from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import agentlab.runner as runner_module
from agentlab.adapters import (
    AgentAdapter,
    AgentExecutionError,
    AgentInfo,
    AgentResumeHandle,
    AgentResumeUnsupportedError,
    AgentRunResult,
    AgentSuspended,
)
from agentlab.adapters.repo_doctor import WORKSPACE_MARKER, RepoDoctorAdapter
from agentlab.execution_sessions import (
    ExecutionSessionError,
    ExecutionStatus,
    execution_session_lock,
    list_execution_sessions,
    load_active_execution_session,
    load_execution_session,
    save_execution_session,
    update_execution_status,
)
from agentlab.models import EvalCase, EvalResult, EvaluationSuspended
from agentlab.runner import (
    ExperimentSuspensionUnsupportedError,
    PytestRunResult,
    evaluate_case,
    resume_evaluation,
    workspace_manifest,
    workspace_manifest_digest,
)
from agentlab.storage import SQLiteStorage
from agentlab.tracer import TraceEvent


class SimulatedCrash(BaseException):
    pass


class ResumableAgent(AgentAdapter):
    def __init__(self, *, resuspend_once: bool = False, fail_on_resume: bool = False):
        self.workspace: Path | None = None
        self.repair_calls = 0
        self.resume_calls = 0
        self.resuspend_once = resuspend_once
        self.fail_on_resume = fail_on_resume

    @property
    def info(self) -> AgentInfo:
        return AgentInfo("resumable_test")

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        self.repair_calls += 1
        self.workspace = workspace
        raise AgentSuspended(
            AgentResumeHandle(
                adapter=self.info.name,
                session_id="adapter-session-1",
                reason="approval_required",
            ),
            AgentRunResult(0),
        )

    def resume(
        self,
        workspace: Path,
        handle: AgentResumeHandle,
    ) -> AgentRunResult:
        self.resume_calls += 1
        assert workspace == self.workspace
        assert handle.session_id == "adapter-session-1"
        if self.resuspend_once and self.resume_calls == 1:
            raise AgentSuspended(handle, AgentRunResult(0))
        if self.fail_on_resume:
            raise AgentExecutionError("terminal adapter failure", AgentRunResult(2))
        (workspace / "completed.txt").write_text("done\n", encoding="utf-8")
        return AgentRunResult(0, "completed")


def _case(tmp_path: Path) -> EvalCase:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    return EvalCase("case-1", str(repository), "Fix VALUE")


def _pytest_result(passed: bool) -> PytestRunResult:
    return PytestRunResult(passed, 0 if passed else 1, "", "")


def _suspend(
    tmp_path: Path,
    monkeypatch,
    *,
    adapter: ResumableAgent | None = None,
) -> tuple[ResumableAgent, EvaluationSuspended]:
    active = adapter or ResumableAgent()
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(False)
    )
    result = evaluate_case(
        _case(tmp_path),
        adapter=active,
        dataset="dataset.yaml",
        database_path=tmp_path / "agentlab.db",
        state_root=tmp_path / "state",
    )
    assert isinstance(result, EvaluationSuspended)
    return active, result


def test_agent_suspension_is_distinct_and_non_final(
    tmp_path: Path, monkeypatch
) -> None:
    adapter, suspended = _suspend(tmp_path, monkeypatch)
    assert not issubclass(AgentSuspended, AgentExecutionError)
    session = load_execution_session(suspended.execution_id, root=tmp_path / "state")
    assert session.status is ExecutionStatus.WAITING_FOR_APPROVAL
    assert session.run_id == suspended.run_id
    assert Path(session.workspace).is_dir()
    assert (Path(session.workspace) / WORKSPACE_MARKER).is_file()
    assert adapter.repair_calls == 1
    assert "agent_suspended" in [event.event_type for event in session.trace]
    assert "run_suspended" in [event.event_type for event in session.trace]
    assert "run_end" not in [event.event_type for event in session.trace]

    storage = SQLiteStorage(tmp_path / "agentlab.db")
    assert storage.list_runs() == ()
    shutil.rmtree(session.workspace)


def test_successful_resume_preserves_run_workspace_and_persists_final_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(tmp_path, monkeypatch)
    session = load_execution_session(suspended.execution_id, root=tmp_path / "state")
    workspace = Path(session.workspace)
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(True)
    )

    result = resume_evaluation(
        suspended.execution_id,
        adapter=adapter,
        state_root=tmp_path / "state",
    )

    assert isinstance(result, EvalResult)
    assert result.passed is True
    assert result.run_id == suspended.run_id
    assert adapter.repair_calls == 1
    assert adapter.resume_calls == 1
    assert not workspace.exists()
    events = [event.event_type for event in result.trace]
    assert events.count("run_start") == 1
    assert "resume_start" in events
    assert "agent_resume_start" in events
    assert "pytest_after_start" in events
    assert events[-1] == "run_end"
    assert [event.sequence for event in result.trace] == list(
        range(1, len(result.trace) + 1)
    )

    storage = SQLiteStorage(tmp_path / "agentlab.db")
    assert storage.get_run(result.run_id).status == "PASS"
    with pytest.raises(ExecutionSessionError, match="terminal"):
        load_active_execution_session(suspended.execution_id, root=tmp_path / "state")


def test_resume_can_suspend_again_without_restart_or_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(
        tmp_path,
        monkeypatch,
        adapter=ResumableAgent(resuspend_once=True),
    )
    workspace = Path(
        load_execution_session(
            suspended.execution_id, root=tmp_path / "state"
        ).workspace
    )
    again = resume_evaluation(
        suspended.execution_id,
        adapter=adapter,
        state_root=tmp_path / "state",
    )
    assert isinstance(again, EvaluationSuspended)
    assert again.execution_id == suspended.execution_id
    assert again.run_id == suspended.run_id
    assert workspace.is_dir()
    assert adapter.repair_calls == 1
    assert adapter.resume_calls == 1
    updated = load_execution_session(suspended.execution_id, root=tmp_path / "state")
    assert updated.status is ExecutionStatus.WAITING_FOR_APPROVAL
    assert "run_end" not in [event.event_type for event in updated.trace]
    shutil.rmtree(workspace)


def test_interrupted_resuming_session_can_be_recovered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(tmp_path, monkeypatch)
    root = tmp_path / "state"
    session = load_execution_session(suspended.execution_id, root=root)
    update_execution_status(session, ExecutionStatus.RESUMING, root=root)
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(True)
    )

    result = resume_evaluation(
        suspended.execution_id,
        adapter=adapter,
        state_root=root,
    )

    assert isinstance(result, EvalResult)
    assert result.passed is True
    assert adapter.resume_calls == 1
    assert load_execution_session(suspended.execution_id, root=root).status is (
        ExecutionStatus.COMPLETED
    )


def test_crash_after_agent_checkpoint_recovers_without_repeating_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(tmp_path, monkeypatch)
    root = tmp_path / "state"
    database = tmp_path / "agentlab.db"
    workspace = Path(load_execution_session(suspended.execution_id, root=root).workspace)
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(True)
    )
    original_save = save_execution_session
    crashed = False

    def crash_after_checkpoint(session, *, root=None):
        nonlocal crashed
        path = original_save(session, root=root)
        if session.status is ExecutionStatus.VERIFYING and not crashed:
            crashed = True
            raise SimulatedCrash
        return path

    monkeypatch.setattr(
        "agentlab.runner.save_execution_session", crash_after_checkpoint
    )
    with pytest.raises(SimulatedCrash):
        resume_evaluation(
            suspended.execution_id,
            adapter=adapter,
            state_root=root,
        )

    checkpoint = load_execution_session(suspended.execution_id, root=root)
    assert checkpoint.status is ExecutionStatus.VERIFYING
    assert checkpoint.trace[-1].event_type == "agent_end"
    assert adapter.resume_calls == 1
    assert workspace.is_dir()
    assert SQLiteStorage(database).get_run(suspended.run_id) is None

    monkeypatch.setattr("agentlab.runner.save_execution_session", original_save)
    result = resume_evaluation(
        suspended.execution_id,
        adapter=adapter,
        state_root=root,
    )

    assert isinstance(result, EvalResult)
    assert result.run_id == suspended.run_id
    assert adapter.resume_calls == 1
    assert [event.sequence for event in result.trace] == list(
        range(1, len(result.trace) + 1)
    )
    assert [event.event_type for event in result.trace].count("resume_start") == 1
    assert SQLiteStorage(database).get_run(suspended.run_id) is not None
    assert load_execution_session(suspended.execution_id, root=root).status is (
        ExecutionStatus.COMPLETED
    )
    assert not workspace.exists()


def test_finalizing_snapshot_recovers_before_sqlite_without_repeating_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(tmp_path, monkeypatch)
    root = tmp_path / "state"
    database = tmp_path / "agentlab.db"
    workspace = Path(load_execution_session(suspended.execution_id, root=root).workspace)
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(True)
    )
    original_persist = SQLiteStorage.save_run_idempotently

    def crash_before_sqlite(self, result, dataset):
        raise SimulatedCrash

    monkeypatch.setattr(SQLiteStorage, "save_run_idempotently", crash_before_sqlite)
    with pytest.raises(SimulatedCrash):
        resume_evaluation(
            suspended.execution_id,
            adapter=adapter,
            state_root=root,
        )

    finalizing = load_execution_session(suspended.execution_id, root=root)
    assert finalizing.status is ExecutionStatus.FINALIZING
    assert finalizing.final_result is not None
    assert finalizing.trace[-1].event_type == "run_end"
    assert adapter.resume_calls == 1
    assert workspace.is_dir()
    assert SQLiteStorage(database).get_run(suspended.run_id) is None

    monkeypatch.setattr(SQLiteStorage, "save_run_idempotently", original_persist)
    result = resume_evaluation(
        suspended.execution_id,
        adapter=adapter,
        state_root=root,
    )

    stored = SQLiteStorage(database)
    assert isinstance(result, EvalResult)
    assert result.run_id == suspended.run_id
    assert adapter.resume_calls == 1
    assert [event.sequence for event in result.trace] == list(
        range(1, len(result.trace) + 1)
    )
    assert stored.get_run(suspended.run_id) is not None
    assert stored.get_trace_events(suspended.run_id) == result.trace
    assert load_execution_session(suspended.execution_id, root=root).status is (
        ExecutionStatus.COMPLETED
    )
    assert not workspace.exists()


def test_sqlite_commit_then_cleanup_crash_is_idempotently_recovered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(tmp_path, monkeypatch)
    root = tmp_path / "state"
    database = tmp_path / "agentlab.db"
    workspace = Path(load_execution_session(suspended.execution_id, root=root).workspace)
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(True)
    )
    original_cleanup = runner_module._cleanup_workspace

    def crash_before_cleanup(_workspace):
        raise SimulatedCrash

    monkeypatch.setattr(runner_module, "_cleanup_workspace", crash_before_cleanup)
    with pytest.raises(SimulatedCrash):
        resume_evaluation(
            suspended.execution_id,
            adapter=adapter,
            state_root=root,
        )

    storage = SQLiteStorage(database)
    persisted_events = storage.get_trace_events(suspended.run_id)
    assert storage.get_run(suspended.run_id) is not None
    assert load_execution_session(suspended.execution_id, root=root).status is (
        ExecutionStatus.FINALIZING
    )
    assert workspace.is_dir()
    assert adapter.resume_calls == 1

    monkeypatch.setattr(runner_module, "_cleanup_workspace", original_cleanup)
    result = resume_evaluation(
        suspended.execution_id,
        adapter=adapter,
        state_root=root,
    )

    assert isinstance(result, EvalResult)
    assert result.run_id == suspended.run_id
    assert adapter.resume_calls == 1
    assert storage.get_trace_events(suspended.run_id) == persisted_events
    assert len(storage.list_runs()) == 1
    assert load_execution_session(suspended.execution_id, root=root).status is (
        ExecutionStatus.COMPLETED
    )
    assert not workspace.exists()


def test_resume_lock_rejects_concurrent_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(tmp_path, monkeypatch)
    root = tmp_path / "state"
    session = load_execution_session(suspended.execution_id, root=root)
    try:
        with (
            execution_session_lock(suspended.execution_id, root=root),
            pytest.raises(ExecutionSessionError, match="already being resumed"),
        ):
            resume_evaluation(
                suspended.execution_id,
                adapter=adapter,
                state_root=root,
            )
        assert adapter.resume_calls == 0
        assert list_execution_sessions(root=root) == (session,)
    finally:
        shutil.rmtree(session.workspace, ignore_errors=True)


def test_resume_rejects_changed_original_repository(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = ResumableAgent()
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(False)
    )
    case = _case(tmp_path)
    case.expected = {"modified_files": ["completed.txt"]}
    suspended = evaluate_case(case, adapter=adapter, state_root=tmp_path / "state")
    assert isinstance(suspended, EvaluationSuspended)
    session = load_execution_session(suspended.execution_id, root=tmp_path / "state")
    (Path(case.repository) / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    try:
        with pytest.raises(ExecutionSessionError, match="changed while execution"):
            resume_evaluation(
                suspended.execution_id,
                adapter=adapter,
                state_root=tmp_path / "state",
            )
        unchanged = load_execution_session(
            suspended.execution_id, root=tmp_path / "state"
        )
        assert unchanged.status is ExecutionStatus.WAITING_FOR_APPROVAL
    finally:
        shutil.rmtree(session.workspace, ignore_errors=True)


def test_resume_rejects_legacy_directory_symlink_before_baseline_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = ResumableAgent()
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(False)
    )
    case = _case(tmp_path)
    case.expected = {"modified_files": ["completed.txt"]}
    suspended = evaluate_case(
        case,
        adapter=adapter,
        dataset="dataset.yaml",
        database_path=tmp_path / "agentlab.db",
        state_root=tmp_path / "state",
    )
    assert isinstance(suspended, EvaluationSuspended)
    root = tmp_path / "state"
    session = load_execution_session(suspended.execution_id, root=root)
    workspace = Path(session.workspace)
    source = Path(case.repository)
    source_real = source / "realdir"
    source_real.mkdir()
    (source_real / "value.txt").write_text("value\n", encoding="utf-8")
    source_link = source / "linked-directory"
    _symlink_or_skip(source_link, source_real, target_is_directory=True)

    workspace_real = workspace / "realdir"
    workspace_link = workspace / "linked-directory"
    workspace_real.mkdir()
    workspace_link.mkdir()
    for directory in (workspace_real, workspace_link):
        (directory / "value.txt").write_text("value\n", encoding="utf-8")
    legacy = replace(
        session,
        baseline_digest=workspace_manifest_digest(workspace_manifest(workspace)),
    )
    save_execution_session(legacy, root=root)

    try:
        with pytest.raises(
            ExecutionSessionError,
            match="directory symlinks are unsupported.*linked-directory",
        ) as captured:
            resume_evaluation(
                suspended.execution_id,
                adapter=adapter,
                state_root=root,
            )
        assert "changed while execution" not in str(captured.value)
        assert adapter.resume_calls == 0
        assert load_execution_session(suspended.execution_id, root=root).status is (
            ExecutionStatus.WAITING_FOR_APPROVAL
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        if os.name == "nt":
            pytest.skip(f"Windows symlink privilege is unavailable: {error}")
        raise


def test_terminal_failure_after_resume_is_final_and_cleans_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter, suspended = _suspend(
        tmp_path,
        monkeypatch,
        adapter=ResumableAgent(fail_on_resume=True),
    )
    workspace = Path(
        load_execution_session(
            suspended.execution_id, root=tmp_path / "state"
        ).workspace
    )
    result = resume_evaluation(
        suspended.execution_id,
        adapter=adapter,
        state_root=tmp_path / "state",
    )
    assert isinstance(result, EvalResult)
    assert result.passed is False
    assert "terminal adapter failure" in result.error
    assert result.trace[-1].event_type == "run_end"
    assert not workspace.exists()
    terminal = load_execution_session(suspended.execution_id, root=tmp_path / "state")
    assert terminal.status is ExecutionStatus.FAILED


def test_non_resumable_adapter_fails_explicitly(tmp_path: Path) -> None:
    class OrdinaryAgent(AgentAdapter):
        def repair(self, workspace: Path, task: str) -> AgentRunResult:
            return AgentRunResult(0)

    with pytest.raises(AgentResumeUnsupportedError, match="does not support"):
        OrdinaryAgent().resume(
            tmp_path,
            AgentResumeHandle("ordinary", "opaque", "approval_required"),
        )


def test_experiment_mode_suspension_aborts_explicitly_and_cleans(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = ResumableAgent()
    monkeypatch.setattr(
        "agentlab.runner.run_pytest", lambda _workspace: _pytest_result(False)
    )
    with pytest.raises(ExperimentSuspensionUnsupportedError, match="unsupported"):
        evaluate_case(
            _case(tmp_path),
            adapter=adapter,
            state_root=tmp_path / "state",
            suspension_supported=False,
        )
    assert adapter.workspace is not None
    assert not adapter.workspace.exists()
    assert not (tmp_path / "state" / "executions").exists()


def test_execution_session_rejects_malformed_future_and_oversized_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, suspended = _suspend(tmp_path, monkeypatch)
    root = tmp_path / "state"
    session = load_execution_session(suspended.execution_id, root=root)
    path = root / "executions" / f"{suspended.execution_id}.json"
    original = path.read_bytes()

    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ExecutionSessionError, match="valid UTF-8 JSON"):
        load_execution_session(suspended.execution_id, root=root)
    path.write_bytes(original)

    future = json.loads(original)
    future["schema_version"] = 999
    path.write_text(json.dumps(future), encoding="utf-8")
    with pytest.raises(ExecutionSessionError, match="Unsupported"):
        load_execution_session(suspended.execution_id, root=root)
    path.write_bytes(original)

    large_trace = tuple(
        TraceEvent(
            run_id=session.run_id,
            sequence=index,
            event_type="evidence",
            timestamp=session.created_at,
            data={"output": "x" * 8_000},
        )
        for index in range(1, 700)
    )
    with pytest.raises(ExecutionSessionError, match="exceeds"):
        save_execution_session(replace(session, trace=large_trace), root=root)
    assert path.read_bytes() == original
    shutil.rmtree(session.workspace)


def test_session_security_rejects_paths_markers_mismatch_and_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, suspended = _suspend(tmp_path, monkeypatch)
    root = tmp_path / "state"
    session = load_execution_session(suspended.execution_id, root=root)
    with pytest.raises(ExecutionSessionError, match="execution_id is invalid"):
        load_execution_session("../escape", root=root)

    mismatched = replace(
        session,
        resume_handle=AgentResumeHandle(
            "different_adapter",
            session.resume_handle.session_id,
            "approval_required",
        ),
    )
    with pytest.raises(ExecutionSessionError, match="does not match"):
        save_execution_session(mismatched, root=root)

    with pytest.raises(ExecutionSessionError, match="outside"):
        save_execution_session(session, root=Path(session.workspace) / "state")

    workspace = Path(session.workspace)
    (workspace / WORKSPACE_MARKER).unlink()
    with pytest.raises(ExecutionSessionError, match="marked AgentLab"):
        resume_evaluation(
            suspended.execution_id,
            adapter=ResumableAgent(),
            state_root=root,
        )

    forged = AgentResumeHandle(
        "repo_doctor",
        "a" * 32,
        "approval_required",
        {"command": "attacker.execute"},
    )
    with pytest.raises(ValueError, match="unexpected metadata"):
        RepoDoctorAdapter()._validate_resume_handle(forged)
    shutil.rmtree(workspace)

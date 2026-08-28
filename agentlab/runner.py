"""Single-case evaluation execution, suspension, and resumption."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from agentlab.adapters import (
    AgentAdapter,
    AgentExecutionError,
    AgentRunResult,
    AgentSuspended,
    RepoDoctorAdapter,
    create_default_registry,
)
from agentlab.adapters.repo_doctor import WORKSPACE_MARKER
from agentlab.evaluators import EvaluationOutcome, Evaluator
from agentlab.execution_sessions import (
    ExecutionSession,
    ExecutionSessionError,
    ExecutionStatus,
    load_active_execution_session,
    new_execution_id,
    now_timestamp,
    save_execution_session,
    update_execution_status,
)
from agentlab.models import EvalCase, EvalResult, EvaluationSuspended
from agentlab.tracer import Tracer, summarize_text


@dataclass(frozen=True)
class PytestRunResult:
    """Captured result of one pytest invocation."""

    passed: bool
    returncode: int
    stdout: str
    stderr: str


class ExperimentSuspensionUnsupportedError(RuntimeError):
    """V1 deliberately does not resume repeated-trial experiment context."""


def create_workspace(repository: str) -> Path:
    source = Path(repository).resolve()
    temp_dir = Path(tempfile.mkdtemp(prefix="agentlab_"))
    try:
        shutil.copytree(
            source,
            temp_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        (temp_dir / WORKSPACE_MARKER).write_text(
            "AgentLab temporary evaluation workspace.\n",
            encoding="utf-8",
        )
    except Exception:
        _cleanup_workspace(temp_dir)
        raise
    return temp_dir.resolve(strict=True)


def validate_preserved_workspace(workspace: str | Path) -> Path:
    """Require the exact canonical AgentLab temporary-workspace shape."""
    candidate = Path(workspace)
    if not candidate.is_absolute() or candidate.resolve(strict=False) != candidate:
        raise ExecutionSessionError(
            "Preserved workspace path must be canonical and absolute."
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ExecutionSessionError("Preserved workspace no longer exists.") from error
    marker = resolved / WORKSPACE_MARKER
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    if (
        resolved.parent != temporary_root
        or not resolved.name.startswith("agentlab_")
        or not marker.is_file()
        or marker.is_symlink()
    ):
        raise ExecutionSessionError(
            "Preserved workspace is not a marked AgentLab temporary workspace."
        )
    return resolved


def run_pytest(workspace: Path) -> PytestRunResult:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return PytestRunResult(
        passed=result.returncode == 0,
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def _elapsed(tracer: Tracer, started_at: float) -> float:
    return round(tracer.elapsed_since(started_at), 6)


def _run_traced_pytest(workspace: Path, tracer: Tracer, phase: str) -> PytestRunResult:
    tracer.emit(f"{phase}_start", workspace=workspace)
    started_at = tracer.start_timer()
    try:
        result = run_pytest(workspace)
    except Exception as error:
        tracer.emit(
            f"{phase}_end",
            status="error",
            passed=False,
            elapsed_time=_elapsed(tracer, started_at),
            error_type=type(error).__name__,
        )
        raise
    tracer.emit(
        f"{phase}_end",
        status="pass" if result.passed else "fail",
        passed=result.passed,
        returncode=result.returncode,
        stdout=summarize_text(result.stdout),
        stderr=summarize_text(result.stderr),
        elapsed_time=_elapsed(tracer, started_at),
    )
    return result


def _agent_result_data(result: AgentRunResult | None) -> dict:
    if result is None:
        return {"returncode": None, "stdout": "", "stderr": ""}
    data = {
        "returncode": result.returncode,
        "stdout": summarize_text(result.stdout),
        "stderr": summarize_text(result.stderr),
    }
    if result.diagnostics is not None:
        data["diagnostics"] = result.diagnostics.to_trace_data()
    return data


def _run_traced_agent(
    adapter: AgentAdapter,
    workspace: Path | None,
    task: str,
    tracer: Tracer,
) -> AgentRunResult | None:
    adapter_name = type(adapter).__name__
    metadata = {**adapter.trace_metadata(), "task_provided": bool(task.strip())}
    tracer.emit("agent_start", adapter=adapter_name, workspace=workspace, **metadata)
    started_at = tracer.start_timer()
    try:
        result = adapter.repair(workspace, task)
    except AgentSuspended as suspended:
        tracer.emit(
            "agent_suspended",
            adapter=adapter_name,
            status="suspended",
            reason=suspended.handle.reason,
            elapsed_time=_elapsed(tracer, started_at),
            **metadata,
            **_agent_result_data(suspended.result),
        )
        raise
    except Exception as error:
        data = {
            "adapter": adapter_name,
            "status": "error",
            "elapsed_time": _elapsed(tracer, started_at),
            **metadata,
        }
        if isinstance(error, AgentExecutionError):
            data.update(_agent_result_data(error.result))
        tracer.emit("agent_end", **data)
        raise
    tracer.emit(
        "agent_end",
        adapter=adapter_name,
        status="ok",
        elapsed_time=_elapsed(tracer, started_at),
        **metadata,
        **_agent_result_data(result),
    )
    return result


def _run_traced_resume(
    adapter: AgentAdapter,
    workspace: Path,
    session: ExecutionSession,
    tracer: Tracer,
) -> AgentRunResult | None:
    adapter_name = type(adapter).__name__
    metadata = adapter.trace_metadata()
    tracer.emit("resume_start", execution_id=session.execution_id)
    tracer.emit(
        "agent_resume_start",
        adapter=adapter_name,
        workspace=workspace,
        reason=session.resume_handle.reason,
        **metadata,
    )
    started_at = tracer.start_timer()
    try:
        result = adapter.resume(workspace, session.resume_handle)
    except AgentSuspended as suspended:
        tracer.emit(
            "agent_suspended",
            adapter=adapter_name,
            status="suspended",
            reason=suspended.handle.reason,
            elapsed_time=_elapsed(tracer, started_at),
            **metadata,
            **_agent_result_data(suspended.result),
        )
        raise
    except Exception as error:
        data = {
            "adapter": adapter_name,
            "status": "error",
            "elapsed_time": _elapsed(tracer, started_at),
            **metadata,
        }
        if isinstance(error, AgentExecutionError):
            data.update(_agent_result_data(error.result))
        tracer.emit("agent_end", **data)
        raise
    tracer.emit(
        "agent_end",
        adapter=adapter_name,
        status="ok",
        resumed=True,
        elapsed_time=_elapsed(tracer, started_at),
        **metadata,
        **_agent_result_data(result),
    )
    return result


def _remove_readonly(function, path: str, _error) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _cleanup_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace, onerror=_remove_readonly)


def _run_traced_evaluator(
    evaluator: Evaluator,
    workspace: Path,
    case: EvalCase,
    tracer: Tracer,
) -> EvaluationOutcome:
    evaluator_name = type(evaluator).__name__
    tracer.emit("evaluator_start", evaluator=evaluator_name, case_id=case.id)
    started_at = tracer.start_timer()
    try:
        outcome = evaluator.evaluate(workspace, case)
    except Exception as error:
        tracer.emit(
            "evaluator_end",
            evaluator=evaluator_name,
            status="error",
            passed=False,
            elapsed_time=_elapsed(tracer, started_at),
            error_type=type(error).__name__,
        )
        raise
    tracer.emit(
        "evaluator_end",
        evaluator=evaluator_name,
        status="pass" if outcome.passed else "fail",
        passed=outcome.passed,
        score=outcome.score,
        feedback=summarize_text(outcome.feedback),
        metadata=outcome.metadata,
        elapsed_time=_elapsed(tracer, started_at),
    )
    return outcome


def _error_trace_data(error: Exception, phase: str) -> dict:
    data = {
        "phase": phase,
        "error_type": type(error).__name__,
        "message": summarize_text(error),
    }
    if isinstance(error, AgentExecutionError) and error.diagnostics is not None:
        diagnostics = error.diagnostics
        data.update(
            failure_type=(
                diagnostics.failure_type.value
                if diagnostics.failure_type is not None
                else None
            ),
            failure_phase=diagnostics.failure_phase,
        )
    return data


def _cleanup_and_finish(
    *,
    workspace: Path,
    tracer: Tracer,
    case: EvalCase,
    before_passed: bool,
    after_passed: bool,
    evaluator_passed: bool,
    error_message: str | None,
) -> EvalResult:
    try:
        if workspace is not None:
            _cleanup_workspace(workspace)
    except Exception as cleanup_error:  # noqa: BLE001 - cleanup failure is trace evidence.
        cleanup_message = summarize_text(cleanup_error)
        tracer.emit(
            "error",
            phase="cleanup",
            error_type=type(cleanup_error).__name__,
            message=cleanup_message,
        )
        error_message = (
            f"{error_message}; workspace cleanup failed: {cleanup_message}"
            if error_message
            else f"Workspace cleanup failed: {cleanup_message}"
        )
    passed = error_message is None and after_passed and evaluator_passed
    tracer.emit(
        "run_end",
        case_id=case.id,
        final_status="pass" if passed else "fail",
        passed=passed,
        tests_before_passed=before_passed,
        tests_after_passed=after_passed,
        elapsed_time=round(tracer.total_elapsed(), 6),
    )
    return EvalResult(
        case_id=case.id,
        passed=passed,
        tests_before_passed=before_passed,
        tests_after_passed=after_passed,
        error=error_message,
        run_id=tracer.run_id,
        trace=tracer.events,
    )


def evaluate_case(
    case: EvalCase,
    adapter: AgentAdapter | None = None,
    tracer: Tracer | None = None,
    evaluator: Evaluator | None = None,
    *,
    dataset: str | None = None,
    evaluator_name: str | None = None,
    database_path: str | Path | None = None,
    state_root: Path | None = None,
    suspension_supported: bool = True,
) -> EvalResult | EvaluationSuspended:
    """Evaluate once, returning a non-final control object if the agent suspends."""
    active_tracer = tracer or Tracer()
    active_adapter = adapter or RepoDoctorAdapter()
    workspace: Path | None = None
    before_passed = False
    after_passed = False
    evaluator_passed = True
    error_message: str | None = None
    phase = "workspace"
    active_tracer.emit(
        "run_start",
        case_id=case.id,
        task=case.task,
        repository=case.repository,
        adapter=type(active_adapter).__name__,
        adapter_identity=active_adapter.info.name,
    )
    try:
        workspace = create_workspace(case.repository)
        phase = "pytest_before"
        before_result = _run_traced_pytest(workspace, active_tracer, phase)
        before_passed = before_result.passed
        phase = "agent"
        _run_traced_agent(active_adapter, workspace, case.task, active_tracer)
        phase = "pytest_after"
        after_result = _run_traced_pytest(workspace, active_tracer, phase)
        after_passed = after_result.passed
        if evaluator is not None and after_passed:
            phase = "evaluator"
            evaluator_outcome = _run_traced_evaluator(
                evaluator, workspace, case, active_tracer
            )
            evaluator_passed = evaluator_outcome.passed
    except AgentSuspended as suspended:
        if workspace is None:
            raise RuntimeError(
                "Agent suspended before a workspace existed."
            ) from suspended
        if suspended.handle.adapter != active_adapter.info.name:
            error = ValueError(
                "Suspended handle adapter does not match the active adapter."
            )
            active_tracer.emit("error", **_error_trace_data(error, phase))
            return _cleanup_and_finish(
                workspace=workspace,
                tracer=active_tracer,
                case=case,
                before_passed=before_passed,
                after_passed=False,
                evaluator_passed=True,
                error_message=summarize_text(error),
            )
        active_tracer.emit(
            "run_suspended",
            case_id=case.id,
            reason=suspended.handle.reason,
            elapsed_time=round(active_tracer.total_elapsed(), 6),
        )
        if not suspension_supported:
            _cleanup_workspace(workspace)
            raise ExperimentSuspensionUnsupportedError(
                "Resumable agent suspension is unsupported for experiments in V1."
            ) from suspended
        identifier = new_execution_id()
        timestamp = now_timestamp()
        session = ExecutionSession(
            execution_id=identifier,
            run_id=active_tracer.run_id,
            case=case,
            adapter=active_adapter.info.name,
            resume_handle=suspended.handle,
            workspace=str(workspace),
            phase="agent",
            tests_before_passed=before_passed,
            trace=active_tracer.events,
            elapsed_seconds=active_tracer.total_elapsed(),
            status=ExecutionStatus.WAITING_FOR_APPROVAL,
            created_at=timestamp,
            updated_at=timestamp,
            dataset=dataset,
            evaluator=evaluator_name,
            database_path=(
                str(Path(database_path).expanduser().resolve(strict=False))
                if database_path is not None
                else None
            ),
        )
        try:
            save_execution_session(session, root=state_root)
        except (
            OSError,
            TypeError,
            ValueError,
            ExecutionSessionError,
        ) as persistence_error:
            active_tracer.emit(
                "error", **_error_trace_data(persistence_error, "suspension")
            )
            return _cleanup_and_finish(
                workspace=workspace,
                tracer=active_tracer,
                case=case,
                before_passed=before_passed,
                after_passed=False,
                evaluator_passed=True,
                error_message=summarize_text(persistence_error),
            )
        return EvaluationSuspended(identifier, active_tracer.run_id, case.id)
    except Exception as error:  # noqa: BLE001 - evaluation errors become trace evidence.
        error_message = summarize_text(error)
        active_tracer.emit("error", **_error_trace_data(error, phase))

    return _cleanup_and_finish(
        workspace=workspace,
        tracer=active_tracer,
        case=case,
        before_passed=before_passed,
        after_passed=after_passed,
        evaluator_passed=evaluator_passed,
        error_message=error_message,
    )


def resume_evaluation(
    execution_id: str,
    *,
    adapter: AgentAdapter | None = None,
    evaluator: Evaluator | None = None,
    state_root: Path | None = None,
) -> EvalResult | EvaluationSuspended:
    """Resume one persisted execution without restarting its original repair."""
    session = load_active_execution_session(execution_id, root=state_root)
    workspace = validate_preserved_workspace(session.workspace)
    active_adapter = adapter
    if active_adapter is None:
        try:
            active_adapter = create_default_registry().create(session.adapter)
        except KeyError as error:
            raise ExecutionSessionError(
                f"Persisted adapter is not registered: {session.adapter}"
            ) from error
    if active_adapter.info.name != session.adapter:
        raise ExecutionSessionError(
            "Active adapter does not match the execution session."
        )
    if session.evaluator is not None and evaluator is None:
        raise ExecutionSessionError(
            "This execution requires its persisted evaluator configuration to resume."
        )

    session = update_execution_status(
        session, ExecutionStatus.RESUMING, root=state_root
    )
    tracer = Tracer(
        run_id=session.run_id,
        events=session.trace,
        elapsed_offset=session.elapsed_seconds,
    )
    after_passed = False
    evaluator_passed = True
    error_message: str | None = None
    phase = "agent"
    try:
        _run_traced_resume(active_adapter, workspace, session, tracer)
        phase = "pytest_after"
        after_result = _run_traced_pytest(workspace, tracer, phase)
        after_passed = after_result.passed
        if evaluator is not None and after_passed:
            phase = "evaluator"
            evaluator_outcome = _run_traced_evaluator(
                evaluator, workspace, session.case, tracer
            )
            evaluator_passed = evaluator_outcome.passed
    except AgentSuspended as suspended:
        if suspended.handle.adapter != session.adapter:
            error = ValueError(
                "Re-suspended handle adapter does not match the session."
            )
            tracer.emit("error", **_error_trace_data(error, phase))
            error_message = summarize_text(error)
        else:
            tracer.emit(
                "run_suspended",
                case_id=session.case.id,
                reason=suspended.handle.reason,
                elapsed_time=round(tracer.total_elapsed(), 6),
            )
            updated = replace(
                session,
                resume_handle=suspended.handle,
                trace=tracer.events,
                elapsed_seconds=tracer.total_elapsed(),
                status=ExecutionStatus.WAITING_FOR_APPROVAL,
                updated_at=now_timestamp(),
            )
            save_execution_session(updated, root=state_root)
            return EvaluationSuspended(
                session.execution_id,
                session.run_id,
                session.case.id,
            )
    except Exception as error:  # noqa: BLE001 - terminal errors become final evidence.
        error_message = summarize_text(error)
        tracer.emit("error", **_error_trace_data(error, phase))

    result = _cleanup_and_finish(
        workspace=workspace,
        tracer=tracer,
        case=session.case,
        before_passed=session.tests_before_passed,
        after_passed=after_passed,
        evaluator_passed=evaluator_passed,
        error_message=error_message,
    )
    terminal_status = (
        ExecutionStatus.COMPLETED if result.passed else ExecutionStatus.FAILED
    )
    terminal = replace(
        session,
        trace=result.trace,
        elapsed_seconds=tracer.total_elapsed(),
        status=terminal_status,
        updated_at=now_timestamp(),
    )
    save_execution_session(terminal, root=state_root)
    return result

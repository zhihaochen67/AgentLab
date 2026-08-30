"""Single-case evaluation execution, suspension, and resumption."""

from __future__ import annotations

import hashlib
import json
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
    ExecutionFinalResult,
    ExecutionSession,
    ExecutionSessionError,
    ExecutionStatus,
    execution_session_lock,
    load_active_execution_session,
    new_execution_id,
    now_timestamp,
    save_execution_session,
    update_execution_status,
)
from agentlab.models import EvalCase, EvalResult, EvaluationSuspended
from agentlab.storage import SQLiteStorage, default_database_path
from agentlab.subprocesses import run_process
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


class PytestTimeoutError(RuntimeError):
    """The deterministic test gate exceeded its bounded runtime."""


class UnsupportedWorkspaceSymlinkError(ValueError):
    """A source or preserved workspace violates the symlink policy."""


@dataclass(frozen=True)
class WorkspaceChangeResult:
    """Deterministic comparison of actual and declared repository changes."""

    passed: bool
    modified_files: tuple[str, ...]
    missing_expected_files: tuple[str, ...]
    unexpected_files: tuple[str, ...]


PYTEST_TIMEOUT_SECONDS = 120
_IGNORED_WORKSPACE_DIRECTORIES = frozenset({".git", ".pytest_cache", "__pycache__"})
_IGNORED_WORKSPACE_FILES = frozenset({WORKSPACE_MARKER})
_MAX_TRACE_PATHS = 200


def create_workspace(repository: str) -> Path:
    source = validate_workspace_symlinks(repository)
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


def workspace_manifest(workspace: str | Path) -> dict[str, str]:
    """Hash material workspace files in stable relative-path order."""
    root = validate_workspace_symlinks(workspace)
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if any(part in _IGNORED_WORKSPACE_DIRECTORIES for part in relative.parts):
            continue
        if relative.as_posix() in _IGNORED_WORKSPACE_FILES:
            continue
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[relative.as_posix()] = digest.hexdigest()
    return manifest


def validate_workspace_symlinks(workspace: str | Path) -> Path:
    """Reject directory and non-file symlinks before copy or manifest use."""
    candidate = Path(workspace)
    if candidate.is_symlink() and candidate.is_dir():
        raise UnsupportedWorkspaceSymlinkError(
            "directory symlinks are unsupported in AgentLab workspaces: ."
        )
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        return root

    def raise_scan_error(error: OSError) -> None:
        raise error

    for current, directories, files in os.walk(
        root,
        followlinks=False,
        onerror=raise_scan_error,
    ):
        directory = Path(current)
        for name in sorted(directories):
            path = directory / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                raise UnsupportedWorkspaceSymlinkError(
                    "directory symlinks are unsupported in AgentLab workspaces: "
                    f"{relative}"
                )
        for name in sorted(files):
            path = directory / name
            if path.is_symlink() and not path.is_file():
                relative = path.relative_to(root).as_posix()
                raise UnsupportedWorkspaceSymlinkError(
                    "non-file symlinks are unsupported in AgentLab workspaces: "
                    f"{relative}"
                )
    return root


def workspace_manifest_digest(manifest: dict[str, str]) -> str:
    """Return a stable digest used to detect source changes across suspension."""
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_modified_files(case: EvalCase) -> tuple[str, ...] | None:
    if not isinstance(case.expected, dict) or "modified_files" not in case.expected:
        return None
    values = case.expected["modified_files"]
    if not isinstance(values, list) or not values:
        raise ValueError("expected.modified_files must be a non-empty list.")
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected.modified_files entries must be non-empty paths.")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("expected.modified_files must stay inside the repository.")
        normalized.add(relative.as_posix())
    return tuple(sorted(normalized))


def verify_workspace_changes(
    baseline: dict[str, str],
    current: dict[str, str],
    expected_files: tuple[str, ...],
) -> WorkspaceChangeResult:
    """Require actual file-content changes to match the dataset contract exactly."""
    modified = tuple(
        sorted(
            path
            for path in set(baseline) | set(current)
            if baseline.get(path) != current.get(path)
        )
    )
    actual = set(modified)
    expected = set(expected_files)
    missing = tuple(sorted(expected - actual))
    unexpected = tuple(sorted(actual - expected))
    return WorkspaceChangeResult(
        passed=not missing and not unexpected,
        modified_files=modified,
        missing_expected_files=missing,
        unexpected_files=unexpected,
    )


def _workspace_change_error(result: WorkspaceChangeResult) -> str:
    details = []
    if result.missing_expected_files:
        details.append(
            "expected files were unchanged: "
            + ", ".join(result.missing_expected_files[:_MAX_TRACE_PATHS])
        )
    if result.unexpected_files:
        details.append(
            "unexpected files changed: "
            + ", ".join(result.unexpected_files[:_MAX_TRACE_PATHS])
        )
    return "Workspace change contract failed (" + "; ".join(details) + ")."


def _combine_error(current: str | None, additional: str) -> str:
    return f"{current}; {additional}" if current else additional


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
    try:
        validate_workspace_symlinks(resolved)
    except UnsupportedWorkspaceSymlinkError as error:
        raise ExecutionSessionError(f"Cannot resume execution: {error}") from error
    return resolved


def run_pytest(workspace: Path) -> PytestRunResult:
    try:
        result = run_process(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                ".",
                "-p",
                "no:cacheprovider",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise PytestTimeoutError(
            f"pytest exceeded the {PYTEST_TIMEOUT_SECONDS}-second limit"
        ) from error
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


def _run_traced_workspace_verification(
    workspace: Path,
    baseline: dict[str, str],
    expected_files: tuple[str, ...],
    tracer: Tracer,
) -> WorkspaceChangeResult:
    tracer.emit(
        "workspace_verification_start",
        expected_modified_files=list(expected_files),
    )
    started_at = tracer.start_timer()
    try:
        result = verify_workspace_changes(
            baseline,
            workspace_manifest(workspace),
            expected_files,
        )
    except Exception as error:
        tracer.emit(
            "workspace_verification_end",
            status="error",
            passed=False,
            elapsed_time=_elapsed(tracer, started_at),
            error_type=type(error).__name__,
        )
        raise
    tracer.emit(
        "workspace_verification_end",
        status="pass" if result.passed else "fail",
        passed=result.passed,
        modified_file_count=len(result.modified_files),
        modified_files=list(result.modified_files[:_MAX_TRACE_PATHS]),
        missing_expected_files=list(
            result.missing_expected_files[:_MAX_TRACE_PATHS]
        ),
        unexpected_files=list(result.unexpected_files[:_MAX_TRACE_PATHS]),
        paths_truncated=len(result.modified_files) > _MAX_TRACE_PATHS,
        elapsed_time=_elapsed(tracer, started_at),
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
    workspace_changes_passed: bool,
    error_message: str | None,
    failure_reason: str | None = None,
    cleanup: bool = True,
) -> EvalResult:
    try:
        if cleanup and workspace is not None:
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
        failure_reason = failure_reason or "cleanup_error"
    passed = (
        error_message is None
        and after_passed
        and evaluator_passed
        and workspace_changes_passed
    )
    if not passed and failure_reason is None:
        if not workspace_changes_passed:
            failure_reason = "workspace_contract_failed"
        elif not after_passed:
            failure_reason = "tests_after_failed"
        elif not evaluator_passed:
            failure_reason = "evaluator_failed"
        else:
            failure_reason = "runtime_error"
    tracer.emit(
        "run_end",
        case_id=case.id,
        final_status="pass" if passed else "fail",
        failure_reason=failure_reason,
        passed=passed,
        tests_before_passed=before_passed,
        tests_after_passed=after_passed,
        workspace_changes_passed=workspace_changes_passed,
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
    baseline: dict[str, str] | None = None
    baseline_digest: str | None = None
    expected_files: tuple[str, ...] | None = None
    before_passed = False
    after_passed = False
    evaluator_passed = True
    workspace_changes_passed = True
    error_message: str | None = None
    failure_reason: str | None = None
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
        expected_files = _expected_modified_files(case)
        workspace = create_workspace(case.repository)
        if expected_files is not None:
            baseline = workspace_manifest(workspace)
            baseline_digest = workspace_manifest_digest(baseline)
            active_tracer.emit(
                "workspace_baseline",
                file_count=len(baseline),
                digest=baseline_digest,
                expected_modified_files=list(expected_files),
            )
        phase = "pytest_before"
        before_result = _run_traced_pytest(workspace, active_tracer, phase)
        before_passed = before_result.passed
        phase = "agent"
        _run_traced_agent(active_adapter, workspace, case.task, active_tracer)
        if baseline is not None and expected_files is not None:
            phase = "workspace_verification"
            changes = _run_traced_workspace_verification(
                workspace,
                baseline,
                expected_files,
                active_tracer,
            )
            workspace_changes_passed = changes.passed
            if not changes.passed:
                error_message = _workspace_change_error(changes)
                failure_reason = "workspace_contract_failed"
        phase = "pytest_after"
        after_result = _run_traced_pytest(workspace, active_tracer, phase)
        after_passed = after_result.passed
        if evaluator is not None and after_passed and workspace_changes_passed:
            phase = "evaluator"
            evaluator_outcome = _run_traced_evaluator(
                evaluator, workspace, case, active_tracer
            )
            evaluator_passed = evaluator_outcome.passed
            if not evaluator_passed:
                failure_reason = "evaluator_failed"
        if not after_passed and failure_reason is None:
            failure_reason = "tests_after_failed"
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
                workspace_changes_passed=False,
                error_message=summarize_text(error),
                failure_reason="invalid_resume_handle",
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
            case=replace(
                case,
                repository=str(Path(case.repository).resolve(strict=True)),
            ),
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
            baseline_digest=baseline_digest,
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
                workspace_changes_passed=False,
                error_message=summarize_text(persistence_error),
                failure_reason="suspension_persistence_error",
            )
        return EvaluationSuspended(identifier, active_tracer.run_id, case.id)
    except Exception as error:  # noqa: BLE001 - evaluation errors become trace evidence.
        error_message = _combine_error(error_message, summarize_text(error))
        failure_reason = failure_reason or f"{phase}_error"
        active_tracer.emit("error", **_error_trace_data(error, phase))

    return _cleanup_and_finish(
        workspace=workspace,
        tracer=active_tracer,
        case=case,
        before_passed=before_passed,
        after_passed=after_passed,
        evaluator_passed=evaluator_passed,
        workspace_changes_passed=workspace_changes_passed,
        error_message=error_message,
        failure_reason=failure_reason,
    )


def resume_evaluation(
    execution_id: str,
    *,
    adapter: AgentAdapter | None = None,
    evaluator: Evaluator | None = None,
    state_root: Path | None = None,
) -> EvalResult | EvaluationSuspended:
    """Resume one persisted execution without restarting its original repair."""
    with execution_session_lock(execution_id, root=state_root):
        return _resume_evaluation_locked(
            execution_id,
            adapter=adapter,
            evaluator=evaluator,
            state_root=state_root,
        )


def _result_from_finalizing_session(session: ExecutionSession) -> EvalResult:
    final = session.final_result
    if final is None:
        raise ExecutionSessionError("Finalizing execution has no persisted result.")
    return EvalResult(
        case_id=session.case.id,
        passed=final.passed,
        tests_before_passed=session.tests_before_passed,
        tests_after_passed=final.tests_after_passed,
        error=final.error,
        run_id=session.run_id,
        trace=session.trace,
    )


def _finalizing_workspace_path(workspace: str) -> Path:
    """Validate the fixed cleanup target even after a partial prior cleanup."""
    candidate = Path(workspace)
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    if (
        not candidate.is_absolute()
        or candidate.resolve(strict=False) != candidate
        or candidate.parent != temporary_root
        or not candidate.name.startswith("agentlab_")
    ):
        raise ExecutionSessionError(
            "Finalizing workspace is not a canonical AgentLab temporary path."
        )
    return candidate


def _finalize_execution(
    session: ExecutionSession,
    *,
    state_root: Path | None,
) -> EvalResult:
    """Idempotently persist, clean, and terminalize one staged result."""
    if session.status is not ExecutionStatus.FINALIZING:
        raise ExecutionSessionError("Execution is not ready for finalization.")
    result = _result_from_finalizing_session(session)
    database = (
        Path(session.database_path)
        if session.database_path is not None
        else default_database_path()
    )
    dataset = session.dataset or "resumed-single-evaluation"
    SQLiteStorage(database).save_run_idempotently(result, dataset)

    workspace = _finalizing_workspace_path(session.workspace)
    try:
        _cleanup_workspace(workspace)
    except Exception as error:
        raise ExecutionSessionError(
            "Final run is persisted but workspace cleanup remains pending."
        ) from error

    terminal = replace(
        session,
        status=(
            ExecutionStatus.COMPLETED if result.passed else ExecutionStatus.FAILED
        ),
        updated_at=now_timestamp(),
    )
    try:
        save_execution_session(terminal, root=state_root)
    except (OSError, TypeError, ValueError, ExecutionSessionError) as error:
        raise ExecutionSessionError(
            "Final run is persisted and workspace is clean, but terminal session "
            "persistence remains pending."
        ) from error
    return result


def _resume_evaluation_locked(
    execution_id: str,
    *,
    adapter: AgentAdapter | None,
    evaluator: Evaluator | None,
    state_root: Path | None,
) -> EvalResult | EvaluationSuspended:
    session = load_active_execution_session(execution_id, root=state_root)
    if session.status is ExecutionStatus.FINALIZING:
        return _finalize_execution(session, state_root=state_root)

    workspace = validate_preserved_workspace(session.workspace)
    expected_files = _expected_modified_files(session.case)
    baseline: dict[str, str] | None = None
    if expected_files is not None:
        try:
            baseline = workspace_manifest(session.case.repository)
        except UnsupportedWorkspaceSymlinkError as error:
            raise ExecutionSessionError(
                f"Cannot resume execution: {error}"
            ) from error
        except OSError as error:
            raise ExecutionSessionError(
                "Could not read the original repository baseline for resume."
            ) from error
        current_digest = workspace_manifest_digest(baseline)
        if (
            session.baseline_digest is not None
            and current_digest != session.baseline_digest
        ):
            raise ExecutionSessionError(
                "Original repository changed while execution was suspended."
            )
    if session.evaluator is not None and evaluator is None:
        raise ExecutionSessionError(
            "This execution requires its persisted evaluator configuration to resume."
        )

    tracer = Tracer(
        run_id=session.run_id,
        events=session.trace,
        elapsed_offset=session.elapsed_seconds,
    )
    after_passed = False
    evaluator_passed = True
    workspace_changes_passed = True
    error_message: str | None = None
    failure_reason: str | None = None
    phase = "agent"
    if session.status is not ExecutionStatus.VERIFYING:
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
        session = update_execution_status(
            session, ExecutionStatus.RESUMING, root=state_root
        )
        try:
            _run_traced_resume(active_adapter, workspace, session, tracer)
        except AgentSuspended as suspended:
            if suspended.handle.adapter != session.adapter:
                error = ValueError(
                    "Re-suspended handle adapter does not match the session."
                )
                tracer.emit("error", **_error_trace_data(error, phase))
                error_message = summarize_text(error)
                failure_reason = "invalid_resume_handle"
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
        except Exception as error:  # noqa: BLE001 - terminal error is staged.
            error_message = summarize_text(error)
            failure_reason = f"{phase}_error"
            tracer.emit("error", **_error_trace_data(error, phase))
        else:
            session = replace(
                session,
                trace=tracer.events,
                elapsed_seconds=tracer.total_elapsed(),
                status=ExecutionStatus.VERIFYING,
                updated_at=now_timestamp(),
            )
            save_execution_session(session, root=state_root)

    if error_message is None:
        try:
            if baseline is not None and expected_files is not None:
                phase = "workspace_verification"
                changes = _run_traced_workspace_verification(
                    workspace,
                    baseline,
                    expected_files,
                    tracer,
                )
                workspace_changes_passed = changes.passed
                if not changes.passed:
                    error_message = _workspace_change_error(changes)
                    failure_reason = "workspace_contract_failed"
            phase = "pytest_after"
            after_result = _run_traced_pytest(workspace, tracer, phase)
            after_passed = after_result.passed
            if evaluator is not None and after_passed and workspace_changes_passed:
                phase = "evaluator"
                evaluator_outcome = _run_traced_evaluator(
                    evaluator, workspace, session.case, tracer
                )
                evaluator_passed = evaluator_outcome.passed
                if not evaluator_passed:
                    failure_reason = "evaluator_failed"
            if not after_passed and failure_reason is None:
                failure_reason = "tests_after_failed"
        except Exception as error:  # noqa: BLE001 - terminal error is staged.
            error_message = _combine_error(error_message, summarize_text(error))
            failure_reason = failure_reason or f"{phase}_error"
            tracer.emit("error", **_error_trace_data(error, phase))

    result = _cleanup_and_finish(
        workspace=workspace,
        tracer=tracer,
        case=session.case,
        before_passed=session.tests_before_passed,
        after_passed=after_passed,
        evaluator_passed=evaluator_passed,
        workspace_changes_passed=workspace_changes_passed,
        error_message=error_message,
        failure_reason=failure_reason,
        cleanup=False,
    )
    finalizing = replace(
        session,
        trace=result.trace,
        elapsed_seconds=tracer.total_elapsed(),
        status=ExecutionStatus.FINALIZING,
        final_result=ExecutionFinalResult(
            passed=result.passed,
            tests_after_passed=result.tests_after_passed,
            error=result.error,
        ),
        updated_at=now_timestamp(),
    )
    save_execution_session(finalizing, root=state_root)
    return _finalize_execution(finalizing, state_root=state_root)

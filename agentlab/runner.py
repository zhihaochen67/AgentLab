import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab.adapters import (
    AgentAdapter,
    AgentExecutionError,
    AgentRunResult,
    RepoDoctorAdapter,
)
from agentlab.adapters.repo_doctor import WORKSPACE_MARKER
from agentlab.models import EvalCase, EvalResult
from agentlab.tracer import Tracer, summarize_text


@dataclass(frozen=True)
class PytestRunResult:
    """Captured result of one pytest invocation."""

    passed: bool
    returncode: int
    stdout: str
    stderr: str


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

    return temp_dir


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
    workspace: Path,
    task: str,
    tracer: Tracer,
) -> AgentRunResult | None:
    adapter_name = type(adapter).__name__
    metadata = {
        **adapter.trace_metadata(),
        "task_provided": bool(task.strip()),
    }
    tracer.emit(
        "agent_start",
        adapter=adapter_name,
        workspace=workspace,
        **metadata,
    )
    started_at = tracer.start_timer()
    try:
        result = adapter.repair(workspace, task)
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


def _remove_readonly(function, path: str, _error) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _cleanup_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace, onerror=_remove_readonly)


def evaluate_case(
    case: EvalCase,
    adapter: AgentAdapter | None = None,
    tracer: Tracer | None = None,
) -> EvalResult:
    active_tracer = tracer or Tracer()
    active_adapter = adapter or RepoDoctorAdapter()
    workspace: Path | None = None
    before_passed = False
    after_passed = False
    error_message: str | None = None
    phase = "workspace"

    active_tracer.emit(
        "run_start",
        case_id=case.id,
        task=case.task,
        repository=case.repository,
        adapter=type(active_adapter).__name__,
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

    except Exception as error:  # noqa: BLE001 - evaluation errors become trace evidence.
        error_message = summarize_text(error)
        error_data = {
            "phase": phase,
            "error_type": type(error).__name__,
            "message": error_message,
        }
        if isinstance(error, AgentExecutionError) and error.diagnostics is not None:
            diagnostics = error.diagnostics
            error_data.update(
                failure_type=(
                    diagnostics.failure_type.value
                    if diagnostics.failure_type is not None
                    else None
                ),
                failure_phase=diagnostics.failure_phase,
            )
        active_tracer.emit("error", **error_data)

    finally:
        if workspace is not None:
            try:
                _cleanup_workspace(workspace)
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup failure is traced.
                cleanup_message = summarize_text(cleanup_error)
                active_tracer.emit(
                    "error",
                    phase="cleanup",
                    error_type=type(cleanup_error).__name__,
                    message=cleanup_message,
                )
                if error_message:
                    error_message += f"; workspace cleanup failed: {cleanup_message}"
                else:
                    error_message = f"Workspace cleanup failed: {cleanup_message}"

        passed = error_message is None and after_passed
        active_tracer.emit(
            "run_end",
            case_id=case.id,
            final_status="pass" if passed else "fail",
            passed=passed,
            tests_before_passed=before_passed,
            tests_after_passed=after_passed,
            elapsed_time=round(active_tracer.total_elapsed(), 6),
        )

    return EvalResult(
        case_id=case.id,
        passed=error_message is None and after_passed,
        tests_before_passed=before_passed,
        tests_after_passed=after_passed,
        error=error_message,
        run_id=active_tracer.run_id,
        trace=active_tracer.events,
    )

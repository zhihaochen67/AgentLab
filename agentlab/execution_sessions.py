"""Bounded persistence for non-final resumable evaluation executions."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from agentlab.adapters.base import AgentResumeHandle
from agentlab.models import EvalCase
from agentlab.tracer import TraceEvent, sanitize_data, summarize_text

STATE_ROOT_ENV = "AGENTLAB_STATE_ROOT"
SESSION_SCHEMA_VERSION = 1
MAX_EXECUTION_SESSION_BYTES = 1_000_000
MAX_CASE_EXPECTED_ITEMS = 200
MAX_TRACE_EVENTS = 2_000
_EXECUTION_ID = re.compile(r"[0-9a-f]{32}")
_IDENTITY = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class ExecutionStatus(StrEnum):
    """AgentLab-owned execution progress, separate from final PASS/FAIL runs."""

    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    RESUMING = "RESUMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


TERMINAL_EXECUTION_STATUSES = {
    ExecutionStatus.COMPLETED,
    ExecutionStatus.FAILED,
}


class ExecutionSessionError(RuntimeError):
    """A persisted execution session is missing, malformed, or unsafe."""


@dataclass(frozen=True)
class ExecutionSession:
    """The AgentLab-owned state required to finish one suspended evaluation."""

    execution_id: str
    run_id: str
    case: EvalCase
    adapter: str
    resume_handle: AgentResumeHandle
    workspace: str
    phase: str
    tests_before_passed: bool
    trace: tuple[TraceEvent, ...]
    elapsed_seconds: float
    status: ExecutionStatus
    created_at: str
    updated_at: str
    dataset: str | None = None
    evaluator: str | None = None
    database_path: str | None = None
    schema_version: int = SESSION_SCHEMA_VERSION


def default_state_root() -> Path:
    """Return a cross-platform state directory outside evaluation workspaces."""
    configured = os.environ.get(STATE_ROOT_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ExecutionSessionError(f"{STATE_ROOT_ENV} must be an absolute path.")
        return path.resolve(strict=False)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return (Path(base) / "AgentLab" / "State").resolve(strict=False)
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return (Path(xdg).expanduser() / "agentlab").resolve(strict=False)
    return (Path.home() / ".local" / "state" / "agentlab").resolve(strict=False)


def repo_doctor_state_root(root: Path | None = None) -> Path:
    """Return the fixed AgentLab-managed Repo Doctor state root."""
    return _state_root(root) / "repo-doctor"


def new_execution_id() -> str:
    return secrets.token_hex(16)


def now_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def save_execution_session(
    session: ExecutionSession,
    *,
    root: Path | None = None,
) -> Path:
    """Strictly validate and atomically replace one execution-session file."""
    validated = _session_from_data(_session_to_data(session))
    payload = json.dumps(
        _session_to_data(validated),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_EXECUTION_SESSION_BYTES:
        raise ExecutionSessionError(
            f"Execution session exceeds {MAX_EXECUTION_SESSION_BYTES} bytes."
        )

    state = _state_root(root)
    workspace = Path(validated.workspace)
    if state == workspace or workspace in state.parents:
        raise ExecutionSessionError(
            "AgentLab state root must remain outside the evaluation workspace."
        )
    directory = _sessions_directory(state, create=True)
    destination = directory / f"{validated.execution_id}.json"
    if destination.is_symlink():
        raise ExecutionSessionError(
            "Execution-session destination cannot be a symlink."
        )
    temporary = directory / f".{validated.execution_id}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        if temporary.stat().st_size != len(payload):
            raise ExecutionSessionError(
                "Execution-session temporary write was incomplete."
            )
        os.replace(temporary, destination)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_execution_session(
    execution_id: str,
    *,
    root: Path | None = None,
) -> ExecutionSession:
    """Load one bounded session from its fixed state-root location."""
    identifier = _execution_id(execution_id)
    path = _sessions_directory(root, create=False) / f"{identifier}.json"
    if path.is_symlink():
        raise ExecutionSessionError("Execution-session files cannot be symlinks.")
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read(MAX_EXECUTION_SESSION_BYTES + 1)
    except FileNotFoundError as error:
        raise ExecutionSessionError(
            f"Execution session not found: {identifier}"
        ) from error
    except OSError as error:
        raise ExecutionSessionError(
            f"Could not read execution session: {error}"
        ) from error
    if len(payload) > MAX_EXECUTION_SESSION_BYTES:
        raise ExecutionSessionError(
            f"Execution session exceeds {MAX_EXECUTION_SESSION_BYTES} bytes."
        )
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionSessionError(
            f"Execution session is not valid UTF-8 JSON: {error}"
        ) from error
    session = _session_from_data(data)
    if session.execution_id != identifier:
        raise ExecutionSessionError(
            "Execution-session ID does not match its file name."
        )
    return session


def load_active_execution_session(
    execution_id: str,
    *,
    root: Path | None = None,
) -> ExecutionSession:
    session = load_execution_session(execution_id, root=root)
    if session.status in TERMINAL_EXECUTION_STATUSES:
        raise ExecutionSessionError(
            f"Execution session {session.execution_id} is terminal and cannot be replayed."
        )
    if session.status is not ExecutionStatus.WAITING_FOR_APPROVAL:
        raise ExecutionSessionError(
            f"Execution session {session.execution_id} is {session.status.value}, not resumable."
        )
    return session


def update_execution_status(
    session: ExecutionSession,
    status: ExecutionStatus,
    *,
    root: Path | None = None,
) -> ExecutionSession:
    updated = replace(session, status=status, updated_at=now_timestamp())
    save_execution_session(updated, root=root)
    return updated


def _state_root(root: Path | None) -> Path:
    candidate = (root or default_state_root()).expanduser()
    if not candidate.is_absolute():
        raise ExecutionSessionError("AgentLab state root must be absolute.")
    resolved = candidate.resolve(strict=False)
    if resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise ExecutionSessionError("AgentLab state root must be a real directory.")
    return resolved


def _sessions_directory(root: Path | None, *, create: bool) -> Path:
    state = _state_root(root)
    directory = state / "executions"
    if create:
        state.mkdir(parents=True, exist_ok=True)
        if state.is_symlink():
            raise ExecutionSessionError("AgentLab state root cannot be a symlink.")
        directory.mkdir(exist_ok=True)
    if not directory.is_dir() or directory.is_symlink():
        if not create and not directory.exists():
            raise ExecutionSessionError("AgentLab has no persisted execution sessions.")
        raise ExecutionSessionError(
            "Execution-session directory must be a real directory."
        )
    return directory.resolve(strict=True)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _session_to_data(session: ExecutionSession) -> dict[str, Any]:
    return {
        "schema_version": session.schema_version,
        "execution_id": session.execution_id,
        "run_id": session.run_id,
        "case": {
            "id": session.case.id,
            "repository": session.case.repository,
            "task": session.case.task,
            "expected": sanitize_data(session.case.expected),
        },
        "adapter": session.adapter,
        "resume_handle": {
            "adapter": session.resume_handle.adapter,
            "session_id": session.resume_handle.session_id,
            "reason": session.resume_handle.reason,
            "metadata": dict(session.resume_handle.metadata),
        },
        "workspace": session.workspace,
        "phase": session.phase,
        "tests_before_passed": session.tests_before_passed,
        "trace": [
            {
                "run_id": event.run_id,
                "sequence": event.sequence,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "data": sanitize_data(event.data),
            }
            for event in session.trace
        ],
        "elapsed_seconds": session.elapsed_seconds,
        "status": session.status.value,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "dataset": session.dataset,
        "evaluator": session.evaluator,
        "database_path": session.database_path,
    }


def _session_from_data(value: object) -> ExecutionSession:
    keys = {
        "schema_version",
        "execution_id",
        "run_id",
        "case",
        "adapter",
        "resume_handle",
        "workspace",
        "phase",
        "tests_before_passed",
        "trace",
        "elapsed_seconds",
        "status",
        "created_at",
        "updated_at",
        "dataset",
        "evaluator",
        "database_path",
    }
    data = _object(value, "execution session", keys)
    version = _integer(data["schema_version"], "schema_version")
    if version != SESSION_SCHEMA_VERSION:
        raise ExecutionSessionError(
            f"Unsupported execution-session schema version {version}; "
            f"expected {SESSION_SCHEMA_VERSION}."
        )
    execution_id = _execution_id(data["execution_id"])
    run_id = _identity(data["run_id"], "run_id")
    case_data = _object(data["case"], "case", {"id", "repository", "task", "expected"})
    expected = _bounded_json_object(case_data["expected"], "case.expected")
    case = EvalCase(
        id=_text(case_data["id"], "case.id", 512),
        repository=_text(case_data["repository"], "case.repository", 8_000),
        task=_text(case_data["task"], "case.task", 64_000),
        expected=expected,
    )
    adapter = _identity(data["adapter"], "adapter")
    handle_data = _object(
        data["resume_handle"],
        "resume_handle",
        {"adapter", "session_id", "reason", "metadata"},
    )
    metadata_data = _object(handle_data["metadata"], "resume_handle.metadata")
    if len(metadata_data) > 8:
        raise ExecutionSessionError("resume_handle.metadata has too many entries.")
    metadata = {
        _identity(key, "resume_handle.metadata key"): _text(
            item, f"resume_handle.metadata.{key}", 512
        )
        for key, item in metadata_data.items()
    }
    handle = AgentResumeHandle(
        adapter=_identity(handle_data["adapter"], "resume_handle.adapter"),
        session_id=_identity(handle_data["session_id"], "resume_handle.session_id"),
        reason=_identity(handle_data["reason"], "resume_handle.reason"),
        metadata=metadata,
    )
    if handle.adapter != adapter:
        raise ExecutionSessionError(
            "Resume-handle adapter does not match the session adapter."
        )
    workspace = Path(_text(data["workspace"], "workspace", 8_000))
    if not workspace.is_absolute() or workspace.resolve(strict=False) != workspace:
        raise ExecutionSessionError("workspace must be a canonical absolute path.")
    if data["phase"] != "agent":
        raise ExecutionSessionError("Only the agent phase is resumable in V1.")
    traces = data["trace"]
    if not isinstance(traces, list) or len(traces) > MAX_TRACE_EVENTS:
        raise ExecutionSessionError("trace must be a bounded list.")
    trace = tuple(
        _trace_event(item, index, run_id) for index, item in enumerate(traces)
    )
    elapsed = data["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
        or not math.isfinite(elapsed)
    ):
        raise ExecutionSessionError("elapsed_seconds must be a non-negative number.")
    try:
        status = ExecutionStatus(_text(data["status"], "status", 64))
    except ValueError as error:
        raise ExecutionSessionError("Execution-session status is invalid.") from error
    created_at = _timestamp(data["created_at"], "created_at")
    updated_at = _timestamp(data["updated_at"], "updated_at")
    database_path = _optional_text(data["database_path"], "database_path", 8_000)
    if database_path is not None and not Path(database_path).is_absolute():
        raise ExecutionSessionError("database_path must be absolute when present.")
    return ExecutionSession(
        execution_id=execution_id,
        run_id=run_id,
        case=case,
        adapter=adapter,
        resume_handle=handle,
        workspace=str(workspace),
        phase="agent",
        tests_before_passed=_boolean(
            data["tests_before_passed"], "tests_before_passed"
        ),
        trace=trace,
        elapsed_seconds=float(elapsed),
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        dataset=_optional_text(data["dataset"], "dataset", 8_000),
        evaluator=_optional_text(data["evaluator"], "evaluator", 128),
        database_path=database_path,
        schema_version=version,
    )


def _trace_event(value: object, index: int, run_id: str) -> TraceEvent:
    data = _object(
        value,
        f"trace[{index}]",
        {"run_id", "sequence", "event_type", "timestamp", "data"},
    )
    event_run_id = _identity(data["run_id"], f"trace[{index}].run_id")
    if event_run_id != run_id:
        raise ExecutionSessionError(
            "Trace event run_id does not match the session run_id."
        )
    sequence = _integer(data["sequence"], f"trace[{index}].sequence")
    if sequence != index + 1:
        raise ExecutionSessionError("Trace sequence numbers must be contiguous.")
    return TraceEvent(
        run_id=event_run_id,
        sequence=sequence,
        event_type=_identity(data["event_type"], f"trace[{index}].event_type"),
        timestamp=_timestamp(data["timestamp"], f"trace[{index}].timestamp"),
        data=_bounded_json_object(data["data"], f"trace[{index}].data"),
    )


def _bounded_json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExecutionSessionError(f"{label} must be an object.")
    count = 0

    def validate(item: object, depth: int) -> Any:
        nonlocal count
        count += 1
        if count > MAX_CASE_EXPECTED_ITEMS:
            raise ExecutionSessionError(f"{label} has too many values.")
        if depth > 12:
            raise ExecutionSessionError(f"{label} is nested too deeply.")
        if isinstance(item, dict):
            return {
                _text(key, f"{label} key", 256): validate(child, depth + 1)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [validate(child, depth + 1) for child in item]
        if item is None or isinstance(item, (bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ExecutionSessionError(f"{label} contains a non-finite number.")
            return item
        if isinstance(item, str):
            return summarize_text(item, limit=8_000)
        raise ExecutionSessionError(f"{label} contains a non-JSON value.")

    return validate(value, 0)


def _object(
    value: object,
    label: str,
    keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExecutionSessionError(f"{label} must be an object with string keys.")
    if keys is not None and set(value) != keys:
        raise ExecutionSessionError(f"{label} has missing or unexpected fields.")
    return value


def _execution_id(value: object) -> str:
    text = _text(value, "execution_id", 32)
    if _EXECUTION_ID.fullmatch(text) is None:
        raise ExecutionSessionError("execution_id is invalid.")
    return text


def _identity(value: object, label: str) -> str:
    text = _text(value, label, 128)
    if _IDENTITY.fullmatch(text) is None:
        raise ExecutionSessionError(f"{label} is invalid.")
    return text


def _text(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ExecutionSessionError(f"{label} must be non-empty bounded text.")
    return value


def _optional_text(value: object, label: str, limit: int) -> str | None:
    if value is None:
        return None
    return _text(value, label, limit)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionSessionError(f"{label} must be an integer.")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ExecutionSessionError(f"{label} must be a boolean.")
    return value


def _timestamp(value: object, label: str) -> str:
    text = _text(value, label, 128)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ExecutionSessionError(
            f"{label} must be an ISO-8601 timestamp."
        ) from error
    if parsed.tzinfo is None:
        raise ExecutionSessionError(f"{label} must include a timezone.")
    return text

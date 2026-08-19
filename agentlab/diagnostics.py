"""Structured, evidence-based diagnostics for agent repair failures."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agentlab.tracer import sanitize_data, summarize_text


class AgentFailureType(str, Enum):
    """Stable taxonomy for failures reported by an agent integration."""

    PROVIDER_CONFIGURATION_ERROR = "provider_configuration_error"
    PROVIDER_REQUEST_ERROR = "provider_request_error"
    TIMEOUT = "timeout"
    PATCH_GENERATION_ERROR = "patch_generation_error"
    PATCH_APPLY_ERROR = "patch_apply_error"
    REPAIR_VERIFICATION_FAILED = "repair_verification_failed"
    ROLLBACK_ERROR = "rollback_error"
    AGENT_PROCESS_ERROR = "agent_process_error"
    UNKNOWN_AGENT_ERROR = "unknown_agent_error"


@dataclass(frozen=True)
class AgentDiagnostics:
    """Facts that can be established from one agent process execution."""

    failure_type: AgentFailureType | None
    failure_phase: str | None
    returncode: int | None
    patch_applied: bool | None = None
    verification_failed: bool | None = None
    rollback_attempted: bool | None = None
    rollback_succeeded: bool | None = None
    verification_command: str | None = None
    verification_returncode: int | None = None
    verification_output: str | None = None
    patch_diff: str | None = None
    stdout_summary: str = ""
    stderr_summary: str = ""

    def to_trace_data(self) -> dict[str, Any]:
        """Return bounded, redacted diagnostics suitable for trace persistence."""
        return sanitize_data(
            {
                "failure_type": (
                    self.failure_type.value if self.failure_type is not None else None
                ),
                "failure_phase": self.failure_phase,
                "returncode": self.returncode,
                "patch_applied": self.patch_applied,
                "verification_failed": self.verification_failed,
                "rollback_attempted": self.rollback_attempted,
                "rollback_succeeded": self.rollback_succeeded,
                "verification_command": self.verification_command,
                "verification_returncode": self.verification_returncode,
                "verification_output": self.verification_output,
                "patch_diff": self.patch_diff,
                "stdout_summary": self.stdout_summary,
                "stderr_summary": self.stderr_summary,
            }
        )


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_VERIFICATION_FAILED = re.compile(r"(?im)^Verification failed:\s*(.+)$")


def diagnose_repo_doctor(
    returncode: int | None,
    stdout: str = "",
    stderr: str = "",
    *,
    patch_diff: str | None = None,
    process_timed_out: bool = False,
) -> AgentDiagnostics:
    """Classify Repo Doctor using its documented CLI evidence, not exit code alone."""
    clean_stdout = _ANSI_ESCAPE.sub("", stdout)
    clean_stderr = _ANSI_ESCAPE.sub("", stderr)
    evidence = f"{clean_stdout}\n{clean_stderr}".strip()
    lowered = evidence.lower()

    patch_applied = "patch applied" in lowered
    verification_failed = True if "verification failed" in lowered else None
    if "verification passed" in lowered:
        verification_failed = False
    rollback_attempted = "rolling back" in lowered or "rollback failed" in lowered
    rollback_succeeded: bool | None = None
    if "repository restored successfully" in lowered:
        rollback_succeeded = True
    elif rollback_attempted:
        rollback_succeeded = False

    verification_output = None
    match = _VERIFICATION_FAILED.search(evidence)
    if match:
        verification_output = match.group(1).strip()
    elif "verification passed" in lowered:
        verification_output = "Verification passed"

    failure_type: AgentFailureType | None = None
    failure_phase: str | None = None
    if process_timed_out:
        failure_type = AgentFailureType.TIMEOUT
        failure_phase = "process"
    elif returncode not in (None, 0):
        failure_type, failure_phase = _classify_repo_doctor_failure(
            returncode,
            lowered,
        )

    return AgentDiagnostics(
        failure_type=failure_type,
        failure_phase=failure_phase,
        returncode=returncode,
        patch_applied=patch_applied,
        verification_failed=verification_failed,
        rollback_attempted=rollback_attempted,
        rollback_succeeded=rollback_succeeded,
        verification_output=(
            summarize_text(verification_output) if verification_output else None
        ),
        patch_diff=summarize_text(patch_diff) if patch_diff else None,
        stdout_summary=summarize_text(clean_stdout),
        stderr_summary=summarize_text(clean_stderr),
    )


def diagnostics_from_trace_data(data: Any) -> dict[str, Any] | None:
    """Load and re-sanitize optional diagnostics from new or legacy trace data."""
    if not isinstance(data, Mapping):
        return None
    diagnostics = data.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        return sanitize_data(diagnostics)

    returncode = data.get("returncode")
    if (
        data.get("adapter") == "RepoDoctorAdapter"
        and data.get("status") == "error"
        and (returncode is None or isinstance(returncode, int))
    ):
        inferred = diagnose_repo_doctor(
            returncode,
            str(data.get("stdout", "")),
            str(data.get("stderr", "")),
        )
        if inferred.failure_type is not None:
            return inferred.to_trace_data()
    return None


def _classify_repo_doctor_failure(
    returncode: int,
    evidence: str,
) -> tuple[AgentFailureType, str]:
    if "rollback failed" in evidence:
        return AgentFailureType.ROLLBACK_ERROR, "rollback"
    if "patch applied" in evidence and (
        "verification failed" in evidence or "rolling back" in evidence
    ):
        return AgentFailureType.REPAIR_VERIFICATION_FAILED, "verification"
    if "required configuration is missing" in evidence or (
        "repo_doctor_request_timeout" in evidence
        and "positive finite number" in evidence
    ):
        return AgentFailureType.PROVIDER_CONFIGURATION_ERROR, "configuration"
    if "timed out" in evidence or "timeout" in evidence:
        return AgentFailureType.TIMEOUT, "provider_request"
    if "could not apply patch" in evidence or "before patch application" in evidence:
        return AgentFailureType.PATCH_APPLY_ERROR, "patch_apply"
    patch_generation_markers = (
        "ai patch response",
        "unexpected patch object",
        "patch path is unsafe",
        "patch targets an ignored",
        "patch confidence",
        "patch replacement",
        "patch changes too much",
        "patch target",
        "patch old_text",
        "patch file does not match",
    )
    if any(marker in evidence for marker in patch_generation_markers):
        return AgentFailureType.PATCH_GENERATION_ERROR, "patch_generation"
    provider_markers = (
        "ai provider",
        "network error",
        "rate limit",
        "httpx",
    )
    if any(marker in evidence for marker in provider_markers):
        return AgentFailureType.PROVIDER_REQUEST_ERROR, "provider_request"
    if "verification" in evidence:
        return AgentFailureType.REPAIR_VERIFICATION_FAILED, "verification"
    if returncode < 0 or not evidence:
        return AgentFailureType.AGENT_PROCESS_ERROR, "process"
    return AgentFailureType.UNKNOWN_AGENT_ERROR, "agent"

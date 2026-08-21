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
    analysis_summary: str | None = None
    selected_finding: dict[str, Any] | None = None
    behavioral_contract: dict[str, Any] | None = None
    final_status: str | None = None
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
                "analysis_summary": self.analysis_summary,
                "selected_finding": self.selected_finding,
                "behavioral_contract": self.behavioral_contract,
                "final_status": self.final_status,
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


def diagnose_repo_doctor_report(
    report: Any,
    returncode: int | None,
    stdout: str = "",
    stderr: str = "",
    *,
    fallback_patch_diff: str | None = None,
    process_timed_out: bool = False,
) -> AgentDiagnostics:
    """Prefer Repo Doctor's structured report and retain stdout parsing as fallback."""
    fallback = diagnose_repo_doctor(
        returncode,
        stdout,
        stderr,
        patch_diff=fallback_patch_diff,
        process_timed_out=process_timed_out,
    )
    if not isinstance(report, Mapping):
        return fallback

    safe = sanitize_data(report)
    patch = safe.get("patch")
    patch_mapping = patch if isinstance(patch, Mapping) else {}
    verification = safe.get("verification")
    verification_mapping = verification if isinstance(verification, Mapping) else {}
    commands = verification_mapping.get("commands")
    command_items = [item for item in commands if isinstance(item, Mapping)] if isinstance(
        commands, list
    ) else []
    selected_command = next(
        (
            item
            for item in command_items
            if isinstance(item.get("returncode"), int)
            and not isinstance(item.get("returncode"), bool)
            and item.get("returncode") != 0
        ),
        command_items[-1] if command_items else None,
    )
    verification_command = None
    verification_returncode = None
    if selected_command is not None:
        raw_command = selected_command.get("command")
        if isinstance(raw_command, list):
            verification_command = " ".join(str(item) for item in raw_command)
        raw_returncode = selected_command.get("returncode")
        if isinstance(raw_returncode, int) and not isinstance(raw_returncode, bool):
            verification_returncode = raw_returncode

    output_blocks = []
    for item in command_items:
        name = item.get("name") or "Verification"
        output = "\n".join(
            str(value)
            for value in (item.get("stdout_summary"), item.get("stderr_summary"))
            if value
        )
        return_code = item.get("returncode")
        output_blocks.append(f"{name} (return code {return_code})\n{output}".strip())
    verification_output = "\n\n".join(output_blocks)
    if not verification_output and verification_mapping.get("summary"):
        verification_output = str(verification_mapping["summary"])

    final_status = safe.get("final_status")
    report_patch_applied = safe.get("patch_applied")
    patch_applied = (
        report_patch_applied if isinstance(report_patch_applied, bool) else fallback.patch_applied
    )
    report_rollback_attempted = safe.get("rollback_attempted")
    rollback_attempted = (
        report_rollback_attempted
        if isinstance(report_rollback_attempted, bool)
        else fallback.rollback_attempted
    )
    report_rollback_succeeded = safe.get("rollback_succeeded")
    rollback_succeeded = (
        report_rollback_succeeded
        if isinstance(report_rollback_succeeded, bool)
        else fallback.rollback_succeeded
    )
    verification_failed = fallback.verification_failed
    if command_items:
        verification_failed = any(item.get("passed") is not True for item in command_items)
    elif final_status == "kept":
        verification_failed = False
    elif final_status in {"rolled_back", "rollback_failed", "verification_failed_pending_rollback"}:
        verification_failed = True

    failure_type = fallback.failure_type
    failure_phase = fallback.failure_phase
    if final_status == "rollback_failed":
        failure_type = AgentFailureType.ROLLBACK_ERROR
        failure_phase = "rollback"
    elif returncode not in (None, 0) and verification_failed:
        failure_type = AgentFailureType.REPAIR_VERIFICATION_FAILED
        failure_phase = "verification"

    selected_finding = safe.get("selected_finding")
    behavioral_contract = safe.get("behavioral_contract")
    return AgentDiagnostics(
        failure_type=failure_type,
        failure_phase=failure_phase,
        returncode=returncode,
        patch_applied=patch_applied,
        verification_failed=verification_failed,
        rollback_attempted=rollback_attempted,
        rollback_succeeded=rollback_succeeded,
        verification_command=verification_command or fallback.verification_command,
        verification_returncode=(
            verification_returncode
            if verification_returncode is not None
            else fallback.verification_returncode
        ),
        verification_output=(
            summarize_text(verification_output)
            if verification_output
            else fallback.verification_output
        ),
        patch_diff=(
            summarize_text(patch_mapping["diff"])
            if isinstance(patch_mapping.get("diff"), str)
            else fallback.patch_diff
        ),
        analysis_summary=(
            summarize_text(safe["analysis_summary"])
            if isinstance(safe.get("analysis_summary"), str)
            else None
        ),
        selected_finding=(
            dict(selected_finding) if isinstance(selected_finding, Mapping) else None
        ),
        behavioral_contract=(
            dict(behavioral_contract) if isinstance(behavioral_contract, Mapping) else None
        ),
        final_status=(summarize_text(final_status) if isinstance(final_status, str) else None),
        stdout_summary=fallback.stdout_summary,
        stderr_summary=fallback.stderr_summary,
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

import tempfile
from pathlib import Path

import pytest

from agentlab.adapters import AgentAdapter, AgentExecutionError, AgentRunResult
from agentlab.diagnostics import AgentFailureType, diagnose_repo_doctor
from agentlab.models import EvalCase, EvalResult
from agentlab.runner import evaluate_case
from agentlab.storage import SQLiteStorage
from agentlab.tracer import TraceEvent


def test_repair_verification_failure_and_successful_rollback_are_classified() -> None:
    diagnostics = diagnose_repo_doctor(
        1,
        "Patch applied\n"
        "Verification failed: Verification failed: Python tests.\n"
        "Rolling back\n"
        "Repository restored successfully\n",
    )

    assert diagnostics.failure_type is AgentFailureType.REPAIR_VERIFICATION_FAILED
    assert diagnostics.failure_phase == "verification"
    assert diagnostics.returncode == 1
    assert diagnostics.patch_applied is True
    assert diagnostics.verification_failed is True
    assert diagnostics.rollback_attempted is True
    assert diagnostics.rollback_succeeded is True
    assert diagnostics.verification_output == "Verification failed: Python tests."
    assert diagnostics.verification_command is None
    assert diagnostics.verification_returncode is None


def test_provider_configuration_error_is_classified() -> None:
    diagnostics = diagnose_repo_doctor(
        2,
        "Error: AI analysis requested, but required configuration is missing: "
        "REPO_DOCTOR_API_KEY, REPO_DOCTOR_BASE_URL, REPO_DOCTOR_MODEL.",
    )

    assert diagnostics.failure_type is AgentFailureType.PROVIDER_CONFIGURATION_ERROR
    assert diagnostics.failure_phase == "configuration"
    assert diagnostics.patch_applied is False


def test_timeout_is_classified() -> None:
    provider_timeout = diagnose_repo_doctor(
        2,
        "Error: AI provider request timed out.",
    )
    process_timeout = diagnose_repo_doctor(None, process_timed_out=True)

    assert provider_timeout.failure_type is AgentFailureType.TIMEOUT
    assert provider_timeout.failure_phase == "provider_request"
    assert process_timeout.failure_type is AgentFailureType.TIMEOUT
    assert process_timeout.failure_phase == "process"


def test_unknown_error_is_the_safe_fallback() -> None:
    diagnostics = diagnose_repo_doctor(7, "Unexpected failure with no known marker")

    assert diagnostics.failure_type is AgentFailureType.UNKNOWN_AGENT_ERROR
    assert diagnostics.failure_phase == "agent"


@pytest.mark.parametrize(
    ("stdout", "expected"),
    (
        (
            "Error: AI provider request failed due to a network error.",
            AgentFailureType.PROVIDER_REQUEST_ERROR,
        ),
        (
            "Error: AI patch response is missing fields: new_text.",
            AgentFailureType.PATCH_GENERATION_ERROR,
        ),
        (
            "Error: Could not apply patch to parser.py.",
            AgentFailureType.PATCH_APPLY_ERROR,
        ),
        (
            "Error: Rollback failed for parser.py; restore it from Git.",
            AgentFailureType.ROLLBACK_ERROR,
        ),
    ),
)
def test_additional_failure_taxonomy(stdout, expected) -> None:
    assert diagnose_repo_doctor(2, stdout).failure_type is expected


class DiagnosticFailureAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.workspace: Path | None = None

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        del task
        self.workspace = workspace
        stdout = (
            "Patch applied\nVerification failed: Python tests.\n"
            "Rolling back\nRepository restored successfully\n"
        )
        diagnostics = diagnose_repo_doctor(1, stdout)
        result = AgentRunResult(1, stdout, "", diagnostics)
        raise AgentExecutionError(
            "Repo Doctor failed during verification (repair_verification_failed)",
            result,
        )


def test_runner_emits_structured_diagnostics_and_cleans_workspace() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-diagnostics-") as directory:
        repository = Path(directory) / "repository"
        repository.mkdir()
        (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repository / "test_module.py").write_text(
            "from module import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
            encoding="utf-8",
        )
        adapter = DiagnosticFailureAdapter()

        result = evaluate_case(
            EvalCase("case-001", str(repository), "Fix VALUE"),
            adapter=adapter,
        )

        agent_end = next(event for event in result.trace if event.event_type == "agent_end")
        error = next(event for event in result.trace if event.event_type == "error")
        assert agent_end.data["diagnostics"]["failure_type"] == (
            "repair_verification_failed"
        )
        assert agent_end.data["diagnostics"]["rollback_succeeded"] is True
        assert error.data["failure_type"] == "repair_verification_failed"
        assert error.data["failure_phase"] == "verification"
        assert adapter.workspace is not None
        assert not adapter.workspace.exists()


def test_patch_diff_is_redacted_and_persisted(monkeypatch) -> None:
    secret = "diagnostic-secret-value-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    diagnostics = diagnose_repo_doctor(
        1,
        "Patch applied\nVerification failed: Python tests\nRolling back\n"
        "Repository restored successfully",
        patch_diff=f"-password=old\n+password={secret}\n",
    )
    trace_data = diagnostics.to_trace_data()
    events = (
        TraceEvent(
            "diagnostic-run",
            1,
            "run_start",
            "2026-08-18T01:00:00+00:00",
            {"case_id": "case-001", "adapter": "RepoDoctorAdapter"},
        ),
        TraceEvent(
            "diagnostic-run",
            2,
            "agent_end",
            "2026-08-18T01:00:01+00:00",
            {"status": "error", "returncode": 1, "diagnostics": trace_data},
        ),
        TraceEvent(
            "diagnostic-run",
            3,
            "run_end",
            "2026-08-18T01:00:02+00:00",
            {"passed": False, "elapsed_time": 2.0},
        ),
    )
    result = EvalResult(
        case_id="case-001",
        passed=False,
        tests_before_passed=False,
        tests_after_passed=False,
        error="repair failed",
        run_id="diagnostic-run",
        trace=events,
    )

    with tempfile.TemporaryDirectory(prefix="agentlab-diagnostics-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        storage.save_run(result, "dataset.yaml")
        stored = storage.get_trace_events("diagnostic-run")[1].data["diagnostics"]

        assert stored["failure_type"] == "repair_verification_failed"
        assert "[REDACTED]" in stored["patch_diff"]
        assert secret not in repr(stored)
        assert secret.encode() not in database.read_bytes()

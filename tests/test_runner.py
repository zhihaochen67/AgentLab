import shutil
import tempfile
from pathlib import Path

from agentlab.adapters import AgentAdapter, AgentRunResult
from agentlab.models import EvalCase
from agentlab.runner import create_workspace, evaluate_case, load_dataset


class FixingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        self.calls.append((workspace, task))
        (workspace / "calculator.py").write_text(
            "def add(left, right):\n    return left + right\n",
            encoding="utf-8",
        )
        return AgentRunResult(0, "repair complete", "")


class FailingAdapter(AgentAdapter):
    def __init__(self, message: str) -> None:
        self.message = message
        self.workspace: Path | None = None

    def repair(self, workspace: Path, task: str) -> None:
        self.workspace = workspace
        raise RuntimeError(self.message)


class SecretReportingAdapter(FixingAdapter):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        super().repair(workspace, task)
        return AgentRunResult(
            0,
            f"Authorization: Bearer {self.secret}; API_KEY={self.secret}",
            f"token={self.secret}",
        )


class CalculateTotalAdapter(AgentAdapter):
    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        path = workspace / "inventory.py"
        source = path.read_text(encoding="utf-8")
        path.write_text(
            source.replace("total -= price", "total += price"),
            encoding="utf-8",
        )
        return AgentRunResult(0, "fixed inventory.py", "")


def make_failing_repository(root: Path) -> Path:
    repository = root / "repository"
    repository.mkdir()
    (repository / "calculator.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    (repository / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return repository


def test_load_dataset() -> None:
    cases = load_dataset("datasets/repo_doctor_basic.yaml")

    assert len(cases) == 1
    assert cases[0].id == "calculate_total_001"
    assert cases[0].expected["tests_pass"] is True


def test_workspace_is_an_isolated_copy() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        workspace = create_workspace(str(repository))

        try:
            (workspace / "calculator.py").write_text("changed\n", encoding="utf-8")
            assert "left - right" in (repository / "calculator.py").read_text(
                encoding="utf-8"
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def test_evaluate_case_runs_before_agent_and_after() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        original = (repository / "calculator.py").read_bytes()
        adapter = FixingAdapter()
        case = EvalCase("addition", str(repository), "Fix addition")

        result = evaluate_case(case, adapter=adapter)

        assert result.tests_before_passed is False
        assert result.tests_after_passed is True
        assert result.passed is True
        assert result.error is None
        assert len(adapter.calls) == 1
        assert adapter.calls[0][1] == "Fix addition"
        assert (repository / "calculator.py").read_bytes() == original
        assert not adapter.calls[0][0].exists()

        event_types = [event.event_type for event in result.trace]
        assert event_types == [
            "run_start",
            "pytest_before_start",
            "pytest_before_end",
            "agent_start",
            "agent_end",
            "pytest_after_start",
            "pytest_after_end",
            "run_end",
        ]
        assert result.run_id
        assert all(event.run_id == result.run_id for event in result.trace)
        assert result.trace[2].data["passed"] is False
        assert result.trace[4].data["returncode"] == 0
        assert result.trace[6].data["passed"] is True
        assert result.trace[-1].data["final_status"] == "pass"


def test_adapter_failure_records_error_and_cleans_workspace() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        adapter = FailingAdapter("agent exploded")
        case = EvalCase("addition", str(repository), "Fix addition")

        result = evaluate_case(case, adapter=adapter)

        assert result.passed is False
        assert result.error == "agent exploded"
        assert adapter.workspace is not None
        assert not adapter.workspace.exists()
        assert [event.event_type for event in result.trace] == [
            "run_start",
            "pytest_before_start",
            "pytest_before_end",
            "agent_start",
            "agent_end",
            "error",
            "run_end",
        ]
        assert result.trace[4].data["status"] == "error"
        assert result.trace[5].data["phase"] == "agent"
        assert result.trace[-1].data["final_status"] == "fail"


def test_pytest_exception_records_error_and_run_end(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        adapter = FixingAdapter()

        def fail_pytest(_workspace: Path):
            raise OSError("pytest unavailable")

        monkeypatch.setattr("agentlab.runner.run_pytest", fail_pytest)
        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=adapter,
        )

        assert result.error == "pytest unavailable"
        assert [event.event_type for event in result.trace] == [
            "run_start",
            "pytest_before_start",
            "pytest_before_end",
            "error",
            "run_end",
        ]
        assert result.trace[2].data["status"] == "error"
        assert result.trace[3].data["phase"] == "pytest_before"


def test_trace_redacts_secret_from_task_and_agent_output(monkeypatch) -> None:
    secret = "test-secret-value-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        result = evaluate_case(
            EvalCase("addition", str(repository), f"Fix addition using {secret}"),
            adapter=SecretReportingAdapter(secret),
        )

        rendered_trace = repr(result.trace)
        assert secret not in rendered_trace
        assert "[REDACTED]" in rendered_trace


def test_evaluation_does_not_modify_original_fixture() -> None:
    fixture = Path("fixtures/calculate_total_001")
    before = {path.name: path.read_bytes() for path in fixture.iterdir() if path.is_file()}
    case = EvalCase("calculate_total", str(fixture), "Fix calculate_total")

    result = evaluate_case(case, adapter=CalculateTotalAdapter())

    after = {path.name: path.read_bytes() for path in fixture.iterdir() if path.is_file()}
    assert result.tests_before_passed is False
    assert result.tests_after_passed is True
    assert before == after

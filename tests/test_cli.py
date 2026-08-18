import tempfile
from pathlib import Path

from typer.testing import CliRunner

from agentlab.adapters import AgentAdapter, AgentRunResult
from agentlab.cli import app
from agentlab.models import EvalCase
from agentlab.runner import evaluate_case as run_evaluation
from agentlab.storage import StorageError


class CapturingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.workspace: Path | None = None

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        self.workspace = workspace
        (workspace / "calculator.py").write_text(
            "def add(left, right):\n    return left + right\n",
            encoding="utf-8",
        )
        return AgentRunResult(0, "fixed", "")


class FailingStorage:
    def save_run(self, result, dataset) -> None:
        raise StorageError("simulated database failure")


def test_database_write_failure_does_not_leak_workspace(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-cli-") as directory:
        repository = Path(directory) / "repository"
        repository.mkdir()
        (repository / "calculator.py").write_text(
            "def add(left, right):\n    return left - right\n",
            encoding="utf-8",
        )
        (repository / "test_calculator.py").write_text(
            "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        case = EvalCase("addition", str(repository), "Fix addition")
        adapter = CapturingAdapter()

        monkeypatch.setattr("agentlab.cli.load_dataset", lambda _dataset: [case])
        monkeypatch.setattr(
            "agentlab.cli.evaluate_case",
            lambda selected_case: run_evaluation(selected_case, adapter=adapter),
        )
        monkeypatch.setattr("agentlab.cli._open_storage", lambda: FailingStorage())

        result = CliRunner().invoke(app, ["eval", "dataset.yaml"])

        assert result.exit_code == 1
        assert "Could not persist run" in result.stdout
        assert "simulated" in result.stdout
        assert adapter.workspace is not None
        assert not adapter.workspace.exists()

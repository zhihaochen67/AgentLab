import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentlab.adapters import AgentAdapter, AgentRunResult
from agentlab.cli import app
from agentlab.models import EvalCase, Experiment
from agentlab.runner import evaluate_case as run_evaluation
from agentlab.storage import SQLiteStorage, StorageError


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


def test_cli_experiment_preflight_aborts_before_any_run(monkeypatch) -> None:
    secret = "cli-provider-secret-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    monkeypatch.delenv("REPO_DOCTOR_BASE_URL", raising=False)
    monkeypatch.delenv("REPO_DOCTOR_MODEL", raising=False)
    with tempfile.TemporaryDirectory(prefix="agentlab-cli-experiment-") as directory:
        database = Path(directory) / "agentlab.db"
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))
        monkeypatch.setattr(
            "agentlab.cli.load_dataset",
            lambda _dataset, **_kwargs: [EvalCase("case-a", "unused", "Fix A")],
        )
        monkeypatch.setattr(
            "agentlab.experiments.evaluate_case",
            lambda *_args, **_kwargs: pytest.fail("evaluation must not run"),
        )

        result = CliRunner().invoke(
            app,
            [
                "experiment",
                "dataset.yaml",
                "--trials",
                "2",
                "--case",
                "case-a",
                "--label",
                "blocked",
            ],
        )
        storage = SQLiteStorage(database)
        experiments = storage.list_experiments()

        assert result.exit_code == 1
        assert "REPO_DOCTOR_BASE_URL" in result.stdout
        assert "REPO_DOCTOR_MODEL" in result.stdout
        assert "Runs executed: 0" in result.stdout
        assert secret not in result.stdout
        assert len(experiments) == 1
        assert experiments[0].status == "aborted"
        assert experiments[0].total_runs == 0
        assert experiments[0].prompt_variant == "baseline-v1"
        assert storage.list_runs() == ()
        assert secret.encode() not in database.read_bytes()


def test_cli_invalid_api_key_is_redacted_and_executes_no_runs(monkeypatch) -> None:
    secret = "your api key must be replaced in cli"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "model-name")
    with tempfile.TemporaryDirectory(prefix="agentlab-cli-experiment-") as directory:
        database = Path(directory) / "agentlab.db"
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))
        monkeypatch.setattr(
            "agentlab.cli.load_dataset",
            lambda _dataset, **_kwargs: [EvalCase("case-a", "unused", "Fix A")],
        )
        monkeypatch.setattr(
            "agentlab.experiments.evaluate_case",
            lambda *_args, **_kwargs: pytest.fail("evaluation must not run"),
        )

        result = CliRunner().invoke(
            app,
            ["experiment", "dataset.yaml", "--trials", "2"],
        )
        storage = SQLiteStorage(database)
        experiments = storage.list_experiments()

        assert result.exit_code == 1
        assert "REPO_DOCTOR_API_KEY appears invalid" in result.stdout
        assert "Runs executed: 0" in result.stdout
        assert secret not in result.stdout
        assert len(experiments) == 1
        assert experiments[0].status == "aborted"
        assert experiments[0].total_runs == 0
        assert storage.list_runs() == ()
        assert secret.encode() not in database.read_bytes()


def test_cli_experiment_records_explicit_variant_metadata(monkeypatch) -> None:
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", "placeholder api key")
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "deepseek-v4-flash")
    with tempfile.TemporaryDirectory(prefix="agentlab-cli-variant-") as directory:
        database = Path(directory) / "agentlab.db"
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))
        monkeypatch.setattr(
            "agentlab.cli.load_dataset",
            lambda _dataset, **_kwargs: [EvalCase("case-a", "unused", "Fix A")],
        )
        monkeypatch.setattr(
            "agentlab.experiments.evaluate_case",
            lambda *_args, **_kwargs: pytest.fail("evaluation must not run"),
        )

        result = CliRunner().invoke(
            app,
            [
                "experiment",
                "dataset.yaml",
                "--agent-version",
                "repo-doctor-0.2.0",
                "--prompt-variant",
                "candidate-v2",
                "--notes",
                "candidate prompt trial",
            ],
        )
        experiments = SQLiteStorage(database).list_experiments()

        assert result.exit_code == 1
        assert len(experiments) == 1
        assert experiments[0].agent_version == "repo-doctor-0.2.0"
        assert experiments[0].prompt_variant == "candidate-v2"
        assert experiments[0].notes == "candidate prompt trial"
        assert experiments[0].total_runs == 0


def test_cli_lists_and_shows_persisted_experiment(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-cli-experiment-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        experiment = Experiment(
            experiment_id="experiment-cli",
            label="CLI baseline",
            dataset="dataset.yaml",
            adapter="CapturingAdapter",
            model=None,
            trials_per_case=1,
            total_cases=1,
            total_runs=0,
            started_at="2026-08-19T01:00:00+00:00",
            finished_at=None,
            status="running",
            agent_version="repo-doctor-0.2.0",
            prompt_variant="baseline-v1",
            notes="official baseline",
        )
        storage.create_experiment(experiment)
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
        evaluated = run_evaluation(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=CapturingAdapter(),
        )
        storage.save_run(
            replace(evaluated, experiment_id="experiment-cli", trial_index=1),
            "dataset.yaml",
        )
        storage.finish_experiment(
            "experiment-cli",
            "completed",
            "2026-08-19T01:00:02+00:00",
        )
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))

        listed = CliRunner().invoke(app, ["experiments"])
        shown = CliRunner().invoke(app, ["experiment-show", "experiment-cli"])

        assert listed.exit_code == 0
        assert "experiment-cli" in listed.stdout
        assert "CLI baseline" in listed.stdout
        assert "repo-doctor-0.2.0" in listed.stdout
        assert "baseline-v1" in listed.stdout
        assert "not recorded" in listed.stdout
        assert shown.exit_code == 0
        assert "Agent Version:" in shown.stdout
        assert "repo-doctor-0.2.0" in shown.stdout
        assert "Prompt Variant:" in shown.stdout
        assert "baseline-v1" in shown.stdout
        assert "Notes:" in shown.stdout
        assert "official baseline" in shown.stdout
        assert "Success Rate:" in shown.stdout
        assert "100.0%" in shown.stdout
        assert "addition" in shown.stdout

def test_cli_lists_available_agents() -> None:
    result = CliRunner().invoke(
        app,
        [
            "agents",
            "list",
        ],
    )

    assert result.exit_code == 0
    assert "repo_doctor" in result.stdout
    assert "AI coding repair agent evaluated by AgentLab" in result.stdout

def test_cli_experiment_accepts_agent_selection(monkeypatch) -> None:
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", "placeholder api key")
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "deepseek-v4-flash")

    with tempfile.TemporaryDirectory(prefix="agentlab-cli-agent-") as directory:
        database = Path(directory) / "agentlab.db"
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))

        monkeypatch.setattr(
            "agentlab.cli.load_dataset",
            lambda _dataset, **_kwargs: [
                EvalCase("case-a", "unused", "Fix A")
            ],
        )

        monkeypatch.setattr(
            "agentlab.experiments.evaluate_case",
            lambda *_args, **_kwargs: pytest.fail(
                "evaluation must not run"
            ),
        )

        result = CliRunner().invoke(
            app,
            [
                "experiment",
                "dataset.yaml",
                "--agent",
                "repo_doctor",
            ],
        )

        experiments = SQLiteStorage(database).list_experiments()

        assert result.exit_code == 1
        assert len(experiments) == 1
        assert experiments[0].status == "aborted"
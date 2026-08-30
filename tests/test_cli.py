import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from agentlab.adapters import AgentAdapter, AgentRunResult
from agentlab.cli import app
from agentlab.execution_sessions import ExecutionStatus
from agentlab.models import (
    CaseExperimentMetrics,
    EvalCase,
    EvalResult,
    EvaluationSuspended,
    Experiment,
    ExperimentMetrics,
)
from agentlab.runner import evaluate_case as run_evaluation
from agentlab.storage import SQLiteStorage, StorageError
from agentlab.tracer import TraceEvent


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
            lambda selected_case, **_kwargs: run_evaluation(
                selected_case, adapter=adapter
            ),
        )
        monkeypatch.setattr("agentlab.cli._open_storage", lambda: FailingStorage())

        result = CliRunner().invoke(
            app, ["eval", "dataset.yaml", "--agent", "mock_agent"]
        )

        assert result.exit_code == 1
        assert "Could not persist run" in result.stdout
        assert "simulated" in result.stdout
        assert adapter.workspace is not None
        assert not adapter.workspace.exists()


def test_cli_experiment_preflight_aborts_before_any_run(
    monkeypatch,
    fake_repo_doctor_project: Path,
) -> None:
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


def test_cli_invalid_api_key_is_redacted_and_executes_no_runs(
    monkeypatch,
    fake_repo_doctor_project: Path,
) -> None:
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


def test_cli_experiment_records_explicit_variant_metadata(
    monkeypatch,
    fake_repo_doctor_project: Path,
) -> None:
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


def test_cli_input_errors_are_concise_and_do_not_render_tracebacks() -> None:
    missing = CliRunner().invoke(
        app,
        ["eval", "does-not-exist.yaml", "--agent", "mock_agent"],
    )
    unknown = CliRunner().invoke(
        app,
        ["eval", "does-not-exist.yaml", "--agent", "no-such-agent"],
    )

    assert missing.exit_code == 1
    assert "Could not start evaluation" in missing.stdout
    assert "Cannot read dataset" in missing.stdout
    assert "Traceback" not in missing.stdout
    assert unknown.exit_code == 1
    assert "Unknown agent: no-such-agent" in unknown.stdout
    assert "Traceback" not in unknown.stdout


def test_cli_execution_history_is_inspectable(monkeypatch) -> None:
    timestamp = "2026-08-28T00:00:00+00:00"
    session = SimpleNamespace(
        execution_id="a" * 32,
        run_id="run-1",
        case=SimpleNamespace(id="case-a"),
        adapter="repo_doctor",
        status=SimpleNamespace(value="WAITING_FOR_APPROVAL"),
        resume_handle=SimpleNamespace(reason="approval_required"),
        dataset="dataset.yaml",
        created_at=timestamp,
        updated_at="2026-08-28T00:01:00+00:00",
        elapsed_seconds=1.0,
        trace=(
            TraceEvent(
                "run-1",
                1,
                "run_start",
                timestamp,
                {"case_id": "case-a", "adapter": "RepoDoctorAdapter"},
            ),
            TraceEvent(
                "run-1",
                2,
                "run_suspended",
                timestamp,
                {"reason": "approval_required", "elapsed_time": 1.0},
            ),
        ),
    )
    monkeypatch.setattr("agentlab.cli.list_execution_sessions", lambda limit: (session,))
    monkeypatch.setattr("agentlab.cli.load_execution_session", lambda _value: session)

    listing = CliRunner().invoke(app, ["executions"])
    detail = CliRunner().invoke(app, ["execution-show", "a" * 32])

    assert listing.exit_code == 0
    assert "WAITING_FOR_APPROVAL" in listing.stdout
    assert "case-a" in listing.stdout
    assert detail.exit_code == 0
    assert "Trace events: 2" in detail.stdout
    assert "approval_required" in detail.stdout
    assert "run_suspended" in detail.stdout
    assert "WAITING" in detail.stdout


def test_cli_experiment_accepts_agent_selection(
    monkeypatch,
    fake_repo_doctor_project: Path,
) -> None:
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", "placeholder api key")
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "deepseek-v4-flash")

    with tempfile.TemporaryDirectory(prefix="agentlab-cli-agent-") as directory:
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
                "--agent",
                "repo_doctor",
            ],
        )

        experiments = SQLiteStorage(database).list_experiments()

        assert result.exit_code == 1
        assert len(experiments) == 1
        assert experiments[0].status == "aborted"


def test_cli_report_writes_json_file(monkeypatch, tmp_path) -> None:
    experiment = Experiment(
        experiment_id="experiment-cli",
        label="CLI report",
        dataset="dataset.yaml",
        adapter="MockAgentAdapter",
        model=None,
        agent_version="mock-v1",
        prompt_variant="baseline",
        notes=None,
        status="completed",
        trials_per_case=1,
        total_cases=1,
        total_runs=1,
        started_at="2026-08-26T00:00:00+00:00",
        finished_at="2026-08-26T00:00:01+00:00",
    )
    metrics = ExperimentMetrics(
        experiment_id="experiment-cli",
        total_runs=1,
        passed_runs=1,
        failed_runs=0,
        success_rate=1.0,
        average_latency=0.25,
        per_case=(
            CaseExperimentMetrics(
                case_id="case-a",
                total_runs=1,
                passed_runs=1,
                failed_runs=0,
                success_rate=1.0,
                average_latency=0.25,
            ),
        ),
        failure_types=(),
    )

    class ReportStorage:
        def get_experiment(self, experiment_id):
            assert experiment_id == "experiment-cli"
            return experiment

        def get_experiment_metrics(self, experiment_id):
            assert experiment_id == "experiment-cli"
            return metrics

    monkeypatch.setattr(
        "agentlab.cli._open_storage",
        lambda **_kwargs: ReportStorage(),
    )

    output_path = tmp_path / "reports" / "result.json"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "experiment-cli",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Report written to" in result.stdout
    assert output_path.exists()

    report_text = output_path.read_text(encoding="utf-8")
    assert '"id": "experiment-cli"' in report_text
    assert '"label": "CLI report"' in report_text
    assert '"total_runs": 1' in report_text
    assert '"case_id": "case-a"' in report_text


def test_cli_suspension_prints_actionable_resume_and_uses_distinct_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = EvalCase("case-a", str(tmp_path), "Fix it")

    class EmptyStorage:
        database_path = tmp_path / "agentlab.db"

        def save_run(self, *_args):
            pytest.fail("a suspended evaluation must not be persisted as a final run")

    monkeypatch.setattr("agentlab.cli.load_dataset", lambda _dataset: [case])
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: EmptyStorage())
    monkeypatch.setattr(
        "agentlab.cli.evaluate_case",
        lambda *_args, **_kwargs: EvaluationSuspended(
            "a" * 32,
            "run-1",
            "case-a",
        ),
    )

    result = CliRunner().invoke(app, ["eval", "dataset.yaml", "--agent", "mock_agent"])
    assert result.exit_code == 75
    assert "WAITING_FOR_APPROVAL" in result.stdout
    assert f"Execution ID: {'a' * 32}" in result.stdout
    assert f"agentlab resume {'a' * 32}" in result.stdout
    assert "externally" in result.stdout


def test_cli_resume_uses_stored_context_and_persists_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "agentlab.db"
    session = SimpleNamespace(
        adapter="mock_agent",
        evaluator=None,
        database_path=str(database),
        dataset="dataset.yaml",
        status=ExecutionStatus.WAITING_FOR_APPROVAL,
    )
    timestamp = "2026-08-28T00:00:00+00:00"
    trace = (
        TraceEvent(
            "run-1",
            1,
            "run_start",
            timestamp,
            {"adapter": "MockAgentAdapter"},
        ),
        TraceEvent(
            "run-1",
            2,
            "run_end",
            timestamp,
            {"passed": True, "elapsed_time": 1.0},
        ),
    )
    completed = EvalResult("case-a", True, False, True, run_id="run-1", trace=trace)
    observed = {}
    monkeypatch.setattr(
        "agentlab.cli.load_execution_session", lambda value: session
    )

    def fake_resume(value, *, adapter, evaluator):
        observed.update(value=value, adapter=adapter.info.name, evaluator=evaluator)
        SQLiteStorage(database).save_run_idempotently(completed, "dataset.yaml")
        return completed

    monkeypatch.setattr("agentlab.cli.resume_evaluation", fake_resume)
    result = CliRunner().invoke(app, ["resume", "a" * 32])
    assert result.exit_code == 0
    assert observed == {"value": "a" * 32, "adapter": "mock_agent", "evaluator": None}
    assert "PASS case-a" in result.stdout
    assert SQLiteStorage(database).get_run("run-1") is not None


def test_cli_resume_rejects_invalid_or_replayed_execution(monkeypatch) -> None:
    def rejected(_execution_id):
        from agentlab.execution_sessions import ExecutionSessionError

        raise ExecutionSessionError("terminal and cannot be replayed")

    monkeypatch.setattr("agentlab.cli.load_execution_session", rejected)
    result = CliRunner().invoke(app, ["resume", "b" * 32])
    assert result.exit_code == 1
    assert "cannot be replayed" in " ".join(result.stdout.split())

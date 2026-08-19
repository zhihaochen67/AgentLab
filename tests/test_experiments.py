import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from agentlab.adapters import AgentAdapter, AgentPreflightResult, AgentRunResult
from agentlab.adapters.repo_doctor import RepoDoctorAdapter
from agentlab.experiments import (
    ExperimentPreflightError,
    new_experiment_id,
    run_experiment,
)
from agentlab.models import EvalCase, EvalResult, Experiment
from agentlab.storage import SQLiteStorage
from agentlab.tracer import TraceEvent


class FakeExperimentAdapter(AgentAdapter):
    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        raise AssertionError("fake evaluator should replace repair")

    def preflight(self) -> AgentPreflightResult:
        return AgentPreflightResult(model="fake-model")


class SequenceEvaluator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, case: EvalCase, _adapter: AgentAdapter) -> EvalResult:
        self.calls.append(case.id)
        call_index = len(self.calls)
        passed = call_index != 2
        failure_type = "timeout" if not passed else None
        return make_result(
            case.id,
            passed=passed,
            latency=float(call_index),
            failure_type=failure_type,
        )


def make_result(
    case_id: str,
    *,
    passed: bool,
    latency: float,
    failure_type: str | None = None,
) -> EvalResult:
    run_id = str(uuid4())
    diagnostics = (
        {
            "failure_type": failure_type,
            "failure_phase": "provider_request",
            "returncode": 2,
        }
        if failure_type
        else None
    )
    agent_data = {
        "adapter": "FakeExperimentAdapter",
        "status": "ok" if passed else "error",
        "returncode": 0 if passed else 2,
    }
    if diagnostics is not None:
        agent_data["diagnostics"] = diagnostics
    trace = (
        TraceEvent(
            run_id,
            1,
            "run_start",
            "2026-08-19T01:00:00+00:00",
            {"case_id": case_id, "adapter": "FakeExperimentAdapter"},
        ),
        TraceEvent(
            run_id,
            2,
            "agent_end",
            "2026-08-19T01:00:01+00:00",
            agent_data,
        ),
        TraceEvent(
            run_id,
            3,
            "run_end",
            "2026-08-19T01:00:02+00:00",
            {"passed": passed, "elapsed_time": latency},
        ),
    )
    return EvalResult(
        case_id=case_id,
        passed=passed,
        tests_before_passed=False,
        tests_after_passed=passed,
        error=None if passed else "trial failed",
        run_id=run_id,
        trace=trace,
    )


def make_experiment(experiment_id: str = "experiment-one") -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        label="baseline",
        dataset="dataset.yaml",
        adapter="FakeExperimentAdapter",
        model="fake-model",
        trials_per_case=2,
        total_cases=2,
        total_runs=0,
        started_at="2026-08-19T01:00:00+00:00",
        finished_at=None,
        status="running",
    )


def test_experiment_ids_are_unique_uuids() -> None:
    first = new_experiment_id()
    second = new_experiment_id()

    assert first != second
    assert str(UUID(first)) == first
    assert str(UUID(second)) == second


def test_trials_filter_persistence_continuation_and_aggregates() -> None:
    cases = [
        EvalCase("case-a", "repository-a", "Fix A"),
        EvalCase("case-b", "repository-b", "Fix B"),
        EvalCase("case-c", "repository-c", "Fix C"),
    ]
    evaluator = SequenceEvaluator()
    with tempfile.TemporaryDirectory(prefix="agentlab-experiment-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        execution = run_experiment(
            cases=cases,
            dataset="dataset.yaml",
            storage=storage,
            adapter=FakeExperimentAdapter(),
            trials_per_case=2,
            label="two-case-baseline",
            case_ids=("case-c", "case-a", "case-a"),
            evaluator=evaluator,
            validator=lambda _cases: None,
        )

        experiment = storage.get_experiment(execution.experiment.experiment_id)
        runs = storage.list_runs(
            limit=20,
            experiment_id=execution.experiment.experiment_id,
        )
        metrics = storage.get_experiment_metrics(execution.experiment.experiment_id)

        assert experiment is not None
        assert experiment.label == "two-case-baseline"
        assert experiment.model == "fake-model"
        assert experiment.total_cases == 2
        assert experiment.total_runs == 4
        assert experiment.status == "completed_with_failures"
        assert evaluator.calls == ["case-a", "case-a", "case-c", "case-c"]
        assert sorted((run.case_id, run.trial_index) for run in runs) == [
            ("case-a", 1),
            ("case-a", 2),
            ("case-c", 1),
            ("case-c", 2),
        ]
        assert all(run.experiment_id == experiment.experiment_id for run in runs)
        assert metrics.total_runs == 4
        assert metrics.passed_runs == 3
        assert metrics.failed_runs == 1
        assert metrics.success_rate == 75.0
        assert metrics.average_latency == 2.5
        assert [(case.case_id, case.passed_runs, case.success_rate) for case in metrics.per_case] == [
            ("case-a", 1, 50.0),
            ("case-c", 2, 100.0),
        ]
        assert metrics.failure_types == (("timeout", 1),)


def test_repo_doctor_preflight_aborts_without_runs_or_secret(monkeypatch) -> None:
    secret = "experiment-provider-secret-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    monkeypatch.delenv("REPO_DOCTOR_BASE_URL", raising=False)
    monkeypatch.delenv("REPO_DOCTOR_MODEL", raising=False)
    called = False

    def forbidden_evaluator(_case: EvalCase, _adapter: AgentAdapter) -> EvalResult:
        nonlocal called
        called = True
        raise AssertionError("preflight must prevent evaluation")

    with tempfile.TemporaryDirectory(prefix="agentlab-experiment-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        with pytest.raises(ExperimentPreflightError) as captured:
            run_experiment(
                cases=[EvalCase("case-a", "unused", "Fix A")],
                dataset="dataset.yaml",
                storage=storage,
                adapter=RepoDoctorAdapter(),
                trials_per_case=3,
                evaluator=forbidden_evaluator,
                validator=lambda _cases: pytest.fail("validation must not run"),
            )

        experiments = storage.list_experiments()
        assert called is False
        assert captured.value.missing_variables == (
            "REPO_DOCTOR_BASE_URL",
            "REPO_DOCTOR_MODEL",
        )
        assert len(experiments) == 1
        assert experiments[0].status == "aborted"
        assert experiments[0].total_runs == 0
        assert storage.list_runs() == ()
        assert secret.encode() not in database.read_bytes()


def test_invalid_api_key_aborts_before_evaluation_repair_or_network(
    monkeypatch,
) -> None:
    secret = "your api key must be replaced"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "model-name")
    evaluation_called = False
    repair_called = False
    network_called = False

    def forbidden_evaluator(_case: EvalCase, _adapter: AgentAdapter) -> EvalResult:
        nonlocal evaluation_called
        evaluation_called = True
        raise AssertionError("invalid preflight must prevent evaluation")

    def forbidden_repair(*_args, **_kwargs):
        nonlocal repair_called
        repair_called = True
        raise AssertionError("invalid preflight must prevent Repo Doctor")

    def forbidden_network(*_args, **_kwargs):
        nonlocal network_called
        network_called = True
        raise AssertionError("invalid preflight must prevent network access")

    monkeypatch.setattr(RepoDoctorAdapter, "repair", forbidden_repair)
    monkeypatch.setattr("socket.create_connection", forbidden_network)

    with tempfile.TemporaryDirectory(prefix="agentlab-experiment-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        with pytest.raises(ExperimentPreflightError) as captured:
            run_experiment(
                cases=[EvalCase("case-a", "unused", "Fix A")],
                dataset="dataset.yaml",
                storage=storage,
                adapter=RepoDoctorAdapter(),
                trials_per_case=3,
                evaluator=forbidden_evaluator,
                validator=lambda _cases: pytest.fail("validation must not run"),
            )

        experiments = storage.list_experiments()
        assert str(captured.value) == "REPO_DOCTOR_API_KEY appears invalid"
        assert secret not in str(captured.value)
        assert captured.value.missing_variables == ()
        assert captured.value.invalid_variables == ("REPO_DOCTOR_API_KEY",)
        assert evaluation_called is False
        assert repair_called is False
        assert network_called is False
        assert len(experiments) == 1
        assert experiments[0].status == "aborted"
        assert experiments[0].total_runs == 0
        assert storage.list_runs() == ()
        assert secret.encode() not in database.read_bytes()


def test_old_database_is_readable_and_migrates_without_losing_runs() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-migration-") as directory:
        database = Path(directory) / "legacy.db"
        with closing(sqlite3.connect(database)) as connection:
            connection.executescript(
                """
                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    adapter TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    total_latency REAL NOT NULL,
                    tests_before_passed INTEGER NOT NULL,
                    tests_after_passed INTEGER NOT NULL,
                    error TEXT
                );
                CREATE TABLE trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    UNIQUE (run_id, sequence)
                );
                INSERT INTO runs VALUES (
                    'legacy-run', 'legacy-case', 'old.yaml', 'OldAdapter', 'PASS',
                    '2026-08-18T00:00:00+00:00', '2026-08-18T00:00:01+00:00',
                    1.0, 0, 1, NULL
                );
                INSERT INTO trace_events (
                    run_id, sequence, event_type, timestamp, data_json
                ) VALUES (
                    'legacy-run', 1, 'run_start',
                    '2026-08-18T00:00:00+00:00', '{}'
                );
                """
            )
            connection.commit()

        read_only = SQLiteStorage(database, read_only=True)
        legacy = read_only.get_run("legacy-run")
        assert legacy is not None
        assert legacy.experiment_id is None
        assert legacy.trial_index is None
        assert read_only.list_experiments() == ()
        assert [run.run_id for run in read_only.list_runs()] == ["legacy-run"]
        assert read_only.list_runs(experiment_id="missing") == ()

        migrated = SQLiteStorage(database)
        migrated_legacy = migrated.get_run("legacy-run")
        with closing(sqlite3.connect(database)) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(runs)")
            }
            experiment_table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'experiments'
                """
            ).fetchone()

        assert migrated_legacy is not None
        assert migrated_legacy.experiment_id is None
        assert {"experiment_id", "trial_index"}.issubset(columns)
        assert experiment_table is not None

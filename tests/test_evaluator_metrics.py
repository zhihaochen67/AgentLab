import sqlite3
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from agentlab.comparison import build_experiment_comparison
from agentlab.models import (
    CaseExperimentMetrics,
    EvalResult,
    EvaluatorExperimentMetrics,
    Experiment,
    ExperimentMetrics,
)
from agentlab.storage import SQLiteStorage, StorageError, _evaluator_experiment_metrics
from agentlab.tracer import TraceEvent

_STARTED_AT = "2026-08-18T01:00:00+00:00"
_FINISHED_AT = "2026-08-18T01:00:06+00:00"

_LEGACY_SCHEMA = """
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    dataset TEXT NOT NULL,
    adapter TEXT NOT NULL,
    model TEXT,
    trials_per_case INTEGER NOT NULL CHECK (trials_per_case > 0),
    total_cases INTEGER NOT NULL CHECK (total_cases >= 0),
    total_runs INTEGER NOT NULL DEFAULT 0 CHECK (total_runs >= 0),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'completed_with_failures', 'aborted')
    )
);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    adapter TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PASS', 'FAIL')),
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    total_latency REAL NOT NULL,
    tests_before_passed INTEGER NOT NULL,
    tests_after_passed INTEGER NOT NULL,
    error TEXT,
    experiment_id TEXT,
    trial_index INTEGER
);
CREATE TABLE trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    data_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, sequence)
);
"""


def outcome(
    status: str = "pass",
    *,
    score: float | None = 0.75,
    evaluator: str = "JudgeA",
) -> dict:
    data = {"evaluator": evaluator, "status": status, "passed": status == "pass"}
    if score is not None:
        data["score"] = score
    return data


def make_result(
    run_id: str,
    *,
    evaluator_events: list[dict] | None = None,
    passed: bool = True,
) -> EvalResult:
    events = [
        TraceEvent(
            run_id,
            1,
            "run_start",
            _STARTED_AT,
            {"case_id": "case-001", "adapter": "FakeAdapter"},
        ),
        TraceEvent(run_id, 2, "pytest_before_end", _FINISHED_AT, {"passed": False}),
        TraceEvent(run_id, 3, "agent_end", _FINISHED_AT, {"status": "ok"}),
        TraceEvent(run_id, 4, "pytest_after_end", _FINISHED_AT, {"passed": passed}),
    ]
    for sequence, data in enumerate(evaluator_events or [], start=5):
        events.append(TraceEvent(run_id, sequence, "evaluator_end", _FINISHED_AT, data))
    events.append(
        TraceEvent(
            run_id,
            99,
            "run_end",
            _FINISHED_AT,
            {"passed": passed, "elapsed_time": 1.25},
        )
    )
    return EvalResult(
        case_id="case-001",
        passed=passed,
        tests_before_passed=False,
        tests_after_passed=passed,
        error=None,
        run_id=run_id,
        trace=tuple(events),
    )


def create_experiment(storage: SQLiteStorage, experiment_id: str) -> None:
    storage.create_experiment(
        Experiment(
            experiment_id=experiment_id,
            label=experiment_id,
            dataset="dataset.yaml",
            adapter="FakeAdapter",
            model=None,
            trials_per_case=1,
            total_cases=1,
            total_runs=0,
            started_at=_STARTED_AT,
            finished_at=None,
            status="running",
        )
    )


def save_eval_run(
    storage: SQLiteStorage,
    experiment_id: str,
    run_id: str,
    *,
    evaluator_events: list[dict] | None = None,
    trial_index: int = 1,
    passed: bool = True,
) -> None:
    result = replace(
        make_result(run_id, evaluator_events=evaluator_events, passed=passed),
        experiment_id=experiment_id,
        trial_index=trial_index,
    )
    storage.save_run(result, "dataset.yaml")


def create_legacy_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_LEGACY_SCHEMA)
    connection.execute(
        """
        INSERT INTO experiments (
            experiment_id, label, dataset, adapter, model,
            trials_per_case, total_cases, total_runs, started_at, finished_at, status
        ) VALUES ('legacy-exp', 'legacy', 'dataset.yaml', 'FakeAdapter', NULL,
                  1, 1, 1, ?, NULL, 'running')
        """,
        (_STARTED_AT,),
    )
    connection.execute(
        """
        INSERT INTO runs (
            run_id, case_id, dataset, adapter, status, started_at, finished_at,
            total_latency, tests_before_passed, tests_after_passed, error,
            experiment_id, trial_index
        ) VALUES ('legacy-run', 'case-legacy', 'dataset.yaml', 'FakeAdapter',
                  'PASS', ?, ?, 1.0, 0, 1, NULL, 'legacy-exp', 1)
        """,
        (_STARTED_AT, _FINISHED_AT),
    )
    connection.execute(
        """
        INSERT INTO trace_events (run_id, sequence, event_type, timestamp, data_json)
        VALUES ('legacy-run', 1, 'run_start', ?, '{}')
        """,
        (_STARTED_AT,),
    )
    connection.commit()
    connection.close()


def make_comparison_experiment(
    experiment_id: str,
    *,
    dataset: str = "dataset.yaml",
    status: str = "completed_with_failures",
) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        label=f"{experiment_id}-label",
        dataset=dataset,
        adapter="FakeAdapter",
        model="fake-model",
        trials_per_case=2,
        total_cases=1,
        total_runs=2,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
        status=status,
    )


def make_comparison_metrics(
    experiment_id: str,
    *,
    evaluator_metrics: tuple[EvaluatorExperimentMetrics, ...] = (),
) -> ExperimentMetrics:
    per_case = (
        CaseExperimentMetrics(
            case_id="case-a",
            total_runs=2,
            passed_runs=1,
            failed_runs=1,
            success_rate=50.0,
            average_latency=1.0,
        ),
    )
    return ExperimentMetrics(
        experiment_id=experiment_id,
        total_runs=2,
        passed_runs=1,
        failed_runs=1,
        success_rate=50.0,
        average_latency=1.0,
        per_case=per_case,
        failure_types=(),
        evaluator_metrics=evaluator_metrics,
    )


def make_evaluator_metrics(
    evaluator: str,
    *,
    total_outcomes: int = 2,
    evaluated_runs: int = 2,
    passed_outcomes: int = 1,
    failed_outcomes: int = 1,
    error_outcomes: int = 0,
    pass_rate: float = 50.0,
    score_count: int = 2,
    average_score: float | None = 0.6,
    min_score: float | None = 0.5,
    max_score: float | None = 0.7,
    coverage_rate: float = 100.0,
) -> EvaluatorExperimentMetrics:
    return EvaluatorExperimentMetrics(
        evaluator=evaluator,
        total_outcomes=total_outcomes,
        evaluated_runs=evaluated_runs,
        passed_outcomes=passed_outcomes,
        failed_outcomes=failed_outcomes,
        error_outcomes=error_outcomes,
        verdict_outcomes=passed_outcomes + failed_outcomes,
        pass_rate=pass_rate,
        score_count=score_count,
        average_score=average_score,
        min_score=min_score,
        max_score=max_score,
        coverage_rate=coverage_rate,
    )


# --- Aggregation tests --------------------------------------------------------


def test_metrics_without_evaluator_outcomes_are_empty() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-plain")
        save_eval_run(storage, "exp-plain", "run-1")

        metrics = storage.get_experiment_metrics("exp-plain")

        assert metrics.evaluator_metrics == ()


def test_metrics_single_pass_outcome() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-pass")
        save_eval_run(
            storage,
            "exp-pass",
            "run-1",
            evaluator_events=[outcome(score=0.8, evaluator="judge")],
        )

        metrics = storage.get_experiment_metrics("exp-pass")

        assert len(metrics.evaluator_metrics) == 1
        judge = metrics.evaluator_metrics[0]
        assert judge.evaluator == "judge"
        assert judge.total_outcomes == 1
        assert judge.evaluated_runs == 1
        assert judge.passed_outcomes == 1
        assert judge.failed_outcomes == 0
        assert judge.error_outcomes == 0
        assert judge.verdict_outcomes == 1
        assert judge.pass_rate == 100.0
        assert judge.score_count == 1
        assert judge.average_score == 0.8
        assert judge.min_score == 0.8
        assert judge.max_score == 0.8
        assert judge.coverage_rate == 100.0


def test_metrics_single_fail_outcome() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-fail")
        save_eval_run(
            storage,
            "exp-fail",
            "run-1",
            passed=False,
            evaluator_events=[outcome("fail", score=0.1, evaluator="judge")],
        )

        judge = storage.get_experiment_metrics("exp-fail").evaluator_metrics[0]

        assert judge.failed_outcomes == 1
        assert judge.passed_outcomes == 0
        assert judge.verdict_outcomes == 1
        assert judge.pass_rate == 0.0


def test_metrics_single_error_outcome() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-error")
        save_eval_run(
            storage,
            "exp-error",
            "run-1",
            passed=False,
            evaluator_events=[outcome("error", score=None, evaluator="judge")],
        )

        judge = storage.get_experiment_metrics("exp-error").evaluator_metrics[0]

        assert judge.error_outcomes == 1
        assert judge.verdict_outcomes == 0
        assert judge.pass_rate == 0.0
        assert judge.score_count == 0
        assert judge.average_score is None
        assert judge.min_score is None
        assert judge.max_score is None


def test_pass_rate_denominator_uses_verdicts_only() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-mixed")
        save_eval_run(storage, "exp-mixed", "run-1", evaluator_events=[outcome(score=0.9)])
        save_eval_run(
            storage, "exp-mixed", "run-2", passed=False,
            evaluator_events=[outcome("fail", score=0.1)],
        )
        save_eval_run(
            storage, "exp-mixed", "run-3", passed=False,
            evaluator_events=[outcome("error", score=None)],
        )

        judge = storage.get_experiment_metrics("exp-mixed").evaluator_metrics[0]

        assert judge.total_outcomes == 3
        assert judge.passed_outcomes == 1
        assert judge.failed_outcomes == 1
        assert judge.error_outcomes == 1
        assert judge.verdict_outcomes == 2
        assert judge.pass_rate == 50.0


def test_null_scores_excluded_from_score_count() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-null-score")
        save_eval_run(
            storage, "exp-null-score", "run-1",
            evaluator_events=[outcome(score=None)],
        )
        save_eval_run(
            storage, "exp-null-score", "run-2",
            evaluator_events=[outcome(score=0.5)],
        )

        judge = storage.get_experiment_metrics("exp-null-score").evaluator_metrics[0]

        assert judge.score_count == 1
        assert judge.average_score == 0.5


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_boundary_scores_are_counted(score: float) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-boundary")
        save_eval_run(
            storage, "exp-boundary", "run-1",
            evaluator_events=[outcome(score=score)],
        )

        judge = storage.get_experiment_metrics("exp-boundary").evaluator_metrics[0]

        assert judge.score_count == 1
        assert judge.average_score == score
        assert judge.min_score == score
        assert judge.max_score == score


def test_average_min_max_scores() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-spread")
        for index, score in enumerate([0.2, 0.4, 0.8]):
            save_eval_run(
                storage, "exp-spread", f"run-{index}",
                evaluator_events=[outcome(score=score)],
            )

        judge = storage.get_experiment_metrics("exp-spread").evaluator_metrics[0]

        assert judge.score_count == 3
        assert judge.average_score == pytest.approx(0.466666, abs=1e-5)
        assert judge.min_score == 0.2
        assert judge.max_score == 0.8


def test_evaluated_runs_uses_distinct_runs() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-distinct")
        save_eval_run(
            storage, "exp-distinct", "run-1",
            evaluator_events=[outcome(score=0.5), outcome(score=0.7)],
        )

        judge = storage.get_experiment_metrics("exp-distinct").evaluator_metrics[0]

        assert judge.total_outcomes == 2
        assert judge.evaluated_runs == 1
        assert judge.coverage_rate == 100.0


def test_multiple_evaluators_aggregated_separately() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-multi")
        save_eval_run(
            storage, "exp-multi", "run-1",
            evaluator_events=[outcome(score=0.9, evaluator="JudgeA")],
        )
        save_eval_run(
            storage, "exp-multi", "run-2",
            evaluator_events=[outcome("fail", score=0.2, evaluator="JudgeB")],
        )

        by_evaluator = {
            item.evaluator: item
            for item in storage.get_experiment_metrics("exp-multi").evaluator_metrics
        }

        assert set(by_evaluator) == {"JudgeA", "JudgeB"}
        assert by_evaluator["JudgeA"].passed_outcomes == 1
        assert by_evaluator["JudgeA"].average_score == 0.9
        assert by_evaluator["JudgeB"].failed_outcomes == 1
        assert by_evaluator["JudgeB"].pass_rate == 0.0


def test_evaluator_metrics_sorted_by_name() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-sorted")
        save_eval_run(
            storage, "exp-sorted", "run-1",
            evaluator_events=[outcome(evaluator="zeta")],
        )
        save_eval_run(
            storage, "exp-sorted", "run-2",
            evaluator_events=[outcome(evaluator="alpha")],
        )

        metrics = storage.get_experiment_metrics("exp-sorted")

        assert [item.evaluator for item in metrics.evaluator_metrics] == [
            "alpha",
            "zeta",
        ]


def test_coverage_rate_uses_total_runs() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-coverage")
        save_eval_run(storage, "exp-coverage", "run-1", evaluator_events=[outcome()])
        save_eval_run(storage, "exp-coverage", "run-2", trial_index=2)

        judge = storage.get_experiment_metrics("exp-coverage").evaluator_metrics[0]

        assert judge.evaluated_runs == 1
        assert judge.coverage_rate == 50.0


def test_metrics_helper_coverage_zero_when_no_runs() -> None:
    metrics = _evaluator_experiment_metrics(
        [
            {
                "evaluator": "judge",
                "total_outcomes": 1,
                "evaluated_runs": 1,
                "passed_outcomes": 1,
                "failed_outcomes": 0,
                "error_outcomes": 0,
                "score_count": 1,
                "average_score": 0.5,
                "min_score": 0.5,
                "max_score": 0.5,
            }
        ],
        total_runs=0,
    )

    assert metrics[0].coverage_rate == 0.0
    assert metrics[0].verdict_outcomes == 1


def test_metrics_helper_pass_rate_zero_without_verdicts() -> None:
    metrics = _evaluator_experiment_metrics(
        [
            {
                "evaluator": "judge",
                "total_outcomes": 1,
                "evaluated_runs": 1,
                "passed_outcomes": 0,
                "failed_outcomes": 0,
                "error_outcomes": 1,
                "score_count": 0,
                "average_score": None,
                "min_score": None,
                "max_score": None,
            }
        ],
        total_runs=1,
    )

    assert metrics[0].pass_rate == 0.0
    assert metrics[0].score_count == 0
    assert metrics[0].average_score is None


def test_read_only_legacy_database_metrics_are_empty() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        database = Path(directory) / "agentlab.db"
        create_legacy_database(database)

        read_only = SQLiteStorage(database, read_only=True)

        metrics = read_only.get_experiment_metrics("legacy-exp")

        assert metrics.evaluator_metrics == ()
        assert metrics.total_runs == 1
        assert metrics.passed_runs == 1


def test_unknown_experiment_still_raises() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        with pytest.raises(StorageError, match="Experiment not found"):
            storage.get_experiment_metrics("missing-exp")


# --- status/passed invariant tests --------------------------------------------


@pytest.mark.parametrize(
    ("status", "passed"),
    [
        ("pass", False),
        ("fail", True),
        ("error", True),
    ],
)
def test_inconsistent_status_passed_combinations_rejected(
    status: str, passed: bool
) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "inconsistent-run",
            evaluator_events=[
                {"evaluator": "judge", "status": status, "passed": passed}
            ],
        )

        with pytest.raises(ValueError, match="inconsistent status/passed"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("inconsistent-run") is None


def test_consistent_status_passed_combinations_accepted() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        create_experiment(storage, "exp-consistent")
        save_eval_run(storage, "exp-consistent", "run-1", evaluator_events=[outcome()])
        save_eval_run(
            storage, "exp-consistent", "run-2", passed=False,
            evaluator_events=[outcome("fail")],
        )
        save_eval_run(
            storage, "exp-consistent", "run-3", passed=False,
            evaluator_events=[outcome("error")],
        )

        statuses = [
            item.status
            for run_id in ("run-1", "run-2", "run-3")
            for item in storage.get_evaluator_outcomes(run_id)
        ]

        assert statuses == ["PASS", "FAIL", "ERROR"]


# --- Comparison tests ---------------------------------------------------------


def test_comparison_same_evaluator_score_delta() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(make_evaluator_metrics("judge", average_score=0.6),),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(make_evaluator_metrics("judge", average_score=0.8),),
        ),
    )

    assert len(comparison.evaluator_metrics) == 1
    row = comparison.evaluator_metrics[0]
    assert row.evaluator == "judge"
    assert row.baseline_average_score == 0.6
    assert row.candidate_average_score == 0.8
    assert row.average_score_delta == pytest.approx(0.2)


def test_comparison_score_delta_none_when_one_side_missing() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(
                make_evaluator_metrics("judge", average_score=None, score_count=0),
            ),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(make_evaluator_metrics("judge", average_score=0.9),),
        ),
    )

    row = comparison.evaluator_metrics[0]
    assert row.baseline_average_score is None
    assert row.average_score_delta is None


def test_comparison_pass_rate_delta() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(make_evaluator_metrics("judge", pass_rate=40.0),),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(make_evaluator_metrics("judge", pass_rate=70.0),),
        ),
    )

    row = comparison.evaluator_metrics[0]
    assert row.baseline_pass_rate == 40.0
    assert row.candidate_pass_rate == 70.0
    assert row.pass_rate_delta == 30.0


def test_comparison_preserves_coverage_both_sides() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(
                make_evaluator_metrics("judge", coverage_rate=50.0, evaluated_runs=1),
            ),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(
                make_evaluator_metrics("judge", coverage_rate=100.0, evaluated_runs=2),
            ),
        ),
    )

    row = comparison.evaluator_metrics[0]
    assert row.baseline_coverage_rate == 50.0
    assert row.candidate_coverage_rate == 100.0
    assert row.baseline_evaluated_runs == 1
    assert row.candidate_evaluated_runs == 2


def test_comparison_identical_evaluator_sets_match() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(make_evaluator_metrics("judge"),),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(make_evaluator_metrics("judge"),),
        ),
    )

    assert comparison.compatibility.evaluator_sets_match is True
    assert comparison.compatibility.common_evaluators == ("judge",)
    assert comparison.compatibility.baseline_only_evaluators == ()
    assert comparison.compatibility.candidate_only_evaluators == ()
    assert comparison.compatibility.is_equivalent is True
    assert not any(
        "Evaluator mismatch" in warning
        for warning in comparison.compatibility.warnings
    )


def test_baseline_only_evaluator_warns_and_excludes_comparison() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(make_evaluator_metrics("old-judge"),),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics("candidate"),
    )

    assert comparison.compatibility.evaluator_sets_match is False
    assert comparison.compatibility.baseline_only_evaluators == ("old-judge",)
    assert comparison.compatibility.candidate_only_evaluators == ()
    assert comparison.evaluator_metrics == ()
    assert any(
        "Evaluator mismatch" in warning and "old-judge" in warning
        for warning in comparison.compatibility.warnings
    )


def test_candidate_only_evaluator_warns() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics("baseline"),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(make_evaluator_metrics("new-judge"),),
        ),
    )

    assert comparison.compatibility.evaluator_sets_match is False
    assert comparison.compatibility.candidate_only_evaluators == ("new-judge",)
    assert comparison.evaluator_metrics == ()
    assert any(
        "Evaluator mismatch" in warning and "new-judge" in warning
        for warning in comparison.compatibility.warnings
    )


def test_both_sides_without_evaluators_have_no_mismatch_warning() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics("baseline"),
        make_comparison_experiment("candidate"),
        make_comparison_metrics("candidate"),
    )

    assert comparison.evaluator_metrics == ()
    assert comparison.compatibility.evaluator_sets_match is True
    assert not any(
        "Evaluator mismatch" in warning
        for warning in comparison.compatibility.warnings
    )
    assert comparison.compatibility.is_equivalent is True


def test_evaluator_mismatch_preserves_other_compatibility() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(make_evaluator_metrics("judge-a"),),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(make_evaluator_metrics("judge-b"),),
        ),
    )

    assert comparison.compatibility.dataset_matches is True
    assert comparison.compatibility.case_sets_match is True
    assert comparison.compatibility.trials_per_case_matches is True
    assert comparison.compatibility.is_equivalent is False
    assert [case.case_id for case in comparison.per_case] == ["case-a"]
    assert comparison.evaluator_metrics == ()
    assert comparison.compatibility.baseline_only_evaluators == ("judge-a",)
    assert comparison.compatibility.candidate_only_evaluators == ("judge-b",)


def test_multiple_evaluator_comparison_sorted_by_name() -> None:
    comparison = build_experiment_comparison(
        make_comparison_experiment("baseline"),
        make_comparison_metrics(
            "baseline",
            evaluator_metrics=(
                make_evaluator_metrics("beta", average_score=0.3),
                make_evaluator_metrics("alpha", average_score=0.5),
            ),
        ),
        make_comparison_experiment("candidate"),
        make_comparison_metrics(
            "candidate",
            evaluator_metrics=(
                make_evaluator_metrics("alpha", average_score=0.7),
                make_evaluator_metrics("beta", average_score=0.4),
            ),
        ),
    )

    assert [row.evaluator for row in comparison.evaluator_metrics] == [
        "alpha",
        "beta",
    ]
    assert comparison.compatibility.common_evaluators == ("alpha", "beta")
    assert comparison.evaluator_metrics[0].average_score_delta == pytest.approx(0.2)
    assert comparison.evaluator_metrics[1].average_score_delta == pytest.approx(0.1)


def test_storage_aggregation_to_comparison_integration() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-metrics-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        for experiment_id in ("base-exp", "cand-exp"):
            create_experiment(storage, experiment_id)
            storage.finish_experiment(experiment_id, "completed", _FINISHED_AT)
        save_eval_run(
            storage, "base-exp", "base-1",
            evaluator_events=[outcome(score=0.5, evaluator="judge")],
        )
        save_eval_run(
            storage, "base-exp", "base-2", trial_index=2,
            evaluator_events=[outcome(score=0.7, evaluator="judge")],
        )
        save_eval_run(
            storage, "cand-exp", "cand-1",
            evaluator_events=[outcome(score=0.9, evaluator="judge")],
        )
        save_eval_run(
            storage, "cand-exp", "cand-2", trial_index=2,
            evaluator_events=[outcome(score=1.0, evaluator="judge")],
        )

        baseline = storage.get_experiment("base-exp")
        candidate = storage.get_experiment("cand-exp")
        assert baseline is not None and candidate is not None
        comparison = build_experiment_comparison(
            baseline,
            storage.get_experiment_metrics("base-exp"),
            candidate,
            storage.get_experiment_metrics("cand-exp"),
        )

    assert comparison.compatibility.evaluator_sets_match is True
    assert len(comparison.evaluator_metrics) == 1
    row = comparison.evaluator_metrics[0]
    assert row.evaluator == "judge"
    assert row.baseline_average_score == pytest.approx(0.6)
    assert row.candidate_average_score == pytest.approx(0.95)
    assert row.average_score_delta == pytest.approx(0.35)
    assert row.baseline_pass_rate == 100.0
    assert row.candidate_pass_rate == 100.0
    assert row.baseline_evaluated_runs == 2
    assert row.candidate_evaluated_runs == 2

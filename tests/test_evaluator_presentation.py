import json
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
from agentlab.reporting import build_experiment_report
from agentlab.storage import SQLiteStorage, StoredEvaluatorOutcome
from agentlab.tracer import TraceEvent
from dashboard.view_models import (
    comparison_evaluator_rows,
    comparison_view_data,
    evaluator_outcome_rows,
    experiment_evaluator_rows,
)

_STARTED_AT = "2026-08-18T01:00:00+00:00"
_FINISHED_AT = "2026-08-18T01:00:06+00:00"


def make_outcome(
    *,
    run_id: str = "run-1",
    sequence: int = 5,
    evaluator: str = "LLMJudgeEvaluator",
    status: str = "PASS",
    passed: bool = True,
    score: float | None = 0.82,
    feedback: str | None = "Semantics look good.",
    metadata: dict | None = None,
    error_type: str | None = None,
    elapsed_time: float | None = 0.12,
) -> StoredEvaluatorOutcome:
    return StoredEvaluatorOutcome(
        run_id=run_id,
        sequence=sequence,
        evaluator=evaluator,
        status=status,
        passed=passed,
        score=score,
        feedback=feedback,
        metadata=metadata if metadata is not None else {"rubric": "semantic"},
        error_type=error_type,
        elapsed_time=elapsed_time,
    )


def make_evaluator_metrics(
    evaluator: str,
    *,
    total_outcomes: int = 3,
    evaluated_runs: int = 3,
    passed_outcomes: int = 2,
    failed_outcomes: int = 1,
    error_outcomes: int = 0,
    pass_rate: float = 200 / 3,
    score_count: int = 3,
    average_score: float | None = 0.75,
    min_score: float | None = 0.5,
    max_score: float | None = 0.9,
    coverage_rate: float = 75.0,
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


def make_experiment(experiment_id: str) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        label=f"{experiment_id}-label",
        dataset="dataset.yaml",
        adapter="FakeAdapter",
        model="fake-model",
        trials_per_case=2,
        total_cases=1,
        total_runs=2,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
        status="completed_with_failures",
    )


def make_metrics(
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


# --- Report tests -------------------------------------------------------------


def test_report_includes_evaluator_metrics() -> None:
    report = build_experiment_report(
        make_experiment("exp-report"),
        make_metrics(
            "exp-report",
            evaluator_metrics=(make_evaluator_metrics("judge"),),
        ),
    )

    assert report["evaluator_metrics"] == [
        {
            "evaluator": "judge",
            "total_outcomes": 3,
            "evaluated_runs": 3,
            "passed_outcomes": 2,
            "failed_outcomes": 1,
            "error_outcomes": 0,
            "verdict_outcomes": 3,
            "pass_rate": 200 / 3,
            "score_count": 3,
            "average_score": 0.75,
            "min_score": 0.5,
            "max_score": 0.9,
            "coverage_rate": 75.0,
        }
    ]


def test_report_evaluator_metrics_include_error_aggregates() -> None:
    report = build_experiment_report(
        make_experiment("exp-error-report"),
        make_metrics(
            "exp-error-report",
            evaluator_metrics=(
                make_evaluator_metrics(
                    "judge",
                    passed_outcomes=1,
                    failed_outcomes=1,
                    error_outcomes=1,
                    pass_rate=50.0,
                ),
            ),
        ),
    )

    entry = report["evaluator_metrics"][0]
    assert entry["passed_outcomes"] == 1
    assert entry["failed_outcomes"] == 1
    assert entry["error_outcomes"] == 1
    assert entry["verdict_outcomes"] == 2
    assert entry["pass_rate"] == 50.0


def test_report_preserves_zero_scores() -> None:
    report = build_experiment_report(
        make_experiment("exp-zero-score"),
        make_metrics(
            "exp-zero-score",
            evaluator_metrics=(
                make_evaluator_metrics(
                    "judge",
                    average_score=0.0,
                    min_score=0.0,
                    max_score=0.0,
                ),
            ),
        ),
    )

    entry = report["evaluator_metrics"][0]
    assert entry["average_score"] == 0.0
    assert entry["average_score"] is not None
    assert entry["min_score"] == 0.0
    assert entry["max_score"] == 0.0


def test_report_serializes_missing_scores_as_null() -> None:
    report = build_experiment_report(
        make_experiment("exp-null-score"),
        make_metrics(
            "exp-null-score",
            evaluator_metrics=(
                make_evaluator_metrics(
                    "judge",
                    score_count=0,
                    average_score=None,
                    min_score=None,
                    max_score=None,
                ),
            ),
        ),
    )

    entry = report["evaluator_metrics"][0]
    assert entry["average_score"] is None
    assert entry["min_score"] is None
    assert entry["max_score"] is None
    assert "null" in json.dumps(report)


def test_report_without_evaluators_outputs_empty_list() -> None:
    report = build_experiment_report(
        make_experiment("exp-plain"),
        make_metrics("exp-plain"),
    )

    assert report["evaluator_metrics"] == []


def test_report_existing_fields_are_unchanged() -> None:
    report = build_experiment_report(
        make_experiment("exp-existing"),
        make_metrics("exp-existing"),
    )

    assert report["experiment"]["id"] == "exp-existing"
    assert report["experiment"]["adapter"] == "FakeAdapter"
    assert report["metrics"]["total_runs"] == 2
    assert report["metrics"]["success_rate"] == 50.0
    assert report["cases"] == [
        {
            "case_id": "case-a",
            "runs": 2,
            "passed": 1,
            "failed": 1,
            "success_rate": 50.0,
            "average_latency": 1.0,
        }
    ]
    assert report["failure_types"] == []


# --- Run Detail view model tests ----------------------------------------------


def test_evaluator_outcome_rows_pass_fail_and_error() -> None:
    rows = evaluator_outcome_rows(
        (
            make_outcome(status="PASS", passed=True, score=0.9),
            make_outcome(
                status="FAIL", passed=False, score=0.1, feedback="wrong semantics"
            ),
            make_outcome(
                status="ERROR",
                passed=False,
                score=None,
                feedback=None,
                error_type="JudgeResponseError",
            ),
        )
    )

    assert rows[0]["status"] == "✅ PASS"
    assert rows[0]["passed"] is True
    assert rows[0]["score"] == 0.9
    assert rows[1]["status"] == "❌ FAIL"
    assert rows[1]["passed"] is False
    assert rows[1]["score"] == 0.1
    assert rows[2]["status"] == "❌ ERROR"
    assert rows[2]["error_type"] == "JudgeResponseError"
    assert rows[2]["feedback"] == "—"


def test_evaluator_outcome_rows_none_score_is_not_zero() -> None:
    rows = evaluator_outcome_rows((make_outcome(score=None),))

    assert rows[0]["score"] == "—"
    assert rows[0]["elapsed_time"] == 0.12


def test_evaluator_outcome_rows_keep_metadata_as_object() -> None:
    rows = evaluator_outcome_rows(
        (make_outcome(metadata={"rubric": "semantic", "tags": ["a", "b"]}),)
    )

    assert rows[0]["metadata"] == {"rubric": "semantic", "tags": ["a", "b"]}
    assert isinstance(rows[0]["metadata"], dict)


def test_evaluator_outcome_rows_redact_feedback_and_sensitive_keys(monkeypatch) -> None:
    secret = "presentation-secret-111222"
    monkeypatch.setenv("JUDGE_API_KEY", secret)

    rows = evaluator_outcome_rows(
        (
            make_outcome(
                feedback=(
                    f"Authorization: Bearer {secret}; api_key={secret}; "
                    f"password={secret}; sk-{secret}"
                ),
                metadata={
                    "api_key": secret,
                    "token": secret,
                    "password": secret,
                    "note": f"token={secret}",
                },
                evaluator=f"judge-{secret}",
            ),
        )
    )

    row = rows[0]
    assert secret not in repr(rows)
    assert "[REDACTED]" in row["feedback"]
    assert row["metadata"]["api_key"] == "[REDACTED]"
    assert row["metadata"]["token"] == "[REDACTED]"
    assert row["metadata"]["password"] == "[REDACTED]"
    assert "[REDACTED]" in row["metadata"]["note"]
    assert secret not in row["evaluator"]


def test_evaluator_outcome_rows_empty_input() -> None:
    assert evaluator_outcome_rows(()) == []


def test_evaluator_outcome_rows_reject_none_feedback_cleanly() -> None:
    rows = evaluator_outcome_rows(
        (make_outcome(feedback=None, error_type=None, elapsed_time=None),)
    )

    assert rows[0]["feedback"] == "—"
    assert rows[0]["error_type"] == "—"
    assert rows[0]["elapsed_time"] == "—"


# --- Experiment Detail view model tests ---------------------------------------


def test_experiment_evaluator_rows_complete() -> None:
    rows = experiment_evaluator_rows((make_evaluator_metrics("judge"),))

    assert rows[0] == {
        "evaluator": "judge",
        "evaluated_runs": 3,
        "total_outcomes": 3,
        "coverage_rate": 75.0,
        "passed": 2,
        "failed": 1,
        "errors": 0,
        "pass_rate": 66.7,
        "score_count": 3,
        "average_score": 0.75,
        "min_score": 0.5,
        "max_score": 0.9,
    }


def test_experiment_evaluator_rows_empty() -> None:
    assert experiment_evaluator_rows(()) == []


def test_experiment_evaluator_rows_rounding_and_raw_scale() -> None:
    rows = experiment_evaluator_rows(
        (
            make_evaluator_metrics(
                "judge",
                pass_rate=66.666,
                coverage_rate=49.999,
                average_score=0.826,
            ),
        )
    )

    assert rows[0]["pass_rate"] == 66.7
    assert rows[0]["coverage_rate"] == 50.0
    assert rows[0]["average_score"] == 0.826


def test_experiment_evaluator_rows_none_scores_display_dash() -> None:
    rows = experiment_evaluator_rows(
        (
            make_evaluator_metrics(
                "judge",
                score_count=0,
                average_score=None,
                min_score=None,
                max_score=None,
            ),
        )
    )

    assert rows[0]["average_score"] == "—"
    assert rows[0]["min_score"] == "—"
    assert rows[0]["max_score"] == "—"
    assert rows[0]["score_count"] == 0


# --- Comparison view model tests ----------------------------------------------


def build_comparison(
    baseline_evaluators: tuple[EvaluatorExperimentMetrics, ...],
    candidate_evaluators: tuple[EvaluatorExperimentMetrics, ...],
):
    return build_experiment_comparison(
        make_experiment("baseline"),
        make_metrics("baseline", evaluator_metrics=baseline_evaluators),
        make_experiment("candidate"),
        make_metrics("candidate", evaluator_metrics=candidate_evaluators),
    )


def test_comparison_evaluator_rows_common_evaluator() -> None:
    comparison = build_comparison(
        (make_evaluator_metrics("judge", average_score=0.6, pass_rate=40.0),),
        (make_evaluator_metrics("judge", average_score=0.8, pass_rate=70.0),),
    )

    rows = comparison_evaluator_rows(comparison.evaluator_metrics)

    assert rows[0]["evaluator"] == "judge"
    assert rows[0]["baseline_average_score"] == 0.6
    assert rows[0]["candidate_average_score"] == 0.8
    assert rows[0]["average_score_delta"] == pytest.approx(0.2)
    assert rows[0]["pass_rate_delta"] == 30.0
    assert rows[0]["baseline_pass_rate"] == 40.0
    assert rows[0]["candidate_pass_rate"] == 70.0


def test_comparison_evaluator_rows_none_score_delta_shows_dash() -> None:
    comparison = build_comparison(
        (
            make_evaluator_metrics(
                "judge", score_count=0, average_score=None
            ),
        ),
        (make_evaluator_metrics("judge", average_score=0.9),),
    )

    rows = comparison_evaluator_rows(comparison.evaluator_metrics)

    assert rows[0]["baseline_average_score"] == "—"
    assert rows[0]["average_score_delta"] == "—"


def test_comparison_view_data_exposes_coverage_and_errors() -> None:
    comparison = build_comparison(
        (
            make_evaluator_metrics(
                "judge",
                evaluated_runs=1,
                coverage_rate=50.0,
                error_outcomes=2,
            ),
        ),
        (
            make_evaluator_metrics(
                "judge",
                evaluated_runs=2,
                coverage_rate=100.0,
                error_outcomes=0,
            ),
        ),
    )

    row = comparison_view_data(comparison)["evaluator_metrics"][0]

    assert row["baseline_coverage_rate"] == 50.0
    assert row["candidate_coverage_rate"] == 100.0
    assert row["baseline_error_outcomes"] == 2
    assert row["candidate_error_outcomes"] == 0
    assert row["baseline_evaluated_runs"] == 1
    assert row["candidate_evaluated_runs"] == 2


def test_comparison_view_data_exposes_baseline_only_evaluators() -> None:
    comparison = build_comparison(
        (make_evaluator_metrics("old-judge"),),
        (),
    )

    data = comparison_view_data(comparison)

    assert data["evaluator_sets_match"] is False
    assert data["common_evaluators"] == ()
    assert data["baseline_only_evaluators"] == ("old-judge",)
    assert data["candidate_only_evaluators"] == ()
    assert data["evaluator_metrics"] == []


def test_comparison_view_data_exposes_candidate_only_evaluators() -> None:
    comparison = build_comparison(
        (),
        (make_evaluator_metrics("new-judge"),),
    )

    data = comparison_view_data(comparison)

    assert data["evaluator_sets_match"] is False
    assert data["candidate_only_evaluators"] == ("new-judge",)
    assert data["baseline_only_evaluators"] == ()
    assert data["evaluator_metrics"] == []


def test_comparison_view_data_both_empty_evaluators() -> None:
    comparison = build_comparison((), ())

    data = comparison_view_data(comparison)

    assert data["evaluator_sets_match"] is True
    assert data["common_evaluators"] == ()
    assert data["baseline_only_evaluators"] == ()
    assert data["candidate_only_evaluators"] == ()
    assert data["evaluator_metrics"] == []


def test_evaluator_mismatch_keeps_run_level_view_data() -> None:
    comparison = build_comparison(
        (make_evaluator_metrics("judge-a"),),
        (make_evaluator_metrics("judge-b"),),
    )

    data = comparison_view_data(comparison)

    assert data["is_equivalent"] is False
    assert data["success_rate_delta"] == 0.0
    assert data["latency_delta"] == 0.0
    assert [row["case_id"] for row in data["per_case"]] == ["case-a"]
    assert data["baseline"]["total_runs"] == 2
    assert data["candidate"]["total_runs"] == 2
    assert data["evaluator_metrics"] == []
    assert any(
        "Evaluator mismatch" in warning for warning in data["warnings"]
    )


def test_comparison_evaluator_rows_preserve_order() -> None:
    comparison = build_comparison(
        (
            make_evaluator_metrics("beta", average_score=0.3),
            make_evaluator_metrics("alpha", average_score=0.5),
        ),
        (
            make_evaluator_metrics("alpha", average_score=0.7),
            make_evaluator_metrics("beta", average_score=0.4),
        ),
    )

    rows = comparison_evaluator_rows(comparison.evaluator_metrics)

    assert [row["evaluator"] for row in rows] == ["alpha", "beta"]


# --- Storage-backed smoke test ------------------------------------------------


def test_report_json_serializes_from_persisted_evaluator_data() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-presentation-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.create_experiment(
            Experiment(
                experiment_id="exp-smoke",
                label="smoke",
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
        result = EvalResult(
            case_id="case-001",
            passed=True,
            tests_before_passed=False,
            tests_after_passed=True,
            error=None,
            run_id="smoke-run",
            trace=(
                TraceEvent(
                    "smoke-run",
                    1,
                    "run_start",
                    _STARTED_AT,
                    {"case_id": "case-001", "adapter": "FakeAdapter"},
                ),
                TraceEvent(
                    "smoke-run",
                    2,
                    "evaluator_end",
                    _FINISHED_AT,
                    {
                        "evaluator": "LLMJudgeEvaluator",
                        "status": "pass",
                        "passed": True,
                        "score": 0.82,
                        "feedback": "ok",
                        "metadata": {"judge": "mock"},
                        "elapsed_time": 0.1,
                    },
                ),
                TraceEvent(
                    "smoke-run",
                    3,
                    "run_end",
                    _FINISHED_AT,
                    {"passed": True, "elapsed_time": 1.0},
                ),
            ),
        )
        storage.save_run(
            replace(result, experiment_id="exp-smoke", trial_index=1),
            "dataset.yaml",
        )
        experiment = storage.get_experiment("exp-smoke")
        assert experiment is not None
        metrics = storage.get_experiment_metrics("exp-smoke")
        report = build_experiment_report(experiment, metrics)

    assert report["evaluator_metrics"][0]["evaluator"] == "LLMJudgeEvaluator"
    assert report["evaluator_metrics"][0]["average_score"] == 0.82
    assert report["evaluator_metrics"][0]["pass_rate"] == 100.0
    serialized = json.dumps(report)
    assert "LLMJudgeEvaluator" in serialized

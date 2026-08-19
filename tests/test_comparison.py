from __future__ import annotations

from dataclasses import replace

import pytest
from typer.testing import CliRunner

from agentlab.cli import app
from agentlab.comparison import build_experiment_comparison
from agentlab.models import (
    CaseExperimentMetrics,
    Experiment,
    ExperimentMetrics,
)
from dashboard.view_models import comparison_view_data


def make_experiment(
    experiment_id: str,
    *,
    dataset: str = "dataset.yaml",
    trials: int = 3,
    total_cases: int = 2,
    status: str = "completed_with_failures",
    agent_version: str | None = None,
    prompt_variant: str | None = None,
    model: str | None = "fake-model",
) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        label=f"{experiment_id}-label",
        dataset=dataset,
        adapter="FakeAdapter",
        model=model,
        trials_per_case=trials,
        total_cases=total_cases,
        total_runs=0,
        started_at="2026-08-19T01:00:00+00:00",
        finished_at="2026-08-19T01:01:00+00:00",
        status=status,
        agent_version=agent_version,
        prompt_variant=prompt_variant,
    )


def make_metrics(
    experiment_id: str,
    cases: dict[str, tuple[int, int, float]],
    *,
    failure_types: tuple[tuple[str, int], ...] = (),
) -> ExperimentMetrics:
    per_case = tuple(
        CaseExperimentMetrics(
            case_id=case_id,
            total_runs=runs,
            passed_runs=passes,
            failed_runs=runs - passes,
            success_rate=(passes / runs * 100.0) if runs else 0.0,
            average_latency=average_latency,
        )
        for case_id, (runs, passes, average_latency) in sorted(cases.items())
    )
    total_runs = sum(case.total_runs for case in per_case)
    passed_runs = sum(case.passed_runs for case in per_case)
    total_latency = sum(
        case.average_latency * case.total_runs for case in per_case
    )
    return ExperimentMetrics(
        experiment_id=experiment_id,
        total_runs=total_runs,
        passed_runs=passed_runs,
        failed_runs=total_runs - passed_runs,
        success_rate=(passed_runs / total_runs * 100.0) if total_runs else 0.0,
        average_latency=(total_latency / total_runs) if total_runs else 0.0,
        per_case=per_case,
        failure_types=failure_types,
    )


def make_mixed_comparison():
    baseline = make_experiment(
        "baseline",
        trials=2,
        agent_version="repo-doctor-0.2.0",
        prompt_variant="baseline-v1",
        model="deepseek-v4-flash",
    )
    candidate = make_experiment(
        "candidate",
        trials=2,
        agent_version="repo-doctor-0.2.0",
        prompt_variant="candidate-v2",
        model="deepseek-v4-flash",
    )
    baseline_metrics = make_metrics(
        "baseline",
        {"case-a": (2, 1, 12.0), "case-b": (2, 2, 8.0)},
        failure_types=(
            ("patch_generation_error", 1),
            ("repair_verification_failed", 2),
        ),
    )
    candidate_metrics = make_metrics(
        "candidate",
        {"case-a": (2, 2, 8.0), "case-b": (2, 1, 4.0)},
        failure_types=(
            ("repair_verification_failed", 1),
            ("timeout", 2),
        ),
    )
    return build_experiment_comparison(
        baseline,
        baseline_metrics,
        candidate,
        candidate_metrics,
    )


def test_identical_experiments_are_equivalent_and_unchanged() -> None:
    baseline = make_experiment("baseline")
    candidate = make_experiment("candidate")
    baseline_metrics = make_metrics(
        "baseline",
        {"case-a": (3, 3, 5.0), "case-b": (3, 2, 7.0)},
        failure_types=(("timeout", 1),),
    )
    candidate_metrics = replace(baseline_metrics, experiment_id="candidate")

    comparison = build_experiment_comparison(
        baseline,
        baseline_metrics,
        candidate,
        candidate_metrics,
    )

    assert comparison.compatibility.is_equivalent is True
    assert comparison.success_rate_delta == 0.0
    assert comparison.latency_delta == 0.0
    assert {case.change for case in comparison.per_case} == {"unchanged"}
    assert comparison.failure_types[0].delta == 0


def test_success_rate_improvement_regression_and_per_case_delta() -> None:
    comparison = make_mixed_comparison()
    by_case = {case.case_id: case for case in comparison.per_case}

    assert by_case["case-a"].delta == 50.0
    assert by_case["case-a"].change == "improved"
    assert by_case["case-b"].delta == -50.0
    assert by_case["case-b"].change == "regressed"
    assert comparison.success_rate_delta == 0.0


def test_overall_success_rate_improvement() -> None:
    baseline = make_experiment("baseline", total_cases=1)
    candidate = make_experiment("candidate", total_cases=1)
    comparison = build_experiment_comparison(
        baseline,
        make_metrics("baseline", {"case-a": (3, 2, 10.0)}),
        candidate,
        make_metrics("candidate", {"case-a": (3, 3, 10.0)}),
    )

    assert comparison.success_rate_delta == pytest.approx(100 / 3)


def test_failure_taxonomy_and_latency_deltas_use_candidate_minus_baseline() -> None:
    comparison = make_mixed_comparison()
    failures = {
        failure.failure_type: failure for failure in comparison.failure_types
    }

    assert comparison.latency_delta == -4.0
    assert failures["patch_generation_error"].baseline_count == 1
    assert failures["patch_generation_error"].candidate_count == 0
    assert failures["patch_generation_error"].delta == -1
    assert failures["repair_verification_failed"].delta == -1
    assert failures["timeout"].delta == 2


def test_mismatched_dataset_is_non_equivalent() -> None:
    baseline = make_experiment("baseline")
    candidate = make_experiment("candidate", dataset="other.yaml")
    comparison = build_experiment_comparison(
        baseline,
        make_metrics("baseline", {"case-a": (3, 3, 1.0), "case-b": (3, 3, 1.0)}),
        candidate,
        make_metrics("candidate", {"case-a": (3, 3, 1.0), "case-b": (3, 3, 1.0)}),
    )

    assert comparison.compatibility.dataset_matches is False
    assert comparison.compatibility.is_equivalent is False
    assert "Dataset mismatch" in comparison.compatibility.warnings[0]


def test_mismatched_cases_only_compare_the_intersection() -> None:
    baseline = make_experiment("baseline")
    candidate = make_experiment("candidate")
    comparison = build_experiment_comparison(
        baseline,
        make_metrics("baseline", {"case-a": (3, 2, 1.0), "case-b": (3, 3, 1.0)}),
        candidate,
        make_metrics("candidate", {"case-a": (3, 3, 1.0), "case-c": (3, 3, 1.0)}),
    )

    assert comparison.compatibility.case_sets_match is False
    assert comparison.compatibility.common_case_ids == ("case-a",)
    assert comparison.compatibility.baseline_only_case_ids == ("case-b",)
    assert comparison.compatibility.candidate_only_case_ids == ("case-c",)
    assert [case.case_id for case in comparison.per_case] == ["case-a"]


def test_mismatched_trials_are_non_equivalent() -> None:
    baseline = make_experiment("baseline", trials=3, total_cases=1)
    candidate = make_experiment("candidate", trials=5, total_cases=1)
    comparison = build_experiment_comparison(
        baseline,
        make_metrics("baseline", {"case-a": (3, 3, 1.0)}),
        candidate,
        make_metrics("candidate", {"case-a": (5, 5, 1.0)}),
    )

    assert comparison.compatibility.trials_per_case_matches is False
    assert comparison.compatibility.is_equivalent is False
    assert any(
        "Trials-per-case mismatch" in warning
        for warning in comparison.compatibility.warnings
    )


def test_zero_run_experiments_do_not_divide_by_zero() -> None:
    baseline = make_experiment("baseline", total_cases=0, status="completed")
    candidate = make_experiment("candidate", total_cases=0, status="completed")
    comparison = build_experiment_comparison(
        baseline,
        make_metrics("baseline", {}),
        candidate,
        make_metrics("candidate", {}),
    )

    assert comparison.baseline.total_runs == 0
    assert comparison.success_rate_delta == 0.0
    assert comparison.latency_delta == 0.0
    assert comparison.per_case == ()
    assert comparison.compatibility.is_equivalent is True


def test_aborted_experiment_and_unknown_case_set_are_non_equivalent() -> None:
    baseline = make_experiment("baseline", total_cases=1, status="aborted")
    candidate = make_experiment("candidate", total_cases=1, status="completed")
    comparison = build_experiment_comparison(
        baseline,
        make_metrics("baseline", {}),
        candidate,
        make_metrics("candidate", {"case-a": (3, 3, 1.0)}),
    )

    assert comparison.compatibility.is_equivalent is False
    assert comparison.compatibility.baseline_case_set_complete is False
    assert any("could not be fully verified" in warning for warning in comparison.compatibility.warnings)
    assert any("status is 'aborted'" in warning for warning in comparison.compatibility.warnings)


def test_dashboard_comparison_view_model_has_changes_and_taxonomy() -> None:
    data = comparison_view_data(make_mixed_comparison())

    assert data["is_equivalent"] is True
    assert data["baseline"]["total_runs"] == 4
    assert data["baseline"]["agent_version"] == "repo-doctor-0.2.0"
    assert data["baseline"]["prompt_variant"] == "baseline-v1"
    assert data["baseline"]["model"] == "deepseek-v4-flash"
    assert data["candidate"]["prompt_variant"] == "candidate-v2"
    assert data["candidate"]["average_latency"] == 6.0
    assert data["latency_delta"] == -4.0
    assert [row["case_id"] for row in data["improvements"]] == ["case-a"]
    assert [row["case_id"] for row in data["regressions"]] == ["case-b"]
    assert data["failure_types"][0] == {
        "failure_type": "patch_generation_error",
        "baseline": 1,
        "candidate": 0,
        "delta": -1,
    }


def test_cli_comparison_formatting(monkeypatch) -> None:
    comparison = make_mixed_comparison()
    opened_with = {}

    def fake_open_storage(**kwargs):
        opened_with.update(kwargs)
        return object()

    monkeypatch.setattr("agentlab.cli._open_storage", fake_open_storage)
    monkeypatch.setattr(
        "agentlab.cli.compare_experiments",
        lambda *_args: comparison,
    )

    result = CliRunner().invoke(app, ["compare", "baseline", "candidate"])

    assert result.exit_code == 0
    assert "Experiment Comparison" in result.stdout
    assert "Baseline:" in result.stdout
    assert "Candidate:" in result.stdout
    assert "repo-doctor-0.2.0 / baseline-v1 / deepseek-v4-flash" in result.stdout
    assert "repo-doctor-0.2.0 / candidate-v2 / deepseek-v4-flash" in result.stdout
    assert "Success Rate" in result.stdout
    assert "Avg Latency" in result.stdout
    assert "case-a" in result.stdout
    assert "IMPROVED" in result.stdout
    assert "case-b" in result.stdout
    assert "REGRESSED" in result.stdout
    assert "patch_generation_error" in result.stdout
    assert opened_with == {"read_only": True}

"""Pure business logic for comparing persisted experiments."""

from __future__ import annotations

import math

from agentlab.models import (
    CaseExperimentComparison,
    CaseExperimentMetrics,
    ComparisonChange,
    ComparisonCompatibility,
    EvaluatorExperimentMetrics,
    EvaluatorMetricsComparison,
    Experiment,
    ExperimentComparison,
    ExperimentComparisonSummary,
    ExperimentMetrics,
    FailureTypeComparison,
)
from agentlab.storage import RunStorage

COMPARABLE_STATUSES = {"completed", "completed_with_failures"}


class ExperimentComparisonError(ValueError):
    """A requested experiment comparison could not be constructed."""


def compare_experiments(
    storage: RunStorage,
    baseline_experiment_id: str,
    candidate_experiment_id: str,
) -> ExperimentComparison:
    """Load two persisted experiments and build their comparison."""
    baseline = storage.get_experiment(baseline_experiment_id)
    if baseline is None:
        raise ExperimentComparisonError(
            f"Baseline experiment not found: {baseline_experiment_id}"
        )
    candidate = storage.get_experiment(candidate_experiment_id)
    if candidate is None:
        raise ExperimentComparisonError(
            f"Candidate experiment not found: {candidate_experiment_id}"
        )
    return build_experiment_comparison(
        baseline,
        storage.get_experiment_metrics(baseline_experiment_id),
        candidate,
        storage.get_experiment_metrics(candidate_experiment_id),
    )


def build_experiment_comparison(
    baseline: Experiment,
    baseline_metrics: ExperimentMetrics,
    candidate: Experiment,
    candidate_metrics: ExperimentMetrics,
) -> ExperimentComparison:
    """Build a deterministic comparison without performing storage I/O."""
    _validate_metrics(baseline, baseline_metrics)
    _validate_metrics(candidate, candidate_metrics)

    baseline_cases = {case.case_id: case for case in baseline_metrics.per_case}
    candidate_cases = {case.case_id: case for case in candidate_metrics.per_case}
    baseline_evaluators = {
        metrics.evaluator for metrics in baseline_metrics.evaluator_metrics
    }
    candidate_evaluators = {
        metrics.evaluator for metrics in candidate_metrics.evaluator_metrics
    }
    compatibility = _compatibility(
        baseline,
        baseline_cases,
        candidate,
        candidate_cases,
        baseline_evaluators,
        candidate_evaluators,
    )
    per_case = tuple(
        _compare_case(baseline_cases[case_id], candidate_cases[case_id])
        for case_id in compatibility.common_case_ids
    )

    baseline_failures = dict(baseline_metrics.failure_types)
    candidate_failures = dict(candidate_metrics.failure_types)
    failure_types = tuple(
        FailureTypeComparison(
            failure_type=failure_type,
            baseline_count=baseline_failures.get(failure_type, 0),
            candidate_count=candidate_failures.get(failure_type, 0),
            delta=(
                candidate_failures.get(failure_type, 0)
                - baseline_failures.get(failure_type, 0)
            ),
        )
        for failure_type in sorted(baseline_failures.keys() | candidate_failures.keys())
    )

    baseline_by_evaluator = {
        metrics.evaluator: metrics for metrics in baseline_metrics.evaluator_metrics
    }
    candidate_by_evaluator = {
        metrics.evaluator: metrics for metrics in candidate_metrics.evaluator_metrics
    }
    evaluator_metrics = tuple(
        _compare_evaluator(
            baseline_by_evaluator[evaluator],
            candidate_by_evaluator[evaluator],
        )
        for evaluator in compatibility.common_evaluators
    )

    return ExperimentComparison(
        baseline=_summary(baseline, baseline_metrics),
        candidate=_summary(candidate, candidate_metrics),
        success_rate_delta=(
            candidate_metrics.success_rate - baseline_metrics.success_rate
        ),
        latency_delta=(
            candidate_metrics.average_latency - baseline_metrics.average_latency
        ),
        compatibility=compatibility,
        per_case=per_case,
        failure_types=failure_types,
        evaluator_metrics=evaluator_metrics,
    )


def _compare_evaluator(
    baseline: EvaluatorExperimentMetrics,
    candidate: EvaluatorExperimentMetrics,
) -> EvaluatorMetricsComparison:
    """Compare evaluator metrics for one evaluator name shared by both sides."""
    average_score_delta: float | None
    if baseline.average_score is None or candidate.average_score is None:
        average_score_delta = None
    else:
        average_score_delta = candidate.average_score - baseline.average_score
    return EvaluatorMetricsComparison(
        evaluator=baseline.evaluator,
        baseline_total_outcomes=baseline.total_outcomes,
        candidate_total_outcomes=candidate.total_outcomes,
        baseline_evaluated_runs=baseline.evaluated_runs,
        candidate_evaluated_runs=candidate.evaluated_runs,
        baseline_coverage_rate=baseline.coverage_rate,
        candidate_coverage_rate=candidate.coverage_rate,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        pass_rate_delta=candidate.pass_rate - baseline.pass_rate,
        baseline_average_score=baseline.average_score,
        candidate_average_score=candidate.average_score,
        average_score_delta=average_score_delta,
        baseline_score_count=baseline.score_count,
        candidate_score_count=candidate.score_count,
        baseline_error_outcomes=baseline.error_outcomes,
        candidate_error_outcomes=candidate.error_outcomes,
    )


def _validate_metrics(experiment: Experiment, metrics: ExperimentMetrics) -> None:
    if experiment.experiment_id != metrics.experiment_id:
        raise ExperimentComparisonError(
            "Experiment metadata and metrics have different experiment ids: "
            f"{experiment.experiment_id} != {metrics.experiment_id}"
        )


def _summary(
    experiment: Experiment,
    metrics: ExperimentMetrics,
) -> ExperimentComparisonSummary:
    return ExperimentComparisonSummary(
        experiment=experiment,
        total_runs=metrics.total_runs,
        passed_runs=metrics.passed_runs,
        failed_runs=metrics.failed_runs,
        success_rate=metrics.success_rate,
        average_latency=metrics.average_latency,
    )


def _compatibility(
    baseline: Experiment,
    baseline_cases: dict[str, CaseExperimentMetrics],
    candidate: Experiment,
    candidate_cases: dict[str, CaseExperimentMetrics],
    baseline_evaluators: set[str],
    candidate_evaluators: set[str],
) -> ComparisonCompatibility:
    baseline_case_ids = set(baseline_cases)
    candidate_case_ids = set(candidate_cases)
    common = tuple(sorted(baseline_case_ids & candidate_case_ids))
    baseline_only = tuple(sorted(baseline_case_ids - candidate_case_ids))
    candidate_only = tuple(sorted(candidate_case_ids - baseline_case_ids))
    common_evaluators = tuple(sorted(baseline_evaluators & candidate_evaluators))
    baseline_only_evaluators = tuple(sorted(baseline_evaluators - candidate_evaluators))
    candidate_only_evaluators = tuple(sorted(candidate_evaluators - baseline_evaluators))
    evaluator_sets_match = baseline_evaluators == candidate_evaluators
    baseline_complete = len(baseline_case_ids) == baseline.total_cases
    candidate_complete = len(candidate_case_ids) == candidate.total_cases
    dataset_matches = baseline.dataset == candidate.dataset
    observed_case_sets_match = baseline_case_ids == candidate_case_ids
    case_sets_match = (
        observed_case_sets_match and baseline_complete and candidate_complete
    )
    trials_match = baseline.trials_per_case == candidate.trials_per_case

    warnings: list[str] = []
    if not dataset_matches:
        warnings.append(
            "Dataset mismatch: "
            f"baseline={baseline.dataset!r}, candidate={candidate.dataset!r}."
        )
    if not observed_case_sets_match:
        details = []
        if baseline_only:
            details.append(f"baseline-only={', '.join(baseline_only)}")
        if candidate_only:
            details.append(f"candidate-only={', '.join(candidate_only)}")
        warnings.append(f"Case set mismatch: {'; '.join(details)}.")
    if not baseline_complete or not candidate_complete:
        warnings.append(
            "Case set could not be fully verified from persisted runs: "
            f"baseline observed {len(baseline_case_ids)}/{baseline.total_cases}, "
            f"candidate observed {len(candidate_case_ids)}/{candidate.total_cases}."
        )
    if not trials_match:
        warnings.append(
            "Trials-per-case mismatch: "
            f"baseline={baseline.trials_per_case}, "
            f"candidate={candidate.trials_per_case}."
        )
    if not evaluator_sets_match:
        details = []
        if baseline_only_evaluators:
            details.append(f"baseline-only={', '.join(baseline_only_evaluators)}")
        if candidate_only_evaluators:
            details.append(f"candidate-only={', '.join(candidate_only_evaluators)}")
        warnings.append(f"Evaluator mismatch: {'; '.join(details)}.")
    if baseline.status not in COMPARABLE_STATUSES:
        warnings.append(
            f"Baseline experiment status is {baseline.status!r}; results may be incomplete."
        )
    if candidate.status not in COMPARABLE_STATUSES:
        warnings.append(
            f"Candidate experiment status is {candidate.status!r}; results may be incomplete."
        )

    return ComparisonCompatibility(
        is_equivalent=not warnings,
        dataset_matches=dataset_matches,
        case_sets_match=case_sets_match,
        trials_per_case_matches=trials_match,
        baseline_case_set_complete=baseline_complete,
        candidate_case_set_complete=candidate_complete,
        common_case_ids=common,
        baseline_only_case_ids=baseline_only,
        candidate_only_case_ids=candidate_only,
        warnings=tuple(warnings),
        evaluator_sets_match=evaluator_sets_match,
        common_evaluators=common_evaluators,
        baseline_only_evaluators=baseline_only_evaluators,
        candidate_only_evaluators=candidate_only_evaluators,
    )


def _compare_case(
    baseline: CaseExperimentMetrics,
    candidate: CaseExperimentMetrics,
) -> CaseExperimentComparison:
    delta = candidate.success_rate - baseline.success_rate
    change: ComparisonChange
    if math.isclose(delta, 0.0, abs_tol=1e-9):
        change = "unchanged"
    elif delta > 0:
        change = "improved"
    else:
        change = "regressed"
    return CaseExperimentComparison(
        case_id=baseline.case_id,
        baseline_runs=baseline.total_runs,
        baseline_passes=baseline.passed_runs,
        baseline_success_rate=baseline.success_rate,
        candidate_runs=candidate.total_runs,
        candidate_passes=candidate.passed_runs,
        candidate_success_rate=candidate.success_rate,
        delta=delta,
        change=change,
    )

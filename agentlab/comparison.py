"""Pure business logic for comparing persisted experiments."""

from __future__ import annotations

import math

from agentlab.models import (
    CaseExperimentComparison,
    CaseExperimentMetrics,
    ComparisonChange,
    ComparisonCompatibility,
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
    compatibility = _compatibility(
        baseline,
        baseline_cases,
        candidate,
        candidate_cases,
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
) -> ComparisonCompatibility:
    baseline_case_ids = set(baseline_cases)
    candidate_case_ids = set(candidate_cases)
    common = tuple(sorted(baseline_case_ids & candidate_case_ids))
    baseline_only = tuple(sorted(baseline_case_ids - candidate_case_ids))
    candidate_only = tuple(sorted(candidate_case_ids - baseline_case_ids))
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

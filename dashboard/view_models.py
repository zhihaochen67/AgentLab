"""Pure presentation helpers shared by Dashboard views and tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentlab.diagnostics import diagnostics_from_trace_data
from agentlab.models import (
    CaseExperimentComparison,
    CaseExperimentMetrics,
    EvaluatorExperimentMetrics,
    EvaluatorMetricsComparison,
    Experiment,
    ExperimentComparison,
    ExperimentMetrics,
    FailureTypeComparison,
)
from agentlab.replay import event_elapsed, event_status
from agentlab.storage import StoredEvaluatorOutcome, StoredRun
from agentlab.tracer import TraceEvent, sanitize_data, summarize_text


def status_label(status: str) -> str:
    """Return a visually distinct, text-safe status label."""
    if status == "PASS":
        return "✅ PASS"
    if status == "FAIL":
        return "❌ FAIL"
    if status == "OK":
        return "✅ OK"
    if status == "ERROR":
        return "❌ ERROR"
    return status


def run_table_rows(runs: Iterable[StoredRun]) -> list[dict[str, Any]]:
    """Convert stored runs into rows suitable for a read-only data table."""
    return [
        {
            "run_id": summarize_text(run.run_id),
            "case_id": summarize_text(run.case_id),
            "adapter": summarize_text(run.adapter),
            "status": status_label(run.status),
            "latency": round(run.total_latency, 3),
            "started_at": summarize_text(run.started_at),
        }
        for run in runs
    ]


def run_detail_data(run: StoredRun) -> dict[str, Any]:
    """Return re-sanitized metadata for the run detail view."""
    return {
        "run_id": summarize_text(run.run_id),
        "case_id": summarize_text(run.case_id),
        "dataset": summarize_text(run.dataset),
        "adapter": summarize_text(run.adapter),
        "status": run.status,
        "total_latency": run.total_latency,
        "tests_before_passed": run.tests_before_passed,
        "tests_after_passed": run.tests_after_passed,
        "started_at": summarize_text(run.started_at),
        "finished_at": summarize_text(run.finished_at),
        "error": summarize_text(run.error) if run.error is not None else None,
        "experiment_id": (
            summarize_text(run.experiment_id) if run.experiment_id is not None else None
        ),
        "trial_index": run.trial_index,
    }


def evaluator_outcome_rows(
    outcomes: Iterable[StoredEvaluatorOutcome],
) -> list[dict[str, Any]]:
    """Convert persisted evaluator outcomes into sanitized display rows.

    Scores keep the raw 0.0~1.0 scale; missing values render as "—" so a
    NULL score is never shown as 0. Feedback and metadata are defensively
    re-sanitized at this output boundary.
    """
    return [
        {
            "sequence": outcome.sequence,
            "evaluator": summarize_text(outcome.evaluator),
            "status": status_label(outcome.status),
            "passed": outcome.passed,
            "score": outcome.score if outcome.score is not None else "—",
            "feedback": (
                summarize_text(outcome.feedback)
                if outcome.feedback is not None
                else "—"
            ),
            "error_type": (
                summarize_text(outcome.error_type)
                if outcome.error_type is not None
                else "—"
            ),
            "elapsed_time": (
                outcome.elapsed_time if outcome.elapsed_time is not None else "—"
            ),
            "metadata": sanitize_data(outcome.metadata),
        }
        for outcome in outcomes
    ]


def experiment_table_rows(experiments: Iterable[Experiment]) -> list[dict[str, Any]]:
    """Convert experiment metadata into read-only table rows."""
    return [
        {
            "experiment_id": summarize_text(experiment.experiment_id),
            "label": summarize_text(experiment.label),
            "agent_version": _metadata_for_display(experiment.agent_version),
            "prompt_variant": _metadata_for_display(experiment.prompt_variant),
            "model": _metadata_for_display(experiment.model),
            "status": experiment.status.upper(),
            "runs": experiment.total_runs,
            "trials_per_case": experiment.trials_per_case,
            "started_at": summarize_text(experiment.started_at),
        }
        for experiment in experiments
    ]


def experiment_detail_data(
    experiment: Experiment,
    metrics: ExperimentMetrics,
) -> dict[str, Any]:
    """Create a sanitized experiment detail view model."""
    return {
        "experiment_id": summarize_text(experiment.experiment_id),
        "label": summarize_text(experiment.label),
        "dataset": summarize_text(experiment.dataset),
        "adapter": summarize_text(experiment.adapter),
        "agent_version": _metadata_for_display(experiment.agent_version),
        "prompt_variant": _metadata_for_display(experiment.prompt_variant),
        "model": _metadata_for_display(experiment.model),
        "notes": _metadata_for_display(experiment.notes),
        "trials_per_case": experiment.trials_per_case,
        "total_cases": experiment.total_cases,
        "total_runs": metrics.total_runs,
        "passed_runs": metrics.passed_runs,
        "failed_runs": metrics.failed_runs,
        "success_rate": metrics.success_rate,
        "average_latency": metrics.average_latency,
        "started_at": summarize_text(experiment.started_at),
        "finished_at": (
            summarize_text(experiment.finished_at)
            if experiment.finished_at is not None
            else "In progress"
        ),
        "status": experiment.status.upper(),
    }


def experiment_case_rows(
    cases: Iterable[CaseExperimentMetrics],
) -> list[dict[str, Any]]:
    """Convert per-case experiment aggregates into display rows."""
    return [
        {
            "case_id": summarize_text(case.case_id),
            "passed": case.passed_runs,
            "failed": case.failed_runs,
            "runs": case.total_runs,
            "success_rate": round(case.success_rate, 1),
            "average_latency": round(case.average_latency, 3),
        }
        for case in cases
    ]


def experiment_failure_rows(
    failure_types: Iterable[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Convert failure taxonomy counts into display rows."""
    return [
        {"failure_type": summarize_text(failure_type), "count": count}
        for failure_type, count in failure_types
    ]


def experiment_evaluator_rows(
    evaluator_metrics: Iterable[EvaluatorExperimentMetrics],
) -> list[dict[str, Any]]:
    """Convert evaluator experiment metrics into display rows.

    Pass rate and coverage are percentages; scores keep the raw 0.0~1.0
    scale and missing scores render as "—".
    """
    return [
        {
            "evaluator": summarize_text(item.evaluator),
            "evaluated_runs": item.evaluated_runs,
            "total_outcomes": item.total_outcomes,
            "coverage_rate": round(item.coverage_rate, 1),
            "passed": item.passed_outcomes,
            "failed": item.failed_outcomes,
            "errors": item.error_outcomes,
            "pass_rate": round(item.pass_rate, 1),
            "score_count": item.score_count,
            "average_score": (
                item.average_score if item.average_score is not None else "—"
            ),
            "min_score": item.min_score if item.min_score is not None else "—",
            "max_score": item.max_score if item.max_score is not None else "—",
        }
        for item in evaluator_metrics
    ]


def comparison_view_data(comparison: ExperimentComparison) -> dict[str, Any]:
    """Create a sanitized, UI-ready experiment comparison view model."""
    baseline = comparison.baseline
    candidate = comparison.candidate
    return {
        "is_equivalent": comparison.compatibility.is_equivalent,
        "comparison_label": (
            "EQUIVALENT"
            if comparison.compatibility.is_equivalent
            else "NON-EQUIVALENT COMPARISON"
        ),
        "warnings": [
            summarize_text(warning)
            for warning in comparison.compatibility.warnings
        ],
        "baseline": {
            "experiment_id": summarize_text(baseline.experiment.experiment_id),
            "label": summarize_text(baseline.experiment.label),
            "agent_version": _metadata_for_display(
                baseline.experiment.agent_version
            ),
            "prompt_variant": _metadata_for_display(
                baseline.experiment.prompt_variant
            ),
            "model": _metadata_for_display(baseline.experiment.model),
            "total_runs": baseline.total_runs,
            "passed_runs": baseline.passed_runs,
            "failed_runs": baseline.failed_runs,
            "success_rate": round(baseline.success_rate, 1),
            "average_latency": round(baseline.average_latency, 3),
        },
        "candidate": {
            "experiment_id": summarize_text(candidate.experiment.experiment_id),
            "label": summarize_text(candidate.experiment.label),
            "agent_version": _metadata_for_display(
                candidate.experiment.agent_version
            ),
            "prompt_variant": _metadata_for_display(
                candidate.experiment.prompt_variant
            ),
            "model": _metadata_for_display(candidate.experiment.model),
            "total_runs": candidate.total_runs,
            "passed_runs": candidate.passed_runs,
            "failed_runs": candidate.failed_runs,
            "success_rate": round(candidate.success_rate, 1),
            "average_latency": round(candidate.average_latency, 3),
        },
        "success_rate_delta": round(comparison.success_rate_delta, 1),
        "latency_delta": round(comparison.latency_delta, 3),
        "evaluator_sets_match": comparison.compatibility.evaluator_sets_match,
        "common_evaluators": comparison.compatibility.common_evaluators,
        "baseline_only_evaluators": comparison.compatibility.baseline_only_evaluators,
        "candidate_only_evaluators": comparison.compatibility.candidate_only_evaluators,
        "evaluator_metrics": comparison_evaluator_rows(comparison.evaluator_metrics),
        "per_case": comparison_case_rows(comparison.per_case),
        "improvements": comparison_case_rows(
            case for case in comparison.per_case if case.change == "improved"
        ),
        "regressions": comparison_case_rows(
            case for case in comparison.per_case if case.change == "regressed"
        ),
        "failure_types": comparison_failure_rows(comparison.failure_types),
    }


def comparison_overall_rows(comparison: ExperimentComparison) -> list[dict[str, Any]]:
    """Convert overall comparison KPIs into a compact display table."""
    baseline = comparison.baseline
    candidate = comparison.candidate
    return [
        {
            "metric": "Total Runs",
            "baseline": baseline.total_runs,
            "candidate": candidate.total_runs,
            "delta": candidate.total_runs - baseline.total_runs,
        },
        {
            "metric": "Passed",
            "baseline": baseline.passed_runs,
            "candidate": candidate.passed_runs,
            "delta": candidate.passed_runs - baseline.passed_runs,
        },
        {
            "metric": "Failed",
            "baseline": baseline.failed_runs,
            "candidate": candidate.failed_runs,
            "delta": candidate.failed_runs - baseline.failed_runs,
        },
        {
            "metric": "Success Rate (%)",
            "baseline": round(baseline.success_rate, 1),
            "candidate": round(candidate.success_rate, 1),
            "delta": round(comparison.success_rate_delta, 1),
        },
        {
            "metric": "Average Latency (s)",
            "baseline": round(baseline.average_latency, 3),
            "candidate": round(candidate.average_latency, 3),
            "delta": round(comparison.latency_delta, 3),
        },
    ]


def comparison_case_rows(
    cases: Iterable[CaseExperimentComparison],
) -> list[dict[str, Any]]:
    """Convert common per-case comparisons into display rows."""
    return [
        {
            "case_id": summarize_text(case.case_id),
            "baseline_runs": case.baseline_runs,
            "baseline_passes": case.baseline_passes,
            "baseline_success_rate": round(case.baseline_success_rate, 1),
            "candidate_runs": case.candidate_runs,
            "candidate_passes": case.candidate_passes,
            "candidate_success_rate": round(case.candidate_success_rate, 1),
            "delta": round(case.delta, 1),
            "change": case.change.upper(),
        }
        for case in cases
    ]


def comparison_failure_rows(
    failure_types: Iterable[FailureTypeComparison],
) -> list[dict[str, Any]]:
    """Convert failure taxonomy comparisons into display rows."""
    return [
        {
            "failure_type": summarize_text(failure.failure_type),
            "baseline": failure.baseline_count,
            "candidate": failure.candidate_count,
            "delta": failure.delta,
        }
        for failure in failure_types
    ]


def comparison_evaluator_rows(
    evaluator_metrics: Iterable[EvaluatorMetricsComparison],
) -> list[dict[str, Any]]:
    """Convert evaluator metrics comparisons into display rows.

    Pass rate / coverage values and their delta are percentages
    (delta in percentage points); scores keep the raw 0.0~1.0 scale and
    missing scores render as "—".
    """
    return [
        {
            "evaluator": summarize_text(row.evaluator),
            "baseline_evaluated_runs": row.baseline_evaluated_runs,
            "candidate_evaluated_runs": row.candidate_evaluated_runs,
            "baseline_coverage_rate": round(row.baseline_coverage_rate, 1),
            "candidate_coverage_rate": round(row.candidate_coverage_rate, 1),
            "baseline_pass_rate": round(row.baseline_pass_rate, 1),
            "candidate_pass_rate": round(row.candidate_pass_rate, 1),
            "pass_rate_delta": round(row.pass_rate_delta, 1),
            "baseline_average_score": (
                row.baseline_average_score
                if row.baseline_average_score is not None
                else "—"
            ),
            "candidate_average_score": (
                row.candidate_average_score
                if row.candidate_average_score is not None
                else "—"
            ),
            "average_score_delta": (
                row.average_score_delta if row.average_score_delta is not None else "—"
            ),
            "baseline_score_count": row.baseline_score_count,
            "candidate_score_count": row.candidate_score_count,
            "baseline_error_outcomes": row.baseline_error_outcomes,
            "candidate_error_outcomes": row.candidate_error_outcomes,
        }
        for row in evaluator_metrics
    ]


def trace_table_rows(events: Iterable[TraceEvent]) -> list[dict[str, Any]]:
    """Convert trace events into strictly sequence-ordered summary rows."""
    rows = []
    for event in sorted(events, key=lambda item: item.sequence):
        status = event_status(event)
        rows.append(
            {
                "sequence": event.sequence,
                "event_type": summarize_text(event.event_type),
                "timestamp": summarize_text(event.timestamp),
                "status": status_label(status) if status else "",
                "elapsed": event_elapsed(event),
            }
        )
    return rows


def event_data_for_display(event: TraceEvent) -> dict[str, Any]:
    """Re-sanitize event data at the final UI boundary."""
    return sanitize_data(event.data)


def format_failure_diagnostics(data: Any) -> dict[str, Any] | None:
    """Format optional diagnostics for a stable, secret-safe Dashboard view."""
    if not isinstance(data, Mapping):
        return None
    repair_fields = (
        "analysis_summary",
        "selected_finding",
        "behavioral_contract",
        "patch_diff",
        "final_status",
    )
    if not data.get("failure_type") and not any(data.get(key) is not None for key in repair_fields):
        return None
    safe = sanitize_data(data)
    verification_failed = safe.get("verification_failed")
    if verification_failed is True:
        verification_result = "FAILED"
    elif verification_failed is False:
        verification_result = "PASSED"
    else:
        verification_result = "NOT AVAILABLE"

    rollback_attempted = safe.get("rollback_attempted")
    rollback_succeeded = safe.get("rollback_succeeded")
    if rollback_succeeded is True:
        rollback_result = "SUCCESS"
    elif rollback_succeeded is False:
        rollback_result = "FAILED"
    elif rollback_attempted is False:
        rollback_result = "NOT ATTEMPTED"
    else:
        rollback_result = "NOT AVAILABLE"

    return {
        **safe,
        "failure_type": summarize_text(safe.get("failure_type") or "none").upper(),
        "failure_phase": summarize_text(safe.get("failure_phase") or "not available"),
        "patch_applied_display": _boolean_label(safe.get("patch_applied")),
        "verification_result": verification_result,
        "rollback_result": rollback_result,
    }


def failure_diagnostics_for_display(
    events: Iterable[TraceEvent],
) -> dict[str, Any] | None:
    """Find the latest structured failure diagnostics in a trace."""
    ordered = sorted(events, key=lambda item: item.sequence, reverse=True)
    for event in ordered:
        diagnostics = diagnostics_from_trace_data(event.data)
        formatted = format_failure_diagnostics(diagnostics)
        if formatted is not None:
            return formatted
    return None


def _boolean_label(value: Any) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "NOT AVAILABLE"


def _metadata_for_display(value: str | None) -> str:
    return summarize_text(value) if value is not None else "Not recorded"

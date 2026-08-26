"""Experiment report generation."""

from __future__ import annotations

from agentlab.models import Experiment, ExperimentMetrics


def build_experiment_report(
    experiment: Experiment,
    metrics: ExperimentMetrics,
) -> dict:
    """Build a JSON-serializable experiment report.

    Evaluator metrics are aggregate-only: no feedback, metadata, prompts,
    or raw judge responses are included. Scores stay on the raw 0.0~1.0
    scale and missing scores serialize as JSON null.
    """
    evaluator_metrics = [
        {
            "evaluator": item.evaluator,
            "total_outcomes": item.total_outcomes,
            "evaluated_runs": item.evaluated_runs,
            "passed_outcomes": item.passed_outcomes,
            "failed_outcomes": item.failed_outcomes,
            "error_outcomes": item.error_outcomes,
            "verdict_outcomes": item.verdict_outcomes,
            "pass_rate": item.pass_rate,
            "score_count": item.score_count,
            "average_score": item.average_score,
            "min_score": item.min_score,
            "max_score": item.max_score,
            "coverage_rate": item.coverage_rate,
        }
        for item in metrics.evaluator_metrics
    ]

    return {
        "experiment": {
            "id": experiment.experiment_id,
            "label": experiment.label,
            "dataset": experiment.dataset,
            "adapter": experiment.adapter,
            "model": experiment.model,
            "agent_version": experiment.agent_version,
            "prompt_variant": experiment.prompt_variant,
            "notes": experiment.notes,
            "status": experiment.status,
            "trials_per_case": experiment.trials_per_case,
            "total_cases": experiment.total_cases,
            "started_at": experiment.started_at,
            "finished_at": experiment.finished_at,
        },
        "metrics": {
            "total_runs": metrics.total_runs,
            "passed_runs": metrics.passed_runs,
            "failed_runs": metrics.failed_runs,
            "success_rate": metrics.success_rate,
            "average_latency": metrics.average_latency,
        },
        "evaluator_metrics": evaluator_metrics,
        "cases": [
            {
                "case_id": case.case_id,
                "runs": case.total_runs,
                "passed": case.passed_runs,
                "failed": case.failed_runs,
                "success_rate": case.success_rate,
                "average_latency": case.average_latency,
            }
            for case in metrics.per_case
        ],
        "failure_types": [
            {
                "type": failure_type,
                "count": count,
            }
            for failure_type, count in metrics.failure_types
        ],
    }
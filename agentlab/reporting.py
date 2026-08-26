"""Experiment report generation."""

from __future__ import annotations

from agentlab.models import Experiment, ExperimentMetrics


def build_experiment_report(
    experiment: Experiment,
    metrics: ExperimentMetrics,
) -> dict:
    """Build a JSON-serializable experiment report."""

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
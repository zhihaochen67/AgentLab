from types import SimpleNamespace

from agentlab.reporting import build_experiment_report


def test_build_experiment_report() -> None:
    experiment = SimpleNamespace(
        experiment_id="experiment-report",
        label="Report baseline",
        dataset="dataset.yaml",
        adapter="MockAgentAdapter",
        model=None,
        agent_version="mock-v1",
        prompt_variant="baseline",
        notes="report test",
        status="completed",
        trials_per_case=2,
        total_cases=1,
        started_at="2026-08-26T00:00:00+00:00",
        finished_at="2026-08-26T00:00:02+00:00",
    )

    case_metrics = SimpleNamespace(
        case_id="case-a",
        total_runs=2,
        passed_runs=1,
        failed_runs=1,
        success_rate=0.5,
        average_latency=0.4,
    )

    metrics = SimpleNamespace(
        total_runs=2,
        passed_runs=1,
        failed_runs=1,
        success_rate=0.5,
        average_latency=0.4,
        per_case=(case_metrics,),
        failure_types=(("test_failure", 1),),
    )

    report = build_experiment_report(experiment, metrics)

    assert report["experiment"]["id"] == "experiment-report"
    assert report["experiment"]["agent_version"] == "mock-v1"
    assert report["metrics"]["total_runs"] == 2
    assert report["metrics"]["success_rate"] == 0.5

    assert report["cases"] == [
        {
            "case_id": "case-a",
            "runs": 2,
            "passed": 1,
            "failed": 1,
            "success_rate": 0.5,
            "average_latency": 0.4,
        }
    ]

    assert report["failure_types"] == [
        {
            "type": "test_failure",
            "count": 1,
        }
    ]
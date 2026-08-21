from agentlab.models import CaseExperimentMetrics, Experiment, ExperimentMetrics
from agentlab.storage import StoredRun
from agentlab.tracer import TraceEvent
from dashboard.view_models import (
    event_data_for_display,
    event_elapsed,
    event_status,
    experiment_case_rows,
    experiment_detail_data,
    experiment_failure_rows,
    experiment_table_rows,
    failure_diagnostics_for_display,
    format_failure_diagnostics,
    run_detail_data,
    run_table_rows,
    status_label,
    trace_table_rows,
)


def make_stored_run(status: str) -> StoredRun:
    return StoredRun(
        run_id=f"run-{status.lower()}",
        case_id="case-001",
        dataset="dataset.yaml",
        adapter="FakeAdapter",
        status=status,
        started_at="2026-08-18T01:00:00+00:00",
        finished_at="2026-08-18T01:00:01+00:00",
        total_latency=1.23456,
        tests_before_passed=False,
        tests_after_passed=status == "PASS",
        error=None,
    )


def test_run_table_rows_make_pass_and_fail_visually_distinct() -> None:
    rows = run_table_rows((make_stored_run("PASS"), make_stored_run("FAIL")))

    assert rows[0]["status"] == "✅ PASS"
    assert rows[1]["status"] == "❌ FAIL"
    assert rows[0]["latency"] == 1.235
    assert status_label("ERROR") == "❌ ERROR"


def test_trace_rows_are_sorted_and_derive_status_and_elapsed() -> None:
    events = (
        TraceEvent(
            "run-one",
            3,
            "run_end",
            "2026-08-18T01:00:01+00:00",
            {"passed": True, "elapsed_time": 1.5},
        ),
        TraceEvent(
            "run-one",
            1,
            "pytest_before_end",
            "2026-08-18T01:00:00+00:00",
            {"passed": False, "elapsed_time": 0.25},
        ),
        TraceEvent(
            "run-one",
            2,
            "agent_end",
            "2026-08-18T01:00:00+00:00",
            {"status": "ok", "elapsed_time": 1.0},
        ),
    )

    rows = trace_table_rows(events)

    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["status"] for row in rows] == ["❌ FAIL", "✅ OK", "✅ PASS"]
    assert event_status(events[0]) == "PASS"
    assert event_elapsed(events[0]) == 1.5


def test_event_data_is_redacted_again_at_dashboard_boundary(monkeypatch) -> None:
    secret = "dashboard-secret-value-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    event = TraceEvent(
        "run-one",
        1,
        "agent_end",
        "2026-08-18T01:00:00+00:00",
        {
            "stdout": f"Authorization: Bearer {secret}",
            "api_key": secret,
            "nested": {"message": secret},
        },
    )

    data = event_data_for_display(event)

    assert secret not in repr(data)
    assert "[REDACTED]" in repr(data)


def test_run_metadata_is_redacted_again_at_dashboard_boundary(monkeypatch) -> None:
    secret = "dashboard-run-secret-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    run = StoredRun(
        run_id="run-one",
        case_id=f"case-{secret}",
        dataset=f"dataset-{secret}",
        adapter=f"adapter-{secret}",
        status="FAIL",
        started_at="2026-08-18T01:00:00+00:00",
        finished_at="2026-08-18T01:00:01+00:00",
        total_latency=1.0,
        tests_before_passed=False,
        tests_after_passed=False,
        error=f"Authorization: Bearer {secret}",
    )

    detail = run_detail_data(run)
    rows = run_table_rows((run,))

    assert secret not in repr(detail)
    assert secret not in repr(rows)
    assert "[REDACTED]" in repr(detail)


def test_dashboard_formats_failure_diagnostics_and_redacts_patch(monkeypatch) -> None:
    secret = "dashboard-diagnostic-secret-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    raw = {
        "failure_type": "repair_verification_failed",
        "failure_phase": "verification",
        "returncode": 1,
        "patch_applied": True,
        "verification_failed": True,
        "rollback_attempted": True,
        "rollback_succeeded": True,
        "verification_output": "Python tests failed",
        "patch_diff": f"+password={secret}",
    }

    formatted = format_failure_diagnostics(raw)

    assert formatted is not None
    assert formatted["failure_type"] == "REPAIR_VERIFICATION_FAILED"
    assert formatted["failure_phase"] == "verification"
    assert formatted["patch_applied_display"] == "YES"
    assert formatted["verification_result"] == "FAILED"
    assert formatted["rollback_result"] == "SUCCESS"
    assert secret not in repr(formatted)
    assert "[REDACTED]" in formatted["patch_diff"]


def test_dashboard_and_replay_format_contract_selected_finding_and_patch() -> None:
    event = TraceEvent(
        "report-run",
        2,
        "agent_end",
        "2026-08-18T01:00:01+00:00",
        {
            "status": "ok",
            "diagnostics": {
                "failure_type": None,
                "failure_phase": None,
                "returncode": 0,
                "analysis_summary": "One finding and one complete contract.",
                "selected_finding": {"id": "finding-1", "title": "Boundary"},
                "behavioral_contract": {
                    "must_fix": ["Accept the boundary."],
                    "must_preserve": ["Keep smaller values valid."],
                    "evidence": ["All smaller-value tests pass."],
                    "rationale": "Preserve the passing behavior.",
                },
                "patch_diff": "-before\n+after\n",
                "patch_applied": True,
                "verification_failed": False,
                "verification_output": "5 passed",
                "rollback_attempted": False,
                "rollback_succeeded": None,
                "final_status": "kept",
            },
        },
    )

    formatted = failure_diagnostics_for_display((event,))

    assert formatted is not None
    assert formatted["failure_type"] == "NONE"
    assert formatted["selected_finding"]["id"] == "finding-1"
    assert formatted["behavioral_contract"]["must_preserve"] == [
        "Keep smaller values valid."
    ]
    assert formatted["patch_diff"] == "-before\n+after\n"
    assert formatted["verification_output"] == "5 passed"
    assert formatted["rollback_result"] == "NOT ATTEMPTED"


def test_legacy_trace_without_diagnostics_is_compatible() -> None:
    legacy = TraceEvent(
        "legacy-run",
        1,
        "agent_end",
        "2026-08-18T01:00:00+00:00",
        {"status": "error", "returncode": 1, "stdout": "old output"},
    )

    assert failure_diagnostics_for_display((legacy,)) is None
    assert format_failure_diagnostics(None) is None


def test_legacy_repo_doctor_trace_is_diagnosed_without_database_migration() -> None:
    legacy = TraceEvent(
        "legacy-repo-doctor-run",
        5,
        "agent_end",
        "2026-08-18T01:00:00+00:00",
        {
            "adapter": "RepoDoctorAdapter",
            "status": "error",
            "returncode": 1,
            "stdout": (
                "Patch applied\nVerification failed: Python tests.\n"
                "Rolling back\nRepository restored successfully\n"
            ),
            "stderr": "",
        },
    )

    formatted = failure_diagnostics_for_display((legacy,))

    assert formatted is not None
    assert formatted["failure_type"] == "REPAIR_VERIFICATION_FAILED"
    assert formatted["rollback_result"] == "SUCCESS"


def test_dashboard_experiment_view_models_are_aggregated_and_redacted(monkeypatch) -> None:
    secret = "dashboard-experiment-secret-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    experiment = Experiment(
        experiment_id="experiment-one",
        label=f"baseline-{secret}",
        dataset="dataset.yaml",
        adapter="RepoDoctorAdapter",
        model="deepseek-chat",
        trials_per_case=3,
        total_cases=1,
        total_runs=3,
        started_at="2026-08-19T01:00:00+00:00",
        finished_at="2026-08-19T01:01:00+00:00",
        status="completed_with_failures",
        agent_version="repo-doctor-0.2.0",
        prompt_variant="baseline-v1",
        notes="official baseline",
    )
    metrics = ExperimentMetrics(
        experiment_id="experiment-one",
        total_runs=3,
        passed_runs=2,
        failed_runs=1,
        success_rate=200 / 3,
        average_latency=10.25,
        per_case=(
            CaseExperimentMetrics(
                "settings_parser_001",
                3,
                2,
                1,
                200 / 3,
                10.25,
            ),
        ),
        failure_types=(("repair_verification_failed", 1),),
    )

    table = experiment_table_rows((experiment,))
    detail = experiment_detail_data(experiment, metrics)
    cases = experiment_case_rows(metrics.per_case)
    failures = experiment_failure_rows(metrics.failure_types)

    assert secret not in repr(table)
    assert secret not in repr(detail)
    assert "[REDACTED]" in table[0]["label"]
    assert table[0]["agent_version"] == "repo-doctor-0.2.0"
    assert table[0]["prompt_variant"] == "baseline-v1"
    assert table[0]["model"] == "deepseek-chat"
    assert detail["agent_version"] == "repo-doctor-0.2.0"
    assert detail["prompt_variant"] == "baseline-v1"
    assert detail["model"] == "deepseek-chat"
    assert detail["notes"] == "official baseline"
    assert detail["total_runs"] == 3
    assert detail["success_rate"] == 200 / 3
    assert cases == [
        {
            "case_id": "settings_parser_001",
            "passed": 2,
            "failed": 1,
            "runs": 3,
            "success_rate": 66.7,
            "average_latency": 10.25,
        }
    ]
    assert failures == [{"failure_type": "repair_verification_failed", "count": 1}]


def test_dashboard_legacy_experiment_metadata_is_rendered_as_not_recorded() -> None:
    experiment = Experiment(
        experiment_id="legacy-experiment",
        label="legacy",
        dataset="dataset.yaml",
        adapter="RepoDoctorAdapter",
        model=None,
        trials_per_case=3,
        total_cases=0,
        total_runs=0,
        started_at="2026-08-19T01:00:00+00:00",
        finished_at="2026-08-19T01:01:00+00:00",
        status="completed",
    )
    metrics = ExperimentMetrics(
        experiment_id="legacy-experiment",
        total_runs=0,
        passed_runs=0,
        failed_runs=0,
        success_rate=0.0,
        average_latency=0.0,
        per_case=(),
        failure_types=(),
    )

    table = experiment_table_rows((experiment,))
    detail = experiment_detail_data(experiment, metrics)

    assert table[0]["agent_version"] == "Not recorded"
    assert table[0]["prompt_variant"] == "Not recorded"
    assert table[0]["model"] == "Not recorded"
    assert detail["agent_version"] == "Not recorded"
    assert detail["prompt_variant"] == "Not recorded"
    assert detail["notes"] == "Not recorded"

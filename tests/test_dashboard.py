from agentlab.storage import StoredRun
from agentlab.tracer import TraceEvent
from dashboard.view_models import (
    event_data_for_display,
    event_elapsed,
    event_status,
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

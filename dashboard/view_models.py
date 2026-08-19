"""Pure presentation helpers shared by Dashboard views and tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentlab.diagnostics import diagnostics_from_trace_data
from agentlab.replay import event_elapsed, event_status
from agentlab.storage import StoredRun
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
    }


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
    if not isinstance(data, Mapping) or not data.get("failure_type"):
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
        "failure_type": summarize_text(safe["failure_type"]).upper(),
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

"""Pure presentation helpers shared by Dashboard views and tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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

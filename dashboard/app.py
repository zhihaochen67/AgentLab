"""Streamlit entry point for the read-only AgentLab trace viewer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from agentlab.storage import (
    RunStats,
    RunStorage,
    SQLiteStorage,
    StorageError,
    StoredRun,
    default_database_path,
)
from agentlab.tracer import TraceEvent
from dashboard.view_models import (
    event_data_for_display,
    event_status,
    run_detail_data,
    run_table_rows,
    status_label,
    trace_table_rows,
)

RECENT_RUN_LIMIT = 10
DASHBOARD_RUN_LIMIT = 500
LONG_TEXT_FIELDS = {"stdout", "stderr"}


def render_status(status: str) -> None:
    css_class = "status-pass" if status == "PASS" else "status-fail"
    st.markdown(
        f'<span class="status-pill {css_class}">{status}</span>',
        unsafe_allow_html=True,
    )


def render_overview(storage: RunStorage, stats: RunStats) -> None:
    st.header("Overview")
    columns = st.columns(5)
    columns[0].metric("Total Runs", stats.total_runs)
    columns[1].metric("Successful Runs", stats.successful_runs)
    columns[2].metric("Failed Runs", stats.failed_runs)
    columns[3].metric("Success Rate", f"{stats.success_rate:.1f}%")
    columns[4].metric("Average Latency", f"{stats.average_latency:.2f}s")

    st.subheader("Recent Runs")
    recent = storage.list_runs(limit=RECENT_RUN_LIMIT)
    st.dataframe(run_table_rows(recent), width="stretch", hide_index=True)


def run_option(run: StoredRun) -> str:
    data = run_detail_data(run)
    return f"{status_label(run.status)} · {data['case_id']} · {data['run_id']}"


def render_runs(storage: RunStorage) -> None:
    st.header("Runs")
    filter_columns = st.columns(2)
    status_filter = filter_columns[0].selectbox("Status", ("ALL", "PASS", "FAIL"))
    case_ids = storage.list_case_ids()
    case_filter = filter_columns[1].selectbox("Case", ("ALL", *case_ids))

    runs = storage.list_runs(
        limit=DASHBOARD_RUN_LIMIT,
        status=None if status_filter == "ALL" else status_filter,
        case_id=None if case_filter == "ALL" else case_filter,
    )
    if not runs:
        st.info("No runs match the selected filters.")
        return

    st.dataframe(run_table_rows(runs), width="stretch", hide_index=True)
    selected = st.selectbox(
        "Select a run to inspect",
        runs,
        format_func=run_option,
        key="runs_selected_run",
    )
    render_run_detail(storage, selected.run_id)


def render_run_detail(storage: RunStorage, run_id: str) -> None:
    run = storage.get_run(run_id)
    if run is None:
        st.warning(f"Run not found: {run_id}")
        return
    events = storage.get_trace_events(run_id)
    data = run_detail_data(run)

    st.header("Run Detail / Trace Viewer")
    st.code(data["run_id"], language=None)
    render_status(run.status)

    first, second = st.columns(2)
    with first:
        st.write("**Case:**", data["case_id"])
        st.write("**Dataset:**", data["dataset"])
        st.write("**Adapter:**", data["adapter"])
        st.write("**Total latency:**", f"{data['total_latency']:.3f}s")
    with second:
        before_status = "PASS" if data["tests_before_passed"] else "FAIL"
        after_status = "PASS" if data["tests_after_passed"] else "FAIL"
        st.write("**Pytest before:**", status_label(before_status))
        st.write("**Pytest after:**", status_label(after_status))
        st.write("**Started:**", data["started_at"])
        st.write("**Finished:**", data["finished_at"])

    if data["error"]:
        st.error(data["error"])

    st.subheader("Trace")
    st.dataframe(trace_table_rows(events), width="stretch", hide_index=True)
    render_event_details(events)


def render_event_details(events: Sequence[TraceEvent]) -> None:
    st.subheader("Event Data")
    for event in sorted(events, key=lambda item: item.sequence):
        status = event_status(event)
        label = f"{event.sequence} · {event.event_type}"
        if status:
            label += f" · {status_label(status)}"
        with st.expander(label, expanded=False):
            data = event_data_for_display(event)
            regular = {key: value for key, value in data.items() if key not in LONG_TEXT_FIELDS}
            if regular:
                st.json(regular)
            for key in LONG_TEXT_FIELDS:
                value = data.get(key)
                if value:
                    st.text_area(
                        key,
                        value=str(value),
                        height=160,
                        disabled=True,
                        key=f"{event.run_id}-{event.sequence}-{key}",
                    )
            if not regular and not any(data.get(key) for key in LONG_TEXT_FIELDS):
                st.caption("No event data.")


def render_detail_selector(storage: RunStorage) -> None:
    runs = storage.list_runs(limit=DASHBOARD_RUN_LIMIT)
    selected = st.selectbox(
        "Run",
        runs,
        format_func=run_option,
        key="detail_selected_run",
    )
    render_run_detail(storage, selected.run_id)


def render_missing_database(path: Path) -> None:
    st.header("AgentLab Dashboard")
    st.info("No AgentLab database found. Run an evaluation to create the first run.")
    st.caption(f"Expected database: {path}")


def configure_page() -> None:
    st.set_page_config(page_title="AgentLab Dashboard", page_icon="🔬", layout="wide")
    st.markdown(
        """
        <style>
        .status-pill { padding: 0.3rem 0.7rem; border-radius: 999px; font-weight: 700; }
        .status-pass { color: #166534; background: #dcfce7; }
        .status-fail { color: #991b1b; background: #fee2e2; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    database_path = default_database_path()
    if not database_path.is_file():
        render_missing_database(database_path)
        return

    try:
        storage = SQLiteStorage(database_path, read_only=True)
        stats = storage.get_stats()
        if stats.total_runs == 0:
            st.header("AgentLab Dashboard")
            st.info("No evaluation runs yet.")
            return

        view = st.sidebar.radio("View", ("Overview", "Runs", "Run Detail"))
        st.sidebar.caption(f"Database: {database_path}")
        if view == "Overview":
            render_overview(storage, stats)
        elif view == "Runs":
            render_runs(storage)
        else:
            render_detail_selector(storage)
    except (StorageError, ValueError) as error:
        st.error(f"Could not load AgentLab data: {error}")


if __name__ == "__main__":
    main()

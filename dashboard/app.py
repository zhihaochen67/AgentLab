"""Streamlit entry point for the read-only AgentLab trace viewer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from agentlab.comparison import compare_experiments
from agentlab.replay import (
    ReplayEventView,
    ReplayState,
    ReplayTrace,
    format_replay_event,
    prepare_replay,
)
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
    comparison_overall_rows,
    comparison_view_data,
    event_data_for_display,
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

RECENT_RUN_LIMIT = 10
DASHBOARD_RUN_LIMIT = 500
LONG_TEXT_FIELDS = {"stdout", "stderr"}
DIAGNOSTIC_LONG_FIELDS = (
    "verification_output",
    "patch_diff",
    "stdout_summary",
    "stderr_summary",
)


def render_status(status: str) -> None:
    css_class = "status-pass" if status in {"PASS", "OK"} else "status-fail"
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

    diagnostics = failure_diagnostics_for_display(events)
    if diagnostics is not None:
        render_failure_diagnostics(diagnostics)

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
    if not runs:
        st.info("No evaluation runs yet.")
        return
    selected = st.selectbox(
        "Run",
        runs,
        format_func=run_option,
        key="detail_selected_run",
    )
    render_run_detail(storage, selected.run_id)


def experiment_option(experiment) -> str:
    return f"{experiment.status.upper()} · {experiment.label} · {experiment.experiment_id}"


def render_experiments(storage: RunStorage) -> None:
    st.header("Experiments")
    experiments = storage.list_experiments(limit=DASHBOARD_RUN_LIMIT)
    if not experiments:
        st.info("No experiments yet.")
        return

    st.subheader("Experiment List")
    st.dataframe(
        experiment_table_rows(experiments),
        width="stretch",
        hide_index=True,
    )
    selected = st.selectbox(
        "Experiment",
        experiments,
        format_func=experiment_option,
        key="experiment_selected",
    )
    metrics = storage.get_experiment_metrics(selected.experiment_id)
    detail = experiment_detail_data(selected, metrics)

    st.subheader("Experiment Detail")
    st.code(detail["experiment_id"], language=None)
    first, second = st.columns(2)
    with first:
        st.write("**Label:**", detail["label"])
        st.write("**Dataset:**", detail["dataset"])
        st.write("**Adapter:**", detail["adapter"])
        st.write("**Model:**", detail["model"])
        st.write("**Status:**", detail["status"])
    with second:
        st.write("**Trials / case:**", detail["trials_per_case"])
        st.write("**Total cases:**", detail["total_cases"])
        st.write("**Started:**", detail["started_at"])
        st.write("**Finished:**", detail["finished_at"])

    metrics_columns = st.columns(5)
    metrics_columns[0].metric("Total Runs", detail["total_runs"])
    metrics_columns[1].metric("Passed", detail["passed_runs"])
    metrics_columns[2].metric("Failed", detail["failed_runs"])
    metrics_columns[3].metric("Success Rate", f"{detail['success_rate']:.1f}%")
    metrics_columns[4].metric(
        "Average Latency", f"{detail['average_latency']:.2f}s"
    )

    st.subheader("Per-Case Results")
    case_rows = experiment_case_rows(metrics.per_case)
    if case_rows:
        st.dataframe(case_rows, width="stretch", hide_index=True)
    else:
        st.caption("No evaluation runs were executed.")

    st.subheader("Failure Type Distribution")
    failure_rows = experiment_failure_rows(metrics.failure_types)
    if failure_rows:
        st.dataframe(failure_rows, width="stretch", hide_index=True)
    else:
        st.caption("No failures recorded.")


def render_comparison(storage: RunStorage) -> None:
    st.header("Experiment Comparison")
    experiments = storage.list_experiments(limit=DASHBOARD_RUN_LIMIT)
    if not experiments:
        st.info("No experiments available for comparison.")
        return

    selectors = st.columns(2)
    baseline = selectors[0].selectbox(
        "Baseline",
        experiments,
        format_func=experiment_option,
        key="comparison_baseline",
    )
    candidate = selectors[1].selectbox(
        "Candidate",
        experiments,
        index=1 if len(experiments) > 1 else 0,
        format_func=experiment_option,
        key="comparison_candidate",
    )
    comparison = compare_experiments(
        storage,
        baseline.experiment_id,
        candidate.experiment_id,
    )
    data = comparison_view_data(comparison)

    st.write(
        f"**Baseline:** {data['baseline']['label']} "
        f"(`{data['baseline']['experiment_id']}`)"
    )
    st.write(
        f"**Candidate:** {data['candidate']['label']} "
        f"(`{data['candidate']['experiment_id']}`)"
    )
    if data["is_equivalent"]:
        st.success("Equivalent comparison")
    else:
        warning_text = "\n".join(f"- {warning}" for warning in data["warnings"])
        st.warning(f"Non-equivalent comparison\n\n{warning_text}")

    st.subheader("Overall")
    kpis = st.columns(2)
    kpis[0].metric(
        "Candidate Success Rate",
        f"{data['candidate']['success_rate']:.1f}%",
        f"{data['success_rate_delta']:+.1f} pp vs baseline",
    )
    kpis[1].metric(
        "Candidate Average Latency",
        f"{data['candidate']['average_latency']:.3f}s",
        f"{data['latency_delta']:+.3f}s vs baseline",
        delta_color="inverse",
    )
    st.dataframe(
        comparison_overall_rows(comparison),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Per-Case Results (Common Cases)")
    if data["per_case"]:
        st.dataframe(data["per_case"], width="stretch", hide_index=True)
    else:
        st.caption("No common executed cases.")

    changes = st.columns(2)
    with changes[0]:
        st.subheader("Improvements")
        if data["improvements"]:
            st.dataframe(data["improvements"], width="stretch", hide_index=True)
        else:
            st.caption("None")
    with changes[1]:
        st.subheader("Regressions")
        if data["regressions"]:
            st.dataframe(data["regressions"], width="stretch", hide_index=True)
        else:
            st.caption("None")

    st.subheader("Failure Type Comparison")
    if data["failure_types"]:
        st.dataframe(data["failure_types"], width="stretch", hide_index=True)
    else:
        st.caption("No failures recorded.")


def render_replay_event(event: ReplayEventView) -> None:
    st.subheader(f"{event.sequence} · {event.event_type}")
    summary = st.columns(3)
    summary[0].write("**Timestamp**")
    summary[0].write(event.timestamp)
    summary[1].write("**Status**")
    if event.status:
        with summary[1]:
            render_status(event.status)
    else:
        summary[1].write("—")
    summary[2].write("**Elapsed**")
    summary[2].write(f"{event.elapsed:.3f}s" if event.elapsed is not None else "—")

    diagnostics = format_failure_diagnostics(event.data.get("diagnostics"))
    if diagnostics is not None:
        render_failure_diagnostics(diagnostics)

    regular = {
        key: value
        for key, value in event.highlights.items()
        if key not in LONG_TEXT_FIELDS
    }
    if regular:
        st.json(regular)
    for key in LONG_TEXT_FIELDS:
        value = event.highlights.get(key)
        if value:
            with st.expander(f"{key} summary", expanded=False):
                st.text(str(value))

    with st.expander("Sanitized event data", expanded=False):
        event_regular = {
            key: value for key, value in event.data.items() if key not in LONG_TEXT_FIELDS
        }
        if event_regular:
            st.json(event_regular)
        for key in LONG_TEXT_FIELDS:
            value = event.data.get(key)
            if value:
                st.text_area(
                    key,
                    value=str(value),
                    height=160,
                    disabled=True,
                    key=f"replay-{event.sequence}-{key}",
                )
        if not event.data:
            st.caption("No event data.")


def render_failure_diagnostics(diagnostics: dict) -> None:
    st.subheader("Failure Diagnostics")
    first, second, third = st.columns(3)
    first.write("**Failure Type**")
    first.write(diagnostics["failure_type"])
    first.write("**Failure Phase**")
    first.write(diagnostics["failure_phase"])
    second.write("**Return Code**")
    second.write(
        diagnostics.get("returncode")
        if diagnostics.get("returncode") is not None
        else "Not available"
    )
    second.write("**Patch Applied**")
    second.write(diagnostics["patch_applied_display"])
    third.write("**Verification Result**")
    third.write(diagnostics["verification_result"])
    third.write("**Rollback Result**")
    third.write(diagnostics["rollback_result"])

    verification = st.columns(2)
    verification[0].write("**Verification Command**")
    verification[0].write(diagnostics.get("verification_command") or "Not available")
    verification[1].write("**Verification Return Code**")
    verification[1].write(
        diagnostics.get("verification_returncode")
        if diagnostics.get("verification_returncode") is not None
        else "Not available"
    )

    labels = {
        "verification_output": "Verification Output",
        "patch_diff": "Patch / Diff",
        "stdout_summary": "Agent stdout summary",
        "stderr_summary": "Agent stderr summary",
    }
    for key in DIAGNOSTIC_LONG_FIELDS:
        value = diagnostics.get(key)
        if value:
            with st.expander(labels[key], expanded=False):
                st.text(str(value))


def replay_cursor(storage_key: str, trace: ReplayTrace) -> ReplayState:
    """Read a replay cursor from UI state and clamp it to the prepared trace."""
    raw_index = st.session_state.get(storage_key, 0)
    index = raw_index if isinstance(raw_index, int) else 0
    return ReplayState(trace, index)


def render_replay(storage: RunStorage) -> None:
    st.header("Historical Replay")
    st.info("No side effects are re-executed. This view only reads persisted history.")

    runs = storage.list_runs(limit=DASHBOARD_RUN_LIMIT)
    if not runs:
        st.info("No evaluation runs are available for replay.")
        return
    selected = st.selectbox(
        "Historical run",
        runs,
        format_func=run_option,
        key="replay_selected_run",
    )
    st.code(run_detail_data(selected)["run_id"], language=None)

    trace = prepare_replay(storage.get_trace_events(selected.run_id))
    for warning in trace.warnings:
        st.warning(f"Trace integrity: {warning}")
    if not trace.events:
        st.info("This run has no replayable trace events.")
        return

    cursor_key = f"replay_index_{selected.run_id}"
    state = replay_cursor(cursor_key, trace)
    controls = st.columns(4)
    if controls[0].button("First", disabled=state.is_first, width="stretch"):
        state = state.first()
    if controls[1].button("Previous", disabled=state.is_first, width="stretch"):
        state = state.previous()
    if controls[2].button("Next", disabled=state.is_last, width="stretch"):
        state = state.next()
    if controls[3].button("Last", disabled=state.is_last, width="stretch"):
        state = state.last()
    st.session_state[cursor_key] = state.index

    st.write(f"**Step {state.current_step} / {state.total_steps}**")
    st.progress(state.current_step / state.total_steps)
    current = state.current_event
    if current is not None:
        render_replay_event(format_replay_event(current))


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
        has_experiments = bool(storage.list_experiments(limit=1))
        if stats.total_runs == 0 and not has_experiments:
            st.header("AgentLab Dashboard")
            st.info("No evaluation runs yet.")
            return

        view = st.sidebar.radio(
            "View",
            (
                "Overview",
                "Runs",
                "Run Detail",
                "Replay",
                "Experiments",
                "Comparison",
            ),
        )
        st.sidebar.caption(f"Database: {database_path}")
        if view == "Overview":
            render_overview(storage, stats)
        elif view == "Runs":
            render_runs(storage)
        elif view == "Run Detail":
            render_detail_selector(storage)
        elif view == "Replay":
            render_replay(storage)
        elif view == "Experiments":
            render_experiments(storage)
        else:
            render_comparison(storage)
    except (StorageError, ValueError) as error:
        st.error(f"Could not load AgentLab data: {error}")


if __name__ == "__main__":
    main()

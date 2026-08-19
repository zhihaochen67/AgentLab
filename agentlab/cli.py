from collections.abc import Sequence
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from agentlab.adapters import RepoDoctorAdapter
from agentlab.dataset import load_dataset
from agentlab.diagnostics import diagnostics_from_trace_data
from agentlab.experiments import (
    ExperimentAbortedError,
    ExperimentPreflightError,
    run_experiment,
)
from agentlab.models import EvalResult, Experiment, ExperimentMetrics
from agentlab.runner import evaluate_case
from agentlab.storage import SQLiteStorage, StorageError, default_database_path
from agentlab.tracer import TraceEvent

app = typer.Typer(
    help="Agent evaluation and observability platform."
)

console = Console()


@app.callback()
def main():
    """AgentLab CLI."""


@app.command("eval")
def run_eval(
    dataset: str,
    show_trace: bool = typer.Option(
        False,
        "--show-trace",
        help="Show the in-memory structured trace for each evaluation case.",
    ),
):
    """Run an AgentLab evaluation dataset."""

    cases = load_dataset(dataset)
    storage = _open_storage()

    console.print()
    console.print("[bold cyan]AgentLab Evaluation[/bold cyan]")
    console.print(f"Dataset: {dataset}")
    console.print()

    table = Table()

    table.add_column("Case")
    table.add_column("Run ID")
    table.add_column("Before")
    table.add_column("After")
    table.add_column("Result")

    passed_count = 0
    results: list[EvalResult] = []
    persistence_failed = False

    for case in cases:
        result = evaluate_case(case)
        results.append(result)

        try:
            storage.save_run(result, dataset)
        except (StorageError, TypeError, ValueError) as error:
            persistence_failed = True
            console.print(f"[red]Could not persist run {result.run_id}: {error}[/red]")

        if result.passed:
            passed_count += 1

        table.add_row(
            result.case_id,
            result.run_id,
            "[green]PASS[/green]"
            if result.tests_before_passed
            else "[red]FAIL[/red]",
            "[green]PASS[/green]"
            if result.tests_after_passed
            else "[red]FAIL[/red]",
            "[green]PASS[/green]"
            if result.passed
            else "[red]FAIL[/red]",
        )

    console.print(table)

    total = len(cases)
    success_rate = passed_count / total * 100 if total else 0

    console.print()
    console.print(f"Passed: {passed_count}/{total}")
    console.print(f"Success Rate: [bold]{success_rate:.1f}%[/bold]")
    for result in results:
        console.print(f"Run: {result.run_id}  Case: {result.case_id}")

    if show_trace:
        for result in results:
            _render_trace(
                run_id=result.run_id,
                case_id=result.case_id,
                status="PASS" if result.passed else "FAIL",
                events=result.trace,
                total_latency=_trace_total_latency(result.trace),
            )

    if persistence_failed:
        raise typer.Exit(1)


@app.command("trace")
def show_stored_trace(run_id: str):
    """Show a completed run loaded from SQLite."""
    storage = _open_storage()
    run = storage.get_run(run_id)
    if run is None:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)
    events = storage.get_trace_events(run_id)
    _render_trace(
        run_id=run.run_id,
        case_id=run.case_id,
        status=run.status,
        events=events,
        total_latency=run.total_latency,
    )


@app.command("runs")
def show_recent_runs():
    """Show the 20 most recently completed runs from SQLite."""
    storage = _open_storage()
    runs = storage.list_runs(limit=20)

    table = Table(box=None, pad_edge=False)
    table.add_column("Run ID", no_wrap=True)
    table.add_column("Case")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    for run in runs:
        status = "[green]PASS[/green]" if run.status == "PASS" else "[red]FAIL[/red]"
        table.add_row(run.run_id, run.case_id, status, f"{run.total_latency:.2f}s")
    console.print(table)


@app.command("experiment")
def run_experiment_command(
    dataset: str,
    trials: Annotated[
        int,
        typer.Option(
            "--trials",
            min=1,
            help="Number of independent trials for each selected case.",
        ),
    ] = 3,
    label: Annotated[
        str | None,
        typer.Option("--label", help="Human-readable label."),
    ] = None,
    case_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--case",
            help="Run only this case id; repeat the option to select multiple cases.",
        ),
    ] = None,
):
    """Run a persisted repeated-trial evaluation experiment."""
    cases = load_dataset(dataset, validate_initial_state=False)
    storage = _open_storage()
    try:
        execution = run_experiment(
            cases=cases,
            dataset=dataset,
            storage=storage,
            adapter=RepoDoctorAdapter(),
            trials_per_case=trials,
            label=label,
            case_ids=case_ids,
        )
    except ExperimentPreflightError as error:
        console.print(f"[red]{error}[/red]")
        console.print(f"Experiment: {error.experiment_id}")
        console.print("Status: [red]ABORTED[/red]")
        console.print("Runs executed: 0")
        raise typer.Exit(1) from error
    except (ExperimentAbortedError, StorageError, ValueError) as error:
        console.print(f"[red]Experiment aborted: {error}[/red]")
        if isinstance(error, ExperimentAbortedError):
            console.print(f"Experiment: {error.experiment_id}")
        raise typer.Exit(1) from error
    _render_experiment(execution.experiment, execution.metrics)


@app.command("experiments")
def show_recent_experiments():
    """Show the 20 most recent persisted experiments."""
    storage = _open_storage()
    experiments = storage.list_experiments(limit=20)
    if not experiments:
        console.print("No experiments yet.")
        return
    table = Table(box=None, pad_edge=False)
    table.add_column("Experiment ID", no_wrap=True)
    table.add_column("Label")
    table.add_column("Status")
    table.add_column("Runs", justify="right")
    table.add_column("Trials/Case", justify="right")
    table.add_column("Started")
    for experiment in experiments:
        table.add_row(
            experiment.experiment_id,
            experiment.label,
            _experiment_status_markup(experiment.status),
            str(experiment.total_runs),
            str(experiment.trials_per_case),
            experiment.started_at,
        )
    console.print(table)


@app.command("experiment-show")
def show_experiment(experiment_id: str):
    """Show metadata and aggregate metrics for one persisted experiment."""
    storage = _open_storage()
    experiment = storage.get_experiment(experiment_id)
    if experiment is None:
        console.print(f"[red]Experiment not found: {experiment_id}[/red]")
        raise typer.Exit(1)
    metrics = storage.get_experiment_metrics(experiment_id)
    _render_experiment(experiment, metrics)


def _render_experiment(
    experiment: Experiment,
    metrics: ExperimentMetrics,
) -> None:
    console.print()
    console.print(f"[bold cyan]Experiment:[/bold cyan] {experiment.experiment_id}")
    console.print(f"[bold]Label:[/bold] {experiment.label}")
    console.print(f"[bold]Dataset:[/bold] {experiment.dataset}")
    console.print(f"[bold]Adapter:[/bold] {experiment.adapter}")
    console.print(f"[bold]Model:[/bold] {experiment.model or 'not recorded'}")
    console.print(
        f"[bold]Status:[/bold] {_experiment_status_markup(experiment.status)}"
    )
    console.print(f"[bold]Trials per case:[/bold] {experiment.trials_per_case}")
    console.print(f"[bold]Runs:[/bold] {metrics.total_runs}")
    console.print(f"[bold]Passed:[/bold] {metrics.passed_runs}")
    console.print(f"[bold]Failed:[/bold] {metrics.failed_runs}")
    console.print(f"[bold]Success Rate:[/bold] {metrics.success_rate:.1f}%")
    console.print(f"[bold]Average Latency:[/bold] {metrics.average_latency:.2f}s")

    console.print()
    console.print("[bold]Per Case[/bold]")
    case_table = Table(box=None, pad_edge=False)
    case_table.add_column("Case")
    case_table.add_column("Passed", justify="right")
    case_table.add_column("Runs", justify="right")
    case_table.add_column("Success Rate", justify="right")
    case_table.add_column("Avg Latency", justify="right")
    for case in metrics.per_case:
        case_table.add_row(
            case.case_id,
            str(case.passed_runs),
            str(case.total_runs),
            f"{case.success_rate:.1f}%",
            f"{case.average_latency:.2f}s",
        )
    console.print(case_table)

    console.print()
    console.print("[bold]Failure Types[/bold]")
    if not metrics.failure_types:
        console.print("None")
    else:
        failure_table = Table(box=None, pad_edge=False)
        failure_table.add_column("Failure Type")
        failure_table.add_column("Count", justify="right")
        for failure_type, count in metrics.failure_types:
            failure_table.add_row(failure_type, str(count))
        console.print(failure_table)


def _experiment_status_markup(status: str) -> str:
    if status == "completed":
        return "[green]COMPLETED[/green]"
    if status == "completed_with_failures":
        return "[yellow]COMPLETED WITH FAILURES[/yellow]"
    if status == "aborted":
        return "[red]ABORTED[/red]"
    return "[cyan]RUNNING[/cyan]"


def _open_storage() -> SQLiteStorage:
    try:
        return SQLiteStorage(default_database_path())
    except StorageError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error


def _trace_total_latency(events: Sequence[TraceEvent]) -> float:
    if not events:
        return 0.0
    elapsed = events[-1].data.get("elapsed_time")
    return float(elapsed) if isinstance(elapsed, (int, float)) else 0.0


def _render_trace(
    *,
    run_id: str,
    case_id: str,
    status: str,
    events: Sequence[TraceEvent],
    total_latency: float,
) -> None:
    console.print()
    console.print(f"[bold cyan]Run:[/bold cyan] {run_id}")
    console.print(f"[bold]Case:[/bold] {case_id}")
    status_markup = "[green]PASS[/green]" if status == "PASS" else "[red]FAIL[/red]"
    console.print(f"[bold]Status:[/bold] {status_markup}")
    console.print()

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Sequence", justify="right", style="dim")
    table.add_column("Event")
    table.add_column("Status")
    table.add_column("Elapsed", justify="right")
    table.add_column("Diagnostics")

    for event in events:
        status = ""
        if event.event_type.startswith("pytest_") and event.event_type.endswith("_end"):
            if event.data.get("status") == "error":
                status = "[red]ERROR[/red]"
            else:
                status = "[green]PASS[/green]" if event.data.get("passed") else "[red]FAIL[/red]"
        elif event.event_type == "agent_end":
            status = "[green]OK[/green]" if event.data.get("status") == "ok" else "[red]ERROR[/red]"
        elif event.event_type == "run_end":
            status = "[green]PASS[/green]" if event.data.get("passed") else "[red]FAIL[/red]"
        elif event.event_type == "error":
            status = "[red]ERROR[/red]"

        elapsed = event.data.get("elapsed_time")
        elapsed_text = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else ""
        diagnostics = diagnostics_from_trace_data(event.data)
        detail = ""
        if diagnostics and diagnostics.get("failure_type"):
            detail = str(diagnostics["failure_type"]).upper()
            if diagnostics.get("failure_phase"):
                detail += f" / {diagnostics['failure_phase']}"
        table.add_row(str(event.sequence), event.event_type, status, elapsed_text, detail)

    console.print(table)
    _render_failure_diagnostics(events)
    console.print()
    console.print(f"Total latency: {total_latency:.2f}s")


def _render_failure_diagnostics(events: Sequence[TraceEvent]) -> None:
    diagnostics = next(
        (
            item
            for event in reversed(events)
            if (item := diagnostics_from_trace_data(event.data))
            and item.get("failure_type")
        ),
        None,
    )
    if diagnostics is None:
        return

    console.print()
    console.print("[bold red]Failure Diagnostics[/bold red]")
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    fields = (
        ("Failure Type", str(diagnostics.get("failure_type", "")).upper()),
        ("Failure Phase", diagnostics.get("failure_phase")),
        ("Return Code", diagnostics.get("returncode")),
        ("Patch Applied", diagnostics.get("patch_applied")),
        ("Verification Failed", diagnostics.get("verification_failed")),
        ("Rollback Attempted", diagnostics.get("rollback_attempted")),
        ("Rollback Succeeded", diagnostics.get("rollback_succeeded")),
        ("Verification Command", diagnostics.get("verification_command")),
        ("Verification Return Code", diagnostics.get("verification_returncode")),
    )
    for label, value in fields:
        table.add_row(label, "not available" if value is None else str(value))
    console.print(table)
    for label, key in (
        ("Verification Output", "verification_output"),
        ("Patch / Diff", "patch_diff"),
    ):
        value = diagnostics.get(key)
        if value:
            console.print(f"[bold]{label}:[/bold]")
            console.print(str(value), markup=False)


if __name__ == "__main__":
    app()

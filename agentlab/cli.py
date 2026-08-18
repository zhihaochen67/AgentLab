from collections.abc import Sequence

import typer
from rich.console import Console
from rich.table import Table

from agentlab.dataset import load_dataset
from agentlab.models import EvalResult
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
        table.add_row(str(event.sequence), event.event_type, status, elapsed_text)

    console.print(table)
    console.print()
    console.print(f"Total latency: {total_latency:.2f}s")


if __name__ == "__main__":
    app()

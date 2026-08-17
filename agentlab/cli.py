import typer
from rich.console import Console
from rich.table import Table

from agentlab.models import EvalResult
from agentlab.runner import evaluate_case, load_dataset

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

    console.print()
    console.print("[bold cyan]AgentLab Evaluation[/bold cyan]")
    console.print(f"Dataset: {dataset}")
    console.print()

    table = Table()

    table.add_column("Case")
    table.add_column("Before")
    table.add_column("After")
    table.add_column("Result")

    passed_count = 0
    results: list[EvalResult] = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)

        if result.passed:
            passed_count += 1

        table.add_row(
            result.case_id,
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

    if show_trace:
        for result in results:
            _render_trace(result)


def _render_trace(result: EvalResult) -> None:
    console.print()
    console.print(f"[bold cyan]Run:[/bold cyan] {result.run_id}")
    console.print(f"[bold]Case:[/bold] {result.case_id}")
    console.print()

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Sequence", justify="right", style="dim")
    table.add_column("Event")
    table.add_column("Status")
    table.add_column("Elapsed", justify="right")

    for event in result.trace:
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
    total = result.trace[-1].data.get("elapsed_time") if result.trace else None
    if isinstance(total, (int, float)):
        console.print()
        console.print(f"Total latency: {total:.2f}s")


if __name__ == "__main__":
    app()

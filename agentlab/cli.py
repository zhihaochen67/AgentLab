import typer
from rich.console import Console
from rich.table import Table

from agentlab.runner import evaluate_case, load_dataset


app = typer.Typer(
    help="Agent evaluation and observability platform."
)

console = Console()


@app.callback()
def main():
    """AgentLab CLI."""
    pass


@app.command("eval")
def run_eval(dataset: str):
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

    for case in cases:
        result = evaluate_case(case)

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


if __name__ == "__main__":
    app()

import shutil
import tempfile
from pathlib import Path

from agentlab.adapters import AgentAdapter
from agentlab.models import EvalCase
from agentlab.runner import create_workspace, evaluate_case, load_dataset


class FixingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def repair(self, workspace: Path, task: str) -> None:
        self.calls.append((workspace, task))
        (workspace / "calculator.py").write_text(
            "def add(left, right):\n    return left + right\n",
            encoding="utf-8",
        )


def make_failing_repository(root: Path) -> Path:
    repository = root / "repository"
    repository.mkdir()
    (repository / "calculator.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    (repository / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return repository


def test_load_dataset() -> None:
    cases = load_dataset("datasets/repo_doctor_basic.yaml")

    assert len(cases) == 1
    assert cases[0].id == "calculate_total_001"
    assert cases[0].expected["tests_pass"] is True


def test_workspace_is_an_isolated_copy() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        workspace = create_workspace(str(repository))

        try:
            (workspace / "calculator.py").write_text("changed\n", encoding="utf-8")
            assert "left - right" in (repository / "calculator.py").read_text(
                encoding="utf-8"
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def test_evaluate_case_runs_before_agent_and_after() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        original = (repository / "calculator.py").read_bytes()
        adapter = FixingAdapter()
        case = EvalCase("addition", str(repository), "Fix addition")

        result = evaluate_case(case, adapter=adapter)

        assert result.tests_before_passed is False
        assert result.tests_after_passed is True
        assert result.passed is True
        assert result.error is None
        assert len(adapter.calls) == 1
        assert adapter.calls[0][1] == "Fix addition"
        assert (repository / "calculator.py").read_bytes() == original
        assert not adapter.calls[0][0].exists()

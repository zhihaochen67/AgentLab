import tempfile
from pathlib import Path

import pytest
import yaml

from agentlab.adapters import AgentAdapter, AgentRunResult
from agentlab.dataset import (
    DatasetValidationError,
    FixturePytestResult,
    load_dataset,
    run_fixture_pytest,
    validate_dataset,
)
from agentlab.models import EvalCase
from agentlab.runner import evaluate_case


def make_case(
    repository: Path,
    *,
    case_id: str = "case-001",
    task: str = "Fix the implementation.",
) -> EvalCase:
    return EvalCase(
        id=case_id,
        repository=str(repository),
        task=task,
        expected={"tests_pass": True, "modified_files": ["implementation.py"]},
    )


def make_repository(root: Path, name: str = "repository") -> Path:
    repository = root / name
    repository.mkdir()
    (repository / "implementation.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    (repository / "test_implementation.py").write_text(
        "from implementation import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return repository


def failing_baseline(_repository: Path) -> FixturePytestResult:
    return FixturePytestResult(1, "one failed", "")


def write_dataset(path: Path, cases: list[EvalCase]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test-dataset",
                "cases": [
                    {
                        "id": case.id,
                        "repository": case.repository,
                        "task": case.task,
                        "expected": case.expected,
                    }
                    for case in cases
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_duplicate_case_id_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-dataset-") as directory:
        repository = make_repository(Path(directory))
        cases = [make_case(repository), make_case(repository)]

        with pytest.raises(DatasetValidationError, match="duplicate case id"):
            validate_dataset(cases, validate_initial_state=False)


def test_missing_repository_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-dataset-") as directory:
        missing = Path(directory) / "missing"

        with pytest.raises(DatasetValidationError, match="repository does not exist"):
            validate_dataset([make_case(missing)], validate_initial_state=False)


def test_empty_task_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-dataset-") as directory:
        repository = make_repository(Path(directory))

        with pytest.raises(DatasetValidationError, match="task must be non-empty"):
            validate_dataset(
                [make_case(repository, task="   ")],
                validate_initial_state=False,
            )


def test_valid_dataset_loads_all_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-dataset-") as directory:
        root = Path(directory)
        repository = make_repository(root)
        dataset = root / "dataset.yaml"
        write_dataset(dataset, [make_case(repository)])

        cases = load_dataset(dataset, pytest_runner=failing_baseline)

        assert [case.id for case in cases] == ["case-001"]
        assert cases[0].expected == {
            "tests_pass": True,
            "modified_files": ["implementation.py"],
        }


def test_every_benchmark_fixture_initially_fails_tests() -> None:
    cases = load_dataset(
        "datasets/repo_doctor_basic.yaml",
        validate_initial_state=False,
    )

    outcomes = {
        case.id: run_fixture_pytest(Path(case.repository).resolve()).returncode
        for case in cases
    }

    assert len(outcomes) == 11
    assert set(outcomes.values()) == {1}


def test_dataset_validation_does_not_modify_fixtures() -> None:
    fixture_root = Path("fixtures")
    before = {
        path.relative_to(fixture_root): path.read_bytes()
        for path in fixture_root.rglob("*")
        if path.is_file()
    }

    load_dataset("datasets/repo_doctor_basic.yaml")

    after = {
        path.relative_to(fixture_root): path.read_bytes()
        for path in fixture_root.rglob("*")
        if path.is_file()
    }
    assert after == before


class SequentialFixingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        self.tasks.append(task)
        implementation = workspace / "implementation.py"
        source = implementation.read_text(encoding="utf-8")
        implementation.write_text(
            source.replace("return left - right", "return left + right"),
            encoding="utf-8",
        )
        return AgentRunResult(0, "fixed", "")


def test_multi_case_dataset_loads_and_executes_in_order() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-dataset-") as directory:
        root = Path(directory)
        first = make_case(
            make_repository(root, "first"),
            case_id="first-case",
            task="Fix first.",
        )
        second = make_case(
            make_repository(root, "second"),
            case_id="second-case",
            task="Fix second.",
        )
        dataset = root / "multi.yaml"
        write_dataset(dataset, [first, second])
        cases = load_dataset(dataset)
        adapter = SequentialFixingAdapter()

        results = [evaluate_case(case, adapter=adapter) for case in cases]

        assert [case.id for case in cases] == ["first-case", "second-case"]
        assert adapter.tasks == ["Fix first.", "Fix second."]
        assert [result.passed for result in results] == [True, True]

"""Evaluation dataset loading and deterministic fixture validation."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentlab.models import EvalCase
from agentlab.runner import (
    UnsupportedWorkspaceSymlinkError,
    validate_workspace_symlinks,
)
from agentlab.subprocesses import run_process
from agentlab.tracer import summarize_text

FIXTURE_PYTEST_TIMEOUT = 30


class DatasetValidationError(ValueError):
    """An evaluation dataset or one of its fixtures is invalid."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("Dataset validation failed:\n- " + "\n- ".join(self.issues))


@dataclass(frozen=True)
class FixturePytestResult:
    """Minimal evidence from a fixture baseline pytest run."""

    returncode: int
    stdout: str
    stderr: str


FixturePytestRunner = Callable[[Path], FixturePytestResult]


def load_dataset(
    path: str | Path,
    *,
    validate_initial_state: bool = True,
    pytest_runner: FixturePytestRunner | None = None,
) -> list[EvalCase]:
    """Load YAML cases and validate their structure and broken baseline."""
    dataset_path = Path(path)
    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as error:
        raise DatasetValidationError((f"Cannot read dataset {dataset_path}: {error}",)) from error
    except yaml.YAMLError as error:
        raise DatasetValidationError((f"Invalid YAML in {dataset_path}: {error}",)) from error

    cases = _parse_cases(data)
    validate_dataset(
        cases,
        validate_initial_state=validate_initial_state,
        pytest_runner=pytest_runner,
    )
    return cases


def _parse_cases(data: Any) -> list[EvalCase]:
    if not isinstance(data, dict):
        raise DatasetValidationError(("Dataset root must be a mapping.",))
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DatasetValidationError(("Dataset cases must be a non-empty list.",))

    issues: list[str] = []
    cases: list[EvalCase] = []
    for index, item in enumerate(raw_cases, 1):
        if not isinstance(item, dict):
            issues.append(f"Case #{index} must be a mapping.")
            continue
        cases.append(
            EvalCase(
                id=item.get("id"),
                repository=item.get("repository"),
                task=item.get("task"),
                expected=item.get("expected", {}),
            )
        )
    if issues:
        raise DatasetValidationError(issues)
    return cases


def validate_dataset(
    cases: Sequence[EvalCase],
    *,
    validate_initial_state: bool = True,
    pytest_runner: FixturePytestRunner | None = None,
) -> None:
    """Validate case metadata, expected files, and initially failing tests."""
    issues: list[str] = []
    seen_ids: set[str] = set()
    repositories: list[tuple[EvalCase, Path]] = []

    for index, case in enumerate(cases, 1):
        label = _case_label(case, index)
        if not isinstance(case.id, str) or not case.id.strip():
            issues.append(f"{label}: id must be a non-empty string.")
        elif case.id in seen_ids:
            issues.append(f"{label}: duplicate case id {case.id!r}.")
        else:
            seen_ids.add(case.id)

        repository = _validate_repository(case, label, issues)
        if repository is not None:
            repositories.append((case, repository))

        if not isinstance(case.task, str) or not case.task.strip():
            issues.append(f"{label}: task must be non-empty.")

        _validate_expected(case, repository, label, issues)

    if issues:
        raise DatasetValidationError(issues)
    if not validate_initial_state:
        return

    run_pytest = pytest_runner or run_fixture_pytest
    baseline_issues = []
    for case, repository in repositories:
        result = run_pytest(repository)
        if result.returncode == 1:
            continue
        if result.returncode == 0:
            detail = "fixture tests unexpectedly PASS"
        else:
            output = result.stderr or result.stdout
            detail = (
                f"pytest could not establish a failing baseline "
                f"(exit {result.returncode}): {summarize_text(output, limit=300)}"
            )
        baseline_issues.append(f"Case {case.id!r}: {detail}.")
    if baseline_issues:
        raise DatasetValidationError(baseline_issues)


def _case_label(case: EvalCase, index: int) -> str:
    if isinstance(case.id, str) and case.id:
        return f"Case {case.id!r}"
    return f"Case #{index}"


def _validate_repository(
    case: EvalCase,
    label: str,
    issues: list[str],
) -> Path | None:
    if not isinstance(case.repository, str) or not case.repository.strip():
        issues.append(f"{label}: repository must be a non-empty path.")
        return None
    try:
        repository = validate_workspace_symlinks(case.repository)
    except UnsupportedWorkspaceSymlinkError as error:
        issues.append(f"{label}: {error}")
        return None
    except OSError:
        repository = Path(case.repository).resolve()
    if not repository.is_dir():
        issues.append(f"{label}: repository does not exist: {case.repository}.")
        return None
    return repository


def _validate_expected(
    case: EvalCase,
    repository: Path | None,
    label: str,
    issues: list[str],
) -> None:
    if not isinstance(case.expected, dict):
        issues.append(f"{label}: expected must be a mapping.")
        return
    if case.expected.get("tests_pass") is not True:
        issues.append(f"{label}: expected.tests_pass must be true.")

    modified_files = case.expected.get("modified_files")
    if not isinstance(modified_files, list) or not modified_files:
        issues.append(f"{label}: expected.modified_files must be a non-empty list.")
        return

    seen_files: set[str] = set()
    for value in modified_files:
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{label}: modified file paths must be non-empty strings.")
            continue
        if value in seen_files:
            issues.append(f"{label}: duplicate modified file path {value!r}.")
            continue
        seen_files.add(value)
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(f"{label}: modified file must stay inside repository: {value}.")
            continue
        if repository is not None and not (repository / relative).is_file():
            issues.append(f"{label}: modified file does not exist: {value}.")


def run_fixture_pytest(repository: Path) -> FixturePytestResult:
    """Run a fixture's tests without cache or bytecode side effects."""
    try:
        completed = run_process(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                ".",
                "-p",
                "no:cacheprovider",
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=FIXTURE_PYTEST_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DatasetValidationError(
            (f"Could not run fixture pytest in {repository}: {error}",)
        ) from error
    return FixturePytestResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )

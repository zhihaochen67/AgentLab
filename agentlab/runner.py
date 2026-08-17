import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from agentlab.adapters import AgentAdapter, RepoDoctorAdapter
from agentlab.adapters.repo_doctor import WORKSPACE_MARKER
from agentlab.models import EvalCase, EvalResult


def load_dataset(path: str) -> list[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases = []

    for item in data["cases"]:
        cases.append(
            EvalCase(
                id=item["id"],
                repository=item["repository"],
                task=item["task"],
                expected=item.get("expected", {}),
            )
        )

    return cases


def create_workspace(repository: str) -> Path:
    source = Path(repository).resolve()

    temp_dir = Path(tempfile.mkdtemp(prefix="agentlab_"))

    shutil.copytree(
        source,
        temp_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    (temp_dir / WORKSPACE_MARKER).write_text(
        "AgentLab temporary evaluation workspace.\n",
        encoding="utf-8",
    )

    return temp_dir


def run_pytest(workspace: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.returncode == 0


def evaluate_case(case: EvalCase, adapter: AgentAdapter | None = None) -> EvalResult:
    workspace = create_workspace(case.repository)
    before_passed = False
    after_passed = False

    try:
        before_passed = run_pytest(workspace)

        active_adapter = adapter or RepoDoctorAdapter()
        active_adapter.repair(workspace, case.task)

        after_passed = run_pytest(workspace)

        return EvalResult(
            case_id=case.id,
            passed=after_passed,
            tests_before_passed=before_passed,
            tests_after_passed=after_passed,
        )

    except Exception as error:  # Agent failures are represented in EvalResult.
        return EvalResult(
            case_id=case.id,
            passed=False,
            tests_before_passed=before_passed,
            tests_after_passed=after_passed,
            error=str(error),
        )

    finally:
        shutil.rmtree(workspace, ignore_errors=True)

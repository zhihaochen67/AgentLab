import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

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
    )

    return temp_dir


def run_pytest(workspace: Path) -> bool:
    result = subprocess.run(
        ["python", "-m", "pytest", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    return result.returncode == 0


def evaluate_case(case: EvalCase) -> EvalResult:
    workspace = create_workspace(case.repository)

    try:
        before_passed = run_pytest(workspace)

        # 这里以后接入 Repo Doctor。
        # 目前故意不修改代码。

        after_passed = run_pytest(workspace)

        return EvalResult(
            case_id=case.id,
            passed=after_passed,
            tests_before_passed=before_passed,
            tests_after_passed=after_passed,
        )

    except Exception as e:
        return EvalResult(
            case_id=case.id,
            passed=False,
            tests_before_passed=False,
            tests_after_passed=False,
            error=str(e),
        )

    finally:
        shutil.rmtree(workspace, ignore_errors=True)

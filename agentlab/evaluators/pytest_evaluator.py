"""Pytest-based evaluator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agentlab.evaluators.base import EvaluationOutcome, Evaluator
from agentlab.models import EvalCase


class PytestEvaluator(Evaluator):
    """Evaluate a workspace by running its pytest suite."""

    def evaluate(
        self,
        workspace: Path,
        case: EvalCase,
    ) -> EvaluationOutcome:
        """Run pytest and convert its exit status into an evaluation verdict."""

        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        return EvaluationOutcome(
            passed=result.returncode == 0,
            score=1.0 if result.returncode == 0 else 0.0,
            feedback=stderr or stdout,
            metadata={
                "evaluator": "pytest",
                "returncode": result.returncode,
            },
        )
"""Common interface for AgentLab evaluators."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlab.models import EvalCase


@dataclass(frozen=True)
class EvaluationOutcome:
    """Structured verdict produced by an evaluator."""

    passed: bool
    score: float | None = None
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("EvaluationOutcome.passed must be a boolean.")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise TypeError("EvaluationOutcome.score must be numeric or None.")
            score = float(self.score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("EvaluationOutcome.score must be within [0.0, 1.0].")
            object.__setattr__(self, "score", score)
        if not isinstance(self.feedback, str):
            raise TypeError("EvaluationOutcome.feedback must be text.")
        if not isinstance(self.metadata, dict):
            raise TypeError("EvaluationOutcome.metadata must be an object.")


class Evaluator(ABC):
    """Evaluate an agent workspace against an evaluation case."""

    @abstractmethod
    def evaluate(
        self,
        workspace: Path,
        case: EvalCase,
    ) -> EvaluationOutcome:
        """Return a structured verdict for the current workspace."""

"""Common interface for AgentLab evaluators."""

from __future__ import annotations

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


class Evaluator(ABC):
    """Evaluate an agent workspace against an evaluation case."""

    @abstractmethod
    def evaluate(
        self,
        workspace: Path,
        case: EvalCase,
    ) -> EvaluationOutcome:
        """Return a structured verdict for the current workspace."""
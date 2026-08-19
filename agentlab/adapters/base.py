"""Common interface for agents evaluated by AgentLab."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from agentlab.diagnostics import AgentDiagnostics


@dataclass(frozen=True)
class AgentRunResult:
    """Observable result returned by an agent adapter."""

    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    diagnostics: AgentDiagnostics | None = None


class AgentExecutionError(RuntimeError):
    """An adapter failure that retains safe-to-sanitize process evidence."""

    def __init__(self, message: str, result: AgentRunResult) -> None:
        super().__init__(message)
        self.result = result

    @property
    def diagnostics(self) -> AgentDiagnostics | None:
        return self.result.diagnostics


class AgentAdapter(ABC):
    """Apply an agent's repair workflow to an isolated evaluation workspace."""

    @abstractmethod
    def repair(self, workspace: Path, task: str) -> AgentRunResult | None:
        """Attempt *task* by modifying only *workspace*."""

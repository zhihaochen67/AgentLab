"""Common interface for agents evaluated by AgentLab."""

from abc import ABC, abstractmethod
from pathlib import Path


class AgentAdapter(ABC):
    """Apply an agent's repair workflow to an isolated evaluation workspace."""

    @abstractmethod
    def repair(self, workspace: Path, task: str) -> None:
        """Attempt *task* by modifying only *workspace*."""

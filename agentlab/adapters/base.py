"""Common interface for agents evaluated by AgentLab."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from agentlab.diagnostics import AgentDiagnostics


@dataclass(frozen=True)
class AgentInfo:
    """Identity metadata for an evaluated agent."""

    name: str
    description: str = ""
    agent_type: str = "general"
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentRunResult:
    """Observable result returned by an agent adapter."""

    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    diagnostics: AgentDiagnostics | None = None


@dataclass(frozen=True)
class AgentResumeHandle:
    """Opaque adapter-owned authority needed to resume one agent execution."""

    adapter: str
    session_id: str
    reason: str
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class AgentSuspended(RuntimeError):
    """An agent paused without reaching either success or terminal failure."""

    def __init__(
        self,
        handle: AgentResumeHandle,
        result: AgentRunResult | None = None,
    ) -> None:
        self.handle = handle
        self.result = result
        super().__init__(f"Agent execution suspended: {handle.reason}")


class AgentResumeUnsupportedError(RuntimeError):
    """An adapter was asked to resume but has no resumable execution support."""


class AgentExecutionError(RuntimeError):
    """An adapter failure that retains safe-to-sanitize process evidence."""

    def __init__(self, message: str, result: AgentRunResult) -> None:
        super().__init__(message)
        self.result = result

    @property
    def diagnostics(self) -> AgentDiagnostics | None:
        return self.result.diagnostics


@dataclass(frozen=True)
class AgentPreflightResult:
    """Non-secret adapter metadata established before evaluation begins."""

    model: str | None = None


class AgentPreflightError(RuntimeError):
    """An adapter cannot safely start an experiment."""

    def __init__(
        self,
        missing_variables: tuple[str, ...] = (),
        *,
        invalid_variables: tuple[str, ...] = (),
    ) -> None:
        self.missing_variables = missing_variables
        self.invalid_variables = invalid_variables
        if invalid_variables:
            message = ", ".join(f"{name} appears invalid" for name in invalid_variables)
        else:
            names = ", ".join(missing_variables)
            message = f"Missing provider configuration: {names}"
        super().__init__(message)


class AgentAdapter(ABC):
    """Apply an agent's workflow to an isolated evaluation workspace."""

    @property
    def info(self) -> AgentInfo:
        """Return agent identity metadata."""
        return AgentInfo(
            name=self.__class__.__name__,
        )

    @abstractmethod
    def repair(self, workspace: Path, task: str) -> AgentRunResult | None:
        """Attempt *task* by modifying only *workspace*."""

    def preflight(self) -> AgentPreflightResult:
        """Validate experiment prerequisites without executing an evaluation."""
        return AgentPreflightResult()

    def resume(
        self,
        workspace: Path,
        handle: AgentResumeHandle,
    ) -> AgentRunResult | None:
        """Resume an adapter-owned execution, or fail explicitly if unsupported."""
        raise AgentResumeUnsupportedError(
            f"{type(self).__name__} does not support resumable execution."
        )

    def trace_metadata(self) -> dict[str, str]:
        """Return non-secret metadata describing the agent execution."""
        return {}

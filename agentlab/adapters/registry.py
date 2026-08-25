"""Agent adapter registry."""

from collections.abc import Callable

from agentlab.adapters.base import AgentAdapter
from agentlab.adapters.repo_doctor import RepoDoctorAdapter

AgentFactory = Callable[..., AgentAdapter]


class AgentRegistry:
    """Store and create available agent adapters."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentFactory] = {}

    def register(
        self,
        name: str,
        factory: AgentFactory,
    ) -> None:
        """Register an agent adapter factory."""
        self._agents[name] = factory

    def create(
        self,
        name: str,
        **kwargs,
    ) -> AgentAdapter:
        """Create an agent adapter by name."""
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        return self._agents[name](**kwargs)

    def list_agents(self) -> list[str]:
        """Return registered agent names."""
        return sorted(self._agents)


def create_default_registry() -> AgentRegistry:
    """Create registry with built-in agents."""

    registry = AgentRegistry()

    registry.register(
        "repo_doctor",
        RepoDoctorAdapter,
    )

    return registry
"""Agent integrations available to AgentLab."""

from agentlab.adapters.base import (
    AgentAdapter,
    AgentExecutionError,
    AgentPreflightError,
    AgentPreflightResult,
    AgentRunResult,
)
from agentlab.adapters.registry import (
    AgentRegistry,
    create_default_registry,
)
from agentlab.adapters.repo_doctor import RepoDoctorAdapter
from agentlab.diagnostics import AgentDiagnostics, AgentFailureType

__all__ = [
    "AgentAdapter",
    "AgentDiagnostics",
    "AgentExecutionError",
    "AgentFailureType",
    "AgentPreflightError",
    "AgentPreflightResult",
    "AgentRegistry",
    "AgentRunResult",
    "RepoDoctorAdapter",
    "create_default_registry",
]
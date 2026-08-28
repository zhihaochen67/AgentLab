"""Agent integrations available to AgentLab."""

from agentlab.adapters.base import (
    AgentAdapter,
    AgentExecutionError,
    AgentInfo,
    AgentPreflightError,
    AgentPreflightResult,
    AgentResumeHandle,
    AgentResumeUnsupportedError,
    AgentRunResult,
    AgentSuspended,
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
    "AgentInfo",
    "AgentPreflightError",
    "AgentPreflightResult",
    "AgentRegistry",
    "AgentResumeHandle",
    "AgentResumeUnsupportedError",
    "AgentRunResult",
    "AgentSuspended",
    "RepoDoctorAdapter",
    "create_default_registry",
]

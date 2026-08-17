"""Agent integrations available to AgentLab."""

from agentlab.adapters.base import AgentAdapter, AgentExecutionError, AgentRunResult
from agentlab.adapters.repo_doctor import RepoDoctorAdapter

__all__ = ["AgentAdapter", "AgentExecutionError", "AgentRunResult", "RepoDoctorAdapter"]

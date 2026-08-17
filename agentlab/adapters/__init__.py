"""Agent integrations available to AgentLab."""

from agentlab.adapters.base import AgentAdapter
from agentlab.adapters.repo_doctor import RepoDoctorAdapter

__all__ = ["AgentAdapter", "RepoDoctorAdapter"]

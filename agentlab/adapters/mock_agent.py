"""Simple mock agent for benchmark pipeline testing."""

from dataclasses import dataclass
from pathlib import Path

from agentlab.adapters.base import (
    AgentAdapter,
    AgentInfo,
    AgentRunResult,
)


@dataclass(frozen=True)
class MockAgentAdapter(AgentAdapter):
    """A deterministic agent used for benchmark pipeline testing."""

    agent_version: str | None = None
    prompt_variant: str | None = None

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name="mock_agent",
            description="Deterministic benchmark test agent",
            agent_type="test-agent",
            capabilities=(
                "benchmark",
                "testing",
            ),
        )

    def trace_metadata(self) -> dict[str, str]:
        metadata = {}

        if self.agent_version:
            metadata["agent_version"] = self.agent_version

        if self.prompt_variant:
            metadata["prompt_variant"] = self.prompt_variant

        return metadata

    def repair(
        self,
        workspace: Path,
        task: str,
    ) -> AgentRunResult:
        """Apply deterministic fixes for benchmark fixtures."""

        inventory = workspace / "inventory.py"

        if inventory.exists():
            content = inventory.read_text(encoding="utf-8")
            content = content.replace(
                "total -= price",
                "total += price",
            )
            inventory.write_text(
                content,
                encoding="utf-8",
            )

        return AgentRunResult(
            returncode=0,
            stdout="Mock agent completed deterministic repair",
            stderr="",
        )
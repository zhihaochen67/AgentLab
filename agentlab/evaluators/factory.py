"""Evaluator factory used by CLI/experiment entry points."""

from __future__ import annotations

from agentlab.evaluators.base import Evaluator
from agentlab.evaluators.llm_judge import LLMJudgeEvaluator
from agentlab.evaluators.openai_compatible import (
    JudgeProviderConfig,
    OpenAICompatibleJudge,
)

VALID_EVALUATOR_NAMES = ("none", "llm_judge")


def create_evaluator(name: str) -> Evaluator | None:
    """Build the evaluator selected by name, or None for 'none'.

    Judge configuration is validated eagerly at construction, so invalid
    AGENTLAB_JUDGE_* settings fail here — before any case execution and
    before any network call. Unknown names raise ValueError listing the
    valid choices; there is no silent fallback.
    """
    normalized = name.strip()
    if normalized == "none":
        return None
    if normalized == "llm_judge":
        config = JudgeProviderConfig.from_env()
        judge = OpenAICompatibleJudge(config)
        return LLMJudgeEvaluator(
            judge,
            model=config.model,
            judge_name="openai_compatible",
        )
    raise ValueError(
        f"Unknown evaluator {name!r}; expected one of: "
        f"{', '.join(VALID_EVALUATOR_NAMES)}."
    )

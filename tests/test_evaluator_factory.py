import pytest

from agentlab.evaluators import (
    VALID_EVALUATOR_NAMES,
    JudgeConfigurationError,
    LLMJudgeEvaluator,
    create_evaluator,
)


def valid_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENTLAB_JUDGE_API_KEY", "sk-judge-1234567890abcdef")
    monkeypatch.setenv("AGENTLAB_JUDGE_BASE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("AGENTLAB_JUDGE_MODEL", "deepseek-v4-flash")
    monkeypatch.delenv("AGENTLAB_JUDGE_TIMEOUT_SECONDS", raising=False)


def test_factory_none_returns_none() -> None:
    assert create_evaluator("none") is None


def test_factory_llm_judge_builds_evaluator_without_network(monkeypatch) -> None:
    valid_env(monkeypatch)
    network_called = []

    def forbidden_network(*_args, **_kwargs):
        network_called.append(True)
        raise AssertionError("factory construction must not use the network")

    monkeypatch.setattr("socket.create_connection", forbidden_network)

    evaluator = create_evaluator("llm_judge")

    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator.model == "deepseek-v4-flash"
    assert evaluator.judge_name == "openai_compatible"
    assert network_called == []


def test_factory_unknown_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown evaluator 'fancy_judge'"):
        create_evaluator("fancy_judge")

    assert "llm_judge" in str(
        pytest.raises(ValueError, create_evaluator, "fancy_judge").value
    )
    assert VALID_EVALUATOR_NAMES == ("none", "llm_judge")


def test_factory_invalid_env_fails_fast(monkeypatch) -> None:
    monkeypatch.delenv("AGENTLAB_JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("AGENTLAB_JUDGE_BASE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("AGENTLAB_JUDGE_MODEL", "deepseek-v4-flash")

    with pytest.raises(JudgeConfigurationError, match="AGENTLAB_JUDGE_API_KEY"):
        create_evaluator("llm_judge")

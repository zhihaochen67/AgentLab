import subprocess
from pathlib import Path

import pytest

from agentlab.evaluators import (
    EvaluationOutcome,
    JudgeResponseError,
    JudgeVerdict,
    LLMJudgeEvaluator,
    PytestEvaluator,
    build_judge_prompt,
    collect_evidence,
)
from agentlab.evaluators.llm_judge import EvidenceBundle
from agentlab.models import EvalCase


def test_pytest_evaluator_passes_for_green_test_suite(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "def test_ok():\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    case = EvalCase(
        id="pytest-pass",
        repository=str(tmp_path),
        task="Run tests",
    )

    outcome = PytestEvaluator().evaluate(tmp_path, case)

    assert outcome.passed is True
    assert outcome.score == 1.0
    assert outcome.metadata["evaluator"] == "pytest"
    assert outcome.metadata["returncode"] == 0


def test_pytest_evaluator_fails_for_red_test_suite(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "def test_fail():\n"
        "    assert 1 == 2\n",
        encoding="utf-8",
    )

    case = EvalCase(
        id="pytest-fail",
        repository=str(tmp_path),
        task="Run tests",
    )

    outcome = PytestEvaluator().evaluate(tmp_path, case)

    assert outcome.passed is False
    assert outcome.score == 0.0
    assert outcome.metadata["evaluator"] == "pytest"
    assert outcome.metadata["returncode"] != 0


def test_pytest_evaluator_preserves_timeout_taxonomy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def timeout(*_args, **kwargs):
        assert kwargs["timeout"] == 120
        raise subprocess.TimeoutExpired(["pytest"], timeout=120)

    monkeypatch.setattr("agentlab.evaluators.pytest_evaluator.run_process", timeout)

    with pytest.raises(RuntimeError, match="PytestEvaluator exceeded the 120-second"):
        PytestEvaluator().evaluate(
            tmp_path,
            EvalCase("pytest-timeout", str(tmp_path), "Run tests"),
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"passed": 1}, TypeError),
        ({"passed": True, "score": float("nan")}, ValueError),
        ({"passed": True, "score": 1.1}, ValueError),
        ({"passed": True, "feedback": None}, TypeError),
        ({"passed": True, "metadata": []}, TypeError),
    ],
)
def test_evaluation_outcome_rejects_invalid_adapter_values(kwargs, error_type) -> None:
    with pytest.raises(error_type):
        EvaluationOutcome(**kwargs)


class ScriptedJudge:
    """Mock judge that returns a canned response and records its prompts."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class ExplodingJudge:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __call__(self, prompt: str) -> str:
        raise self.error


def test_judge_verdict_parses_valid_payload() -> None:
    verdict = JudgeVerdict.parse(
        '{"passed": true, "score": 0.75, "feedback": "mostly fine", '
        '"model": "mock-llm-1", "extra": "ignored"}'
    )

    assert verdict.passed is True
    assert verdict.score == 0.75
    assert verdict.feedback == "mostly fine"
    assert verdict.model == "mock-llm-1"


def test_judge_verdict_accepts_integer_score() -> None:
    verdict = JudgeVerdict.parse('{"passed": true, "score": 1, "feedback": "ok"}')

    assert verdict.score == 1.0


def test_judge_verdict_rejects_invalid_json() -> None:
    with pytest.raises(JudgeResponseError, match="invalid JSON"):
        JudgeVerdict.parse("{not valid json")


def test_judge_verdict_rejects_non_object_payload() -> None:
    with pytest.raises(JudgeResponseError, match="JSON object"):
        JudgeVerdict.parse("[1, 2, 3]")


def test_judge_verdict_rejects_oversized_response() -> None:
    with pytest.raises(JudgeResponseError, match="exceeds"):
        JudgeVerdict.parse("x" * 200_000)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"score": 0.5, "feedback": "ok"}',
        '{"passed": true, "feedback": "ok"}',
        '{"passed": true, "score": 0.5}',
    ],
)
def test_judge_verdict_rejects_missing_required_fields(payload: str) -> None:
    with pytest.raises(JudgeResponseError):
        JudgeVerdict.parse(payload)


@pytest.mark.parametrize(
    "score",
    ["1.5", "-0.01", "null"],
)
def test_judge_verdict_rejects_out_of_range_score(score: str) -> None:
    payload = f'{{"passed": true, "score": {score}, "feedback": "ok"}}'

    with pytest.raises(JudgeResponseError, match="score"):
        JudgeVerdict.parse(payload)


@pytest.mark.parametrize(
    "payload",
    [
        '{"passed": true, "score": "0.8", "feedback": "ok"}',
        '{"passed": true, "score": true, "feedback": "ok"}',
        '{"passed": "yes", "score": 0.8, "feedback": "ok"}',
        '{"passed": true, "score": 0.8, "feedback": 5}',
        '{"passed": true, "score": 0.8, "feedback": "   "}',
        '{"passed": true, "score": 0.8, "feedback": "ok", "model": 7}',
    ],
)
def test_judge_verdict_rejects_invalid_field_types(payload: str) -> None:
    with pytest.raises(JudgeResponseError):
        JudgeVerdict.parse(payload)


def test_collect_evidence_skips_ignored_directories_and_binary_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Example\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("plain text\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git data", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "site.py").write_text("venv data", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"\x00binary")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04\x00\x00\x00")
    (tmp_path / ".env").write_text("SECRET_TOKEN=hidden\n", encoding="utf-8")
    (tmp_path / "data.db").write_text("not really sqlite", encoding="utf-8")

    bundle = collect_evidence(tmp_path)

    paths = [item.relative_path for item in bundle.files]
    assert paths == ["app.py", "notes.txt", "README.md"]
    assert bundle.skipped_binary == 3  # image.png, archive.zip, data.db
    assert bundle.truncated is False
    assert bundle.total_bytes == sum(len(item.content) for item in bundle.files)


def test_collect_evidence_respects_total_budget(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"module_{index}.py").write_text(
            "def value():\n    return " + "x" * 1_000 + "\n",
            encoding="utf-8",
        )

    bundle = collect_evidence(tmp_path, max_bytes=2_000)

    assert bundle.total_bytes <= 2_000
    assert bundle.truncated is True
    assert len(bundle.files) < 10


def test_collect_evidence_respects_max_files(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"module_{index}.py").write_text(f"value = {index}\n", encoding="utf-8")

    bundle = collect_evidence(tmp_path, max_files=3)

    assert len(bundle.files) == 3
    assert bundle.truncated is True


def test_collect_evidence_truncates_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "large.py").write_text("x = " + "y" * 10_000, encoding="utf-8")

    bundle = collect_evidence(tmp_path, max_file_bytes=2_000)

    assert len(bundle.files) == 1
    assert bundle.files[0].truncated is True
    assert len(bundle.files[0].content) == 2_000
    assert bundle.truncated is True


def test_collect_evidence_redacts_secrets_from_file_content(
    monkeypatch, tmp_path: Path
) -> None:
    secret = "evidence-secret-123456"
    monkeypatch.setenv("EVIDENCE_API_KEY", secret)
    (tmp_path / "app.py").write_text(
        f"API_KEY={secret}\nAuthorization: Bearer {secret}\n",
        encoding="utf-8",
    )

    bundle = collect_evidence(tmp_path)

    assert secret not in bundle.files[0].content
    assert "[REDACTED]" in bundle.files[0].content


def test_collect_evidence_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        collect_evidence(tmp_path / "missing")


def test_collect_evidence_rejects_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        collect_evidence(tmp_path, max_bytes=0)


def test_llm_judge_evaluator_passes_with_valid_verdict(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    judge = ScriptedJudge(
        '{"passed": true, "score": 1.0, "feedback": "correct", "model": "mock-llm-2"}'
    )
    case = EvalCase("addition", str(tmp_path), "Implement add")

    outcome = LLMJudgeEvaluator(judge, judge_name="mock-judge").evaluate(tmp_path, case)

    assert outcome.passed is True
    assert outcome.score == 1.0
    assert outcome.feedback == "correct"
    assert outcome.metadata["evaluator"] == "llm_judge"
    assert outcome.metadata["judge"] == "mock-judge"
    assert outcome.metadata["model"] == "mock-llm-2"
    assert outcome.metadata["evidence_files"] == 1
    assert outcome.metadata["evidence_bytes"] > 0
    assert outcome.metadata["evidence_truncated"] is False

    prompt = judge.prompts[0]
    assert "Implement add" in prompt
    assert "return a + b" in prompt
    assert '"passed"' in prompt


def test_llm_judge_evaluator_fails_with_failing_verdict(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    judge = ScriptedJudge('{"passed": false, "score": 0.1, "feedback": "wrong op"}')

    outcome = LLMJudgeEvaluator(judge).evaluate(
        tmp_path,
        EvalCase("addition", str(tmp_path), "Implement add"),
    )

    assert outcome.passed is False
    assert outcome.score == 0.1
    assert outcome.feedback == "wrong op"


def test_llm_judge_evaluator_metadata_prefers_constructor_model(
    tmp_path: Path,
) -> None:
    judge = ScriptedJudge('{"passed": true, "score": 1.0, "feedback": "ok", "model": "judge-says"}')

    outcome = LLMJudgeEvaluator(judge, model="declared-model").evaluate(
        tmp_path,
        EvalCase("addition", str(tmp_path), "Implement add"),
    )

    assert outcome.metadata["model"] == "declared-model"


def test_llm_judge_evaluator_infers_judge_name_from_callable(tmp_path: Path) -> None:
    judge = ScriptedJudge('{"passed": true, "score": 1.0, "feedback": "ok"}')

    outcome = LLMJudgeEvaluator(judge).evaluate(
        tmp_path,
        EvalCase("addition", str(tmp_path), "Implement add"),
    )

    assert outcome.metadata["judge"] == "ScriptedJudge"


def test_llm_judge_evaluator_raises_on_invalid_judge_response(tmp_path: Path) -> None:
    judge = ScriptedJudge("{broken json")

    with pytest.raises(JudgeResponseError, match="invalid JSON"):
        LLMJudgeEvaluator(judge).evaluate(
            tmp_path,
            EvalCase("addition", str(tmp_path), "Implement add"),
        )


def test_llm_judge_evaluator_propagates_judge_exceptions(tmp_path: Path) -> None:
    judge = ExplodingJudge(RuntimeError("provider offline"))

    with pytest.raises(RuntimeError, match="provider offline"):
        LLMJudgeEvaluator(judge).evaluate(
            tmp_path,
            EvalCase("addition", str(tmp_path), "Implement add"),
        )


def test_llm_judge_evaluator_rejects_invalid_evidence_limits(tmp_path: Path) -> None:
    judge = ScriptedJudge('{"passed": true, "score": 1.0, "feedback": "ok"}')

    with pytest.raises(ValueError, match="max_file_bytes"):
        LLMJudgeEvaluator(judge, max_file_bytes=0).evaluate(
            tmp_path,
            EvalCase("addition", str(tmp_path), "Implement add"),
        )


def test_llm_judge_prompt_marks_empty_evidence(tmp_path: Path) -> None:
    prompt = build_judge_prompt(
        EvalCase("empty", str(tmp_path), "Nothing to do"),
        EvidenceBundle(files=(), truncated=False),
    )

    assert "no readable evidence files" in prompt
    assert "Nothing to do" in prompt

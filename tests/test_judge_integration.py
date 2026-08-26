import json
import tempfile
from pathlib import Path

import httpx
import pytest

from agentlab.adapters import AgentAdapter, AgentRunResult
from agentlab.evaluators import (
    JudgeProviderConfig,
    LLMJudgeEvaluator,
    OpenAICompatibleJudge,
)
from agentlab.experiments import run_experiment
from agentlab.models import EvalCase
from agentlab.storage import SQLiteStorage


def make_failing_repository(root: Path) -> Path:
    repository = root / "repository"
    repository.mkdir()
    (repository / "calculator.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    (repository / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return repository


class FixingAdapter(AgentAdapter):
    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        (workspace / "calculator.py").write_text(
            "def add(left, right):\n    return left + right\n",
            encoding="utf-8",
        )
        return AgentRunResult(0, "fixed", "")


class NoOpAdapter(AgentAdapter):
    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        return AgentRunResult(0, "no changes", "")


class FirstTrialFixingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        self.calls += 1
        if self.calls == 1:
            (workspace / "calculator.py").write_text(
                "def add(left, right):\n    return left + right\n",
                encoding="utf-8",
            )
        return AgentRunResult(0, "maybe fixed", "")


def make_judge_evaluator(
    *,
    passed: bool = True,
    score: float = 1.0,
    feedback: str = "all good",
    calls: list[httpx.Request] | None = None,
) -> LLMJudgeEvaluator:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        content = json.dumps(
            {"passed": passed, "score": score, "feedback": feedback}
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": content}}
                ]
            },
        )

    config = JudgeProviderConfig(
        api_key="sk-test-1234567890abcdef",
        base_url="http://judge.test/v1",
        model="judge-model",
        timeout_seconds=5.0,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = OpenAICompatibleJudge(config, client=client)
    return LLMJudgeEvaluator(judge, model=config.model, judge_name="openai_compatible")


def test_fake_provider_experiment_metrics_round_trip() -> None:
    calls: list[httpx.Request] = []
    evaluator = make_judge_evaluator(calls=calls)

    with tempfile.TemporaryDirectory(prefix="agentlab-judge-int-") as directory:
        repository = make_failing_repository(Path(directory))
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        execution = run_experiment(
            cases=[EvalCase("addition", str(repository), "Fix addition")],
            dataset="dataset.yaml",
            storage=storage,
            adapter=FixingAdapter(),
            trials_per_case=2,
            evaluator=evaluator,
            validator=lambda _cases: None,
        )

        experiment_id = execution.experiment.experiment_id
        metrics = storage.get_experiment_metrics(experiment_id)
        runs = storage.list_runs(experiment_id=experiment_id)
        outcomes = storage.get_evaluator_outcomes(runs[0].run_id)

    assert len(calls) == 2
    assert metrics.total_runs == 2
    assert metrics.passed_runs == 2
    assert [run.status for run in runs] == ["PASS", "PASS"]
    assert len(outcomes) == 1
    assert outcomes[0].status == "PASS"
    assert outcomes[0].metadata["judge"] == "openai_compatible"
    assert outcomes[0].metadata["model"] == "judge-model"

    assert len(metrics.evaluator_metrics) == 1
    judge_metrics = metrics.evaluator_metrics[0]
    assert judge_metrics.evaluator == "LLMJudgeEvaluator"
    assert judge_metrics.total_outcomes == 2
    assert judge_metrics.evaluated_runs == 2
    assert judge_metrics.passed_outcomes == 2
    assert judge_metrics.pass_rate == 100.0
    assert judge_metrics.average_score == 1.0
    assert judge_metrics.coverage_rate == 100.0


def test_failing_judge_verdict_fails_the_run() -> None:
    evaluator = make_judge_evaluator(
        passed=False,
        score=0.1,
        feedback="wrong semantics",
    )

    with tempfile.TemporaryDirectory(prefix="agentlab-judge-int-") as directory:
        repository = make_failing_repository(Path(directory))
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        execution = run_experiment(
            cases=[EvalCase("addition", str(repository), "Fix addition")],
            dataset="dataset.yaml",
            storage=storage,
            adapter=FixingAdapter(),
            trials_per_case=2,
            evaluator=evaluator,
            validator=lambda _cases: None,
        )

        experiment_id = execution.experiment.experiment_id
        metrics = storage.get_experiment_metrics(experiment_id)
        runs = storage.list_runs(experiment_id=experiment_id)

    assert [run.status for run in runs] == ["FAIL", "FAIL"]
    assert all(run.tests_after_passed is True for run in runs)
    assert metrics.passed_runs == 0
    assert metrics.failed_runs == 2
    judge_metrics = metrics.evaluator_metrics[0]
    assert judge_metrics.passed_outcomes == 0
    assert judge_metrics.failed_outcomes == 2
    assert judge_metrics.pass_rate == 0.0
    assert judge_metrics.average_score == 0.1


def test_red_test_suite_skips_judge_entirely() -> None:
    calls: list[httpx.Request] = []
    evaluator = make_judge_evaluator(calls=calls)

    with tempfile.TemporaryDirectory(prefix="agentlab-judge-int-") as directory:
        repository = make_failing_repository(Path(directory))
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        execution = run_experiment(
            cases=[EvalCase("addition", str(repository), "Do not fix addition")],
            dataset="dataset.yaml",
            storage=storage,
            adapter=NoOpAdapter(),
            trials_per_case=2,
            evaluator=evaluator,
            validator=lambda _cases: None,
        )

        metrics = storage.get_experiment_metrics(execution.experiment.experiment_id)

    assert calls == []
    assert metrics.evaluator_metrics == ()
    assert metrics.passed_runs == 0


def test_judge_only_called_for_green_trials() -> None:
    calls: list[httpx.Request] = []
    evaluator = make_judge_evaluator(calls=calls)

    with tempfile.TemporaryDirectory(prefix="agentlab-judge-int-") as directory:
        repository = make_failing_repository(Path(directory))
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        execution = run_experiment(
            cases=[EvalCase("addition", str(repository), "Fix addition")],
            dataset="dataset.yaml",
            storage=storage,
            adapter=FirstTrialFixingAdapter(),
            trials_per_case=2,
            evaluator=evaluator,
            validator=lambda _cases: None,
        )

        metrics = storage.get_experiment_metrics(execution.experiment.experiment_id)

    assert len(calls) == 1
    judge_metrics = metrics.evaluator_metrics[0]
    assert judge_metrics.total_outcomes == 1
    assert judge_metrics.evaluated_runs == 1
    assert judge_metrics.coverage_rate == 50.0
    assert metrics.total_runs == 2
    assert metrics.passed_runs == 1


def test_case_executor_and_evaluator_conflict_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-judge-int-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        with pytest.raises(ValueError, match="case_executor"):
            run_experiment(
                cases=[EvalCase("case-a", "unused", "Fix A")],
                dataset="dataset.yaml",
                storage=storage,
                adapter=NoOpAdapter(),
                trials_per_case=1,
                evaluator=make_judge_evaluator(),
                case_executor=lambda _case, _adapter: None,
                validator=lambda _cases: None,
            )

        assert storage.list_experiments() == ()

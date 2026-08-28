import pytest
from typer.testing import CliRunner

from agentlab.cli import app
from agentlab.evaluators import LLMJudgeEvaluator
from agentlab.experiments import ExperimentExecution
from agentlab.models import EvalCase, EvalResult, Experiment, ExperimentMetrics


def make_completed(case: EvalCase) -> EvalResult:
    return EvalResult(
        case_id=case.id,
        passed=True,
        tests_before_passed=False,
        tests_after_passed=True,
        error=None,
        run_id=f"run-{case.id}",
        trace=(),
    )


def make_execution() -> ExperimentExecution:
    experiment = Experiment(
        experiment_id="exp-cli-evaluator",
        label="cli-evaluator",
        dataset="dataset.yaml",
        adapter="RepoDoctorAdapter",
        model=None,
        trials_per_case=1,
        total_cases=1,
        total_runs=0,
        started_at="2026-08-19T01:00:00+00:00",
        finished_at="2026-08-19T01:01:00+00:00",
        status="completed",
    )
    metrics = ExperimentMetrics(
        experiment_id="exp-cli-evaluator",
        total_runs=0,
        passed_runs=0,
        failed_runs=0,
        success_rate=0.0,
        average_latency=0.0,
        per_case=(),
        failure_types=(),
    )
    return ExperimentExecution(experiment, metrics)


class RecordingStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str]] = []

    def save_run(self, result: EvalResult, dataset: str) -> None:
        self.saved.append((result.run_id, dataset))


def clear_judge_env(monkeypatch) -> None:
    for name in (
        "AGENTLAB_JUDGE_API_KEY",
        "AGENTLAB_JUDGE_BASE_URL",
        "AGENTLAB_JUDGE_MODEL",
        "AGENTLAB_JUDGE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def set_valid_judge_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENTLAB_JUDGE_API_KEY", "sk-judge-1234567890abcdef")
    monkeypatch.setenv("AGENTLAB_JUDGE_BASE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("AGENTLAB_JUDGE_MODEL", "deepseek-v4-flash")


def test_eval_default_does_not_require_judge_env(monkeypatch) -> None:
    clear_judge_env(monkeypatch)
    storage = RecordingStorage()
    calls: dict = {}

    def fake_evaluate(case, **kwargs):
        calls["kwargs"] = kwargs
        return make_completed(case)

    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: [EvalCase("case-a", "repo", "Fix A")],
    )
    monkeypatch.setattr("agentlab.cli.evaluate_case", fake_evaluate)
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: storage)

    result = CliRunner().invoke(app, ["eval", "dataset.yaml"])

    assert result.exit_code == 0
    assert calls["kwargs"]["evaluator"] is None
    assert calls["kwargs"]["evaluator_name"] is None
    assert calls["kwargs"]["dataset"] == "dataset.yaml"
    assert [run_id for run_id, _dataset in storage.saved] == ["run-case-a"]


def test_experiment_default_does_not_require_judge_env(monkeypatch) -> None:
    clear_judge_env(monkeypatch)
    captured: dict = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return make_execution()

    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: [EvalCase("case-a", "repo", "Fix A")],
    )
    monkeypatch.setattr("agentlab.cli.run_experiment", fake_run_experiment)
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: object())

    result = CliRunner().invoke(app, ["experiment", "dataset.yaml", "--trials", "1"])

    assert result.exit_code == 0
    assert captured["evaluator"] is None


def test_benchmark_default_does_not_require_judge_env(monkeypatch) -> None:
    clear_judge_env(monkeypatch)
    captured: dict = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return make_execution()

    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: [EvalCase("case-a", "repo", "Fix A")],
    )
    monkeypatch.setattr("agentlab.cli.run_experiment", fake_run_experiment)
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: object())

    result = CliRunner().invoke(app, ["benchmark", "dataset.yaml", "--trials", "1"])

    assert result.exit_code == 0
    assert captured["evaluator"] is None


def test_eval_evaluator_none_preserves_old_behavior(monkeypatch) -> None:
    clear_judge_env(monkeypatch)
    storage = RecordingStorage()
    calls: dict = {}

    def fake_evaluate(case, **kwargs):
        calls["kwargs"] = kwargs
        return make_completed(case)

    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: [EvalCase("case-a", "repo", "Fix A")],
    )
    monkeypatch.setattr("agentlab.cli.evaluate_case", fake_evaluate)
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: storage)

    result = CliRunner().invoke(app, ["eval", "dataset.yaml", "--evaluator", "none"])

    assert result.exit_code == 0
    assert calls["kwargs"]["evaluator"] is None
    assert calls["kwargs"]["evaluator_name"] is None
    assert calls["kwargs"]["dataset"] == "dataset.yaml"


def test_eval_llm_judge_is_wired(monkeypatch) -> None:
    set_valid_judge_env(monkeypatch)
    storage = RecordingStorage()
    captured: dict = {}

    def fake_evaluate(case, **kwargs):
        captured["kwargs"] = kwargs
        return make_completed(case)

    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: [EvalCase("case-a", "repo", "Fix A")],
    )
    monkeypatch.setattr("agentlab.cli.evaluate_case", fake_evaluate)
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: storage)

    result = CliRunner().invoke(
        app, ["eval", "dataset.yaml", "--evaluator", "llm_judge"]
    )

    assert result.exit_code == 0
    evaluator = captured["kwargs"]["evaluator"]
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator.judge_name == "openai_compatible"
    assert evaluator.model == "deepseek-v4-flash"


def test_experiment_llm_judge_is_wired(monkeypatch) -> None:
    set_valid_judge_env(monkeypatch)
    captured: dict = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return make_execution()

    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: [EvalCase("case-a", "repo", "Fix A")],
    )
    monkeypatch.setattr("agentlab.cli.run_experiment", fake_run_experiment)
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: object())

    result = CliRunner().invoke(
        app,
        ["experiment", "dataset.yaml", "--trials", "1", "--evaluator", "llm_judge"],
    )

    assert result.exit_code == 0
    evaluator = captured["evaluator"]
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator.judge_name == "openai_compatible"


def test_benchmark_llm_judge_is_wired(monkeypatch) -> None:
    set_valid_judge_env(monkeypatch)
    captured: dict = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return make_execution()

    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: [EvalCase("case-a", "repo", "Fix A")],
    )
    monkeypatch.setattr("agentlab.cli.run_experiment", fake_run_experiment)
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: object())

    result = CliRunner().invoke(
        app,
        ["benchmark", "dataset.yaml", "--trials", "1", "--evaluator", "llm_judge"],
    )

    assert result.exit_code == 0
    evaluator = captured["evaluator"]
    assert isinstance(evaluator, LLMJudgeEvaluator)
    assert evaluator.judge_name == "openai_compatible"


def test_invalid_judge_config_exits_before_case_execution(monkeypatch) -> None:
    monkeypatch.setenv("AGENTLAB_JUDGE_API_KEY", "your api key replace me")
    monkeypatch.setenv("AGENTLAB_JUDGE_BASE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("AGENTLAB_JUDGE_MODEL", "deepseek-v4-flash")
    called: list[str] = []

    def forbidden_load(*_args, **_kwargs):
        called.append("load_dataset")
        return []

    monkeypatch.setattr("agentlab.cli.load_dataset", forbidden_load)
    monkeypatch.setattr(
        "agentlab.cli.run_experiment",
        lambda **_kwargs: pytest.fail("run_experiment must not run"),
    )
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: object())

    result = CliRunner().invoke(
        app,
        ["experiment", "dataset.yaml", "--trials", "1", "--evaluator", "llm_judge"],
    )

    assert result.exit_code == 1
    assert "AGENTLAB_JUDGE_API_KEY appears invalid" in result.stdout
    assert called == []


def test_cli_error_never_prints_secret(monkeypatch) -> None:
    secret = "judge-cli-secret-987654"
    monkeypatch.setenv("AGENTLAB_JUDGE_API_KEY", f"your api key {secret}")
    monkeypatch.setenv("AGENTLAB_JUDGE_BASE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("AGENTLAB_JUDGE_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(
        "agentlab.cli.load_dataset",
        lambda _dataset, **_kwargs: pytest.fail("load must not run"),
    )
    monkeypatch.setattr("agentlab.cli._open_storage", lambda: object())

    result = CliRunner().invoke(
        app,
        ["eval", "dataset.yaml", "--evaluator", "llm_judge"],
    )

    assert result.exit_code == 1
    assert secret not in result.stdout
    assert "AGENTLAB_JUDGE_API_KEY appears invalid" in result.stdout

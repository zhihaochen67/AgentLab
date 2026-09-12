import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from agentlab.adapters import AgentAdapter, AgentRunResult
from agentlab.dataset import load_dataset
from agentlab.evaluators import EvaluationOutcome, Evaluator, LLMJudgeEvaluator
from agentlab.models import EvalCase
from agentlab.runner import (
    PytestRunResult,
    PytestTimeoutError,
    UnsupportedWorkspaceSymlinkError,
    create_workspace,
    evaluate_case,
    run_pytest,
    workspace_manifest,
)
from agentlab.storage import SQLiteStorage


class FixingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        self.calls.append((workspace, task))
        (workspace / "calculator.py").write_text(
            "def add(left, right):\n    return left + right\n",
            encoding="utf-8",
        )
        return AgentRunResult(0, "repair complete", "")


class FailingAdapter(AgentAdapter):
    def __init__(self, message: str) -> None:
        self.message = message
        self.workspace: Path | None = None

    def repair(self, workspace: Path, task: str) -> None:
        self.workspace = workspace
        raise RuntimeError(self.message)


class SecretReportingAdapter(FixingAdapter):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        super().repair(workspace, task)
        return AgentRunResult(
            0,
            f"Authorization: Bearer {self.secret}; API_KEY={self.secret}",
            f"token={self.secret}",
        )


class CalculateTotalAdapter(AgentAdapter):
    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        path = workspace / "inventory.py"
        source = path.read_text(encoding="utf-8")
        path.write_text(
            source.replace("total -= price", "total += price"),
            encoding="utf-8",
        )
        return AgentRunResult(0, "fixed inventory.py", "")


class MetadataFixingAdapter(FixingAdapter):
    def trace_metadata(self) -> dict[str, str]:
        return {
            "prompt_variant": "candidate-v2",
            "agent_version": "repo-doctor-0.2.0",
            "model": "deepseek-v4-flash",
        }


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


def test_load_dataset() -> None:
    cases = load_dataset("datasets/repo_doctor_basic.yaml")

    assert len(cases) == 11
    assert cases[0].id == "calculate_total_001"
    assert cases[0].expected["tests_pass"] is True


def test_workspace_is_an_isolated_copy() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        workspace = create_workspace(str(repository))

        try:
            (workspace / "calculator.py").write_text("changed\n", encoding="utf-8")
            assert "left - right" in (repository / "calculator.py").read_text(
                encoding="utf-8"
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def test_directory_symlink_is_rejected_before_workspace_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    real_directory = repository / "realdir"
    real_directory.mkdir(parents=True)
    (real_directory / "value.txt").write_text("value\n", encoding="utf-8")
    link = repository / "linked-directory"
    _symlink_or_skip(link, real_directory, target_is_directory=True)

    monkeypatch.setattr(
        "agentlab.runner.shutil.copytree",
        lambda *_args, **_kwargs: pytest.fail(
            "workspace copy must not start before symlink validation"
        ),
    )

    with pytest.raises(
        UnsupportedWorkspaceSymlinkError,
        match="directory symlinks are unsupported.*linked-directory",
    ):
        create_workspace(str(repository))


def test_regular_file_symlink_copy_and_manifest_semantics_are_consistent(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    target = repository / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    link = repository / "linked.py"
    _symlink_or_skip(link, target, target_is_directory=False)

    source_manifest = workspace_manifest(repository)
    workspace = create_workspace(str(repository))
    try:
        assert not (workspace / "linked.py").is_symlink()
        assert (workspace / "linked.py").read_bytes() == target.read_bytes()
        assert workspace_manifest(workspace) == source_manifest
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        if os.name == "nt":
            pytest.skip(f"Windows symlink privilege is unavailable: {error}")
        raise


def test_evaluate_case_runs_before_agent_and_after() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        original = (repository / "calculator.py").read_bytes()
        adapter = FixingAdapter()
        case = EvalCase("addition", str(repository), "Fix addition")

        result = evaluate_case(case, adapter=adapter)

        assert result.tests_before_passed is False
        assert result.tests_after_passed is True
        assert result.passed is True
        assert result.error is None
        assert len(adapter.calls) == 1
        assert adapter.calls[0][1] == "Fix addition"
        assert (repository / "calculator.py").read_bytes() == original
        assert not adapter.calls[0][0].exists()

        event_types = [event.event_type for event in result.trace]
        assert event_types == [
            "run_start",
            "pytest_before_start",
            "pytest_before_end",
            "agent_start",
            "agent_end",
            "pytest_after_start",
            "pytest_after_end",
            "run_end",
        ]
        assert result.run_id
        assert all(event.run_id == result.run_id for event in result.trace)
        assert result.trace[2].data["passed"] is False
        assert result.trace[4].data["returncode"] == 0
        assert result.trace[6].data["passed"] is True
        assert result.trace[-1].data["final_status"] == "pass"


def test_declared_workspace_change_contract_is_enforced() -> None:
    class TamperingAdapter(FixingAdapter):
        def repair(self, workspace: Path, task: str) -> AgentRunResult:
            result = super().repair(workspace, task)
            (workspace / "test_calculator.py").write_text(
                "def test_trivial():\n    assert True\n",
                encoding="utf-8",
            )
            return result

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        result = evaluate_case(
            EvalCase(
                "addition",
                str(repository),
                "Fix addition",
                {"modified_files": ["calculator.py"]},
            ),
            adapter=TamperingAdapter(),
        )

    assert result.tests_after_passed is True
    assert result.passed is False
    assert result.error is not None
    assert "unexpected files changed: test_calculator.py" in result.error
    verification = next(
        event
        for event in result.trace
        if event.event_type == "workspace_verification_end"
    )
    assert verification.data == {
        "status": "fail",
        "passed": False,
        "modified_file_count": 2,
        "modified_files": ["calculator.py", "test_calculator.py"],
        "missing_expected_files": [],
        "unexpected_files": ["test_calculator.py"],
        "paths_truncated": False,
        "elapsed_time": verification.data["elapsed_time"],
    }
    assert result.trace[-1].data["failure_reason"] == "workspace_contract_failed"


def test_declared_workspace_change_contract_passes_exact_repair() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        result = evaluate_case(
            EvalCase(
                "addition",
                str(repository),
                "Fix addition",
                {"modified_files": ["calculator.py"]},
            ),
            adapter=FixingAdapter(),
        )

    assert result.passed is True
    assert [
        event.event_type for event in result.trace if "workspace_" in event.event_type
    ] == [
        "workspace_baseline",
        "workspace_verification_start",
        "workspace_verification_end",
        "final_workspace_verification_start",
        "final_workspace_verification_end",
    ]
    assert result.trace[-1].data["workspace_changes_passed"] is True


def test_pytest_before_side_effects_are_part_of_pre_agent_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = make_failing_repository(tmp_path)
    calls = 0

    def side_effecting_pytest(workspace: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            (workspace / "generated.txt").write_text("baseline\n", encoding="utf-8")
            return PytestRunResult(False, 1, "", "")
        return PytestRunResult(True, 0, "", "")

    monkeypatch.setattr("agentlab.runner.run_pytest", side_effecting_pytest)
    result = evaluate_case(
        EvalCase(
            "addition",
            str(repository),
            "Fix addition",
            {"modified_files": ["calculator.py"]},
        ),
        adapter=FixingAdapter(),
    )

    assert result.passed is True
    verification = next(
        event
        for event in result.trace
        if event.event_type == "workspace_verification_end"
    )
    assert verification.data["modified_files"] == ["calculator.py"]


@pytest.mark.parametrize("mutation_path", ["calculator.py", "pytest-output.txt"])
def test_pytest_after_workspace_mutation_fails_final_identity_gate(
    tmp_path: Path,
    monkeypatch,
    mutation_path: str,
) -> None:
    repository = make_failing_repository(tmp_path)
    calls = 0

    def side_effecting_pytest(workspace: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (workspace / mutation_path).write_text(
                "changed by verification\n",
                encoding="utf-8",
            )
        return PytestRunResult(calls != 1, 1 if calls == 1 else 0, "", "")

    class MustNotRunEvaluator:
        def evaluate(self, workspace: Path, case: EvalCase) -> EvaluationOutcome:
            raise AssertionError("evaluator must not run after workspace mutation")

    monkeypatch.setattr("agentlab.runner.run_pytest", side_effecting_pytest)
    result = evaluate_case(
        EvalCase(
            "addition",
            str(repository),
            "Fix addition",
            {"modified_files": ["calculator.py"]},
        ),
        adapter=FixingAdapter(),
        evaluator=MustNotRunEvaluator(),
    )

    assert result.tests_after_passed is True
    assert result.passed is False
    assert "pytest-after modified the workspace" in result.error
    assert result.trace[-1].data["failure_reason"] == (
        "verification_modified_workspace"
    )
    final = next(
        event
        for event in result.trace
        if event.event_type == "final_workspace_verification_end"
    )
    assert final.data["passed"] is False
    assert final.data["verification_modified_files"] == [mutation_path]
    assert "evaluator_start" not in [event.event_type for event in result.trace]


def test_runtime_rejects_already_passing_repair_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = make_failing_repository(tmp_path)
    adapter = FixingAdapter()
    monkeypatch.setattr(
        "agentlab.runner.run_pytest",
        lambda _workspace: PytestRunResult(True, 0, "", ""),
    )

    result = evaluate_case(
        EvalCase("addition", str(repository), "Fix addition"),
        adapter=adapter,
    )

    assert result.tests_before_passed is True
    assert result.tests_after_passed is False
    assert result.passed is False
    assert result.error == (
        "Repair baseline unexpectedly passed pytest; a repair evaluation "
        "requires a failing before state."
    )
    assert adapter.calls == []
    assert result.trace[-1].data["failure_reason"] == "baseline_already_passing"


def test_agent_trace_records_variant_metadata_without_full_prompt() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=MetadataFixingAdapter(),
        )

    agent_events = [
        event
        for event in result.trace
        if event.event_type in {"agent_start", "agent_end"}
    ]
    assert len(agent_events) == 2
    for event in agent_events:
        assert event.data["prompt_variant"] == "candidate-v2"
        assert event.data["agent_version"] == "repo-doctor-0.2.0"
        assert event.data["model"] == "deepseek-v4-flash"
        assert event.data["task_provided"] is True
        assert "system_prompt" not in event.data


def test_adapter_failure_records_error_and_cleans_workspace() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        adapter = FailingAdapter("agent exploded")
        case = EvalCase("addition", str(repository), "Fix addition")

        result = evaluate_case(case, adapter=adapter)

        assert result.passed is False
        assert result.error == "agent exploded"
        assert adapter.workspace is not None
        assert not adapter.workspace.exists()
        assert [event.event_type for event in result.trace] == [
            "run_start",
            "pytest_before_start",
            "pytest_before_end",
            "agent_start",
            "agent_end",
            "error",
            "run_end",
        ]
        assert result.trace[4].data["status"] == "error"
        assert result.trace[5].data["phase"] == "agent"
        assert result.trace[-1].data["final_status"] == "fail"


def test_pytest_exception_records_error_and_run_end(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        adapter = FixingAdapter()

        def fail_pytest(_workspace: Path):
            raise OSError("pytest unavailable")

        monkeypatch.setattr("agentlab.runner.run_pytest", fail_pytest)
        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=adapter,
        )

        assert result.error == "pytest unavailable"
        assert [event.event_type for event in result.trace] == [
            "run_start",
            "pytest_before_start",
            "pytest_before_end",
            "error",
            "run_end",
        ]
        assert result.trace[2].data["status"] == "error"
        assert result.trace[3].data["phase"] == "pytest_before"


def test_pytest_gate_has_a_clear_timeout_failure(monkeypatch, tmp_path: Path) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["pytest"], timeout=120)

    monkeypatch.setattr("agentlab.runner.run_process", timeout)

    with pytest.raises(PytestTimeoutError, match="120-second limit"):
        run_pytest(tmp_path)


def test_trace_redacts_secret_from_task_and_agent_output(monkeypatch) -> None:
    secret = "test-secret-value-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        result = evaluate_case(
            EvalCase("addition", str(repository), f"Fix addition using {secret}"),
            adapter=SecretReportingAdapter(secret),
        )

        rendered_trace = repr(result.trace)
        assert secret not in rendered_trace
        assert "[REDACTED]" in rendered_trace


def test_evaluation_does_not_modify_original_fixture() -> None:
    fixture = Path("fixtures/calculate_total_001")
    before = {
        path.name: path.read_bytes() for path in fixture.iterdir() if path.is_file()
    }
    case = EvalCase("calculate_total", str(fixture), "Fix calculate_total")

    result = evaluate_case(case, adapter=CalculateTotalAdapter())

    after = {
        path.name: path.read_bytes() for path in fixture.iterdir() if path.is_file()
    }
    assert result.tests_before_passed is False
    assert result.tests_after_passed is True
    assert before == after


def test_optional_evaluator_can_reject_green_test_suite() -> None:
    class RejectingEvaluator:
        def evaluate(self, workspace: Path, case: EvalCase) -> EvaluationOutcome:
            assert workspace.exists()
            assert case.id == "addition"

            return EvaluationOutcome(
                passed=False,
                score=0.25,
                feedback="Semantic requirements were not satisfied.",
                metadata={"evaluator": "rejecting-test"},
            )

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=RejectingEvaluator(),
        )

    assert result.tests_after_passed is True
    assert result.passed is False
    assert result.error is None

    evaluator_events = [
        event
        for event in result.trace
        if event.event_type in {"evaluator_start", "evaluator_end"}
    ]

    assert len(evaluator_events) == 2
    assert evaluator_events[1].data["status"] == "fail"
    assert evaluator_events[1].data["passed"] is False
    assert evaluator_events[1].data["score"] == 0.25
    assert evaluator_events[1].data["feedback"] == (
        "Semantic requirements were not satisfied."
    )
    assert evaluator_events[1].data["metadata"] == {"evaluator": "rejecting-test"}
    assert result.trace[-1].data["final_status"] == "fail"


def test_optional_evaluator_is_skipped_when_tests_fail() -> None:
    class NoOpAdapter(AgentAdapter):
        def repair(self, workspace: Path, task: str) -> AgentRunResult:
            return AgentRunResult(0, "no changes", "")

    class MustNotRunEvaluator:
        def evaluate(self, workspace: Path, case: EvalCase) -> EvaluationOutcome:
            raise AssertionError("evaluator must not run")

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Do not fix addition"),
            adapter=NoOpAdapter(),
            evaluator=MustNotRunEvaluator(),
        )

    assert result.tests_after_passed is False
    assert result.passed is False

    event_types = [event.event_type for event in result.trace]

    assert "evaluator_start" not in event_types
    assert "evaluator_end" not in event_types
    assert result.trace[-1].data["final_status"] == "fail"


def test_optional_evaluator_pass_records_verdict_in_trace() -> None:
    class PassingEvaluator(Evaluator):
        def evaluate(self, workspace: Path, case: EvalCase) -> EvaluationOutcome:
            return EvaluationOutcome(
                passed=True,
                score=0.9,
                feedback="Semantics look good.",
                metadata={"evaluator": "passing-test", "rubric": "semantic"},
            )

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=PassingEvaluator(),
        )

    assert result.tests_after_passed is True
    assert result.passed is True
    assert result.error is None

    evaluator_events = [
        event
        for event in result.trace
        if event.event_type in {"evaluator_start", "evaluator_end"}
    ]

    assert len(evaluator_events) == 2
    assert evaluator_events[0].data["evaluator"] == "PassingEvaluator"
    assert evaluator_events[1].data["status"] == "pass"
    assert evaluator_events[1].data["passed"] is True
    assert evaluator_events[1].data["score"] == 0.9
    assert evaluator_events[1].data["feedback"] == "Semantics look good."
    assert evaluator_events[1].data["metadata"] == {
        "evaluator": "passing-test",
        "rubric": "semantic",
    }
    assert result.trace[-1].data["final_status"] == "pass"


def test_evaluator_exception_fails_run_with_diagnostic_trace() -> None:
    class ExplodingEvaluator(Evaluator):
        def evaluate(self, workspace: Path, case: EvalCase) -> EvaluationOutcome:
            raise RuntimeError("judge exploded")

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=ExplodingEvaluator(),
        )

    assert result.tests_after_passed is True
    assert result.passed is False
    assert result.error == "judge exploded"

    event_types = [event.event_type for event in result.trace]
    assert event_types == [
        "run_start",
        "pytest_before_start",
        "pytest_before_end",
        "agent_start",
        "agent_end",
        "pytest_after_start",
        "pytest_after_end",
        "evaluator_start",
        "evaluator_end",
        "error",
        "run_end",
    ]
    assert result.trace[7].data["evaluator"] == "ExplodingEvaluator"
    assert result.trace[8].data["status"] == "error"
    assert result.trace[8].data["passed"] is False
    assert result.trace[8].data["error_type"] == "RuntimeError"
    assert result.trace[9].data["phase"] == "evaluator"
    assert result.trace[-1].data["final_status"] == "fail"


def test_llm_judge_evaluator_pass_verdict_passes_run() -> None:
    judge = ScriptedJudge(
        '{"passed": true, "score": 1.0, "feedback": "all good", "model": "mock-llm-1"}'
    )

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=LLMJudgeEvaluator(judge, judge_name="mock-judge"),
        )

    assert result.tests_after_passed is True
    assert result.passed is True
    assert result.error is None

    evaluator_end = next(
        event for event in result.trace if event.event_type == "evaluator_end"
    )
    assert evaluator_end.data["status"] == "pass"
    assert evaluator_end.data["score"] == 1.0
    assert evaluator_end.data["feedback"] == "all good"
    assert evaluator_end.data["metadata"]["judge"] == "mock-judge"
    assert evaluator_end.data["metadata"]["model"] == "mock-llm-1"
    assert evaluator_end.data["metadata"]["evidence_files"] > 0
    assert result.trace[-1].data["final_status"] == "pass"


def test_llm_judge_evaluator_fail_verdict_fails_run() -> None:
    judge = ScriptedJudge(
        '{"passed": false, "score": 0.2, "feedback": "wrong semantics"}'
    )

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=LLMJudgeEvaluator(judge, judge_name="mock-judge"),
        )

    assert result.tests_after_passed is True
    assert result.passed is False
    assert result.error is None
    assert result.trace[-1].data["final_status"] == "fail"


def test_invalid_judge_response_fails_run_diagnostically() -> None:
    judge = ScriptedJudge("{not valid json")

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=LLMJudgeEvaluator(judge, judge_name="mock-judge"),
        )

    assert result.tests_after_passed is True
    assert result.passed is False
    assert result.error is not None
    assert "invalid JSON" in result.error

    error_event = next(event for event in result.trace if event.event_type == "error")
    assert error_event.data["phase"] == "evaluator"
    assert error_event.data["error_type"] == "JudgeResponseError"
    assert result.trace[-1].data["final_status"] == "fail"


def test_evaluator_trace_redacts_judge_secrets(monkeypatch) -> None:
    secret = "judge-secret-value-98765"
    monkeypatch.setenv("JUDGE_API_KEY", secret)

    class SecretLeakingEvaluator(Evaluator):
        def evaluate(self, workspace: Path, case: EvalCase) -> EvaluationOutcome:
            return EvaluationOutcome(
                passed=False,
                score=0.0,
                feedback=f"Authorization: Bearer {secret} and api_key={secret}",
                metadata={"rubric": f"leaked {secret}", "api_key": secret},
            )

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))

        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=SecretLeakingEvaluator(),
        )

    rendered_trace = repr(result.trace)
    assert secret not in rendered_trace
    assert "[REDACTED]" in rendered_trace
    assert result.passed is False


class ScriptedJudge:
    """Mock LLM judge that returns a canned response."""

    def __init__(self, response: str) -> None:
        self.response = response

    def __call__(self, prompt: str) -> str:
        return self.response


def test_evaluator_outcome_persists_from_runner_trace() -> None:
    judge = ScriptedJudge(
        '{"passed": true, "score": 1.0, "feedback": "all good", "model": "mock-llm-1"}'
    )

    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        repository = make_failing_repository(Path(directory))
        result = evaluate_case(
            EvalCase("addition", str(repository), "Fix addition"),
            adapter=FixingAdapter(),
            evaluator=LLMJudgeEvaluator(judge, judge_name="mock-judge"),
        )
        database = Path(directory) / "persist.db"
        storage = SQLiteStorage(database)
        storage.save_run(result, "dataset.yaml")
        outcomes = storage.get_evaluator_outcomes(result.run_id)

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.run_id == result.run_id
    assert outcome.evaluator == "LLMJudgeEvaluator"
    assert outcome.status == "PASS"
    assert outcome.passed is True
    assert outcome.score == 1.0
    assert outcome.feedback == "all good"
    assert outcome.error_type is None
    assert outcome.metadata["judge"] == "mock-judge"
    assert outcome.metadata["model"] == "mock-llm-1"
    assert outcome.metadata["evidence_files"] > 0

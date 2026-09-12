import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentlab.cli import app
from agentlab.models import EvalCase, EvalResult
from agentlab.storage import SQLiteStorage, StorageError, default_database_path
from agentlab.tracer import TraceEvent


def make_result(
    run_id: str,
    *,
    case_id: str = "case-001",
    passed: bool = True,
    started_at: str = "2026-08-18T01:00:00+00:00",
    finished_at: str = "2026-08-18T01:00:01+00:00",
    error: str | None = None,
    latency: float = 1.25,
    extra_data: dict | None = None,
    reverse_trace: bool = False,
) -> EvalResult:
    events = (
        TraceEvent(
            run_id,
            1,
            "run_start",
            started_at,
            {"case_id": case_id, "adapter": "FakeAdapter"},
        ),
        TraceEvent(
            run_id,
            2,
            "pytest_before_end",
            finished_at,
            {"status": "fail", "passed": False, "returncode": 1},
        ),
        TraceEvent(
            run_id,
            3,
            "agent_end",
            finished_at,
            {"status": "ok", **(extra_data or {})},
        ),
        TraceEvent(
            run_id,
            4,
            "pytest_after_end",
            finished_at,
            {"status": "pass" if passed else "fail", "passed": passed},
        ),
        TraceEvent(
            run_id,
            5,
            "run_end",
            finished_at,
            {
                "passed": passed,
                "tests_before_passed": False,
                "tests_after_passed": passed,
                "workspace_changes_passed": True,
                "final_status": "pass" if passed else "fail",
                "elapsed_time": latency,
            },
        ),
    )
    if reverse_trace:
        events = tuple(reversed(events))
    return EvalResult(
        case_id=case_id,
        passed=passed,
        tests_before_passed=False,
        tests_after_passed=passed,
        error=error,
        run_id=run_id,
        trace=events,
    )


def test_save_and_load_run_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result("run-one")

        storage.save_run(result, "datasets/example.yaml")
        stored = storage.get_run("run-one")

        assert stored is not None
        assert stored.run_id == "run-one"
        assert stored.case_id == "case-001"
        assert stored.dataset == "datasets/example.yaml"
        assert stored.adapter == "FakeAdapter"
        assert stored.status == "PASS"
        assert stored.total_latency == 1.25
        assert stored.tests_before_passed is False
        assert stored.tests_after_passed is True


def test_trace_events_are_saved_and_loaded_in_sequence_order() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(make_result("ordered-run", reverse_trace=True), "dataset.yaml")

        events = storage.get_trace_events("ordered-run")

        assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
        assert [event.event_type for event in events] == [
            "run_start",
            "pytest_before_end",
            "agent_end",
            "pytest_after_end",
            "run_end",
        ]


def contract_backed_result(run_id: str) -> EvalResult:
    timestamp = "2026-08-18T01:00:01+00:00"
    events = (
        TraceEvent(
            run_id,
            1,
            "run_start",
            timestamp,
            {
                "case_id": "case-001",
                "adapter": "FakeAdapter",
                "workspace_contract_required": True,
                "evaluator": None,
            },
        ),
        TraceEvent(
            run_id,
            2,
            "pytest_before_end",
            timestamp,
            {"status": "fail", "passed": False},
        ),
        TraceEvent(run_id, 3, "workspace_baseline", timestamp, {}),
        TraceEvent(run_id, 4, "agent_end", timestamp, {"status": "ok"}),
        TraceEvent(
            run_id,
            5,
            "workspace_verification_end",
            timestamp,
            {"status": "pass", "passed": True},
        ),
        TraceEvent(
            run_id,
            6,
            "pytest_after_end",
            timestamp,
            {"status": "pass", "passed": True},
        ),
        TraceEvent(
            run_id,
            7,
            "final_workspace_verification_end",
            timestamp,
            {"status": "pass", "passed": True, "contract_passed": True},
        ),
        TraceEvent(
            run_id,
            8,
            "run_end",
            timestamp,
            {
                "passed": True,
                "tests_before_passed": False,
                "tests_after_passed": True,
                "workspace_changes_passed": True,
                "final_status": "pass",
                "elapsed_time": 1.0,
            },
        ),
    )
    return EvalResult(
        case_id="case-001",
        passed=True,
        tests_before_passed=False,
        tests_after_passed=True,
        run_id=run_id,
        trace=events,
    )


def without_event(result: EvalResult, event_type: str) -> EvalResult:
    events = tuple(
        replace(event, sequence=sequence)
        for sequence, event in enumerate(
            (event for event in result.trace if event.event_type != event_type),
            start=1,
        )
    )
    return replace(result, trace=events)


def test_storage_rejects_pass_with_only_run_boundaries(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "agentlab.db")
    result = make_result("boundary-only")
    result = replace(
        result,
        trace=(result.trace[0], replace(result.trace[-1], sequence=2)),
    )

    with pytest.raises(ValueError, match="pytest_before_end"):
        storage.save_run(result, "dataset.yaml")


def test_storage_rejects_pass_missing_pytest_before(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "agentlab.db")

    with pytest.raises(ValueError, match="pytest_before_end"):
        storage.save_run(
            without_event(make_result("missing-before"), "pytest_before_end"),
            "dataset.yaml",
        )


def test_storage_rejects_contract_pass_missing_final_verification(
    tmp_path: Path,
) -> None:
    storage = SQLiteStorage(tmp_path / "agentlab.db")

    with pytest.raises(ValueError, match="final_workspace_verification_end"):
        storage.save_run(
            without_event(
                contract_backed_result("missing-final"),
                "final_workspace_verification_end",
            ),
            "dataset.yaml",
        )


def test_storage_rejects_configured_evaluator_without_evidence(
    tmp_path: Path,
) -> None:
    storage = SQLiteStorage(tmp_path / "agentlab.db")
    result = make_result("missing-evaluator")
    start = replace(
        result.trace[0],
        data={**result.trace[0].data, "evaluator": "FakeEvaluator"},
    )

    with pytest.raises(ValueError, match="evaluator_end"):
        storage.save_run(
            replace(result, trace=(start, *result.trace[1:])),
            "dataset.yaml",
        )


def test_complete_contract_pass_persists_and_replays(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "agentlab.db")
    result = contract_backed_result("complete-pass")

    storage.save_run(result, "dataset.yaml")

    assert storage.get_run(result.run_id).status == "PASS"
    assert storage.get_trace_events(result.run_id) == result.trace


def test_ordinary_fail_trace_still_persists(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "agentlab.db")
    result = make_result("ordinary-fail", passed=False, error="tests failed")

    storage.save_run(result, "dataset.yaml")

    assert storage.get_run(result.run_id).status == "FAIL"


def test_missing_run_id_returns_no_data() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        assert storage.get_run("missing") is None
        assert storage.get_trace_events("missing") == ()


def test_default_database_path_uses_external_state_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_root = (tmp_path / "state").resolve()
    monkeypatch.delenv("AGENTLAB_DB_PATH", raising=False)
    monkeypatch.setenv("AGENTLAB_STATE_ROOT", str(state_root))

    assert default_database_path() == state_root / "agentlab.db"


def test_storage_imports_independently_in_fresh_interpreter(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("AGENTLAB_DB_PATH", None)
    state_root = (tmp_path / "state").resolve()
    environment["AGENTLAB_STATE_ROOT"] = str(state_root)
    script = (
        "import sys; "
        "assert 'agentlab.storage' not in sys.modules; "
        "from agentlab.storage import default_database_path; "
        "assert 'agentlab.execution_sessions' not in sys.modules; "
        "print(default_database_path())"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == state_root / "agentlab.db"


def test_multiple_runs_are_listed_most_recent_first() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result("older", finished_at="2026-08-18T01:00:01+00:00"),
            "dataset.yaml",
        )
        storage.save_run(
            make_result("newer", finished_at="2026-08-18T02:00:01+00:00"),
            "dataset.yaml",
        )

        assert [run.run_id for run in storage.list_runs()] == ["newer", "older"]
        assert [run.run_id for run in storage.list_runs(limit=1)] == ["newer"]


def test_stats_case_ids_and_run_filters() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        empty_stats = storage.get_stats()
        assert empty_stats.total_runs == 0
        assert empty_stats.success_rate == 0.0
        assert empty_stats.average_latency == 0.0
        assert storage.list_case_ids() == ()

        storage.save_run(
            make_result("passing", case_id="case-a", latency=2.0),
            "dataset.yaml",
        )
        storage.save_run(
            make_result("failing", case_id="case-b", passed=False, latency=4.0),
            "dataset.yaml",
        )

        stats = storage.get_stats()
        assert stats.total_runs == 2
        assert stats.successful_runs == 1
        assert stats.failed_runs == 1
        assert stats.success_rate == 50.0
        assert stats.average_latency == 3.0
        assert storage.list_case_ids() == ("case-a", "case-b")
        assert [run.run_id for run in storage.list_runs(status="PASS")] == ["passing"]
        assert [run.run_id for run in storage.list_runs(status="FAIL")] == ["failing"]
        assert [run.run_id for run in storage.list_runs(case_id="case-b")] == [
            "failing"
        ]
        assert storage.list_runs(status="PASS", case_id="case-b") == ()

        with pytest.raises(ValueError, match="status must be"):
            storage.list_runs(status="UNKNOWN")


def test_read_only_storage_cannot_modify_database() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        writable = SQLiteStorage(database)
        writable.save_run(make_result("existing-run"), "dataset.yaml")
        before = database.read_bytes()

        read_only = SQLiteStorage(database, read_only=True)
        assert read_only.get_stats().total_runs == 1
        assert read_only.get_run("existing-run") is not None
        with pytest.raises(StorageError, match="read-only"):
            read_only.save_run(make_result("forbidden-run"), "dataset.yaml")

        assert database.read_bytes() == before


def test_duplicate_run_id_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result("duplicate")
        storage.save_run(result, "dataset.yaml")

        with pytest.raises(StorageError, match="UNIQUE constraint failed"):
            storage.save_run(result, "dataset.yaml")


def test_idempotent_run_persistence_accepts_exact_replay_and_rejects_conflict() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result("finalizing-run")
        result = replace(
            result,
            trace=(
                *result.trace[:-1],
                TraceEvent(
                    result.run_id,
                    len(result.trace),
                    "evaluator_end",
                    result.trace[-1].timestamp,
                    {
                        "evaluator": "FakeEvaluator",
                        "status": "pass",
                        "passed": True,
                        "score": 0.9,
                        "feedback": "accepted",
                        "metadata": {"source": "test"},
                        "elapsed_time": 0.1,
                    },
                ),
                replace(result.trace[-1], sequence=len(result.trace) + 1),
            ),
        )

        storage.save_run_idempotently(result, "dataset.yaml")
        storage.save_run_idempotently(result, "dataset.yaml")

        assert [run.run_id for run in storage.list_runs()] == ["finalizing-run"]
        assert [
            event.sequence for event in storage.get_trace_events("finalizing-run")
        ] == [1, 2, 3, 4, 5, 6]
        assert len(storage.get_evaluator_outcomes("finalizing-run")) == 1
        with pytest.raises(StorageError, match="conflicts"):
            storage.save_run_idempotently(result, "different-dataset.yaml")


def test_storage_rejects_contradictory_result_and_trace_invariants() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        with pytest.raises(ValueError, match="no error"):
            storage.save_run(
                make_result("pass-with-error", error="contradiction"),
                "dataset.yaml",
            )

        mismatched_end = make_result("mismatched-end")
        mismatched_trace = tuple(
            replace(event, data={**event.data, "passed": False})
            if event.event_type == "run_end"
            else event
            for event in mismatched_end.trace
        )
        with pytest.raises(ValueError, match="run_end passed"):
            storage.save_run(
                replace(mismatched_end, trace=mismatched_trace),
                "dataset.yaml",
            )

        mismatched_tests = make_result("mismatched-tests")
        mismatched_test_trace = tuple(
            replace(
                event,
                data={**event.data, "tests_after_passed": False},
            )
            if event.event_type == "run_end"
            else event
            for event in mismatched_tests.trace
        )
        with pytest.raises(ValueError, match="tests_after_passed"):
            storage.save_run(
                replace(mismatched_tests, trace=mismatched_test_trace),
                "dataset.yaml",
            )

        failed_contract = make_result("failed-contract")
        failed_contract_trace = tuple(
            replace(
                event,
                data={**event.data, "workspace_changes_passed": False},
            )
            if event.event_type == "run_end"
            else event
            for event in failed_contract.trace
        )
        with pytest.raises(ValueError, match="workspace-change contract"):
            storage.save_run(
                replace(failed_contract, trace=failed_contract_trace),
                "dataset.yaml",
            )

        noncontiguous = make_result("noncontiguous")
        broken_trace = tuple(
            replace(event, sequence=4) if event.sequence == 3 else event
            for event in noncontiguous.trace
        )
        with pytest.raises(ValueError, match="unique and contiguous"):
            storage.save_run(
                replace(noncontiguous, trace=broken_trace),
                "dataset.yaml",
            )

        nonfinite = make_result("nonfinite", latency=float("nan"))
        with pytest.raises(ValueError, match="finite non-negative"):
            storage.save_run(nonfinite, "dataset.yaml")


def test_secret_is_never_written_to_sqlite(monkeypatch) -> None:
    secret = "storage-secret-value-987654"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        result = make_result(
            "secret-run",
            case_id=f"case-{secret}",
            passed=False,
            error=f"password={secret}",
            extra_data={
                "stdout": f"Authorization: Bearer {secret}",
                "api_key": secret,
            },
        )

        storage.save_run(result, f"dataset-{secret}.yaml")
        stored = storage.get_run("secret-run")
        events = storage.get_trace_events("secret-run")

        assert stored is not None
        assert secret not in repr(stored)
        assert secret not in repr(events)
        assert "[REDACTED]" in repr(stored)
        assert "[REDACTED]" in repr(events)
        assert secret.encode() not in database.read_bytes()


def test_cli_trace_loads_persisted_run(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        storage.save_run(make_result("persisted-run"), "dataset.yaml")
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))
        monkeypatch.setattr(
            "agentlab.cli.evaluate_case",
            lambda _case: pytest.fail("trace command must not evaluate a case"),
        )

        result = CliRunner().invoke(app, ["trace", "persisted-run"])

        assert result.exit_code == 0
        assert "persisted-run" in result.stdout
        assert "case-001" in result.stdout
        assert "run_start" in result.stdout
        assert "run_end" in result.stdout
        assert "PASS" in result.stdout


def test_cli_eval_persists_completed_run(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        completed = make_result("eval-persisted-run")
        case = EvalCase("case-001", "repository", "fix it")
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))
        monkeypatch.setattr("agentlab.cli.load_dataset", lambda _dataset: [case])
        monkeypatch.setattr(
            "agentlab.cli.evaluate_case", lambda _case, **_kwargs: completed
        )

        result = CliRunner().invoke(
            app, ["eval", "dataset.yaml", "--agent", "mock_agent"]
        )
        stored = SQLiteStorage(database).get_run("eval-persisted-run")

        assert result.exit_code == 0
        assert "eval-persisted-run" in result.stdout
        assert stored is not None
        assert stored.dataset == "dataset.yaml"


def test_cli_runs_lists_persisted_history(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        storage.save_run(make_result("history-run"), "dataset.yaml")
        monkeypatch.setenv("AGENTLAB_DB_PATH", str(database))

        result = CliRunner().invoke(app, ["runs"])

        assert result.exit_code == 0
        assert "history-run" in result.stdout
        assert "case-001" in result.stdout
        assert "PASS" in result.stdout

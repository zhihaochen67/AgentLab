import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentlab.cli import app
from agentlab.models import EvalCase, EvalResult
from agentlab.storage import SQLiteStorage, StorageError
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
            "agent_end",
            finished_at,
            {"status": "ok", **(extra_data or {})},
        ),
        TraceEvent(
            run_id,
            3,
            "run_end",
            finished_at,
            {"passed": passed, "elapsed_time": latency},
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

        assert [event.sequence for event in events] == [1, 2, 3]
        assert [event.event_type for event in events] == [
            "run_start",
            "agent_end",
            "run_end",
        ]


def test_missing_run_id_returns_no_data() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        assert storage.get_run("missing") is None
        assert storage.get_trace_events("missing") == ()


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


def test_secret_is_never_written_to_sqlite(monkeypatch) -> None:
    secret = "storage-secret-value-987654"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    with tempfile.TemporaryDirectory(prefix="agentlab-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        result = make_result(
            "secret-run",
            case_id=f"case-{secret}",
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

        result = CliRunner().invoke(app, ["eval", "dataset.yaml"])
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

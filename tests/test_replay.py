import tempfile
from pathlib import Path

from agentlab.adapters.repo_doctor import RepoDoctorAdapter
from agentlab.models import EvalResult
from agentlab.replay import ReplayState, format_replay_event, prepare_replay
from agentlab.storage import SQLiteStorage
from agentlab.tracer import TraceEvent


def event(
    sequence: int,
    event_type: str = "run_start",
    data: dict | None = None,
) -> TraceEvent:
    return TraceEvent(
        run_id="historical-run",
        sequence=sequence,
        event_type=event_type,
        timestamp=f"2026-08-18T01:00:0{sequence}+00:00",
        data=data or {},
    )


def test_replay_orders_events_by_sequence() -> None:
    replay = prepare_replay((event(3, "run_end"), event(1), event(2, "agent_end")))

    assert [item.sequence for item in replay.events] == [1, 2, 3]
    assert [item.event_type for item in replay.events] == [
        "run_start",
        "agent_end",
        "run_end",
    ]
    assert replay.warnings == ()


def test_replay_navigation_clamps_first_previous_next_and_last() -> None:
    state = ReplayState.from_events((event(1), event(2), event(3)))

    assert state.current_step == 1
    assert state.first().current_step == 1
    assert state.previous().current_step == 1
    assert state.next().current_step == 2
    assert state.next().next().current_step == 3
    assert state.next().next().next().current_step == 3
    assert state.last().current_step == 3
    assert state.last().first().current_step == 1


def test_empty_trace_has_safe_navigation() -> None:
    state = ReplayState.from_events(())

    assert state.total_steps == 0
    assert state.current_step == 0
    assert state.current_event is None
    assert state.is_first is True
    assert state.is_last is True
    assert state.first().previous().next().last().current_event is None


def test_invalid_duplicate_and_missing_sequences_are_non_fatal() -> None:
    invalid = event(2)
    object.__setattr__(invalid, "sequence", "not-a-number")
    replay = prepare_replay(
        (
            event(3, "run_end"),
            event(1),
            event(1, "duplicate"),
            event(0, "invalid"),
            invalid,
        )
    )

    assert [item.sequence for item in replay.events] == [1, 3]
    assert any("duplicate sequence 1" in warning for warning in replay.warnings)
    assert any("invalid sequence 0" in warning for warning in replay.warnings)
    assert any("invalid sequence 'not-a-number'" in warning for warning in replay.warnings)
    assert any("Missing sequence 2" in warning for warning in replay.warnings)


def test_event_formatting_is_specific_to_historical_event_type() -> None:
    pytest_view = format_replay_event(
        event(
            1,
            "pytest_before_end",
            {
                "passed": False,
                "returncode": 1,
                "stdout": "one failed",
                "stderr": "warning",
                "elapsed_time": 0.25,
            },
        )
    )
    agent_view = format_replay_event(
        event(
            2,
            "agent_end",
            {
                "adapter": "FakeAdapter",
                "status": "ok",
                "returncode": 0,
                "stdout": "fixed",
                "elapsed_time": 1.5,
            },
        )
    )
    error_view = format_replay_event(
        event(3, "error", {"error_type": "RuntimeError", "message": "failed"})
    )
    end_view = format_replay_event(
        event(4, "run_end", {"passed": True, "elapsed_time": 2.0})
    )

    assert pytest_view.status == "FAIL"
    assert pytest_view.elapsed == 0.25
    assert pytest_view.highlights["returncode"] == 1
    assert pytest_view.highlights["stdout"] == "one failed"
    assert agent_view.status == "OK"
    assert agent_view.elapsed == 1.5
    assert agent_view.highlights["adapter"] == "FakeAdapter"
    assert error_view.status == "ERROR"
    assert error_view.highlights["error_type"] == "RuntimeError"
    assert end_view.status == "PASS"
    assert end_view.highlights["elapsed_time"] == 2.0


def test_replay_redacts_secrets_at_display_boundary(monkeypatch) -> None:
    secret = "replay-secret-value-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    view = format_replay_event(
        event(
            1,
            "error",
            {
                "message": f"Authorization: Bearer {secret}",
                "api_key": secret,
                "nested": {"raw": secret},
            },
        )
    )

    assert secret not in repr(view)
    assert "[REDACTED]" in repr(view)


def test_replay_never_calls_agent_adapter(monkeypatch) -> None:
    called = False

    def forbidden_repair(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("historical replay must not call an agent")

    monkeypatch.setattr(RepoDoctorAdapter, "repair", forbidden_repair)

    state = ReplayState.from_events((event(1), event(2, "agent_end")))
    format_replay_event(state.last().current_event)

    assert called is False


def test_replay_reads_sqlite_without_writing() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-replay-") as directory:
        database = Path(directory) / "replay.db"
        writable = SQLiteStorage(database)
        trace = (
            event(1, "run_start", {"adapter": "FakeAdapter"}),
            event(2, "run_end", {"passed": True, "elapsed_time": 1.0}),
        )
        writable.save_run(
            EvalResult(
                case_id="case-001",
                passed=True,
                tests_before_passed=False,
                tests_after_passed=True,
                run_id="historical-run",
                trace=trace,
            ),
            "dataset.yaml",
        )
        before = database.read_bytes()

        read_only = SQLiteStorage(database, read_only=True)
        state = ReplayState.from_events(read_only.get_trace_events("historical-run"))
        format_replay_event(state.last().current_event)

        assert state.current_event is not None
        assert state.last().current_event.event_type == "run_end"
        assert database.read_bytes() == before

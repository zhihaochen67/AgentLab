import sqlite3
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from agentlab.models import EvalResult, Experiment
from agentlab.storage import SQLiteStorage, StorageError
from agentlab.tracer import TraceEvent

_STARTED_AT = "2026-08-18T01:00:00+00:00"
_FINISHED_AT = "2026-08-18T01:00:06+00:00"

_LEGACY_SCHEMA = """
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    dataset TEXT NOT NULL,
    adapter TEXT NOT NULL,
    model TEXT,
    trials_per_case INTEGER NOT NULL CHECK (trials_per_case > 0),
    total_cases INTEGER NOT NULL CHECK (total_cases >= 0),
    total_runs INTEGER NOT NULL DEFAULT 0 CHECK (total_runs >= 0),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'completed_with_failures', 'aborted')
    )
);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    adapter TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PASS', 'FAIL')),
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    total_latency REAL NOT NULL,
    tests_before_passed INTEGER NOT NULL,
    tests_after_passed INTEGER NOT NULL,
    error TEXT
);
CREATE TABLE trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    data_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, sequence)
);
"""


def evaluator_end_data(
    *,
    evaluator: str = "LLMJudgeEvaluator",
    status: str = "pass",
    passed: bool = True,
    score: float | None = 0.75,
    feedback: str | None = "judge feedback",
    metadata: dict | None = None,
    error_type: str | None = None,
    elapsed_time: float | None = 0.12,
) -> dict:
    data: dict = {"evaluator": evaluator, "status": status, "passed": passed}
    if score is not None:
        data["score"] = score
    if feedback is not None:
        data["feedback"] = feedback
    if metadata is not None:
        data["metadata"] = metadata
    if error_type is not None:
        data["error_type"] = error_type
    if elapsed_time is not None:
        data["elapsed_time"] = elapsed_time
    return data


def make_result(
    run_id: str,
    *,
    passed: bool = True,
    evaluator_events: list[tuple[int, dict]] | None = None,
) -> EvalResult:
    events = [
        TraceEvent(
            run_id,
            1,
            "run_start",
            _STARTED_AT,
            {"case_id": "case-001", "adapter": "FakeAdapter"},
        ),
        TraceEvent(run_id, 2, "pytest_before_end", _FINISHED_AT, {"passed": False}),
        TraceEvent(run_id, 3, "agent_end", _FINISHED_AT, {"status": "ok"}),
        TraceEvent(run_id, 4, "pytest_after_end", _FINISHED_AT, {"passed": passed}),
    ]
    for sequence, data in evaluator_events or []:
        events.append(TraceEvent(run_id, sequence, "evaluator_end", _FINISHED_AT, data))
    events.append(
        TraceEvent(
            run_id,
            len(events) + 1,
            "run_end",
            _FINISHED_AT,
            {"passed": passed, "elapsed_time": 1.25},
        )
    )
    return EvalResult(
        case_id="case-001",
        passed=passed,
        tests_before_passed=False,
        tests_after_passed=passed,
        error=None,
        run_id=run_id,
        trace=tuple(events),
    )


def create_legacy_database(path: Path, *, with_run: bool = False) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_LEGACY_SCHEMA)
    if with_run:
        connection.execute(
            """
            INSERT INTO runs (
                run_id, case_id, dataset, adapter, status,
                started_at, finished_at, total_latency,
                tests_before_passed, tests_after_passed, error
            ) VALUES (
                'legacy-run', 'case-legacy', 'dataset.yaml', 'FakeAdapter',
                'PASS', ?, ?, 1.0, 0, 1, NULL
            )
            """,
            (_STARTED_AT, _FINISHED_AT),
        )
        connection.execute(
            """
            INSERT INTO trace_events (run_id, sequence, event_type, timestamp, data_json)
            VALUES ('legacy-run', 1, 'run_start', ?, '{}')
            """,
            (_STARTED_AT,),
        )
    connection.commit()
    connection.close()


def test_evaluator_pass_round_trip() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "pass-run",
                evaluator_events=[
                    (
                        5,
                        evaluator_end_data(
                            score=0.8,
                            feedback="semantics ok",
                            metadata={"rubric": "semantic"},
                            elapsed_time=0.34,
                        ),
                    )
                ],
            ),
            "dataset.yaml",
        )

        outcomes = storage.get_evaluator_outcomes("pass-run")

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.run_id == "pass-run"
        assert outcome.sequence == 5
        assert outcome.evaluator == "LLMJudgeEvaluator"
        assert outcome.status == "PASS"
        assert outcome.passed is True
        assert outcome.score == 0.8
        assert outcome.feedback == "semantics ok"
        assert outcome.metadata == {"rubric": "semantic"}
        assert outcome.error_type is None
        assert outcome.elapsed_time == 0.34


def test_evaluator_fail_round_trip() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "fail-run",
                passed=False,
                evaluator_events=[
                    (5, evaluator_end_data(status="fail", passed=False, score=0.1))
                ],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("fail-run")[0]

        assert outcome.status == "FAIL"
        assert outcome.passed is False
        assert outcome.score == 0.1


def test_evaluator_error_round_trip() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "error-run",
                passed=False,
                evaluator_events=[
                    (
                        5,
                        evaluator_end_data(
                            status="error",
                            passed=False,
                            score=None,
                            feedback=None,
                            error_type="JudgeResponseError",
                        ),
                    )
                ],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("error-run")[0]

        assert outcome.status == "ERROR"
        assert outcome.passed is False
        assert outcome.score is None
        assert outcome.feedback is None
        assert outcome.error_type == "JudgeResponseError"


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_score_bounds_round_trip(score: float) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "score-run",
                evaluator_events=[(5, evaluator_end_data(score=score))],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("score-run")[0]

        assert outcome.score == score
        assert outcome.score is not None


def test_score_none_round_trip() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "no-score-run",
                evaluator_events=[(5, evaluator_end_data(score=None))],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("no-score-run")[0]

        assert outcome.score is None


def test_feedback_round_trip() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "feedback-run",
                evaluator_events=[
                    (5, evaluator_end_data(feedback="Long multiline justification."))
                ],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("feedback-run")[0]

        assert outcome.feedback == "Long multiline justification."


def test_metadata_json_round_trip() -> None:
    metadata = {
        "rubric": {"strictness": 0.5},
        "tags": ["semantic", "style"],
        "judge": "mock-judge",
    }
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "metadata-run",
                evaluator_events=[(5, evaluator_end_data(metadata=metadata))],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("metadata-run")[0]

        assert isinstance(outcome.metadata, dict)
        assert outcome.metadata == metadata


def test_missing_metadata_defaults_to_empty_object() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "bare-run",
                evaluator_events=[(5, evaluator_end_data(metadata=None))],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("bare-run")[0]

        assert outcome.metadata == {}


def test_sensitive_metadata_key_is_redacted(monkeypatch) -> None:
    secret = "evaluator-metadata-secret-321"
    monkeypatch.setenv("JUDGE_API_KEY", secret)
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        storage.save_run(
            make_result(
                "metadata-secret-run",
                evaluator_events=[
                    (
                        5,
                        evaluator_end_data(
                            metadata={"api_key": secret, "note": f"key={secret}"}
                        ),
                    )
                ],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("metadata-secret-run")[0]

        assert secret not in repr(outcome)
        assert outcome.metadata["api_key"] == "[REDACTED]"
        assert "[REDACTED]" in outcome.metadata["note"]
        assert secret.encode() not in database.read_bytes()


def test_feedback_secret_is_redacted(monkeypatch) -> None:
    secret = "evaluator-feedback-secret-654"
    monkeypatch.setenv("JUDGE_API_KEY", secret)
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        storage.save_run(
            make_result(
                "feedback-secret-run",
                evaluator_events=[
                    (
                        5,
                        evaluator_end_data(
                            feedback=(
                                f"Authorization: Bearer {secret}; "
                                f"api_key={secret}"
                            )
                        ),
                    )
                ],
            ),
            "dataset.yaml",
        )

        outcome = storage.get_evaluator_outcomes("feedback-secret-run")[0]

        assert secret not in outcome.feedback
        assert "[REDACTED]" in outcome.feedback
        assert secret.encode() not in database.read_bytes()


def test_no_evaluator_returns_empty_outcomes() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(make_result("plain-run"), "dataset.yaml")

        assert storage.get_evaluator_outcomes("plain-run") == ()


def test_missing_run_returns_empty_outcomes() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")

        assert storage.get_evaluator_outcomes("missing-run") == ()


def test_multiple_evaluator_end_events_in_sequence_order() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.save_run(
            make_result(
                "multi-eval-run",
                evaluator_events=[
                    (6, evaluator_end_data(evaluator="SecondEvaluator", score=0.9)),
                    (5, evaluator_end_data(evaluator="FirstEvaluator", score=0.3)),
                ],
            ),
            "dataset.yaml",
        )

        outcomes = storage.get_evaluator_outcomes("multi-eval-run")

        assert [outcome.sequence for outcome in outcomes] == [5, 6]
        assert [outcome.evaluator for outcome in outcomes] == [
            "FirstEvaluator",
            "SecondEvaluator",
        ]


def test_invalid_status_rejected_without_partial_state() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "bad-status-run",
            evaluator_events=[
                (
                    5,
                    evaluator_end_data() | {"status": "pending"},
                )
            ],
        )

        with pytest.raises(ValueError, match="invalid status"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("bad-status-run") is None
        assert storage.get_trace_events("bad-status-run") == ()
        assert storage.get_evaluator_outcomes("bad-status-run") == ()


def test_missing_evaluator_name_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "no-name-run",
            evaluator_events=[(5, evaluator_end_data(evaluator="   "))],
        )

        with pytest.raises(ValueError, match="evaluator"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("no-name-run") is None


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_out_of_range_score_rejected(score: float) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "bad-score-run",
            evaluator_events=[(5, evaluator_end_data(score=score))],
        )

        with pytest.raises(ValueError, match="score"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("bad-score-run") is None


def test_bool_score_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "bool-score-run",
            evaluator_events=[(5, evaluator_end_data(score=True))],
        )

        with pytest.raises(ValueError, match="score"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("bool-score-run") is None


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_non_finite_score_rejected(score: float) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "non-finite-score-run",
            evaluator_events=[(5, evaluator_end_data(score=score))],
        )

        with pytest.raises(ValueError, match="score"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("non-finite-score-run") is None


def test_non_bool_passed_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "bad-passed-run",
            evaluator_events=[(5, evaluator_end_data(passed="yes"))],
        )

        with pytest.raises(ValueError, match="passed"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("bad-passed-run") is None


def test_invalid_metadata_type_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "bad-metadata-run",
            evaluator_events=[(5, evaluator_end_data(metadata=[1, 2, 3]))],
        )

        with pytest.raises(ValueError, match="metadata"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("bad-metadata-run") is None


@pytest.mark.parametrize("elapsed_time", [-0.5, float("nan"), "fast"])
def test_invalid_elapsed_time_rejected(elapsed_time) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        result = make_result(
            "bad-elapsed-run",
            evaluator_events=[(5, evaluator_end_data(elapsed_time=elapsed_time))],
        )

        with pytest.raises(ValueError, match="elapsed_time"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("bad-elapsed-run") is None


def test_evaluator_persistence_failure_rolls_back_entire_save() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        storage = SQLiteStorage(database)
        storage.create_experiment(
            Experiment(
                experiment_id="exp-rollback",
                label="rollback",
                dataset="dataset.yaml",
                adapter="FakeAdapter",
                model=None,
                trials_per_case=1,
                total_cases=1,
                total_runs=0,
                started_at=_STARTED_AT,
                finished_at=None,
                status="running",
            )
        )

        connection = sqlite3.connect(database)
        connection.execute(
            """
            CREATE TRIGGER abort_evaluator_insert
            BEFORE INSERT ON evaluator_outcomes
            BEGIN
                SELECT RAISE(ABORT, 'simulated evaluator persistence failure');
            END
            """
        )
        connection.commit()
        connection.close()

        result = replace(
            make_result(
                "rollback-run",
                evaluator_events=[(5, evaluator_end_data())],
            ),
            experiment_id="exp-rollback",
            trial_index=1,
        )

        with pytest.raises(StorageError, match="simulated evaluator persistence failure"):
            storage.save_run(result, "dataset.yaml")

        assert storage.get_run("rollback-run") is None
        assert storage.get_trace_events("rollback-run") == ()
        assert storage.get_evaluator_outcomes("rollback-run") == ()
        experiment = storage.get_experiment("exp-rollback")
        assert experiment is not None
        assert experiment.total_runs == 0

        connection = sqlite3.connect(database)
        connection.execute("DROP TRIGGER abort_evaluator_insert")
        connection.commit()
        connection.close()

        storage.save_run(result, "dataset.yaml")

        assert storage.get_run("rollback-run") is not None
        assert len(storage.get_evaluator_outcomes("rollback-run")) == 1


def test_writable_legacy_database_gains_evaluator_outcomes() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        create_legacy_database(database, with_run=True)

        storage = SQLiteStorage(database)

        legacy = storage.get_run("legacy-run")
        assert legacy is not None
        assert legacy.case_id == "case-legacy"
        assert storage.get_evaluator_outcomes("legacy-run") == ()

        storage.save_run(
            make_result(
                "new-run",
                evaluator_events=[(5, evaluator_end_data(score=0.6))],
            ),
            "dataset.yaml",
        )

        outcomes = storage.get_evaluator_outcomes("new-run")
        assert len(outcomes) == 1
        assert outcomes[0].score == 0.6


def test_read_only_legacy_database_returns_empty_outcomes() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-storage-") as directory:
        database = Path(directory) / "agentlab.db"
        create_legacy_database(database, with_run=True)
        before = database.read_bytes()

        read_only = SQLiteStorage(database, read_only=True)

        assert read_only.get_evaluator_outcomes("legacy-run") == ()
        assert read_only.get_run("legacy-run") is not None
        assert read_only.get_trace_events("legacy-run")
        assert database.read_bytes() == before

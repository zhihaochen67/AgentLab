"""Persistent storage abstractions for completed evaluation runs."""

from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentlab.diagnostics import diagnostics_from_trace_data
from agentlab.models import (
    CaseExperimentMetrics,
    EvalResult,
    EvaluatorExperimentMetrics,
    Experiment,
    ExperimentMetrics,
)
from agentlab.tracer import TraceEvent, sanitize_data, summarize_text

DEFAULT_DATABASE_PATH = Path(".agentlab") / "agentlab.db"
DATABASE_PATH_ENV = "AGENTLAB_DB_PATH"


@dataclass(frozen=True)
class StoredRun:
    """Metadata for one completed evaluation run."""

    run_id: str
    case_id: str
    dataset: str
    adapter: str
    status: str
    started_at: str
    finished_at: str
    total_latency: float
    tests_before_passed: bool
    tests_after_passed: bool
    error: str | None
    experiment_id: str | None = None
    trial_index: int | None = None


@dataclass(frozen=True)
class StoredEvaluatorOutcome:
    """Immutable persisted outcome of one evaluator execution."""

    run_id: str
    sequence: int
    evaluator: str
    status: str
    passed: bool
    score: float | None
    feedback: str | None
    metadata: dict[str, Any]
    error_type: str | None
    elapsed_time: float | None


@dataclass(frozen=True)
class RunStats:
    """Aggregate read-only statistics for the dashboard overview."""

    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    average_latency: float


def _evaluator_experiment_metrics(
    rows: list[dict[str, Any]],
    total_runs: int,
) -> tuple[EvaluatorExperimentMetrics, ...]:
    """Convert SQL-aggregated evaluator_outcomes rows into metrics.

    Only derived ratios are computed here; every count/statistic comes from
    SQL aggregation over the evaluator_outcomes table. Scores stay on the
    raw 0.0~1.0 scale and ERROR outcomes never enter the pass-rate verdict
    denominator.
    """
    metrics: list[EvaluatorExperimentMetrics] = []
    for row in rows:
        passed = int(row["passed_outcomes"])
        failed = int(row["failed_outcomes"])
        verdict = passed + failed
        evaluated_runs = int(row["evaluated_runs"])
        metrics.append(
            EvaluatorExperimentMetrics(
                evaluator=row["evaluator"],
                total_outcomes=int(row["total_outcomes"]),
                evaluated_runs=evaluated_runs,
                passed_outcomes=passed,
                failed_outcomes=failed,
                error_outcomes=int(row["error_outcomes"]),
                verdict_outcomes=verdict,
                pass_rate=(passed / verdict * 100.0) if verdict else 0.0,
                score_count=int(row["score_count"]),
                average_score=row["average_score"],
                min_score=row["min_score"],
                max_score=row["max_score"],
                coverage_rate=(
                    (evaluated_runs / total_runs * 100.0) if total_runs else 0.0
                ),
            )
        )
    return tuple(metrics)


class StorageError(RuntimeError):
    """A persistent storage operation could not be completed."""


class RunStorage(ABC):
    """Storage contract for completed runs and their trace events."""

    @abstractmethod
    def save_run(self, result: EvalResult, dataset: str) -> None:
        """Atomically persist one evaluation result and all trace events."""

    @abstractmethod
    def get_run(self, run_id: str) -> StoredRun | None:
        """Load run metadata, or return None when it does not exist."""

    @abstractmethod
    def get_trace_events(self, run_id: str) -> tuple[TraceEvent, ...]:
        """Load trace events in stable sequence order."""

    @abstractmethod
    def get_evaluator_outcomes(self, run_id: str) -> tuple[StoredEvaluatorOutcome, ...]:
        """Load persisted evaluator outcomes in stable sequence order."""

    @abstractmethod
    def list_runs(
        self,
        limit: int = 20,
        *,
        status: str | None = None,
        case_id: str | None = None,
        experiment_id: str | None = None,
    ) -> tuple[StoredRun, ...]:
        """Load recent runs with optional exact status and case filters."""

    @abstractmethod
    def list_case_ids(self) -> tuple[str, ...]:
        """Load distinct case ids available for filtering."""

    @abstractmethod
    def get_stats(self) -> RunStats:
        """Load aggregate run statistics."""

    @abstractmethod
    def create_experiment(self, experiment: Experiment) -> None:
        """Persist experiment metadata before any associated run."""

    @abstractmethod
    def finish_experiment(
        self,
        experiment_id: str,
        status: str,
        finished_at: str,
    ) -> Experiment:
        """Set a final experiment status and completion timestamp."""

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Load one experiment, or return None when it is absent."""

    @abstractmethod
    def list_experiments(self, limit: int = 20) -> tuple[Experiment, ...]:
        """Load recent experiments in stable reverse chronological order."""

    @abstractmethod
    def get_experiment_metrics(self, experiment_id: str) -> ExperimentMetrics:
        """Aggregate persisted runs belonging to one experiment."""


def default_database_path() -> Path:
    """Return the configured database path, defaulting to the current project."""
    configured = os.environ.get(DATABASE_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / DEFAULT_DATABASE_PATH


class SQLiteStorage(RunStorage):
    """SQLite-backed run storage with transactional writes."""

    def __init__(self, database_path: str | Path, *, read_only: bool = False) -> None:
        self.database_path = Path(database_path)
        self.read_only = read_only
        try:
            if read_only:
                if not self.database_path.is_file():
                    raise StorageError(f"AgentLab database does not exist: {self.database_path}")
            else:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                self._initialize_schema()
        except (OSError, sqlite3.Error) as error:
            raise StorageError(f"Could not initialize AgentLab database: {error}") from error

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = self.database_path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    adapter TEXT NOT NULL,
                    model TEXT,
                    agent_version TEXT,
                    prompt_variant TEXT,
                    notes TEXT,
                    trials_per_case INTEGER NOT NULL CHECK (trials_per_case > 0),
                    total_cases INTEGER NOT NULL CHECK (total_cases >= 0),
                    total_runs INTEGER NOT NULL DEFAULT 0 CHECK (total_runs >= 0),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'running', 'completed',
                            'completed_with_failures', 'aborted'
                        )
                    )
                );

                CREATE TABLE IF NOT EXISTS runs (
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
                    error TEXT,
                    experiment_id TEXT,
                    trial_index INTEGER,
                    FOREIGN KEY (experiment_id)
                        REFERENCES experiments(experiment_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    UNIQUE (run_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_trace_events_run_sequence
                    ON trace_events(run_id, sequence);

                CREATE TABLE IF NOT EXISTS evaluator_outcomes (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    evaluator TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('PASS', 'FAIL', 'ERROR')),
                    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
                    score REAL CHECK (score IS NULL OR (score >= 0.0 AND score <= 1.0)),
                    feedback TEXT,
                    metadata_json TEXT NOT NULL,
                    error_type TEXT,
                    elapsed_time REAL CHECK (
                        elapsed_time IS NULL OR elapsed_time >= 0.0
                    ),
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_evaluator_outcomes_run_sequence
                    ON evaluator_outcomes(run_id, sequence);
                """
            )
            self._migrate_experiments_schema(connection)
            self._migrate_runs_schema(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_experiment_case_trial
                    ON runs(experiment_id, case_id, trial_index)
                """
            )

    @staticmethod
    def _migrate_experiments_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
        }
        for column in ("agent_version", "prompt_variant", "notes"):
            if column not in columns:
                connection.execute(f"ALTER TABLE experiments ADD COLUMN {column} TEXT")

    @staticmethod
    def _migrate_runs_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "experiment_id" not in columns:
            connection.execute(
                """
                ALTER TABLE runs ADD COLUMN experiment_id TEXT
                    REFERENCES experiments(experiment_id) ON DELETE SET NULL
                """
            )
        if "trial_index" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN trial_index INTEGER")

    @staticmethod
    def _required_event(result: EvalResult, event_type: str) -> TraceEvent:
        event = next((item for item in result.trace if item.event_type == event_type), None)
        if event is None:
            raise ValueError(f"Evaluation trace is missing required event: {event_type}")
        return event

    _EVALUATOR_STATUSES = frozenset({"pass", "fail", "error"})

    @classmethod
    def _evaluator_outcome_rows(cls, result: EvalResult) -> list[tuple[Any, ...]]:
        """Extract validated evaluator_outcomes rows from evaluator_end events."""
        rows: list[tuple[Any, ...]] = []
        for event in sorted(result.trace, key=lambda item: item.sequence):
            if event.event_type != "evaluator_end":
                continue
            rows.append(cls._validated_evaluator_outcome_row(result.run_id, event))
        return rows

    @classmethod
    def _validated_evaluator_outcome_row(
        cls,
        run_id: str,
        event: TraceEvent,
    ) -> tuple[Any, ...]:
        """Validate one structured evaluator_end event into a persisted row."""
        context = f"evaluator_end sequence {event.sequence} for run {run_id}"
        data = event.data

        evaluator = data.get("evaluator")
        if not isinstance(evaluator, str) or not evaluator.strip():
            raise ValueError(f"{context} is missing a non-empty 'evaluator'.")

        raw_status = data.get("status")
        status = raw_status.lower() if isinstance(raw_status, str) else None
        if status not in cls._EVALUATOR_STATUSES:
            raise ValueError(
                f"{context} has invalid status: {raw_status!r} "
                "(expected 'pass', 'fail', or 'error')."
            )

        passed = data.get("passed")
        if not isinstance(passed, bool):
            raise ValueError(  # noqa: TRY004 - persisted evaluator data validation.
                f"{context} is missing a boolean 'passed'."
            )
        if (status == "pass") != passed:
            raise ValueError(
                f"{context} has inconsistent status/passed: "
                f"status={status!r} passed={passed!r}."
            )

        score = data.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{context} has a non-numeric 'score': {score!r}.")
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(
                    f"{context} score must be within [0.0, 1.0], got {score!r}."
                )
            score = float(score)

        feedback = data.get("feedback")
        if feedback is not None and not isinstance(feedback, str):
            raise ValueError(f"{context} 'feedback' must be a string when present.")

        metadata = data.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(  # noqa: TRY004 - persisted evaluator data validation.
                f"{context} 'metadata' must be an object, got "
                f"{type(metadata).__name__}."
            )

        error_type = data.get("error_type")
        if error_type is not None and not isinstance(error_type, str):
            raise ValueError(f"{context} 'error_type' must be a string when present.")

        elapsed_time = data.get("elapsed_time")
        if elapsed_time is not None:
            if isinstance(elapsed_time, bool) or not isinstance(
                elapsed_time, (int, float)
            ):
                raise ValueError(
                    f"{context} 'elapsed_time' must be numeric when present."
                )
            if elapsed_time < 0:
                raise ValueError(
                    f"{context} 'elapsed_time' must be non-negative, got {elapsed_time!r}."
                )
            elapsed_time = float(elapsed_time)

        sanitized_metadata = sanitize_data(metadata)
        try:
            metadata_json = json.dumps(
                sanitized_metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context} metadata is not JSON-serializable: {error}") from error

        return (
            run_id,
            event.sequence,
            summarize_text(evaluator),
            status.upper(),
            int(passed),
            score,
            summarize_text(feedback) if feedback is not None else None,
            metadata_json,
            summarize_text(error_type) if error_type is not None else None,
            elapsed_time,
        )

    def save_run(self, result: EvalResult, dataset: str) -> None:
        if self.read_only:
            raise StorageError("Cannot save an evaluation run through read-only storage.")
        start = self._required_event(result, "run_start")
        end = self._required_event(result, "run_end")
        mismatched = [event.sequence for event in result.trace if event.run_id != result.run_id]
        if mismatched:
            raise ValueError("Every trace event must match EvalResult.run_id.")
        if (result.experiment_id is None) != (result.trial_index is None):
            raise ValueError("experiment_id and trial_index must be set together.")
        if result.trial_index is not None and result.trial_index < 1:
            raise ValueError("trial_index must be at least 1.")

        adapter = str(start.data.get("adapter", "unknown"))
        total_latency = end.data.get("elapsed_time", 0.0)
        if not isinstance(total_latency, (int, float)):
            raise TypeError("run_end elapsed_time must be numeric.")

        run_values = (
            result.run_id,
            summarize_text(result.case_id),
            summarize_text(dataset),
            summarize_text(adapter),
            "PASS" if result.passed else "FAIL",
            summarize_text(start.timestamp),
            summarize_text(end.timestamp),
            float(total_latency),
            int(result.tests_before_passed),
            int(result.tests_after_passed),
            summarize_text(result.error) if result.error is not None else None,
            result.experiment_id,
            result.trial_index,
        )
        event_values = [
            (
                result.run_id,
                event.sequence,
                summarize_text(event.event_type),
                summarize_text(event.timestamp),
                json.dumps(
                    sanitize_data(event.data),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for event in sorted(result.trace, key=lambda item: item.sequence)
        ]
        evaluator_values = self._evaluator_outcome_rows(result)

        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, case_id, dataset, adapter, status,
                        started_at, finished_at, total_latency,
                        tests_before_passed, tests_after_passed, error,
                        experiment_id, trial_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    run_values,
                )
                connection.executemany(
                    """
                    INSERT INTO trace_events (
                        run_id, sequence, event_type, timestamp, data_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    event_values,
                )
                if evaluator_values:
                    connection.executemany(
                        """
                        INSERT INTO evaluator_outcomes (
                            run_id, sequence, evaluator, status, passed, score,
                            feedback, metadata_json, error_type, elapsed_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        evaluator_values,
                    )
                if result.experiment_id is not None:
                    updated = connection.execute(
                        """
                        UPDATE experiments
                        SET total_runs = total_runs + 1
                        WHERE experiment_id = ?
                        """,
                        (result.experiment_id,),
                    )
                    if updated.rowcount != 1:
                        raise sqlite3.IntegrityError(
                            f"Unknown experiment: {result.experiment_id}"
                        )
        except sqlite3.Error as error:
            raise StorageError(f"Could not save evaluation run {result.run_id}: {error}") from error

    def get_run(self, run_id: str) -> StoredRun | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise StorageError(f"Could not load evaluation run {run_id}: {error}") from error
        return self._stored_run(row) if row is not None else None

    def get_trace_events(self, run_id: str) -> tuple[TraceEvent, ...]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT run_id, sequence, event_type, timestamp, data_json
                    FROM trace_events
                    WHERE run_id = ?
                    ORDER BY run_id, sequence
                    """,
                    (run_id,),
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"Could not load trace for run {run_id}: {error}") from error
        try:
            return tuple(
                TraceEvent(
                    run_id=row["run_id"],
                    sequence=row["sequence"],
                    event_type=row["event_type"],
                    timestamp=row["timestamp"],
                    data=json.loads(row["data_json"]),
                )
                for row in rows
            )
        except (TypeError, json.JSONDecodeError) as error:
            raise StorageError(f"Stored trace for run {run_id} is invalid: {error}") from error

    def get_evaluator_outcomes(self, run_id: str) -> tuple[StoredEvaluatorOutcome, ...]:
        try:
            with self._connection() as connection:
                if not self._table_exists(connection, "evaluator_outcomes"):
                    return ()
                rows = connection.execute(
                    """
                    SELECT run_id, sequence, evaluator, status, passed, score,
                           feedback, metadata_json, error_type, elapsed_time
                    FROM evaluator_outcomes
                    WHERE run_id = ?
                    ORDER BY run_id, sequence
                    """,
                    (run_id,),
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(
                f"Could not load evaluator outcomes for run {run_id}: {error}"
            ) from error

        outcomes: list[StoredEvaluatorOutcome] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise StorageError(
                    f"Stored evaluator outcome for run {run_id} is invalid: {error}"
                ) from error
            if not isinstance(metadata, dict):
                raise StorageError(
                    f"Stored evaluator outcome for run {run_id} has invalid metadata."
                )
            outcomes.append(
                StoredEvaluatorOutcome(
                    run_id=row["run_id"],
                    sequence=int(row["sequence"]),
                    evaluator=row["evaluator"],
                    status=row["status"],
                    passed=bool(row["passed"]),
                    score=(
                        float(row["score"]) if row["score"] is not None else None
                    ),
                    feedback=row["feedback"],
                    metadata=metadata,
                    error_type=row["error_type"],
                    elapsed_time=(
                        float(row["elapsed_time"])
                        if row["elapsed_time"] is not None
                        else None
                    ),
                )
            )
        return tuple(outcomes)

    def list_runs(
        self,
        limit: int = 20,
        *,
        status: str | None = None,
        case_id: str | None = None,
        experiment_id: str | None = None,
    ) -> tuple[StoredRun, ...]:
        if limit < 1:
            return ()
        normalized_status = status.upper() if status else None
        if normalized_status not in (None, "PASS", "FAIL"):
            raise ValueError("status must be PASS, FAIL, or None.")
        try:
            with self._connection() as connection:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(runs)").fetchall()
                }
                if "experiment_id" in columns:
                    rows = connection.execute(
                        """
                        SELECT * FROM runs
                        WHERE (? IS NULL OR status = ?)
                          AND (? IS NULL OR case_id = ?)
                          AND (? IS NULL OR experiment_id = ?)
                        ORDER BY finished_at DESC, rowid DESC
                        LIMIT ?
                        """,
                        (
                            normalized_status,
                            normalized_status,
                            case_id,
                            case_id,
                            experiment_id,
                            experiment_id,
                            limit,
                        ),
                    ).fetchall()
                elif experiment_id is not None:
                    rows = []
                else:
                    rows = connection.execute(
                        """
                        SELECT * FROM runs
                        WHERE (? IS NULL OR status = ?)
                          AND (? IS NULL OR case_id = ?)
                        ORDER BY finished_at DESC, rowid DESC
                        LIMIT ?
                        """,
                        (
                            normalized_status,
                            normalized_status,
                            case_id,
                            case_id,
                            limit,
                        ),
                    ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"Could not list evaluation runs: {error}") from error
        return tuple(self._stored_run(row) for row in rows)

    def list_case_ids(self) -> tuple[str, ...]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT DISTINCT case_id FROM runs ORDER BY case_id"
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"Could not list evaluation case ids: {error}") from error
        return tuple(row["case_id"] for row in rows)

    def get_stats(self) -> RunStats:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(*) AS total_runs,
                        COALESCE(SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END), 0)
                            AS successful_runs,
                        COALESCE(SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END), 0)
                            AS failed_runs,
                        COALESCE(AVG(total_latency), 0.0) AS average_latency
                    FROM runs
                    """
                ).fetchone()
        except sqlite3.Error as error:
            raise StorageError(f"Could not load evaluation statistics: {error}") from error
        total_runs = int(row["total_runs"])
        successful_runs = int(row["successful_runs"])
        return RunStats(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=int(row["failed_runs"]),
            success_rate=(successful_runs / total_runs * 100.0) if total_runs else 0.0,
            average_latency=float(row["average_latency"]),
        )

    def create_experiment(self, experiment: Experiment) -> None:
        if self.read_only:
            raise StorageError("Cannot create an experiment through read-only storage.")
        values = (
            experiment.experiment_id,
            summarize_text(experiment.label),
            summarize_text(experiment.dataset),
            summarize_text(experiment.adapter),
            summarize_text(experiment.model) if experiment.model is not None else None,
            (
                summarize_text(experiment.agent_version)
                if experiment.agent_version is not None
                else None
            ),
            (
                summarize_text(experiment.prompt_variant)
                if experiment.prompt_variant is not None
                else None
            ),
            summarize_text(experiment.notes) if experiment.notes is not None else None,
            experiment.trials_per_case,
            experiment.total_cases,
            experiment.total_runs,
            summarize_text(experiment.started_at),
            (
                summarize_text(experiment.finished_at)
                if experiment.finished_at is not None
                else None
            ),
            experiment.status,
        )
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO experiments (
                        experiment_id, label, dataset, adapter, model,
                        agent_version, prompt_variant, notes,
                        trials_per_case, total_cases, total_runs,
                        started_at, finished_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.Error as error:
            raise StorageError(
                f"Could not create experiment {experiment.experiment_id}: {error}"
            ) from error

    def finish_experiment(
        self,
        experiment_id: str,
        status: str,
        finished_at: str,
    ) -> Experiment:
        if self.read_only:
            raise StorageError("Cannot finish an experiment through read-only storage.")
        if status not in {"completed", "completed_with_failures", "aborted"}:
            raise ValueError("Invalid final experiment status.")
        try:
            with self._connection() as connection:
                updated = connection.execute(
                    """
                    UPDATE experiments
                    SET status = ?, finished_at = ?
                    WHERE experiment_id = ?
                    """,
                    (status, summarize_text(finished_at), experiment_id),
                )
                if updated.rowcount != 1:
                    raise StorageError(f"Experiment not found: {experiment_id}")
        except sqlite3.Error as error:
            raise StorageError(
                f"Could not finish experiment {experiment_id}: {error}"
            ) from error
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise StorageError(f"Experiment not found after update: {experiment_id}")
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        try:
            with self._connection() as connection:
                if not self._table_exists(connection, "experiments"):
                    return None
                row = connection.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise StorageError(f"Could not load experiment {experiment_id}: {error}") from error
        return self._experiment(row) if row is not None else None

    def list_experiments(self, limit: int = 20) -> tuple[Experiment, ...]:
        if limit < 1:
            return ()
        try:
            with self._connection() as connection:
                if not self._table_exists(connection, "experiments"):
                    return ()
                rows = connection.execute(
                    """
                    SELECT * FROM experiments
                    ORDER BY started_at DESC, rowid DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"Could not list experiments: {error}") from error
        return tuple(self._experiment(row) for row in rows)

    def get_experiment_metrics(self, experiment_id: str) -> ExperimentMetrics:
        if self.get_experiment(experiment_id) is None:
            raise StorageError(f"Experiment not found: {experiment_id}")
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT run_id, case_id, status, total_latency
                    FROM runs
                    WHERE experiment_id = ?
                    ORDER BY case_id, trial_index, rowid
                    """,
                    (experiment_id,),
                ).fetchall()
                failure_rows = connection.execute(
                    """
                    SELECT trace_events.data_json
                    FROM runs
                    LEFT JOIN trace_events
                      ON trace_events.run_id = runs.run_id
                     AND trace_events.event_type = 'agent_end'
                    WHERE runs.experiment_id = ? AND runs.status = 'FAIL'
                    ORDER BY runs.rowid
                    """,
                    (experiment_id,),
                ).fetchall()
                evaluator_rows: list[dict[str, Any]] = []
                if self._table_exists(connection, "evaluator_outcomes"):
                    evaluator_rows = [
                        dict(row)
                        for row in connection.execute(
                            """
                            SELECT evaluator,
                                   COUNT(*) AS total_outcomes,
                                   COUNT(DISTINCT run_id) AS evaluated_runs,
                                   COALESCE(
                                       SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END),
                                       0
                                   ) AS passed_outcomes,
                                   COALESCE(
                                       SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END),
                                       0
                                   ) AS failed_outcomes,
                                   COALESCE(
                                       SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END),
                                       0
                                   ) AS error_outcomes,
                                   COUNT(score) AS score_count,
                                   AVG(score) AS average_score,
                                   MIN(score) AS min_score,
                                   MAX(score) AS max_score
                            FROM evaluator_outcomes
                            WHERE run_id IN (
                                SELECT run_id FROM runs WHERE experiment_id = ?
                            )
                            GROUP BY evaluator
                            ORDER BY evaluator
                            """,
                            (experiment_id,),
                        ).fetchall()
                    ]
        except sqlite3.Error as error:
            raise StorageError(
                f"Could not aggregate experiment {experiment_id}: {error}"
            ) from error

        case_totals: dict[str, list[float | int]] = {}
        for row in rows:
            values = case_totals.setdefault(row["case_id"], [0, 0, 0.0])
            values[0] += 1
            values[1] += int(row["status"] == "PASS")
            values[2] += float(row["total_latency"])
        per_case = tuple(
            CaseExperimentMetrics(
                case_id=case_id,
                total_runs=int(values[0]),
                passed_runs=int(values[1]),
                failed_runs=int(values[0] - values[1]),
                success_rate=(values[1] / values[0] * 100.0),
                average_latency=float(values[2] / values[0]),
            )
            for case_id, values in sorted(case_totals.items())
        )
        failure_counts: dict[str, int] = {}
        for row in failure_rows:
            failure_type = "unknown_agent_error"
            if row["data_json"] is not None:
                try:
                    data = json.loads(row["data_json"])
                except (TypeError, json.JSONDecodeError):
                    data = {}
                diagnostics = diagnostics_from_trace_data(data)
                if diagnostics and diagnostics.get("failure_type"):
                    failure_type = str(diagnostics["failure_type"])
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

        total_runs = len(rows)
        passed_runs = sum(int(row["status"] == "PASS") for row in rows)
        total_latency = sum(float(row["total_latency"]) for row in rows)
        return ExperimentMetrics(
            experiment_id=experiment_id,
            total_runs=total_runs,
            passed_runs=passed_runs,
            failed_runs=total_runs - passed_runs,
            success_rate=(passed_runs / total_runs * 100.0) if total_runs else 0.0,
            average_latency=(total_latency / total_runs) if total_runs else 0.0,
            per_case=per_case,
            failure_types=tuple(sorted(failure_counts.items())),
            evaluator_metrics=_evaluator_experiment_metrics(evaluator_rows, total_runs),
        )

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _stored_run(row: sqlite3.Row) -> StoredRun:
        columns = set(row.keys())
        return StoredRun(
            run_id=row["run_id"],
            case_id=row["case_id"],
            dataset=row["dataset"],
            adapter=row["adapter"],
            status=row["status"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            total_latency=row["total_latency"],
            tests_before_passed=bool(row["tests_before_passed"]),
            tests_after_passed=bool(row["tests_after_passed"]),
            error=row["error"],
            experiment_id=(row["experiment_id"] if "experiment_id" in columns else None),
            trial_index=(row["trial_index"] if "trial_index" in columns else None),
        )

    @staticmethod
    def _experiment(row: sqlite3.Row) -> Experiment:
        columns = set(row.keys())
        return Experiment(
            experiment_id=row["experiment_id"],
            label=row["label"],
            dataset=row["dataset"],
            adapter=row["adapter"],
            model=row["model"],
            trials_per_case=int(row["trials_per_case"]),
            total_cases=int(row["total_cases"]),
            total_runs=int(row["total_runs"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            agent_version=(
                row["agent_version"] if "agent_version" in columns else None
            ),
            prompt_variant=(
                row["prompt_variant"] if "prompt_variant" in columns else None
            ),
            notes=row["notes"] if "notes" in columns else None,
        )

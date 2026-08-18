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

from agentlab.models import EvalResult
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
    def list_runs(self, limit: int = 20) -> tuple[StoredRun, ...]:
        """Load the most recently finished runs."""


def default_database_path() -> Path:
    """Return the configured database path, defaulting to the current project."""
    configured = os.environ.get(DATABASE_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / DEFAULT_DATABASE_PATH


class SQLiteStorage(RunStorage):
    """SQLite-backed run storage with transactional writes."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_schema()
        except (OSError, sqlite3.Error) as error:
            raise StorageError(f"Could not initialize AgentLab database: {error}") from error

    def _connect(self) -> sqlite3.Connection:
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
                    error TEXT
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
                """
            )

    @staticmethod
    def _required_event(result: EvalResult, event_type: str) -> TraceEvent:
        event = next((item for item in result.trace if item.event_type == event_type), None)
        if event is None:
            raise ValueError(f"Evaluation trace is missing required event: {event_type}")
        return event

    def save_run(self, result: EvalResult, dataset: str) -> None:
        start = self._required_event(result, "run_start")
        end = self._required_event(result, "run_end")
        mismatched = [event.sequence for event in result.trace if event.run_id != result.run_id]
        if mismatched:
            raise ValueError("Every trace event must match EvalResult.run_id.")

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

        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, case_id, dataset, adapter, status,
                        started_at, finished_at, total_latency,
                        tests_before_passed, tests_after_passed, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def list_runs(self, limit: int = 20) -> tuple[StoredRun, ...]:
        if limit < 1:
            return ()
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM runs
                    ORDER BY finished_at DESC, rowid DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error as error:
            raise StorageError(f"Could not list evaluation runs: {error}") from error
        return tuple(self._stored_run(row) for row in rows)

    @staticmethod
    def _stored_run(row: sqlite3.Row) -> StoredRun:
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
        )

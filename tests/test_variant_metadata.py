from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from agentlab.models import Experiment
from agentlab.storage import SQLiteStorage


def make_experiment(
    experiment_id: str,
    *,
    agent_version: str | None = None,
    prompt_variant: str | None = None,
    model: str | None = None,
    notes: str | None = None,
) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        label="variant test",
        dataset="dataset.yaml",
        adapter="RepoDoctorAdapter",
        model=model,
        trials_per_case=3,
        total_cases=1,
        total_runs=0,
        started_at="2026-08-19T01:00:00+00:00",
        finished_at=None,
        status="running",
        agent_version=agent_version,
        prompt_variant=prompt_variant,
        notes=notes,
    )


def test_variant_metadata_is_persisted() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-variant-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.create_experiment(
            make_experiment(
                "variant-experiment",
                agent_version="repo-doctor-0.2.0",
                prompt_variant="baseline-v1",
                model="deepseek-v4-flash",
                notes="Official baseline",
            )
        )

        stored = storage.get_experiment("variant-experiment")

        assert stored is not None
        assert stored.agent_version == "repo-doctor-0.2.0"
        assert stored.prompt_variant == "baseline-v1"
        assert stored.model == "deepseek-v4-flash"
        assert stored.notes == "Official baseline"


def test_variant_metadata_fields_are_nullable() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-variant-") as directory:
        storage = SQLiteStorage(Path(directory) / "agentlab.db")
        storage.create_experiment(make_experiment("nullable-experiment"))

        stored = storage.get_experiment("nullable-experiment")

        assert stored is not None
        assert stored.agent_version is None
        assert stored.prompt_variant is None
        assert stored.model is None
        assert stored.notes is None


def test_legacy_experiment_is_readable_and_migrates_nullable_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-legacy-variant-") as directory:
        database = Path(directory) / "legacy.db"
        with closing(sqlite3.connect(database)) as connection:
            connection.executescript(
                """
                CREATE TABLE experiments (
                    experiment_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    adapter TEXT NOT NULL,
                    model TEXT,
                    trials_per_case INTEGER NOT NULL,
                    total_cases INTEGER NOT NULL,
                    total_runs INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL
                );
                INSERT INTO experiments VALUES (
                    'legacy-experiment', 'legacy baseline', 'dataset.yaml',
                    'RepoDoctorAdapter', 'deepseek-v4-flash', 3, 11, 33,
                    '2026-08-19T01:00:00+00:00',
                    '2026-08-19T01:10:00+00:00',
                    'completed_with_failures'
                );
                """
            )
            connection.commit()

        legacy = SQLiteStorage(database, read_only=True).get_experiment(
            "legacy-experiment"
        )

        assert legacy is not None
        assert legacy.model == "deepseek-v4-flash"
        assert legacy.agent_version is None
        assert legacy.prompt_variant is None
        assert legacy.notes is None

        migrated_storage = SQLiteStorage(database)
        migrated = migrated_storage.get_experiment("legacy-experiment")
        with closing(sqlite3.connect(database)) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(experiments)")
            }

        assert migrated is not None
        assert migrated.agent_version is None
        assert migrated.prompt_variant is None
        assert migrated.notes is None
        assert {"agent_version", "prompt_variant", "notes"}.issubset(columns)

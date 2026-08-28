from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def fake_repo_doctor_project(tmp_path: Path, monkeypatch) -> Path:
    """Configure a minimal provenance-valid Repo Doctor checkout for a test."""
    project = (tmp_path / "repo-doctor").resolve()
    cli = project / "repo_doctor" / "cli.py"
    cli.parent.mkdir(parents=True)
    cli.write_text("# test Repo Doctor CLI\n", encoding="utf-8")

    interpreter = project / ".venv"
    if os.name == "nt":
        interpreter = interpreter / "Scripts" / "python.exe"
    else:
        interpreter = interpreter / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"test interpreter")

    monkeypatch.setenv("AGENTLAB_REPO_DOCTOR_PROJECT", str(project))
    return project

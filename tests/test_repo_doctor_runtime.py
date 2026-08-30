from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agentlab.adapters import AgentExecutionError, AgentResumeHandle, AgentSuspended
from agentlab.adapters.repo_doctor import (
    REPO_DOCTOR_PROJECT_ENV,
    REPO_DOCTOR_STATE_ENV,
    REPO_DOCTOR_TOOLHUB_PROJECT_ENV,
    RepoDoctorAdapter,
    _checkout_interpreter,
    _configured_repo_doctor_project,
)
from agentlab.runner import create_workspace

SESSION_ID = "a" * 32


def _active_interpreter(project: Path) -> Path:
    if os.name == "nt":
        return (project / ".venv" / "Scripts" / "python.exe").resolve()
    return (project / ".venv" / "bin" / "python").resolve()


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "repo-doctor"
    (project / "repo_doctor").mkdir(parents=True)
    (project / "repo_doctor" / "cli.py").write_text("# cli\n", encoding="utf-8")
    windows = project / ".venv" / "Scripts" / "python.exe"
    windows.parent.mkdir(parents=True)
    windows.write_bytes(b"python")
    posix = project / ".venv" / "bin" / "python"
    posix.parent.mkdir(parents=True)
    posix.write_bytes(b"python")
    return project.resolve()


def _workspace(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "test_sample.py").write_text("def test_sample():\n    assert False\n")
    return create_workspace(str(source))


def _session_data(workspace: Path, phase: str, *, session_id: str = SESSION_ID) -> dict:
    return {
        "schema_version": 2,
        "session_type": "repair",
        "session_id": session_id,
        "created_at": "2026-08-28T00:00:00+00:00",
        "target_repository": str(workspace),
        "backend": "mcp",
        "phase": phase,
        "finding_id": "finding-1",
        "finding_title": "Bug",
        "target_file": "module.py",
        "expected_hash": "0" * 64,
        "proposed_hash": "1" * 64,
        "patch_trace_id": "trace-1",
        "patch_new_hash": None,
        "verification_plan": [],
        "operations": [
            {
                "request_id": "toolhub-request-must-remain-repo-doctor-owned",
                "resume_tool": "filesystem.apply_patch_approved",
            }
        ],
        "diff_summary": None,
        "error": "",
    }


def _configure(monkeypatch, tmp_path: Path) -> Path:
    project = _project(tmp_path)
    monkeypatch.setenv(REPO_DOCTOR_PROJECT_ENV, str(project))
    monkeypatch.setenv(REPO_DOCTOR_TOOLHUB_PROJECT_ENV, str(tmp_path / "toolhub"))
    monkeypatch.setenv("AGENTLAB_STATE_ROOT", str((tmp_path / "state").resolve()))
    monkeypatch.setattr(RepoDoctorAdapter, "_create_git_baseline", lambda *_args: None)
    return project


def _write_state(environment: dict[str, str], data: dict) -> Path:
    directory = Path(environment[REPO_DOCTOR_STATE_ENV]) / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['session_id']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_structured_suspension_and_resume_use_only_repo_doctor_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _configure(monkeypatch, tmp_path)
    workspace = _workspace(tmp_path)
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def fake_run(
        command,
        *,
        cwd,
        env,
        capture_output,
        text,
        check=False,
        timeout=None,
    ):
        calls.append((tuple(command), Path(cwd), env))
        if "fix" in command:
            _write_state(env, _session_data(workspace, "patch_pending"))
            return subprocess.CompletedProcess(
                command, 0, "Approval request: secret-id", ""
            )
        path = Path(env[REPO_DOCTOR_STATE_ENV]) / "sessions" / f"{SESSION_ID}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["phase"] = "verified_pass"
        path.write_text(json.dumps(data), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "request secret-id consumed", "")

    monkeypatch.setattr("agentlab.adapters.repo_doctor.run_process", fake_run)
    adapter = RepoDoctorAdapter()
    try:
        with pytest.raises(AgentSuspended) as captured:
            adapter.repair(workspace, "fix it")
        handle = captured.value.handle
        assert handle.session_id == SESSION_ID
        assert handle.adapter == "repo_doctor"
        assert "toolhub-request" not in repr(handle)
        assert (workspace / "requirements.txt").is_file()

        result = adapter.resume(workspace, handle)
        assert result.returncode == 0
        assert result.stdout == ""
        assert not (workspace / "requirements.txt").exists()
        assert calls[0][0][:4] == (
            str(_active_interpreter(project)),
            "-B",
            "-m",
            "repo_doctor.cli",
        )
        assert "--tool-backend" in calls[0][0]
        assert "--report-json" not in calls[0][0]
        assert calls[1][0][-2:] == ("resume", SESSION_ID)
        assert all(call[1] == project for call in calls)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_resume_reconciles_already_terminal_repo_doctor_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure(monkeypatch, tmp_path)
    workspace = _workspace(tmp_path)
    state_root = (tmp_path / "state" / "repo-doctor").resolve()
    _write_state(
        {REPO_DOCTOR_STATE_ENV: str(state_root)},
        _session_data(workspace, "verified_pass"),
    )
    (workspace / "requirements.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "agentlab.adapters.repo_doctor.run_process",
        lambda *_args, **_kwargs: pytest.fail("terminal reconciliation must not relaunch"),
    )

    result = RepoDoctorAdapter().resume(
        workspace,
        AgentResumeHandle(
            "repo_doctor",
            SESSION_ID,
            "approval_required",
            {"scaffold": "requirements.txt"},
        ),
    )

    assert result.returncode == 0
    assert result.diagnostics is not None
    assert result.diagnostics.final_status == "verified_pass"
    assert not (workspace / "requirements.txt").exists()


def test_repo_doctor_process_timeout_is_enforced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def timeout(*_args, **kwargs):
        assert kwargs["timeout"] == 7
        raise subprocess.TimeoutExpired(["repo-doctor"], timeout=7)

    monkeypatch.setattr("agentlab.adapters.repo_doctor.run_process", timeout)

    with pytest.raises(AgentExecutionError, match="timed out") as captured:
        RepoDoctorAdapter(process_timeout=7)._run_process(
            ["repo-doctor"],
            cwd=tmp_path,
            environment={},
            report_path=None,
        )

    assert captured.value.diagnostics is not None
    assert captured.value.diagnostics.failure_type.value == "timeout"


@pytest.mark.parametrize(
    "phase",
    ["patch_rejected", "verification_failed", "error"],
)
def test_structured_terminal_failure_is_not_suspension(
    tmp_path: Path,
    monkeypatch,
    phase: str,
) -> None:
    _configure(monkeypatch, tmp_path)
    workspace = _workspace(tmp_path)

    def fake_run(command, *, env, **_kwargs):
        _write_state(env, _session_data(workspace, phase))
        return subprocess.CompletedProcess(command, 0, "approval", "")

    monkeypatch.setattr("agentlab.adapters.repo_doctor.run_process", fake_run)
    try:
        with pytest.raises(AgentExecutionError, match=phase):
            RepoDoctorAdapter().repair(workspace, "fix it")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_malformed_lifecycle_fails_closed_and_stdout_never_suspends(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure(monkeypatch, tmp_path)
    workspace = _workspace(tmp_path)

    def malformed(command, *, env, **_kwargs):
        data = _session_data(workspace, "patch_pending")
        del data["phase"]
        _write_state(env, data)
        return subprocess.CompletedProcess(command, 0, "WAITING FOR APPROVAL", "")

    monkeypatch.setattr("agentlab.adapters.repo_doctor.run_process", malformed)
    with pytest.raises(ValueError, match="unexpected schema"):
        RepoDoctorAdapter().repair(workspace, "fix it")
    shutil.rmtree(workspace, ignore_errors=True)

    other_root = tmp_path / "other"
    monkeypatch.setenv("AGENTLAB_STATE_ROOT", str(other_root.resolve()))
    workspace = _workspace(tmp_path / "second")

    def prose_only(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, "WAITING FOR APPROVAL", "")

    monkeypatch.setattr("agentlab.adapters.repo_doctor.run_process", prose_only)
    try:
        result = RepoDoctorAdapter().repair(workspace, "fix it")
        assert result.returncode == 0
        assert result.stdout == ""
        assert result.stderr == ""
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_repo_doctor_provenance_is_checkout_local_and_sanitized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _project(tmp_path)
    monkeypatch.setenv(REPO_DOCTOR_PROJECT_ENV, str(project))
    monkeypatch.setenv("PYTHONPATH", "attacker")
    monkeypatch.setenv("PYTHONHOME", "attacker")
    monkeypatch.setenv("PYTHONSTARTUP", "attacker.py")
    monkeypatch.setenv(REPO_DOCTOR_TOOLHUB_PROJECT_ENV, str(tmp_path / "toolhub"))
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "provider-model")
    context = RepoDoctorAdapter._launch_context()
    assert context.project == project
    assert context.prefix == (
        str(_active_interpreter(project)),
        "-B",
        "-m",
        "repo_doctor.cli",
    )
    assert context.environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in context.environment
    assert "PYTHONHOME" not in context.environment
    assert "PYTHONSTARTUP" not in context.environment
    assert context.environment[REPO_DOCTOR_TOOLHUB_PROJECT_ENV] == str(
        tmp_path / "toolhub"
    )
    assert context.environment["REPO_DOCTOR_MODEL"] == "provider-model"

    assert _checkout_interpreter(project, platform="windows").name == "python.exe"
    assert _checkout_interpreter(project, platform="posix").name == "python"


def test_repo_doctor_provenance_rejects_relative_missing_and_global_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(REPO_DOCTOR_PROJECT_ENV, "relative/repo-doctor")
    with pytest.raises(ValueError, match="absolute"):
        _configured_repo_doctor_project()

    monkeypatch.setenv(REPO_DOCTOR_PROJECT_ENV, str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="does not exist"):
        _configured_repo_doctor_project()

    project = _project(tmp_path)
    noncanonical = project / ".." / project.name
    monkeypatch.setenv(REPO_DOCTOR_PROJECT_ENV, str(noncanonical))
    with pytest.raises(ValueError, match="canonical"):
        _configured_repo_doctor_project()

    monkeypatch.delenv(REPO_DOCTOR_PROJECT_ENV)
    monkeypatch.setenv("PATH", str(tmp_path / "global-bin"))
    with pytest.raises(ValueError, match=REPO_DOCTOR_PROJECT_ENV):
        _configured_repo_doctor_project(platform="posix")

    project = tmp_path / "checkout"
    (project / "repo_doctor").mkdir(parents=True)
    (project / "repo_doctor" / "cli.py").write_text("# cli\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkout-local"):
        _checkout_interpreter(project, platform="posix")

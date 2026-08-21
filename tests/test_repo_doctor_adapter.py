import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agentlab.adapters import (
    AgentAdapter,
    AgentExecutionError,
    AgentPreflightError,
    RepoDoctorAdapter,
)
from agentlab.adapters.repo_doctor import WORKSPACE_MARKER
from agentlab.diagnostics import AgentFailureType
from agentlab.runner import create_workspace


def test_agent_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        AgentAdapter()


def test_repo_doctor_preflight_accepts_valid_looking_ascii_key(monkeypatch) -> None:
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", "  sk-test_0123456789abcdef  ")
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", " https://provider.invalid/v1 ")
    monkeypatch.setenv("REPO_DOCTOR_MODEL", " model-name ")

    result = RepoDoctorAdapter().preflight()

    assert result.model == "model-name"


@pytest.mark.parametrize(
    "api_key",
    [
        None,
        "",
        "   \t",
        "这是你的真实 DeepSeek API Key",
        "your api key goes here",
        "sk-0123456789abcd密钥",
        "short-key",
    ],
)
def test_repo_doctor_preflight_rejects_obviously_invalid_api_keys(
    monkeypatch,
    api_key,
) -> None:
    if api_key is None:
        monkeypatch.delenv("REPO_DOCTOR_API_KEY", raising=False)
    else:
        monkeypatch.setenv("REPO_DOCTOR_API_KEY", api_key)
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "model-name")

    with pytest.raises(
        AgentPreflightError,
        match="^REPO_DOCTOR_API_KEY appears invalid$",
    ) as captured:
        RepoDoctorAdapter().preflight()

    if api_key:
        assert api_key not in str(captured.value)


def test_repo_doctor_rejects_non_agentlab_workspace() -> None:
    with tempfile.TemporaryDirectory(prefix="not-agentlab-") as directory:
        adapter = RepoDoctorAdapter()

        with pytest.raises(ValueError, match="only run in a temporary workspace"):
            adapter.repair(Path(directory), "fix it")


@pytest.mark.parametrize(
    ("configured_variant", "expected_variant"),
    [(None, "baseline-v1"), ("candidate-v2", "candidate-v2")],
)
def test_repo_doctor_uses_selected_prompt_variant_in_verified_cli_shape(
    monkeypatch,
    configured_variant: str | None,
    expected_variant: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (source / "test_module.py").write_text(
            "from module import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
            encoding="utf-8",
        )
        workspace = create_workspace(str(source))
        calls: list[tuple[tuple[str, ...], Path, bool]] = []
        observed_task = []
        executable = str(Path("C:/tools/repo-doctor.exe"))

        class Result:
            def __init__(self, returncode=0, stdout="", stderr="") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        def fake_run(command, *, cwd, capture_output, text, check=False):
            calls.append((tuple(command), Path(cwd), check))
            if tuple(command)[:2] == ("git", "diff"):
                return Result(stdout="--- a/module.py\n+++ b/module.py\n")
            if next(iter(command)) == executable:
                task_path = Path(command[command.index("--task-file") + 1])
                observed_task.append(task_path.read_text(encoding="utf-8"))
                return Result(stdout="Patch applied\nVerification passed\nChange kept\n")
            return Result()

        monkeypatch.setattr("agentlab.adapters.repo_doctor.shutil.which", lambda _: executable)
        monkeypatch.setattr("agentlab.adapters.repo_doctor.subprocess.run", fake_run)

        try:
            adapter = (
                RepoDoctorAdapter(verification_timeout=45)
                if configured_variant is None
                else RepoDoctorAdapter(
                    verification_timeout=45,
                    prompt_variant=configured_variant,
                )
            )
            result = adapter.repair(workspace, "fix VALUE")

            agent_command, agent_cwd, agent_check = calls[-2]
            assert agent_command[:6] == (
                executable,
                "fix",
                str(workspace.resolve()),
                "--ai",
                "--prompt-variant",
                expected_variant,
            )
            assert "--task-file" in agent_command
            assert "--report-json" in agent_command
            assert agent_command[-2:] == ("--timeout", "45")
            assert agent_cwd == workspace.resolve()
            assert agent_check is False
            assert observed_task == ["fix VALUE"]
            assert calls[-1][0][:2] == ("git", "diff")
            assert [call[0][:2] for call in calls[:-2]] == [
                ("git", "init"),
                ("git", "add"),
                ("git", "-c"),
            ]
            assert not (workspace / "requirements.txt").exists()
            assert not (workspace / WORKSPACE_MARKER).exists()
            assert result.returncode == 0
            assert result.stdout.startswith("Patch applied")
            assert result.diagnostics is not None
            assert result.diagnostics.failure_type is None
            assert result.diagnostics.patch_applied is True
            assert result.diagnostics.patch_diff == "--- a/module.py\n+++ b/module.py\n"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def test_repo_doctor_exposes_non_secret_execution_metadata(monkeypatch) -> None:
    monkeypatch.setenv("REPO_DOCTOR_MODEL", "deepseek-v4-flash")
    adapter = RepoDoctorAdapter(
        prompt_variant="candidate-v2",
        agent_version="repo-doctor-0.2.0",
    )

    assert adapter.trace_metadata() == {
        "prompt_variant": "candidate-v2",
        "agent_version": "repo-doctor-0.2.0",
        "model": "deepseek-v4-flash",
    }


def test_repo_doctor_exposes_verification_failure_diagnostics(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (source / "test_module.py").write_text(
            "from module import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
            encoding="utf-8",
        )
        workspace = create_workspace(str(source))
        executable = str(Path("C:/tools/repo-doctor.exe"))

        class Result:
            returncode = 1
            stdout = (
                "Patch applied\n"
                "Verification failed: Verification failed: Python tests.\n"
                "Rolling back\n"
                "Repository restored successfully\n"
            )
            stderr = ""

        def fake_run(command, **_kwargs):
            if tuple(command)[:2] == ("git", "diff"):
                return type("DiffResult", (), {"returncode": 0, "stdout": ""})()
            return Result()

        monkeypatch.setattr("agentlab.adapters.repo_doctor.shutil.which", lambda _: executable)
        monkeypatch.setattr("agentlab.adapters.repo_doctor.subprocess.run", fake_run)

        try:
            with pytest.raises(AgentExecutionError) as captured:
                RepoDoctorAdapter().repair(workspace, "fix VALUE")

            diagnostics = captured.value.diagnostics
            assert diagnostics is not None
            assert (
                diagnostics.failure_type
                is AgentFailureType.REPAIR_VERIFICATION_FAILED
            )
            assert diagnostics.failure_phase == "verification"
            assert diagnostics.patch_applied is True
            assert diagnostics.rollback_attempted is True
            assert diagnostics.rollback_succeeded is True
            assert diagnostics.patch_diff is None
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def test_repo_doctor_parses_structured_report_and_preserves_attempted_patch(
    monkeypatch,
) -> None:
    secret = "adapter-report-secret-123456"
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", secret)
    with tempfile.TemporaryDirectory(prefix="agentlab-test-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (source / "test_module.py").write_text(
            "from module import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
            encoding="utf-8",
        )
        workspace = create_workspace(str(source))
        executable = str(Path("C:/tools/repo-doctor.exe"))
        observed = {}

        class Result:
            returncode = 1
            stdout = "opaque failure"
            stderr = ""

        def fake_run(command, **_kwargs):
            if next(iter(command)) == executable:
                task_path = Path(command[command.index("--task-file") + 1])
                report_path = Path(command[command.index("--report-json") + 1])
                observed["task"] = task_path.read_text(encoding="utf-8")
                observed["report_path"] = report_path
                report_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "prompt_variant": "candidate-v3",
                            "task_provided": True,
                            "analysis_summary": "Two related findings.",
                            "selected_finding": {"id": "finding-2", "title": "Coupled bug"},
                            "behavioral_contract": {
                                "must_fix": ["Fix both related failures."],
                                "must_preserve": ["Keep existing valid values."],
                                "evidence": ["2 failed, 3 passed"],
                                "rationale": "One patch must satisfy the contract.",
                            },
                            "patch": {
                                "file": "module.py",
                                "diff": f"-VALUE = 1\n+VALUE = 2\n# password={secret}",
                            },
                            "patch_applied": True,
                            "verification": {
                                "summary": "Verification failed.",
                                "commands": [
                                    {
                                        "name": "Python tests",
                                        "command": ["pytest", "-q"],
                                        "returncode": 1,
                                        "passed": False,
                                        "stdout_summary": "3 passed, 2 failed",
                                        "stderr_summary": f"Authorization: Bearer {secret}",
                                    }
                                ],
                            },
                            "rollback_attempted": True,
                            "rollback_succeeded": True,
                            "final_status": "rolled_back",
                        }
                    ),
                    encoding="utf-8",
                )
                return Result()
            return type("GitResult", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr("agentlab.adapters.repo_doctor.shutil.which", lambda _: executable)
        monkeypatch.setattr("agentlab.adapters.repo_doctor.subprocess.run", fake_run)

        try:
            with pytest.raises(AgentExecutionError) as captured:
                RepoDoctorAdapter(prompt_variant="candidate-v3").repair(
                    workspace,
                    f"Fix VALUE without exposing api_key={secret}",
                )

            diagnostics = captured.value.diagnostics
            assert diagnostics is not None
            assert observed["task"] == "Fix VALUE without exposing api_key=[REDACTED]"
            assert observed["report_path"].name == "repair-report.json"
            assert diagnostics.selected_finding == {
                "id": "finding-2",
                "title": "Coupled bug",
            }
            assert diagnostics.behavioral_contract is not None
            assert diagnostics.behavioral_contract["must_preserve"] == [
                "Keep existing valid values."
            ]
            assert diagnostics.patch_diff is not None
            assert "+VALUE = 2" in diagnostics.patch_diff
            assert diagnostics.verification_command == "pytest -q"
            assert diagnostics.verification_returncode == 1
            assert "3 passed, 2 failed" in diagnostics.verification_output
            assert diagnostics.rollback_succeeded is True
            assert diagnostics.final_status == "rolled_back"
            assert secret not in repr(diagnostics)
            assert "[REDACTED]" in diagnostics.patch_diff
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

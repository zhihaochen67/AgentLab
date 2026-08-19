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


def test_repo_doctor_uses_verified_cli_shape_and_workspace(monkeypatch) -> None:
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
                return Result(stdout="Patch applied\nVerification passed\nChange kept\n")
            return Result()

        monkeypatch.setattr("agentlab.adapters.repo_doctor.shutil.which", lambda _: executable)
        monkeypatch.setattr("agentlab.adapters.repo_doctor.subprocess.run", fake_run)

        try:
            result = RepoDoctorAdapter(verification_timeout=45).repair(workspace, "fix VALUE")

            assert calls[-2] == (
                (
                    executable,
                    "fix",
                    str(workspace.resolve()),
                    "--ai",
                    "--timeout",
                    "45",
                ),
                workspace.resolve(),
                False,
            )
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

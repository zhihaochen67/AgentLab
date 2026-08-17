import shutil
import tempfile
from pathlib import Path

import pytest

from agentlab.adapters import AgentAdapter, RepoDoctorAdapter
from agentlab.adapters.repo_doctor import WORKSPACE_MARKER
from agentlab.runner import create_workspace


def test_agent_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        AgentAdapter()


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
            returncode = 0
            stdout = "Patch applied"
            stderr = ""

        def fake_run(command, *, cwd, capture_output, text, check=False):
            calls.append((tuple(command), Path(cwd), check))
            return Result()

        monkeypatch.setattr("agentlab.adapters.repo_doctor.shutil.which", lambda _: executable)
        monkeypatch.setattr("agentlab.adapters.repo_doctor.subprocess.run", fake_run)

        try:
            result = RepoDoctorAdapter(verification_timeout=45).repair(workspace, "fix VALUE")

            assert calls[-1] == (
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
            assert [call[0][:2] for call in calls[:-1]] == [
                ("git", "init"),
                ("git", "add"),
                ("git", "-c"),
            ]
            assert not (workspace / "requirements.txt").exists()
            assert not (workspace / WORKSPACE_MARKER).exists()
            assert result.returncode == 0
            assert result.stdout == "Patch applied"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

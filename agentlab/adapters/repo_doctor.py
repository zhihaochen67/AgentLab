"""Repo Doctor CLI integration."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab.adapters.base import AgentAdapter, AgentExecutionError, AgentRunResult

WORKSPACE_MARKER = ".agentlab-workspace"
_PYTHON_MANIFESTS = ("pyproject.toml", "requirements.txt", "setup.py")


@dataclass(frozen=True)
class RepoDoctorAdapter(AgentAdapter):
    """Run Repo Doctor's verified AI repair against an AgentLab workspace."""

    executable: str = "repo-doctor"
    verification_timeout: int = 120

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        """Run one Repo Doctor semantic repair in the temporary workspace.

        Repo Doctor discovers the problem from failed verification output, so its
        CLI does not accept AgentLab's natural-language task as an argument.
        """
        del task
        workspace = self._validated_workspace(workspace)
        executable = shutil.which(self.executable)
        if executable is None:
            raise RuntimeError(f"Repo Doctor CLI was not found: {self.executable}")

        scaffold = self._ensure_python_manifest(workspace)
        git_directory = workspace / ".git"
        try:
            self._create_git_baseline(workspace)
            result = subprocess.run(
                [
                    executable,
                    "fix",
                    str(workspace),
                    "--ai",
                    "--timeout",
                    str(self.verification_timeout),
                ],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            agent_result = AgentRunResult(result.returncode, result.stdout, result.stderr)
            if result.returncode != 0:
                raise AgentExecutionError(
                    f"Repo Doctor exited with status {result.returncode}",
                    agent_result,
                )
            return agent_result
        finally:
            if git_directory.is_dir():
                shutil.rmtree(git_directory, onerror=self._remove_readonly)
            if scaffold is not None:
                scaffold.unlink(missing_ok=True)
            (workspace / WORKSPACE_MARKER).unlink(missing_ok=True)

    @staticmethod
    def _remove_readonly(function, path: str, _error) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    @staticmethod
    def _validated_workspace(workspace: Path) -> Path:
        try:
            resolved = workspace.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"AgentLab workspace does not exist: {workspace}") from error

        temporary_root = Path(tempfile.gettempdir()).resolve()
        if (
            resolved.parent != temporary_root
            or not resolved.name.startswith("agentlab_")
            or not (resolved / WORKSPACE_MARKER).is_file()
        ):
            raise ValueError(
                "Repo Doctor may only run in a temporary workspace created by AgentLab."
            )
        return resolved

    @staticmethod
    def _ensure_python_manifest(workspace: Path) -> Path | None:
        if any((workspace / name).exists() for name in _PYTHON_MANIFESTS):
            return None
        has_python_tests = any(workspace.glob("test_*.py")) or (workspace / "tests").is_dir()
        if not has_python_tests:
            return None
        manifest = workspace / "requirements.txt"
        manifest.write_text("", encoding="utf-8")
        return manifest

    @staticmethod
    def _create_git_baseline(workspace: Path) -> None:
        commands = (
            ("git", "init", "--quiet"),
            ("git", "add", "--all"),
            (
                "git",
                "-c",
                "user.name=AgentLab",
                "-c",
                "user.email=agentlab@localhost",
                "commit",
                "--quiet",
                "-m",
                "AgentLab evaluation baseline",
            ),
        )
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except FileNotFoundError as error:
                raise RuntimeError("Git is required by Repo Doctor fix mode.") from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                message = "Could not create the temporary Git baseline"
                if detail:
                    message += f": {detail}"
                raise RuntimeError(message) from error

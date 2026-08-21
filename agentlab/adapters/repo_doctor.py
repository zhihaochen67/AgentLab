"""Repo Doctor CLI integration."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab.adapters.base import (
    AgentAdapter,
    AgentExecutionError,
    AgentPreflightError,
    AgentPreflightResult,
    AgentRunResult,
)
from agentlab.diagnostics import (
    AgentDiagnostics,
    AgentFailureType,
    diagnose_repo_doctor_report,
)
from agentlab.tracer import summarize_text

WORKSPACE_MARKER = ".agentlab-workspace"
DEFAULT_PROMPT_VARIANT = "baseline-v1"
_PYTHON_MANIFESTS = ("pyproject.toml", "requirements.txt", "setup.py")
_MIN_API_KEY_LENGTH = 16
_MAX_REPORT_BYTES = 2_000_000
_MAX_TASK_TEXT = 8_000
_API_KEY_PLACEHOLDERS = (
    "api key",
    "your key",
    "你的",
    "真实 deepseek",
    "placeholder",
    "replace me",
    "change me",
    "changeme",
    "insert key",
    "paste key",
    "example key",
    "dummy key",
    "test key",
)


def is_plausible_api_key(value: str | None) -> bool:
    """Reject obviously invalid API keys without assuming a provider format."""
    if value is None:
        return False
    candidate = value.strip()
    if not candidate or not candidate.isascii():
        return False
    if len(candidate) < _MIN_API_KEY_LENGTH:
        return False
    normalized = candidate.casefold().replace("_", " ").replace("-", " ")
    return not any(placeholder in normalized for placeholder in _API_KEY_PLACEHOLDERS)


@dataclass(frozen=True)
class RepoDoctorAdapter(AgentAdapter):
    """Run Repo Doctor's verified AI repair against an AgentLab workspace."""

    executable: str = "repo-doctor"
    verification_timeout: int = 120
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    agent_version: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt_variant.strip():
            raise ValueError("Repo Doctor prompt_variant must be non-empty.")

    def preflight(self) -> AgentPreflightResult:
        """Require Repo Doctor provider settings without exposing their values."""
        api_key_name = "REPO_DOCTOR_API_KEY"
        if not is_plausible_api_key(os.environ.get(api_key_name)):
            raise AgentPreflightError(invalid_variables=(api_key_name,))

        names = ("REPO_DOCTOR_BASE_URL", "REPO_DOCTOR_MODEL")
        values = {name: os.environ.get(name, "").strip() for name in names}
        missing = tuple(name for name in names if not values[name])
        if missing:
            raise AgentPreflightError(missing)
        return AgentPreflightResult(model=values["REPO_DOCTOR_MODEL"])

    def trace_metadata(self) -> dict[str, str]:
        """Return the exact non-secret variant metadata used by Repo Doctor."""
        metadata = {"prompt_variant": self.prompt_variant}
        if self.agent_version is not None:
            metadata["agent_version"] = self.agent_version
        model = os.environ.get("REPO_DOCTOR_MODEL", "").strip()
        if model:
            metadata["model"] = model
        return metadata

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        """Run one task-aware Repo Doctor repair in the temporary workspace."""
        workspace = self._validated_workspace(workspace)
        executable = shutil.which(self.executable)
        if executable is None:
            diagnostics = AgentDiagnostics(
                failure_type=AgentFailureType.AGENT_PROCESS_ERROR,
                failure_phase="process",
                returncode=None,
                stderr_summary=f"Repo Doctor CLI was not found: {self.executable}",
            )
            raise AgentExecutionError(
                "Repo Doctor process could not start (agent_process_error)",
                AgentRunResult(None, "", diagnostics.stderr_summary, diagnostics),
            )

        scaffold = self._ensure_python_manifest(workspace)
        git_directory = workspace / ".git"
        try:
            self._create_git_baseline(workspace)
            with tempfile.TemporaryDirectory(prefix="agentlab-repo-doctor-") as directory:
                artifacts = Path(directory)
                report_path = artifacts / "repair-report.json"
                command = [
                    executable,
                    "fix",
                    str(workspace),
                    "--ai",
                    "--prompt-variant",
                    self.prompt_variant,
                ]
                if task.strip():
                    task_path = artifacts / "task.txt"
                    task_path.write_text(
                        summarize_text(task, limit=_MAX_TASK_TEXT),
                        encoding="utf-8",
                    )
                    command.extend(("--task-file", str(task_path)))
                command.extend(
                    (
                        "--report-json",
                        str(report_path),
                        "--timeout",
                        str(self.verification_timeout),
                    )
                )
                try:
                    result = subprocess.run(
                        command,
                        cwd=workspace,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    stdout = self._process_output(error.stdout)
                    stderr = self._process_output(error.stderr)
                    report = self._read_repair_report(report_path)
                    diagnostics = diagnose_repo_doctor_report(
                        report,
                        None,
                        stdout,
                        stderr,
                        process_timed_out=True,
                    )
                    raise AgentExecutionError(
                        "Repo Doctor process timed out (timeout)",
                        AgentRunResult(None, stdout, stderr, diagnostics),
                    ) from error
                except OSError as error:
                    diagnostics = AgentDiagnostics(
                        failure_type=AgentFailureType.AGENT_PROCESS_ERROR,
                        failure_phase="process",
                        returncode=None,
                        stderr_summary=str(error),
                    )
                    raise AgentExecutionError(
                        "Repo Doctor process could not start (agent_process_error)",
                        AgentRunResult(None, "", str(error), diagnostics),
                    ) from error

                report = self._read_repair_report(report_path)
                patch_diff = None if report is not None else self._capture_git_diff(workspace)
                diagnostics = diagnose_repo_doctor_report(
                    report,
                    result.returncode,
                    result.stdout,
                    result.stderr,
                    fallback_patch_diff=patch_diff,
                )
            agent_result = AgentRunResult(
                result.returncode,
                result.stdout,
                result.stderr,
                diagnostics,
            )
            if result.returncode != 0:
                failure_type = diagnostics.failure_type or AgentFailureType.UNKNOWN_AGENT_ERROR
                raise AgentExecutionError(
                    f"Repo Doctor failed during {diagnostics.failure_phase or 'agent'} "
                    f"({failure_type.value})",
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
    def _process_output(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""

    @staticmethod
    def _capture_git_diff(workspace: Path) -> str | None:
        """Capture a surviving real patch without changing repository state."""
        try:
            result = subprocess.run(
                ("git", "diff", "--no-ext-diff", "--no-color", "--binary", "--"),
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return result.stdout if result.returncode == 0 and result.stdout.strip() else None

    @staticmethod
    def _read_repair_report(path: Path) -> dict | None:
        """Load one bounded JSON object; legacy/malformed reports use old diagnostics."""
        try:
            if not path.is_file() or path.stat().st_size > _MAX_REPORT_BYTES:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

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

"""Repo Doctor CLI integration with opaque repair-session resumption."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab.adapters.base import (
    AgentAdapter,
    AgentExecutionError,
    AgentInfo,
    AgentPreflightError,
    AgentPreflightResult,
    AgentResumeHandle,
    AgentRunResult,
    AgentSuspended,
)
from agentlab.diagnostics import (
    AgentDiagnostics,
    AgentFailureType,
    diagnose_repo_doctor_report,
)
from agentlab.platform_paths import repo_doctor_state_root
from agentlab.providers import is_plausible_api_key
from agentlab.subprocesses import run_process
from agentlab.tracer import summarize_text

WORKSPACE_MARKER = ".agentlab-workspace"
DEFAULT_PROMPT_VARIANT = "baseline-v1"
REPO_DOCTOR_PROJECT_ENV = "AGENTLAB_REPO_DOCTOR_PROJECT"
REPO_DOCTOR_STATE_ENV = "REPO_DOCTOR_STATE_ROOT"
REPO_DOCTOR_TOOLHUB_PROJECT_ENV = "REPO_DOCTOR_TOOLHUB_PROJECT"
_PYTHON_MANIFESTS = ("pyproject.toml", "requirements.txt", "setup.py")
_PYTHON_REDIRECTION_VARIABLES = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONINSPECT",
    "PYTHONBREAKPOINT",
    "PYTHONSAFEPATH",
    "PYTHONEXECUTABLE",
    "__PYVENV_LAUNCHER__",
)
_MAX_REPORT_BYTES = 2_000_000
_MAX_REPO_DOCTOR_SESSION_BYTES = 1_000_000
_MAX_TASK_TEXT = 8_000
_REPAIR_SESSION_SCHEMA_VERSION = 2
_SESSION_ID = re.compile(r"[0-9a-f]{32}")
_REPAIR_SESSION_KEYS = {
    "schema_version",
    "session_type",
    "session_id",
    "created_at",
    "target_repository",
    "backend",
    "phase",
    "finding_id",
    "finding_title",
    "target_file",
    "expected_hash",
    "proposed_hash",
    "patch_trace_id",
    "patch_new_hash",
    "verification_plan",
    "operations",
    "diff_summary",
    "error",
}
_SUSPENDED_PHASES = {"patch_pending", "verification_pending"}
_SUCCESS_PHASES = {"verified_pass"}
_FAILURE_PHASES = {
    "patch_rejected",
    "patch_expired",
    "patch_conflict",
    "verification_failed",
    "verification_rejected",
    "verification_expired",
    "error",
}


@dataclass(frozen=True)
class _RepoDoctorRepairState:
    session_id: str
    target_repository: str
    phase: str


@dataclass(frozen=True)
class _LaunchContext:
    prefix: tuple[str, ...]
    project: Path
    environment: dict[str, str]


@dataclass(frozen=True)
class RepoDoctorAdapter(AgentAdapter):
    """Run Repo Doctor from one verified checkout and resume its own sessions."""

    verification_timeout: int = 120
    process_timeout: int = 300
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    agent_version: str | None = None
    trusted_execution: bool = False

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name="repo_doctor",
            description="AI coding repair agent evaluated by AgentLab",
            agent_type="coding-agent",
            capabilities=(
                "code-repair",
                "test-fixing",
                "repository-analysis",
                "resumable-execution",
            ),
        )

    def __post_init__(self) -> None:
        if not self.prompt_variant.strip():
            raise ValueError("Repo Doctor prompt_variant must be non-empty.")
        if self.verification_timeout < 1:
            raise ValueError("Repo Doctor verification_timeout must be positive.")
        if self.process_timeout < 1:
            raise ValueError("Repo Doctor process_timeout must be positive.")

    def preflight(self) -> AgentPreflightResult:
        """Require provider settings and verified Repo Doctor provenance."""
        api_key_name = "REPO_DOCTOR_API_KEY"
        if not is_plausible_api_key(os.environ.get(api_key_name)):
            raise AgentPreflightError(invalid_variables=(api_key_name,))

        names = ("REPO_DOCTOR_BASE_URL", "REPO_DOCTOR_MODEL")
        values = {name: os.environ.get(name, "").strip() for name in names}
        missing = tuple(name for name in names if not values[name])
        if missing:
            raise AgentPreflightError(missing)
        self._require_trusted_execution()
        launch = self._launch_context()
        self._require_trusted_execution_capability(launch)
        return AgentPreflightResult(model=values["REPO_DOCTOR_MODEL"])

    def trace_metadata(self) -> dict[str, str]:
        """Return the exact non-secret variant metadata used by Repo Doctor."""
        metadata = {"prompt_variant": self.prompt_variant}
        metadata["trusted_execution"] = (
            "enabled" if self.trusted_execution else "disabled"
        )
        if self.agent_version is not None:
            metadata["agent_version"] = self.agent_version
        model = os.environ.get("REPO_DOCTOR_MODEL", "").strip()
        if model:
            metadata["model"] = model
        return metadata

    def repair(self, workspace: Path, task: str) -> AgentRunResult:
        """Start one task-aware repair; suspend only from Repo Doctor session JSON."""
        workspace = self._validated_workspace(workspace)
        self._require_trusted_execution()
        launch = self._launch_context()
        scaffold = self._ensure_python_manifest(workspace)
        suspended = False
        use_mcp = bool(os.environ.get(REPO_DOCTOR_TOOLHUB_PROJECT_ENV, "").strip())
        state_root = repo_doctor_state_root() if use_mcp else None
        if state_root is not None and (
            state_root == workspace or workspace in state_root.parents
        ):
            raise ValueError(
                "AgentLab state root must remain outside the evaluation workspace."
            )
        known_sessions = (
            self._session_ids(state_root) if state_root is not None else set()
        )
        try:
            self._create_git_baseline(workspace)
            with tempfile.TemporaryDirectory(
                prefix="agentlab-repo-doctor-"
            ) as directory:
                artifacts = Path(directory)
                report_path = artifacts / "repair-report.json"
                command = [
                    *launch.prefix,
                    "fix",
                    str(workspace),
                    "--ai",
                    "--trusted-execution",
                    "--prompt-variant",
                    self.prompt_variant,
                ]
                if use_mcp:
                    command.extend(("--tool-backend", "mcp"))
                if task.strip():
                    task_path = artifacts / "task.txt"
                    task_path.write_text(
                        summarize_text(task, limit=_MAX_TASK_TEXT),
                        encoding="utf-8",
                    )
                    command.extend(("--task-file", str(task_path)))
                if not use_mcp:
                    command.extend(("--report-json", str(report_path)))
                command.extend(("--timeout", str(self.verification_timeout)))
                environment = launch.environment
                if state_root is not None:
                    environment = dict(environment)
                    environment[REPO_DOCTOR_STATE_ENV] = str(state_root)
                result = self._run_process(
                    command,
                    cwd=launch.project,
                    environment=environment,
                    report_path=report_path if not use_mcp else None,
                )

                if use_mcp:
                    state = self._new_repair_state(
                        state_root, known_sessions, workspace
                    )
                    outcome = self._mcp_outcome(result, state, scaffold)
                    if isinstance(outcome, AgentSuspended):
                        suspended = True
                        raise outcome
                    return outcome

                report = self._read_repair_report(report_path)
                patch_diff = (
                    None if report is not None else self._capture_git_diff(workspace)
                )
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
                if diagnostics.final_status in {
                    "safe_preview",
                    "preview",
                    "dry_run",
                }:
                    raise AgentExecutionError(
                        "Repo Doctor returned a preview-only result instead of "
                        "a trusted repair execution",
                        agent_result,
                    )
                if result.returncode != 0:
                    failure_type = (
                        diagnostics.failure_type or AgentFailureType.UNKNOWN_AGENT_ERROR
                    )
                    raise AgentExecutionError(
                        f"Repo Doctor failed during "
                        f"{diagnostics.failure_phase or 'agent'} ({failure_type.value})",
                        agent_result,
                    )
                return agent_result
        finally:
            if not suspended:
                self._cleanup_after_terminal(workspace, scaffold)

    def resume(
        self,
        workspace: Path,
        handle: AgentResumeHandle,
    ) -> AgentRunResult:
        """Resume exactly one Repo Doctor-owned repair session."""
        workspace = self._validated_workspace(workspace)
        self._require_trusted_execution()
        scaffold = self._validate_resume_handle(handle)
        state_root = repo_doctor_state_root()
        if state_root == workspace or workspace in state_root.parents:
            raise ValueError(
                "AgentLab state root must remain outside the evaluation workspace."
            )
        state = self._read_repair_state(state_root, handle.session_id, workspace)
        suspended = False
        try:
            if state.phase not in _SUSPENDED_PHASES:
                return self._mcp_outcome(
                    subprocess.CompletedProcess((), 0, "", ""),
                    state,
                    scaffold,
                )
            launch = self._launch_context()
            environment = dict(launch.environment)
            environment[REPO_DOCTOR_STATE_ENV] = str(state_root)
            result = self._run_process(
                [*launch.prefix, "resume", handle.session_id],
                cwd=launch.project,
                environment=environment,
                report_path=None,
            )
            updated = self._read_repair_state(state_root, handle.session_id, workspace)
            outcome = self._mcp_outcome(result, updated, scaffold)
            if isinstance(outcome, AgentSuspended):
                suspended = True
                raise outcome
            return outcome
        finally:
            if not suspended:
                self._cleanup_after_terminal(workspace, scaffold)

    def _require_trusted_execution(self) -> None:
        if not self.trusted_execution:
            raise ValueError(
                "Repo Doctor repair is disabled until trusted execution is "
                "explicitly authorized; pass --repo-doctor-trusted-execution."
            )

    @staticmethod
    def _require_trusted_execution_capability(launch: _LaunchContext) -> None:
        cli_path = launch.project / "repo_doctor" / "cli.py"
        try:
            if cli_path.stat().st_size > _MAX_REPORT_BYTES:
                raise ValueError("Repo Doctor CLI source is unexpectedly large.")
            syntax = ast.parse(
                cli_path.read_text(encoding="utf-8"), filename=str(cli_path)
            )
        except (OSError, SyntaxError, UnicodeError) as error:
            raise ValueError(
                "Could not validate the configured Repo Doctor CLI capability."
            ) from error
        if not any(
            isinstance(node, ast.Constant) and node.value == "--trusted-execution"
            for node in ast.walk(syntax)
        ):
            raise ValueError(
                "Configured Repo Doctor is incompatible: its CLI does not expose "
                "the required --trusted-execution option."
            )

    def _mcp_outcome(
        self,
        process: subprocess.CompletedProcess[str],
        state: _RepoDoctorRepairState | None,
        scaffold: Path | None,
    ) -> AgentRunResult | AgentSuspended:
        if process.returncode != 0:
            diagnostics = AgentDiagnostics(
                failure_type=AgentFailureType.AGENT_PROCESS_ERROR,
                failure_phase="process",
                returncode=process.returncode,
            )
            raise AgentExecutionError(
                "Repo Doctor process failed while managing its repair session",
                AgentRunResult(
                    process.returncode,
                    "",
                    "",
                    diagnostics,
                ),
            )
        if state is None:
            diagnostics = AgentDiagnostics(None, None, process.returncode)
            return AgentRunResult(process.returncode, "", "", diagnostics)
        safe_result = AgentRunResult(
            process.returncode,
            "",
            "",
            AgentDiagnostics(
                None
                if state.phase in _SUCCESS_PHASES | _SUSPENDED_PHASES
                else AgentFailureType.UNKNOWN_AGENT_ERROR,
                None if state.phase in _SUCCESS_PHASES | _SUSPENDED_PHASES else "agent",
                process.returncode,
                final_status=state.phase,
            ),
        )
        if state.phase in _SUSPENDED_PHASES:
            metadata = {"scaffold": scaffold.name} if scaffold is not None else {}
            return AgentSuspended(
                AgentResumeHandle(
                    adapter=self.info.name,
                    session_id=state.session_id,
                    reason="approval_required",
                    metadata=metadata,
                ),
                safe_result,
            )
        if state.phase in _SUCCESS_PHASES:
            return safe_result
        if state.phase in _FAILURE_PHASES:
            raise AgentExecutionError(
                f"Repo Doctor repair session ended in {state.phase}",
                safe_result,
            )
        raise AgentExecutionError(
            "Repo Doctor repair session has an incomplete or unknown lifecycle phase",
            AgentRunResult(
                process.returncode,
                "",
                "",
                AgentDiagnostics(
                    AgentFailureType.UNKNOWN_AGENT_ERROR,
                    "agent",
                    process.returncode,
                    final_status=state.phase,
                ),
            ),
        )

    def _run_process(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        report_path: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return run_process(
                command,
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.process_timeout,
            )
        except subprocess.TimeoutExpired as error:
            stdout = self._process_output(error.stdout)
            stderr = self._process_output(error.stderr)
            report = self._read_repair_report(report_path) if report_path else None
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

    @staticmethod
    def _launch_context() -> _LaunchContext:
        project = _configured_repo_doctor_project()
        interpreter = _checkout_interpreter(project)
        environment = dict(os.environ)
        for name in _PYTHON_REDIRECTION_VARIABLES:
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        return _LaunchContext(
            prefix=(str(interpreter), "-B", "-m", "repo_doctor.cli"),
            project=project,
            environment=environment,
        )

    @staticmethod
    def _session_ids(root: Path) -> set[str]:
        directory = root / "sessions"
        if not directory.exists():
            return set()
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Repo Doctor session directory is unsafe.")
        return {
            path.stem
            for path in directory.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and _SESSION_ID.fullmatch(path.stem) is not None
            and path.suffix == ".json"
        }

    @classmethod
    def _new_repair_state(
        cls,
        root: Path,
        known: set[str],
        workspace: Path,
    ) -> _RepoDoctorRepairState | None:
        created = cls._session_ids(root) - known
        if not created:
            return None
        if len(created) != 1:
            raise AgentExecutionError(
                "Repo Doctor created an ambiguous number of repair sessions",
                AgentRunResult(
                    0,
                    "",
                    "",
                    AgentDiagnostics(
                        AgentFailureType.UNKNOWN_AGENT_ERROR,
                        "agent",
                        0,
                    ),
                ),
            )
        return cls._read_repair_state(root, created.pop(), workspace)

    @staticmethod
    def _read_repair_state(
        root: Path,
        session_id: str,
        workspace: Path,
    ) -> _RepoDoctorRepairState:
        if _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Repo Doctor repair session identifier is invalid.")
        directory = (root / "sessions").resolve(strict=False)
        path = directory / f"{session_id}.json"
        if path.is_symlink():
            raise ValueError("Repo Doctor repair session cannot be a symlink.")
        try:
            if (
                not path.is_file()
                or path.stat().st_size > _MAX_REPO_DOCTOR_SESSION_BYTES
            ):
                raise ValueError("Repo Doctor repair session is missing or oversized.")
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Repo Doctor repair session is malformed.") from error
        if not isinstance(value, dict) or set(value) != _REPAIR_SESSION_KEYS:
            raise ValueError("Repo Doctor repair session has an unexpected schema.")
        if value.get("schema_version") != _REPAIR_SESSION_SCHEMA_VERSION:
            raise ValueError(
                "Repo Doctor repair session schema version is unsupported."
            )
        if value.get("session_type") != "repair" or value.get("backend") != "mcp":
            raise ValueError("Repo Doctor session is not an MCP repair session.")
        if value.get("session_id") != session_id:
            raise ValueError("Repo Doctor repair session ID is mismatched.")
        target_value = value.get("target_repository")
        phase = value.get("phase")
        if not isinstance(target_value, str) or not isinstance(phase, str):
            raise TypeError(
                "Repo Doctor repair session lifecycle fields are malformed."
            )
        target = Path(target_value)
        if (
            not target.is_absolute()
            or target.resolve(strict=False) != target
            or not _same_path(target, workspace)
        ):
            raise ValueError(
                "Repo Doctor repair session targets a different workspace."
            )
        return _RepoDoctorRepairState(session_id, target_value, phase)

    def _validate_resume_handle(self, handle: AgentResumeHandle) -> Path | None:
        if handle.adapter != self.info.name:
            raise ValueError("Resume handle belongs to a different adapter.")
        if handle.reason != "approval_required":
            raise ValueError("Repo Doctor resume reason is invalid.")
        if _SESSION_ID.fullmatch(handle.session_id) is None:
            raise ValueError("Repo Doctor repair session identifier is invalid.")
        if set(handle.metadata) - {"scaffold"}:
            raise ValueError("Repo Doctor resume handle has unexpected metadata.")
        scaffold = handle.metadata.get("scaffold")
        if scaffold is None:
            return None
        if scaffold != "requirements.txt":
            raise ValueError(
                "Repo Doctor resume handle has an invalid scaffold marker."
            )
        return Path(scaffold)

    @classmethod
    def _cleanup_after_terminal(cls, workspace: Path, scaffold: Path | None) -> None:
        git_directory = workspace / ".git"
        if git_directory.is_dir():
            shutil.rmtree(git_directory, onerror=cls._remove_readonly)
        if scaffold is not None:
            path = scaffold if scaffold.is_absolute() else workspace / scaffold
            path.unlink(missing_ok=True)

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
        return (
            result.stdout if result.returncode == 0 and result.stdout.strip() else None
        )

    @staticmethod
    def _read_repair_report(path: Path) -> dict | None:
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
            raise ValueError(
                f"AgentLab workspace does not exist: {workspace}"
            ) from error
        temporary_root = Path(tempfile.gettempdir()).resolve()
        marker = resolved / WORKSPACE_MARKER
        if (
            resolved.parent != temporary_root
            or not resolved.name.startswith("agentlab_")
            or not marker.is_file()
            or marker.is_symlink()
        ):
            raise ValueError(
                "Repo Doctor may only run in a temporary workspace created by AgentLab."
            )
        return resolved

    @staticmethod
    def _ensure_python_manifest(workspace: Path) -> Path | None:
        if any((workspace / name).exists() for name in _PYTHON_MANIFESTS):
            return None
        has_python_tests = (
            any(workspace.glob("test_*.py")) or (workspace / "tests").is_dir()
        )
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
                raise RuntimeError(
                    "Git is required by Repo Doctor fix mode."
                ) from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                message = "Could not create the temporary Git baseline"
                if detail:
                    message += f": {detail}"
                raise RuntimeError(message) from error


def _configured_repo_doctor_project(*, platform: str | None = None) -> Path:
    active_platform = platform or ("windows" if os.name == "nt" else "posix")
    configured = os.environ.get(REPO_DOCTOR_PROJECT_ENV)
    explicit = bool(configured)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise ValueError(f"{REPO_DOCTOR_PROJECT_ENV} must be an absolute path.")
    elif active_platform == "windows" and Path(r"D:\repo-doctor").is_dir():
        candidate = Path(r"D:\repo-doctor")
    else:
        raise ValueError(
            f"{REPO_DOCTOR_PROJECT_ENV} must identify the Repo Doctor checkout."
        )
    try:
        project = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("Configured Repo Doctor checkout does not exist.") from error
    if explicit and (candidate != project or candidate.is_symlink()):
        raise ValueError(
            f"{REPO_DOCTOR_PROJECT_ENV} must be a canonical, non-symlink path."
        )
    if not project.is_dir():
        raise ValueError("Configured Repo Doctor checkout must be a real directory.")
    if not (project / "repo_doctor" / "cli.py").is_file():
        raise ValueError(
            "Configured checkout does not contain Repo Doctor production CLI."
        )
    return project


def _checkout_interpreter(project: Path, *, platform: str | None = None) -> Path:
    active_platform = platform or ("windows" if os.name == "nt" else "posix")
    if active_platform == "windows":
        candidate = project / ".venv" / "Scripts" / "python.exe"
    elif active_platform == "posix":
        candidate = project / ".venv" / "bin" / "python"
    else:
        raise ValueError("Unsupported Repo Doctor launch platform.")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("Repo Doctor checkout-local Python interpreter was not found.")
    return candidate.resolve(strict=True)


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve(strict=False))) == os.path.normcase(
        str(right.resolve(strict=False))
    )

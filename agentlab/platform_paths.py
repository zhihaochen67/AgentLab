"""Dependency-neutral resolution of AgentLab platform paths."""

from __future__ import annotations

import os
from pathlib import Path

STATE_ROOT_ENV = "AGENTLAB_STATE_ROOT"


class StateRootConfigurationError(ValueError):
    """The configured AgentLab state root is invalid."""


def default_state_root() -> Path:
    """Return a cross-platform state directory outside evaluation workspaces."""
    configured = os.environ.get(STATE_ROOT_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise StateRootConfigurationError(
                f"{STATE_ROOT_ENV} must be an absolute path."
            )
        return path.resolve(strict=False)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return (Path(base) / "AgentLab" / "State").resolve(strict=False)
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return (Path(xdg).expanduser() / "agentlab").resolve(strict=False)
    return (Path.home() / ".local" / "state" / "agentlab").resolve(strict=False)


def repo_doctor_state_root(root: Path | None = None) -> Path:
    """Return the validated state root reserved for Repo Doctor."""
    candidate = (root or default_state_root()).expanduser()
    if not candidate.is_absolute():
        raise StateRootConfigurationError("AgentLab state root must be absolute.")
    resolved = candidate.resolve(strict=False)
    if resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise StateRootConfigurationError(
            "AgentLab state root must be a real directory."
        )
    return resolved / "repo-doctor"

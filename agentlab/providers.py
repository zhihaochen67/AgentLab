"""Shared, provider-agnostic configuration validation helpers."""

from __future__ import annotations

_MIN_API_KEY_LENGTH = 16
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


def is_placeholder_text(value: str) -> bool:
    """Return True when a value reads like an obvious placeholder."""
    normalized = value.casefold().replace("_", " ").replace("-", " ")
    return any(placeholder in normalized for placeholder in _API_KEY_PLACEHOLDERS)


def is_plausible_api_key(value: str | None) -> bool:
    """Reject obviously invalid API keys without assuming a provider format."""
    if value is None:
        return False
    candidate = value.strip()
    if not candidate or not candidate.isascii():
        return False
    if len(candidate) < _MIN_API_KEY_LENGTH:
        return False
    return not is_placeholder_text(candidate)

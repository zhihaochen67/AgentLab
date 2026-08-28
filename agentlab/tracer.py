"""In-memory structured execution tracing."""

from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_TRACE_TEXT = 2_000
REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|credential|password|secret|token)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_API_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


@dataclass(frozen=True)
class TraceEvent:
    """One ordered, sanitized event from an evaluation run."""

    run_id: str
    sequence: int
    event_type: str
    timestamp: str
    data: dict[str, Any]


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact_text(value: object) -> str:
    """Redact common credentials and secret environment values from text."""
    text = str(value)
    environment_secrets = {
        secret
        for name, secret in os.environ.items()
        if _is_sensitive_key(name) and len(secret) >= 4
    }
    for secret in sorted(environment_secrets, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    text = _BEARER_TOKEN.sub("Bearer " + REDACTED, text)
    text = _API_TOKEN.sub(REDACTED, text)
    text = _CREDENTIAL_ASSIGNMENT.sub(
        lambda match: match.group(1) + match.group(2) + REDACTED, text
    )
    return text


def summarize_text(value: object, limit: int = MAX_TRACE_TEXT) -> str:
    """Return a redacted, bounded text summary suitable for trace storage."""
    text = redact_text(value)
    if len(text) <= limit:
        return text
    suffix = f"... [truncated {len(text) - limit} chars]"
    return text[: max(0, limit - len(suffix))] + suffix


def _sanitize(value: Any, *, key: object | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item, key=item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return summarize_text(value)
    if isinstance(value, str):
        return summarize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return summarize_text(value)


def sanitize_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-compatible trace data with secrets redacted."""
    return {str(key): _sanitize(value, key=key) for key, value in data.items()}


class Tracer:
    """Generate a run id and maintain a stable in-memory event sequence."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        events: tuple[TraceEvent, ...] = (),
        elapsed_offset: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        if any(event.run_id != self.run_id for event in events):
            raise ValueError("Every restored trace event must match run_id.")
        if tuple(event.sequence for event in events) != tuple(
            range(1, len(events) + 1)
        ):
            raise ValueError(
                "Restored trace events must have contiguous sequence numbers."
            )
        if elapsed_offset < 0:
            raise ValueError("elapsed_offset must be non-negative.")
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._started_at = self._clock()
        self._elapsed_offset = elapsed_offset
        self._sequence = len(events)
        self._events = list(events)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return an immutable snapshot of events emitted so far."""
        return tuple(self._events)

    def emit(self, event_type: str, **data: Any) -> TraceEvent:
        """Append a sanitized event with the next sequence number."""
        self._sequence += 1
        event = TraceEvent(
            run_id=self.run_id,
            sequence=self._sequence,
            event_type=event_type,
            timestamp=self._wall_clock().astimezone(timezone.utc).isoformat(),
            data=sanitize_data(data),
        )
        self._events.append(event)
        return event

    def start_timer(self) -> float:
        """Start a monotonic phase timer."""
        return self._clock()

    def elapsed_since(self, started_at: float) -> float:
        """Return elapsed monotonic seconds for a phase."""
        return max(0.0, self._clock() - started_at)

    def total_elapsed(self) -> float:
        """Return elapsed monotonic seconds since tracer creation."""
        return self._elapsed_offset + self.elapsed_since(self._started_at)

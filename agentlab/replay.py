"""Pure, side-effect-free replay of persisted historical trace events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from agentlab.tracer import TraceEvent, sanitize_data, summarize_text


@dataclass(frozen=True)
class ReplayTrace:
    """Validated historical events and non-fatal integrity warnings."""

    events: tuple[TraceEvent, ...]
    warnings: tuple[str, ...] = ()

    @property
    def total_steps(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class ReplayState:
    """Immutable cursor over a prepared historical replay."""

    trace: ReplayTrace
    index: int = 0

    def __post_init__(self) -> None:
        maximum = max(0, self.trace.total_steps - 1)
        object.__setattr__(self, "index", min(max(0, self.index), maximum))

    @classmethod
    def from_events(cls, events: Iterable[TraceEvent]) -> ReplayState:
        return cls(prepare_replay(events))

    @property
    def total_steps(self) -> int:
        return self.trace.total_steps

    @property
    def current_step(self) -> int:
        return self.index + 1 if self.total_steps else 0

    @property
    def current_event(self) -> TraceEvent | None:
        return self.trace.events[self.index] if self.total_steps else None

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return not self.total_steps or self.index == self.total_steps - 1

    def first(self) -> ReplayState:
        return ReplayState(self.trace, 0)

    def previous(self) -> ReplayState:
        return ReplayState(self.trace, self.index - 1)

    def next(self) -> ReplayState:
        return ReplayState(self.trace, self.index + 1)

    def last(self) -> ReplayState:
        return ReplayState(self.trace, self.total_steps - 1)


@dataclass(frozen=True)
class ReplayEventView:
    """Sanitized, event-aware presentation data for one replay step."""

    sequence: int
    event_type: str
    timestamp: str
    status: str
    elapsed: float | None
    highlights: dict[str, Any]
    data: dict[str, Any]


def prepare_replay(events: Iterable[TraceEvent]) -> ReplayTrace:
    """Return strictly ordered valid events without failing on corrupt sequences."""
    warnings: list[str] = []
    valid: list[tuple[int, TraceEvent]] = []
    for position, event in enumerate(events, 1):
        sequence = getattr(event, "sequence", None)
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            warnings.append(
                f"Skipped event at input position {position}: invalid sequence {sequence!r}."
            )
            continue
        valid.append((position, event))

    valid.sort(key=lambda item: (item[1].sequence, item[0]))
    ordered: list[TraceEvent] = []
    seen: set[int] = set()
    for _, event in valid:
        if event.sequence in seen:
            warnings.append(f"Skipped duplicate sequence {event.sequence}.")
            continue
        seen.add(event.sequence)
        ordered.append(event)

    expected = 1
    for event in ordered:
        if event.sequence > expected:
            warnings.append(_missing_sequence_warning(expected, event.sequence - 1))
        expected = event.sequence + 1
    return ReplayTrace(tuple(ordered), tuple(warnings))


def _missing_sequence_warning(first: int, last: int) -> str:
    if first == last:
        return f"Missing sequence {first}."
    return f"Missing sequence range {first}-{last}."


def _data_mapping(event: TraceEvent) -> Mapping[str, Any]:
    return event.data if isinstance(event.data, Mapping) else {}


def event_status(event: TraceEvent) -> str:
    """Derive the historical status for one event without executing anything."""
    event_type = str(event.event_type)
    data = _data_mapping(event)
    if event_type.startswith("pytest_") and event_type.endswith("_end"):
        if data.get("status") == "error":
            return "ERROR"
        return "PASS" if data.get("passed") else "FAIL"
    if event_type == "workspace_verification_end":
        if data.get("status") == "error":
            return "ERROR"
        return "PASS" if data.get("passed") else "FAIL"
    if event_type in {"agent_suspended", "run_suspended"}:
        return "WAITING"
    if event_type == "agent_end":
        return "OK" if data.get("status") == "ok" else "ERROR"
    if event_type == "run_end":
        final_status = str(data.get("final_status", "")).upper()
        if final_status in {"PASS", "FAIL"}:
            return final_status
        return "PASS" if data.get("passed") else "FAIL"
    if event_type == "error":
        return "ERROR"
    return ""


def event_elapsed(event: TraceEvent) -> float | None:
    """Return a historical event duration when it is numeric."""
    elapsed = _data_mapping(event).get("elapsed_time")
    numeric = isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
    return float(elapsed) if numeric else None


def format_replay_event(event: TraceEvent) -> ReplayEventView:
    """Create a sanitized, type-aware view of one historical event."""
    raw_data = _data_mapping(event)
    data = sanitize_data(raw_data)
    event_type = summarize_text(event.event_type)
    keys: tuple[str, ...]
    if event_type.startswith("pytest_") and event_type.endswith("_end"):
        keys = ("passed", "returncode", "stdout", "stderr")
    elif event_type == "workspace_verification_end":
        keys = (
            "passed",
            "modified_files",
            "missing_expected_files",
            "unexpected_files",
        )
    elif event_type == "agent_end":
        keys = ("adapter", "status", "returncode", "stdout", "stderr")
    elif event_type == "error":
        keys = ("phase", "error_type", "message")
    elif event_type == "run_end":
        keys = (
            "final_status",
            "failure_reason",
            "passed",
            "tests_before_passed",
            "tests_after_passed",
            "workspace_changes_passed",
            "elapsed_time",
        )
    else:
        keys = tuple(data)
    highlights = {key: data[key] for key in keys if key in data}
    if not isinstance(event.data, Mapping):
        data = {"warning": "Historical event data was not an object."}
    return ReplayEventView(
        sequence=event.sequence,
        event_type=event_type,
        timestamp=summarize_text(event.timestamp),
        status=event_status(event),
        elapsed=event_elapsed(event),
        highlights=highlights,
        data=data,
    )

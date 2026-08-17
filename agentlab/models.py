from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlab.tracer import TraceEvent


@dataclass
class EvalCase:
    id: str
    repository: str
    task: str
    expected: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    tests_before_passed: bool
    tests_after_passed: bool
    error: str | None = None
    run_id: str = ""
    trace: tuple[TraceEvent, ...] = field(default_factory=tuple)

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
    experiment_id: str | None = None
    trial_index: int | None = None


@dataclass(frozen=True)
class Experiment:
    """Persisted metadata for one repeated-trial evaluation experiment."""

    experiment_id: str
    label: str
    dataset: str
    adapter: str
    model: str | None
    trials_per_case: int
    total_cases: int
    total_runs: int
    started_at: str
    finished_at: str | None
    status: str


@dataclass(frozen=True)
class CaseExperimentMetrics:
    """Aggregate results for one case within an experiment."""

    case_id: str
    total_runs: int
    passed_runs: int
    failed_runs: int
    success_rate: float
    average_latency: float


@dataclass(frozen=True)
class ExperimentMetrics:
    """Aggregate results calculated from persisted experiment runs."""

    experiment_id: str
    total_runs: int
    passed_runs: int
    failed_runs: int
    success_rate: float
    average_latency: float
    per_case: tuple[CaseExperimentMetrics, ...]
    failure_types: tuple[tuple[str, int], ...]

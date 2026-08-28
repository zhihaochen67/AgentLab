from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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
class EvaluationSuspended:
    """Non-final control result for a persisted, resumable evaluation."""

    execution_id: str
    run_id: str
    case_id: str
    status: str = "WAITING_FOR_APPROVAL"


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
    agent_version: str | None = None
    prompt_variant: str | None = None
    notes: str | None = None


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
class EvaluatorExperimentMetrics:
    """Aggregate evaluator outcomes for one evaluator within an experiment."""

    evaluator: str
    total_outcomes: int
    evaluated_runs: int
    passed_outcomes: int
    failed_outcomes: int
    error_outcomes: int
    verdict_outcomes: int
    pass_rate: float
    score_count: int
    average_score: float | None
    min_score: float | None
    max_score: float | None
    coverage_rate: float


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
    evaluator_metrics: tuple[EvaluatorExperimentMetrics, ...] = ()


ComparisonChange = Literal["improved", "regressed", "unchanged"]


@dataclass(frozen=True)
class ExperimentComparisonSummary:
    """One side of an experiment comparison."""

    experiment: Experiment
    total_runs: int
    passed_runs: int
    failed_runs: int
    success_rate: float
    average_latency: float


@dataclass(frozen=True)
class ComparisonCompatibility:
    """Compatibility verdict and the evidence used to produce it."""

    is_equivalent: bool
    dataset_matches: bool
    case_sets_match: bool
    trials_per_case_matches: bool
    baseline_case_set_complete: bool
    candidate_case_set_complete: bool
    common_case_ids: tuple[str, ...]
    baseline_only_case_ids: tuple[str, ...]
    candidate_only_case_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    evaluator_sets_match: bool = True
    common_evaluators: tuple[str, ...] = ()
    baseline_only_evaluators: tuple[str, ...] = ()
    candidate_only_evaluators: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseExperimentComparison:
    """Success-rate comparison for one case observed on both sides."""

    case_id: str
    baseline_runs: int
    baseline_passes: int
    baseline_success_rate: float
    candidate_runs: int
    candidate_passes: int
    candidate_success_rate: float
    delta: float
    change: ComparisonChange


@dataclass(frozen=True)
class FailureTypeComparison:
    """Failure taxonomy count comparison for one failure type."""

    failure_type: str
    baseline_count: int
    candidate_count: int
    delta: int


@dataclass(frozen=True)
class EvaluatorMetricsComparison:
    """Evaluator-level metrics comparison for one shared evaluator name.

    Note: the same evaluator name is the current comparability boundary;
    differing models/rubrics behind one name are not detected yet.
    """

    evaluator: str
    baseline_total_outcomes: int
    candidate_total_outcomes: int
    baseline_evaluated_runs: int
    candidate_evaluated_runs: int
    baseline_coverage_rate: float
    candidate_coverage_rate: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float
    baseline_average_score: float | None
    candidate_average_score: float | None
    average_score_delta: float | None
    baseline_score_count: int
    candidate_score_count: int
    baseline_error_outcomes: int
    candidate_error_outcomes: int


@dataclass(frozen=True)
class ExperimentComparison:
    """Pure comparison result built from two persisted experiments."""

    baseline: ExperimentComparisonSummary
    candidate: ExperimentComparisonSummary
    success_rate_delta: float
    latency_delta: float
    compatibility: ComparisonCompatibility
    per_case: tuple[CaseExperimentComparison, ...]
    failure_types: tuple[FailureTypeComparison, ...]
    evaluator_metrics: tuple[EvaluatorMetricsComparison, ...] = ()

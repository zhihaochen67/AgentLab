"""Repeated-trial experiment orchestration and provider preflight."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from agentlab.adapters import AgentAdapter, AgentPreflightError
from agentlab.dataset import validate_dataset
from agentlab.evaluators import Evaluator
from agentlab.models import EvalCase, EvalResult, Experiment, ExperimentMetrics
from agentlab.runner import evaluate_case
from agentlab.storage import RunStorage

CaseExecutor = Callable[[EvalCase, AgentAdapter], EvalResult]
DatasetValidator = Callable[[Sequence[EvalCase]], None]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ExperimentExecution:
    """Completed experiment metadata and aggregates."""

    experiment: Experiment
    metrics: ExperimentMetrics


class ExperimentAbortedError(RuntimeError):
    """An experiment was persisted as aborted before it could complete."""

    def __init__(self, experiment_id: str, message: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(message)


class ExperimentPreflightError(ExperimentAbortedError):
    """Provider preflight aborted an experiment before any run executed."""

    def __init__(
        self,
        experiment_id: str,
        missing_variables: tuple[str, ...],
        invalid_variables: tuple[str, ...] = (),
    ) -> None:
        self.missing_variables = missing_variables
        self.invalid_variables = invalid_variables
        if invalid_variables:
            message = ", ".join(
                f"{name} appears invalid" for name in invalid_variables
            )
        else:
            names = ", ".join(missing_variables)
            message = f"Missing provider configuration: {names}"
        super().__init__(experiment_id, message)


def new_experiment_id() -> str:
    """Return a globally unique experiment identifier."""
    return str(uuid.uuid4())


def select_experiment_cases(
    cases: Sequence[EvalCase],
    case_ids: Sequence[str] | None,
) -> list[EvalCase]:
    """Filter cases in dataset order and reject unknown identifiers."""
    if not case_ids:
        return list(cases)
    requested = set(case_ids)
    known = {case.id for case in cases}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown case id(s): {', '.join(unknown)}")
    return [case for case in cases if case.id in requested]


def run_experiment(
    *,
    cases: Sequence[EvalCase],
    dataset: str,
    storage: RunStorage,
    adapter: AgentAdapter,
    trials_per_case: int,
    label: str | None = None,
    agent_version: str | None = None,
    prompt_variant: str | None = None,
    notes: str | None = None,
    case_ids: Sequence[str] | None = None,
    evaluator: Evaluator | None = None,
    case_executor: CaseExecutor | None = None,
    validator: DatasetValidator = validate_dataset,
    experiment_id: str | None = None,
    clock: Clock | None = None,
) -> ExperimentExecution:
    """Execute selected cases repeatedly while preserving every completed trial."""
    if trials_per_case < 1:
        raise ValueError("trials_per_case must be at least 1.")
    if evaluator is not None and case_executor is not None:
        raise ValueError(
            "Provide either 'evaluator' or 'case_executor', not both."
        )
    selected = select_experiment_cases(cases, case_ids)
    if not selected:
        raise ValueError("An experiment requires at least one selected case.")

    active_clock = clock or (lambda: datetime.now(timezone.utc))
    identifier = experiment_id or new_experiment_id()
    started_at = _timestamp(active_clock)
    experiment_label = (label or Path(dataset).stem).strip()
    if not experiment_label:
        raise ValueError("Experiment label must be non-empty.")
    recorded_agent_version = _optional_metadata(agent_version)
    recorded_prompt_variant = _optional_metadata(prompt_variant)
    recorded_notes = _optional_metadata(notes)
    adapter_metadata = adapter.trace_metadata()
    recorded_agent_version = _reconcile_metadata(
        "agent_version",
        recorded_agent_version,
        _optional_metadata(adapter_metadata.get("agent_version")),
    )
    recorded_prompt_variant = _reconcile_metadata(
        "prompt_variant",
        recorded_prompt_variant,
        _optional_metadata(adapter_metadata.get("prompt_variant")),
    )

    try:
        preflight = adapter.preflight()
    except AgentPreflightError as error:
        aborted = Experiment(
            experiment_id=identifier,
            label=experiment_label,
            dataset=dataset,
            adapter=type(adapter).__name__,
            model=None,
            trials_per_case=trials_per_case,
            total_cases=len(selected),
            total_runs=0,
            started_at=started_at,
            finished_at=_timestamp(active_clock),
            status="aborted",
            agent_version=recorded_agent_version,
            prompt_variant=recorded_prompt_variant,
            notes=recorded_notes,
        )
        storage.create_experiment(aborted)
        raise ExperimentPreflightError(
            identifier,
            error.missing_variables,
            error.invalid_variables,
        ) from error

    running = Experiment(
        experiment_id=identifier,
        label=experiment_label,
        dataset=dataset,
        adapter=type(adapter).__name__,
        model=preflight.model,
        trials_per_case=trials_per_case,
        total_cases=len(selected),
        total_runs=0,
        started_at=started_at,
        finished_at=None,
        status="running",
        agent_version=recorded_agent_version,
        prompt_variant=recorded_prompt_variant,
        notes=recorded_notes,
    )
    storage.create_experiment(running)
    execute = case_executor or (
        lambda case, adapter: _evaluate(case, adapter, evaluator=evaluator)
    )
    try:
        validator(selected)
        for case in selected:
            for trial_index in range(1, trials_per_case + 1):
                result = execute(case, adapter)
                associated = replace(
                    result,
                    experiment_id=identifier,
                    trial_index=trial_index,
                )
                storage.save_run(associated, dataset)
    except Exception as error:
        storage.finish_experiment(identifier, "aborted", _timestamp(active_clock))
        raise ExperimentAbortedError(identifier, str(error)) from error

    metrics = storage.get_experiment_metrics(identifier)
    status = "completed" if metrics.failed_runs == 0 else "completed_with_failures"
    finished = storage.finish_experiment(identifier, status, _timestamp(active_clock))
    return ExperimentExecution(finished, metrics)


def _evaluate(
    case: EvalCase,
    adapter: AgentAdapter,
    *,
    evaluator: Evaluator | None = None,
) -> EvalResult:
    return evaluate_case(case, adapter=adapter, evaluator=evaluator)


def _timestamp(clock: Clock) -> str:
    return clock().astimezone(timezone.utc).isoformat()


def _optional_metadata(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _reconcile_metadata(
    name: str,
    requested: str | None,
    actual: str | None,
) -> str | None:
    if requested is not None and actual is not None and requested != actual:
        raise ValueError(
            f"Experiment {name} {requested!r} does not match adapter {name} {actual!r}."
        )
    return actual or requested

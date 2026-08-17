from dataclasses import dataclass, field


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

"""Provider-agnostic LLM-as-Judge evaluator foundation.

The judge is any callable that maps a prompt to raw response text, so tests
and integrations can inject a mock or a provider-specific wrapper without
AgentLab depending on a specific LLM vendor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agentlab.evaluators.base import EvaluationOutcome, Evaluator
from agentlab.models import EvalCase
from agentlab.tracer import redact_text

DEFAULT_MAX_EVIDENCE_FILES = 20
DEFAULT_MAX_EVIDENCE_BYTES = 200_000
DEFAULT_MAX_FILE_BYTES = 20_000
_MAX_RAW_RESPONSE_CHARS = 100_000
_BINARY_SNIFF_BYTES = 8_192

_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".nox",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "site-packages",
        "venv",
    }
)

_SKIPPED_SUFFIXES = frozenset(
    {
        ".7z",
        ".avi",
        ".bin",
        ".bmp",
        ".bz2",
        ".db",
        ".dll",
        ".dylib",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".lib",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".pdf",
        ".png",
        ".pyc",
        ".pyo",
        ".so",
        ".sqlite",
        ".tar",
        ".tiff",
        ".webp",
        ".whl",
        ".woff",
        ".woff2",
        ".xz",
        ".zip",
    }
)

_PRIORITY_SUFFIXES = frozenset(
    {
        ".c",
        ".cfg",
        ".cpp",
        ".css",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".js",
        ".jsx",
        ".json",
        ".md",
        ".ps1",
        ".py",
        ".rst",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class JudgeResponseError(ValueError):
    """A judge response that could not be parsed or validated."""


class JudgeCallable(Protocol):
    """Provider-agnostic judge: prompt in, raw response text out."""

    def __call__(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class JudgeVerdict:
    """Validated structured verdict produced by a judge."""

    passed: bool
    score: float
    feedback: str
    model: str | None = None

    @classmethod
    def parse(cls, raw_response: str) -> JudgeVerdict:
        """Parse and validate a raw judge response.

        Raises JudgeResponseError for invalid JSON, non-object payloads,
        missing required fields, wrong field types, or out-of-range scores.
        Extra unknown fields are ignored for forward compatibility.
        """
        if len(raw_response) > _MAX_RAW_RESPONSE_CHARS:
            raise JudgeResponseError(
                f"judge response exceeds {_MAX_RAW_RESPONSE_CHARS} characters"
            )
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise JudgeResponseError(
                f"judge returned invalid JSON: {error.msg} (line {error.lineno})"
            ) from error
        if not isinstance(payload, dict):
            raise JudgeResponseError("judge response must be a JSON object")
        return cls._from_payload(payload)

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> JudgeVerdict:
        score = payload.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeResponseError(
                "judge response must contain a numeric 'score'"
            )
        numeric_score = float(score)
        if not 0.0 <= numeric_score <= 1.0:
            raise JudgeResponseError(
                f"judge 'score' must be within [0.0, 1.0], got {score}"
            )

        passed = payload.get("passed")
        if not isinstance(passed, bool):
            raise JudgeResponseError(
                "judge response must contain a boolean 'passed'"
            )

        feedback = payload.get("feedback")
        if not isinstance(feedback, str) or not feedback.strip():
            raise JudgeResponseError(
                "judge response must contain non-empty 'feedback' text"
            )

        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise JudgeResponseError(
                "judge 'model' must be a string when present"
            )

        return cls(
            passed=passed,
            score=numeric_score,
            feedback=feedback,
            model=model,
        )


@dataclass(frozen=True)
class EvidenceFile:
    """One bounded, redacted text file collected from a workspace."""

    relative_path: str
    content: str
    truncated: bool = False


@dataclass(frozen=True)
class EvidenceBundle:
    """Bounded workspace evidence summary handed to the judge."""

    files: tuple[EvidenceFile, ...]
    truncated: bool
    skipped_binary: int = 0
    skipped_unreadable: int = 0

    @property
    def total_bytes(self) -> int:
        """Total character count of collected evidence."""
        return sum(len(item.content) for item in self.files)


def _looks_binary(head: bytes) -> bool:
    return b"\x00" in head


def collect_evidence(
    workspace: Path,
    *,
    max_files: int = DEFAULT_MAX_EVIDENCE_FILES,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> EvidenceBundle:
    """Collect bounded, redacted text evidence from a workspace.

    Skips hidden entries, well-known non-source directories (.git, .venv,
    caches, build output, ...) and binary files. Reads at most
    *max_file_bytes* per file and *max_bytes* in total; any evidence that
    cannot fit is dropped and reflected by ``truncated``. Text is decoded
    leniently and passed through AgentLab's secret redaction so environment
    secrets never reach the judge prompt.
    """
    for name, value in (
        ("max_files", max_files),
        ("max_bytes", max_bytes),
        ("max_file_bytes", max_file_bytes),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    workspace = Path(workspace)
    if not workspace.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace}")

    candidates: list[Path] = []
    for root, directories, files in os.walk(workspace):
        directories[:] = sorted(
            directory
            for directory in directories
            if not directory.startswith(".")
            and directory not in _SKIPPED_DIRECTORIES
        )
        for filename in files:
            if filename.startswith("."):
                continue
            candidates.append(Path(root) / filename)

    candidates.sort(
        key=lambda path: (
            _priority(path.suffix.lower()),
            str(path.relative_to(workspace)).casefold(),
        )
    )

    collected: list[EvidenceFile] = []
    total_bytes = 0
    truncated = False
    skipped_binary = 0
    skipped_unreadable = 0

    for path in candidates:
        if len(collected) >= max_files:
            truncated = True
            break
        try:
            with path.open("rb") as handle:
                raw = handle.read(max_file_bytes + 1)
        except OSError:
            skipped_unreadable += 1
            continue
        if path.suffix.lower() in _SKIPPED_SUFFIXES or _looks_binary(
            raw[:_BINARY_SNIFF_BYTES]
        ):
            skipped_binary += 1
            continue

        file_truncated = len(raw) > max_file_bytes
        decoded = redact_text(raw[:max_file_bytes].decode("utf-8", errors="replace"))
        remaining = max_bytes - total_bytes
        if remaining <= 0:
            truncated = True
            break
        budget_cut = len(decoded) > remaining
        if budget_cut:
            decoded = decoded[:remaining]
            truncated = True
        if file_truncated:
            truncated = True

        total_bytes += len(decoded)
        collected.append(
            EvidenceFile(
                relative_path=str(path.relative_to(workspace)),
                content=decoded,
                truncated=file_truncated or budget_cut,
            )
        )

    return EvidenceBundle(
        files=tuple(collected),
        truncated=truncated,
        skipped_binary=skipped_binary,
        skipped_unreadable=skipped_unreadable,
    )


def _priority(suffix: str) -> int:
    return 0 if suffix in _PRIORITY_SUFFIXES else 1


def build_judge_prompt(case: EvalCase, bundle: EvidenceBundle) -> str:
    """Build the prompt handed to the judge, including the evidence blocks."""
    blocks: list[str] = []
    for item in bundle.files:
        marker = " [truncated]" if item.truncated else ""
        blocks.append(
            f"<file path={item.relative_path!r}{marker}>\n"
            f"{item.content}\n"
            "</file>"
        )
    evidence = "\n\n".join(blocks) if blocks else "(no readable evidence files)"
    if bundle.truncated:
        evidence += "\n(evidence truncated by evaluator size limits)"

    return (
        "You are an evaluation judge for a coding agent. Decide whether the "
        "agent completed the task based on the workspace evidence below.\n\n"
        f"Task: {case.task}\n"
        f"Case: {case.id}\n\n"
        "Workspace evidence:\n"
        f"{evidence}\n\n"
        "Respond with a single JSON object and nothing else, using exactly "
        "this shape:\n"
        '{"passed": true|false, "score": 0.0-1.0, '
        '"feedback": "short justification", "model": "optional model name"}'
    )


class LLMJudgeEvaluator(Evaluator):
    """Evaluate a workspace with a provider-agnostic LLM judge.

    The judge callable receives the full prompt (task plus bounded workspace
    evidence) and returns raw response text. The response must be a JSON
    object with ``passed``, ``score`` and ``feedback`` fields; anything else
    raises JudgeResponseError, which the runner records as a diagnostic
    evaluation error and a failed run.
    """

    def __init__(
        self,
        judge: JudgeCallable,
        *,
        model: str | None = None,
        judge_name: str | None = None,
        max_evidence_files: int = DEFAULT_MAX_EVIDENCE_FILES,
        max_evidence_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._judge = judge
        self.model = model
        self.judge_name = judge_name or type(judge).__name__
        self.max_evidence_files = max_evidence_files
        self.max_evidence_bytes = max_evidence_bytes
        self.max_file_bytes = max_file_bytes

    def evaluate(
        self,
        workspace: Path,
        case: EvalCase,
    ) -> EvaluationOutcome:
        """Collect evidence, call the judge and return a structured verdict."""
        bundle = collect_evidence(
            workspace,
            max_files=self.max_evidence_files,
            max_bytes=self.max_evidence_bytes,
            max_file_bytes=self.max_file_bytes,
        )
        prompt = build_judge_prompt(case, bundle)
        verdict = JudgeVerdict.parse(self._judge(prompt))

        metadata: dict[str, Any] = {
            "evaluator": "llm_judge",
            "judge": self.judge_name,
            "evidence_files": len(bundle.files),
            "evidence_bytes": bundle.total_bytes,
            "evidence_truncated": bundle.truncated,
            "evidence_skipped_binary": bundle.skipped_binary,
        }
        model = self.model or verdict.model
        if model is not None:
            metadata["model"] = model
        if bundle.skipped_unreadable:
            metadata["evidence_skipped_unreadable"] = bundle.skipped_unreadable

        return EvaluationOutcome(
            passed=verdict.passed,
            score=verdict.score,
            feedback=verdict.feedback,
            metadata=metadata,
        )

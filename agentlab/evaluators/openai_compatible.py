"""OpenAI-compatible chat-completions transport for the LLM judge.

This module only moves bytes: it builds the HTTP request, extracts
``choices[0].message.content`` from a valid provider response, and raises
safe, diagnostic errors otherwise. It never parses judge verdict fields
(passed/score/feedback) — that is JudgeVerdict's responsibility — and it
never exposes API keys, authorization headers, prompts, or raw response
bodies in error messages.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import httpx

from agentlab.providers import is_placeholder_text, is_plausible_api_key

DEFAULT_TIMEOUT_SECONDS = 60.0

API_KEY_ENV = "AGENTLAB_JUDGE_API_KEY"
BASE_URL_ENV = "AGENTLAB_JUDGE_BASE_URL"
MODEL_ENV = "AGENTLAB_JUDGE_MODEL"
TIMEOUT_ENV = "AGENTLAB_JUDGE_TIMEOUT_SECONDS"


class JudgeConfigurationError(ValueError):
    """Judge provider configuration is missing or invalid."""


class JudgeProviderError(RuntimeError):
    """A judge provider request failed; messages never expose secrets."""


@dataclass(frozen=True)
class JudgeProviderConfig:
    """Validated OpenAI-compatible judge provider configuration."""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> JudgeProviderConfig:
        """Build a validated configuration from environment variables.

        Reads only the AGENTLAB_JUDGE_* variables; the agent-under-test
        provider (REPO_DOCTOR_*) is never consulted.
        """
        env = os.environ if environ is None else environ

        api_key = env.get(API_KEY_ENV)
        if not is_plausible_api_key(api_key):
            raise JudgeConfigurationError(f"{API_KEY_ENV} appears invalid")

        base_url = cls._validate_base_url(env.get(BASE_URL_ENV, ""))

        model = env.get(MODEL_ENV, "").strip()
        if not model:
            raise JudgeConfigurationError(f"{MODEL_ENV} must be set")
        if is_placeholder_text(model):
            raise JudgeConfigurationError(f"{MODEL_ENV} appears invalid")

        raw_timeout = env.get(TIMEOUT_ENV, "").strip()
        if not raw_timeout:
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        else:
            try:
                timeout_seconds = float(raw_timeout)
            except ValueError as error:
                raise JudgeConfigurationError(
                    f"{TIMEOUT_ENV} must be a positive number"
                ) from error
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise JudgeConfigurationError(
                    f"{TIMEOUT_ENV} must be a positive number"
                )

        return cls(
            api_key=api_key.strip(),
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _validate_base_url(raw: str) -> str:
        """Normalize and validate a base URL without echoing secret parts."""
        raw = raw.strip()
        if not raw:
            raise JudgeConfigurationError(f"{BASE_URL_ENV} must be set")
        try:
            parsed = urlsplit(raw)
        except ValueError as error:
            raise JudgeConfigurationError(
                f"{BASE_URL_ENV} is not a valid URL"
            ) from error
        if parsed.scheme.lower() not in {"http", "https"}:
            raise JudgeConfigurationError(
                f"{BASE_URL_ENV} must use an http or https scheme"
            )
        if not parsed.hostname:
            raise JudgeConfigurationError(f"{BASE_URL_ENV} must include a host")
        if parsed.username or parsed.password:
            raise JudgeConfigurationError(
                f"{BASE_URL_ENV} must not contain embedded credentials"
            )
        path = parsed.path.rstrip("/")
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class OpenAICompatibleJudge:
    """OpenAI-compatible chat-completions transport satisfying JudgeCallable.

    Call it with a prompt and it returns the assistant message content,
    validated only for shape. Construction performs no network I/O; a
    lazily created httpx.Client is used unless one is injected.
    """

    def __init__(
        self,
        config: JudgeProviderConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client

    def __call__(self, prompt: str) -> str:
        owns_client = self._client is None
        client = self._client or httpx.Client(
            timeout=self._config.timeout_seconds,
            follow_redirects=False,
        )
        try:
            return self._request(client, prompt)
        finally:
            if owns_client:
                client.close()

    def _request(self, client: httpx.Client, prompt: str) -> str:
        url = f"{self._config.base_url}/chat/completions"
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        try:
            response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as error:
            raise JudgeProviderError(
                f"Judge provider request timed out "
                f"after {self._config.timeout_seconds:g}s."
            ) from error
        except httpx.HTTPError as error:
            raise JudgeProviderError(
                f"Judge provider request failed: {type(error).__name__}."
            ) from error

        if not 200 <= response.status_code < 300:
            raise JudgeProviderError(
                f"Judge provider returned HTTP {response.status_code}."
            )
        try:
            body = response.json()
        except ValueError as error:
            raise JudgeProviderError(
                "Judge provider returned a non-JSON response."
            ) from error
        if not isinstance(body, dict):
            raise JudgeProviderError(
                "Judge provider response must be a JSON object."
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise JudgeProviderError(
                "Judge provider response is missing choices."
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise JudgeProviderError(
                "Judge provider response has an invalid choice."
            )
        first_message = first_choice.get("message")
        if not isinstance(first_message, dict):
            raise JudgeProviderError(
                "Judge provider response is missing a message."
            )
        content = first_message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise JudgeProviderError(
                "Judge provider response has no message content."
            )
        return content

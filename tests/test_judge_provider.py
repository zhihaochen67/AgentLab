import json

import httpx
import pytest

from agentlab.evaluators import (
    JudgeConfigurationError,
    JudgeProviderConfig,
    JudgeProviderError,
    OpenAICompatibleJudge,
)

_VALID_API_KEY = "sk-judge-1234567890abcdef"
_VALID_BASE_URL = "https://judge.example.com/v1"
_VALID_MODEL = "deepseek-v4-flash"


def valid_env(**overrides: str) -> dict[str, str]:
    env = {
        "AGENTLAB_JUDGE_API_KEY": _VALID_API_KEY,
        "AGENTLAB_JUDGE_BASE_URL": _VALID_BASE_URL,
        "AGENTLAB_JUDGE_MODEL": _VALID_MODEL,
    }
    env.update(overrides)
    return env


def make_judge(
    handler,
    *,
    api_key: str = _VALID_API_KEY,
    base_url: str = "http://judge.test/v1",
    model: str = "judge-model",
    timeout_seconds: float = 5.0,
):
    config = JudgeProviderConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleJudge(config, client=client), config


def success_handler(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"passed": true, "score": 1.0, "feedback": "ok"}',
                        }
                    }
                ]
            },
        )

    return handler


# --- Configuration tests ------------------------------------------------------


def test_valid_config_from_env() -> None:
    config = JudgeProviderConfig.from_env(valid_env())

    assert config.api_key == _VALID_API_KEY
    assert config.base_url == "https://judge.example.com/v1"
    assert config.model == _VALID_MODEL
    assert config.timeout_seconds == 60.0


def test_missing_api_key_rejected() -> None:
    env = valid_env()
    env.pop("AGENTLAB_JUDGE_API_KEY")

    with pytest.raises(JudgeConfigurationError, match="AGENTLAB_JUDGE_API_KEY"):
        JudgeProviderConfig.from_env(env)


def test_placeholder_api_key_rejected() -> None:
    with pytest.raises(JudgeConfigurationError, match="appears invalid"):
        JudgeProviderConfig.from_env(
            valid_env(AGENTLAB_JUDGE_API_KEY="your api key must be replaced")
        )


def test_short_api_key_rejected() -> None:
    with pytest.raises(JudgeConfigurationError, match="appears invalid"):
        JudgeProviderConfig.from_env(valid_env(AGENTLAB_JUDGE_API_KEY="short"))


def test_missing_base_url_rejected() -> None:
    env = valid_env()
    env.pop("AGENTLAB_JUDGE_BASE_URL")

    with pytest.raises(JudgeConfigurationError, match="AGENTLAB_JUDGE_BASE_URL"):
        JudgeProviderConfig.from_env(env)


@pytest.mark.parametrize(
    "raw_url",
    [
        "ftp://judge.example.com/v1",
        "not-a-url",
    ],
)
def test_invalid_base_url_scheme_rejected(raw_url: str) -> None:
    with pytest.raises(JudgeConfigurationError, match="scheme|must be set"):
        JudgeProviderConfig.from_env(valid_env(AGENTLAB_JUDGE_BASE_URL=raw_url))


def test_base_url_without_host_rejected() -> None:
    with pytest.raises(JudgeConfigurationError, match="host"):
        JudgeProviderConfig.from_env(valid_env(AGENTLAB_JUDGE_BASE_URL="https:///v1"))


def test_base_url_with_embedded_credentials_rejected() -> None:
    with pytest.raises(JudgeConfigurationError, match="credentials"):
        JudgeProviderConfig.from_env(
            valid_env(AGENTLAB_JUDGE_BASE_URL="https://user:secret@judge.example.com/v1")
        )


def test_base_url_is_normalized_without_query() -> None:
    config = JudgeProviderConfig.from_env(
        valid_env(AGENTLAB_JUDGE_BASE_URL="https://judge.example.com/v1/?token=abc")
    )

    assert config.base_url == "https://judge.example.com/v1"


def test_missing_model_rejected() -> None:
    env = valid_env()
    env.pop("AGENTLAB_JUDGE_MODEL")

    with pytest.raises(JudgeConfigurationError, match="AGENTLAB_JUDGE_MODEL"):
        JudgeProviderConfig.from_env(env)


def test_placeholder_model_rejected() -> None:
    with pytest.raises(JudgeConfigurationError, match="appears invalid"):
        JudgeProviderConfig.from_env(
            valid_env(AGENTLAB_JUDGE_MODEL="replace-me-model")
        )


@pytest.mark.parametrize("raw_timeout", ["abc", "True", "-1", "0", "nan", "inf", "1e999"])
def test_invalid_timeout_rejected(raw_timeout: str) -> None:
    with pytest.raises(JudgeConfigurationError, match="positive number"):
        JudgeProviderConfig.from_env(
            valid_env(AGENTLAB_JUDGE_TIMEOUT_SECONDS=raw_timeout)
        )


def test_valid_timeout_is_parsed() -> None:
    config = JudgeProviderConfig.from_env(
        valid_env(AGENTLAB_JUDGE_TIMEOUT_SECONDS="12.5")
    )

    assert config.timeout_seconds == 12.5


def test_api_key_absent_from_config_and_judge_repr() -> None:
    config = JudgeProviderConfig(
        api_key=_VALID_API_KEY,
        base_url="https://judge.example.com/v1",
        model=_VALID_MODEL,
    )
    judge = OpenAICompatibleJudge(config)

    assert _VALID_API_KEY not in repr(config)
    assert _VALID_API_KEY not in repr(judge)
    assert "judge.example.com" in repr(config)


def test_api_key_absent_from_exception_strings() -> None:
    secret = "secret-fragment-987654321"
    with pytest.raises(JudgeConfigurationError) as captured:
        JudgeProviderConfig.from_env(
            valid_env(AGENTLAB_JUDGE_API_KEY=f"your api key {secret}")
        )

    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)


def test_config_does_not_read_repo_doctor_environment(monkeypatch) -> None:
    monkeypatch.delenv("AGENTLAB_JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("REPO_DOCTOR_API_KEY", _VALID_API_KEY)
    monkeypatch.setenv("REPO_DOCTOR_BASE_URL", _VALID_BASE_URL)
    monkeypatch.setenv("REPO_DOCTOR_MODEL", _VALID_MODEL)

    with pytest.raises(JudgeConfigurationError, match="AGENTLAB_JUDGE_API_KEY"):
        JudgeProviderConfig.from_env()


# --- HTTP transport tests -----------------------------------------------------


def test_successful_response_extraction() -> None:
    requests = []
    judge, _config = make_judge(success_handler(requests))

    content = judge("Judge this evidence.")

    assert content == '{"passed": true, "score": 1.0, "feedback": "ok"}'
    assert len(requests) == 1


def test_request_targets_chat_completions_endpoint() -> None:
    requests = []
    judge, _config = make_judge(success_handler(requests), base_url="http://judge.test/v1")

    judge("prompt")

    assert str(requests[0].url) == "http://judge.test/v1/chat/completions"


@pytest.mark.parametrize(
    "base_url",
    ["https://api.deepseek.com", "https://api.deepseek.com/v1"],
)
def test_deepseek_style_base_urls_form_correct_endpoint(base_url: str) -> None:
    requests = []
    judge, _config = make_judge(success_handler(requests), base_url=base_url)

    judge("prompt")

    assert str(requests[0].url) == f"{base_url}/chat/completions"


def test_injected_client_is_not_closed_by_transport() -> None:
    requests = []
    handler = success_handler(requests)
    config = JudgeProviderConfig(
        api_key=_VALID_API_KEY,
        base_url="http://judge.test/v1",
        model="judge-model",
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = OpenAICompatibleJudge(config, client=client)

    judge("prompt")

    assert client.is_closed is False
    client.close()


def test_lazily_created_client_is_closed_after_call(monkeypatch) -> None:
    closed: list[bool] = []
    requests: list[httpx.Request] = []
    handler = success_handler(requests)

    class RecordingClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.MockTransport(handler))
            super().__init__(*args, **kwargs)

        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setattr(
        "agentlab.evaluators.openai_compatible.httpx.Client",
        RecordingClient,
    )
    config = JudgeProviderConfig(
        api_key=_VALID_API_KEY,
        base_url="http://judge.test/v1",
        model="judge-model",
    )
    judge = OpenAICompatibleJudge(config)

    judge("prompt")

    assert closed == [True]
    assert len(requests) == 1


def test_request_sends_authorization_header() -> None:
    requests = []
    judge, config = make_judge(success_handler(requests))

    judge("prompt")

    assert requests[0].headers["Authorization"] == f"Bearer {config.api_key}"


def test_request_sends_model_prompt_temperature_and_json_mode() -> None:
    requests = []
    judge, _config = make_judge(success_handler(requests))

    judge("the prompt text")

    body = json.loads(requests[0].content)
    assert body["model"] == "judge-model"
    assert body["messages"] == [{"role": "user", "content": "the prompt text"}]
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("status_code", [401, 429, 500])
def test_http_errors_are_safe(status_code: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": "secret response body content"}},
        )

    judge, _config = make_judge(handler)

    with pytest.raises(JudgeProviderError, match=f"HTTP {status_code}"):
        judge("prompt with secrets")


def test_http_error_message_never_leaks_body_prompt_or_key() -> None:
    secret = "body-secret-abcdef123456"
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["prompt"] = json.loads(request.content)["messages"][0]["content"]
        captured["auth"] = request.headers["Authorization"]
        return httpx.Response(500, json={"error": {"message": secret}})

    judge, config = make_judge(handler, api_key="sk-test-1234567890abcdef")

    with pytest.raises(JudgeProviderError) as caught:
        judge("prompt-content-xyz")

    message = str(caught.value)
    assert message == "Judge provider returned HTTP 500."
    assert secret not in message
    assert "prompt-content-xyz" not in message
    assert config.api_key not in message
    assert captured["auth"] not in message


def test_timeout_is_safe() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    judge, _config = make_judge(handler)

    with pytest.raises(JudgeProviderError, match="timed out"):
        judge("prompt")


def test_network_error_is_safe() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", "http://judge.test"))

    judge, _config = make_judge(handler)

    with pytest.raises(JudgeProviderError, match="ConnectError"):
        judge("prompt")


def test_non_json_response_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    judge, _config = make_judge(handler)

    with pytest.raises(JudgeProviderError, match="non-JSON"):
        judge("prompt")


def test_non_object_response_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    judge, _config = make_judge(handler)

    with pytest.raises(JudgeProviderError, match="JSON object"):
        judge("prompt")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"choices": None}, "missing choices"),
        ({"choices": []}, "missing choices"),
        ({"choices": ["not-an-object"]}, "invalid choice"),
        ({"choices": [{}]}, "missing a message"),
        ({"choices": [{"message": None}]}, "missing a message"),
        ({"choices": [{"message": {"content": ""}}]}, "no message content"),
        ({"choices": [{"message": {"content": None}}]}, "no message content"),
        ({"choices": [{"message": {"content": 42}}]}, "no message content"),
    ],
)
def test_malformed_provider_responses_rejected(payload, match: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    judge, _config = make_judge(handler)

    with pytest.raises(JudgeProviderError, match=match):
        judge("prompt")

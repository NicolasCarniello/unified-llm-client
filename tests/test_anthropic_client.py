import logging

import pytest

from llm_client import LLMClientError
from llm_client.anthropic_client import AnthropicClient
from llm_client.schemas import ChatMessage, ErrorType, ModelConfig
from tests.fakes import (
    FAST_RETRY,
    FakeServer,
    anthropic_error,
    anthropic_message,
    anthropic_stream,
    connection_error,
)

MSGS = [ChatMessage(role="system", content="Sé breve."), ChatMessage(role="user", content="¿Qué es la entropía?")]
CFG = ModelConfig(model="claude-opus-5-5", max_tokens=500)


def make_client(server: FakeServer) -> AnthropicClient:
    return AnthropicClient("sk-ant-test", retry_policy=FAST_RETRY, http_client=server.http_client())


class TestRequest:
    async def test_system_goes_as_separate_parameter(self):
        server = FakeServer([anthropic_message()])
        await make_client(server).generate(MSGS, CFG)
        body = server.last_body
        assert body["system"] == "Sé breve."
        assert body["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
        assert body["max_tokens"] == 500

    async def test_no_system_parameter_when_absent(self):
        server = FakeServer([anthropic_message()])
        await make_client(server).generate([ChatMessage(role="user", content="hola")], CFG)
        assert "system" not in server.last_body

    async def test_multiple_system_messages_are_joined(self):
        server = FakeServer([anthropic_message()])
        msgs = [ChatMessage(role="system", content="A"), ChatMessage(role="system", content="B"), MSGS[1]]
        await make_client(server).generate(msgs, CFG)
        assert server.last_body["system"] == "A\n\nB"

    async def test_sampling_params_are_never_sent(self, caplog):
        server = FakeServer([anthropic_message()])
        with caplog.at_level(logging.WARNING):
            await make_client(server).generate(MSGS, ModelConfig(model="claude-opus-5-5", temperature=1.5, top_p=0.5))
        assert "temperature" not in server.last_body and "top_p" not in server.last_body
        assert "se ignoran" in caplog.text

    async def test_no_warning_with_default_temperature(self, caplog):
        with caplog.at_level(logging.WARNING):
            await make_client(FakeServer([anthropic_message()])).generate(MSGS, CFG)
        assert "se ignoran" not in caplog.text

    @pytest.mark.parametrize(
        ("model", "expects_fallbacks"),
        [("claude-opus-5-5", True), ("claude-sonnet-5-5", True), ("claude-haiku-5-5", False)],
    )
    async def test_fallbacks_only_for_supported_models(self, model, expects_fallbacks):
        server = FakeServer([anthropic_message(model=model)])
        await make_client(server).generate(MSGS, ModelConfig(model=model))
        assert (server.last_body.get("fallbacks") == "default") is expects_fallbacks


class TestGenerate:
    async def test_parses_response_ignoring_thinking_blocks(self):
        resp = await make_client(FakeServer([anthropic_message("Hola")])).generate(MSGS, CFG)
        assert resp.success
        assert resp.content == "Hola"
        assert resp.provider == "anthropic"
        assert resp.finish_reason == "end_turn"
        assert resp.usage.total_tokens == 28

    async def test_refusal_is_a_controlled_error(self):
        server = FakeServer([anthropic_message(text="", stop_reason="refusal")])
        resp = await make_client(server).generate(MSGS, CFG)
        assert resp.error.type == ErrorType.REFUSAL and not resp.error.retryable
        assert server.calls == 1


class TestStream:
    async def test_yields_text(self):
        server = FakeServer([anthropic_stream(["Medida ", "del ", "desorden."])])
        assert [t async for t in make_client(server).stream(MSGS, CFG)] == ["Medida ", "del ", "desorden."]

    async def test_refusal_at_end_of_stream_raises_controlled_error(self):
        server = FakeServer([anthropic_stream(["Parcial"], stop_reason="refusal")])
        with pytest.raises(LLMClientError) as exc_info:
            [t async for t in make_client(server).stream(MSGS, CFG)]
        assert exc_info.value.error.type == ErrorType.REFUSAL


class TestErrors:
    @pytest.mark.parametrize(
        ("response", "expected_type"),
        [
            (anthropic_error(401, "authentication_error", "invalid x-api-key"), ErrorType.AUTH),
            (anthropic_error(403, "permission_error"), ErrorType.AUTH),
            (anthropic_error(400, "invalid_request_error"), ErrorType.BAD_REQUEST),
            (anthropic_error(404, "not_found_error"), ErrorType.BAD_REQUEST),
        ],
        ids=["401", "403", "400", "404"],
    )
    async def test_non_retryable_errors(self, response, expected_type):
        server = FakeServer([response])
        resp = await make_client(server).generate(MSGS, CFG)
        assert resp.error.type == expected_type and not resp.error.retryable
        assert server.calls == 1

    @pytest.mark.parametrize(
        "failure",
        [
            anthropic_error(429, "rate_limit_error"),
            anthropic_error(529, "overloaded_error", "Overloaded"),
            anthropic_error(500, "api_error"),
            connection_error(),
        ],
        ids=["429", "529-overloaded", "500", "network"],
    )
    async def test_retryable_errors_recover(self, failure):
        server = FakeServer([failure, anthropic_message()])
        resp = await make_client(server).generate(MSGS, CFG)
        assert resp.success and server.calls == 2

    async def test_stream_retries_before_first_token(self):
        server = FakeServer([anthropic_error(529, "overloaded_error"), anthropic_stream(["ok"])])
        assert [t async for t in make_client(server).stream(MSGS, CFG)] == ["ok"]
        assert server.calls == 2

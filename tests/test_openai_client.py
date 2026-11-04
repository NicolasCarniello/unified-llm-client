import pytest

from llm_client import LLMClientError
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, ErrorType, ModelConfig
from tests.fakes import FAST_RETRY, FakeServer, connection_error, openai_completion, openai_error, openai_stream

MSGS = [ChatMessage(role="system", content="Sé breve."), ChatMessage(role="user", content="¿Qué es la entropía?")]
CFG = ModelConfig(model="gpt-4o-mini", temperature=0.3, max_tokens=100)


def make_client(server: FakeServer) -> OpenAIClient:
    return OpenAIClient("sk-test", retry_policy=FAST_RETRY, http_client=server.http_client())


class TestRequest:
    async def test_sends_messages_and_params_in_openai_format(self):
        server = FakeServer([openai_completion()])
        await make_client(server).generate(MSGS, CFG)
        body = server.last_body
        assert body["messages"] == [
            {"role": "system", "content": "Sé breve."},
            {"role": "user", "content": "¿Qué es la entropía?"},
        ]
        assert body["temperature"] == 0.3
        assert body["max_completion_tokens"] == 100
        assert "top_p" not in body  # solo se envía si se configuró

    async def test_sends_top_p_when_configured(self):
        server = FakeServer([openai_completion()])
        await make_client(server).generate(MSGS, ModelConfig(model="gpt-4o-mini", top_p=0.9))
        assert server.last_body["top_p"] == 0.9


class TestGenerate:
    async def test_parses_response(self):
        resp = await make_client(FakeServer([openai_completion("Hola")])).generate(MSGS, CFG)
        assert resp.success
        assert resp.content == "Hola"
        assert resp.provider == "openai"
        assert resp.finish_reason == "stop"
        assert (resp.usage.input_tokens, resp.usage.output_tokens) == (12, 5)


class TestStream:
    async def test_yields_text_and_skips_empty_chunks(self):
        server = FakeServer([openai_stream(["Medida ", "del ", "desorden."])])
        tokens = [t async for t in make_client(server).stream(MSGS, CFG)]
        assert tokens == ["Medida ", "del ", "desorden."]
        assert server.last_body["stream"] is True

    async def test_stream_error_is_controlled(self):
        server = FakeServer([openai_error(401, "Incorrect API key", "invalid_api_key")])
        with pytest.raises(LLMClientError) as exc_info:
            [t async for t in make_client(server).stream(MSGS, CFG)]
        assert exc_info.value.error.type == ErrorType.AUTH


class TestErrors:
    @pytest.mark.parametrize(
        ("response", "expected_type"),
        [
            (openai_error(401, "Incorrect API key", "invalid_api_key"), ErrorType.AUTH),
            (openai_error(403, "Forbidden"), ErrorType.AUTH),
            (openai_error(429, "You exceeded your quota", "insufficient_quota"), ErrorType.QUOTA),
            (openai_error(400, "Bad param"), ErrorType.BAD_REQUEST),
            (openai_error(404, "Model not found", "model_not_found"), ErrorType.BAD_REQUEST),
        ],
        ids=["401", "403", "429-quota", "400", "404"],
    )
    async def test_non_retryable_errors(self, response, expected_type):
        server = FakeServer([response])
        resp = await make_client(server).generate(MSGS, CFG)
        assert not resp.success
        assert resp.error.type == expected_type
        assert not resp.error.retryable
        assert server.calls == 1  # no reintentó

    @pytest.mark.parametrize(
        ("failure", "expected_type"),
        [
            (openai_error(429, "Rate limit"), ErrorType.RATE_LIMIT),
            (openai_error(500, "Internal error"), ErrorType.API),
            (openai_error(503, "Unavailable"), ErrorType.API),
            (connection_error(), ErrorType.NETWORK),
        ],
        ids=["429", "500", "503", "network"],
    )
    async def test_retryable_errors_recover(self, failure, expected_type):
        server = FakeServer([failure, openai_completion()])
        resp = await make_client(server).generate(MSGS, CFG)
        assert resp.success and server.calls == 2

    async def test_retryable_errors_exhausted(self):
        server = FakeServer([openai_error(429, "Rate limit")] * 4)
        resp = await make_client(server).generate(MSGS, CFG)
        assert resp.error.type == ErrorType.RATE_LIMIT and resp.error.retryable
        assert server.calls == 4  # 1 intento + 3 reintentos

    async def test_reads_retry_after_header(self):
        server = FakeServer([openai_error(429, "Rate limit", headers={"retry-after": "7"})] * 4)
        resp = await make_client(server).generate(MSGS, CFG)
        assert resp.error.retry_after == 7.0


async def test_timeout_is_classified_as_timeout_not_network():
    """APITimeoutError hereda de APIConnectionError: si el orden de los chequeos
    estuviera invertido, un timeout se reportaría como error de red."""
    import httpx2

    server = FakeServer([httpx2.ReadTimeout("lento")] * 4)
    resp = await make_client(server).generate(MSGS, CFG)
    assert resp.error.type == ErrorType.TIMEOUT
    assert resp.error.retryable and server.calls == 4

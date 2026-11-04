"""Tests de la lógica común (reintentos, streaming seguro, no bloqueo) con un cliente falso."""

import asyncio
import time

import pytest

from llm_client.base import BaseLLMClient, LLMClientError
from llm_client.schemas import ChatMessage, ErrorType, LLMError, ModelConfig, ModelResponse, Provider
from tests.fakes import FAST_RETRY

MSGS = [ChatMessage(role="user", content="hola")]
CFG = ModelConfig(model="fake")


class Transient(Exception):
    pass


class Fatal(Exception):
    pass


class FakeClient(BaseLLMClient):
    """Falla `failures` veces con `exc` y después responde."""

    provider = Provider.OPENAI

    def __init__(self, failures=0, exc=Transient, fail_mid_stream=False, delay=0.0):
        super().__init__(FAST_RETRY)
        self.failures, self.exc, self.fail_mid_stream, self.delay = failures, exc, fail_mid_stream, delay
        self.calls = 0

    async def _generate(self, messages, config):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.calls <= self.failures:
            raise self.exc("boom")
        return ModelResponse(provider=self.provider, model=config.model, content="ok")

    async def _stream(self, messages, config):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("boom")
        for i, token in enumerate(["a", "b", "c"]):
            if self.fail_mid_stream and i == 2:
                raise Transient("corte")
            yield token

    async def aclose(self):
        pass

    def _to_llm_error(self, exc):
        if isinstance(exc, Transient):
            return LLMError(type=ErrorType.RATE_LIMIT, message=str(exc), retryable=True)
        if isinstance(exc, Fatal):
            return LLMError(type=ErrorType.AUTH, message=str(exc), retryable=False)
        return super()._to_llm_error(exc)


class TestGenerate:
    async def test_success_first_try(self):
        client = FakeClient()
        resp = await client.generate(MSGS, CFG)
        assert resp.success and resp.content == "ok" and client.calls == 1
        assert resp.latency_ms is not None

    async def test_retries_transient_errors_then_succeeds(self):
        client = FakeClient(failures=2)
        resp = await client.generate(MSGS, CFG)
        assert resp.success and client.calls == 3

    async def test_does_not_retry_fatal_errors(self):
        client = FakeClient(failures=99, exc=Fatal)
        resp = await client.generate(MSGS, CFG)
        assert not resp.success and resp.error.type == ErrorType.AUTH and client.calls == 1

    async def test_gives_up_after_max_retries(self):
        client = FakeClient(failures=99)
        resp = await client.generate(MSGS, CFG)
        assert not resp.success and resp.error.type == ErrorType.RATE_LIMIT
        assert client.calls == 1 + FAST_RETRY.max_retries

    async def test_unknown_exceptions_never_escape(self):
        client = FakeClient(failures=99, exc=RuntimeError)
        resp = await client.generate(MSGS, CFG)
        assert resp.error.type == ErrorType.UNKNOWN and not resp.error.retryable

    async def test_value_error_maps_to_bad_request(self):
        client = FakeClient(failures=99, exc=ValueError)
        assert (await client.generate(MSGS, CFG)).error.type == ErrorType.BAD_REQUEST


class TestStream:
    async def test_yields_all_tokens(self):
        assert [t async for t in FakeClient().stream(MSGS, CFG)] == ["a", "b", "c"]

    async def test_retries_if_fails_before_first_token(self):
        client = FakeClient(failures=1)
        assert [t async for t in client.stream(MSGS, CFG)] == ["a", "b", "c"]
        assert client.calls == 2

    async def test_does_not_retry_after_emitting_tokens(self):
        client = FakeClient(fail_mid_stream=True)
        received = []
        with pytest.raises(LLMClientError) as exc_info:
            async for token in client.stream(MSGS, CFG):
                received.append(token)
        assert received == ["a", "b"]          # no se duplicó texto
        assert client.calls == 1               # no reintentó
        assert exc_info.value.error.type == ErrorType.RATE_LIMIT

    async def test_raises_only_llm_client_error(self):
        with pytest.raises(LLMClientError) as exc_info:
            [t async for t in FakeClient(failures=99, exc=Fatal).stream(MSGS, CFG)]
        assert isinstance(exc_info.value.__cause__, Fatal)  # la causa original queda encadenada


class TestNonBlocking:
    async def test_concurrent_calls_run_in_parallel(self):
        """5 llamadas de 0.2 s en paralelo deben tardar ~0.2 s, no 1 s."""
        start = time.perf_counter()
        results = await asyncio.gather(*(FakeClient(delay=0.2).generate(MSGS, CFG) for _ in range(5)))
        elapsed = time.perf_counter() - start
        assert all(r.success for r in results)
        assert elapsed < 0.5


async def test_async_context_manager_closes_client():
    closed = False

    class Closable(FakeClient):
        async def aclose(self):
            nonlocal closed
            closed = True

    async with Closable():
        pass
    assert closed

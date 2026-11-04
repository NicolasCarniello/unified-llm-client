"""Cliente asíncrono para OpenAI (Chat Completions API)."""

from collections.abc import AsyncIterator
from typing import Any

import httpx2
import openai
from openai import AsyncOpenAI

from ._errors import classify_sdk_error
from .base import BaseLLMClient
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Provider, RetryPolicy, Usage


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(
        self,
        api_key: str | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 60.0,
        http_client: httpx2.AsyncClient | None = None,  # inyectable para tests
    ) -> None:
        super().__init__(retry_policy)
        # max_retries=0: desactivamos los reintentos internos del SDK porque
        # los maneja BaseLLMClient (si no, se multiplicarían: 3 nuestros x 3 del SDK).
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=timeout, http_client=http_client)

    # ---------------- traducción a formato OpenAI ----------------
    @staticmethod
    def _build_params(messages: list[ChatMessage], config: ModelConfig) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": config.model,
            # En OpenAI el mensaje "system" va dentro de la lista, igual que en nuestro formato.
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": config.temperature,
            "max_completion_tokens": config.max_tokens,  # reemplaza al viejo `max_tokens`
        }
        if config.top_p is not None:
            params["top_p"] = config.top_p
        return params

    # ---------------- implementación de la interfaz ----------------
    async def _generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        # `await` sobre el cliente ASYNC: el event loop queda libre mientras la API responde.
        resp = await self._client.chat.completions.create(**self._build_params(messages, config))
        choice = resp.choices[0]
        return ModelResponse(
            provider=self.provider,
            model=resp.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens)
            if resp.usage
            else None,
        )

    async def _stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(**self._build_params(messages, config), stream=True)
        async for chunk in stream:
            # Algunos chunks no traen texto (el primero trae solo el rol, el último
            # solo finish_reason, y con include_usage llega uno con choices vacío).
            if chunk.choices and (text := chunk.choices[0].delta.content):
                yield text

    def _to_llm_error(self, exc: Exception) -> LLMError:
        return classify_sdk_error(exc, openai) or super()._to_llm_error(exc)

    async def aclose(self) -> None:
        await self._client.close()

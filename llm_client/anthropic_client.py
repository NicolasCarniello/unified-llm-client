"""Cliente asíncrono para Anthropic (Messages API).

Diferencias con OpenAI que este cliente "esconde" detrás de la interfaz común:
  1. El mensaje `system` NO va en la lista de mensajes: es un parámetro aparte.
  2. `max_tokens` es obligatorio.
  3. La respuesta es una lista de bloques (texto, razonamiento, ...): hay que quedarse con los de texto.
  4. Los modelos actuales de Claude no aceptan `temperature` ni `top_p` (la API devuelve 400
     y el SDK ni siquiera los expone): la profundidad se controla con `effort`. Si el usuario
     los configura, los ignoramos y avisamos por log en lugar de romper la llamada.
  5. Un rechazo por políticas de seguridad llega como respuesta 200 con stop_reason="refusal".
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic
import httpx2
from anthropic import AsyncAnthropic

from ._errors import ModelRefusalError, classify_sdk_error
from .base import BaseLLMClient
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Provider, RetryPolicy, Role, Usage

logger = logging.getLogger(__name__)

# Modelos que soportan "fallbacks" del lado del servidor: si el modelo rechaza la
# request por un falso positivo de seguridad, la API la reintenta con otro modelo
# dentro de la misma llamada.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(
        self,
        api_key: str | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 60.0,
        http_client: httpx2.AsyncClient | None = None,  # inyectable para tests
    ) -> None:
        super().__init__(retry_policy)
        self._client = AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout, http_client=http_client)

    # ---------------- traducción a formato Anthropic ----------------
    @staticmethod
    def _build_params(messages: list[ChatMessage], config: ModelConfig) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        chat = [{"role": m.role.value, "content": m.content} for m in messages if m.role is not Role.SYSTEM]

        if config.temperature != ModelConfig.model_fields["temperature"].default or config.top_p is not None:
            logger.warning(
                "anthropic: temperature/top_p no son soportados por los modelos actuales de Claude; se ignoran."
            )

        params: dict[str, Any] = {"model": config.model, "max_tokens": config.max_tokens, "messages": chat}
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        if config.model in _FALLBACK_MODELS:
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    # ---------------- implementación de la interfaz ----------------
    async def _generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        # Usamos client.beta.messages porque `fallbacks` es una función beta.
        resp = await self._client.beta.messages.create(**self._build_params(messages, config))
        usage = Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)

        if resp.stop_reason == "refusal":
            raise ModelRefusalError("El modelo se negó a responder esta solicitud.")

        return ModelResponse(
            provider=self.provider,
            model=resp.model,
            content="".join(block.text for block in resp.content if block.type == "text"),
            finish_reason=resp.stop_reason,
            usage=usage,
        )

    async def _stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        # `async with` garantiza que la conexión HTTP se cierre aunque el consumidor corte el loop.
        async with self._client.beta.messages.stream(**self._build_params(messages, config)) as stream:
            async for text in stream.text_stream:  # el SDK ya filtra solo los deltas de texto
                yield text
            final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                raise ModelRefusalError("El modelo se negó a continuar la respuesta.")

    def _to_llm_error(self, exc: Exception) -> LLMError:
        return classify_sdk_error(exc, anthropic) or super()._to_llm_error(exc)

    async def aclose(self) -> None:
        await self._client.close()

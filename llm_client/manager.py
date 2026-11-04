"""AsyncLLMManager: punto de entrada único del paquete.

Elige el proveedor a partir de la configuración (patrón *Factory*) y expone
`generate()` y `stream()` sin que el código que lo usa sepa si por detrás
hay OpenAI o Anthropic. Cambiar de proveedor = cambiar LLM_PROVIDER en el .env.
"""

import os
from collections.abc import AsyncIterator
from typing import Any, Self

from pydantic import ValidationError

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, RetryPolicy, Role


class LLMConfigError(ValueError):
    """Configuración inválida (proveedor desconocido, falta la API key, parámetros fuera de rango).

    A diferencia de los errores de la API, este SÍ se lanza: es un error del
    programador/configuración y conviene fallar rápido, al arrancar, con un mensaje claro.
    """


# Registro de proveedores: agregar uno nuevo es sumar una línea acá.
_REGISTRY: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}
_API_KEY_VARS = {Provider.OPENAI: "OPENAI_API_KEY", Provider.ANTHROPIC: "ANTHROPIC_API_KEY"}
_MODEL_VARS = {Provider.OPENAI: "OPENAI_MODEL", Provider.ANTHROPIC: "ANTHROPIC_MODEL"}
_DEFAULT_MODELS = {Provider.OPENAI: "gpt-4o-mini", Provider.ANTHROPIC: "claude-opus-5-5"}


class AsyncLLMManager:
    """Cliente LLM unificado y asíncrono."""

    def __init__(
        self,
        provider: Provider | str,
        *,
        api_key: str | None = None,
        config: ModelConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        **client_kwargs: Any,  # se pasan al cliente concreto (p. ej. timeout, http_client)
    ) -> None:
        self.provider = _parse_provider(provider)

        key_var = _API_KEY_VARS[self.provider]
        api_key = api_key or os.getenv(key_var)
        if not api_key:
            raise LLMConfigError(f"Falta la API key: definí {key_var} en tu archivo .env (ver .env.example).")

        self.config = config or ModelConfig(model=_DEFAULT_MODELS[self.provider])
        self._client: BaseLLMClient = _REGISTRY[self.provider](
            api_key, retry_policy=retry_policy, **client_kwargs
        )

    @classmethod
    def from_env(cls, **client_kwargs: Any) -> Self:
        """Crea el manager leyendo las variables de entorno (cargá el .env antes con load_dotenv())."""
        provider = _parse_provider(os.getenv("LLM_PROVIDER", Provider.OPENAI))
        try:
            config = ModelConfig(
                model=os.getenv(_MODEL_VARS[provider]) or _DEFAULT_MODELS[provider],
                **_optional_env(temperature="LLM_TEMPERATURE", max_tokens="LLM_MAX_TOKENS"),
            )
            retry_policy = RetryPolicy(**_optional_env(max_retries="LLM_MAX_RETRIES"))
        except ValidationError as exc:
            problems = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors())
            raise LLMConfigError(f"Configuración inválida en el .env → {problems}") from exc
        return cls(provider, config=config, retry_policy=retry_policy, **client_kwargs)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    async def generate(
        self, prompt: str | list[ChatMessage], *, system: str | None = None, config: ModelConfig | None = None
    ) -> ModelResponse:
        """Respuesta completa. Nunca lanza errores de API: revisá `response.success`."""
        return await self._client.generate(_to_messages(prompt, system), config or self.config)

    async def stream(
        self, prompt: str | list[ChatMessage], *, system: str | None = None, config: ModelConfig | None = None
    ) -> AsyncIterator[str]:
        """Fragmentos de texto a medida que llegan. Ante un fallo lanza `LLMClientError`."""
        async for token in self._client.stream(_to_messages(prompt, system), config or self.config):
            yield token

    @staticmethod
    def available_providers() -> list[str]:
        return [p.value for p in _REGISTRY]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncLLMManager(provider={self.provider.value!r}, model={self.config.model!r})"


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _parse_provider(value: Provider | str) -> Provider:
    try:
        return Provider(str(value).strip().lower())
    except ValueError:
        options = ", ".join(p.value for p in _REGISTRY)
        raise LLMConfigError(f"Proveedor desconocido {value!r}. Opciones válidas: {options}.") from None


def _optional_env(**fields_to_vars: str) -> dict[str, str]:
    """Solo incluye las variables definidas: las ausentes usan el default del modelo Pydantic."""
    return {field: v for field, var in fields_to_vars.items() if (v := os.getenv(var))}


def _to_messages(prompt: str | list[ChatMessage], system: str | None) -> list[ChatMessage]:
    """Atajo: permite pasar un string en vez de armar la lista de ChatMessage a mano."""
    messages = [ChatMessage(role=Role.USER, content=prompt)] if isinstance(prompt, str) else list(prompt)
    if system:
        messages.insert(0, ChatMessage(role=Role.SYSTEM, content=system))
    return messages

"""Clase base abstracta de todos los clientes LLM.

Usa el patrón *Template Method*: la base implementa la lógica común
(reintentos, medición de latencia, conversión de errores) en los métodos
públicos `generate()` y `stream()`, y cada proveedor implementa solo las
piezas específicas:

    _generate()      -> una llamada "cruda" al SDK (puede lanzar excepciones)
    _stream()        -> el stream "crudo" del SDK (puede lanzar excepciones)
    _to_llm_error()  -> traduce las excepciones del SDK a un LLMError
    aclose()         -> libera la conexión HTTP

Así la lógica de robustez se escribe una sola vez y todos los proveedores la heredan.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar, Self

from .schemas import ChatMessage, ErrorType, LLMError, ModelConfig, ModelResponse, Provider, RetryPolicy

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Única excepción que puede salir del cliente (solo en streaming).

    Las excepciones de los SDKs nunca se filtran hacia afuera: se envuelven
    aquí con un `LLMError` estructurado en el atributo `.error`.
    """

    def __init__(self, error: LLMError) -> None:
        super().__init__(f"[{error.type}] {error.message}")
        self.error = error


class BaseLLMClient(ABC):
    """Interfaz común para cualquier proveedor de LLM."""

    provider: ClassVar[Provider]  # cada subclase declara su proveedor

    def __init__(self, retry_policy: RetryPolicy | None = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()

    # ------------------------------------------------------------------ #
    # API pública (común a todos los proveedores)
    # ------------------------------------------------------------------ #
    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        """Genera una respuesta completa. Nunca lanza: los errores vienen en `response.error`."""
        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                response = await self._generate(messages, config)
                response.latency_ms = _elapsed_ms(start)
                return response
            except Exception as exc:  # noqa: BLE001 - convertimos cualquier fallo en un error controlado
                error = self._to_llm_error(exc)
                if not self._should_retry(error, attempt):
                    logger.error("%s: falló sin reintentar más (%s)", self.provider, error.message)
                    return ModelResponse(
                        provider=self.provider, model=config.model, error=error, latency_ms=_elapsed_ms(start)
                    )
                await self._backoff(attempt, error)
                attempt += 1

    async def stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        """Genera la respuesta token a token (generador asíncrono).

        Reintenta solo si el error ocurre ANTES del primer token: si ya
        emitimos texto, reintentar duplicaría la respuesta. En ese caso, o si
        se agotan los reintentos, lanza `LLMClientError` (nunca una excepción del SDK).
        """
        attempt = 0
        while True:
            emitted_any = False
            try:
                async for token in self._stream(messages, config):
                    emitted_any = True
                    yield token
                return
            except Exception as exc:  # noqa: BLE001
                error = self._to_llm_error(exc)
                if emitted_any or not self._should_retry(error, attempt):
                    logger.error("%s: stream interrumpido (%s)", self.provider, error.message)
                    raise LLMClientError(error) from exc
                await self._backoff(attempt, error)
                attempt += 1

    # ------------------------------------------------------------------ #
    # Lo que cada proveedor debe implementar
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def _generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        """Llamada directa al SDK. Puede lanzar excepciones: la base las maneja."""

    @abstractmethod
    def _stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        """Stream directo del SDK, implementado como `async def` con `yield`."""

    @abstractmethod
    async def aclose(self) -> None:
        """Cierra el cliente HTTP subyacente."""

    def _to_llm_error(self, exc: Exception) -> LLMError:
        """Traduce una excepción a LLMError. Las subclases la extienden con los
        errores de su SDK y llaman a `super()` para los casos no reconocidos."""
        if isinstance(exc, ValueError):  # p. ej. un parámetro incompatible con el proveedor
            return LLMError(type=ErrorType.BAD_REQUEST, message=str(exc), retryable=False)
        return LLMError(type=ErrorType.UNKNOWN, message=f"{type(exc).__name__}: {exc}", retryable=False)

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    def _should_retry(self, error: LLMError, attempt: int) -> bool:
        return error.retryable and attempt < self.retry_policy.max_retries

    async def _backoff(self, attempt: int, error: LLMError) -> None:
        delay = self.retry_policy.delay_for(attempt, error)
        logger.warning(
            "%s: %s. Reintento %d/%d en %.2fs",
            self.provider, error.type, attempt + 1, self.retry_policy.max_retries, delay,
        )
        # asyncio.sleep (NO time.sleep): libera el event loop mientras esperamos.
        await asyncio.sleep(delay)

    # Permite usar el cliente con `async with ... as client:` y que se cierre solo.
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)

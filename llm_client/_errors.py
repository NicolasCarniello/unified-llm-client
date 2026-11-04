"""Traducción de excepciones de los SDKs a `LLMError`.

Los SDKs oficiales de OpenAI y Anthropic tienen la MISMA jerarquía de
excepciones (los dos se generan con la misma herramienta, Stainless):

    APIError
    ├── APIConnectionError
    │   └── APITimeoutError          <- hereda de APIConnectionError: ¡va primero!
    └── APIStatusError (tiene .status_code y .response)
        ├── BadRequestError (400)    ├── RateLimitError (429)
        ├── AuthenticationError (401)├── InternalServerError (>=500)
        └── PermissionDeniedError (403) ...

Por eso una sola función, que recibe el módulo del SDK, sirve para ambos.
"""

from types import ModuleType

from .schemas import ErrorType, LLMError


class ModelRefusalError(Exception):
    """El modelo terminó con stop_reason='refusal'. No es un error HTTP, así que lo modelamos aparte."""


def classify_sdk_error(exc: Exception, sdk: ModuleType) -> LLMError | None:
    """Devuelve el LLMError correspondiente, o None si `exc` no es una excepción de ese SDK."""
    if isinstance(exc, ModelRefusalError):
        return LLMError(type=ErrorType.REFUSAL, message=str(exc), retryable=False)

    # El orden importa: de lo más específico a lo más general.
    if isinstance(exc, sdk.APITimeoutError):
        return LLMError(type=ErrorType.TIMEOUT, message="La API tardó demasiado en responder.", retryable=True)
    if isinstance(exc, sdk.APIConnectionError):
        return LLMError(type=ErrorType.NETWORK, message=f"Error de red: {exc}", retryable=True)

    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return LLMError(
            type=ErrorType.AUTH,
            message="API key inválida o sin permisos. Revisá tu archivo .env.",
            retryable=False,
            status_code=exc.status_code,
        )
    if isinstance(exc, sdk.RateLimitError):
        # OpenAI devuelve 429 tanto por "demasiadas requests" como por "sin saldo".
        # El segundo caso NO se soluciona esperando, así que no lo reintentamos.
        if getattr(exc, "code", None) == "insufficient_quota":
            return LLMError(
                type=ErrorType.QUOTA,
                message="Cuota agotada: revisá el saldo/billing de tu cuenta.",
                retryable=False,
                status_code=429,
            )
        return LLMError(
            type=ErrorType.RATE_LIMIT,
            message="Límite de tasa alcanzado (rate limit).",
            retryable=True,
            status_code=429,
            retry_after=_retry_after(exc),
        )
    if isinstance(exc, (sdk.BadRequestError, sdk.NotFoundError, sdk.UnprocessableEntityError)):
        return LLMError(
            type=ErrorType.BAD_REQUEST,
            message=f"Request inválida (¿modelo o parámetros incorrectos?): {exc.message}",
            retryable=False,
            status_code=exc.status_code,
        )
    if isinstance(exc, sdk.APIStatusError):
        server_side = exc.status_code >= 500  # 500, 502, 503, 529 (overloaded)...
        return LLMError(
            type=ErrorType.API,
            message=f"Error del servidor ({exc.status_code}): {exc.message}",
            retryable=server_side,
            status_code=exc.status_code,
            retry_after=_retry_after(exc) if server_side else None,
        )
    return None


def _retry_after(exc: Exception) -> float | None:
    """Lee el header Retry-After (en segundos) si el servidor lo envió."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:  # ausente, o en formato fecha HTTP: usamos nuestro backoff
        return None

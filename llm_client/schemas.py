"""Contratos de datos del cliente unificado.

Todo lo que entra al cliente (mensajes, configuración) y todo lo que sale
(respuestas, errores) es un modelo Pydantic. Así nunca manipulamos dicts
anidados "a mano" y los datos inválidos se rechazan en el borde, antes de
gastar una llamada a la API.
"""

import random
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Provider(StrEnum):
    """Proveedores soportados. StrEnum permite comparar con strings: Provider.OPENAI == "openai"."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(StrEnum):
    """Roles válidos de un mensaje de chat."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación, en formato neutral (independiente del proveedor)."""

    model_config = ConfigDict(frozen=True)  # inmutable: un mensaje enviado no se modifica

    role: Role
    content: str = Field(min_length=1, description="Texto del mensaje; no puede estar vacío.")


class ModelConfig(BaseModel):
    """Parámetros de generación, validados antes de llamar a la API."""

    model: str = Field(min_length=1, description="Nombre del modelo, p. ej. 'gpt-4o-mini'.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # 2048 por defecto: los modelos con razonamiento interno (p. ej. Claude Opus 5.5)
    # gastan parte de max_tokens pensando; con un tope muy bajo la respuesta sale cortada.
    max_tokens: int = Field(default=2048, gt=0, le=16384)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class ErrorType(StrEnum):
    """Categorías de error normalizadas entre proveedores."""

    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    NETWORK = "network"
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"
    QUOTA = "quota"  # sin saldo/cuota: no se arregla esperando
    REFUSAL = "refusal"  # el modelo se negó a responder (respuesta 200, no excepción HTTP)
    API = "api"
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    """Error controlado: lo que devolvemos en lugar de dejar escapar una excepción."""

    type: ErrorType
    message: str
    retryable: bool = Field(description="True si tiene sentido reintentar (errores transitorios).")
    status_code: int | None = None
    retry_after: float | None = Field(
        default=None, description="Segundos sugeridos por el servidor (header Retry-After), si los informó."
    )


class RetryPolicy(BaseModel):
    """Política de reintentos con backoff exponencial y jitter."""

    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, gt=0, description="Espera base en segundos.")
    max_delay: float = Field(default=30.0, gt=0, description="Tope de espera entre intentos.")

    def delay_for(self, attempt: int, error: LLMError) -> float:
        """Segundos a esperar antes del reintento número `attempt` (empieza en 0)."""
        if error.retry_after is not None:
            # Si el servidor nos dice cuánto esperar, le hacemos caso.
            return min(error.retry_after, self.max_delay)
        # "Full jitter": aleatorio entre 0 y base*2^attempt. Evita que muchos
        # clientes reintenten todos al mismo tiempo y vuelvan a saturar la API.
        ceiling = min(self.base_delay * 2**attempt, self.max_delay)
        return random.uniform(0, ceiling)


class Usage(BaseModel):
    """Consumo de tokens de una llamada."""

    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Respuesta unificada. Si `error` tiene valor, la llamada falló de forma controlada."""

    provider: Provider
    model: str
    content: str | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    error: LLMError | None = None

    @computed_field
    @property
    def success(self) -> bool:
        return self.error is None

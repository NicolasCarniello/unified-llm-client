"""Contratos de datos del cliente unificado.

Todo lo que entra al cliente (mensajes, configuración) y todo lo que sale
(respuestas, errores) es un modelo Pydantic. Así nunca manipulamos dicts
anidados "a mano" y los datos inválidos se rechazan en el borde, antes de
gastar una llamada a la API.
"""

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
    max_tokens: int = Field(default=512, gt=0, le=8192)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class ErrorType(StrEnum):
    """Categorías de error normalizadas entre proveedores."""

    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    NETWORK = "network"
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"
    API = "api"
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    """Error controlado: lo que devolvemos en lugar de dejar escapar una excepción."""

    type: ErrorType
    message: str
    retryable: bool = Field(description="True si tiene sentido reintentar (errores transitorios).")
    status_code: int | None = None


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

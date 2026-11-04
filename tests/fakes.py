"""Servidor HTTP falso para testear los clientes contra los SDKs reales, sin red ni API keys.

Los SDKs aceptan un `http_client` propio: le pasamos uno con `MockTransport`,
que en vez de salir a internet llama a una función nuestra. Así el SDK parsea
respuestas, streams (SSE) y errores exactamente igual que en producción.
"""

import json
from collections.abc import Callable, Iterable

import httpx2

from llm_client.schemas import RetryPolicy

# Reintentos casi instantáneos para que los tests corran rápido.
FAST_RETRY = RetryPolicy(max_retries=3, base_delay=0.001, max_delay=0.001)


class FakeServer:
    """Devuelve las respuestas configuradas en orden y registra las requests recibidas."""

    def __init__(self, responses: Iterable[httpx2.Response | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def last_body(self) -> dict:
        return self.requests[-1]

    def http_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))


# ---------------------------------------------------------------- helpers genéricos
def sse(events: Iterable[tuple[str | None, dict | str]]) -> httpx2.Response:
    """Arma una respuesta Server-Sent Events (el formato del streaming de ambas APIs)."""
    body = ""
    for event, data in events:
        if event:
            body += f"event: {event}\n"
        body += f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def error(status: int, body: dict, headers: dict | None = None) -> httpx2.Response:
    return httpx2.Response(status, json=body, headers=headers or {})


def connection_error() -> Exception:
    return httpx2.ConnectError("sin conexión")


# ---------------------------------------------------------------- OpenAI
def openai_completion(content: str = "Medida del desorden.", model: str = "gpt-4o-mini") -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    })


def openai_stream(parts: list[str]) -> httpx2.Response:
    def chunk(delta: dict, finish: str | None = None, with_choice: bool = True) -> tuple[None, dict]:
        choices = [{"index": 0, "delta": delta, "finish_reason": finish}] if with_choice else []
        return None, {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
                      "choices": choices}

    events = [chunk({"role": "assistant"})]                 # 1er chunk: solo el rol, sin texto
    events += [chunk({"content": p}) for p in parts]
    events += [chunk({}, finish="stop"), chunk({}, with_choice=False), (None, "[DONE]")]
    return sse(events)


def openai_error(status: int, message: str = "error", code: str | None = None,
                 headers: dict | None = None) -> httpx2.Response:
    return error(status, {"error": {"message": message, "type": code or "error", "code": code}}, headers)


# ---------------------------------------------------------------- Anthropic
def anthropic_message(text: str = "Medida del desorden.", stop_reason: str = "end_turn",
                      model: str = "claude-opus-5-5") -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},  # razonamiento: debe ignorarse
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 8},
    })


def anthropic_stream(parts: list[str], stop_reason: str = "end_turn") -> httpx2.Response:
    events: list[tuple[str | None, dict]] = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
    ]
    events += [("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": p}}) for p in parts]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 8}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return sse(events)


def anthropic_error(status: int, err_type: str = "api_error", message: str = "error",
                    headers: dict | None = None) -> httpx2.Response:
    return error(status, {"type": "error", "error": {"type": err_type, "message": message}}, headers)


ResponseFactory = Callable[[], httpx2.Response]

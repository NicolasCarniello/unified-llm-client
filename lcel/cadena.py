"""Refactorización a LCEL asíncrono (Módulo 2 · Unidad 1).

Misma tarea que el Módulo 1 (responder "¿Qué es la entropía?"), pero en lugar
de orquestar a mano el SDK, armamos una cadena declarativa:

    ChatPromptTemplate  ->  ChatOpenAI / ChatAnthropic  ->  StrOutputParser
    dict de variables       ChatPromptValue -> AIMessage     AIMessage -> str

Uso:
    python -m lcel.cadena                        # proveedor según LLM_PROVIDER (.env)
    python -m lcel.cadena --provider anthropic   # fuerza un proveedor
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-opus-5-5"}
API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


# ---------------------------------------------------------------------------
# 1. El modelo: reemplaza a AsyncOpenAI / AsyncAnthropic del Módulo 1
# ---------------------------------------------------------------------------
def build_model(provider: str | None = None) -> BaseChatModel:
    """Crea el chat model de LangChain según el proveedor (o LLM_PROVIDER del .env).

    Ambas clases implementan la misma interfaz Runnable, así que el resto de la
    cadena no cambia al cambiar de proveedor: la intercambiabilidad que en el
    Módulo 1 resolvimos con una clase base abstracta, acá viene de fábrica.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Proveedor desconocido {provider!r}. Opciones: {', '.join(DEFAULT_MODELS)}.")
    if not os.getenv(API_KEY_VARS[provider]):
        raise ValueError(f"Falta {API_KEY_VARS[provider]} en el .env (ver .env.example).")

    model_name = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    # max_retries: el SDK reintenta solo los errores transitorios (429, 5xx, red)
    # con backoff exponencial, lo que en el Módulo 1 implementamos a mano.
    common = {"max_tokens": max_tokens, "max_retries": 3, "timeout": 60}

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")), **common)

    from langchain_anthropic import ChatAnthropic

    # Sin `temperature`: los modelos actuales de Claude rechazan ese parámetro (error 400).
    # max_tokens explícito: sin él, ChatAnthropic usa el máximo del modelo (128k en Opus 5.5).
    return ChatAnthropic(model=model_name, **common)


# ---------------------------------------------------------------------------
# 2. El prompt: roles System/Human y variables {materia} y {pregunta}
# ---------------------------------------------------------------------------
def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Sos un profesor de {materia} que explica con claridad a estudiantes universitarios. "
                "Respondé en español, en un párrafo breve (máximo 4 oraciones).",
            ),
            ("human", "{pregunta}"),
        ]
    )


# ---------------------------------------------------------------------------
# 3. La cadena: prompt | modelo | parser
# ---------------------------------------------------------------------------
def build_chain(model: BaseChatModel) -> Runnable[dict, str]:
    """dict {"materia", "pregunta"} -> str. StrOutputParser extrae el texto del AIMessage."""
    return build_prompt() | model | StrOutputParser()


# ---------------------------------------------------------------------------
# 4. Ejecución asíncrona
# ---------------------------------------------------------------------------
async def main() -> int:
    parser = argparse.ArgumentParser(description="Cadena LCEL asíncrona: prompt | modelo | parser.")
    parser.add_argument("--provider", choices=list(DEFAULT_MODELS), help="Sobrescribe LLM_PROVIDER.")
    args = parser.parse_args()
    load_dotenv()

    try:
        model = build_model(args.provider)
    except ValueError as exc:  # configuración inválida: fallamos rápido y claro
        print(f"⚠️  {exc}")
        return 1

    chain = build_chain(model)
    entrada = {"materia": "física", "pregunta": "¿Qué es la entropía?"}
    print(f"Cadena: {' | '.join(type(step).__name__ for step in chain.steps)}")
    model_id = getattr(model, "model_name", None) or getattr(model, "model", None) or "?"
    print(f"Modelo: {type(model).__name__} ({model_id})")

    try:
        # --- a) ainvoke: el ejercicio pedido ---
        print("\n── 1. chain.ainvoke ──")
        respuesta = await chain.ainvoke(entrada)  # ¡con await! sin él, obtendrías una corrutina
        print(respuesta)
        es_texto = isinstance(respuesta, str)  # StrOutputParser devuelve str (o una subclase de str)
        print(f"\n  ✅ ¿la salida es texto plano? {es_texto} · no es un AIMessage, gracias a StrOutputParser")

        # --- b) astream: la MISMA cadena hace streaming sin cambiar nada ---
        print("\n── 2. chain.astream (bonus) ──")
        async for fragmento in chain.astream(entrada):
            print(fragmento, end="", flush=True)
        print()

        # --- c) abatch: varias preguntas en paralelo con la misma cadena ---
        print("\n── 3. chain.abatch (bonus) ──")
        preguntas = [
            {"materia": "física", "pregunta": "¿Qué es la entropía? Respondé en una oración."},
            {"materia": "teoría de la información", "pregunta": "¿Qué es la entropía? Respondé en una oración."},
        ]
        for entrada_i, salida in zip(preguntas, await chain.abatch(preguntas)):
            print(f"• [{entrada_i['materia']}] {salida}")
    except Exception as exc:  # noqa: BLE001 - un error de API (key inválida, cuota) no debe romper el programa
        print(f"\n  ❌ Error controlado ({type(exc).__name__}): {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Script de validación del Unified Async LLM Client.

Uso:
    python main.py                        # usa el proveedor de LLM_PROVIDER (.env)
    python main.py --provider anthropic   # fuerza un proveedor

Ejecuta:
    1. Modo normal       -> respuesta completa + tokens + latencia
    2. Modo streaming    -> tokens impresos a medida que llegan
    3. Error controlado  -> una API key inválida NO rompe el programa
    4. Bonus             -> ambos proveedores en paralelo (si hay keys de los dos)
"""

import argparse
import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, LLMClientError, LLMConfigError, ModelResponse

QUESTION = "¿Qué es la entropía?"
SYSTEM = "Respondé en español, en un párrafo breve (máximo 4 oraciones)."


def header(title: str) -> None:
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


def show(resp: ModelResponse) -> None:
    if resp.success:
        print(resp.content)
        usage = f"{resp.usage.input_tokens} in / {resp.usage.output_tokens} out" if resp.usage else "n/d"
        print(f"\n  ✅ {resp.provider}/{resp.model} · tokens: {usage} · {resp.latency_ms:.0f} ms")
    else:
        print(f"  ❌ Error controlado [{resp.error.type}]: {resp.error.message}")
        print(f"     retryable={resp.error.retryable} · status={resp.error.status_code}")


async def demo_normal(llm: AsyncLLMManager) -> None:
    header(f"1. Modo normal · {llm.provider.value} ({llm.config.model})")
    print(f"Pregunta: {QUESTION}\n")
    show(await llm.generate(QUESTION, system=SYSTEM))


async def demo_stream(llm: AsyncLLMManager) -> None:
    header(f"2. Modo streaming · {llm.provider.value} ({llm.config.model})")
    print(f"Pregunta: {QUESTION}\n")
    start = time.perf_counter()
    first_token_at: float | None = None
    chunks = 0
    try:
        async for token in llm.stream(QUESTION, system=SYSTEM):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks += 1
            print(token, end="", flush=True)  # flush: mostrar el texto apenas llega
        total = time.perf_counter() - start
        ttft = (first_token_at - start) if first_token_at else total
        print(f"\n\n  ✅ {chunks} fragmentos · primer token en {ttft:.2f}s · total {total:.2f}s")
    except LLMClientError as exc:  # la única excepción que puede salir del cliente
        print(f"\n  ❌ Error controlado en streaming [{exc.error.type}]: {exc.error.message}")


async def demo_controlled_error(provider: str) -> None:
    header("3. Error controlado · API key inválida")
    print("Llamamos a la API con una key falsa: el programa debe seguir funcionando.\n")
    async with AsyncLLMManager(provider, api_key="sk-invalid-key-for-demo") as bad:
        show(await bad.generate(QUESTION))


async def demo_parallel() -> None:
    header("4. Bonus · OpenAI y Anthropic en paralelo (asyncio.gather)")
    if not all(_real_key(var) for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")):
        print("Omitido: se necesitan OPENAI_API_KEY y ANTHROPIC_API_KEY en el .env.")
        return
    async with AsyncLLMManager("openai") as oa, AsyncLLMManager("anthropic") as an:
        start = time.perf_counter()
        results = await asyncio.gather(oa.generate(QUESTION, system=SYSTEM), an.generate(QUESTION, system=SYSTEM))
        wall = time.perf_counter() - start
    for resp in results:
        print(f"\n── {resp.provider} ──")
        show(resp)
    sequential = sum((r.latency_ms or 0) for r in results) / 1000
    print(f"\n  ⏱  En paralelo: {wall:.2f}s · secuencial habría tardado ~{sequential:.2f}s")


def _real_key(var: str) -> bool:
    value = os.getenv(var, "")
    return bool(value) and "your" not in value  # descarta los placeholders de .env.example


async def main() -> int:
    parser = argparse.ArgumentParser(description="Valida el Unified Async LLM Client.")
    parser.add_argument("--provider", choices=AsyncLLMManager.available_providers(), help="Sobrescribe LLM_PROVIDER.")
    args = parser.parse_args()

    load_dotenv()  # carga el .env en os.environ
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    logging.basicConfig(level=logging.WARNING, format="  [log] %(name)s: %(message)s")

    try:
        llm = AsyncLLMManager.from_env()
    except LLMConfigError as exc:  # error de configuración: fallamos rápido y claro
        print(f"⚠️  {exc}")
        return 1

    async with llm:
        await demo_normal(llm)
        await demo_stream(llm)
    await demo_controlled_error(llm.provider.value)
    await demo_parallel()
    print("\nListo. ✔")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

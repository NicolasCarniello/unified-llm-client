# Refactorización a LCEL asíncrono

Ejercicio del **Módulo 2 · Unidad 1**. Toma la tarea del Módulo 1 (responder *"¿Qué es la entropía?"* con OpenAI
o Anthropic) y la reescribe como una cadena declarativa de **LangChain Expression Language**:

```python
chain = prompt | model | StrOutputParser()
respuesta = await chain.ainvoke({"materia": "física", "pregunta": "¿Qué es la entropía?"})
```

Todo el código está en [`cadena.py`](cadena.py).

## Cómo ejecutarlo

Desde la raíz del repositorio, con el entorno virtual activado y el `.env` configurado (las mismas variables
que el Módulo 1, ver [`.env.example`](../.env.example)):

```bash
pip install -r requirements.txt          # incluye langchain-core, langchain-openai, langchain-anthropic
python -m lcel.cadena                    # proveedor según LLM_PROVIDER
python -m lcel.cadena --provider anthropic
```

El script ejecuta la cadena tres veces:

1. **`ainvoke`**: lo que pide el ejercicio. Imprime la respuesta como texto plano.
2. **`astream`** (bonus): la *misma* cadena, sin cambios, imprime el texto a medida que llega.
3. **`abatch`** (bonus): dos preguntas en paralelo con una sola llamada.

## La cadena, paso a paso

```
{"materia": "física", "pregunta": "¿Qué es la entropía?"}
        │
        ▼  ChatPromptTemplate.from_messages([("system", ...{materia}...), ("human", "{pregunta}")])
ChatPromptValue  [SystemMessage, HumanMessage]
        │
        ▼  ChatOpenAI / ChatAnthropic
AIMessage(content="La entropía es...")
        │
        ▼  StrOutputParser()
"La entropía es..."   ← str
```

## Criterios de aceptación

| Criterio | Dónde se cumple |
|---|---|
| Flujo principal con la sintaxis LCEL (`\|`) | `build_chain()`: `build_prompt() \| model \| StrOutputParser()` |
| Ejecución asíncrona con `await chain.ainvoke(...)` | `main()` |
| Salida en texto plano mediante `StrOutputParser` | El script verifica `isinstance(respuesta, str)` |
| `ChatPromptTemplate` con roles System/Human | `build_prompt()` |
| Modelo de LangChain con `temperature` y nombre de modelo | `build_model()`: `ChatOpenAI(model=..., temperature=...)` |

Los tests (`tests/test_lcel_cadena.py`) verifican cada criterio con el modelo falso `GenericFakeChatModel`
de LangChain, sin red ni API keys: `pytest tests/test_lcel_cadena.py`.

## Antes (Módulo 1) vs. después (LCEL)

| Necesidad | Módulo 1: SDK directo | LCEL |
|---|---|---|
| Interfaz común entre proveedores | Clase abstracta `BaseLLMClient` + dos subclases | `ChatOpenAI` y `ChatAnthropic` ya implementan `Runnable` |
| Separar el `system` en Anthropic | Código propio en `AnthropicClient` | Lo resuelve `ChatAnthropic` |
| Streaming | Generador `async` propio en cada cliente | `chain.astream()`, sin código extra |
| Varias consultas en paralelo | `asyncio.gather` a mano | `chain.abatch()` |
| Extraer el texto de la respuesta | `resp.choices[0].message.content` / bloques `text` | `StrOutputParser()` |
| Reintentos ante 429 / 5xx / red | Backoff exponencial propio | `max_retries=3` del modelo |

El cambio de paradigma: el Módulo 1 describe **cómo** hacer cada paso. La cadena LCEL describe **qué** flujo
siguen los datos, y la infraestructura (async, streaming, batch) viene del protocolo `Runnable`.

## Decisiones y detalles

- **Sin `temperature` en Anthropic.** Los modelos actuales de Claude devuelven error 400 si se envía.
  `ChatAnthropic` no lo envía si no se configura, y un test verifica que el payload no lo incluya.
- **`max_tokens` explícito.** Sin él, `ChatAnthropic` pide el máximo del modelo (128.000 tokens en Opus 5.5),
  que no es razonable para una respuesta corta sin streaming.
- **Errores controlados.** Una configuración inválida (falta la key, proveedor inexistente) se informa antes de
  crear la cadena. Un error de API (key inválida, cuota) se captura en `main()` y no rompe el programa.
- **Variables que coinciden.** Las claves del diccionario de entrada (`materia`, `pregunta`) son exactamente las
  variables del prompt. Si falta una, `ChatPromptTemplate` lanza `KeyError`, y hay un test para ese caso.

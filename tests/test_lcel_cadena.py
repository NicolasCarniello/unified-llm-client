"""Tests de la cadena LCEL con un modelo falso de LangChain (sin red ni API keys)."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from lcel.cadena import build_chain, build_model, build_prompt

ENTRADA = {"materia": "física", "pregunta": "¿Qué es la entropía?"}


def fake_model(*texts: str) -> GenericFakeChatModel:
    """Modelo falso que responde los textos dados, en orden (también soporta streaming)."""
    return GenericFakeChatModel(messages=iter(AIMessage(content=t) for t in texts))


class TestPrompt:
    def test_has_system_and_human_roles(self):
        messages = build_prompt().format_messages(**ENTRADA)
        assert [m.type for m in messages] == ["system", "human"]

    def test_variables_are_injected(self):
        system, human = build_prompt().format_messages(**ENTRADA)
        assert "profesor de física" in system.content
        assert human.content == "¿Qué es la entropía?"

    def test_declares_expected_variables(self):
        assert set(build_prompt().input_variables) == {"materia", "pregunta"}

    def test_missing_variable_fails(self):
        with pytest.raises(KeyError):
            build_prompt().format_messages(pregunta="hola")  # falta "materia"


class TestChain:
    def test_is_prompt_model_parser_sequence(self):
        chain = build_chain(fake_model("x"))
        assert isinstance(chain, RunnableSequence)  # construida con el operador |
        first, middle, last = chain.steps
        assert isinstance(first, ChatPromptTemplate)
        assert isinstance(middle, GenericFakeChatModel)
        assert isinstance(last, StrOutputParser)

    async def test_ainvoke_returns_plain_string(self):
        result = await build_chain(fake_model("Medida del desorden.")).ainvoke(ENTRADA)
        assert result == "Medida del desorden."
        assert isinstance(result, str) and not isinstance(result, AIMessage)

    async def test_astream_yields_string_chunks(self):
        chunks = [c async for c in build_chain(fake_model("Medida del desorden.")).astream(ENTRADA)]
        assert len(chunks) > 1  # llegó en fragmentos, no de una vez
        assert all(isinstance(c, str) for c in chunks)
        assert "".join(chunks) == "Medida del desorden."

    async def test_abatch_processes_several_inputs(self):
        chain = build_chain(fake_model("uno", "dos"))
        assert await chain.abatch([ENTRADA, ENTRADA], config={"max_concurrency": 1}) == ["uno", "dos"]

    async def test_without_await_you_get_a_coroutine(self):
        """El error común del ejercicio: olvidar el await."""
        pending = build_chain(fake_model("x")).ainvoke(ENTRADA)
        assert not isinstance(pending, str)
        assert await pending == "x"


@pytest.fixture
def clean_env(monkeypatch):
    for var in ["LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL", "ANTHROPIC_MODEL",
                "LLM_TEMPERATURE", "LLM_MAX_TOKENS"]:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class TestBuildModel:
    def test_openai_by_default(self, clean_env):
        clean_env.setenv("OPENAI_API_KEY", "sk-test")
        clean_env.setenv("LLM_TEMPERATURE", "0.2")
        model = build_model()
        assert type(model).__name__ == "ChatOpenAI"
        assert model.model_name == "gpt-4o-mini" and model.temperature == 0.2 and model.max_tokens == 2048

    def test_anthropic_never_sends_temperature(self, clean_env):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        clean_env.setenv("LLM_TEMPERATURE", "1.5")
        model = build_model("anthropic")
        payload = model._get_request_payload([("human", "hola")])
        assert type(model).__name__ == "ChatAnthropic"
        assert "temperature" not in payload
        assert payload["max_tokens"] == 2048  # no el máximo del modelo

    def test_model_from_env(self, clean_env):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        clean_env.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        assert build_model("anthropic").model == "claude-haiku-5-5"

    def test_unknown_provider(self, clean_env):
        with pytest.raises(ValueError, match="Proveedor desconocido"):
            build_model("gemini")

    def test_missing_key(self, clean_env):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            build_model("anthropic")

import pytest

from llm_client import AnthropicClient, AsyncLLMManager, LLMConfigError, OpenAIClient
from tests.fakes import FakeServer, anthropic_message, openai_completion

ENV_VARS = [
    "LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL", "ANTHROPIC_MODEL",
    "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_MAX_RETRIES",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Cada test arranca sin variables de entorno del sistema (aislamiento)."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestFromEnv:
    def test_defaults_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        llm = AsyncLLMManager.from_env()
        assert llm.provider == "openai"
        assert llm.config.model == "gpt-4o-mini"
        assert isinstance(llm._client, OpenAIClient)

    def test_selects_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        llm = AsyncLLMManager.from_env()
        assert isinstance(llm._client, AnthropicClient)
        assert llm.config.model == "claude-opus-5-5"

    def test_provider_is_normalized(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "  Anthropic ")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        assert AsyncLLMManager.from_env().provider == "anthropic"

    def test_reads_model_and_params(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
        monkeypatch.setenv("LLM_MAX_TOKENS", "300")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        llm = AsyncLLMManager.from_env()
        assert (llm.config.model, llm.config.temperature, llm.config.max_tokens) == ("gpt-4.1", 0.2, 300)
        assert llm._client.retry_policy.max_retries == 5


class TestConfigErrors:
    def test_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        with pytest.raises(LLMConfigError, match="Proveedor desconocido 'gemini'"):
            AsyncLLMManager.from_env()

    @pytest.mark.parametrize(("provider", "var"), [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")])
    def test_missing_api_key(self, monkeypatch, provider, var):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        with pytest.raises(LLMConfigError, match=var):
            AsyncLLMManager.from_env()

    @pytest.mark.parametrize(
        ("var", "value", "field"),
        [("LLM_TEMPERATURE", "5", "temperature"), ("LLM_MAX_TOKENS", "mucho", "max_tokens"),
         ("LLM_MAX_RETRIES", "99", "max_retries")],
    )
    def test_invalid_values(self, monkeypatch, var, value, field):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv(var, value)
        with pytest.raises(LLMConfigError, match=field):
            AsyncLLMManager.from_env()


class TestUsage:
    async def test_same_code_works_with_both_providers(self):
        """Intercambiabilidad: el mismo código, distinto proveedor."""
        for provider, response in [("openai", openai_completion("A")), ("anthropic", anthropic_message("A"))]:
            server = FakeServer([response])
            async with AsyncLLMManager(provider, api_key="k", http_client=server.http_client()) as llm:
                resp = await llm.generate("¿Qué es la entropía?")
            assert resp.success and resp.content == "A" and resp.provider == provider

    async def test_string_prompt_and_system_shortcut(self):
        server = FakeServer([openai_completion()])
        llm = AsyncLLMManager("openai", api_key="k", http_client=server.http_client())
        await llm.generate("hola", system="Sé breve.")
        assert server.last_body["messages"] == [
            {"role": "system", "content": "Sé breve."},
            {"role": "user", "content": "hola"},
        ]

    def test_available_providers(self):
        assert AsyncLLMManager.available_providers() == ["openai", "anthropic"]

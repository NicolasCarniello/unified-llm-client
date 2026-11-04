import pytest
from pydantic import ValidationError

from llm_client.schemas import ChatMessage, ErrorType, LLMError, ModelConfig, ModelResponse, RetryPolicy, Usage


class TestChatMessage:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="hola")
        assert msg.role == "user"  # StrEnum compara directo con strings

    @pytest.mark.parametrize("role", ["robot", "", "USER_"])
    def test_rejects_invalid_role(self, role):
        with pytest.raises(ValidationError):
            ChatMessage(role=role, content="hola")

    def test_rejects_empty_content(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_is_immutable(self):
        msg = ChatMessage(role="user", content="hola")
        with pytest.raises(ValidationError):
            msg.content = "otra cosa"


class TestModelConfig:
    @pytest.mark.parametrize("temperature", [0, 0.7, 2])
    def test_accepts_temperature_in_range(self, temperature):
        assert ModelConfig(model="m", temperature=temperature).temperature == temperature

    @pytest.mark.parametrize("temperature", [-0.1, 2.01, 5])
    def test_rejects_temperature_out_of_range(self, temperature):
        with pytest.raises(ValidationError, match="temperature"):
            ModelConfig(model="m", temperature=temperature)

    @pytest.mark.parametrize("max_tokens", [0, -1, 16385])
    def test_rejects_invalid_max_tokens(self, max_tokens):
        with pytest.raises(ValidationError, match="max_tokens"):
            ModelConfig(model="m", max_tokens=max_tokens)

    @pytest.mark.parametrize("top_p", [-0.1, 1.1])
    def test_rejects_invalid_top_p(self, top_p):
        with pytest.raises(ValidationError, match="top_p"):
            ModelConfig(model="m", top_p=top_p)

    def test_rejects_empty_model(self):
        with pytest.raises(ValidationError):
            ModelConfig(model="")

    def test_coerces_numeric_strings(self):
        # Útil al leer del .env, donde todo es string
        cfg = ModelConfig(model="m", temperature="0.5", max_tokens="100")
        assert cfg.temperature == 0.5 and cfg.max_tokens == 100


class TestModelResponse:
    def test_success_is_computed_from_error(self):
        ok = ModelResponse(provider="openai", model="m", content="hola")
        failed = ModelResponse(
            provider="openai", model="m", error=LLMError(type=ErrorType.AUTH, message="x", retryable=False)
        )
        assert ok.success is True
        assert failed.success is False

    def test_success_appears_in_json(self):
        assert ModelResponse(provider="openai", model="m").model_dump()["success"] is True

    def test_total_tokens(self):
        assert Usage(input_tokens=10, output_tokens=5).total_tokens == 15


class TestRetryPolicy:
    def _error(self, retry_after=None):
        return LLMError(type=ErrorType.RATE_LIMIT, message="x", retryable=True, retry_after=retry_after)

    def test_delay_respects_retry_after(self):
        assert RetryPolicy().delay_for(0, self._error(retry_after=4)) == 4

    def test_retry_after_is_capped(self):
        assert RetryPolicy(max_delay=10).delay_for(0, self._error(retry_after=999)) == 10

    @pytest.mark.parametrize("attempt", range(6))
    def test_exponential_backoff_bounds(self, attempt):
        policy = RetryPolicy(base_delay=1, max_delay=8)
        ceiling = min(2**attempt, 8)
        for _ in range(50):  # el jitter es aleatorio: muestreamos varias veces
            assert 0 <= policy.delay_for(attempt, self._error()) <= ceiling

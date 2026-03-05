"""Tests for the capability template method pattern on LLMProvider.

Verifies that:
- get_capabilities() is a concrete template method that calls _detect_capabilities()
- _resolve_capabilities() config overrides work for ALL providers
- None values are filtered from _detect_capabilities() results
- Each provider detects expected capabilities for known models
"""

import pytest

from dataknobs_llm import (
    EchoProvider,
    LLMConfig,
    ModelCapability,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    HuggingFaceProvider,
)


# ---------------------------------------------------------------------------
# Config-override tests — the core fix: ALL providers must respect overrides
# ---------------------------------------------------------------------------


class TestConfigCapabilityOverrides:
    """Config capability overrides must work for every provider, not just Ollama."""

    @pytest.fixture()
    def override_config(self):
        """Config with explicit capability overrides."""
        return {
            "provider": "echo",
            "model": "test",
            "capabilities": ["text_generation", "json_mode"],
        }

    def _make_config(self, provider: str, model: str, capabilities: list[str]):
        return LLMConfig.from_dict({
            "provider": provider,
            "model": model,
            "capabilities": capabilities,
        })

    def test_echo_respects_config_overrides(self):
        config = self._make_config("echo", "test", ["text_generation", "json_mode"])
        provider = EchoProvider(config)
        caps = provider.get_capabilities()
        assert caps == [ModelCapability.TEXT_GENERATION, ModelCapability.JSON_MODE]

    def test_ollama_respects_config_overrides(self):
        config = self._make_config("ollama", "llama3.2", ["text_generation", "embeddings"])
        provider = OllamaProvider(config)
        caps = provider.get_capabilities()
        assert caps == [ModelCapability.TEXT_GENERATION, ModelCapability.EMBEDDINGS]

    def test_openai_respects_config_overrides(self):
        config = self._make_config("openai", "gpt-4o", ["text_generation", "chat"])
        provider = OpenAIProvider(config)
        caps = provider.get_capabilities()
        assert caps == [ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]

    def test_anthropic_respects_config_overrides(self):
        config = self._make_config("anthropic", "claude-3-sonnet", ["text_generation"])
        provider = AnthropicProvider(config)
        caps = provider.get_capabilities()
        assert caps == [ModelCapability.TEXT_GENERATION]

    def test_huggingface_respects_config_overrides(self):
        config = self._make_config("huggingface", "bert-base", ["embeddings"])
        provider = HuggingFaceProvider(config)
        caps = provider.get_capabilities()
        assert caps == [ModelCapability.EMBEDDINGS]


# ---------------------------------------------------------------------------
# None filtering — the HuggingFace bug fix
# ---------------------------------------------------------------------------


class TestNoneFiltering:
    """Template method must filter None from _detect_capabilities()."""

    def test_huggingface_no_none_in_capabilities(self):
        """HuggingFace was returning [TEXT_GENERATION, None] for non-embedding models."""
        config = LLMConfig.from_dict({
            "provider": "huggingface",
            "model": "gpt2",
        })
        provider = HuggingFaceProvider(config)
        caps = provider.get_capabilities()
        assert None not in caps
        assert ModelCapability.TEXT_GENERATION in caps

    def test_huggingface_embedding_model(self):
        config = LLMConfig.from_dict({
            "provider": "huggingface",
            "model": "text-embedding-base",
        })
        provider = HuggingFaceProvider(config)
        caps = provider.get_capabilities()
        assert None not in caps
        assert ModelCapability.EMBEDDINGS in caps


# ---------------------------------------------------------------------------
# Provider-specific capability detection
# ---------------------------------------------------------------------------


class TestOllamaCapabilityDetection:
    """Ollama should detect JSON_MODE and EMBEDDINGS."""

    def _caps(self, model: str) -> list[ModelCapability]:
        config = LLMConfig.from_dict({"provider": "ollama", "model": model})
        return OllamaProvider(config).get_capabilities()

    def test_llama3_has_json_mode(self):
        caps = self._caps("llama3.2")
        assert ModelCapability.JSON_MODE in caps

    def test_llama3_has_function_calling(self):
        caps = self._caps("llama3.2")
        assert ModelCapability.FUNCTION_CALLING in caps

    def test_embeddings_always_present(self):
        caps = self._caps("llama3.2")
        assert ModelCapability.EMBEDDINGS in caps

    def test_deepseek_has_json_mode(self):
        caps = self._caps("deepseek-r1")
        assert ModelCapability.JSON_MODE in caps

    def test_gemma_has_json_mode(self):
        caps = self._caps("gemma2")
        assert ModelCapability.JSON_MODE in caps

    def test_vision_model(self):
        caps = self._caps("llava")
        assert ModelCapability.VISION in caps

    def test_code_model(self):
        caps = self._caps("codellama")
        assert ModelCapability.CODE in caps


class TestOpenAICapabilityDetection:
    """OpenAI should detect capabilities for modern models."""

    def _caps(self, model: str) -> list[ModelCapability]:
        config = LLMConfig.from_dict({"provider": "openai", "model": model})
        return OpenAIProvider(config).get_capabilities()

    def test_gpt4o_has_json_mode(self):
        caps = self._caps("gpt-4o")
        assert ModelCapability.JSON_MODE in caps
        assert ModelCapability.FUNCTION_CALLING in caps

    def test_gpt4o_has_vision(self):
        caps = self._caps("gpt-4o")
        assert ModelCapability.VISION in caps

    def test_o1_has_json_mode(self):
        caps = self._caps("o1-preview")
        assert ModelCapability.JSON_MODE in caps
        assert ModelCapability.FUNCTION_CALLING in caps

    def test_o3_has_json_mode(self):
        caps = self._caps("o3-mini")
        assert ModelCapability.JSON_MODE in caps

    def test_embedding_model(self):
        caps = self._caps("text-embedding-3-small")
        assert ModelCapability.EMBEDDINGS in caps


class TestAnthropicCapabilityDetection:
    """Anthropic should detect JSON_MODE for Claude 3+ models."""

    def _caps(self, model: str) -> list[ModelCapability]:
        config = LLMConfig.from_dict({"provider": "anthropic", "model": model})
        return AnthropicProvider(config).get_capabilities()

    def test_claude3_has_json_mode(self):
        caps = self._caps("claude-3-sonnet-20240229")
        assert ModelCapability.JSON_MODE in caps

    def test_claude35_has_json_mode(self):
        caps = self._caps("claude-3.5-sonnet-20240620")
        assert ModelCapability.JSON_MODE in caps

    def test_claude4_has_json_mode(self):
        caps = self._caps("claude-4-sonnet")
        assert ModelCapability.JSON_MODE in caps

    def test_claude3_has_function_calling(self):
        caps = self._caps("claude-3-opus")
        assert ModelCapability.FUNCTION_CALLING in caps

    def test_claude3_has_vision(self):
        caps = self._caps("claude-3-haiku")
        assert ModelCapability.VISION in caps

    def test_legacy_model_no_json_mode(self):
        """Claude 2.x should not have JSON_MODE."""
        caps = self._caps("claude-2.1")
        assert ModelCapability.JSON_MODE not in caps


class TestHuggingFaceCapabilityDetection:
    """HuggingFace chat-capable model detection."""

    def _caps(self, model: str) -> list[ModelCapability]:
        config = LLMConfig.from_dict({"provider": "huggingface", "model": model})
        return HuggingFaceProvider(config).get_capabilities()

    def test_chat_model_has_chat(self):
        caps = self._caps("meta-llama/Llama-2-7b-chat-hf")
        assert ModelCapability.CHAT in caps

    def test_instruct_model_has_chat(self):
        caps = self._caps("mistralai/Mistral-7B-Instruct-v0.1")
        assert ModelCapability.CHAT in caps

    def test_base_model_no_chat(self):
        caps = self._caps("gpt2")
        assert ModelCapability.CHAT not in caps

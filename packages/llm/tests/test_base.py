"""Tests for dataknobs_llm.llm.base module."""

import pytest
from datetime import datetime
from dataknobs_llm.llm.base import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    CompletionMode,
    ModelCapability,
    normalize_llm_config,
)


def test_completion_mode_enum():
    """Test CompletionMode enum values."""
    assert CompletionMode.CHAT.value == "chat"
    assert CompletionMode.TEXT.value == "text"
    assert CompletionMode.INSTRUCT.value == "instruct"
    assert CompletionMode.EMBEDDING.value == "embedding"
    assert CompletionMode.FUNCTION.value == "function"


def test_model_capability_enum():
    """Test ModelCapability enum values."""
    assert ModelCapability.TEXT_GENERATION.value == "text_generation"
    assert ModelCapability.CHAT.value == "chat"
    assert ModelCapability.EMBEDDINGS.value == "embeddings"
    assert ModelCapability.FUNCTION_CALLING.value == "function_calling"
    assert ModelCapability.VISION.value == "vision"
    assert ModelCapability.CODE.value == "code"
    assert ModelCapability.JSON_MODE.value == "json_mode"
    assert ModelCapability.STREAMING.value == "streaming"


def test_llm_message_basic():
    """Test basic LLMMessage creation."""
    msg = LLMMessage(role="user", content="Hello, world!")
    assert msg.role == "user"
    assert msg.content == "Hello, world!"
    assert msg.name is None
    assert msg.function_call is None
    assert msg.metadata == {}


def test_llm_message_with_metadata():
    """Test LLMMessage with metadata."""
    metadata = {"timestamp": "2025-01-01", "source": "test"}
    msg = LLMMessage(
        role="assistant",
        content="Response",
        metadata=metadata
    )
    assert msg.metadata == metadata


def test_llm_message_with_function_call():
    """Test LLMMessage with function call."""
    func_call = {"name": "get_weather", "arguments": {"city": "SF"}}
    msg = LLMMessage(
        role="assistant",
        content="",
        function_call=func_call
    )
    assert msg.function_call == func_call


def test_llm_response_basic():
    """Test basic LLMResponse creation."""
    response = LLMResponse(content="Hello!", model="gpt-4")
    assert response.content == "Hello!"
    assert response.model == "gpt-4"
    assert response.finish_reason is None
    assert response.usage is None
    assert response.function_call is None
    assert response.metadata == {}
    assert isinstance(response.created_at, datetime)


def test_llm_response_with_usage():
    """Test LLMResponse with token usage."""
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
    response = LLMResponse(
        content="Response",
        model="gpt-4",
        usage=usage,
        finish_reason="stop"
    )
    assert response.usage == usage
    assert response.finish_reason == "stop"


def test_llm_stream_response():
    """Test LLMStreamResponse."""
    stream_resp = LLMStreamResponse(delta="Hello", is_final=False)
    assert stream_resp.delta == "Hello"
    assert stream_resp.is_final is False
    assert stream_resp.finish_reason is None
    assert stream_resp.usage is None

    final_resp = LLMStreamResponse(
        delta="!",
        is_final=True,
        finish_reason="stop",
        usage={"total_tokens": 10}
    )
    assert final_resp.is_final is True
    assert final_resp.finish_reason == "stop"
    assert final_resp.usage == {"total_tokens": 10}


def test_llm_config_basic():
    """Test basic LLMConfig creation."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4"
    )
    assert config.provider == "openai"
    assert config.model == "gpt-4"
    assert config.temperature == 0.7
    assert config.mode == CompletionMode.CHAT
    assert config.stream is False


def test_llm_config_custom_params():
    """Test LLMConfig with custom parameters."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-opus",
        temperature=0.5,
        max_tokens=2000,
        top_p=0.9,
        mode=CompletionMode.INSTRUCT,
        system_prompt="You are a helpful assistant"
    )
    assert config.provider == "anthropic"
    assert config.model == "claude-3-opus"
    assert config.temperature == 0.5
    assert config.max_tokens == 2000
    assert config.top_p == 0.9
    assert config.mode == CompletionMode.INSTRUCT
    assert config.system_prompt == "You are a helpful assistant"


def test_llm_config_from_dict_basic():
    """Test LLMConfig.from_dict with basic dict."""
    config_dict = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.8
    }
    config = LLMConfig.from_dict(config_dict)
    assert config.provider == "openai"
    assert config.model == "gpt-4"
    assert config.temperature == 0.8


def test_llm_config_from_dict_filters_config_attrs():
    """Test that from_dict filters out Config-specific attributes."""
    config_dict = {
        "type": "llm",  # Config-specific
        "name": "my-llm",  # Config-specific
        "factory": "SomeFactory",  # Config-specific
        "provider": "openai",
        "model": "gpt-4"
    }
    config = LLMConfig.from_dict(config_dict)
    assert config.provider == "openai"
    assert config.model == "gpt-4"
    # type, name, factory should not cause errors


def test_llm_config_from_dict_mode_string():
    """Test from_dict converts mode string to CompletionMode enum."""
    config_dict = {
        "provider": "openai",
        "model": "gpt-4",
        "mode": "chat"  # String value
    }
    config = LLMConfig.from_dict(config_dict)
    assert config.mode == CompletionMode.CHAT
    assert isinstance(config.mode, CompletionMode)


def test_llm_config_from_dict_filters_unknown_fields():
    """Test from_dict filters out unknown attributes."""
    config_dict = {
        "provider": "openai",
        "model": "gpt-4",
        "unknown_field": "should_be_ignored",
        "another_unknown": 123
    }
    config = LLMConfig.from_dict(config_dict)
    assert config.provider == "openai"
    assert config.model == "gpt-4"
    # Unknown fields should be filtered without error


def test_llm_config_to_dict_basic():
    """Test LLMConfig.to_dict basic conversion."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.8
    )
    config_dict = config.to_dict()
    assert config_dict["provider"] == "openai"
    assert config_dict["model"] == "gpt-4"
    assert config_dict["temperature"] == 0.8


def test_llm_config_to_dict_enum_conversion():
    """Test to_dict converts enums to values."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        mode=CompletionMode.INSTRUCT
    )
    config_dict = config.to_dict()
    assert config_dict["mode"] == "instruct"
    assert isinstance(config_dict["mode"], str)


def test_llm_config_to_dict_with_config_attrs():
    """Test to_dict with include_config_attrs=True."""
    config = LLMConfig(provider="openai", model="gpt-4")
    config_dict = config.to_dict(include_config_attrs=True)
    assert config_dict["type"] == "llm"
    assert config_dict["provider"] == "openai"


def test_llm_config_to_dict_excludes_none():
    """Test to_dict excludes None values for optional fields."""
    config = LLMConfig(provider="openai", model="gpt-4")
    config_dict = config.to_dict()
    # None values should not be in dict
    assert "api_key" not in config_dict or config_dict.get("api_key") is not None
    # But options should be included even if empty
    assert "options" in config_dict
    assert config_dict["options"] == {}


def test_normalize_llm_config_with_llm_config():
    """Test normalize_llm_config with LLMConfig instance."""
    config = LLMConfig(provider="openai", model="gpt-4")
    normalized = normalize_llm_config(config)
    assert normalized is config
    assert normalized.provider == "openai"
    assert normalized.model == "gpt-4"


def test_normalize_llm_config_with_dict():
    """Test normalize_llm_config with dictionary."""
    config_dict = {
        "provider": "anthropic",
        "model": "claude-3-opus",
        "temperature": 0.5
    }
    normalized = normalize_llm_config(config_dict)
    assert isinstance(normalized, LLMConfig)
    assert normalized.provider == "anthropic"
    assert normalized.model == "claude-3-opus"
    assert normalized.temperature == 0.5


def test_normalize_llm_config_with_invalid_type():
    """Test normalize_llm_config raises TypeError for invalid types."""
    with pytest.raises(TypeError) as exc_info:
        normalize_llm_config("invalid_string")
    assert "Unsupported config type" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        normalize_llm_config(123)
    assert "Unsupported config type" in str(exc_info.value)


def test_llm_config_round_trip():
    """Test LLMConfig round-trip through dict."""
    original = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.8,
        max_tokens=1000,
        mode=CompletionMode.CHAT,
        system_prompt="Test prompt"
    )

    # Convert to dict and back
    config_dict = original.to_dict()
    restored = LLMConfig.from_dict(config_dict)

    assert restored.provider == original.provider
    assert restored.model == original.model
    assert restored.temperature == original.temperature
    assert restored.max_tokens == original.max_tokens
    assert restored.mode == original.mode
    assert restored.system_prompt == original.system_prompt


def test_llm_config_with_options():
    """Test LLMConfig with provider-specific options."""
    config = LLMConfig(
        provider="echo",
        model="echo-model",
        options={
            "echo_prefix": "Test: ",
            "embedding_dim": 384,
            "mock_tokens": True
        }
    )
    assert config.options["echo_prefix"] == "Test: "
    assert config.options["embedding_dim"] == 384
    assert config.options["mock_tokens"] is True

    # Test round-trip with options
    config_dict = config.to_dict()
    restored = LLMConfig.from_dict(config_dict)
    assert restored.options == config.options


def test_llm_config_with_functions():
    """Test LLMConfig with function calling parameters."""
    functions = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    ]
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        functions=functions,
        function_call="auto"
    )
    assert config.functions == functions
    assert config.function_call == "auto"


def test_llm_config_with_stop_sequences():
    """Test LLMConfig with stop sequences."""
    stop_sequences = ["END", "STOP", "\n\n"]
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        stop_sequences=stop_sequences
    )
    assert config.stop_sequences == stop_sequences

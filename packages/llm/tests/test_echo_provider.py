"""Tests for EchoProvider."""

import pytest
import asyncio
from dataknobs_llm.llm.base import LLMConfig, LLMMessage, ModelCapability
from dataknobs_llm.llm.providers import EchoProvider


@pytest.fixture
def echo_config():
    """Create basic echo config."""
    return LLMConfig(
        provider="echo",
        model="echo-model"
    )


@pytest.fixture
def echo_config_custom():
    """Create echo config with custom options."""
    return LLMConfig(
        provider="echo",
        model="echo-model",
        options={
            "echo_prefix": "Test: ",
            "embedding_dim": 384,
            "mock_tokens": True,
            "stream_delay": 0.0
        }
    )


@pytest.fixture
async def echo_provider(echo_config):
    """Create and initialize EchoProvider."""
    provider = EchoProvider(echo_config)
    await provider.initialize()
    yield provider
    await provider.close()


@pytest.fixture
async def echo_provider_custom(echo_config_custom):
    """Create and initialize EchoProvider with custom config."""
    provider = EchoProvider(echo_config_custom)
    await provider.initialize()
    yield provider
    await provider.close()


def test_echo_provider_creation(echo_config):
    """Test EchoProvider creation."""
    provider = EchoProvider(echo_config)
    assert provider.config.provider == "echo"
    assert provider.config.model == "echo-model"
    assert provider.echo_prefix == "Echo: "
    assert provider.embedding_dim == 768
    assert provider.mock_tokens is True


def test_echo_provider_custom_options(echo_config_custom):
    """Test EchoProvider with custom options."""
    provider = EchoProvider(echo_config_custom)
    assert provider.echo_prefix == "Test: "
    assert provider.embedding_dim == 384
    assert provider.mock_tokens is True


def test_echo_provider_from_dict():
    """Test EchoProvider creation from dict."""
    config_dict = {
        "provider": "echo",
        "model": "echo-model",
        "options": {"echo_prefix": ">>>"}
    }
    provider = EchoProvider(config_dict)
    assert provider.echo_prefix == ">>>"


@pytest.mark.asyncio
async def test_echo_provider_initialize_close(echo_config):
    """Test EchoProvider initialization and closing."""
    provider = EchoProvider(echo_config)
    assert not provider.is_initialized

    await provider.initialize()
    assert provider.is_initialized

    await provider.close()
    assert not provider.is_initialized


@pytest.mark.asyncio
async def test_echo_provider_validate_model(echo_provider):
    """Test EchoProvider validate_model."""
    is_valid = await echo_provider.validate_model()
    assert is_valid is True


def test_echo_provider_get_capabilities(echo_provider):
    """Test EchoProvider get_capabilities."""
    capabilities = echo_provider.get_capabilities()
    assert ModelCapability.TEXT_GENERATION in capabilities
    assert ModelCapability.CHAT in capabilities
    assert ModelCapability.EMBEDDINGS in capabilities
    assert ModelCapability.FUNCTION_CALLING in capabilities
    assert ModelCapability.STREAMING in capabilities
    assert ModelCapability.JSON_MODE in capabilities


@pytest.mark.asyncio
async def test_echo_complete_string(echo_provider):
    """Test EchoProvider complete with string input."""
    response = await echo_provider.complete("Hello, world!")

    assert response.content == "Echo: Hello, world!"
    assert response.model == "echo-model"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage["total_tokens"] > 0


@pytest.mark.asyncio
async def test_echo_complete_messages(echo_provider):
    """Test EchoProvider complete with message list."""
    messages = [
        LLMMessage(role="system", content="You are helpful"),
        LLMMessage(role="user", content="What is 2+2?")
    ]
    response = await echo_provider.complete(messages)

    assert response.content == "Echo: What is 2+2?"
    assert response.usage["prompt_tokens"] > 0


@pytest.mark.asyncio
async def test_echo_complete_multiple_user_messages(echo_provider):
    """Test EchoProvider echoes last user message."""
    messages = [
        LLMMessage(role="user", content="First message"),
        LLMMessage(role="assistant", content="Response"),
        LLMMessage(role="user", content="Second message")
    ]
    response = await echo_provider.complete(messages)

    assert response.content == "Echo: Second message"


@pytest.mark.asyncio
async def test_echo_complete_no_user_message(echo_provider):
    """Test EchoProvider with no user messages."""
    messages = [
        LLMMessage(role="system", content="You are helpful")
    ]
    response = await echo_provider.complete(messages)

    assert response.content == "Echo: (no user message)"


@pytest.mark.asyncio
async def test_echo_complete_custom_prefix(echo_provider_custom):
    """Test EchoProvider with custom prefix."""
    response = await echo_provider_custom.complete("Hello")

    assert response.content == "Test: Hello"


@pytest.mark.asyncio
async def test_echo_complete_auto_initialize(echo_config):
    """Test EchoProvider auto-initializes on first use."""
    provider = EchoProvider(echo_config)
    assert not provider.is_initialized

    response = await provider.complete("Test")
    assert provider.is_initialized
    assert response.content == "Echo: Test"

    await provider.close()


@pytest.mark.asyncio
async def test_echo_stream_complete(echo_provider):
    """Test EchoProvider stream_complete."""
    messages = [LLMMessage(role="user", content="Hi")]
    chunks = []
    deltas = []

    async for chunk in echo_provider.stream_complete(messages):
        chunks.append(chunk)
        deltas.append(chunk.delta)

    # Reconstruct full response
    full_content = "".join(deltas)
    assert full_content == "Echo: Hi"

    # Check final chunk
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None

    # Check non-final chunks
    for chunk in chunks[:-1]:
        assert chunk.is_final is False
        assert chunk.finish_reason is None


@pytest.mark.asyncio
async def test_echo_stream_complete_character_by_character(echo_provider):
    """Test EchoProvider streams character by character."""
    response = await echo_provider.complete("AB")
    expected = response.content  # "Echo: AB"

    chunks = []
    async for chunk in echo_provider.stream_complete("AB"):
        chunks.append(chunk.delta)

    # Should have one chunk per character
    assert len(chunks) == len(expected)
    assert "".join(chunks) == expected


@pytest.mark.asyncio
async def test_echo_embed_single_text(echo_provider):
    """Test EchoProvider embed with single text."""
    embedding = await echo_provider.embed("Hello, world!")

    assert isinstance(embedding, list)
    assert len(embedding) == 768  # Default embedding_dim
    assert all(isinstance(x, float) for x in embedding)
    assert all(-1.0 <= x <= 1.0 for x in embedding)


@pytest.mark.asyncio
async def test_echo_embed_multiple_texts(echo_provider):
    """Test EchoProvider embed with multiple texts."""
    texts = ["First text", "Second text", "Third text"]
    embeddings = await echo_provider.embed(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 768
        assert all(-1.0 <= x <= 1.0 for x in emb)


@pytest.mark.asyncio
async def test_echo_embed_deterministic(echo_provider):
    """Test EchoProvider embeddings are deterministic."""
    text = "Test text for embedding"

    embedding1 = await echo_provider.embed(text)
    embedding2 = await echo_provider.embed(text)

    # Same text should produce same embedding
    assert embedding1 == embedding2


@pytest.mark.asyncio
async def test_echo_embed_different_texts(echo_provider):
    """Test EchoProvider different texts produce different embeddings."""
    embedding1 = await echo_provider.embed("First text")
    embedding2 = await echo_provider.embed("Second text")

    # Different texts should produce different embeddings
    assert embedding1 != embedding2


@pytest.mark.asyncio
async def test_echo_embed_custom_dim(echo_provider_custom):
    """Test EchoProvider with custom embedding dimension."""
    embedding = await echo_provider_custom.embed("Test")

    assert len(embedding) == 384


@pytest.mark.asyncio
async def test_echo_function_call_basic(echo_provider):
    """Test EchoProvider function_call basic usage."""
    messages = [
        LLMMessage(role="user", content="What's the weather in NYC?")
    ]
    functions = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units"
                    }
                }
            }
        }
    ]

    response = await echo_provider.function_call(messages, functions)

    assert response.finish_reason == "function_call"
    assert response.function_call is not None
    assert response.function_call["name"] == "get_weather"
    assert "location" in response.function_call["arguments"]
    assert "units" in response.function_call["arguments"]


@pytest.mark.asyncio
async def test_echo_function_call_mock_arguments(echo_provider):
    """Test EchoProvider generates mock arguments by type."""
    messages = [LLMMessage(role="user", content="Test function call")]
    functions = [
        {
            "name": "test_function",
            "parameters": {
                "type": "object",
                "properties": {
                    "str_param": {"type": "string"},
                    "num_param": {"type": "number"},
                    "int_param": {"type": "integer"},
                    "bool_param": {"type": "boolean"},
                    "array_param": {"type": "array"},
                    "obj_param": {"type": "object"}
                }
            }
        }
    ]

    response = await echo_provider.function_call(messages, functions)
    args = response.function_call["arguments"]

    assert isinstance(args["str_param"], str)
    assert "mock_str_param_from_echo" in args["str_param"]
    assert isinstance(args["num_param"], int)
    assert isinstance(args["int_param"], int)
    assert isinstance(args["bool_param"], bool)
    assert isinstance(args["array_param"], list)
    assert isinstance(args["obj_param"], dict)


@pytest.mark.asyncio
async def test_echo_function_call_deterministic(echo_provider):
    """Test EchoProvider function calls are deterministic."""
    messages = [LLMMessage(role="user", content="Same message")]
    functions = [
        {
            "name": "test_func",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"}
                }
            }
        }
    ]

    response1 = await echo_provider.function_call(messages, functions)
    response2 = await echo_provider.function_call(messages, functions)

    # Same input should produce same mock arguments
    assert response1.function_call["arguments"] == response2.function_call["arguments"]


@pytest.mark.asyncio
async def test_echo_function_call_no_functions(echo_provider):
    """Test EchoProvider function_call with no functions falls back to complete."""
    messages = [LLMMessage(role="user", content="Hello")]
    response = await echo_provider.function_call(messages, [])

    # Should fall back to regular complete
    assert response.content == "Echo: Hello"
    assert response.function_call is None


@pytest.mark.asyncio
async def test_echo_function_call_multiple_functions(echo_provider):
    """Test EchoProvider uses first function when multiple provided."""
    messages = [LLMMessage(role="user", content="Test")]
    functions = [
        {"name": "first_function", "parameters": {"type": "object", "properties": {}}},
        {"name": "second_function", "parameters": {"type": "object", "properties": {}}}
    ]

    response = await echo_provider.function_call(messages, functions)

    # Should use first function
    assert response.function_call["name"] == "first_function"


@pytest.mark.asyncio
async def test_echo_token_counting(echo_provider):
    """Test EchoProvider token counting."""
    # Test with different message lengths
    short_response = await echo_provider.complete("Hi")
    long_response = await echo_provider.complete("This is a much longer message with many words")

    assert short_response.usage["total_tokens"] < long_response.usage["total_tokens"]
    assert short_response.usage["prompt_tokens"] < long_response.usage["prompt_tokens"]


@pytest.mark.asyncio
async def test_echo_context_manager(echo_config):
    """Test EchoProvider as async context manager."""
    async with EchoProvider(echo_config) as provider:
        assert provider.is_initialized
        response = await provider.complete("Test")
        assert response.content == "Echo: Test"

    # Should be closed after context
    assert not provider.is_initialized


def test_echo_generate_embedding_consistency():
    """Test internal _generate_embedding method consistency."""
    config = LLMConfig(provider="echo", model="echo-model")
    provider = EchoProvider(config)

    # Same input should always produce same output
    emb1 = provider._generate_embedding("test text")
    emb2 = provider._generate_embedding("test text")
    assert emb1 == emb2

    # Different input should produce different output
    emb3 = provider._generate_embedding("different text")
    assert emb1 != emb3


def test_echo_count_tokens():
    """Test internal _count_tokens method."""
    config = LLMConfig(provider="echo", model="echo-model")
    provider = EchoProvider(config)

    # Roughly 1 token per 4 characters
    assert provider._count_tokens("") == 1  # Minimum 1
    assert provider._count_tokens("1234") == 1
    assert provider._count_tokens("12345678") == 2
    assert provider._count_tokens("x" * 100) == 25


@pytest.mark.asyncio
async def test_echo_with_system_prompt(echo_config):
    """Test EchoProvider with system prompt in config."""
    config = LLMConfig(
        provider="echo",
        model="echo-model",
        system_prompt="You are helpful",
        options={"echo_system": True}
    )
    provider = EchoProvider(config)
    await provider.initialize()

    response = await provider.complete("Hello")

    assert "[System: You are helpful]" in response.content
    assert "Echo: Hello" in response.content

    await provider.close()


@pytest.mark.asyncio
async def test_echo_system_prompt_disabled_by_default(echo_config):
    """Test EchoProvider doesn't echo system prompt by default."""
    config = LLMConfig(
        provider="echo",
        model="echo-model",
        system_prompt="You are helpful"
    )
    provider = EchoProvider(config)
    await provider.initialize()

    response = await provider.complete("Hello")

    assert "[System:" not in response.content
    assert response.content == "Echo: Hello"

    await provider.close()


@pytest.mark.asyncio
async def test_echo_usage_in_integration():
    """Test EchoProvider in integration scenario."""
    # Simulate a complete workflow
    config = LLMConfig(
        provider="echo",
        model="echo-model",
        options={"echo_prefix": "[DEBUG] "}
    )

    async with EchoProvider(config) as provider:
        # Test completion
        messages = [
            LLMMessage(role="system", content="Test system"),
            LLMMessage(role="user", content="Hello")
        ]
        response = await provider.complete(messages)
        assert "[DEBUG] Hello" in response.content

        # Test embedding
        embedding = await provider.embed("test")
        assert len(embedding) == 768

        # Test function calling
        funcs = [{"name": "test", "parameters": {"type": "object", "properties": {}}}]
        func_response = await provider.function_call(messages, funcs)
        assert func_response.function_call["name"] == "test"

        # Test streaming
        chunks = []
        async for chunk in provider.stream_complete("Hi"):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert chunks[-1].is_final

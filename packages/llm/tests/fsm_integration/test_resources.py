"""Tests for FSM LLM resources.

These tests verify the LLM resource providers for FSM,
including the AsyncLLMResource with rate limiting integration.
"""

import pytest
from dataknobs_llm.fsm_integration import (
    LLMProvider,
    LLMSession,
    LLMResource,
    AsyncLLMResource,
)
from dataknobs_llm.llm import LLMConfig, EchoProvider, LLMResponse
from dataknobs_llm.llm.providers.echo import ErrorResponse
from dataknobs_llm.testing import text_response
from dataknobs_common.ratelimit import InMemoryRateLimiter
from dataknobs_fsm.functions.base import ResourceError


def test_resource_imports():
    """Test that resource classes can be imported."""
    assert LLMProvider is not None
    assert LLMSession is not None
    assert LLMResource is not None
    assert AsyncLLMResource is not None


def test_llm_provider_enum():
    """Test LLMProvider enum values."""
    assert LLMProvider.OPENAI.value == "openai"
    assert LLMProvider.ANTHROPIC.value == "anthropic"
    assert LLMProvider.OLLAMA.value == "ollama"
    assert LLMProvider.HUGGINGFACE.value == "huggingface"


def _make_echo_provider(
    prefix: str = "",
    responses: list | None = None,
) -> EchoProvider:
    """Create an EchoProvider instance for testing."""
    config = LLMConfig(
        provider="echo",
        model="echo-test",
        options={"echo_prefix": prefix},
    )
    provider = EchoProvider(config, responses=responses)
    return provider


def _make_resource(
    provider: EchoProvider | None = None,
    requests_per_minute: int = 0,
) -> AsyncLLMResource:
    """Create an AsyncLLMResource with an injected EchoProvider."""
    if provider is None:
        provider = _make_echo_provider()
    return AsyncLLMResource(
        "test-llm",
        provider="echo",
        model="echo-test",
        async_provider=provider,
        requests_per_minute=requests_per_minute,
    )


class TestAsyncLLMResourceInit:
    """Test AsyncLLMResource initialization."""

    def test_default_init(self):
        """Test default construction with Ollama defaults."""
        resource = AsyncLLMResource("test-llm")
        assert resource.provider == LLMProvider.OLLAMA
        assert resource.model == "llama3.2"
        assert resource.api_key is None
        assert resource._async_provider is None
        assert resource._rate_limiter is None

    def test_custom_provider_string(self):
        """Test construction with custom provider string."""
        resource = AsyncLLMResource("test-llm", provider="openai", model="gpt-4")
        assert resource.provider == LLMProvider.OPENAI
        assert resource.model == "gpt-4"

    def test_unknown_provider_preserves_raw_name(self):
        """Test that unknown provider string is preserved for the factory."""
        resource = AsyncLLMResource("test-llm", provider="echo")
        assert resource.provider == LLMProvider.CUSTOM
        assert resource._provider_name == "echo"

    def test_provider_enum_value(self):
        """Test construction with LLMProvider enum."""
        resource = AsyncLLMResource("test-llm", provider=LLMProvider.ANTHROPIC)
        assert resource.provider == LLMProvider.ANTHROPIC

    def test_injected_provider(self):
        """Test construction with a pre-built async provider."""
        echo = _make_echo_provider()
        resource = AsyncLLMResource("test-llm", async_provider=echo)
        assert resource._async_provider is echo

    def test_rate_limiter_created_with_rpm(self):
        """Test that rate limiter is created when requests_per_minute > 0."""
        resource = AsyncLLMResource("test-llm", requests_per_minute=30)
        assert resource._rate_limiter is not None
        assert isinstance(resource._rate_limiter, InMemoryRateLimiter)

    def test_no_rate_limiter_without_rpm(self):
        """Test that no rate limiter is created without requests_per_minute."""
        resource = AsyncLLMResource("test-llm")
        assert resource._rate_limiter is None

    def test_no_rate_limiter_with_zero_rpm(self):
        """Test that no rate limiter is created when requests_per_minute is 0."""
        resource = AsyncLLMResource("test-llm", requests_per_minute=0)
        assert resource._rate_limiter is None

    def test_endpoint_defaults(self):
        """Test that default endpoints are set per provider."""
        ollama = AsyncLLMResource("test", provider="ollama")
        assert ollama.endpoint == "http://localhost:11434"

        openai = AsyncLLMResource("test", provider="openai", api_key="fake")
        assert openai.endpoint == "https://api.openai.com/v1"

    def test_custom_endpoint(self):
        """Test that custom endpoint overrides default."""
        resource = AsyncLLMResource("test-llm", provider="ollama", endpoint="http://custom:8080")
        assert resource.endpoint == "http://custom:8080"

    def test_is_instance_of_llm_resource(self):
        """Test that AsyncLLMResource is an instance of LLMResource."""
        resource = AsyncLLMResource("test-llm")
        assert isinstance(resource, LLMResource)


class TestAsyncLLMResourceGenerate:
    """Test AsyncLLMResource.generate() method."""

    @pytest.mark.asyncio
    async def test_basic_generate(self):
        """Test basic text generation returns correct response format."""
        resource = _make_resource()

        result = await resource.generate(prompt="Hello world")

        assert isinstance(result, dict)
        assert "choices" in result
        assert "model" in result
        assert len(result["choices"]) == 1
        assert "text" in result["choices"][0]
        assert result["choices"][0]["index"] == 0
        assert result["choices"][0]["finish_reason"] == "stop"
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_generate_echoes_prompt(self):
        """Test that EchoProvider echoes back the prompt."""
        resource = _make_resource()

        result = await resource.generate(prompt="Test input")

        # EchoProvider with empty prefix echoes the user message content
        assert "Test input" in result["choices"][0]["text"]
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generate builds messages with system prompt."""
        resource = _make_resource()

        result = await resource.generate(
            prompt="What is 2+2?",
            system_prompt="You are a math teacher",
        )

        assert isinstance(result, dict)
        assert "choices" in result
        assert "What is 2+2?" in result["choices"][0]["text"]
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_generate_with_scripted_response(self):
        """Test generate with scripted EchoProvider responses."""
        echo = _make_echo_provider(responses=[text_response("The answer is 42")])
        resource = _make_resource(provider=echo)

        result = await resource.generate(prompt="What is the answer?")

        assert result["choices"][0]["text"] == "The answer is 42"
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_generate_with_config_overrides(self):
        """Test generate passes config overrides to provider."""
        echo = _make_echo_provider()
        resource = _make_resource(provider=echo)

        await resource.generate(
            prompt="Test",
            model="override-model",
            temperature=0.5,
            max_tokens=100,
        )

        last_call = echo.get_last_call()
        assert last_call is not None
        assert last_call["config_overrides"]["model"] == "override-model"
        assert last_call["config_overrides"]["temperature"] == 0.5
        assert last_call["config_overrides"]["max_tokens"] == 100
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_generate_records_usage(self):
        """Test that generate returns token usage from the provider."""
        echo = _make_echo_provider(
            responses=[
                LLMResponse(
                    content="response text",
                    model="echo-test",
                    usage={"prompt_tokens": 10, "completion_tokens": 20},
                ),
            ]
        )
        resource = _make_resource(provider=echo)

        result = await resource.generate(prompt="Test")

        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
        assert result["choices"][0]["text"] == "response text"
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_generate_session_lifecycle(self):
        """Test that generate acquires and releases session properly."""
        resource = _make_resource()

        initial_sessions = len(resource._sessions)
        await resource.generate(prompt="Test")

        # Session should be released (back to initial count)
        assert len(resource._sessions) == initial_sessions
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_generate_releases_session_on_error(self):
        """Test that session is released even when provider raises."""
        echo = _make_echo_provider(
            responses=[
                ErrorResponse(RuntimeError("provider error")),
            ]
        )
        resource = _make_resource(provider=echo)

        with pytest.raises(RuntimeError, match="provider error"):
            await resource.generate(prompt="Test")

        assert len(resource._sessions) == 0
        await resource.aclose()


class TestAsyncLLMResourceEmbed:
    """Test AsyncLLMResource.embed() method."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text returns list of lists."""
        resource = _make_resource()

        result = await resource.embed("Hello world")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self):
        """Test embedding multiple texts returns list of lists."""
        resource = _make_resource()

        result = await resource.embed(["Hello", "World", "Test"])

        assert isinstance(result, list)
        assert len(result) == 3
        for embedding in result:
            assert isinstance(embedding, list)
            assert all(isinstance(v, float) for v in embedding)
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_embed_deterministic(self):
        """Test that EchoProvider produces deterministic embeddings."""
        resource = _make_resource()

        result1 = await resource.embed("Same text")
        result2 = await resource.embed("Same text")

        assert result1 == result2
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_embed_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        resource = _make_resource()

        result1 = await resource.embed("Text A")
        result2 = await resource.embed("Text B")

        assert result1 != result2
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_embed_with_provided_session(self):
        """Test embed with an externally provided session."""
        resource = _make_resource()
        session = resource.acquire()

        try:
            result = await resource.embed("Test", session=session)
            assert isinstance(result, list)
            assert len(result) == 1
        finally:
            resource.release(session)
            await resource.aclose()

    @pytest.mark.asyncio
    async def test_embed_auto_acquires_session(self):
        """Test embed acquires and releases session when none provided."""
        resource = _make_resource()

        initial_sessions = len(resource._sessions)
        await resource.embed("Test")
        assert len(resource._sessions) == initial_sessions
        await resource.aclose()


class TestAsyncLLMResourceRateLimiting:
    """Test AsyncLLMResource rate limiting integration."""

    @pytest.mark.asyncio
    async def test_generate_with_rate_limiter(self):
        """Test that generate works with rate limiter enabled."""
        resource = _make_resource(requests_per_minute=5)

        result = await resource.generate(prompt="Test")

        assert isinstance(result, dict)
        assert "choices" in result
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_embed_with_rate_limiter(self):
        """Test that embed works with rate limiter enabled."""
        resource = _make_resource(requests_per_minute=5)

        result = await resource.embed("Test")

        assert isinstance(result, list)
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_rate_limiter_tracks_generate_calls(self):
        """Test that rate limiter tracks generate calls."""
        resource = _make_resource(requests_per_minute=5)
        limiter = resource._rate_limiter
        assert limiter is not None

        await resource.generate(prompt="Test")

        status = await limiter.get_status(resource.provider.value)
        assert status.current_count == 1
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_rate_limiter_tracks_embed_calls(self):
        """Test that rate limiter tracks embed calls."""
        resource = _make_resource(requests_per_minute=5)
        limiter = resource._rate_limiter
        assert limiter is not None

        await resource.embed("Test")

        status = await limiter.get_status(resource.provider.value)
        assert status.current_count == 1
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_multiple_calls_tracked(self):
        """Test that multiple calls are tracked by the rate limiter."""
        resource = _make_resource(requests_per_minute=5)
        limiter = resource._rate_limiter
        assert limiter is not None

        for i in range(3):
            await resource.generate(prompt=f"Test {i}")

        status = await limiter.get_status(resource.provider.value)
        assert status.current_count == 3
        assert status.remaining == 2  # 5 limit - 3 used
        await resource.aclose()


class TestAsyncLLMResourceClose:
    """Test AsyncLLMResource.aclose() method."""

    @pytest.mark.asyncio
    async def test_aclose_closes_provider(self):
        """Test that aclose closes the async provider."""
        resource = _make_resource()

        await resource.aclose()

        assert resource._async_provider is None

    @pytest.mark.asyncio
    async def test_aclose_closes_rate_limiter(self):
        """Test that aclose closes both provider and rate limiter."""
        resource = _make_resource(requests_per_minute=10)

        await resource.aclose()

        assert resource._async_provider is None

    @pytest.mark.asyncio
    async def test_aclose_with_no_provider(self):
        """Test that aclose is safe when provider was never initialized."""
        resource = AsyncLLMResource("test-llm")
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_aclose_with_no_rate_limiter(self):
        """Test that aclose is safe when no rate limiter is configured."""
        resource = _make_resource()

        await resource.aclose()

        assert resource._async_provider is None


class TestAsyncLLMResourceLazyInit:
    """Test AsyncLLMResource lazy provider initialization."""

    @pytest.mark.asyncio
    async def test_provider_not_initialized_on_construction(self):
        """Test that provider is not initialized during construction."""
        resource = AsyncLLMResource("test-llm")
        assert resource._async_provider is None

    @pytest.mark.asyncio
    async def test_injected_provider_used_directly(self):
        """Test that injected provider is returned by _get_provider."""
        echo = _make_echo_provider()
        resource = AsyncLLMResource("test-llm", async_provider=echo)

        provider = await resource._get_provider()

        assert provider is echo
        await resource.aclose()

    @pytest.mark.asyncio
    async def test_echo_provider_via_lazy_init(self):
        """Test that provider='echo' creates EchoProvider via the factory."""
        resource = AsyncLLMResource("test-llm", provider="echo", model="echo-test")

        # Trigger lazy initialization — should use create_llm_provider("echo")
        result = await resource.generate(prompt="Hello from factory")

        assert isinstance(result, dict)
        assert "Hello from factory" in result["choices"][0]["text"]
        await resource.aclose()


# ---------------------------------------------------------------------------
# The sync LLMResource path.
#
# Everything above exercises AsyncLLMResource, which OVERRIDES embed() and
# implements generate() separately -- so none of it reaches the sync class's
# own complete() and embed(). These do.
#
# A closed port is used wherever a cell needs a provider call to fail without
# a network: port 1 is privileged, so nothing is listening and the connection
# is refused immediately. That is also what makes the endpoint assertion
# below possible -- a call that fails *there* is a call that went there.
# ---------------------------------------------------------------------------

CLOSED_ENDPOINT = "http://127.0.0.1:1/v1"


class TestSyncResourceEmbedDoesNotInvent:
    """A vector the resource did not compute is not a vector it may return."""

    def test_a_provider_that_cannot_embed_refuses(self):
        """Anthropic has no embeddings API; the provider says so by raising.

        The resource used to answer the same question with 768 floats of
        0.1 -- a plausible-looking vector a store accepts, indexes, and
        returns as a neighbour forever.
        """
        resource = LLMResource("r", provider="anthropic", model="claude-3-5-sonnet", api_key="k")

        with pytest.raises(ResourceError) as excinfo:
            resource.embed(["hello world"])

        assert isinstance(excinfo.value.__cause__, NotImplementedError)

    def test_a_provider_the_enum_does_not_know_is_still_delegated_to(self):
        """``echo`` is a real provider that the FSM-side enum has no member for.

        It degraded to ``LLMProvider.CUSTOM`` and fell to a fabricating
        else-branch, so a working provider was invented over.
        """
        resource = LLMResource("r", provider="echo", model="embed-test")

        vectors = resource.embed(["hello world"])

        assert len(vectors) == 1
        assert len(set(vectors[0])) > 1, "a constant vector is a fabricated one"

    def test_two_texts_do_not_embed_identically(self):
        """The sharpest tell of fabrication: every text got the same vector."""
        resource = LLMResource("r", provider="echo", model="embed-test")

        first, second = resource.embed(["alpha", "beta"])

        assert first != second

    def test_a_single_string_still_returns_a_list_of_vectors(self):
        """The signature promises ``List[List[float]]`` for either input shape."""
        resource = LLMResource("r", provider="echo", model="embed-test")

        vectors = resource.embed("just one")

        assert len(vectors) == 1
        assert isinstance(vectors[0], list)
        assert all(isinstance(value, float) for value in vectors[0])

    def test_a_configured_width_reaches_the_provider(self):
        """The width rule holds through this class too, or it holds nowhere."""
        resource = LLMResource("r", provider="echo", model="embed-test", dimensions=16)

        vectors = resource.embed(["hello world"])

        assert len(vectors[0]) == 16

    def test_the_openai_path_does_not_import_a_module_that_moved(self):
        """``dataknobs_fsm.llm.base`` has not existed since the migration.

        It was the first line of ``_openai_embed``, above the try, so every
        OpenAI embed call through this class raised ``ModuleNotFoundError``
        -- which nothing noticed, because nothing called it.
        """
        resource = LLMResource(
            "r",
            provider="openai",
            model="text-embedding-3-large",
            api_key="sk-configured",
            endpoint=CLOSED_ENDPOINT,
        )

        with pytest.raises(ResourceError) as excinfo:
            resource.embed(["hello world"])

        assert not isinstance(excinfo.value.__cause__, ModuleNotFoundError)


class TestSyncResourceCompleteDoesNotInvent:
    """A failure is not a completion, and must not arrive in the text field."""

    def test_a_failure_is_raised_not_returned_as_the_models_text(self):
        """The swallow put its own error string where model output goes.

        ``{"choices": [{"text": "Error: ..."}]}`` is indistinguishable from
        a completion to every caller that reads ``choices[0]["text"]``.
        """
        resource = LLMResource(
            "r",
            provider="openai",
            model="gpt-4",
            api_key="sk-configured",
            endpoint=CLOSED_ENDPOINT,
        )

        with pytest.raises(ResourceError):
            resource.complete("Say hello")

    def test_the_configured_credentials_are_the_ones_used(self):
        """The method read ``kwargs`` then the environment, never the config.

        A resource built with an explicit key reported 'API key not
        provided' -- and reported it as the model's own words.
        """
        resource = LLMResource(
            "r",
            provider="openai",
            model="gpt-4",
            api_key="sk-configured",
            endpoint=CLOSED_ENDPOINT,
        )

        with pytest.raises(ResourceError) as excinfo:
            resource.complete("Say hello")

        # Reaching the transport at all means the key was accepted upstream
        # of it; failing on the key would mean it was never read.
        assert "API key not provided" not in str(excinfo.value.__cause__)

    def test_a_provider_the_enum_does_not_know_completes(self):
        """``_custom_complete`` refused every provider outside the enum."""
        resource = LLMResource("r", provider="echo", model="echo-test")

        response = resource.complete("Say hello")

        assert response["choices"][0]["text"]
        assert response["choices"][0]["finish_reason"] != "error"

    def test_an_unknown_provider_is_refused_by_both_operations(self):
        """One name nothing can serve, and two operations that must agree.

        They did not: ``complete`` raised and ``embed`` returned constants.
        """
        resource = LLMResource("r", provider="not-a-real-provider", model="m")

        with pytest.raises(ResourceError):
            resource.complete("Say hello")
        with pytest.raises(ResourceError):
            resource.embed(["hello world"])


class TestSyncResourceDelegation:
    """What the sync resource does with one provider it holds across calls."""

    def test_a_session_naming_another_model_completes_with_it(self):
        """One cached provider still serves a session that differs from it.

        ``complete`` has a per-call config surface, so the model travels as
        an override rather than as a second provider.
        """
        resource = LLMResource("r", provider="echo", model="echo-test")
        session = resource.acquire(model="another-model")

        try:
            response = resource.complete("Say hello", session=session)
            assert response["model"] == "another-model"
        finally:
            resource.release(session)
            resource.close()

    def test_a_named_embed_model_gets_a_provider_of_its_own(self):
        """``embed`` has no per-call config surface, so the model needs one.

        Asserted against the provider map rather than the vectors because a
        provider that ignored the name would return the same vectors as one
        that honoured it -- which is exactly how this went unnoticed
        elsewhere.
        """
        resource = LLMResource("r", provider="echo", model="echo-test")

        try:
            resource.embed(["hello world"])
            resource.embed(["hello world"], embed_model="other-embed")

            assert sorted(resource._providers) == ["echo-test", "other-embed"]
        finally:
            resource.close()

    def test_the_provider_is_built_once_and_reused(self):
        """The HuggingFace path re-loaded a model per call because of this."""
        resource = LLMResource("r", provider="echo", model="echo-test")

        try:
            resource.embed(["one"])
            first = resource._providers["echo-test"]
            resource.embed(["two"])
            resource.complete("three")

            assert resource._providers["echo-test"] is first
        finally:
            resource.close()

    def test_close_releases_the_providers_it_built(self):
        """Every provider in the map was built here, so every one is closed."""
        resource = LLMResource("r", provider="echo", model="echo-test")
        resource.embed(["hello world"])
        assert resource._providers

        resource.close()

        assert not resource._providers


class TestAsyncResourceInheritsTheSyncPath:
    """``AsyncLLMResource`` overrides ``embed`` and ``generate`` -- not ``complete``.

    So the sync ``complete`` is reachable on the async class, and the two
    halves must not collide on the helpers it calls.
    """

    def test_the_inherited_sync_complete_still_works(self):
        resource = AsyncLLMResource("r", provider="echo", model="echo-test")

        try:
            response = resource.complete("Say hello")
            assert response["choices"][0]["text"]
        finally:
            resource.close()

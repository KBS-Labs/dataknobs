"""Tests for EchoProvider scripted response features."""

import pytest

from dataknobs_llm.llm import LLMConfig, LLMResponse
from dataknobs_llm.llm.providers.echo import EchoProvider


@pytest.fixture
def echo_config() -> dict:
    """Basic echo provider configuration."""
    return {
        "provider": "echo",
        "model": "echo-test",
        "options": {"echo_prefix": ""},
    }


@pytest.fixture
def provider(echo_config: dict) -> EchoProvider:
    """Create EchoProvider instance."""
    return EchoProvider(echo_config)


class TestResponseQueue:
    """Tests for response queue functionality."""

    @pytest.mark.asyncio
    async def test_single_response(self, provider: EchoProvider) -> None:
        """Test single queued response."""
        provider.set_responses(["Hello from queue"])

        response = await provider.complete("Test input")

        assert response.content == "Hello from queue"

    @pytest.mark.asyncio
    async def test_multiple_responses_consumed_in_order(
        self, provider: EchoProvider
    ) -> None:
        """Test multiple responses are consumed in order."""
        provider.set_responses(["First", "Second", "Third"])

        r1 = await provider.complete("Input 1")
        r2 = await provider.complete("Input 2")
        r3 = await provider.complete("Input 3")

        assert r1.content == "First"
        assert r2.content == "Second"
        assert r3.content == "Third"

    @pytest.mark.asyncio
    async def test_falls_back_to_echo_when_queue_empty(
        self, provider: EchoProvider
    ) -> None:
        """Test falls back to echo when queue is exhausted."""
        provider.set_responses(["Only one"])

        r1 = await provider.complete("First call")
        r2 = await provider.complete("Second call")

        assert r1.content == "Only one"
        assert "Second call" in r2.content  # Echo behavior

    @pytest.mark.asyncio
    async def test_cycle_mode_repeats_responses(
        self, provider: EchoProvider
    ) -> None:
        """Test cycle mode repeats responses."""
        provider.set_responses(["A", "B"], cycle=True)

        r1 = await provider.complete("1")
        r2 = await provider.complete("2")
        r3 = await provider.complete("3")
        r4 = await provider.complete("4")

        assert r1.content == "A"
        assert r2.content == "B"
        assert r3.content == "A"  # Cycles back
        assert r4.content == "B"

    @pytest.mark.asyncio
    async def test_add_response_appends_to_queue(
        self, provider: EchoProvider
    ) -> None:
        """Test add_response appends to existing queue."""
        provider.set_responses(["First"])
        provider.add_response("Second")

        r1 = await provider.complete("1")
        r2 = await provider.complete("2")

        assert r1.content == "First"
        assert r2.content == "Second"


class TestResponseFunction:
    """Tests for response function functionality."""

    @pytest.mark.asyncio
    async def test_simple_response_function(
        self, provider: EchoProvider
    ) -> None:
        """Test simple response function."""
        provider.set_response_function(
            lambda msgs: f"Got {len(msgs)} messages"
        )

        response = await provider.complete("Hello")

        assert response.content == "Got 1 messages"

    @pytest.mark.asyncio
    async def test_response_function_receives_messages(
        self, provider: EchoProvider
    ) -> None:
        """Test response function receives message list."""
        captured = []

        def capture_fn(msgs):
            captured.append(msgs)
            return "Captured"

        provider.set_response_function(capture_fn)

        await provider.complete("Test message")

        assert len(captured) == 1
        assert captured[0][0].content == "Test message"

    @pytest.mark.asyncio
    async def test_response_function_can_return_llm_response(
        self, provider: EchoProvider
    ) -> None:
        """Test response function can return LLMResponse directly."""
        custom_response = LLMResponse(
            content="Custom content",
            model="custom-model",
            finish_reason="custom",
        )

        provider.set_response_function(lambda msgs: custom_response)

        response = await provider.complete("Test")

        assert response.content == "Custom content"
        assert response.model == "custom-model"
        assert response.finish_reason == "custom"

    @pytest.mark.asyncio
    async def test_response_function_has_priority_over_queue(
        self, provider: EchoProvider
    ) -> None:
        """Test response function takes priority over queue."""
        provider.set_responses(["From queue"])
        provider.set_response_function(lambda msgs: "From function")

        response = await provider.complete("Test")

        assert response.content == "From function"


class TestPatternMatching:
    """Tests for pattern matching functionality."""

    @pytest.mark.asyncio
    async def test_simple_pattern_match(
        self, provider: EchoProvider
    ) -> None:
        """Test simple pattern matching."""
        provider.add_pattern_response(r"hello", "Hi there!")

        response = await provider.complete("hello world")

        assert response.content == "Hi there!"

    @pytest.mark.asyncio
    async def test_case_insensitive_by_default(
        self, provider: EchoProvider
    ) -> None:
        """Test patterns are case-insensitive by default."""
        provider.add_pattern_response(r"hello", "Matched!")

        response = await provider.complete("HELLO")

        assert response.content == "Matched!"

    @pytest.mark.asyncio
    async def test_multiple_patterns_first_wins(
        self, provider: EchoProvider
    ) -> None:
        """Test first matching pattern wins."""
        provider.add_pattern_response(r"hello", "Hello response")
        provider.add_pattern_response(r"world", "World response")

        response = await provider.complete("hello world")

        assert response.content == "Hello response"

    @pytest.mark.asyncio
    async def test_no_match_falls_through(
        self, provider: EchoProvider
    ) -> None:
        """Test no match falls through to next priority."""
        provider.add_pattern_response(r"specific", "Matched!")
        provider.set_responses(["From queue"])

        response = await provider.complete("different input")

        assert response.content == "From queue"

    @pytest.mark.asyncio
    async def test_regex_patterns(self, provider: EchoProvider) -> None:
        """Test regex patterns work correctly."""
        provider.add_pattern_response(r"\d{3}-\d{4}", "Phone detected!")

        response = await provider.complete("Call me at 555-1234")

        assert response.content == "Phone detected!"


class TestCallHistory:
    """Tests for call history tracking."""

    @pytest.mark.asyncio
    async def test_call_count(self, provider: EchoProvider) -> None:
        """Test call count tracking."""
        assert provider.call_count == 0

        await provider.complete("First")
        assert provider.call_count == 1

        await provider.complete("Second")
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_get_call_by_index(self, provider: EchoProvider) -> None:
        """Test getting specific calls by index."""
        provider.set_responses(["Response 1", "Response 2"])

        await provider.complete("Input 1")
        await provider.complete("Input 2")

        call_0 = provider.get_call(0)
        call_1 = provider.get_call(1)
        call_neg1 = provider.get_call(-1)  # Last call

        assert call_0["messages"][0].content == "Input 1"
        assert call_1["messages"][0].content == "Input 2"
        assert call_neg1 == call_1

    @pytest.mark.asyncio
    async def test_get_last_call(self, provider: EchoProvider) -> None:
        """Test getting last call."""
        assert provider.get_last_call() is None

        await provider.complete("Test")

        last = provider.get_last_call()
        assert last is not None
        assert last["messages"][0].content == "Test"

    @pytest.mark.asyncio
    async def test_get_last_user_message(self, provider: EchoProvider) -> None:
        """Test getting last user message."""
        assert provider.get_last_user_message() is None

        await provider.complete("User message here")

        assert provider.get_last_user_message() == "User message here"

    @pytest.mark.asyncio
    async def test_call_history_includes_response(
        self, provider: EchoProvider
    ) -> None:
        """Test call history includes the response."""
        provider.set_responses(["Expected response"])

        await provider.complete("Test input")

        call = provider.get_last_call()
        assert call["response"].content == "Expected response"

    @pytest.mark.asyncio
    async def test_clear_history(self, provider: EchoProvider) -> None:
        """Test clearing call history."""
        await provider.complete("Test 1")
        await provider.complete("Test 2")

        assert provider.call_count == 2

        provider.clear_history()

        assert provider.call_count == 0


class TestReset:
    """Tests for reset functionality."""

    @pytest.mark.asyncio
    async def test_clear_responses(self, provider: EchoProvider) -> None:
        """Test clearing responses."""
        provider.set_responses(["Queued"])
        provider.set_response_function(lambda m: "Function")
        provider.add_pattern_response(r"test", "Pattern")

        provider.clear_responses()

        # Should fall back to echo
        response = await provider.complete("test message")
        assert "test message" in response.content

    @pytest.mark.asyncio
    async def test_reset_clears_everything(
        self, provider: EchoProvider
    ) -> None:
        """Test reset clears responses and history."""
        provider.set_responses(["Queued"])
        await provider.complete("Test")

        provider.reset()

        assert provider.call_count == 0
        # Should fall back to echo
        response = await provider.complete("new test")
        assert "new test" in response.content


class TestChaining:
    """Tests for method chaining."""

    @pytest.mark.asyncio
    async def test_fluent_api(self, provider: EchoProvider) -> None:
        """Test fluent API for configuration."""
        provider.set_responses(["R1"]).add_response("R2").add_pattern_response(
            r"special", "Special!"
        )

        r1 = await provider.complete("First")
        r2 = await provider.complete("Second")
        r3 = await provider.complete("This is special")

        assert r1.content == "R1"
        assert r2.content == "R2"
        assert r3.content == "Special!"


class TestInitializationOptions:
    """Tests for initialization with options."""

    @pytest.mark.asyncio
    async def test_init_with_responses(self) -> None:
        """Test initialization with responses parameter."""
        provider = EchoProvider(
            {"provider": "echo", "model": "test"},
            responses=["Init response"],
        )

        response = await provider.complete("Test")

        assert response.content == "Init response"

    @pytest.mark.asyncio
    async def test_init_with_response_function(self) -> None:
        """Test initialization with response_fn parameter."""
        provider = EchoProvider(
            {"provider": "echo", "model": "test"},
            response_fn=lambda msgs: "From init function",
        )

        response = await provider.complete("Test")

        assert response.content == "From init function"

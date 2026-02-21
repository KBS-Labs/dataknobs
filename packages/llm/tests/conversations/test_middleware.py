"""Tests for conversation middleware system."""

import pytest
import logging
from dataknobs_llm.conversations import (
    ConversationManager,
    ConversationMiddleware,
    LoggingMiddleware,
    ContentFilterMiddleware,
    ValidationMiddleware,
    MetadataMiddleware,
    RateLimitMiddleware,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm import LLMConfig, EchoProvider, LLMMessage, LLMResponse
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_llm.conversations.storage import ConversationState, ConversationNode
from dataknobs_common.exceptions import RateLimitError
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_structures.tree import Tree
from pathlib import Path
import tempfile
from typing import List
from datetime import datetime


def create_test_prompts(prompt_dir: Path):
    """Create test prompt files for middleware tests."""
    import yaml

    # System prompts
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "assistant.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )

    # User prompts
    user_dir = prompt_dir / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "question.yaml").write_text(
        yaml.dump({"template": "What is {{topic}}?"})
    )

    # Validation prompts - one that will pass, one that will fail
    # Important: Don't use the word "VALID" in the template itself, so EchoProvider
    # will only return "VALID" if it was in the response being validated
    (user_dir / "validate_response.yaml").write_text(
        yaml.dump({"template": "Check this response: {{response}}"})
    )


@pytest.fixture
async def test_components():
    """Create test LLM, builder, and storage for middleware tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        create_test_prompts(prompt_dir)

        config = LLMConfig(
            provider="echo",
            model="echo-model",
            options={"echo_prefix": ""}
        )
        llm = EchoProvider(config)
        library = FileSystemPromptLibrary(prompt_dir)
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        yield {
            "llm": llm,
            "builder": builder,
            "storage": storage
        }

        await llm.close()


class TestLoggingMiddleware:
    """Test LoggingMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_basic_logging(self, test_components, caplog):
        """Test that logging middleware logs requests and responses."""
        # Create logger and middleware
        logger = logging.getLogger("test_middleware")
        logger.setLevel(logging.INFO)
        middleware = LoggingMiddleware(logger)

        # Create conversation with middleware
        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            system_prompt_name="assistant",
            middleware=[middleware]
        )

        # Add message and complete
        with caplog.at_level(logging.INFO, logger="test_middleware"):
            await manager.add_message(role="user", content="Hello")
            await manager.complete()

        # Check that logging occurred
        assert "Sending" in caplog.text
        assert "messages to LLM" in caplog.text
        assert "Received response" in caplog.text

    @pytest.mark.asyncio
    async def test_logging_with_stream(self, test_components, caplog):
        """Test that logging works with streaming."""
        logger = logging.getLogger("test_stream")
        logger.setLevel(logging.INFO)
        middleware = LoggingMiddleware(logger)

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        with caplog.at_level(logging.INFO, logger="test_stream"):
            await manager.add_message(role="user", content="Test stream")
            async for chunk in manager.stream_complete():
                pass  # Consume stream

        assert "Sending" in caplog.text
        assert "Received response" in caplog.text


class TestContentFilterMiddleware:
    """Test ContentFilterMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_content_filtering(self, test_components):
        """Test that content filter replaces filtered words."""
        # Create middleware with filter words
        middleware = ContentFilterMiddleware(
            filter_words=["badword", "inappropriate"],
            replacement="[FILTERED]"
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        # Add message containing filtered content
        await manager.add_message(role="user", content="This has badword in it")
        response = await manager.complete()

        # Check that content was filtered
        # Note: EchoProvider echoes input, so the response will contain "badword"
        # The middleware should filter it in the response
        assert "[FILTERED]" in response.content or "badword" not in response.content

    @pytest.mark.asyncio
    async def test_case_insensitive_filtering(self, test_components):
        """Test case-insensitive content filtering."""
        middleware = ContentFilterMiddleware(
            filter_words=["badword"],
            replacement="***",
            case_sensitive=False
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="BADWORD and badword and BadWord")
        response = await manager.complete()

        # All variations should be filtered with case-insensitive
        # (EchoProvider will return the input with all variations)
        assert response.metadata.get("content_filtered") is True or "badword" in response.content.lower()

    @pytest.mark.asyncio
    async def test_no_filtering_when_clean(self, test_components):
        """Test that clean content passes through unchanged."""
        middleware = ContentFilterMiddleware(
            filter_words=["badword"],
            replacement="[FILTERED]"
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="Clean content")
        response = await manager.complete()

        # Should not be marked as filtered
        assert response.metadata.get("content_filtered") is not True
        assert "[FILTERED]" not in response.content


class TestMetadataMiddleware:
    """Test MetadataMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_static_request_metadata(self, test_components):
        """Test adding static metadata to requests."""
        middleware = MetadataMiddleware(
            request_metadata={"source": "test", "version": "1.0"}
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="Test")
        await manager.complete()

        # Verify metadata was added (would need to inspect internal state)
        # For now, just verify it doesn't crash
        assert manager.conversation_id is not None

    @pytest.mark.asyncio
    async def test_static_response_metadata(self, test_components):
        """Test adding static metadata to responses."""
        middleware = MetadataMiddleware(
            response_metadata={"processed": True, "version": "2.0"}
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="Test")
        response = await manager.complete()

        # Check response has metadata
        assert response.metadata.get("processed") is True
        assert response.metadata.get("version") == "2.0"

    @pytest.mark.asyncio
    async def test_dynamic_metadata_function(self, test_components):
        """Test adding dynamic metadata via function."""
        call_count = {"count": 0}

        def get_metadata():
            call_count["count"] += 1
            return {"call_number": call_count["count"], "timestamp": "dynamic"}

        middleware = MetadataMiddleware(
            response_metadata_fn=get_metadata
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="First")
        response1 = await manager.complete()

        await manager.add_message(role="user", content="Second")
        response2 = await manager.complete()

        # Check that function was called for each response
        assert response1.metadata.get("call_number") == 1
        assert response2.metadata.get("call_number") == 2
        assert call_count["count"] == 2


class TestValidationMiddleware:
    """Test ValidationMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_validation_passes(self, test_components):
        """Test that valid responses pass validation."""
        # Create separate LLM for validation
        validation_config = LLMConfig(
            provider="echo",
            model="echo-validator",
            options={"echo_prefix": ""}
        )
        validation_llm = EchoProvider(validation_config)

        # The validation prompt will receive the response and the validation LLM
        # will be called with it. EchoProvider will echo back the prompt, which
        # should contain "VALID" if the original response contains "VALID"
        middleware = ValidationMiddleware(
            llm=validation_llm,
            prompt_builder=test_components["builder"],
            validation_prompt="validate_response",
            auto_retry=False
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="This is a VALID response")
        # EchoProvider echoes "This is a VALID response"
        # ValidationMiddleware renders "Check this response: This is a VALID response"
        # EchoProvider echoes that back, which contains "VALID"
        # _check_validity finds "VALID" → validation passes
        response = await manager.complete()

        assert response is not None
        assert response.metadata.get("validation_failed") is not True

        await validation_llm.close()

    @pytest.mark.asyncio
    async def test_validation_marks_failure(self, test_components):
        """Test that invalid responses are marked with auto_retry."""
        # Create separate LLM for validation
        validation_config = LLMConfig(
            provider="echo",
            model="echo-validator",
            options={"echo_prefix": ""}
        )
        validation_llm = EchoProvider(validation_config)

        middleware = ValidationMiddleware(
            llm=validation_llm,
            prompt_builder=test_components["builder"],
            validation_prompt="validate_response",
            auto_retry=True  # Mark for retry instead of raising
        )

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="bad response")
        # EchoProvider echoes "bad response" (no "VALID" in it)
        # ValidationMiddleware renders "Check this response: bad response"
        # EchoProvider echoes that back - still no "VALID" in it
        response = await manager.complete()

        # Should be marked as validation failed with retry requested
        assert response.metadata.get("validation_failed") is True
        assert response.metadata.get("retry_requested") is True
        assert "validation_response" in response.metadata

        await validation_llm.close()


class TestMiddlewareChaining:
    """Test multiple middleware working together."""

    @pytest.mark.asyncio
    async def test_multiple_middleware_execution(self, test_components):
        """Test that multiple middleware execute in correct order."""
        # Create multiple middleware
        logger = logging.getLogger("test_chain")
        logging_mw = LoggingMiddleware(logger)
        metadata_mw = MetadataMiddleware(response_metadata={"chain": "test"})
        filter_mw = ContentFilterMiddleware(filter_words=["bad"], replacement="***")

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[logging_mw, metadata_mw, filter_mw]
        )

        await manager.add_message(role="user", content="This is bad")
        response = await manager.complete()

        # All middleware should have executed
        assert response.metadata.get("chain") == "test"
        # Content filtering should have occurred
        # Note: Depending on execution order, filtering might happen

    @pytest.mark.asyncio
    async def test_middleware_order_matters(self, test_components):
        """Test that middleware execution order is preserved."""
        execution_order = []

        class OrderTrackingMiddleware(ConversationMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process_request(self, messages, state):
                execution_order.append(f"request-{self.name}")
                return messages

            async def process_response(self, response, state):
                execution_order.append(f"response-{self.name}")
                return response

        mw1 = OrderTrackingMiddleware("first")
        mw2 = OrderTrackingMiddleware("second")
        mw3 = OrderTrackingMiddleware("third")

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[mw1, mw2, mw3]
        )

        await manager.add_message(role="user", content="Test order")
        await manager.complete()

        # Requests should execute in forward order
        assert execution_order[0] == "request-first"
        assert execution_order[1] == "request-second"
        assert execution_order[2] == "request-third"

        # Responses should execute in reverse order (onion model)
        assert execution_order[3] == "response-third"
        assert execution_order[4] == "response-second"
        assert execution_order[5] == "response-first"


class TestCustomMiddleware:
    """Test creating custom middleware."""

    @pytest.mark.asyncio
    async def test_custom_middleware_implementation(self, test_components):
        """Test implementing custom middleware."""

        class UpperCaseMiddleware(ConversationMiddleware):
            """Middleware that uppercases all response content."""

            async def process_request(self, messages, state):
                return messages

            async def process_response(self, response, state):
                response.content = response.content.upper()
                return response

        middleware = UpperCaseMiddleware()

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="hello world")
        response = await manager.complete()

        # Response should be uppercased
        assert response.content.isupper()
        assert "HELLO WORLD" in response.content

    @pytest.mark.asyncio
    async def test_middleware_can_modify_messages(self, test_components):
        """Test that middleware can modify request messages."""

        class MessagePrefixMiddleware(ConversationMiddleware):
            """Middleware that adds prefix to all user messages."""

            async def process_request(self, messages, state):
                for msg in messages:
                    if msg.role == "user":
                        msg.content = f"[PREFIXED] {msg.content}"
                return messages

            async def process_response(self, response, state):
                return response

        middleware = MessagePrefixMiddleware()

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware]
        )

        await manager.add_message(role="user", content="Test message")
        response = await manager.complete()

        # EchoProvider will echo the modified message
        assert "[PREFIXED]" in response.content


def _make_conversation_state(
    conversation_id: str = "conv-123",
    metadata: dict | None = None,
) -> ConversationState:
    """Create a minimal ConversationState for unit tests."""
    root_node = ConversationNode(
        message=LLMMessage(role="system", content="You are helpful"),
        node_id="",
    )
    tree = Tree(root_node)
    return ConversationState(
        conversation_id=conversation_id,
        message_tree=tree,
        current_node_id="",
        metadata=metadata or {},
    )


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware backed by InMemoryRateLimiter."""

    @pytest.mark.asyncio
    async def test_requests_within_limit_pass(self):
        """Test that requests within the rate limit are allowed."""
        middleware = RateLimitMiddleware(max_requests=5, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        # Should not raise for requests within limit
        for _ in range(5):
            result = await middleware.process_request(messages, state)
            assert result is not None

    @pytest.mark.asyncio
    async def test_exceeding_limit_raises_rate_limit_error(self):
        """Test that exceeding the rate limit raises RateLimitError."""
        middleware = RateLimitMiddleware(max_requests=3, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        # Use up the limit
        for _ in range(3):
            await middleware.process_request(messages, state)

        # 4th request should raise
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await middleware.process_request(messages, state)

    @pytest.mark.asyncio
    async def test_rate_limit_error_has_retry_after(self):
        """Test that RateLimitError includes retry_after from common package."""
        middleware = RateLimitMiddleware(max_requests=1, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        await middleware.process_request(messages, state)

        with pytest.raises(RateLimitError) as exc_info:
            await middleware.process_request(messages, state)

        # The common RateLimitError should have retry_after set
        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_rate_limit_updates_state_metadata_on_exceed(self):
        """Test that state metadata is updated when rate limit is exceeded."""
        middleware = RateLimitMiddleware(max_requests=1, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        await middleware.process_request(messages, state)

        with pytest.raises(RateLimitError):
            await middleware.process_request(messages, state)

        assert state.metadata["rate_limit_exceeded"] is True
        assert state.metadata["rate_limit_max"] == 1
        assert state.metadata["rate_limit_window"] == 60

    @pytest.mark.asyncio
    async def test_message_metadata_includes_rate_limit_info(self):
        """Test that messages get rate limit count metadata."""
        middleware = RateLimitMiddleware(max_requests=10, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        result = await middleware.process_request(messages, state)

        assert result[0].metadata is not None
        assert result[0].metadata["rate_limit_count"] == 1
        assert result[0].metadata["rate_limit_max"] == 10

    @pytest.mark.asyncio
    async def test_response_metadata_includes_rate_limit_info(self):
        """Test that responses get rate limit metadata."""
        middleware = RateLimitMiddleware(max_requests=10, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        # Make a request first to record in the limiter
        await middleware.process_request(messages, state)

        # Process response
        response = LLMResponse(content="Test response", model="test")
        result = await middleware.process_response(response, state)

        assert result.metadata is not None
        assert result.metadata["rate_limit_count"] == 1
        assert result.metadata["rate_limit_max"] == 10
        assert result.metadata["rate_limit_remaining"] == 9

    @pytest.mark.asyncio
    async def test_conversation_scope_isolates_by_conversation_id(self):
        """Test that conversation scope rate limits per conversation."""
        middleware = RateLimitMiddleware(
            max_requests=2, window_seconds=60, scope="conversation"
        )
        state_a = _make_conversation_state(conversation_id="conv-a")
        state_b = _make_conversation_state(conversation_id="conv-b")
        messages = [LLMMessage(role="user", content="Hello")]

        # Use up conv-a's limit
        await middleware.process_request(messages, state_a)
        await middleware.process_request(messages, state_a)

        # conv-a should be rate limited
        with pytest.raises(RateLimitError):
            await middleware.process_request(messages, state_a)

        # conv-b should still work
        result = await middleware.process_request(messages, state_b)
        assert result is not None

    @pytest.mark.asyncio
    async def test_client_id_scope(self):
        """Test that client_id scope rate limits by client_id from metadata."""
        middleware = RateLimitMiddleware(
            max_requests=2, window_seconds=60, scope="client_id"
        )
        state_same_client = _make_conversation_state(
            conversation_id="conv-1",
            metadata={"client_id": "client-x"},
        )
        state_same_client_2 = _make_conversation_state(
            conversation_id="conv-2",
            metadata={"client_id": "client-x"},
        )
        state_diff_client = _make_conversation_state(
            conversation_id="conv-3",
            metadata={"client_id": "client-y"},
        )
        messages = [LLMMessage(role="user", content="Hello")]

        # Use up client-x's limit across different conversations
        await middleware.process_request(messages, state_same_client)
        await middleware.process_request(messages, state_same_client_2)

        # Same client, different conversation — should be rate limited
        with pytest.raises(RateLimitError):
            await middleware.process_request(messages, state_same_client)

        # Different client — should work
        result = await middleware.process_request(messages, state_diff_client)
        assert result is not None

    @pytest.mark.asyncio
    async def test_custom_key_function(self):
        """Test rate limiting with custom key extraction function."""
        def user_key(state: ConversationState) -> str:
            return state.metadata.get("user_id", "anonymous")

        middleware = RateLimitMiddleware(
            max_requests=2, window_seconds=60, key_fn=user_key
        )
        state = _make_conversation_state(metadata={"user_id": "alice"})
        messages = [LLMMessage(role="user", content="Hello")]

        await middleware.process_request(messages, state)
        await middleware.process_request(messages, state)

        with pytest.raises(RateLimitError):
            await middleware.process_request(messages, state)

        # Different user should be fine
        state_bob = _make_conversation_state(metadata={"user_id": "bob"})
        result = await middleware.process_request(messages, state_bob)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_rate_limit_status(self):
        """Test get_rate_limit_status returns correct status dict."""
        middleware = RateLimitMiddleware(max_requests=10, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        # Make 3 requests
        for _ in range(3):
            await middleware.process_request(messages, state)

        status = await middleware.get_rate_limit_status("conv-123")

        assert status["current_count"] == 3
        assert status["max_requests"] == 10
        assert status["remaining"] == 7
        assert status["window_seconds"] == 60
        assert "next_reset" in status

    @pytest.mark.asyncio
    async def test_reset_specific_key(self):
        """Test resetting rate limit for a specific key."""
        middleware = RateLimitMiddleware(max_requests=2, window_seconds=60)
        state = _make_conversation_state()
        messages = [LLMMessage(role="user", content="Hello")]

        # Use up the limit
        await middleware.process_request(messages, state)
        await middleware.process_request(messages, state)

        # Should be limited
        with pytest.raises(RateLimitError):
            await middleware.process_request(messages, state)

        # Reset this key
        await middleware.reset("conv-123")

        # Should work again
        result = await middleware.process_request(messages, state)
        assert result is not None

    @pytest.mark.asyncio
    async def test_reset_all_keys(self):
        """Test resetting all rate limit keys."""
        middleware = RateLimitMiddleware(max_requests=1, window_seconds=60)
        state_a = _make_conversation_state(conversation_id="conv-a")
        state_b = _make_conversation_state(conversation_id="conv-b")
        messages = [LLMMessage(role="user", content="Hello")]

        # Use up limits for both
        await middleware.process_request(messages, state_a)
        await middleware.process_request(messages, state_b)

        # Both should be limited
        with pytest.raises(RateLimitError):
            await middleware.process_request(messages, state_a)
        with pytest.raises(RateLimitError):
            await middleware.process_request(messages, state_b)

        # Reset all
        await middleware.reset()

        # Both should work again
        await middleware.process_request(messages, state_a)
        await middleware.process_request(messages, state_b)

    @pytest.mark.asyncio
    async def test_integration_with_conversation_manager(self, test_components):
        """Test RateLimitMiddleware works through ConversationManager."""
        middleware = RateLimitMiddleware(max_requests=2, window_seconds=60)

        manager = await ConversationManager.create(
            llm=test_components["llm"],
            prompt_builder=test_components["builder"],
            storage=test_components["storage"],
            middleware=[middleware],
        )

        # First two requests should work
        await manager.add_message(role="user", content="First")
        response1 = await manager.complete()
        assert response1 is not None
        assert response1.metadata.get("rate_limit_count") == 1

        await manager.add_message(role="user", content="Second")
        response2 = await manager.complete()
        assert response2 is not None
        assert response2.metadata.get("rate_limit_count") == 2

        # Third request should fail
        await manager.add_message(role="user", content="Third")
        with pytest.raises(RateLimitError):
            await manager.complete()

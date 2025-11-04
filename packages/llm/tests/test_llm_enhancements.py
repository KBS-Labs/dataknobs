"""Tests for LLM package enhancements (clone, cost tracking, etc.)."""

import pytest
from dataknobs_llm.llm import LLMConfig, LLMResponse, CompletionMode
from datetime import datetime


class TestLLMConfigClone:
    """Test LLMConfig clone() method."""

    def test_clone_basic(self):
        """Test basic cloning without overrides."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )

        cloned = config.clone()

        assert cloned.provider == config.provider
        assert cloned.model == config.model
        assert cloned.temperature == config.temperature
        assert cloned.max_tokens == config.max_tokens

        # Ensure it's a different object
        assert cloned is not config

    def test_clone_with_overrides(self):
        """Test cloning with field overrides."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )

        cloned = config.clone(temperature=1.2, max_tokens=500)

        # Overridden fields should change
        assert cloned.temperature == 1.2
        assert cloned.max_tokens == 500

        # Other fields should remain the same
        assert cloned.provider == config.provider
        assert cloned.model == config.model

        # Original should be unchanged
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_clone_with_model_override(self):
        """Test cloning with model override."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7
        )

        cloned = config.clone(model="gpt-3.5-turbo")

        assert cloned.model == "gpt-3.5-turbo"
        assert config.model == "gpt-4"

    def test_clone_with_multiple_overrides(self):
        """Test cloning with multiple overrides."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0
        )

        cloned = config.clone(
            temperature=0.9,
            max_tokens=1500,
            top_p=0.95,
            frequency_penalty=0.5
        )

        assert cloned.temperature == 0.9
        assert cloned.max_tokens == 1500
        assert cloned.top_p == 0.95
        assert cloned.frequency_penalty == 0.5

        # Provider and model unchanged
        assert cloned.provider == "openai"
        assert cloned.model == "gpt-4"

    def test_clone_with_enum_override(self):
        """Test cloning with enum field override."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            mode=CompletionMode.CHAT
        )

        cloned = config.clone(mode=CompletionMode.TEXT)

        assert cloned.mode == CompletionMode.TEXT
        assert config.mode == CompletionMode.CHAT

    def test_clone_preserves_optional_fields(self):
        """Test cloning preserves optional fields."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key",
            api_base="https://custom.api.com",
            seed=42,
            user_id="user-123"
        )

        cloned = config.clone(temperature=0.9)

        assert cloned.api_key == "sk-test-key"
        assert cloned.api_base == "https://custom.api.com"
        assert cloned.seed == 42
        assert cloned.user_id == "user-123"
        assert cloned.temperature == 0.9


class TestCostTracking:
    """Test cost tracking fields in LLMResponse."""

    def test_response_with_cost(self):
        """Test creating response with cost tracking."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost_usd=0.0015,
            cumulative_cost_usd=0.0045
        )

        assert response.cost_usd == 0.0015
        assert response.cumulative_cost_usd == 0.0045

    def test_response_without_cost(self):
        """Test creating response without cost tracking."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4"
        )

        assert response.cost_usd is None
        assert response.cumulative_cost_usd is None

    def test_response_cost_can_be_set(self):
        """Test that cost can be set after response creation."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )

        assert response.cost_usd is None

        # Set cost
        response.cost_usd = 0.0015
        response.cumulative_cost_usd = 0.0045

        assert response.cost_usd == 0.0015
        assert response.cumulative_cost_usd == 0.0045

    def test_response_with_partial_cost(self):
        """Test response with only cost_usd set."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            cost_usd=0.0015
        )

        assert response.cost_usd == 0.0015
        assert response.cumulative_cost_usd is None

    def test_cost_tracking_with_metadata(self):
        """Test cost tracking doesn't interfere with metadata."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            cost_usd=0.0015,
            cumulative_cost_usd=0.0045,
            metadata={"custom_field": "value"}
        )

        assert response.cost_usd == 0.0015
        assert response.cumulative_cost_usd == 0.0045
        assert response.metadata["custom_field"] == "value"

    def test_multiple_responses_cumulative_cost(self):
        """Test tracking cumulative cost across multiple responses."""
        responses = [
            LLMResponse(
                content="First response",
                model="gpt-4",
                cost_usd=0.001,
                cumulative_cost_usd=0.001
            ),
            LLMResponse(
                content="Second response",
                model="gpt-4",
                cost_usd=0.002,
                cumulative_cost_usd=0.003
            ),
            LLMResponse(
                content="Third response",
                model="gpt-4",
                cost_usd=0.0015,
                cumulative_cost_usd=0.0045
            ),
        ]

        # Verify costs accumulate correctly
        assert responses[0].cumulative_cost_usd == 0.001
        assert responses[1].cumulative_cost_usd == 0.003
        assert responses[2].cumulative_cost_usd == 0.0045

        # Verify individual costs
        total_individual = sum(r.cost_usd for r in responses)
        assert abs(total_individual - 0.0045) < 0.0001  # Floating point comparison


class TestLLMResponseFields:
    """Test that existing LLMResponse fields still work correctly."""

    def test_response_basic_fields(self):
        """Test basic response fields."""
        now = datetime.now()
        response = LLMResponse(
            content="Test content",
            model="gpt-4",
            finish_reason="stop",
            usage={"total_tokens": 100},
            created_at=now
        )

        assert response.content == "Test content"
        assert response.model == "gpt-4"
        assert response.finish_reason == "stop"
        assert response.usage == {"total_tokens": 100}
        assert response.created_at == now

    def test_response_with_function_call(self):
        """Test response with function call."""
        response = LLMResponse(
            content="",
            model="gpt-4",
            function_call={
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}'
            }
        )

        assert response.function_call is not None
        assert response.function_call["name"] == "get_weather"

    def test_response_with_metadata(self):
        """Test response with metadata."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            metadata={
                "request_id": "req-123",
                "environment": "production"
            }
        )

        assert response.metadata["request_id"] == "req-123"
        assert response.metadata["environment"] == "production"

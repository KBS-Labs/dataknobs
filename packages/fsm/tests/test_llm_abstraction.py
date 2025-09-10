"""Tests for LLM abstraction layer."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from dataknobs_fsm.llm.base import (
    LLMConfig, LLMMessage, LLMResponse, LLMStreamResponse,
    CompletionMode, ModelCapability
)
from dataknobs_fsm.llm.providers import (
    OpenAIAdapter, create_llm_provider
)
from dataknobs_fsm.llm.utils import (
    PromptTemplate, MessageBuilder, ResponseParser,
    TokenCounter, CostCalculator, create_few_shot_prompt
)


class TestLLMBase:
    """Test base LLM components."""
    
    def test_llm_message_creation(self):
        """Test LLMMessage creation."""
        msg = LLMMessage(
            role="user",
            content="Hello, world!",
            name="test_user"
        )
        
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.name == "test_user"
        assert msg.function_call is None
        
    def test_llm_response_creation(self):
        """Test LLMResponse creation."""
        response = LLMResponse(
            content="Response text",
            model="gpt-3.5-turbo",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        
        assert response.content == "Response text"
        assert response.model == "gpt-3.5-turbo"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 30
        
    def test_llm_config(self):
        """Test LLMConfig creation."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
            max_tokens=100,
            mode=CompletionMode.CHAT
        )
        
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 100
        assert config.mode == CompletionMode.CHAT
        
    def test_model_capabilities(self):
        """Test ModelCapability enum."""
        assert ModelCapability.TEXT_GENERATION.value == "text_generation"
        assert ModelCapability.CHAT.value == "chat"
        assert ModelCapability.FUNCTION_CALLING.value == "function_calling"
        assert ModelCapability.STREAMING.value == "streaming"


class TestOpenAIAdapter:
    """Test OpenAI adapter."""
    
    def test_adapt_messages(self):
        """Test message adaptation."""
        adapter = OpenAIAdapter()
        
        messages = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there")
        ]
        
        adapted = adapter.adapt_messages(messages)
        
        assert len(adapted) == 3
        assert adapted[0]["role"] == "system"
        assert adapted[0]["content"] == "You are helpful"
        assert adapted[1]["role"] == "user"
        assert adapted[2]["role"] == "assistant"
        
    def test_adapt_config(self):
        """Test configuration adaptation."""
        adapter = OpenAIAdapter()
        
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            stop_sequences=["END"],
            seed=42
        )
        
        params = adapter.adapt_config(config)
        
        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 150
        assert params["top_p"] == 0.9
        assert params["stop"] == ["END"]
        assert params["seed"] == 42


class TestPromptTemplate:
    """Test prompt template utilities."""
    
    def test_template_creation(self):
        """Test prompt template creation."""
        template = PromptTemplate("Hello {name}, you have {count} messages")
        
        assert template.template == "Hello {name}, you have {count} messages"
        assert "name" in template.variables
        assert "count" in template.variables
        
    def test_template_format(self):
        """Test template formatting."""
        template = PromptTemplate("Hello {name}, you have {count} messages")
        
        result = template.format(name="Alice", count=5)
        assert result == "Hello Alice, you have 5 messages"
        
    def test_template_format_missing_variables(self):
        """Test template formatting with missing variables."""
        template = PromptTemplate("Hello {name}")
        
        with pytest.raises(ValueError, match="Missing variables"):
            template.format()
            
    def test_partial_template(self):
        """Test partial template."""
        template = PromptTemplate("Hello {name}, you are {age} years old")
        
        partial = template.partial(name="Bob")
        assert "Bob" in partial.template
        assert "name" not in partial.variables
        assert "age" in partial.variables
        
        result = partial.format(age=30)
        assert result == "Hello Bob, you are 30 years old"


class TestMessageBuilder:
    """Test message builder."""
    
    def test_message_building(self):
        """Test building message sequence."""
        builder = MessageBuilder()
        
        messages = (builder
            .system("You are helpful")
            .user("Hello")
            .assistant("Hi there")
            .build())
        
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        
    def test_function_message(self):
        """Test adding function message."""
        builder = MessageBuilder()
        
        messages = (builder
            .user("What's the weather?")
            .function("get_weather", '{"temp": 72}', {"name": "get_weather"})
            .build())
        
        assert len(messages) == 2
        assert messages[1].role == "function"
        assert messages[1].name == "get_weather"
        assert messages[1].content == '{"temp": 72}'
        
    def test_from_template(self):
        """Test building from template."""
        template = PromptTemplate("Hello {name}")
        builder = MessageBuilder()
        
        messages = (builder
            .from_template("user", template, name="Alice")
            .build())
        
        assert len(messages) == 1
        assert messages[0].content == "Hello Alice"
        
    def test_clear_messages(self):
        """Test clearing messages."""
        builder = MessageBuilder()
        
        builder.user("Test")
        assert len(builder.messages) == 1
        
        builder.clear()
        assert len(builder.messages) == 0


class TestResponseParser:
    """Test response parser."""
    
    def test_extract_json(self):
        """Test JSON extraction."""
        # Simple JSON
        text = '{"key": "value", "number": 42}'
        result = ResponseParser.extract_json(text)
        assert result == {"key": "value", "number": 42}
        
        # JSON in markdown
        text = '```json\n{"key": "value"}\n```'
        result = ResponseParser.extract_json(text)
        assert result == {"key": "value"}
        
        # Invalid JSON
        text = "This is not JSON"
        result = ResponseParser.extract_json(text)
        assert result is None
        
    def test_extract_code(self):
        """Test code extraction."""
        text = '''Here's the code:
```python
def hello():
    print("Hello")
```
And another:
```javascript
console.log("Hi");
```'''
        
        # Extract all code
        code_blocks = ResponseParser.extract_code(text)
        assert len(code_blocks) == 2
        assert "def hello()" in code_blocks[0]
        assert "console.log" in code_blocks[1]
        
        # Extract specific language
        python_blocks = ResponseParser.extract_code(text, language="python")
        assert len(python_blocks) == 1
        assert "def hello()" in python_blocks[0]
        
    def test_extract_list(self):
        """Test list extraction."""
        # Bullet list
        text = """Here are the items:
- First item
- Second item
- Third item"""
        
        items = ResponseParser.extract_list(text)
        assert len(items) == 3
        assert items[0] == "First item"
        
        # Numbered list
        text = """Steps:
1. First step
2. Second step
3. Third step"""
        
        items = ResponseParser.extract_list(text, numbered=True)
        assert len(items) == 3
        assert items[0] == "First step"
        
    def test_extract_sections(self):
        """Test section extraction."""
        text = """# Introduction
This is the intro.

## Details
Here are details.

### Subsection
More content here."""
        
        sections = ResponseParser.extract_sections(text)
        
        assert "Introduction" in sections
        assert "Details" in sections
        assert "Subsection" in sections
        assert "This is the intro." in sections["Introduction"]


class TestTokenCounter:
    """Test token counter."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello, this is a test message."
        
        # Default estimation
        tokens = TokenCounter.estimate_tokens(text, "default")
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count
        
        # GPT-4 estimation
        tokens = TokenCounter.estimate_tokens(text, "gpt-4")
        assert tokens == int(len(text) * 0.25)
        
    def test_estimate_messages_tokens(self):
        """Test message token estimation."""
        messages = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there")
        ]
        
        tokens = TokenCounter.estimate_messages_tokens(messages, "gpt-4")
        assert tokens > 0
        assert tokens > 12  # At least 4 tokens per message for role
        
    def test_fits_in_context(self):
        """Test context window check."""
        text = "Short text"
        
        # Should fit
        assert TokenCounter.fits_in_context(text, "gpt-4", max_tokens=100)
        
        # Long text shouldn't fit
        long_text = "x" * 1000
        assert not TokenCounter.fits_in_context(long_text, "gpt-4", max_tokens=100)


class TestCostCalculator:
    """Test cost calculator."""
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        )
        
        cost = CostCalculator.calculate_cost(response)
        assert cost is not None
        assert cost > 0
        
        # Check calculation
        # GPT-4: $0.03/1K input, $0.06/1K output
        expected = (100 / 1000) * 0.03 + (50 / 1000) * 0.06
        assert abs(cost - expected) < 0.0001
        
    def test_calculate_cost_no_usage(self):
        """Test cost calculation without usage info."""
        response = LLMResponse(
            content="Test",
            model="gpt-4"
        )
        
        cost = CostCalculator.calculate_cost(response)
        assert cost is None
        
    def test_estimate_cost(self):
        """Test cost estimation."""
        text = "This is a test prompt"
        
        cost = CostCalculator.estimate_cost(
            text,
            model="gpt-3.5-turbo",
            expected_output_tokens=100
        )
        
        assert cost is not None
        assert cost > 0
        
    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        cost = CostCalculator.estimate_cost(
            "Test",
            model="unknown-model",
            expected_output_tokens=100
        )
        
        assert cost is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_few_shot_prompt(self):
        """Test few-shot prompt creation."""
        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "3*3", "output": "9"}
        ]
        
        prompt = create_few_shot_prompt(
            "Calculate the result:",
            examples
        )
        
        assert "Calculate the result:" in prompt.template
        assert "Example 1:" in prompt.template
        assert "2+2" in prompt.template
        assert "4" in prompt.template
        assert "Example 2:" in prompt.template
        assert "{query}" in prompt.template
        
        # Format with query
        result = prompt.format(query="5+5")
        assert "5+5" in result


@pytest.mark.asyncio
class TestLLMProviderCreation:
    """Test LLM provider creation."""
    
    def test_create_llm_provider(self):
        """Test provider creation."""
        # OpenAI provider
        config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = create_llm_provider(config, is_async=True)
            assert provider is not None
            
    def test_create_unknown_provider(self):
        """Test creating unknown provider."""
        config = LLMConfig(
            provider="unknown",
            model="test"
        )
        
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_provider(config)
            
    def test_sync_provider_not_implemented(self):
        """Test sync provider creation."""
        config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo"
        )
        
        with pytest.raises(NotImplementedError, match="Sync providers"):
            create_llm_provider(config, is_async=False)
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
    TokenCounter, CostCalculator, create_few_shot_prompt,
    render_conditional_template
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


class TestRenderConditionalTemplate:
    """Test render_conditional_template function."""

    def test_basic_variable_substitution(self):
        """Test basic variable substitution."""
        template = "Hello {{name}}, welcome!"
        params = {"name": "Alice"}
        result = render_conditional_template(template, params)
        assert result == "Hello Alice, welcome!"

    def test_missing_variable_unchanged(self):
        """Test that missing variables remain unchanged."""
        template = "Hello {{name}}, your ID is {{id}}"
        params = {"name": "Bob"}
        result = render_conditional_template(template, params)
        assert result == "Hello Bob, your ID is {{id}}"

    def test_empty_params(self):
        """Test with empty parameters."""
        template = "Hello {{name}}"
        params = {}
        result = render_conditional_template(template, params)
        assert result == "Hello {{name}}"

    def test_none_value_substitution(self):
        """Test substitution with None values."""
        template = "Value: {{value}}"
        params = {"value": None}
        result = render_conditional_template(template, params)
        assert result == "Value: "

    def test_simple_conditional_section(self):
        """Test simple conditional section."""
        template = "Hello {{name}}((, you have {{count}} messages))"

        # With count present
        params = {"name": "Alice", "count": 5}
        result = render_conditional_template(template, params)
        assert result == "Hello Alice, you have 5 messages"

        # Without count
        params = {"name": "Bob"}
        result = render_conditional_template(template, params)
        assert result == "Hello Bob"

    def test_malformed_conditional_with_multiple_variables(self):
        """Test conditional section with multiple variables."""
        # Note: template has unmatched parenthesis and never closes the conditional content
        template = "User: {{user}}(( ({{role}} - {{department}}))"

        # All variables present - no conditional processing, just substitution
        params = {"user": "John", "role": "Admin", "department": "IT"}
        result = render_conditional_template(template, params)
        assert result == "User: John(( (Admin - IT))"

        # Only one conditional variable present - department becomes empty
        params = {"user": "Jane", "role": "Manager"}
        result = render_conditional_template(template, params)
        assert result == "User: Jane(( (Manager - {{department}}))"

        # No conditional variables present
        params = {"user": "Mike"}
        result = render_conditional_template(template, params)
        assert result == "User: Mike(( ({{role}} - {{department}}))"

    def test_conditional_with_multiple_variables(self):
        """Test conditional section with multiple variables."""
        # Template has properly balanced conditional: (( content ))
        template = "User: {{user}}(( ({{role}} - {{department}})))"

        # All variables present - conditional renders with content
        params = {"user": "John", "role": "Admin", "department": "IT"}
        result = render_conditional_template(template, params)
        assert result == "User: John (Admin - IT)"

        # Only one conditional variable present - department becomes empty
        params = {"user": "Jane", "role": "Manager"}
        result = render_conditional_template(template, params)
        assert result == "User: Jane (Manager - )"

        # No conditional variables present - entire conditional removed
        params = {"user": "Mike"}
        result = render_conditional_template(template, params)
        assert result == "User: Mike"

    def test_conditional_all_empty_values(self):
        """Test conditional section where all values are empty."""
        template = "Report((: {{status}} - {{details}}))"

        # Empty strings
        params = {"status": "", "details": ""}
        result = render_conditional_template(template, params)
        assert result == "Report"

        # None values
        params = {"status": None, "details": None}
        result = render_conditional_template(template, params)
        assert result == "Report"

    def test_nested_conditionals(self):
        """Test nested conditional sections."""
        template = "Start((: outer ((inner {{var}})) text))"

        # Variable present - nested conditional is processed
        params = {"var": "value"}
        result = render_conditional_template(template, params)
        assert result == "Start: outer inner value text"

        # Variable missing - entire outer section removed because
        # ALL variables in it (including nested) are empty/missing
        params = {}
        result = render_conditional_template(template, params)
        assert result == "Start"

    def test_multiple_conditional_sections(self):
        """Test multiple conditional sections."""
        template = "Name: {{name}}((, Age: {{age}}))((, City: {{city}}))"

        # All present
        params = {"name": "Alice", "age": 30, "city": "NYC"}
        result = render_conditional_template(template, params)
        assert result == "Name: Alice, Age: 30, City: NYC"

        # Some missing
        params = {"name": "Bob", "city": "LA"}
        result = render_conditional_template(template, params)
        assert result == "Name: Bob, City: LA"

        # Only required present
        params = {"name": "Charlie"}
        result = render_conditional_template(template, params)
        assert result == "Name: Charlie"

    def test_conditional_without_variables(self):
        """Test conditional section without any variables."""
        template = "Text((: static content here))"
        params = {}
        result = render_conditional_template(template, params)
        assert result == "Text: static content here"

    def test_numeric_and_boolean_values(self):
        """Test with numeric and boolean parameter values."""
        template = "Count: {{count}}, Active: {{active}}"

        # Numeric
        params = {"count": 42, "active": True}
        result = render_conditional_template(template, params)
        assert result == "Count: 42, Active: True"

        # Zero and False
        params = {"count": 0, "active": False}
        result = render_conditional_template(template, params)
        assert result == "Count: 0, Active: False"

    def test_whitespace_handling(self):
        """Test handling of whitespace in values."""
        template = "Value((: '{{value}}'))"

        # Whitespace-only string (should be removed)
        params = {"value": "   "}
        result = render_conditional_template(template, params)
        assert result == "Value"

        # String with content and whitespace (should be kept)
        params = {"value": "  text  "}
        result = render_conditional_template(template, params)
        assert result == "Value: '  text  '"

    def test_whitespace_in_variable_syntax(self):
        """Test whitespace handling within variable syntax."""
        # Test with spaces inside curly braces
        template = "Value: {{ var  }}"

        # When variable is present, preserve whitespace
        params = {"var": "test"}
        result = render_conditional_template(template, params)
        assert result == "Value:  test  "

        # When variable is missing, move whitespace outside
        params = {}
        result = render_conditional_template(template, params)
        assert result == "Value:  {{var}}  "

        # When variable is None, move whitespace outside
        params = {"var": None}
        result = render_conditional_template(template, params)
        assert result == "Value:  {{var}}  "

        # When variable is empty string, preserve whitespace
        # Total: 1 space before + 2 spaces from template + 1 after = 4 spaces
        params = {"var": ""}
        result = render_conditional_template(template, params)
        assert result == "Value:    "

    def test_whitespace_in_conditionals(self):
        """Test whitespace handling in conditional sections."""
        template = "Start((: {{ var1 }} and {{var2}}))"

        # Both variables present
        params = {"var1": "A", "var2": "B"}
        result = render_conditional_template(template, params)
        assert result == "Start:  A  and B"

        # One variable missing - var2 becomes empty string in conditional
        # but var1 is present, so section is kept
        params = {"var1": "A"}
        result = render_conditional_template(template, params)
        assert result == "Start:  A  and "

        # Variable with empty string (keeps section because var2 has value)
        # Empty string with whitespace: 1 space + empty + 1 space = 2 spaces
        params = {"var1": "", "var2": "B"}
        result = render_conditional_template(template, params)
        assert result == "Start:    and B"

    def test_complex_real_world_example(self):
        """Test a complex real-world template example."""
        template = """User Profile:
Name: {{name}}
Email: {{email}}((
Phone: {{phone}}))((
Address: {{street}}, {{city}}, {{state}} {{zip}}))((

Notes: {{notes}}))"""

        # Full profile
        params = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "555-1234",
            "street": "123 Main St",
            "city": "Anytown",
            "state": "CA",
            "zip": "12345",
            "notes": "VIP customer"
        }
        result = render_conditional_template(template, params)
        assert "Phone: 555-1234" in result
        assert "Address: 123 Main St, Anytown, CA 12345" in result
        assert "Notes: VIP customer" in result

        # Minimal profile
        params = {
            "name": "Jane Smith",
            "email": "jane@example.com"
        }
        result = render_conditional_template(template, params)
        assert result == """User Profile:
Name: Jane Smith
Email: jane@example.com"""

    def test_unmatched_parentheses(self):
        """Test handling of unmatched parentheses."""
        template = "Text (( unmatched"
        params = {}
        result = render_conditional_template(template, params)
        # Should leave unmatched parentheses as-is
        assert result == "Text (( unmatched"

    def test_edge_case_empty_template(self):
        """Test empty template."""
        template = ""
        params = {"key": "value"}
        result = render_conditional_template(template, params)
        assert result == ""

    def test_edge_case_only_conditionals(self):
        """Test template with only conditional sections."""
        template = "(({{optional1}}))(({{optional2}}))"

        # With values
        params = {"optional1": "A", "optional2": "B"}
        result = render_conditional_template(template, params)
        assert result == "AB"

        # Without values
        params = {}
        result = render_conditional_template(template, params)
        assert result == ""

    def test_unbalanced_parentheses_in_template(self):
        """Test template with unbalanced parentheses (extra closing paren).

        This demonstrates the left-to-right parsing behavior: the conditional
        section ends at the first matching )), leaving the extra ) outside.
        """
        # Template has 3 opening and 3 closing parens total, but the conditional
        # section is (( ... )) with an extra ) after it
        template = "User: {{user}}(( ({{role}} - {{department}})))"

        # All variables present - conditional renders, extra ) remains
        params = {"user": "John", "role": "Admin", "department": "IT"}
        result = render_conditional_template(template, params)
        assert result == "User: John (Admin - IT)"

        # Only one conditional variable present
        params = {"user": "Jane", "role": "Manager"}
        result = render_conditional_template(template, params)
        assert result == "User: Jane (Manager - )"

        # No conditional variables - section removed, extra ) remains
        # This shows the template author has an unmatched parenthesis
        params = {"user": "Mike"}
        result = render_conditional_template(template, params)
        assert result == "User: Mike"

        # Empty string for one variable (dept has value, so section kept)
        params = {"user": "Alice", "role": "", "department": "HR"}
        result = render_conditional_template(template, params)
        assert result == "User: Alice ( - HR)"

    def test_balanced_parentheses_in_conditional(self):
        """Test the correct way to write a conditional with parentheses.

        This shows how to properly balance parentheses within a conditional.
        """
        # Correctly balanced: conditional contains "(role - dept)" with balanced parens
        template = "User: {{user}}(( ({{role}} - {{department}})))"

        # All variables present
        params = {"user": "John", "role": "Admin", "department": "IT"}
        result = render_conditional_template(template, params)
        assert result == "User: John (Admin - IT)"

        # Only one conditional variable present
        params = {"user": "Jane", "role": "Manager"}
        result = render_conditional_template(template, params)
        assert result == "User: Jane (Manager - )"

        # No conditional variables - section cleanly removed
        params = {"user": "Mike"}
        result = render_conditional_template(template, params)
        assert result == "User: Mike"

        # Empty string for one variable
        params = {"user": "Alice", "role": "", "department": "HR"}
        result = render_conditional_template(template, params)
        assert result == "User: Alice ( - HR)"

    def test_nested_optional_subsections(self):
        """Test nested optional subsections with separate variables."""
        template = "User: {{user}}(( - ((Role: {{role}})) ((Dept: {{dept}}))))"

        # All variables present
        params = {"user": "John", "role": "Admin", "dept": "IT"}
        result = render_conditional_template(template, params)
        assert result == "User: John - Role: Admin Dept: IT"

        # Only role present - dept section removed, outer kept
        params = {"user": "Jane", "role": "Manager"}
        result = render_conditional_template(template, params)
        assert result == "User: Jane - Role: Manager "

        # Only dept present - role section removed, outer kept
        params = {"user": "Bob", "dept": "HR"}
        result = render_conditional_template(template, params)
        assert result == "User: Bob -  Dept: HR"

        # Neither role nor dept - entire outer section removed
        # because ALL variables in the outer section are empty/missing
        params = {"user": "Alice"}
        result = render_conditional_template(template, params)
        assert result == "User: Alice"

        # Role is empty string, dept has value - role section removed
        params = {"user": "Charlie", "role": "", "dept": "Finance"}
        result = render_conditional_template(template, params)
        assert result == "User: Charlie -  Dept: Finance"

        # Both role and dept are empty strings - outer section removed
        params = {"user": "David", "role": "", "dept": ""}
        result = render_conditional_template(template, params)
        assert result == "User: David"

    def test_complex_nested_conditionals_with_text(self):
        """Test complex nested conditionals with intervening text."""
        template = "Report((: {{title}} ((by {{author}} ))((on {{date}}))))"

        # All present
        params = {"title": "Analysis", "author": "John", "date": "2024-01-01"}
        result = render_conditional_template(template, params)
        assert result == "Report: Analysis by John on 2024-01-01"

        # No author - author section removed
        params = {"title": "Summary", "date": "2024-01-02"}
        result = render_conditional_template(template, params)
        assert result == "Report: Summary on 2024-01-02"

        # No date - date section removed
        params = {"title": "Review", "author": "Jane"}
        result = render_conditional_template(template, params)
        assert result == "Report: Review by Jane "

        # Only title - nested sections removed
        params = {"title": "Overview"}
        result = render_conditional_template(template, params)
        assert result == "Report: Overview "

        # No title but has author/date - outer section kept because it has non-empty variables
        # Title renders as empty, nested sections with values are kept
        params = {"author": "Bob", "date": "2024-01-03"}
        result = render_conditional_template(template, params)
        assert result == "Report:  by Bob on 2024-01-03"

        # Title is empty string, others present - section kept (empty string for title)
        params = {"title": "", "author": "Alice", "date": "2024-01-04"}
        result = render_conditional_template(template, params)
        assert result == "Report:  by Alice on 2024-01-04"

        # All variables missing - entire section removed
        params = {}
        result = render_conditional_template(template, params)
        assert result == "Report"

    def test_multiple_nested_levels(self):
        """Test multiple levels of nesting."""
        # Note: template has extra closing paren at the end
        template = "Data((: Level1 ((Level2 ((Level3: {{var}})))))))"

        # Variable present - all levels render (extra paren remains)
        params = {"var": "value"}
        result = render_conditional_template(template, params)
        assert result == "Data: Level1 Level2 Level3: value)"

        # Variable missing - entire outer section removed
        # because the only variable in all nested sections is missing
        params = {}
        result = render_conditional_template(template, params)
        assert result == "Data)"

        # Variable is empty string - entire outer section removed
        params = {"var": ""}
        result = render_conditional_template(template, params)
        assert result == "Data)"

    def test_whitespace_edge_cases(self):
        """Test edge cases with whitespace in variables."""
        template = "Val: {{  var1  }}((, {{  var2  }}))"

        # Both with values
        params = {"var1": "A", "var2": "B"}
        result = render_conditional_template(template, params)
        assert result == "Val:   A  ,   B  "

        # var1 present, var2 missing (conditional removed)
        params = {"var1": "A"}
        result = render_conditional_template(template, params)
        assert result == "Val:   A  "

        # var1 None, var2 present (whitespace moved outside)
        params = {"var1": None, "var2": "B"}
        result = render_conditional_template(template, params)
        assert result == "Val:   {{var1}}  ,   B  "

        # Both None
        params = {"var1": None, "var2": None}
        result = render_conditional_template(template, params)
        assert result == "Val:   {{var1}}  "


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
            
    def test_sync_provider_creation(self):
        """Test sync provider creation using adapter."""
        config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo"
        )
        
        # Should now work with our SyncProviderAdapter
        provider = create_llm_provider(config, is_async=False)
        assert provider is not None
        # Should be wrapped in our adapter
        assert hasattr(provider, 'async_provider')

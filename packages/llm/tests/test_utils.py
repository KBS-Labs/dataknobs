"""Tests for dataknobs_llm.llm.utils module."""

import pytest
from dataknobs_llm.llm.utils import (
    TemplateStrategy,
    MessageTemplate,
    render_conditional_template,
    MessageBuilder,
    ResponseParser,
    TokenCounter,
    CostCalculator,
    chain_prompts,
    create_few_shot_prompt,
)
from dataknobs_llm.llm.base import LLMMessage, LLMResponse


def test_template_strategy_enum():
    """Test TemplateStrategy enum values."""
    assert TemplateStrategy.SIMPLE.value == "simple"
    assert TemplateStrategy.CONDITIONAL.value == "conditional"


# MessageTemplate with SIMPLE strategy tests

def test_prompt_template_simple_basic():
    """Test basic MessageTemplate with SIMPLE strategy."""
    template = MessageTemplate("Hello {name}!", ["name"])
    assert template.strategy == TemplateStrategy.SIMPLE
    result = template.format(name="Alice")
    assert result == "Hello Alice!"


def test_prompt_template_simple_auto_extract_variables():
    """Test SIMPLE template auto-extracts variables."""
    template = MessageTemplate("Hello {name}, you are {age} years old.")
    assert set(template.variables) == {"name", "age"}
    result = template.format(name="Bob", age=30)
    assert result == "Hello Bob, you are 30 years old."


def test_prompt_template_simple_missing_variable():
    """Test SIMPLE template raises error for missing variables."""
    template = MessageTemplate("Hello {name}!")
    with pytest.raises(ValueError) as exc_info:
        template.format()
    assert "Missing variables" in str(exc_info.value)


def test_prompt_template_simple_partial():
    """Test SIMPLE template partial substitution."""
    template = MessageTemplate("Hello {name}, you are {age} years old.")
    partial = template.partial(name="Alice")
    assert "Alice" in partial.template
    assert "age" in partial.variables
    assert "name" not in partial.variables

    result = partial.format(age=25)
    assert result == "Hello Alice, you are 25 years old."


def test_prompt_template_simple_multiple_variables():
    """Test SIMPLE template with multiple variables."""
    template = MessageTemplate(
        "Name: {name}\nAge: {age}\nCity: {city}",
        ["name", "age", "city"]
    )
    result = template.format(name="Charlie", age=35, city="NYC")
    assert "Name: Charlie" in result
    assert "Age: 35" in result
    assert "City: NYC" in result


# MessageTemplate with CONDITIONAL strategy tests

def test_prompt_template_conditional_basic():
    """Test basic CONDITIONAL template."""
    template = MessageTemplate.from_conditional("Hello {{name}}!")
    assert template.strategy == TemplateStrategy.CONDITIONAL
    result = template.format(name="Alice")
    assert result == "Hello Alice!"


def test_prompt_template_conditional_auto_extract_variables():
    """Test CONDITIONAL template auto-extracts variables."""
    template = MessageTemplate.from_conditional(
        "Hello {{name}}((, you have {{count}} messages))"
    )
    assert set(template.variables) == {"name", "count"}


def test_prompt_template_conditional_with_value():
    """Test CONDITIONAL template includes section when variable has value."""
    template = MessageTemplate.from_conditional(
        "Hello {{name}}((, you have {{count}} messages))"
    )
    result = template.format(name="Alice", count=5)
    assert result == "Hello Alice, you have 5 messages"


def test_prompt_template_conditional_without_value():
    """Test CONDITIONAL template removes section when variable is missing."""
    template = MessageTemplate.from_conditional(
        "Hello {{name}}((, you have {{count}} messages))"
    )
    result = template.format(name="Bob")
    assert result == "Hello Bob"
    assert "messages" not in result


def test_prompt_template_conditional_missing_variable():
    """Test CONDITIONAL template handles missing variables gracefully."""
    template = MessageTemplate.from_conditional("Hello {{name}}!")
    # Missing variables outside conditionals are left as-is
    result = template.format()
    assert "{{name}}" in result


def test_prompt_template_conditional_partial():
    """Test CONDITIONAL template partial substitution."""
    template = MessageTemplate.from_conditional(
        "Hello {{name}}((, you have {{count}} messages))"
    )
    partial = template.partial(name="Alice")
    # name should be replaced, count should remain
    result = partial.format(count=10)
    assert "Alice" in result
    assert "10" in result


def test_prompt_template_conditional_nested():
    """Test CONDITIONAL template with nested conditionals."""
    template = MessageTemplate.from_conditional(
        "Hello {{name}}((, age {{age}}((, from {{city}}))))"
    )
    # All provided
    result = template.format(name="Alice", age=30, city="NYC")
    assert result == "Hello Alice, age 30, from NYC"

    # Only name and age
    result = template.format(name="Bob", age=25)
    assert result == "Hello Bob, age 25"

    # Only name
    result = template.format(name="Charlie")
    assert result == "Hello Charlie"


def test_render_conditional_template_basic():
    """Test render_conditional_template basic usage."""
    result = render_conditional_template("Hello {{name}}!", {"name": "Alice"})
    assert result == "Hello Alice!"


def test_render_conditional_template_with_conditional():
    """Test render_conditional_template with conditional sections."""
    template = "Hello {{name}}((, you are {{age}} years old))"
    result = render_conditional_template(template, {"name": "Alice", "age": 30})
    assert result == "Hello Alice, you are 30 years old"

    result = render_conditional_template(template, {"name": "Bob"})
    assert result == "Hello Bob"


def test_render_conditional_template_whitespace():
    """Test render_conditional_template whitespace handling."""
    template = "Hello {{ name }}"
    result = render_conditional_template(template, {"name": "Alice"})
    assert result == "Hello  Alice "


def test_render_conditional_template_empty_string():
    """Test render_conditional_template with empty string value."""
    template = "Hello {{name}}((, you have {{count}} items))"
    # Empty string should remove conditional
    result = render_conditional_template(template, {"name": "Alice", "count": ""})
    assert result == "Hello Alice"


def test_render_conditional_missing_variable_unchanged():
    """Test that missing variables remain unchanged."""
    template = "Hello {{name}}, your ID is {{id}}"
    params = {"name": "Bob"}
    result = render_conditional_template(template, params)
    assert result == "Hello Bob, your ID is {{id}}"


def test_render_conditional_empty_params():
    """Test with empty parameters."""
    template = "Hello {{name}}"
    params = {}
    result = render_conditional_template(template, params)
    assert result == "Hello {{name}}"


def test_render_conditional_none_value_substitution():
    """Test substitution with None values."""
    template = "Value: {{value}}"
    params = {"value": None}
    result = render_conditional_template(template, params)
    assert result == "Value: "


def test_render_conditional_malformed_conditional():
    """Test conditional section with unmatched parentheses."""
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


def test_render_conditional_multiple_variables():
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


def test_render_conditional_all_empty_values():
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


def test_render_conditional_nested():
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


def test_render_conditional_multiple_sections():
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


def test_render_conditional_without_variables():
    """Test conditional section without any variables."""
    template = "Text((: static content here))"
    params = {}
    result = render_conditional_template(template, params)
    assert result == "Text: static content here"


def test_render_conditional_numeric_and_boolean():
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


def test_render_conditional_whitespace_handling():
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


def test_render_conditional_whitespace_in_variable_syntax():
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
    params = {"var": ""}
    result = render_conditional_template(template, params)
    assert result == "Value:    "


def test_render_conditional_whitespace_in_conditionals():
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
    params = {"var1": "", "var2": "B"}
    result = render_conditional_template(template, params)
    assert result == "Start:    and B"


def test_render_conditional_complex_real_world():
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


def test_render_conditional_unmatched_parentheses():
    """Test handling of unmatched parentheses."""
    template = "Text (( unmatched"
    params = {}
    result = render_conditional_template(template, params)
    # Should leave unmatched parentheses as-is
    assert result == "Text (( unmatched"


def test_render_conditional_empty_template():
    """Test empty template."""
    template = ""
    params = {"key": "value"}
    result = render_conditional_template(template, params)
    assert result == ""


def test_render_conditional_only_conditionals():
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


def test_render_conditional_unbalanced_extra_closing():
    """Test template with unbalanced parentheses (extra closing paren)."""
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

    # No conditional variables - section removed
    params = {"user": "Mike"}
    result = render_conditional_template(template, params)
    assert result == "User: Mike"

    # Empty string for one variable (dept has value, so section kept)
    params = {"user": "Alice", "role": "", "department": "HR"}
    result = render_conditional_template(template, params)
    assert result == "User: Alice ( - HR)"


def test_render_conditional_balanced_parens_in_content():
    """Test conditional with balanced parentheses in content."""
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


def test_render_conditional_nested_optional_subsections():
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


def test_render_conditional_complex_nested_with_text():
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

    # No title but has author/date - outer section kept
    params = {"author": "Bob", "date": "2024-01-03"}
    result = render_conditional_template(template, params)
    assert result == "Report:  by Bob on 2024-01-03"

    # Title is empty string, others present - section kept
    params = {"title": "", "author": "Alice", "date": "2024-01-04"}
    result = render_conditional_template(template, params)
    assert result == "Report:  by Alice on 2024-01-04"

    # All variables missing - entire section removed
    params = {}
    result = render_conditional_template(template, params)
    assert result == "Report"


def test_render_conditional_multiple_nested_levels():
    """Test multiple levels of nesting."""
    # Note: template has extra closing paren at the end
    template = "Data((: Level1 ((Level2 ((Level3: {{var}})))))))"

    # Variable present - all levels render (extra paren remains)
    params = {"var": "value"}
    result = render_conditional_template(template, params)
    assert result == "Data: Level1 Level2 Level3: value)"

    # Variable missing - entire outer section removed
    params = {}
    result = render_conditional_template(template, params)
    assert result == "Data)"

    # Variable is empty string - entire outer section removed
    params = {"var": ""}
    result = render_conditional_template(template, params)
    assert result == "Data)"


def test_render_conditional_whitespace_edge_cases():
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


# MessageBuilder tests

def test_message_builder_basic():
    """Test MessageBuilder basic usage."""
    builder = MessageBuilder()
    messages = builder.system("You are helpful").user("Hello").build()

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == "You are helpful"
    assert messages[1].role == "user"
    assert messages[1].content == "Hello"


def test_message_builder_chaining():
    """Test MessageBuilder method chaining."""
    messages = (
        MessageBuilder()
        .system("You are helpful")
        .user("What is 2+2?")
        .assistant("4")
        .user("Thanks!")
        .build()
    )

    assert len(messages) == 4
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]


def test_message_builder_function():
    """Test MessageBuilder function messages."""
    builder = MessageBuilder()
    func_call = {"name": "get_weather", "arguments": {"city": "NYC"}}
    messages = builder.function("get_weather", "Sunny, 72°F", func_call).build()

    assert len(messages) == 1
    assert messages[0].role == "function"
    assert messages[0].name == "get_weather"
    assert messages[0].content == "Sunny, 72°F"
    assert messages[0].function_call == func_call


def test_message_builder_from_template():
    """Test MessageBuilder from_template."""
    template = MessageTemplate("Hello {name}, you are {age} years old.")
    messages = (
        MessageBuilder()
        .from_template("user", template, name="Alice", age=30)
        .build()
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello Alice, you are 30 years old."


def test_message_builder_clear():
    """Test MessageBuilder clear."""
    builder = MessageBuilder()
    builder.user("Hello").assistant("Hi")
    assert len(builder.messages) == 2

    builder.clear()
    assert len(builder.messages) == 0

    # Can reuse after clear
    messages = builder.user("New message").build()
    assert len(messages) == 1


# ResponseParser tests

def test_response_parser_extract_json_simple():
    """Test ResponseParser extract_json with simple object."""
    text = '{"name": "Alice", "age": 30}'
    result = ResponseParser.extract_json(text)
    assert result == {"name": "Alice", "age": 30}


def test_response_parser_extract_json_in_text():
    """Test ResponseParser extract_json with JSON in text."""
    text = 'Here is the data: {"name": "Bob", "age": 25} - that is all.'
    result = ResponseParser.extract_json(text)
    assert result == {"name": "Bob", "age": 25}


def test_response_parser_extract_json_code_block():
    """Test ResponseParser extract_json with markdown code block."""
    text = '''Here's the JSON:
```json
{"name": "Charlie", "age": 35}
```
'''
    result = ResponseParser.extract_json(text)
    assert result == {"name": "Charlie", "age": 35}


def test_response_parser_extract_json_from_response():
    """Test ResponseParser extract_json with LLMResponse."""
    response = LLMResponse(
        content='{"result": "success"}',
        model="gpt-4"
    )
    result = ResponseParser.extract_json(response)
    assert result == {"result": "success"}


def test_response_parser_extract_json_none():
    """Test ResponseParser extract_json returns None for invalid JSON."""
    text = "This is not JSON at all"
    result = ResponseParser.extract_json(text)
    assert result is None


def test_response_parser_extract_code_basic():
    """Test ResponseParser extract_code basic usage."""
    text = '''Here's some code:
```python
def hello():
    print("Hello")
```
'''
    result = ResponseParser.extract_code(text)
    assert len(result) == 1
    assert 'def hello():' in result[0]


def test_response_parser_extract_code_multiple():
    """Test ResponseParser extract_code with multiple blocks."""
    text = '''
```python
print("First")
```
Some text
```python
print("Second")
```
'''
    result = ResponseParser.extract_code(text)
    assert len(result) == 2
    assert 'First' in result[0]
    assert 'Second' in result[1]


def test_response_parser_extract_code_language_filter():
    """Test ResponseParser extract_code with language filter."""
    text = '''
```python
print("Python")
```
```javascript
console.log("JS")
```
'''
    python_result = ResponseParser.extract_code(text, language="python")
    assert len(python_result) == 1
    assert "Python" in python_result[0]

    js_result = ResponseParser.extract_code(text, language="javascript")
    assert len(js_result) == 1
    assert "JS" in js_result[0]


def test_response_parser_extract_list_bullets():
    """Test ResponseParser extract_list with bullet points."""
    text = '''
Here are items:
- First item
- Second item
- Third item
'''
    result = ResponseParser.extract_list(text)
    assert len(result) == 3
    assert result[0] == "First item"
    assert result[1] == "Second item"
    assert result[2] == "Third item"


def test_response_parser_extract_list_numbered():
    """Test ResponseParser extract_list with numbered list."""
    text = '''
Steps:
1. First step
2. Second step
3. Third step
'''
    result = ResponseParser.extract_list(text, numbered=True)
    assert len(result) == 3
    assert result[0] == "First step"


def test_response_parser_extract_sections():
    """Test ResponseParser extract_sections."""
    text = '''
# Introduction
This is the intro.

# Body
This is the body.

# Conclusion
This is the end.
'''
    result = ResponseParser.extract_sections(text)
    assert "Introduction" in result
    assert "Body" in result
    assert "Conclusion" in result
    assert "intro" in result["Introduction"]
    assert "body" in result["Body"]
    assert "end" in result["Conclusion"]


# TokenCounter tests

def test_token_counter_estimate_basic():
    """Test TokenCounter estimate_tokens basic usage."""
    text = "Hello, world!"
    tokens = TokenCounter.estimate_tokens(text)
    assert tokens > 0
    assert tokens < len(text)  # Should be less than character count


def test_token_counter_estimate_different_models():
    """Test TokenCounter estimate_tokens with different models."""
    text = "Hello, world! " * 10
    gpt4_tokens = TokenCounter.estimate_tokens(text, "gpt-4")
    llama_tokens = TokenCounter.estimate_tokens(text, "llama")

    # Different models have different ratios
    assert gpt4_tokens > 0
    assert llama_tokens > 0


def test_token_counter_estimate_messages():
    """Test TokenCounter estimate_messages_tokens."""
    messages = [
        LLMMessage(role="system", content="You are helpful"),
        LLMMessage(role="user", content="Hello"),
        LLMMessage(role="assistant", content="Hi there!")
    ]
    tokens = TokenCounter.estimate_messages_tokens(messages)
    # Should include role tokens (4 per message) plus content
    assert tokens >= 12  # At least 3 * 4


def test_token_counter_fits_in_context():
    """Test TokenCounter fits_in_context."""
    short_text = "Hello"
    long_text = "Hello " * 10000

    assert TokenCounter.fits_in_context(short_text, "gpt-4", 100)
    assert not TokenCounter.fits_in_context(long_text, "gpt-4", 100)


# CostCalculator tests

def test_cost_calculator_calculate_cost():
    """Test CostCalculator calculate_cost."""
    response = LLMResponse(
        content="Hello",
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
    # 100 * 0.03/1000 + 50 * 0.06/1000 = 0.003 + 0.003 = 0.006
    assert abs(cost - 0.006) < 0.0001


def test_cost_calculator_calculate_cost_no_usage():
    """Test CostCalculator calculate_cost with no usage info."""
    response = LLMResponse(content="Hello", model="gpt-4")
    cost = CostCalculator.calculate_cost(response)
    assert cost is None


def test_cost_calculator_calculate_cost_unknown_model():
    """Test CostCalculator calculate_cost with unknown model."""
    response = LLMResponse(
        content="Hello",
        model="unknown-model",
        usage={"prompt_tokens": 100, "completion_tokens": 50}
    )
    cost = CostCalculator.calculate_cost(response)
    assert cost is None


def test_cost_calculator_estimate_cost():
    """Test CostCalculator estimate_cost."""
    text = "Hello world " * 100
    cost = CostCalculator.estimate_cost(text, "gpt-3.5-turbo")
    assert cost is not None
    assert cost > 0


# chain_prompts tests

def test_chain_prompts_basic():
    """Test chain_prompts basic usage."""
    t1 = MessageTemplate("First {a}")
    t2 = MessageTemplate("Second {b}")
    combined = chain_prompts(t1, t2)

    assert "First" in combined.template
    assert "Second" in combined.template
    assert set(combined.variables) == {"a", "b"}

    result = combined.format(a="A", b="B")
    assert "First A" in result
    assert "Second B" in result


def test_chain_prompts_shared_variables():
    """Test chain_prompts with shared variables."""
    t1 = MessageTemplate("Hello {name}")
    t2 = MessageTemplate("Goodbye {name}")
    combined = chain_prompts(t1, t2)

    # Should only have 'name' once in variables
    assert combined.variables.count("name") == 1

    result = combined.format(name="Alice")
    assert "Hello Alice" in result
    assert "Goodbye Alice" in result


def test_chain_prompts_empty():
    """Test chain_prompts with no templates."""
    combined = chain_prompts()
    assert combined.template == ""
    assert combined.variables == []


def test_chain_prompts_mixed_strategies_error():
    """Test chain_prompts raises error for mixed strategies."""
    t1 = MessageTemplate("Simple {a}")  # SIMPLE strategy
    t2 = MessageTemplate.from_conditional("Conditional {{b}}")  # CONDITIONAL

    with pytest.raises(ValueError) as exc_info:
        chain_prompts(t1, t2)
    assert "different strategies" in str(exc_info.value)


def test_chain_prompts_conditional():
    """Test chain_prompts with CONDITIONAL templates."""
    t1 = MessageTemplate.from_conditional("First {{a}}")
    t2 = MessageTemplate.from_conditional("Second {{b}}((, optional {{c}}))")
    combined = chain_prompts(t1, t2)

    assert combined.strategy == TemplateStrategy.CONDITIONAL
    result = combined.format(a="A", b="B", c="C")
    assert "First A" in result
    assert "Second B, optional C" in result


# create_few_shot_prompt tests

def test_create_few_shot_prompt_basic():
    """Test create_few_shot_prompt basic usage."""
    instruction = "Translate English to French"
    examples = [
        {"input": "Hello", "output": "Bonjour"},
        {"input": "Goodbye", "output": "Au revoir"}
    ]

    template = create_few_shot_prompt(instruction, examples)
    assert "Translate English to French" in template.template
    assert "Example 1:" in template.template
    assert "Hello" in template.template
    assert "Bonjour" in template.template
    assert "query" in template.variables

    result = template.format(query="Good morning")
    assert "Good morning" in result


def test_create_few_shot_prompt_custom_keys():
    """Test create_few_shot_prompt with custom keys."""
    instruction = "Calculate result"
    examples = [
        {"question": "2+2", "answer": "4"},
        {"question": "3*3", "answer": "9"}
    ]

    template = create_few_shot_prompt(
        instruction,
        examples,
        query_key="question",
        response_key="answer"
    )
    assert "2+2" in template.template
    assert "4" in template.template


def test_create_few_shot_prompt_multiple_examples():
    """Test create_few_shot_prompt with many examples."""
    examples = [
        {"input": f"Test {i}", "output": f"Result {i}"}
        for i in range(5)
    ]
    template = create_few_shot_prompt("Task", examples)

    for i in range(5):
        assert f"Test {i}" in template.template
        assert f"Result {i}" in template.template

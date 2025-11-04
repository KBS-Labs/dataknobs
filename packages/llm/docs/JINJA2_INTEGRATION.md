# Jinja2 Template Integration

## Overview

The DataKnobs LLM package now supports **Jinja2**, the powerful Python templating engine, alongside our custom `(( ))` conditional syntax. This integration provides:

- **50+ built-in filters** (upper, lower, truncate, default, round, etc.)
- **Advanced conditionals** with `{% if/elif/else %}` and boolean logic
- **Template includes** for reusable components
- **Loops** with `{% for %}`
- **Macros** for reusable template functions
- **Template inheritance** with `{% extends %}` and `{% block %}`
- **Custom filters** for domain-specific transformations
- **Full backward compatibility** with existing templates

## Template Modes

Choose between two rendering modes:

### Mixed Mode (Default)

Supports both custom `(( ))` syntax and Jinja2 features:

```python
from dataknobs_llm.prompts.rendering import TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode

renderer = TemplateRenderer()

# Uses MIXED mode by default
result = renderer.render(
    "Hello {{name|upper}}((, age {{age}}))",
    {"name": "alice", "age": 30}
)
# Output: "Hello ALICE, age 30"
```

**Features:**
- Pre-processes `(( ))` conditionals first
- Then applies Jinja2 rendering
- Backward compatible with all existing templates
- **Restriction:** No Jinja2 syntax (`{% %}` or filters) inside `(( ))` blocks

### Jinja2 Mode (Pure Jinja2)

Pure Jinja2 templating without `(( ))` preprocessing:

```python
renderer = TemplateRenderer()

result = renderer.render(
    "Hello {{name|upper}}{% if age %}, age {{age}}{% endif %}",
    {"name": "alice", "age": 30},
    mode=TemplateMode.JINJA2
)
# Output: "Hello ALICE, age 30"
```

**Features:**
- Full Jinja2 power with no restrictions
- Better for new templates
- Slightly faster (one rendering pass)

## Quick Start Examples

### 1. Filters

Transform variables with filters:

```python
renderer = TemplateRenderer()

# Uppercase
result = renderer.render("{{name|upper}}", {"name": "alice"})
# Output: "ALICE"

# Default value for missing variables
result = renderer.render("{{name|default('Guest')}}", {})
# Output: "Guest"

# Chain multiple filters
result = renderer.render("{{name|lower|capitalize}}", {"name": "ALICE"})
# Output: "Alice"

# Length of collections
result = renderer.render("Found {{items|length}} items", {"items": [1, 2, 3]})
# Output: "Found 3 items"

# Join lists
result = renderer.render(
    "Tags: {{tags|join(', ')}}",
    {"tags": ["python", "llm", "jinja2"]}
)
# Output: "Tags: python, llm, jinja2"
```

### 2. Conditionals

Advanced conditional logic:

```python
# Simple if
template = "{% if age >= 18 %}Adult{% else %}Minor{% endif %}"
result = renderer.render(template, {"age": 20}, mode=TemplateMode.JINJA2)
# Output: "Adult"

# If/elif/else
template = """
{% if score >= 90 %}
A grade
{% elif score >= 80 %}
B grade
{% elif score >= 70 %}
C grade
{% else %}
Needs improvement
{% endif %}
"""

# Boolean operators
template = "{% if verified and age >= 18 %}Approved{% endif %}"
result = renderer.render(
    template,
    {"verified": True, "age": 20},
    mode=TemplateMode.JINJA2
)
# Output: "Approved"
```

### 3. Loops

Iterate over collections:

```python
template = """
Files to review:
{% for file in files %}
{{ loop.index }}. {{ file.name }} ({{ file.lines }} lines)
{% endfor %}
"""

result = renderer.render(
    template,
    {
        "files": [
            {"name": "main.py", "lines": 150},
            {"name": "utils.py", "lines": 75},
        ]
    },
    mode=TemplateMode.JINJA2
)
# Output:
# Files to review:
# 1. main.py (150 lines)
# 2. utils.py (75 lines)
```

### 4. Built-in Custom Filters

DataKnobs provides custom filters for common LLM tasks:

```python
# Format code blocks
template = "{{code|format_code('python')}}"
result = renderer.render(template, {"code": "print('hello')"})
# Output:
# ```python
# print('hello')
# ```

# Count tokens (approximate)
template = "This text is approximately {{text|count_tokens}} tokens"
```

### 5. Custom Filters

Add your own domain-specific filters:

```python
renderer = TemplateRenderer()

# Register custom filter
def highlight_keywords(text: str, keywords: list) -> str:
    for keyword in keywords:
        text = text.replace(keyword, f"**{keyword}**")
    return text

renderer.add_custom_filter('highlight', highlight_keywords)

# Use in template
template = "{{description|highlight(['security', 'vulnerability'])}}"
result = renderer.render(
    template,
    {"description": "Found a security issue with potential vulnerability"}
)
# Output: "Found a **security** issue with potential **vulnerability**"
```

## Configuration

### In YAML Files

Specify template mode in your prompt YAML files:

```yaml
# system/code_reviewer.yaml
template: |
  You are an expert code reviewer.

  Review the following {{language|upper}} code:

  ```{{language}}
  {{code}}
  ```

  {% if focus_areas %}
  Focus on: {{ focus_areas|join(', ') }}
  {% endif %}

  ((Guidelines: {{guidelines}}))

template_mode: mixed  # or "jinja2"

defaults:
  language: python
  focus_areas: []

validation:
  level: warn
  required_params:
    - code
```

### In Code

```python
from dataknobs_llm.prompts.implementations import ConfigPromptLibrary
from dataknobs_llm.prompts.rendering import TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode

config = {
    "system": {
        "greet": {
            "template": "Hello {{name|upper}}!",
            "template_mode": "jinja2"
        }
    }
}

library = ConfigPromptLibrary(config)
renderer = TemplateRenderer()

# Mode is read from config automatically
prompt = library.get_system_prompt("greet")
result = renderer.render_prompt_template(prompt, {"name": "alice"})
# Output: "Hello ALICE!"

# Or override mode at render time
result = renderer.render_prompt_template(
    prompt,
    {"name": "alice"},
    mode_override=TemplateMode.MIXED
)
```

## Mixed Mode: Best of Both Worlds

Mixed mode lets you combine our `(( ))` syntax with Jinja2 features:

```python
template = """
Hello {{name|upper}}((, from {{city}}))
{% if premium %}⭐ Premium member{% endif %}
((Account type: {{account_type}}))
"""

result = renderer.render(
    template,
    {
        "name": "alice",
        "city": "NYC",
        "premium": True
        # account_type is missing - whole section removed
    }
)
# Output:
# Hello ALICE, from NYC
# ⭐ Premium member
```

### Syntax Restrictions in Mixed Mode

For clarity and to avoid conflicts, Jinja2 syntax is **not allowed inside** `(( ))` blocks:

```python
# ❌ NOT ALLOWED
template = "((Hello {{name|upper}}))"  # Filter inside (( ))
template = "(({% if age %}{{age}}{% endif %}))"  # Block inside (( ))

# ✅ ALLOWED
template = "{{name|upper}}((, age {{age}}))"  # Filter outside
template = "{% if verified %}✓{% endif %}((, expires {{date}}))"  # Block outside
```

If you need Jinja2 syntax everywhere, use **jinja2 mode** instead.

## Advanced Features

### Template Inheritance

Create reusable base templates:

```yaml
# base_analysis.yaml
template: |
  # {{ analysis_type }} Analysis

  Language: {{ language }}

  {% block content %}
  <!-- Override this block -->
  {% endblock %}

  {% block footer %}
  Generated at {{ timestamp }}
  {% endblock %}
template_mode: jinja2

# security_analysis.yaml
template: |
  {% extends 'base_analysis' %}

  {% block content %}
  Security scan results:
  {% for issue in security_issues %}
  - {{ issue.severity }}: {{ issue.description }}
  {% endfor %}
  {% endblock %}
template_mode: jinja2
```

### Macros

Define reusable template components:

```python
template = """
{% macro code_block(code, language='python') %}
```{{ language }}
{{ code }}
```
{% endmacro %}

{% macro section(title, content) %}
## {{ title }}
{{ content }}
{% endmacro %}

{{ section('Overview', description) }}

{{ section('Code', code_block(code, language)) }}
"""
```

### Whitespace Control

Control whitespace in your templates:

```python
# Trim whitespace with - modifier
template = """
{%- if items %}
  Items:
  {%- for item in items %}
  - {{ item }}
  {%- endfor %}
{%- endif %}
"""

# Or use trim_blocks and lstrip_blocks (already enabled by default)
```

## Common Patterns

### Conditional Sections

```python
# Show different content based on user type
template = """
{% if user_type == 'admin' %}
Admin Dashboard - Full Access
{% elif user_type == 'moderator' %}
Moderator Panel - Limited Access
{% else %}
User View - Read Only
{% endif %}
"""
```

### List Formatting

```python
# Numbered list
template = """
{% for item in items %}
{{ loop.index }}. {{ item }}
{% endfor %}
"""

# Bullet list with conditional formatting
template = """
{% for user in users %}
- {{ user.name }}{% if user.premium %} ⭐{% endif %}
{% endfor %}
"""

# Comma-separated with "and" for last item
template = """
{% for item in items %}
  {{- item }}
  {%- if not loop.last %}
    {%- if loop.index == items|length - 1 %} and {% else %}, {% endif %}
  {%- endif %}
{% endfor %}
"""
```

### Safe Defaults

```python
# Provide defaults for missing values
template = "Welcome {{name|default('Guest')}}"

# Multiple fallbacks
template = "{{primary|default(secondary)|default('Not available')}}"
```

## Built-in Filters Reference

Jinja2 provides 50+ built-in filters. Here are the most useful for LLM prompts:

### String Filters
- `upper` - Convert to uppercase
- `lower` - Convert to lowercase
- `capitalize` - Capitalize first letter
- `title` - Title case
- `trim` - Remove leading/trailing whitespace
- `truncate(length)` - Truncate to length
- `replace(old, new)` - Replace substring
- `wordcount` - Count words

### Collection Filters
- `length` - Get length
- `first` - Get first item
- `last` - Get last item
- `join(separator)` - Join with separator
- `sort` - Sort collection
- `unique` - Remove duplicates
- `reverse` - Reverse order

### Numeric Filters
- `round(precision)` - Round number
- `abs` - Absolute value
- `int` - Convert to integer
- `float` - Convert to float

### Other Filters
- `default(value)` - Default if undefined
- `safe` - Mark as safe HTML (use with caution)
- `format(*args)` - String formatting

[Full Jinja2 filter reference](https://jinja.palletsprojects.com/en/3.1.x/templates/#list-of-builtin-filters)

## Performance

Jinja2 is highly optimized:
- Templates are compiled to Python bytecode
- Compiled templates are cached automatically
- Mixed mode adds negligible overhead (~0.01ms)

**Benchmarks:**
- Simple templates: ~0.01ms
- Complex templates with loops: ~0.1-1ms
- Mixed mode preprocessing: ~0.01ms

## Troubleshooting

### Error: "Jinja2 filters not allowed inside conditional blocks"

**Problem:** Using filters inside `(( ))` in mixed mode.

```python
# ❌ Wrong
template = "((Hello {{name|upper}}))"
```

**Solution:** Move filters outside or use jinja2 mode:

```python
# ✅ Option 1: Move filter outside
template = "{{name|upper}}((, Hello))"

# ✅ Option 2: Use jinja2 mode
result = renderer.render(
    "{% if name %}Hello {{name|upper}}{% endif %}",
    params,
    mode=TemplateMode.JINJA2
)
```

### Error: "Jinja2 block syntax not allowed inside conditional blocks"

**Problem:** Using `{% %}` blocks inside `(( ))` in mixed mode.

**Solution:** Same as above - move outside or use jinja2 mode.

### Undefined Variable Shows as `{{variable}}`

This is **intentional** for backward compatibility. Undefined variables are preserved as placeholders rather than rendered as empty strings.

To get empty strings for undefined variables (pure Jinja2 behavior), use jinja2 mode or the `default` filter:

```python
template = "{{missing_var|default('')}}"
```

## Best Practices

1. **Use jinja2 mode for new templates** - Full power, no restrictions
2. **Use mixed mode for backward compatibility** - Gradually adopt Jinja2 features
3. **Apply filters outside `(( ))` blocks** - Or switch to jinja2 mode
4. **Use `default` filter for optional variables** - Better than checking with `{% if %}`
5. **Keep templates readable** - Break complex logic into macros
6. **Document custom filters** - With clear docstrings
7. **Test both defined and undefined params** - Ensure correct behavior

## Migration from Custom Syntax

See [JINJA2_MIGRATION.md](./JINJA2_MIGRATION.md) for a complete migration guide.

## Additional Resources

- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
- [DataKnobs LLM User Guide](./USER_GUIDE.md)

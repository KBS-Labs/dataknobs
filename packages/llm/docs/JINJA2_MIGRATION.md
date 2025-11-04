# Migrating to Jinja2 Templates

## Do You Need to Migrate?

**No!** Existing templates using `(( ))` syntax work unchanged. The default "mixed" mode ensures full backward compatibility.

**Consider migrating if you want:**
- Filters (uppercase, truncate, etc.)
- Advanced conditionals (if/elif/else with expressions)
- Loops ({% for %})
- Template includes and inheritance
- Macros for reusable components

## Migration Strategies

Choose the approach that fits your needs:

### Strategy 1: Gradual Enhancement (Recommended)

Keep using `(( ))` syntax while adding Jinja2 features gradually.

**Advantages:**
- No breaking changes
- Adopt features as needed
- Test incrementally
- Mixed mode supports both syntaxes

**Example:**

```yaml
# Before (still works!)
template: "Hello {{name}}((, age {{age}}))"

# After: Add filters gradually
template: "Hello {{name|upper}}((, age {{age}}))"

# After: Add Jinja2 conditionals
template: |
  Hello {{name|upper}}((, age {{age}}))
  {% if premium %}⭐ Premium member{% endif %}
```

### Strategy 2: Full Migration to Jinja2

Convert all `(( ))` syntax to pure Jinja2.

**Advantages:**
- Full Jinja2 power
- No syntax restrictions
- Slightly faster (one pass)
- Industry-standard syntax

**Example:**

```yaml
# Before
template: "Hello {{name}}((, age {{age}}))"
template_mode: mixed  # or omitted (default)

# After
template: "Hello {{name}}{% if age %}, age {{age}}{% endif %}"
template_mode: jinja2
```

## Conversion Guide

### Converting `(( ))` to `{% if %}`

The key difference:
- **`(( ))` blocks** are removed if **ALL** variables inside are missing
- **`{% if %}`** checks a **single condition**

#### Simple Conditional

```python
# Before
"Hello {{name}}((, from {{city}}))"

# After (Option 1: Keep (( )) in mixed mode)
"Hello {{name}}((, from {{city}}))"

# After (Option 2: Convert to Jinja2)
"Hello {{name}}{% if city %}, from {{city}}{% endif %}"
```

#### Multiple Variables in Conditional

```python
# Before - removed if BOTH age and city are missing
"{{name}}((, {{age}} years old from {{city}}))"

# After - check each separately
"{{name}}{% if age %}, {{age}} years old{% endif %}{% if city %} from {{city}}{% endif %}"

# Or check both
"{{name}}{% if age and city %}, {{age}} years old from {{city}}{% endif %}"
```

#### Nested Conditionals

```python
# Before
"{{name}}((, from {{city}}((, {{country}}))))"

# After
"{{name}}{% if city %}, from {{city}}{% if country %}, {{country}}{% endif %}{% endif %}"

# Cleaner with separate checks
"{{name}}{% if city %}, from {{city}}{% endif %}{% if country %}, {{country}}{% endif %}"
```

### Converting to Filters

```python
# Before: String manipulation in Python
params = {"name": user.name.upper()}
template = "Hello {{name}}"

# After: Use filters
template = "Hello {{name|upper}}"
params = {"name": user.name}  # Keep original case
```

### Converting Lists

```python
# Before: Pre-formatted string
items_str = ", ".join(items)
template = "Tags: {{items}}"

# After: Use join filter
template = "Tags: {{items|join(', ')}}"
params = {"items": items}  # Pass list directly
```

## Migration Examples

### Example 1: Code Review Prompt

**Before:**

```yaml
template: |
  Review this {{language}} code:

  {{code}}

  ((Focus areas: {{focus_areas}}))

  ((Additional guidelines: {{guidelines}}))
```

**After (Mixed Mode - Gradual):**

```yaml
template: |
  Review this {{language|upper}} code:

  {{code}}

  {% if focus_areas %}
  Focus areas: {{ focus_areas|join(', ') }}
  {% endif %}

  ((Additional guidelines: {{guidelines}}))

template_mode: mixed
```

**After (Pure Jinja2 - Full Migration):**

```yaml
template: |
  Review this {{language|upper}} code:

  {{code}}

  {% if focus_areas %}
  Focus areas: {{ focus_areas|join(', ') }}
  {% endif %}

  {% if guidelines %}
  Additional guidelines: {{guidelines}}
  {% endif %}

template_mode: jinja2
```

### Example 2: User Greeting

**Before:**

```yaml
template: "Welcome {{name}}((! You have {{message_count}} messages))"
```

**After (Mixed Mode):**

```yaml
template: "Welcome {{name|capitalize}}((! You have {{message_count}} messages))"
template_mode: mixed
```

**After (Pure Jinja2):**

```yaml
template: |
  Welcome {{name|capitalize}}{% if message_count %}! You have {{message_count}} message{% if message_count != 1 %}s{% endif %}{% endif %}
template_mode: jinja2
```

### Example 3: Data Analysis Prompt

**Before:**

```yaml
template: |
  Analyze the following data:

  {{data}}

  ((Context: {{context}}))
  ((Previous findings: {{previous_findings}}))
```

**After (Pure Jinja2 with loops):**

```yaml
template: |
  Analyze the following data:

  {% for item in data %}
  {{ loop.index }}. {{ item.name }}: {{ item.value }}
  {% endfor %}

  {% if context %}
  Context: {{context}}
  {% endif %}

  {% if previous_findings %}
  Previous findings:
  {% for finding in previous_findings %}
  - {{ finding }}
  {% endfor %}
  {% endif %}

template_mode: jinja2
```

## Automated Migration Tool

Use this helper function to convert simple cases:

```python
import re

def convert_to_jinja2(template: str) -> str:
    """Convert (( )) syntax to {% if %} syntax.

    Handles simple cases where conditionals contain a single variable.
    Manual review recommended for complex cases.
    """
    # Pattern: ((...{{var}}...))
    pattern = r'\(\(([^()]*\{\{(\w+)\}\}[^()]*)\)\)'

    def replace_conditional(match):
        content = match.group(1)
        var = match.group(2)
        return f"{{% if {var} %}}{content}{{% endif %}}"

    # Replace all (( )) blocks
    result = re.sub(pattern, replace_conditional, template)

    return result

# Usage
old_template = "Hello {{name}}((, age {{age}}))"
new_template = convert_to_jinja2(old_template)
# Output: "Hello {{name}}{% if age %}, age {{age}}{% endif %}"
```

**Note:** This handles simple cases only. For nested conditionals or multiple variables, manual conversion is recommended.

## Testing Your Migration

### 1. Unit Test Both Modes

```python
from dataknobs_llm.prompts.rendering import TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode

def test_template_compatibility():
    """Test that mixed and jinja2 modes produce same output."""
    renderer = TemplateRenderer()

    old_template = "Hello {{name}}((, age {{age}}))"
    new_template = "Hello {{name}}{% if age %}, age {{age}}{% endif %}"

    params = {"name": "Alice", "age": 30}

    # Mixed mode
    result1 = renderer.render(old_template, params, mode=TemplateMode.MIXED)

    # Jinja2 mode
    result2 = renderer.render(new_template, params, mode=TemplateMode.JINJA2)

    assert result1.content == result2.content
```

### 2. Test Edge Cases

```python
def test_edge_cases():
    """Test missing params, nested conditions, etc."""
    renderer = TemplateRenderer()

    # Test missing params
    result = renderer.render(
        "Hello {{name}}((, age {{age}}))",
        {"name": "Alice"}  # age missing
    )
    assert "age" not in result.content

    # Test nested
    result = renderer.render(
        "{{name}}((, from {{city}}((, {{country}}))))",
        {"name": "Alice", "city": "NYC"}  # country missing
    )
    assert "NYC" in result.content
    assert "country" not in result.content
```

### 3. Integration Tests

```python
async def test_with_prompt_builder():
    """Test migrated templates with prompt builder."""
    config = {
        "system": {
            "greet": {
                "template": "Hello {{name|upper}}!",
                "template_mode": "jinja2"
            }
        }
    }

    library = ConfigPromptLibrary(config)
    builder = AsyncPromptBuilder(library=library)

    result = await builder.render_system_prompt("greet", {"name": "alice"})
    assert result.content == "Hello ALICE!"
```

## Decision Matrix

| Feature | Keep `(( ))` | Convert to `{% if %}` |
|---------|-------------|---------------------|
| **Simple optional section** | ✅ Perfect fit | ✅ Works well |
| **Multiple variables that should all be present** | ✅ Perfect fit | ❌ Need AND logic |
| **Need filters** | ⚠️ Use mixed mode | ✅ Best option |
| **Need loops** | ❌ Not possible | ✅ Only option |
| **Need macros** | ❌ Not possible | ✅ Only option |
| **Complex boolean logic** | ❌ Not possible | ✅ Only option |
| **Template includes** | ❌ Not possible | ✅ Only option |
| **Existing templates** | ✅ No change needed | ⚠️ Migration effort |

## When to Use Each Syntax

### Use `(( ))` when:
- ✅ Section has multiple related variables
- ✅ Want "all or nothing" behavior
- ✅ Migrating from existing templates
- ✅ Keeping it simple

### Use `{% if %}` when:
- ✅ Checking single condition
- ✅ Need complex boolean logic (AND/OR)
- ✅ Comparing values (>, <, ==)
- ✅ Using other Jinja2 features (loops, macros)

### Example Comparison

```python
# Scenario: Optional address section with city and ZIP

# Option 1: (( )) - Shows address only if BOTH present
template = "{{name}}((, Address: {{city}}, {{zip}}))"

# Option 2: {% if %} - More control
template = """
{{name}}
{% if city and zip %}
Address: {{city}}, {{zip}}
{% elif city %}
City: {{city}}
{% elif zip %}
ZIP: {{zip}}
{% endif %}
"""
```

## Common Migration Pitfalls

### Pitfall 1: Assuming Same Behavior

```python
# ❌ These are NOT equivalent
"(({{name}}, {{age}}))"  # Removed if name OR age missing
"{% if name and age %}{{name}}, {{age}}{% endif %}"  # Removed if name OR age missing

# ✅ This is equivalent
"(({{name}}, {{age}}))"  # Removed if BOTH missing
"{% if name or age %}{% if name %}{{name}}{% endif %}{% if name and age %}, {% endif %}{% if age %}{{age}}{% endif %}{% endif %}"

# Better: Just keep (( )) for this use case!
```

### Pitfall 2: Forgetting to Change Mode

```yaml
# ❌ Wrong - using {% %} but mode is mixed
template: "{% if age > 18 %}Adult{% endif %}"
# Will fail with syntax error

# ✅ Correct - specify jinja2 mode
template: "{% if age > 18 %}Adult{% endif %}"
template_mode: jinja2
```

### Pitfall 3: Filters Inside `(( ))`

```python
# ❌ Not allowed in mixed mode
template = "((Hello {{name|upper}}))"

# ✅ Move filter outside
template = "{{name|upper}}((, Hello))"

# ✅ Or use jinja2 mode
template = "{% if name %}Hello {{name|upper}}{% endif %}"
template_mode = "jinja2"
```

## Rollback Plan

If you encounter issues, rolling back is simple:

### Option 1: Revert template_mode

```yaml
# Change from
template_mode: jinja2

# Back to
template_mode: mixed  # or remove (mixed is default)
```

### Option 2: Revert to Original Template

Keep a backup of original templates:

```bash
# Before migration
cp prompts/ prompts.backup/

# If issues occur
cp prompts.backup/ prompts/
```

### Option 3: Use Mode Override

```python
# Override mode at render time
result = renderer.render_prompt_template(
    prompt,
    params,
    mode_override=TemplateMode.MIXED
)
```

## Migration Checklist

- [ ] Read Jinja2 documentation
- [ ] Decide on migration strategy (gradual vs. full)
- [ ] Back up existing templates
- [ ] Convert templates (start with simple ones)
- [ ] Add `template_mode` to YAML files (if using jinja2 mode)
- [ ] Write tests for converted templates
- [ ] Test with missing parameters
- [ ] Test with nested conditions
- [ ] Update documentation
- [ ] Deploy and monitor

## Getting Help

If you encounter issues during migration:

1. Check the [JINJA2_INTEGRATION.md](./JINJA2_INTEGRATION.md) documentation
2. Review the [test examples](../tests/prompts/test_jinja2_integration.py)
3. Consult the [Jinja2 documentation](https://jinja.palletsprojects.com/)
4. File an issue on GitHub

## Summary

- **No migration required** - backward compatible by default
- **Gradual adoption** - add Jinja2 features incrementally
- **Full migration** - optional for maximum power
- **Mixed mode** - best of both worlds
- **Easy rollback** - change mode or revert templates
- **Well tested** - 35 new tests covering all scenarios

Happy templating! 🎉

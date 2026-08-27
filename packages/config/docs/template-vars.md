# Template Variable Substitution

The `substitute_template_vars` function provides recursive `{{var}}` placeholder substitution
in configuration data structures.

## Overview

```python
from dataknobs_config import substitute_template_vars

config = {
    "greeting": "Hello, {{name}}!",
    "settings": {
        "max_items": "{{limit}}",
        "features": ["{{feature_a}}", "{{feature_b}}"]
    }
}

variables = {
    "name": "World",
    "limit": 100,
    "feature_a": "search",
    "feature_b": "export"
}

result = substitute_template_vars(config, variables)
# {
#     "greeting": "Hello, World!",
#     "settings": {
#         "max_items": 100,  # Note: int preserved!
#         "features": ["search", "export"]
#     }
# }
```

## Features

### Recursive Substitution

Works with nested dicts, lists, and mixed structures:

```python
data = {
    "level1": {
        "level2": {
            "value": "{{deep_var}}"
        }
    },
    "items": [
        {"name": "{{item1}}"},
        {"name": "{{item2}}"}
    ]
}

result = substitute_template_vars(data, {
    "deep_var": "found",
    "item1": "first",
    "item2": "second"
})
```

### Type Preservation

When a placeholder is the **entire value** (not mixed with other text), the original
Python type is preserved:

```python
# Entire-value placeholders preserve type
result = substitute_template_vars(
    {"count": "{{count}}", "enabled": "{{flag}}"},
    {"count": 42, "flag": True}
)
# {"count": 42, "enabled": True}  # int and bool preserved

# Mixed content always returns string
result = substitute_template_vars(
    {"message": "You have {{count}} items"},
    {"count": 42}
)
# {"message": "You have 42 items"}  # string
```

Supported types for preservation:
- `int`, `float`
- `bool`
- `list`, `dict`
- `None`

### Whitespace Tolerance

Whitespace inside placeholders is handled gracefully:

```python
data = "{{ name }} and {{  value  }}"
result = substitute_template_vars(data, {"name": "a", "value": "b"})
# "a and b"
```

### Missing Variables

By default, missing variables are preserved:

```python
data = "Hello {{name}}, you have {{count}} items"
result = substitute_template_vars(data, {"name": "Alice"})
# "Hello Alice, you have {{count}} items"
```

Use `preserve_missing=False` to replace missing variables with empty strings:

```python
result = substitute_template_vars(data, {"name": "Alice"}, preserve_missing=False)
# "Hello Alice, you have  items"
```

### Disable Type Casting

Use `type_cast=False` to always return strings:

```python
result = substitute_template_vars(
    {"count": "{{count}}"},
    {"count": 42},
    type_cast=False
)
# {"count": "42"}  # string, not int
```

## API Reference

```python
def substitute_template_vars(
    data: Any,
    variables: dict[str, Any],
    *,
    preserve_missing: bool = True,
    type_cast: bool = True,
) -> Any:
    """Recursively substitute {{var}} placeholders in configuration data.

    Args:
        data: Configuration data (dict, list, string, or primitive)
        variables: Dict mapping variable names to values
        preserve_missing: If True, keep {{var}} for missing variables;
                         if False, replace with empty string
        type_cast: If True, preserve Python types for entire-value placeholders;
                  if False, always convert to string

    Returns:
        New data structure with placeholders substituted
        (original data is not modified)
    """
```

## Use Cases

### Configuration Templates

```python
# Base template with placeholders
base_config = {
    "llm": {
        "provider": "{{llm_provider}}",
        "model": "{{llm_model}}",
        "temperature": "{{temperature}}"
    },
    "storage": {
        "backend": "{{storage_backend}}"
    }
}

# Environment-specific values
dev_vars = {
    "llm_provider": "ollama",
    "llm_model": "gemma3:1b",
    "temperature": 0.7,
    "storage_backend": "memory"
}

prod_vars = {
    "llm_provider": "openai",
    "llm_model": "gpt-4",
    "temperature": 0.3,
    "storage_backend": "postgres"
}

dev_config = substitute_template_vars(base_config, dev_vars)
prod_config = substitute_template_vars(base_config, prod_vars)
```

### User-Facing Templates

```python
# Message templates
templates = {
    "welcome": "Welcome, {{user_name}}!",
    "error": "Error in {{component}}: {{message}}",
    "summary": "Processed {{count}} items in {{duration}}s"
}

# Runtime substitution
message = substitute_template_vars(
    templates["summary"],
    {"count": 150, "duration": 2.5}
)
# "Processed 150 items in 2.5s"
```

### Wizard Stage Prompts

```python
# Wizard stage with dynamic content
stage = {
    "prompt": "Configure {{bot_name}}'s settings",
    "suggestions": [
        "Use {{recommended_model}}",
        "Keep default settings"
    ]
}

rendered = substitute_template_vars(stage, {
    "bot_name": "CustomerBot",
    "recommended_model": "GPT-4"
})
```

## Notes

- The original data structure is never modified; a new structure is returned
- Non-string primitives (int, bool, None, etc.) pass through unchanged
- Empty dicts and lists pass through unchanged
- The function is safe to call multiple times for multi-pass substitution

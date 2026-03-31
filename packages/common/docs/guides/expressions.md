# Safe Expression Engine

The `dataknobs_common.expressions` module provides a shared safe expression evaluation engine for evaluating Python expression strings with restricted globals.

## Overview

Many parts of the dataknobs framework need to evaluate user-authored Python expressions safely: wizard transition conditions, field derivation expressions, and other config-driven logic. The expression engine centralizes this pattern with:

- **Restricted builtins** -- only safe type constructors, collection functions, and constants
- **AST validation** -- blocks dunder attribute access to prevent MRO traversal attacks
- **YAML literal aliases** -- `true`/`false`/`null`/`none` for config-authored expressions
- **Structured error reporting** -- `ExpressionResult` with success/failure and error details
- **Bool coercion** -- opt-in for condition evaluation use cases

## Quick Start

```python
from dataknobs_common.expressions import safe_eval, safe_eval_value

# Simple expression
result = safe_eval("1 + 2")
assert result.value == 3
assert result.success is True

# Convenience wrapper (returns just the value)
value = safe_eval_value("1 + 2")
assert value == 3

# Expression with scope variables
result = safe_eval("x * y", scope={"x": 3, "y": 4})
assert result.value == 12

# Condition evaluation with bool coercion
ok = safe_eval_value(
    "data.get('count', 0) > 5",
    scope={"data": {"count": 10}},
    coerce_bool=True,
)
assert ok is True

# Dict lookup with native type return
val = safe_eval_value(
    "{'easy': 30, 'hard': 120}.get(value, 60)",
    scope={"value": "hard"},
)
assert val == 120
```

## API

### `safe_eval(code, scope=None, *, coerce_bool=False, restrict_builtins=True, default=None)`

Evaluate a Python expression string safely.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code` | `str` | (required) | Python expression string. `return` is auto-prepended if missing. |
| `scope` | `dict[str, Any] \| None` | `None` | Variables available in the expression. Merged on top of `SAFE_BUILTINS` and `YAML_ALIASES`. |
| `coerce_bool` | `bool` | `False` | If True, coerce result to `bool` (for condition evaluation). |
| `restrict_builtins` | `bool` | `True` | If True, restrict `__builtins__` and validate AST. Set to False only for trusted code. |
| `default` | `Any` | `None` | Value to return on evaluation failure. |

**Returns:** `ExpressionResult` with `value`, `success`, and `error` fields.

### `safe_eval_value(code, scope=None, **kwargs)`

Convenience wrapper returning just the value. Same as `safe_eval(...).value`.

### `ExpressionResult`

Frozen dataclass with:

| Field | Type | Description |
|-------|------|-------------|
| `value` | `Any` | The evaluated result (native Python type). |
| `success` | `bool` | Whether evaluation succeeded. |
| `error` | `str \| None` | Exception message if evaluation failed. |

## Available in Expression Scope

### Safe Builtins (`SAFE_BUILTINS`)

When `restrict_builtins=True` (default), only these builtins are available:

| Category | Available |
|----------|-----------|
| Type constructors | `str`, `int`, `float`, `bool`, `list`, `dict`, `tuple`, `set` |
| Collection/numeric | `len`, `min`, `max`, `abs`, `round`, `sorted`, `isinstance`, `enumerate`, `range`, `zip` |
| Constants | `True`, `False`, `None` |

**Explicitly blocked:** `exec`, `eval`, `__import__`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `compile`, `breakpoint`.

### YAML Aliases (`YAML_ALIASES`)

Config-authored expressions can use YAML-style literals:

| Alias | Value |
|-------|-------|
| `true` | `True` |
| `false` | `False` |
| `null` | `None` |
| `none` | `None` |

Scope variables with the same name override these aliases.

## Security Model

The engine provides two layers of protection:

1. **Restricted builtins** -- `__builtins__` is set to `SAFE_BUILTINS`, blocking dangerous functions like `exec()`, `eval()`, `open()`, and `__import__()`.

2. **AST validation** -- before execution, the expression's AST is walked to reject any dunder attribute access (`__class__`, `__bases__`, `__subclasses__`, etc.) or dunder name usage (`__builtins__`, `__import__`). This prevents MRO traversal attacks that bypass builtins restrictions by navigating the Python object graph.

```python
# These are all blocked:
safe_eval("__import__('os')")          # NameError -- not in SAFE_BUILTINS
safe_eval("().__class__.__bases__")    # AST validation blocks __class__
safe_eval("open('/etc/passwd')")       # NameError -- not in SAFE_BUILTINS
safe_eval("exec('import os')")         # NameError -- not in SAFE_BUILTINS
```

For trusted code (e.g., developer-authored FSM functions), pass `restrict_builtins=False` to use full Python builtins and skip AST validation.

## Usage Patterns

### Wizard Transition Conditions

```python
from dataknobs_common.expressions import safe_eval_value

result = safe_eval_value(
    "data.get('name') and data.get('email')",
    scope={
        "data": wizard_data,
        "has": lambda key: wizard_data.get(key) is not None,
    },
    coerce_bool=True,
    default=False,
)
```

### Derivation Expressions

```python
from dataknobs_common.expressions import safe_eval

result = safe_eval(
    "10 if value == 'quiz_maker' else 5",
    scope={
        "value": source_value,
        "data": dict(wizard_data),
        "has": lambda key: wizard_data.get(key) is not None,
    },
)
if result.success:
    derived_value = result.value
```

### Error Handling

```python
result = safe_eval("1 / 0", default=-1)
assert result.success is False
assert result.value == -1
assert "ZeroDivisionError" in result.error
```

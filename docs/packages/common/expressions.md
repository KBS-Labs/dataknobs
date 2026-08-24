# Safe Expression Engine

The `dataknobs_common.expressions` module provides a shared safe expression evaluation engine for evaluating Python expression strings with restricted globals.

## Overview

Many parts of the dataknobs framework need to evaluate user-authored Python expressions safely: wizard transition conditions, field derivation expressions, and other config-driven logic. The expression engine centralizes this pattern with:

- **Restricted builtins** -- only safe type constructors, collection functions, and constants
- **AST validation** -- blocks dunder attribute access to prevent MRO traversal attacks
- **YAML literal aliases** -- `true`/`false`/`null`/`none` for config-authored expressions
- **Structured error reporting** -- `ExpressionResult` with success/failure and error details
- **Bool coercion** -- opt-in for condition evaluation use cases
- **Callable static pass** -- `safe_eval_validate` reports why an expression would be refused, without evaluating it

## Quick Start

```python
from dataknobs_common.expressions import (
    safe_eval,
    safe_eval_validate,
    safe_eval_value,
)

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
| `code` | `str` | (required) | Python expression string. `return` is prepended unless the expression already *is* a `return` statement. The test is on the `return` **token**, so an expression starting with a name such as `returned_value` is treated as an expression. |
| `scope` | `dict[str, Any] \| None` | `None` | Variables available in the expression. Merged on top of `SAFE_BUILTINS` and `YAML_ALIASES`. |
| `coerce_bool` | `bool` | `False` | If True, coerce result to `bool` (for condition evaluation). |
| `restrict_builtins` | `bool` | `True` | If True, restrict `__builtins__` and validate AST. Set to False only for trusted code. |
| `default` | `Any` | `None` | Value to return on evaluation failure. |

**Returns:** `ExpressionResult` with `value`, `success`, and `error` fields.

### `safe_eval_value(code, scope=None, **kwargs)`

Convenience wrapper returning just the value. Same as `safe_eval(...).value`.

### `safe_eval_validate(expression, *, restrict_builtins=True)`

Report why `safe_eval` would refuse an expression, **without evaluating it**. Returns the reason as a string, or `None` if `safe_eval` would proceed to evaluation.

Use it to pre-check a config-authored or generated expression while the author is still in the build loop, rather than discovering the refusal as a condition that silently never fires.

```python
reason = safe_eval_validate("data.get('a').__class__")
if reason is not None:
    raise ValueError(f"unusable condition: {reason}")
```

The contract is definitional -- it reports what `safe_eval`'s static pass would say, because it *is* that pass. When the rules below are tightened, both answers change together.

**The static pass rejects, in order:**

| # | Rejected | Example | `restrict_builtins=False` |
|---|----------|---------|---------------------------|
| 1 | Empty expression | `""`, `"   "` | also rejected |
| 2 | Multiline expression | a YAML `\|` or `>` block containing a blank line | also rejected |
| 3 | Syntax error | `"data.get('a' =="` | not checked |
| 4 | Dunder attribute access | `"().__class__"` | not checked |
| 5 | `.format()` / `.format_map()` call | `"'{0}'.format(x)"` | not checked |
| 6 | Dunder name | `"__builtins__"` | not checked |

Rules 3-6 are the AST pass, which `restrict_builtins=False` skips -- so pass the same value the matching `safe_eval` call will use.

**`None` is not a safety review, and not a promise the expression will succeed.** An expression reading a missing key is accepted here and raises at evaluation -- that is the distinction the function exists to draw: "not satisfied yet" is not "will not run".

```python
assert safe_eval_validate("data['x']") is None          # will run
assert safe_eval("data['x']", {"data": {}}).success is False  # and raises
```

In particular it does not check that the expression is free of side effects: `safe_eval` blocks assignment but permits mutation by method call, so `data.update(...)` is reported as acceptable and will take effect. Nor does it check that names resolve.

**It never raises.** `safe_eval` degrades a bad input to `success=False` rather than raising, and the pre-check mirrors that boundary -- so an unquoted YAML scalar (`condition: true`, which arrives as a `bool`) is reported as a reason like any other, and a loop over config-loaded conditions needs no guard around the pre-check.

### `ExpressionResult`

Frozen dataclass with:

| Field | Type | Description |
|-------|------|-------------|
| `value` | `Any` | The evaluated result (native Python type). |
| `success` | `bool` | Whether evaluation succeeded. |
| `error` | `str \| None` | Message if evaluation failed, or if the static pass refused the expression. This is `str(exception)` -- it carries no exception type. |

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

2. **A static pass before execution** -- the six rules tabulated under `safe_eval_validate` in the API section above. Rules 4-6 are what prevent MRO traversal attacks that bypass builtins restrictions by navigating the Python object graph: dunder attribute access (`__class__`, `__bases__`, `__subclasses__`), dunder names (`__builtins__`, `__import__`), and `.format()` / `.format_map()` calls, whose format-spec mini-language performs runtime attribute access via `{N.attr}` that the AST walk cannot see. f-strings are safe -- their substitutions go through normal AST nodes.

Both layers run on every `safe_eval` call, and `safe_eval_validate` runs the second one on its own.

```python
# These are all blocked:
safe_eval("__import__('os')")          # NameError -- not in SAFE_BUILTINS
safe_eval("().__class__.__bases__")    # AST validation blocks __class__
safe_eval("open('/etc/passwd')")       # NameError -- not in SAFE_BUILTINS
safe_eval("exec('import os')")         # NameError -- not in SAFE_BUILTINS
safe_eval("'{0}'.format(())")          # static pass blocks .format()
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
assert result.error == "division by zero"
```

`error` is `str(exception)` and carries no exception type, so it is a message for a human or a log, not something to branch on. To tell a *refusal* from a *runtime failure* -- the distinction that actually matters at a call site -- use `safe_eval_validate`:

```python
reason = safe_eval_validate(condition)
if reason is not None:
    # Will never run. The author needs to know now.
    logger.warning("unusable condition %r: %s", condition, reason)
```

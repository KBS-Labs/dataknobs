# Python Version Compatibility

## Overview

The DataKnobs project requires **Python 3.12 or higher**. The minimum version
is enforced via `requires-python = ">=3.12"` in all package `pyproject.toml`
files.

## Available Language Features

With Python 3.12 as the floor, the following features are available natively
without backport packages:

| Feature | Available Since | Notes |
|---------|----------------|-------|
| `X \| Y` type unions | 3.10 | No need for `Union[X, Y]` |
| `list[str]`, `dict[str, int]` | 3.9 | No need for `List`, `Dict` from `typing` |
| `typing.Self` | 3.11 | No need for `typing_extensions` |
| `ExceptionGroup` / `except*` | 3.11 | Structured concurrent error handling |
| `tomllib` | 3.11 | TOML parsing in stdlib |
| `asyncio.TaskGroup` | 3.11 | Structured concurrency |
| `typing.override` | 3.12 | Decorator for explicit overrides |
| `type` statement | 3.12 | Type alias syntax |

## Type Hint Style

Use modern type hint syntax throughout:

```python
# Preferred
def process_data(
    items: list[str],
    config: dict[str, Any] | None = None,
) -> tuple[list[Record], dict[str, int]]:
    ...

# Avoid — legacy typing imports
from typing import List, Dict, Optional, Tuple, Union
```

The `from __future__ import annotations` import is no longer required for
modern type syntax but remains useful for forward references and reducing
runtime evaluation overhead.

## Validation

```bash
# Type checking
uv run mypy packages/<package>/src

# Linting
uv run ruff check packages/<package>/src

# Auto-fix
uv run ruff check --fix packages/<package>/src
```

## Related Documentation

- [Quality Checks](./quality-checks.md)
- [Testing Guide](./testing-guide.md)

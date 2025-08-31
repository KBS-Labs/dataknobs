# Python 3.9+ Compatibility Guide

## Overview

The DataKnobs project maintains compatibility with Python 3.9 and later versions. While development typically uses Python 3.10+, the codebase must remain compatible with Python 3.9 to support users on older systems.

## Key Requirements

### 1. Future Annotations Import

**All Python files using type hints MUST include the future annotations import:**

```python
from __future__ import annotations
```

This import must be placed at the top of the file, after the module docstring but before any other imports.

### Why This Is Required

Python 3.10 introduced the pipe operator (`|`) for type unions, replacing the need for `Union` and `Optional` from the typing module. However, this syntax causes runtime errors in Python 3.9:

```python
# This causes TypeError in Python 3.9 without future annotations
def process(value: str | None = None) -> list[dict] | None:
    ...
```

The `from __future__ import annotations` import tells Python to treat all annotations as strings and defer their evaluation, making the new syntax work on Python 3.9.

## Type Hint Guidelines

### Preferred Style (with future annotations)

```python
from __future__ import annotations

def process_data(
    items: list[str],
    config: dict[str, Any] | None = None,
    validate: bool = True
) -> tuple[list[Record], dict[str, int]]:
    """Process data with optional configuration."""
    ...
```

### Avoid (verbose and outdated)

```python
from typing import List, Dict, Optional, Tuple, Any, Union

def process_data(
    items: List[str],
    config: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> Tuple[List[Record], Dict[str, int]]:
    """Process data with optional configuration."""
    ...
```

## Current Status (August 31, 2025)

- ✅ All 49 source files in `packages/data/src/dataknobs_data` have the future import
- ✅ All tests pass on Python 3.9.6
- ✅ Type checking works with `uv run mypy`

## Testing Compatibility

### Local Testing

```bash
# Check your development Python version
uv run python --version  # Should show 3.10+

# Test with system Python 3.9 (if available)
python3.9 -m pytest packages/data/tests/
```

### CI/CD Testing

The CI pipeline tests against multiple Python versions including 3.9. Any compatibility issues will be caught during the automated testing phase.

## Common Pitfalls and Solutions

### 1. Missing Future Import

**Problem:**
```python
# This will fail at runtime on Python 3.9
def get_value() -> str | None:
    return None
```

**Solution:**
```python
from __future__ import annotations

def get_value() -> str | None:
    return None
```

### 2. Runtime Type Checking

**Problem:**
```python
# This fails because isinstance doesn't work with stringified annotations
def process(value: str | int):
    if isinstance(value, str | int):  # TypeError!
        ...
```

**Solution:**
```python
from __future__ import annotations
from typing import Union

def process(value: str | int):
    # Use Union for runtime checks
    if isinstance(value, (str, int)):
        ...
```

### 3. Forward References

**Problem:**
```python
class Node:
    def __init__(self, parent: Node | None = None):  # NameError without quotes or future import
        self.parent = parent
```

**Solution:**
```python
from __future__ import annotations

class Node:
    def __init__(self, parent: Node | None = None):
        self.parent = parent
```

## Validation Tools

### Type Checking

Always use `uv run mypy` for type checking to ensure you're using the project's configured environment:

```bash
# Check entire package
uv run mypy packages/data/src/dataknobs_data

# Check specific file
uv run mypy packages/data/src/dataknobs_data/validation/constraints.py
```

### Linting

The project uses Ruff for linting, which is configured to enforce compatibility:

```bash
# Check for issues
uv run ruff check packages/data/src/dataknobs_data

# Auto-fix where possible
uv run ruff check --fix packages/data/src/dataknobs_data
```

## Adding New Files

When creating a new Python file in the project:

1. **Start with the template:**
```python
"""Module description."""

from __future__ import annotations

# Standard library imports
import ...

# Third-party imports
import ...

# Local imports
from ...
```

2. **Use modern type hints** with the pipe operator
3. **Run type checking** before committing
4. **Ensure tests pass** on Python 3.9

## Migration Guide for Existing Code

If you're updating older code that uses `Union` and `Optional`:

1. Add `from __future__ import annotations` at the top
2. Replace `Optional[X]` with `X | None`
3. Replace `Union[X, Y]` with `X | Y`
4. Replace `List[X]` with `list[X]`
5. Replace `Dict[K, V]` with `dict[K, V]`
6. Replace `Tuple[X, ...]` with `tuple[X, ...]`
7. Replace `Set[X]` with `set[X]`

## Benefits of This Approach

1. **Cleaner code** - More readable type hints
2. **Future-proof** - Ready for when Python 3.9 support is dropped
3. **Consistent style** - All code uses the same modern syntax
4. **Better IDE support** - Modern IDEs understand the new syntax better
5. **Reduced imports** - No need to import `Union`, `Optional`, etc.

## Related Documentation

- [Linting Configuration](./linting-configuration.md)
- [Quality Checks](./quality-checks.md)
- [Testing Guide](./testing-guide.md)
- [Contributing](./contributing.md)
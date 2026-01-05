---
globs:
  - "**/*.py"
---

# Dataknobs Code Validation

## Required Validation

All new and edited code in dataknobs MUST pass validation before being considered complete.

## Validation Command

Use `bin/validate.sh` for comprehensive code quality checks:

```bash
# Validate all packages
bin/validate.sh

# Validate specific package
bin/validate.sh common
bin/validate.sh data

# Validate with auto-fix
bin/validate.sh -f
bin/validate.sh -f common

# Quick validation (skip mypy)
bin/validate.sh -q

# Show error statistics
bin/validate.sh -s data
```

## What validate.sh Checks

1. **Python syntax** - Catches syntax errors before they cause runtime failures
2. **Ruff linting** - Style and common bug patterns
3. **Import validation** - Ensures packages can be imported
4. **MyPy type checking** - Type annotation compliance
5. **Print statements** - Ensures logging is used instead of print()
6. **TODO/FIXME comments** - Tracks technical debt

## Workflow for New Code

1. Write or modify code
2. Run `bin/validate.sh -f <package>` to auto-fix what's possible
3. Manually fix remaining issues
4. Re-run `bin/validate.sh <package>` to confirm all checks pass
5. Run tests with `bin/dk test <package>`

## Common Issues and Fixes

### Ruff Issues
```bash
# Auto-fix most ruff issues
bin/validate.sh -f <package>

# Or use ruff directly
uv run ruff check --fix packages/<package>/src
```

### MyPy Issues
- Add missing type annotations
- Fix type mismatches
- Use `cast()` for complex type narrowing
- Check `pyproject.toml` for mypy configuration

### Print Statements
Replace `print()` with proper logging:
```python
import logging
logger = logging.getLogger(__name__)

# Instead of: print(f"Processing {item}")
logger.info("Processing %s", item)

# Instead of: print(f"Error: {e}")
logger.error("Operation failed: %s", e)
```

## Integration with dk Tool

The `bin/dk` tool also runs validation as part of quality checks:
```bash
bin/dk check <package>  # Quick quality check including validation
bin/dk pr               # Full PR checks including validation
```

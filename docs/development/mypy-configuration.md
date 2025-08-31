# MyPy Configuration Strategy

## Overview

The DataKnobs project uses a pragmatic MyPy configuration that focuses on catching critical type errors while allowing gradual typing adoption. This document explains our type checking strategy and how to use it effectively.

## Configuration Files

We maintain two MyPy configurations:

1. **`mypy.ini`** - Focused configuration for catching real issues (141 errors)
   - **Note:** When present, MyPy uses this file by default
2. **`pyproject.toml`** - Comprehensive configuration for full type checking (587 errors when mypy.ini is absent)

## Using MyPy

### Quick Check (Focused on Critical Issues)
```bash
# MyPy automatically uses mypy.ini when present
uv run mypy packages/data/src/dataknobs_data

# Or explicitly specify the config file
uv run mypy --config-file mypy.ini packages/data/src/dataknobs_data
```

This configuration suppresses:
- Missing type annotations (gradual typing)
- Import errors for untyped libraries
- Dynamic attribute access
- Union type issues requiring extensive narrowing
- Method override signature mismatches
- Type assignment false positives

**Result: ~141 critical errors that need attention**

### Full Check (All Type Issues)
```bash
# Temporarily rename mypy.ini to use pyproject.toml settings
mv mypy.ini mypy.ini.bak
uv run mypy packages/data/src/dataknobs_data
mv mypy.ini.bak mypy.ini

# Or explicitly use pyproject.toml (if mypy.ini doesn't exist)
uv run mypy --config-file pyproject.toml packages/data/src/dataknobs_data
```

This shows more type issues including some less critical ones.

**Result: ~587 total errors including less critical issues**

## Error Categories

### Critical Errors (Shown in Focused Mode)
These are real issues that should be fixed:

| Error Code | Description | Count | Priority |
|------------|-------------|-------|----------|
| `arg-type` | Incorrect argument types | 79 | High |
| `return-value` | Wrong return types | 17 | High |
| `operator` | Unsupported operations | 12 | High |
| `call-arg` | Invalid function arguments | 9 | High |
| `misc` | Miscellaneous type errors | 9 | Medium |
| `func-returns-value` | Function missing return | 4 | High |
| `unreachable` | Unreachable code | 3 | Medium |

### Suppressed Errors (Hidden in Focused Mode)
These are less critical for gradual typing:

| Error Code | Description | Count | Reason for Suppression |
|------------|-------------|-------|------------------------|
| `attr-defined` | Dynamic attributes | 134 | Common with untyped libraries |
| `no-untyped-def` | Missing annotations | 82 | Gradual typing adoption |
| `union-attr` | Union attribute access | 76 | Requires extensive narrowing |
| `assignment` | Type mismatches | 62 | Many false positives |
| `import-untyped` | Untyped imports | 59 | Missing py.typed markers |
| `no-any-return` | Returning Any | 44 | Gradual typing |
| `override` | Override signatures | 25 | Base class constraints |
| `var-annotated` | Untyped variables | 15 | Less critical |

## Gradual Typing Strategy

### Phase 1: Fix Critical Errors (Current)
Focus on the 141 critical errors that represent real type safety issues:
- Incorrect argument types
- Wrong return types
- Unsupported operations

### Phase 2: Add py.typed Marker
Once critical errors are fixed:
1. Add `py.typed` marker file to package
2. This resolves 59 import-untyped errors
3. Makes package type-checkable by other projects

### Phase 3: Improve Type Coverage
Gradually add type annotations:
1. Start with public APIs
2. Add return type annotations
3. Type function arguments
4. Add variable annotations where helpful

### Phase 4: Type Narrowing
Address union type issues:
1. Add isinstance checks
2. Use TypeGuard functions
3. Implement proper type narrowing

## Practical Examples

### Fixing arg-type Errors
```python
# Before - arg-type error
def process(value: int) -> str:
    return str(value)

result = process("123")  # Error: arg-type

# After - fixed
result = process(int("123"))  # Or change function signature
```

### Fixing return-value Errors
```python
# Before - return-value error
def get_value() -> int:
    if condition:
        return 42
    return None  # Error: return-value

# After - fixed
def get_value() -> int | None:
    if condition:
        return 42
    return None
```

### Fixing operator Errors
```python
# Before - operator error
value: str | None = get_value()
result = value + "suffix"  # Error: operator

# After - fixed with type narrowing
value: str | None = get_value()
if value is not None:
    result = value + "suffix"
```

## IDE Integration

### VS Code
Add to `.vscode/settings.json`:
```json
{
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": [
    "--config-file=mypy.ini"
  ]
}
```

### PyCharm
1. Go to Settings → Tools → Python Integrated Tools
2. Set MyPy as the type checker
3. Add `--config-file=mypy.ini` to additional arguments

## CI/CD Integration

For CI pipelines, use the focused configuration:
```yaml
- name: Type Check
  run: |
    uv run mypy --config-file mypy.ini packages/data/src/dataknobs_data
    # Fail if critical errors exceed threshold
    ERROR_COUNT=$(uv run mypy --config-file mypy.ini packages/data/src/dataknobs_data 2>&1 | tail -1 | grep -oE '[0-9]+' | head -1)
    if [ "$ERROR_COUNT" -gt "150" ]; then
      echo "Critical type errors increased! Found $ERROR_COUNT errors (threshold: 150)"
      exit 1
    fi
```

## Adding Type Stubs

If you encounter import-untyped errors for third-party libraries:

1. Check if type stubs exist:
   ```bash
   pip search types-<package-name>
   ```

2. Add to dev dependencies:
   ```toml
   [tool.uv]
   dev-dependencies = [
       "types-requests>=2.31.0",
       "pandas-stubs>=2.0.0",
   ]
   ```

3. For packages without stubs, create minimal stubs:
   ```python
   # stubs/some_package.pyi
   def some_function(arg: Any) -> Any: ...
   ```

## Best Practices

1. **Run focused checks regularly** during development
2. **Fix critical errors immediately** - they often indicate real bugs
3. **Add type annotations gradually** - start with public APIs
4. **Use type: ignore sparingly** - only for false positives
5. **Document type: ignore usage** - explain why it's needed
6. **Keep error count trending down** - track progress over time

## Current Status

As of August 31, 2025:
- **Critical errors (focused with mypy.ini)**: 141
- **Total errors (comprehensive with pyproject.toml)**: 587
- **Python 3.9 compatibility**: ✅ Fully compatible
- **Type stub dependencies**: ✅ Installed

Note: MyPy automatically uses `mypy.ini` when present in the project root, which is why the default command shows focused results.

## Related Documentation

- [Python Compatibility Guide](./python-compatibility.md)
- [Linting Configuration](./linting-configuration.md)
- [Quality Checks](./quality-checks.md)
- [Contributing](./contributing.md)
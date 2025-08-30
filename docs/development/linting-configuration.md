# Linting Configuration Guidelines

## Overview
This document explains the rationale behind the linting and type checking configuration for the Dataknobs project. It serves as a reference for understanding which error types are considered important versus cosmetic, and why certain rules are ignored.

The actual configuration is in `pyproject.toml`. For specific errors that need to be fixed in each package, see the package-specific checklists (e.g., `packages/data/docs/linting-errors-checklist.md`).

## Error Categories and Decisions

### 1. Important Errors to Keep (NOT ignored)

#### Critical Bugs
- **F811**: Redefinition of unused variable - Can mask real bugs
- **F821**: Undefined name - Runtime errors
- **B904**: Raise without `from` inside except - Loses exception context
- **PLE0704**: Bare raise not in exception handler - Invalid Python

#### Code Quality
- **F401**: Unused imports (except in __init__.py)
- **F402**: Import shadowing
- **B007**: Unused loop variable not prefixed with underscore
- **B027**: Empty method without abstract decorator
- **PLR1714**: Consider merging multiple comparisons
- **PLR5501**: Consider using elif
- **RUF005**: Consider iterable unpacking
- **RUF006**: Store async task reference

#### Security
- **S3**: Various security issues

### 2. Ignored Error Categories

#### Whitespace/Formatting (Auto-fixable)
- **W291, W293**: Whitespace issues - Cosmetic, can be auto-fixed
- **E501**: Line too long - Already configured at 100 chars

#### Documentation
- **D105, D107**: Missing docstrings in special methods - Often self-explanatory
- **D200, D415, D417**: Docstring formatting - Minor style issues

#### Type Annotations
- **ANN204**: Missing return type for `__init__` - Always returns None
- **ANN001, ANN003**: Missing type annotations - Often obvious from context
- **ANN201, ANN202, ANN205**: Missing return types - Can be inferred

#### Import Location
- **PLC0415**: Import at top-level - Sometimes needed for:
  - Lazy loading for performance
  - Avoiding circular dependencies
  - Conditional imports

#### Code Simplification
- **SIM102**: Combine nested if - Sometimes clearer as nested
- **SIM108**: Use ternary operator - Can reduce readability

#### Complexity Metrics
- **PLR0911**: Too many returns - Already limited to 6
- **PLR0912**: Too many branches - Already limited to 12
- **PLR0915**: Too many statements - Already limited to 50

#### Unused Arguments
- **ARG001, ARG002, ARG004**: Unused arguments - Often required by:
  - Interface contracts
  - Callback signatures
  - Override methods

#### Type System Updates
- **UP035, UP038**: Modern type syntax - Gradual migration
- **UP028**: Yield from - Not always clearer

## Remaining Important Errors

After configuration, focus on these error types that indicate real issues:

### Critical Bugs (Must Fix)
- **F811**: Redefinition of unused variable - Can mask real bugs
- **F821**: Undefined name - Will cause runtime errors  
- **PLE0704**: Bare raise not in exception handler - Invalid Python
- **B904**: Raise without `from` in except - Loses exception context

### Code Quality Issues (Should Fix)
- **F841**: Local variable assigned but never used - Dead code
- **F401**: Unused imports (except in __init__.py) - Dead code
- **F402**: Import shadowing - Can cause confusion
- **B007**: Loop control variable not used - Potential logic error
- **B027**: Empty method without abstract decorator - Missing abstraction

### Type Checking & Advanced
- **TC003/TC004/TC001**: Type checking imports - May need runtime_checkable
- **NPY002**: NumPy legacy random - Should use new random API
- **PLW2901**: Loop variable overwritten - Potential bug

### Security
- **S3**: Various security issues - Always important to address

## MyPy Type Checking Configuration

### Common Error Categories
- **attr-defined**: Accessing undefined attributes, often on None
- **no-untyped-def**: Missing type annotations
- **union-attr**: Union type attribute access without guards
- **assignment**: Type incompatibilities in assignments
- **arg-type**: Wrong argument types passed to functions
- **no-any-return**: Returning Any from typed functions
- **unreachable**: Dead code - indicates logic errors
- **import-untyped**: Missing type stubs for third-party libraries
- **import-not-found**: Optional dependencies not installed

### Configuration Strategy
1. **Third-party libraries**: Add to ignore list when stubs unavailable
2. **Complex modules**: Relax strictness for gradual migration
3. **Optional dependencies**: Ignore imports for feature-specific libraries
4. **Legacy code**: Use per-module overrides to disable strict checking

### Priority for Fixes
1. **Unreachable code** - Always indicates logic errors
2. **None attribute access** - Add proper type guards
3. **Type mismatches** - Fix as you modify code
4. **Missing annotations** - Add gradually, prioritize public APIs

## Running Validation

```bash
# Run linting checks
uv run bin/validate.sh [package-name]

# Run type checking
uv run mypy packages/[package-name]/src

# Run both with detailed output
uv run bin/validate.sh [package-name] --verbose
```

## Package-Specific Checklists

Each package should maintain its own error checklist documenting specific issues to address:
- Location: `packages/[package-name]/docs/linting-errors-checklist.md`
- Format: Checkbox list organized by priority
- Updates: As errors are fixed, check them off and remove when complete

Example: `packages/data/docs/linting-errors-checklist.md`

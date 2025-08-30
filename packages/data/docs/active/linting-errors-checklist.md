# Data Package - Linting & Type Checking Errors Checklist

This document tracks specific errors that need to be addressed in the data package.

## Ruff Linting Errors (~40 remaining)

### Critical Bugs (Must Fix Immediately)

#### Redefinitions & Undefined Names
- [ ] **elasticsearch.py:869** - F811: Duplicate `close` method (also defined at line 135)
- [ ] **elasticsearch_mixins.py:186** - PLE0704: Bare `raise` statement not in exception handler
- [ ] **postgres_vector.py:166** - F841: Variable `operator` assigned but never used

#### Exception Handling
- [ ] **file.py:295** - B904: Raise ImportError without `from` for Parquet support
- [ ] **file.py:324** - B904: Raise ImportError without `from` for Parquet support

#### Unused Variables
- [ ] **file.py:277** - F401: `pyarrow.parquet` imported but unused
- [ ] **postgres.py:336** - F841: Variable `result` assigned but never used
- [ ] **postgres.py:380** - F841: Variable `result` assigned but never used  
- [ ] **postgres.py:418** - F841: Variable `result` assigned but never used
- [ ] **postgres.py:1232** - B007: Loop variable `i` not used (rename to `_i`)
- [ ] **s3.py:538** - F841: Variable `e` assigned but never used
- [ ] **s3.py:822** - F841: Variable `s3_response` assigned but never used
- [ ] **s3.py:942** - F821: Undefined name `aioboto3`
- [ ] **s3.py:1019** - F821: Undefined name `boto3`
- [ ] **sqlite.py:119** - F841: Variable `query` assigned but never used
- [ ] **sqlite.py:307** - F841: Variable `query` assigned but never used

### Code Quality Issues (Should Fix)

#### Type Checking Issues
- [ ] **pandas/converter.py:104** - TC003: Type checking import issue
- [ ] **pandas/converter.py:112** - TC003: Type checking import issue
- [ ] **pandas/converter.py:120** - TC003: Type checking import issue
- [ ] **pandas/converter.py:203** - TC003: Type checking import issue
- [ ] **pandas/type_mapper.py:178** - TC001: Type checking import issue
- [ ] **validation/schema.py:96** - TC004: Type checking import issue

#### NumPy Legacy
- [ ] **vector/benchmarks.py:12** - NPY002: Use `numpy.random.Generator`
- [ ] **vector/operations.py:5** - NPY002: Use `numpy.random.Generator`
- [ ] **vector/optimizations.py:11** - NPY002: Use `numpy.random.Generator`
- [ ] **vector/stores/common.py:6** - NPY002: Use `numpy.random.Generator`
- [ ] **vector/tracker.py:9** - NPY002: Use `numpy.random.Generator`

#### Other Issues
- [ ] **backends/file.py** - Multiple B027: Empty methods without abstract decorator
- [ ] **backends/postgres.py** - Multiple PLW2901: Loop variable overwritten
- [ ] **backends/s3.py** - Multiple F821: Undefined names (boto3, aioboto3)
- [ ] **vector/stores/__init__.py** - PYI056: Use `+=` instead of `.append()` for `__all__`

## MyPy Type Checking Errors (~400-500 remaining)

### Priority 1: Unreachable Code (31 instances)
These indicate logic errors and should be fixed first:

- [ ] **query.py:101** - Statement unreachable after return
- [ ] **query_logic.py:102** - Statement unreachable after condition
- [ ] **pooling/base.py:87, 118, 180** - Multiple unreachable statements
- [ ] **validation/constraints.py:362** - Unreachable after return
- [ ] **validation/coercer.py:78** - Unreachable after return
- [ ] **vector/benchmarks.py:33, 340, 342** - Multiple unreachable statements
- [ ] **vector/elasticsearch_utils.py:290** - Unreachable code
- [ ] **vector/migration.py:1003** - Unreachable statement

### Priority 2: Attribute Access on None (136 instances)
Most common pattern - needs type guards:

#### Vector Stores
- [ ] **faiss.py** - Multiple `None` attribute access (index operations)
- [ ] **chroma.py** - Multiple `None` attribute access (collection operations)
- [ ] Add proper initialization checks and type guards

#### Streaming Mixins
- [ ] **streaming.py** - Multiple missing attribute errors on mixins
- [ ] Define proper protocols or base classes

### Priority 3: Type Incompatibilities (76 assignment + 73 arg-type)

#### Common Patterns
- [ ] Fix `None` default arguments that should be optional
- [ ] Fix union type assignments without proper narrowing
- [ ] Fix list/dict type mismatches in query building

### Priority 4: Missing Type Annotations (82 instances)
- [ ] Add return type annotations to public methods
- [ ] Add type hints to function arguments
- [ ] Focus on public API first, internal methods later

## Tracking Progress

### Summary Statistics
- **Total Ruff Errors**: ~40 (down from ~1500)
- **Total MyPy Errors**: ~400-500 (down from 653 after config)

### By Category
| Category | Ruff | MyPy | Priority |
|----------|------|------|----------|
| Critical Bugs | 15 | 31 | Immediate |
| Type Safety | 6 | 285 | High |
| Code Quality | 15 | 82 | Medium |
| Style/Legacy | 4 | 100+ | Low |

## Next Steps

1. **Fix critical bugs first** - F811, PLE0704, B904, F821
2. **Add type guards** for None checks
3. **Fix unreachable code** - indicates logic errors
4. **Gradually add type annotations** as you modify code
5. **Consider refactoring** complex unions and mixins

## Notes

- Some errors may be in generated or third-party code - verify before fixing
- Vector store implementations have many errors due to optional dependencies
- Consider creating abstract base classes for mixins to improve type checking
- The validation module uses complex type unions that may need redesign
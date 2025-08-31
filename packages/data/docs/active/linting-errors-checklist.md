# Data Package - Linting & Type Checking Errors Checklist

This document tracks specific errors that need to be addressed in the data package.

## Ruff Linting Errors Status (Completed August 2025)

### Summary
**Reduced from ~1500 → ~40 → 5-10 remaining stylistic errors**

All functional Ruff errors have been resolved. Remaining errors are stylistic preferences that have been documented as acceptable in the project's linting configuration.

### Fixed in Final Cleanup ✅

#### Critical Functional Issues
- [x] **F811** - Duplicate method definitions - FIXED
- [x] **F821** - Undefined names - FIXED (added TYPE_CHECKING imports)
- [x] **F841** - Unused variables - FIXED (removed or used appropriately)
- [x] **F402** - Import shadowing - FIXED (renamed loop variables)
- [x] **B904** - Exception handling without `from` - FIXED
- [x] **PLE0704** - Bare raise statements - FIXED

#### Code Quality Improvements
- [x] **B027** - Empty base methods - FIXED (added `# noqa: B027` for intentional empty implementations)
- [x] **B007** - Unused loop variables - FIXED (removed unnecessary enumerate/items() calls)
- [x] **NPY002** - Legacy numpy random - FIXED (modernized to use np.random.default_rng())
- [x] **SIM101** - Multiple isinstance calls - FIXED (merged for clarity)
- [x] **PYI056** - __all__ modifications - FIXED (use += instead of append)
- [x] **PLW2901** - Loop variable overwrites - FIXED (and found a bug in python_vector_search.py!)
- [x] **TC001/003/004** - Type checking imports - FIXED (moved to TYPE_CHECKING blocks)

#### Bug Discovered and Fixed
- [x] **python_vector_search.py** - Record objects were being unnecessarily converted to dicts and back, potentially losing data

### Remaining Stylistic Errors (Ignored in pyproject.toml)

These have been evaluated and determined to be stylistic preferences that don't affect functionality:

- **SIM103** (2 instances) - Return negated condition directly - Sometimes clearer with if/else
- **SIM118** (5 instances) - Use `key in dict` instead of `key in dict.keys()` - Explicit can be clearer
- **PLW3301** (1 instance) - Nested max calls - More readable when nested
- **RUF006** (1 instance) - Store asyncio.create_task reference - Only needed if canceling
- **PLW0127** (1 instance) - Self-assignment - Documented with `# noqa` comment as intentional

## MyPy Type Checking Errors (~625 remaining)

### Priority 1: Unreachable Code (31 instances) 🔴
These indicate logic errors and should be fixed first:

#### Query Module
- [ ] **query.py:104** - Statement unreachable after return
- [ ] **query_logic.py:105** - Statement unreachable after condition

#### Pooling Module
- [ ] **pooling/base.py:87, 118, 180** - Multiple unreachable statements

#### Validation Module
- [ ] **validation/constraints.py:362** - Unreachable after return
- [ ] **validation/coercer.py:78** - Unreachable after return

#### Vector Module
- [ ] **vector/elasticsearch_utils.py:290** - Unreachable code

#### SQLite Mixins
- [ ] **backends/sqlite_mixins.py:122, 126, 128** - Unreachable statements

#### Pandas Module
- [ ] **pandas/type_mapper.py:213, 222, 339** - Unreachable statements
- [ ] **pandas/converter.py:39, 41** - Unreachable statements

#### Migration Module
- [ ] **migration/transformer.py:223, 230** - Unreachable statements

### Priority 2: Attribute Access on None (136 instances) 🟠
Most common pattern - needs type guards:

#### Vector Stores
- [ ] **faiss.py** - Multiple `None` attribute access (index operations)
- [ ] **chroma.py** - Multiple `None` attribute access (collection operations)
- [ ] Add proper initialization checks and type guards

#### Streaming Mixins
- [ ] **streaming.py:355, 359, 391, 392, 409, 410, 437, 441, 473, 474, 491, 492** - Missing attribute errors on mixins
- [ ] Define proper protocols or base classes

### Priority 3: Type Incompatibilities (149 instances) 🟡

#### Assignment Issues (76 instances)
- [ ] Fix `None` default arguments that should be optional
- [ ] Fix union type assignments without proper narrowing
- [ ] Fix list/dict type mismatches in query building

#### Argument Type Issues (73 instances)
- [ ] **query.py:624, 634, 703, 710, 716, 718** - LogicCondition vs FilterCondition mismatches
- [ ] **query_logic.py:123, 127, 335, 351** - Condition list type mismatches
- [ ] **validation/factory.py:141, 147, 155, 160, 163, 171, 177, 182** - Constraint type mismatches

### Priority 4: Missing Type Annotations (82 instances) 🟢
- [ ] Add return type annotations to public methods
- [ ] Add type hints to function arguments
- [ ] Focus on public API first, internal methods later

### Priority 5: Any Type Issues (Multiple instances) ⚪
- [ ] **validation/constraints.py:42, 44, 45** - Any(...) no longer supported, use cast(Any, ...)
- [ ] **no-any-return** - Functions returning Any when typed return expected

## Tracking Progress

### Summary Statistics
- **Ruff Errors**: ✅ 5-10 stylistic only (down from ~1500 originally)
- **MyPy Errors**: 625 (ready to address)

### Error Categories by Priority
| Priority | Category | Count | Status |
|----------|----------|-------|--------|
| 🔴 High | Unreachable code | 31 | To fix |
| 🟠 High | None attribute access | ~136 | To fix |
| 🟡 Medium | Type incompatibilities | 149 | To fix |
| 🟢 Low | Missing annotations | 82 | To fix |
| ⚪ Low | Any type issues | ~227 | To fix |

## Next Steps for MyPy Errors

1. **Fix unreachable code** (Priority 1)
   - Review logic flow in each module
   - Remove or fix dead code paths
   - May reveal actual bugs

2. **Add type guards for None checks** (Priority 2)
   - Add initialization checks for vector stores
   - Define protocols for streaming mixins
   - Use Optional types correctly

3. **Fix type incompatibilities** (Priority 3)
   - Review Query/ComplexQuery/LogicCondition/FilterCondition relationships
   - Fix constraint type hierarchies in validation module
   - Correct assignment type mismatches

4. **Add missing type annotations** (Priority 4)
   - Start with public API methods
   - Add return types to all functions
   - Use gradual typing for complex cases

5. **Resolve Any type issues** (Priority 5)
   - Replace deprecated Any(...) with cast(Any, ...)
   - Add specific types where possible
   - Document where Any is truly needed

## Notes

- Vector store implementations have many errors due to optional dependencies (faiss, chroma)
- Consider creating abstract base classes or protocols for mixins to improve type checking
- The validation module uses complex type unions that may need architectural review
- Some MyPy errors may be false positives due to dynamic typing patterns
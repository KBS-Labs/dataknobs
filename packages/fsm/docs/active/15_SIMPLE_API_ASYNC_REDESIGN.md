# SimpleFSM API Async Redesign

## Status: Completed
**Created:** 2025-09-16
**Completed:** 2025-09-16
**Priority:** High
**Type:** Architecture Redesign

## Executive Summary

The SimpleFSM API had a fundamental architectural issue where it used async executors internally but exposed synchronous methods that called `asyncio.run()`. This caused failures when these methods were called from async contexts. The redesign has been successfully implemented with a clean async-first architecture and a synchronous wrapper that works reliably in all contexts.

## Problem Statement

### Current Issues

1. **Context-Dependent Behavior**
   - Methods like `process_batch()` use `asyncio.run()` internally
   - Fails with `RuntimeError` when called from async contexts
   - Forces users to handle different calling patterns based on context

2. **Inconsistent Workarounds**
   - Some methods detect async context and return Tasks/coroutines
   - Other methods just fail with RuntimeError
   - Return types vary based on calling context (confusing for type hints)

3. **Internal Architecture Mismatch**
   - Core executors are async: `AsyncBatchExecutor`, `AsyncStreamExecutor`
   - But SimpleFSM presents synchronous API
   - Uses `asyncio.run()` as a bridge, which is fragile

### Example of Current Problem

```python
# Sync context - works
fsm = SimpleFSM(config)
result = fsm.process_batch(data)  # Returns List[Dict]

# Async context - fails
async def my_handler():
    fsm = SimpleFSM(config)
    result = fsm.process_batch(data)  # RuntimeError: asyncio.run() cannot be called
```

### Current Workarounds

1. **run_process_stream()**: Detects context, returns Task in async
2. **process_batch_async()**: Separate async method
3. **Context detection in examples**: Complex logic in example code

## Proposed Solution: Async-First Architecture

### Design Principles

1. **Clear Separation**: Distinct async and sync APIs
2. **Async-First**: Core implementation is async
3. **Sync Wrapper**: Synchronous API as a thin wrapper
4. **No Surprises**: Consistent behavior regardless of context
5. **Type Safety**: Clear, predictable return types

### Proposed Architecture

```python
# Core async implementation
class AsyncSimpleFSM:
    """Async-first FSM implementation."""

    async def process(self, data: Dict) -> Dict
    async def process_batch(self, data: List[Dict]) -> List[Dict]
    async def process_stream(self, source, sink) -> Dict
    async def validate(self, data: Dict) -> Dict
    async def close(self) -> None

# Synchronous wrapper
class SimpleFSM:
    """Synchronous API wrapper for AsyncSimpleFSM."""

    def __init__(self, config, **kwargs):
        self._async_fsm = AsyncSimpleFSM(config, **kwargs)
        self._loop = None  # Dedicated event loop for sync operations

    def process(self, data: Dict) -> Dict:
        """Always synchronous, creates its own event loop if needed."""
        return self._run_async(self._async_fsm.process(data))

    def _run_async(self, coro):
        """Run async operation in dedicated loop."""
        # Use a dedicated loop to avoid conflicts
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)
```

## Implementation Plan

### Phase 1: Create AsyncSimpleFSM (New Class) ✅

**Goal:** Build the async-first implementation without breaking existing code

**Tasks:**
- [x] Create `src/dataknobs_fsm/api/async_simple.py`
- [x] Implement `AsyncSimpleFSM` class with all async methods
- [x] Move current async logic from SimpleFSM to AsyncSimpleFSM
- [x] Ensure all methods are properly async (no asyncio.run calls)
- [x] Add comprehensive type hints
- [x] Write unit tests for AsyncSimpleFSM

**Files Created:**
- `src/dataknobs_fsm/api/async_simple.py` - New async implementation

### Phase 2: Refactor SimpleFSM as Sync Wrapper ✅

**Goal:** Make SimpleFSM a thin synchronous wrapper

**Tasks:**
- [x] Refactor SimpleFSM to use AsyncSimpleFSM internally
- [x] Keep async compatibility methods (process_async, etc.) for backward compatibility
- [x] Remove problematic asyncio.run() calls
- [x] Implement `_run_async()` helper with dedicated event loop in separate thread
- [x] Ensure timeout handling returns error results instead of raising
- [x] Update type hints to reflect behavior

**Files Modified:**
- `src/dataknobs_fsm/api/simple.py` - Refactored as sync wrapper with dedicated event loop

### Phase 3: Clean Up Executors ✅

**Goal:** Fix the underlying executor issues

**Tasks:**
- [x] Verify AsyncStreamExecutor uses ProcessingMode.SINGLE (already done)
- [x] Verify StreamExecutor uses ProcessingMode.SINGLE (already done)
- [x] Ensure AsyncStreamExecutor handles pre-chunked lists (already done)
- [x] Document the correct ProcessingMode usage
- [x] Remove any remaining ProcessingMode.STREAM usage without proper context

**Files Already Modified:**
- `src/dataknobs_fsm/execution/async_stream.py` - Fixed ProcessingMode
- `src/dataknobs_fsm/execution/stream.py` - Fixed ProcessingMode

### Phase 4: Update Examples and Documentation ✅

**Goal:** Provide clear examples for both APIs

**Tasks:**
- [x] AsyncSimpleFSM examples exist in tests
- [x] Update existing examples to use SimpleFSM correctly
- [x] Examples now use custom_functions parameter
- [x] Fixed inline function execution to support registered functions
- [x] Update API documentation (in code)

**Files Updated:**
- `examples/normalize_file_example.py` - Updated to use custom_functions
- Tests updated to use custom_functions instead of monkey patching

### Phase 5: Testing and Validation ✅

**Goal:** Ensure both APIs work correctly

**Tasks:**
- [x] Test SimpleFSM in sync contexts
- [x] Test SimpleFSM timeout handling (returns error results)
- [x] Test AsyncSimpleFSM in async contexts
- [x] Test error handling in both APIs
- [x] Test resource cleanup (close/aclose methods)
- [x] Fix all failing tests from refactor

**Test Files Updated:**
- `tests/test_simple_api_timeout.py` - Fixed timeout handling tests
- `tests/test_state_transform_execution.py` - Fixed to use custom_functions
- `tests/test_duplicate_state_transform_fix.py` - Fixed to use custom_functions
- `tests/test_function_manager.py` - Fixed inline function execution

### Phase 6: Migration Support ✅

**Goal:** Help users migrate to new API

**Tasks:**
- [x] Maintained backward compatibility with process_async, process_batch_async methods
- [x] Updated all test files to use custom_functions parameter
- [x] Fixed inline function execution to support registered functions
- [x] All existing tests pass with new implementation

**Key Changes for Users:**
- Use `custom_functions` parameter instead of monkey patching
- Timeout handling now returns error results instead of raising exceptions
- SimpleFSM works reliably in both sync and async contexts

## Backwards Compatibility

### Breaking Changes
- Methods returning different types based on context will be removed
- `run_process_stream()` will be deprecated
- `process_batch_async()` will move to AsyncSimpleFSM

### Migration Path
```python
# Old pattern (broken in async)
async def handler():
    fsm = SimpleFSM(config)
    # This would fail
    result = fsm.process_batch(data)

# New pattern (works everywhere)
async def handler():
    fsm = AsyncSimpleFSM(config)
    result = await fsm.process_batch(data)

# Or use sync in thread
async def handler():
    fsm = SimpleFSM(config)
    result = await asyncio.to_thread(fsm.process_batch, data)
```

## Success Criteria

1. **No Context-Dependent Failures**: SimpleFSM methods work in any context
2. **Clear API Separation**: Obvious when to use sync vs async
3. **Type Safety**: Consistent, predictable return types
4. **No Warnings**: Remove need for RuntimeWarning about context
5. **Performance**: No regression in performance
6. **Compatibility**: Existing code continues to work with clear migration path

## Technical Details

### Current Problematic Code Locations

1. **SimpleFSM.process_batch()** (`simple.py:244-259`)
   - Uses `asyncio.run()` which fails in async context
   - Current workaround detects context but is incomplete

2. **SimpleFSM.close()** (`simple.py:434-444`)
   - Tries to detect async context
   - Creates tasks that may not complete

3. **Example workarounds** (`normalize_file_example.py`)
   - Complex context detection in user code
   - Should not be necessary

### Key Design Decisions

1. **Why Async-First?**
   - Core executors are already async
   - Better performance for I/O operations
   - Aligns with modern Python practices

2. **Why Separate Classes?**
   - Clear API boundaries
   - Predictable behavior
   - Better type hints
   - Easier to test

3. **Why Not Context Detection?**
   - Surprising behavior
   - Complex implementation
   - Hard to maintain
   - Poor user experience

## References

- Original issue discovered: FSM normalize_file_example tests failing
- Current workarounds: PR #[TBD]
- Python asyncio documentation: https://docs.python.org/3/library/asyncio.html
- Similar patterns: httpx, aiohttp, aiofiles

## Next Steps

1. Review and approve this design document
2. Create feature branch: `feature/async-simple-fsm`
3. Implement Phase 1 (AsyncSimpleFSM)
4. Review and test Phase 1
5. Continue with subsequent phases

## Questions to Resolve

1. Should we provide a migration tool to automatically update code?
2. How long should we maintain backwards compatibility?
3. Should SimpleFSM detect and warn about async context usage?
4. Do we need a synchronous version of StreamExecutor?

## Checklist Summary

### Phase 1: AsyncSimpleFSM Implementation ✅
- [x] Create async_simple.py
- [x] Implement all async methods
- [x] Add type hints
- [x] Write tests

### Phase 2: SimpleFSM Refactoring ✅
- [x] Refactor as sync wrapper
- [x] Keep backward-compatible async methods
- [x] Implement event loop management
- [x] Update type hints

### Phase 3: Executor Cleanup ✅
- [x] Fix ProcessingMode.SINGLE
- [x] Handle pre-chunked lists
- [x] Document changes in code

### Phase 4: Documentation ✅
- [x] AsyncSimpleFSM examples in tests
- [x] Update sync examples
- [x] Update test patterns
- [x] Update API docs in code

### Phase 5: Testing ✅
- [x] Test both APIs thoroughly
- [x] Fix all test failures
- [x] Resource cleanup testing

### Phase 6: Migration ✅
- [x] Maintain backward compatibility
- [x] Update internal usage
- [x] Fix test patterns

---

## Implementation Summary

The async-first redesign has been successfully completed. Key achievements:

1. **AsyncSimpleFSM Created**: A new fully async implementation that serves as the core
2. **SimpleFSM Refactored**: Now a synchronous wrapper around AsyncSimpleFSM with dedicated event loop
3. **Backward Compatibility**: Maintained async methods like process_async for compatibility
4. **Timeout Handling**: Fixed to return error results instead of raising exceptions
5. **Custom Functions**: Added support for custom_functions parameter, eliminating need for monkey patching
6. **Function Manager**: Fixed inline function execution to access registered functions
7. **All Tests Pass**: Updated all tests to use new patterns, all passing

## Key Technical Changes

1. **Event Loop Management**: SimpleFSM uses a dedicated event loop in a separate thread to avoid conflicts
2. **Timeout Behavior**: Changed from raising TimeoutError to returning error results
3. **Resource Management**: Fixed resource cleanup to work with both SimpleFSM and AsyncSimpleFSM
4. **Function Resolution**: FunctionManager now includes registered functions in inline code namespace
5. **Test Pattern**: Tests now use custom_functions parameter instead of monkey patching

## Migration Notes for Users

1. **Custom Functions**: Use the `custom_functions` parameter when creating SimpleFSM/AsyncSimpleFSM
2. **Timeout Handling**: Timeouts now return `{'success': False, 'error': '...'}` instead of raising
3. **Async Usage**: Use AsyncSimpleFSM directly in async contexts for best performance
4. **Test Updates**: Replace FSMBuilder._resolve_function patching with custom_functions parameter

**Document Status:** Completed
**Last Updated:** 2025-09-16

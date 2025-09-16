# SimpleFSM API Async Redesign

## Status: Active
**Created:** 2025-09-16
**Priority:** High
**Type:** Architecture Redesign

## Executive Summary

The SimpleFSM API currently has a fundamental architectural issue where it uses async executors internally but exposes synchronous methods that call `asyncio.run()`. This causes failures when these methods are called from async contexts. The current workarounds (context detection, returning different types) are fragile and confusing. This document outlines a plan to properly redesign the API with clear separation between sync and async interfaces.

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

### Phase 1: Create AsyncSimpleFSM (New Class)

**Goal:** Build the async-first implementation without breaking existing code

**Tasks:**
- [ ] Create `src/dataknobs_fsm/api/async_simple.py`
- [ ] Implement `AsyncSimpleFSM` class with all async methods
- [ ] Move current async logic from SimpleFSM to AsyncSimpleFSM
- [ ] Ensure all methods are properly async (no asyncio.run calls)
- [ ] Add comprehensive type hints
- [ ] Write unit tests for AsyncSimpleFSM

**Files to Create:**
- `src/dataknobs_fsm/api/async_simple.py` - New async implementation

### Phase 2: Refactor SimpleFSM as Sync Wrapper

**Goal:** Make SimpleFSM a thin synchronous wrapper

**Tasks:**
- [ ] Refactor SimpleFSM to use AsyncSimpleFSM internally
- [ ] Remove all async methods from SimpleFSM
- [ ] Remove context detection code
- [ ] Implement `_run_async()` helper with dedicated event loop
- [ ] Ensure all methods are purely synchronous
- [ ] Update type hints to reflect sync-only behavior

**Files to Modify:**
- `src/dataknobs_fsm/api/simple.py` - Refactor as sync wrapper

### Phase 3: Clean Up Executors

**Goal:** Fix the underlying executor issues

**Tasks:**
- [ ] Verify AsyncStreamExecutor uses ProcessingMode.SINGLE (already done)
- [ ] Verify StreamExecutor uses ProcessingMode.SINGLE (already done)
- [ ] Ensure AsyncStreamExecutor handles pre-chunked lists (already done)
- [ ] Document the correct ProcessingMode usage
- [ ] Remove any remaining ProcessingMode.STREAM usage without proper context

**Files Already Modified:**
- `src/dataknobs_fsm/execution/async_stream.py` - Fixed ProcessingMode
- `src/dataknobs_fsm/execution/stream.py` - Fixed ProcessingMode

### Phase 4: Update Examples and Documentation

**Goal:** Provide clear examples for both APIs

**Tasks:**
- [ ] Create async examples using AsyncSimpleFSM
- [ ] Update existing examples to use SimpleFSM correctly
- [ ] Remove complex context detection from examples
- [ ] Update mkdocs documentation of examples and quickstart guides to use SimpleFSM correctly
- [ ] Update API documentation

**Files to Update:**
- `examples/normalize_file_example.py` - Simplify, remove context detection
- `examples/async_normalize_example.py` - New async example (to create)

### Phase 5: Testing and Validation

**Goal:** Ensure both APIs work correctly

**Tasks:**
- [ ] Test SimpleFSM in sync contexts
- [ ] Test SimpleFSM in async contexts (should work without errors)
- [ ] Test AsyncSimpleFSM in async contexts
- [ ] Test error handling in both APIs
- [ ] Performance comparison between APIs
- [ ] Test resource cleanup (close/aclose methods)

**Test Files to Update:**
- `tests/test_simple_api.py` - Add sync-specific tests
- `tests/test_async_simple_api.py` - New async API tests (to create)

### Phase 6: Migration Support

**Goal:** Help users migrate to new API

**Tasks:**
- [ ] Add deprecation warnings to old patterns
- [ ] Create migration guide documentation
- [ ] Provide compatibility shim if needed
- [ ] Update all internal usage of SimpleFSM
- [ ] Update integration examples

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

### Phase 1: AsyncSimpleFSM Implementation
- [ ] Create async_simple.py
- [ ] Implement all async methods
- [ ] Add type hints
- [ ] Write tests

### Phase 2: SimpleFSM Refactoring
- [ ] Refactor as sync wrapper
- [ ] Remove async methods
- [ ] Implement event loop management
- [ ] Update type hints

### Phase 3: Executor Cleanup
- [x] Fix ProcessingMode.SINGLE
- [x] Handle pre-chunked lists
- [ ] Document changes

### Phase 4: Documentation
- [ ] Create async examples
- [ ] Update sync examples
- [ ] Write migration guide
- [ ] Update API docs

### Phase 5: Testing
- [ ] Test both APIs thoroughly
- [ ] Performance testing
- [ ] Resource cleanup testing

### Phase 6: Migration
- [ ] Add deprecation warnings
- [ ] Create compatibility layer
- [ ] Update internal usage

---

**Document Status:** Ready for Review
**Last Updated:** 2025-09-16

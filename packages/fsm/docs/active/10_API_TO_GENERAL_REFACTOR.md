# API to General Level Refactor Plan

## Overview

During the process of fixing Simple and Advanced API tests, we discovered several cases where logic had been created or repeated at the API level that should be implemented at the general execution engine level instead. This document outlines the identified areas and provides a checklist for refactoring them.

## Context

The Simple API (`src/dataknobs_fsm/api/simple.py`) and Advanced API (`src/dataknobs_fsm/api/advanced.py`) contain duplicated and API-specific logic that should be centralized in the core execution components. This refactoring will:

- Eliminate code duplication
- Improve maintainability  
- Ensure consistent behavior across all FSM usage patterns
- Reduce API-specific workarounds

## Identified Issues

### 1. Resource Provider Creation Logic (Simple API)

**Current State:**
- Location: `SimpleFSM._create_resource_provider()` and `SimpleFSM._create_simple_resource_provider()` (lines 414-436)
- Issue: Simple API duplicates resource creation logic that should be centralized

**Target State:**
- Move to ResourceManager with factory method for creating providers from dict configs
- Remove duplicated logic from Simple API

**Code Reference:**
```python
# Current duplication in SimpleFSM
def _create_simple_resource_provider(self, name: str, config: Dict[str, Any]):
    class SimpleResourceProvider:
        # ... implementation
```

### 2. Context Initialization Patterns (Both APIs)

**Current State:**
- Simple API: `process()` and `process_async()` methods (lines 116-133, 182-198)
- Advanced API: `execution_context()` method (lines 220-296)
- Issue: Complex context initialization logic duplicated across APIs

**Target State:**
- Create centralized `ContextFactory` class
- Enhance `ExecutionContext` constructor to handle common patterns

**Code Reference:**
```python
# Duplicated pattern in both APIs
context = ExecutionContext(
    data_mode=ProcessingMode.SINGLE,
    resources={}
)
context.data = record.to_dict()
# ... state initialization logic
```

### 3. Result Formatting Logic (Simple API)

**Current State:**
- Location: `SimpleFSM.process()` and `SimpleFSM.process_async()` (lines 143-158, 213-227)
- Issue: Identical result formatting logic duplicated between sync and async methods

**Target State:**
- Create `ResultFormatter` utility class
- Add result formatting to execution engines

**Code Reference:**
```python
# Duplicated result formatting
return {
    'final_state': context.current_state,
    'data': context.data,
    'path': context.state_history + ([context.current_state] if context.current_state else []),
    'success': success,
    'error': None if success else str(result)
}
```

### 4. State History Path Construction (Simple API)

**Current State:**
- Location: Lines 147 and 216 in result formatting
- Issue: Path construction logic duplicated and should be handled by ExecutionContext

**Target State:**
- Add `get_complete_path()` method to ExecutionContext

**Code Reference:**
```python
# Current duplicated logic
'path': context.state_history + ([context.current_state] if context.current_state else [])
```

### 5. Batch Processing Result Transformation (Simple API)

**Current State:**
- Location: `SimpleFSM.process_batch()` lines 269-289
- Issue: Hardcoded result transformation with missing functionality (TODOs)

**Target State:**
- Batch executor should return standardized results
- Create common result transformer

**Code Reference:**
```python
# Current incomplete transformation
'final_state': 'output',  # TODO: Get actual final state from context
'path': [],  # TODO: Get path from context
```

### 6. Async Engine Creation Pattern (Simple API)

**Current State:**
- Location: `SimpleFSM.process_async()` lines 201-202
- Issue: Creates async engines on-demand vs Advanced API's instance variables

**Target State:**
- Standardize engine lifecycle management across APIs

**Code Reference:**
```python
# Inconsistent engine management
from ..execution.async_engine import AsyncExecutionEngine
async_engine = AsyncExecutionEngine(self._fsm)  # Created on-demand
```

### 7. State Definition and Instance Creation (Advanced API)

**Current State:**
- Location: `AdvancedFSM.execution_context()` lines 232-284
- Issue: Complex state resolution logic with fallback handling

**Target State:**
- Move to FSM core or ExecutionContext with factory method

**Code Reference:**
```python
# Complex state resolution logic
if not state_name:
    # Find start state from FSM
    if hasattr(self.fsm, 'get_start_state'):
        state_def = self.fsm.get_start_state()
        # ... more fallback logic
```

## Refactor Checklist

### Phase 1: Core Infrastructure

- [ ] **Create ContextFactory class**
  - [ ] Design interface for context creation patterns
  - [ ] Implement data conversion (dict → Record)
  - [ ] Implement initial state determination logic
  - [ ] Implement context creation with proper modes
  - [ ] Add state instance creation logic
  - [ ] Add unit tests for ContextFactory

- [ ] **Create ResultFormatter utility class**
  - [ ] Design standardized result format
  - [ ] Implement formatting for sync execution results
  - [ ] Implement formatting for async execution results
  - [ ] Implement formatting for batch execution results
  - [ ] Implement formatting for stream execution results
  - [ ] Add unit tests for ResultFormatter

- [ ] **Enhance ExecutionContext**
  - [ ] Add `get_complete_path()` method
  - [ ] Improve state management (dual tracking of string name and StateInstance)
  - [ ] Add better initialization methods
  - [ ] Update existing tests to use new methods

### Phase 2: ResourceManager Enhancement

- [ ] **Enhance ResourceManager with factory methods**
  - [ ] Add `create_provider_from_dict()` method
  - [ ] Add `create_simple_provider()` method
  - [ ] Move resource creation logic from APIs
  - [ ] Update APIs to use ResourceManager factory methods
  - [ ] Add unit tests for new ResourceManager methods

### Phase 3: Engine Lifecycle Standardization

- [ ] **Standardize engine management patterns**
  - [ ] Design consistent engine lifecycle across APIs
  - [ ] Update Simple API to maintain engine instances
  - [ ] Ensure consistent engine initialization
  - [ ] Update Advanced API if needed
  - [ ] Add tests for engine lifecycle

### Phase 4: FSM Core Enhancements

- [ ] **Improve state resolution in FSM core**
  - [ ] Add `find_state_definition()` method to FSM
  - [ ] Add `create_state_instance()` method to FSM
  - [ ] Move complex state finding logic from Advanced API
  - [ ] Add fallback handling for missing states
  - [ ] Add unit tests for state resolution

### Phase 5: API Refactoring

- [ ] **Refactor Simple API**
  - [ ] Replace resource creation logic with ResourceManager calls
  - [ ] Replace context initialization with ContextFactory
  - [ ] Replace result formatting with ResultFormatter
  - [ ] Replace path construction with ExecutionContext method
  - [ ] Update batch processing to use standardized results
  - [ ] Update engine creation pattern
  - [ ] Verify all tests still pass

- [ ] **Refactor Advanced API**
  - [ ] Replace context initialization with ContextFactory
  - [ ] Replace state resolution with FSM core methods
  - [ ] Update result handling if needed
  - [ ] Verify all tests still pass

### Phase 6: Testing and Validation

- [ ] **Run comprehensive tests**
  - [ ] All Simple API tests pass
  - [ ] All Advanced API tests pass
  - [ ] All execution engine tests pass
  - [ ] All integration tests pass
  - [ ] Performance benchmarks maintain or improve

- [ ] **Code quality checks**
  - [ ] No code duplication between APIs
  - [ ] Consistent patterns across codebase
  - [ ] Proper error handling maintained
  - [ ] Documentation updated for new patterns

## Success Criteria

1. **Zero Code Duplication**: No duplicated logic between Simple and Advanced APIs
2. **Consistent Patterns**: All APIs use the same underlying infrastructure
3. **Test Coverage**: All existing tests continue to pass
4. **Maintainability**: New features can be added at the general level and automatically benefit all APIs
5. **Performance**: No performance regressions from refactoring

## Notes

- This refactoring should be done incrementally to avoid breaking existing functionality
- Each phase should include thorough testing before moving to the next
- Consider backwards compatibility for any public API changes
- Document any breaking changes clearly

## Files to be Modified

### New Files to Create:
- `src/dataknobs_fsm/core/context_factory.py`
- `src/dataknobs_fsm/core/result_formatter.py`

### Existing Files to Modify:
- `src/dataknobs_fsm/api/simple.py`
- `src/dataknobs_fsm/api/advanced.py`
- `src/dataknobs_fsm/execution/context.py`
- `src/dataknobs_fsm/resources/manager.py`
- `src/dataknobs_fsm/core/fsm.py`

### Test Files to Update:
- All existing API test files
- Add new unit tests for new components
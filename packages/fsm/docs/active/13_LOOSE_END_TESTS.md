# FSM Package - Loose End Implementation Tests Checklist

This document tracks the test coverage needed for the completed loose end implementations documented in `11_LOOSE_ENDS.md`. Each completed feature needs comprehensive testing to ensure it works correctly.

**Created**: December 2024  
**Status**: 1 of 10 test suites completed, 9 remaining  
**Overall Test Coverage**: 59% (Target: 80%+)

## Test Implementation Priority

### 🔴 Critical Priority (Coverage < 40%)
These have very low coverage and are critical functionality.

### 🟡 High Priority (Coverage 40-70%)
These need significant improvement to ensure reliability.

### 🟢 Medium Priority (Coverage > 70%)
These have reasonable coverage but need specific edge cases tested.

---

## ✅ Completed Test Suites

### ✅ 1. Execution Common Module Tests
**File**: `tests/test_execution_common.py`  
**Coverage**: 97% (improved from 33%)  
**Completed**: December 2024

- ✅ NetworkSelector class (11 test methods)
  - ✅ Network selection from stack
  - ✅ Network selection from metadata
  - ✅ Main network selection (core and wrapper FSM)
  - ✅ Intelligent selection by data mode
  - ✅ Network type preferences
  - ✅ Fallback mechanisms

- ✅ ArcScorer class (9 test methods)
  - ✅ Basic priority scoring
  - ✅ Resource availability scoring
  - ✅ Historical success rate integration
  - ✅ Load balancing penalties
  - ✅ Deterministic arc preferences
  - ✅ Combined factor scoring

- ✅ TransitionSelector class (10 test methods)
  - ✅ Strategy-based selection (depth-first, breadth-first)
  - ✅ Resource-optimized selection
  - ✅ Stream-optimized selection
  - ✅ Priority scoring mode
  - ✅ Round-robin for tied scores
  - ✅ Hybrid mode operation

- ✅ Metrics extraction (6 test methods)
  - ✅ Arc usage metrics
  - ✅ Network warnings
  - ✅ Batch information
  - ✅ Timing metrics

---

## 🔴 Critical Priority Test Suites (Need Immediate Attention)

### ❌ 2. LLM Provider Implementations Tests
**Target File**: `tests/test_llm_providers_integration.py`  
**Current Coverage**: 21% ⚠️  
**Lines Needing Coverage**: 849-884 (SyncProviderAdapter), 512-519, 532-539

**Test Requirements**:
- [ ] Test `SyncProviderAdapter` class initialization
- [ ] Test async-to-sync wrapping with event loop handling
- [ ] Test OpenAI completion using provider system
- [ ] Test Anthropic completion using provider system  
- [ ] Test OpenAI embeddings using provider system
- [ ] Test error handling and fallback mechanisms
- [ ] Test thread safety of sync adapter
- [ ] Test resource cleanup after provider use

**Implementation Notes**:
```python
# Key areas to test:
- SyncProviderAdapter._run_async() method
- Provider registry integration
- Error handling for unsupported features (Anthropic embeddings, HuggingFace function calling)
```

### ❌ 3. Sync I/O Providers Tests
**Target File**: `tests/test_io_sync_providers.py`  
**Current Coverage**: 31% ⚠️  
**Lines Needing Coverage**: 268-454

**Test Requirements**:
- [ ] Test `SyncDatabaseProvider` initialization with sqlite3
- [ ] Test sync database CRUD operations (read, write, update, delete)
- [ ] Test sync database batch operations
- [ ] Test sync database stream operations
- [ ] Test `SyncHTTPProvider` initialization with requests
- [ ] Test sync HTTP GET/POST/PUT/DELETE operations
- [ ] Test sync HTTP error handling (timeouts, connection errors)
- [ ] Test sync HTTP streaming responses
- [ ] Test fallback from async to sync providers

**Implementation Notes**:
```python
# Test both providers with real backends:
- Use in-memory sqlite for database tests
- Use httpbin.org or mock server for HTTP tests
- Verify interface compatibility with async versions
```

---

## 🟡 High Priority Test Suites

### ❌ 4. Streaming Infrastructure Tests
**Target File**: `tests/test_streaming_core_extensions.py`  
**Current Coverage**: 52%  
**Lines Needing Coverage**: 466-542, 563-613

**Test Requirements**:
- [ ] Test `BasicStreamProcessor` initialization and configuration
- [ ] Test stream processing workflows with transforms
- [ ] Test `MemoryStreamSource` data feeding
- [ ] Test `MemoryStreamSink` data collection
- [ ] Test buffer management and overflow handling
- [ ] Test state transitions during streaming
- [ ] Test error handling in stream pipeline
- [ ] Test stream lifecycle (start, process, stop, cleanup)

**Implementation Notes**:
```python
# Focus on:
- End-to-end streaming workflows
- Buffer edge cases (overflow, underflow)
- Concurrent stream operations
```

### ❌ 5. Arc Resource Handling Tests
**Target File**: `tests/test_arc_resource_management.py`  
**Current Coverage**: 69%  
**Lines Needing Coverage**: 397-415 (_allocate_resources, _release_resources)

**Test Requirements**:
- [ ] Test resource acquisition through ResourceManager
- [ ] Test resource owner tracking
- [ ] Test resource cleanup on arc completion
- [ ] Test resource cleanup on arc failure
- [ ] Test resource contention handling
- [ ] Test timeout during resource acquisition
- [ ] Test resource pool integration
- [ ] Test cascading resource failures

**Implementation Notes**:
```python
# Test scenarios:
- Multiple arcs competing for same resource
- Resource leak detection
- Proper cleanup in exception paths
```

---

## 🟢 Medium Priority Test Suites

### ❌ 6. ExecutionHistory Deserialization Tests
**Target File**: `tests/test_storage_deserialization.py`  
**Current Coverage**: 68%  
**Lines Needing Coverage**: 237-305

**Test Requirements**:
- [ ] Test `ExecutionStep.from_dict()` with all field types
- [ ] Test `ExecutionHistory.from_dict()` with nested structures
- [ ] Test tree reconstruction from serialized data
- [ ] Test round-trip serialization/deserialization
- [ ] Test handling of missing/invalid fields
- [ ] Test version compatibility
- [ ] Test large history deserialization performance

**Implementation Notes**:
```python
# Verify:
- All data types are preserved
- Tree structure is maintained
- No data loss in round-trip
```

### ❌ 7. Builder Execution Implementation Tests
**Target File**: `tests/test_builder_execution.py`  
**Current Coverage**: 76%  
**Lines Needing Coverage**: 866-891

**Test Requirements**:
- [ ] Test `execute()` method with simple FSM
- [ ] Test execution with complex multi-state FSM
- [ ] Test execution with resources
- [ ] Test execution error handling
- [ ] Test result formatting
- [ ] Test execution with different data modes
- [ ] Test execution with breakpoints
- [ ] Test execution context creation

**Implementation Notes**:
```python
# Test the full execution pipeline:
- Builder -> Engine -> Context -> Result
- Verify DRY principle adherence
```

### ❌ 8. Simple API Timeout Support Tests
**Target File**: `tests/test_simple_api_timeout.py`  
**Current Coverage**: 74%  
**Lines Needing Coverage**: 428-448, 499-512

**Test Requirements**:
- [ ] Test timeout in `process()` function
- [ ] Test timeout in `process_file()` function
- [ ] Test timeout in `batch_process()` function
- [ ] Test ThreadPoolExecutor timeout handling
- [ ] Test async timeout with asyncio.wait_for()
- [ ] Test timeout error messages
- [ ] Test cleanup after timeout
- [ ] Test partial results on timeout

**Implementation Notes**:
```python
# Use short timeouts with slow operations:
- Mock slow transforms
- Verify proper exception types
- Check resource cleanup
```

### ❌ 9. Specific FSM Exception Types Tests
**Target File**: `tests/test_fsm_exceptions.py`  
**Current Coverage**: 78%  
**Lines Needing Coverage**: Exception usage in patterns

**Test Requirements**:
- [ ] Test `CircuitBreakerError` with wait_time
- [ ] Test `ETLError` for ETL operations
- [ ] Test `BulkheadTimeoutError` for queue timeouts
- [ ] Test exception propagation in patterns/etl.py
- [ ] Test exception handling in patterns/api_orchestration.py
- [ ] Test exception recovery in patterns/error_recovery.py
- [ ] Test exception serialization
- [ ] Test exception context preservation

**Implementation Notes**:
```python
# Verify exceptions are raised in correct contexts:
- ETL failures -> ETLError
- Circuit breaker trips -> CircuitBreakerError
- Resource timeouts -> BulkheadTimeoutError
```

### ❌ 10. Database Storage Factory Tests
**Target File**: `tests/test_database_storage_factory.py`  
**Current Coverage**: 81%  
**Lines Needing Coverage**: 166-214

**Test Requirements**:
- [ ] Test `AsyncDatabaseFactory` with memory backend
- [ ] Test factory with sqlite backend
- [ ] Test factory with invalid backend
- [ ] Test database connection handling
- [ ] Test cleanup with close() method
- [ ] Test concurrent database operations
- [ ] Test transaction support
- [ ] Test error recovery

**Implementation Notes**:
```python
# Test all supported backends:
- memory (no connection needed)
- sqlite (connection required)
- Verify proper cleanup in all cases
```

---

## Testing Strategy

### Phase 1: Critical Tests (Week 1)
1. LLM Provider Tests - Blocks AI features
2. Sync I/O Provider Tests - Blocks sync operations

### Phase 2: High Priority Tests (Week 2)
3. Streaming Infrastructure Tests
4. Arc Resource Management Tests

### Phase 3: Medium Priority Tests (Week 3)
5. ExecutionHistory Deserialization Tests
6. Builder Execution Tests
7. Simple API Timeout Tests
8. FSM Exception Tests
9. Database Storage Factory Tests

### Test Implementation Guidelines

1. **Use Real Implementations**: Follow DRY principle, avoid excessive mocking
2. **Test Edge Cases**: Focus on error paths and boundary conditions
3. **Performance Tests**: Include tests for large datasets where relevant
4. **Integration Tests**: Test component interactions, not just units
5. **Documentation**: Add docstrings explaining what each test validates

### Success Metrics

- [ ] Overall test coverage > 80%
- [ ] All critical components > 90% coverage
- [ ] Zero failing tests in CI/CD pipeline
- [ ] All edge cases documented and tested
- [ ] Performance benchmarks established

### Resources Required

- Mock servers for HTTP testing (httpbin.org or local mock)
- Test databases (in-memory sqlite)
- Sample data files for streaming tests
- Mock LLM responses for provider tests

---

## Notes

- The `execution/common.py` test suite serves as a template for comprehensive testing
- Focus on integration tests that verify the loose ends work together
- Prioritize tests that verify the DRY principle improvements
- Each test file should follow the pattern established in `test_execution_common.py`
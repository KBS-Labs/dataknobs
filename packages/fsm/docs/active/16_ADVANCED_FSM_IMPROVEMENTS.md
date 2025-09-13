# Advanced FSM Improvements

## Overview
This document tracks the improvements needed for the AdvancedFSM API to fully support interactive debugging and monitoring scenarios as originally envisioned in the `advanced_debugging.py` example.

## Current State (2025-01-12)

### Working Features ✅
The following AdvancedFSM features are fully functional:

1. **Breakpoint Management**
   - `add_breakpoint(state_name)` - Add breakpoint at a state
   - `remove_breakpoint(state_name)` - Remove a breakpoint
   - `clear_breakpoints()` - Clear all breakpoints
   - `breakpoints` property - Access current breakpoints

2. **State Inspection**
   - `inspect_state(state_name)` - Get detailed state information
   - `get_available_transitions(state_name)` - List transitions from a state
   - `visualize_fsm()` - Generate DOT graph visualization

3. **Execution Hooks**
   - `set_hooks(ExecutionHook)` - Set enter/exit/error hooks
   - `hooks` property - Access current hooks

4. **History Tracking**
   - `enable_history(max_depth)` - Enable execution history
   - `disable_history()` - Disable history tracking
   - `history_enabled` property - Check if enabled
   - `max_history_depth` property - Get max depth
   - `execution_history` property - Access history steps

5. **Execution Modes**
   - `trace_execution(data)` - Execute with full tracing
   - `profile_execution(data)` - Execute with performance profiling
   - `step(context)` - Execute single step (async)
   - `run_until_breakpoint(context)` - Run to breakpoint (async)

### Implementation Details

#### Properties Added
```python
@property
def breakpoints(self) -> set:
    """Get the current breakpoints."""
    return self._breakpoints.copy()

@property
def hooks(self) -> ExecutionHook:
    """Get the current execution hooks."""
    return self._hooks

@property
def history_enabled(self) -> bool:
    """Check if history tracking is enabled."""
    return self._history is not None

@property
def max_history_depth(self) -> int:
    """Get the maximum history depth."""
    return self._history.max_depth if self._history else 0

@property
def execution_history(self) -> list:
    """Get the execution history steps."""
    return self._history.steps if self._history else []
```

#### Core State Improvements
Added to `StateDefinition` class:
```python
@property
def arcs(self) -> List["ArcDefinition"]:
    """Get the outgoing arcs from this state."""
    return self.outgoing_arcs
```

Added to `ExecutionHistory` class:
```python
@property
def steps(self) -> List[ExecutionStep]:
    """Get all execution steps from the history tree."""
    # Collects all steps from tree structure
```

## Missing Features 🚧

### 1. Synchronous Interface
**Problem**: All execution methods are async, making interactive debugging cumbersome.

**Needed**:
```python
def execute_step_sync(self, context) -> Dict[str, Any]:
    """Synchronous wrapper for step execution."""
    
def run_until_breakpoint_sync(self, context) -> StateInstance:
    """Synchronous wrapper for breakpoint execution."""
```

**Solution Options**:
- Add sync wrappers using `asyncio.run()`
- Provide a `SyncAdvancedFSM` wrapper class
- Add `sync=True` parameter to methods

### 2. Simple Context Creation
**Problem**: Creating execution context requires async context manager.

**Current**:
```python
async with fsm.execution_context(data) as context:
    # Use context
```

**Needed**:
```python
context = fsm.create_context(data)
# Use context directly
```

**Solution**:
```python
def create_context(self, data: Dict[str, Any]) -> ExecutionContext:
    """Create an execution context for manual control."""
    context = ContextFactory.create_context(
        self.fsm,
        data,
        data_mode=self.data_mode
    )
    return context
```

### 3. Context Helper Methods
**Problem**: ExecutionContext lacks convenience methods for debugging.

**Needed**:
```python
class ExecutionContext:
    def is_complete(self) -> bool:
        """Check if FSM has reached an end state."""
        
    def get_current_state(self) -> str:
        """Get the name of the current state."""
        
    def get_data_snapshot(self) -> Dict[str, Any]:
        """Get current data snapshot."""
```

### 4. FSMDebugger Export
**Problem**: Debugger interface exists but isn't exported/documented.

**Current**: Internal `debug()` method returns debugger-like interface.

**Needed**:
```python
class FSMDebugger:
    """Interactive debugger for FSM execution."""
    
    def __init__(self, fsm: AdvancedFSM):
        self.fsm = fsm
        self.context = None
        
    def start(self, data: Dict[str, Any]):
        """Start debugging session."""
        
    def step(self) -> StepResult:
        """Execute one step."""
        
    def continue_to_breakpoint(self) -> StateInstance:
        """Continue execution to next breakpoint."""
        
    def inspect_current_state(self) -> Dict[str, Any]:
        """Inspect current state details."""
```

### 5. Structured Step Results
**Problem**: Step execution doesn't return structured information.

**Needed**:
```python
@dataclass
class StepResult:
    """Result from a single step execution."""
    from_state: str
    to_state: str
    transition: str
    data_before: Dict[str, Any]
    data_after: Dict[str, Any]
    duration: float
    success: bool
    error: Optional[str]
```

## Implementation Plan

### Phase 1: Context Improvements ⏳
1. Add `create_context()` method for simple context creation
2. Add helper methods to ExecutionContext:
   - `is_complete()`
   - `get_current_state()`
   - `get_data_snapshot()`

### Phase 2: Synchronous Support 📋
1. Create `SyncAdvancedFSM` wrapper class
2. Implement sync wrappers for all async methods
3. Add proper error handling for sync/async boundary

### Phase 3: Debugger Interface 🔍
1. Create proper `FSMDebugger` class
2. Export it in `__init__.py`
3. Add interactive debugging methods
4. Create structured result types

### Phase 4: Enhanced Step Results 📊
1. Define `StepResult` dataclass
2. Update `step()` to return StepResult
3. Add data change tracking
4. Include timing information

## Example Use Cases

### Current (Working)
```python
# Async execution with tracing
fsm = create_advanced_fsm(config, custom_functions=funcs)
trace = await fsm.trace_execution(data)

# Inspection
state_info = fsm.inspect_state('validate')
transitions = fsm.get_available_transitions('start')
```

### Desired (After Improvements)
```python
# Simple synchronous debugging
fsm = create_advanced_fsm(config, custom_functions=funcs)
debugger = FSMDebugger(fsm)

# Start debugging
debugger.start({'user_id': '123', 'action': 'create'})

# Step through execution
while not debugger.is_complete():
    result = debugger.step()
    print(f"Transitioned: {result.from_state} -> {result.to_state}")
    print(f"Data changes: {result.data_changes}")
    
    if debugger.at_breakpoint():
        # Inspect and potentially modify
        state = debugger.inspect_current_state()
        print(f"At breakpoint: {state}")
```

## Testing Requirements

### Unit Tests Created ✅
- `test_advanced_fsm_operations.py` - Comprehensive tests for all current features
  - Creation with execution modes
  - Breakpoint management
  - State inspection
  - Transition discovery
  - Visualization
  - Hook configuration
  - History tracking
  - Trace execution
  - Profile execution
  - Validation failure paths
  - Different action types

### Additional Tests Needed
1. Test synchronous wrappers when implemented
2. Test FSMDebugger class functionality
3. Test context helper methods
4. Test structured step results
5. Integration tests with complex FSMs

## Related Files

### Implementation Files
- `/src/dataknobs_fsm/api/advanced.py` - Main AdvancedFSM implementation
- `/src/dataknobs_fsm/execution/context.py` - ExecutionContext class
- `/src/dataknobs_fsm/execution/history.py` - ExecutionHistory with steps property
- `/src/dataknobs_fsm/core/state.py` - StateDefinition with arcs property

### Example Files
- `/examples/advanced_debugging_simple.py` - Working example with current features
- `/examples/advanced_debugging.py` - Original vision (not working, needs missing features)

### Test Files
- `/tests/test_advanced_fsm_operations.py` - Comprehensive unit tests

## Success Metrics

1. **API Simplicity**: Debugging an FSM should require < 10 lines of code
2. **Feature Completeness**: All features from original vision working
3. **Performance**: Step execution overhead < 1ms for simple FSMs
4. **Test Coverage**: > 90% coverage of AdvancedFSM code
5. **Documentation**: Complete examples for all debugging scenarios

## Next Steps

1. **Immediate** (This Week):
   - [ ] Implement `create_context()` method
   - [ ] Add context helper methods
   - [ ] Create basic synchronous wrappers

2. **Short Term** (Next Sprint):
   - [ ] Implement FSMDebugger class
   - [ ] Add structured step results
   - [ ] Update examples to use new features

3. **Long Term** (Next Month):
   - [ ] Add interactive debugging UI support
   - [ ] Implement remote debugging capabilities
   - [ ] Add execution replay from history

## Notes

- The core async functionality is solid and well-tested
- The main gap is in ease-of-use for interactive/debugging scenarios
- Synchronous support is critical for REPL/notebook usage
- Consider creating a separate `fsm-debugger` package for advanced debugging tools
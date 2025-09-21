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

## Completed Improvements ✅ (2025-01-13)

### 1. Synchronous Interface ✅
**Implemented**: Full synchronous execution methods with shared code between sync/async.

**Added Methods**:
```python
def execute_step_sync(self, context, arc_name=None) -> StepResult:
    """Execute a single transition step synchronously."""

def run_until_breakpoint_sync(self, context, max_steps=1000) -> StateInstance:
    """Run execution until a breakpoint is hit (synchronous)."""

def trace_execution_sync(self, data, initial_state=None, max_steps=1000) -> List[Dict]:
    """Execute with full tracing enabled (synchronous)."""

def profile_execution_sync(self, data, initial_state=None, max_steps=1000) -> Dict:
    """Execute with performance profiling (synchronous)."""
```

### 2. Simple Context Creation ✅
**Implemented**: Direct context creation without async context manager.

```python
def create_context(self, data, data_mode=DataHandlingMode.COPY, initial_state=None) -> ExecutionContext:
    """Create an execution context for manual control (synchronous)."""
```

### 3. Context Helper Methods ✅
**Implemented**: All requested helper methods added to ExecutionContext.

```python
def is_complete(self) -> bool:
    """Check if FSM has reached an end state."""

def get_current_state(self) -> str:
    """Get the name of the current state."""

def get_data_snapshot(self) -> Dict[str, Any]:
    """Get a snapshot of current data."""

def get_current_state_instance(self) -> Any:
    """Get the current state instance object."""
```

### 4. FSMDebugger Class ✅
**Implemented**: Full-featured synchronous debugger class.

```python
class FSMDebugger:
    """Interactive debugger for FSM execution (fully synchronous)."""

    def start(self, data, initial_state=None) -> None:
        """Start debugging session."""

    def step(self) -> StepResult:
        """Execute single step and return detailed result."""

    def continue_to_breakpoint(self) -> StateInstance:
        """Continue execution until a breakpoint is hit."""

    def inspect(self, path="") -> Any:
        """Inspect data at path."""

    def watch(self, name, path) -> None:
        """Add a watch expression."""

    def print_state(self) -> None:
        """Print current state information."""

    def inspect_current_state(self) -> Dict[str, Any]:
        """Get detailed information about current state."""

    def get_history(self, limit=10) -> List[StepResult]:
        """Get recent execution history."""
```

### 5. Structured Step Results ✅
**Implemented**: StepResult dataclass with all requested fields.

```python
@dataclass
class StepResult:
    """Result from a single step execution."""
    from_state: str
    to_state: str
    transition: str
    data_before: Dict[str, Any] = field(default_factory=dict)
    data_after: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None
    at_breakpoint: bool = False
    is_complete: bool = False
```

### 6. Code Sharing Improvements ✅
**Implemented**: Extensive refactoring to share code between sync and async implementations.

**Shared Helper Methods**:
- `_get_available_transitions()` - Gets available transitions from current state
- `_execute_arc_transform()` - Executes arc transform functions
- `_update_state_instance()` - Updates the current state instance
- `_is_at_end_state()` - Checks if at an end state
- `_record_trace_entry()` - Records trace entries
- `_record_history_step()` - Records history steps
- `_call_hook_sync()` - Calls hooks with error handling
- `_find_initial_state()` - Finds the initial state

### 7. Exports ✅
All new classes and methods are properly exported in `__init__.py`:
- `StepResult`
- `FSMDebugger`
- `ExecutionContext` (with new helper methods)

## Remaining Enhancements 🚧

### 1. Async Hook Support in Sync Methods
**Issue**: Sync methods currently only support synchronous hooks.

**Potential Solution**:
- Detect if hook is async and use `asyncio.run()` or thread pool
- Or require hooks to be sync when using sync methods

### 2. SyncAdvancedFSM Wrapper Class
**Status**: Not needed - sync methods are directly available on AdvancedFSM.

The implementation provides synchronous methods directly on the AdvancedFSM class rather than requiring a separate wrapper, making the API simpler to use.

## Testing Status ✅

All synchronous implementations have been tested and verified:
- `test_synchronous_debugging.py` - 7 tests passing
- Context creation works synchronously
- Step execution with StepResult
- FSMDebugger functionality
- Breakpoint support
- Trace and profile execution

## Example Use Cases

### Synchronous Debugging (Now Working ✅)
```python
# Create FSM with custom functions
fsm = create_advanced_fsm(config, custom_functions=funcs)

# Create synchronous debugger
debugger = FSMDebugger(fsm)

# Start debugging session
debugger.start({'user_id': '123', 'action': 'create'})

# Step through execution
while True:
    result = debugger.step()
    print(f"Step {debugger.step_count}: {result.from_state} -> {result.to_state}")

    if result.at_breakpoint:
        print("*** Hit breakpoint ***")
        debugger.print_state()

    if result.is_complete:
        print("*** Execution complete ***")
        break

    if not result.success:
        print(f"Error: {result.error}")
        break

# Direct synchronous execution
context = fsm.create_context({'value': 42})
result = fsm.execute_step_sync(context)
print(f"Transition: {result.transition}, Success: {result.success}")

# Trace execution synchronously
trace = fsm.trace_execution_sync({'input': 'test'})
for entry in trace:
    print(f"{entry['from_state']} -> {entry['to_state']}")

# Profile execution synchronously
profile = fsm.profile_execution_sync({'data': 'value'})
print(f"Total time: {profile['total_time']:.3f}s")
print(f"Transitions: {profile['transitions']}")
```

### Async Execution (Also Working)
```python
# Async execution with tracing
fsm = create_advanced_fsm(config, custom_functions=funcs)
trace = await fsm.trace_execution(data)

# Async step execution
async with fsm.execution_context(data) as context:
    while not context.is_complete():
        state = await fsm.step(context)
        if state:
            print(f"Now in: {state.definition.name}")

# Inspection (always synchronous)
state_info = fsm.inspect_state('validate')
transitions = fsm.get_available_transitions('start')
```

## Testing Requirements

### Unit Tests Created ✅
- `test_advanced_fsm_operations.py` - Comprehensive tests for async features
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

- `test_synchronous_debugging.py` - Complete tests for sync features ✅
  - Synchronous context creation
  - Step execution with StepResult
  - FSMDebugger functionality
  - Breakpoint support in sync mode
  - Trace execution sync
  - Profile execution sync
  - ExecutionContext helper methods

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

1. **Completed** ✅:
   - [x] Implement `create_context()` method
   - [x] Add context helper methods
   - [x] Create synchronous execution methods
   - [x] Implement FSMDebugger class
   - [x] Add structured step results
   - [x] Share code between sync/async implementations
   - [x] Export all new classes in `__init__.py`
   - [x] Create comprehensive tests

2. **Short Term** (Next Sprint):
   - [ ] Update advanced_debugging.py example to use new sync features
   - [ ] Add async hook detection for sync methods
   - [ ] Create notebook-friendly debugging examples

3. **Long Term** (Future):
   - [ ] Add interactive debugging UI support
   - [ ] Implement remote debugging capabilities
   - [ ] Add execution replay from history
   - [ ] Create debugging visualizations

## Notes

- ✅ All major gaps identified in the original assessment have been addressed
- ✅ Synchronous support is now fully implemented for REPL/notebook usage
- ✅ Code sharing between sync/async implementations reduces maintenance burden
- ✅ FSMDebugger provides a clean, intuitive interface for interactive debugging
- The implementation exceeds the original requirements by adding features like watches, history tracking, and detailed step results
- Consider creating a separate `fsm-debugger` package for advanced debugging tools in the future
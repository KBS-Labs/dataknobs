# Advanced API Reference

> **📖 Also see:** [Auto-generated API Reference](../../../api/reference/fsm.md) - Complete documentation from source code docstrings

This page provides curated examples and usage patterns. The auto-generated reference provides exhaustive technical documentation with all methods, parameters, and type annotations.

---

## Overview

The Advanced API provides fine-grained control over FSM execution, including step-by-step execution, breakpoints, tracing, profiling, and debugging capabilities. This API is designed for complex workflows requiring detailed control and monitoring.

## AdvancedFSM Class

The `AdvancedFSM` class provides advanced execution control features for FSM workflows.

### Initialization

```python
from dataknobs_fsm import AdvancedFSM, ExecutionMode, FSM

# Create from FSM instance
fsm_instance = FSM(...)
advanced_fsm = AdvancedFSM(
    fsm=fsm_instance,
    execution_mode=ExecutionMode.STEP_BY_STEP,
    custom_functions={'my_func': my_function}
)

# Or use factory function
from dataknobs_fsm import create_advanced_fsm

advanced_fsm = create_advanced_fsm(
    config="path/to/config.yaml",  # or dict or FSM instance
    custom_functions={'my_func': my_function}
)
```

### Execution Modes

```python
from dataknobs_fsm import ExecutionMode

# Available modes
ExecutionMode.STEP_BY_STEP  # Execute one transition at a time
ExecutionMode.BREAKPOINT    # Stop at specific states
ExecutionMode.TRACE         # Full execution tracing
ExecutionMode.PROFILE       # Performance profiling
ExecutionMode.DEBUG         # Debug mode with detailed logging
```

## Step-by-Step Execution

Execute FSM one step at a time with full control:

### Synchronous Step Execution

```python
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Create execution context
context = advanced_fsm.create_context(
    data={'input': 'value'},
    data_mode=DataHandlingMode.COPY,
    initial_state='start'  # Optional
)

# Execute single step
result = advanced_fsm.execute_step_sync(context, arc_name=None)

# Check result
print(f"Transition: {result.from_state} -> {result.to_state}")
print(f"Success: {result.success}")
print(f"At breakpoint: {result.at_breakpoint}")
print(f"Complete: {result.is_complete}")
```

### Asynchronous Step Execution

```python
# Create async context
async with advanced_fsm.execution_context(data, initial_state='start') as context:
    # Execute single step
    new_state = await advanced_fsm.step(context, arc_name=None)

    if new_state:
        print(f"Transitioned to: {new_state.definition.name}")
```

### StepResult Object

```python
from dataknobs_fsm import StepResult

# StepResult contains:
result.from_state      # State transitioning from
result.to_state        # State transitioned to
result.transition      # Arc/transition name
result.data_before     # Data snapshot before transition
result.data_after      # Data snapshot after transition
result.duration        # Execution time in seconds
result.success         # Whether transition succeeded
result.error           # Error message if failed
result.at_breakpoint   # Whether stopped at breakpoint
result.is_complete     # Whether reached end state
```

## Breakpoints

Set breakpoints to pause execution at specific states:

### Managing Breakpoints

```python
# Add breakpoint
advanced_fsm.add_breakpoint("validate_data")

# Remove breakpoint
advanced_fsm.remove_breakpoint("validate_data")

# Clear all breakpoints
advanced_fsm.clear_breakpoints()

# Get current breakpoints
breakpoints = advanced_fsm.breakpoints  # Returns set of state names
```

### Running to Breakpoint

```python
# Synchronous execution
context = advanced_fsm.create_context(data)
state = advanced_fsm.run_until_breakpoint_sync(context, max_steps=1000)

# Asynchronous execution
async with advanced_fsm.execution_context(data) as context:
    state = await advanced_fsm.run_until_breakpoint(context, max_steps=1000)
```

## Execution Hooks

Monitor execution events with hooks:

```python
from dataknobs_fsm import ExecutionHook

# Define hooks
hooks = ExecutionHook(
    on_state_enter=lambda state: print(f"Entering: {state}"),
    on_state_exit=lambda state: print(f"Exiting: {state}"),
    on_arc_execute=lambda arc: print(f"Executing arc: {arc}"),
    on_error=lambda error: print(f"Error: {error}"),
    on_resource_acquire=lambda res: print(f"Acquired: {res}"),
    on_resource_release=lambda res: print(f"Released: {res}"),
    on_transaction_begin=lambda: print("Transaction started"),
    on_transaction_commit=lambda: print("Transaction committed"),
    on_transaction_rollback=lambda: print("Transaction rolled back")
)

# Set hooks
advanced_fsm.set_hooks(hooks)
```

## Tracing

Trace execution for debugging and analysis:

### Synchronous Tracing

```python
# Execute with tracing
trace = advanced_fsm.trace_execution_sync(
    data={'input': 'value'},
    initial_state='start',
    max_steps=1000
)

# Analyze trace
for entry in trace:
    print(f"[{entry['timestamp']}] {entry['from_state']} -> {entry['to_state']}")
    print(f"  Transition: {entry['transition']}")
    print(f"  Data: {entry['data']}")
```

### Asynchronous Tracing

```python
# Execute with async tracing
trace = await advanced_fsm.trace_execution(
    data={'input': 'value'},
    initial_state='start'
)
```

## Profiling

Profile FSM execution for performance optimization:

### Synchronous Profiling

```python
# Profile execution
profile = advanced_fsm.profile_execution_sync(
    data={'input': 'value'},
    initial_state='start',
    max_steps=1000
)

# Analyze profile
print(f"Total time: {profile['total_time']}s")
print(f"Transitions: {profile['transitions']}")
print(f"States visited: {profile['states_visited']}")
print(f"Avg transition time: {profile['avg_transition_time']}s")

# State-specific timings
for state, times in profile['state_times'].items():
    print(f"{state}:")
    print(f"  Count: {times['count']}")
    print(f"  Avg: {times['avg']}s")
    print(f"  Min: {times['min']}s")
    print(f"  Max: {times['max']}s")
```

### Asynchronous Profiling

```python
# Profile async execution
profile = await advanced_fsm.profile_execution(
    data={'input': 'value'},
    initial_state='start'
)
```

## History Tracking

Track execution history for audit and analysis:

### Enable History

```python
from dataknobs_fsm.storage.memory import InMemoryStorage

# Enable history with storage
storage = InMemoryStorage()
advanced_fsm.enable_history(
    storage=storage,
    max_depth=100  # Maximum history steps
)

# Check if enabled
if advanced_fsm.history_enabled:
    print(f"Max depth: {advanced_fsm.max_history_depth}")
```

### Access History

```python
# Get history
history = advanced_fsm.get_history()

# Get execution steps
steps = advanced_fsm.execution_history

# Save history
await advanced_fsm.save_history()

# Load history
await advanced_fsm.load_history("history_id")
```

## Resource Management

Register and manage external resources:

```python
from dataknobs_fsm.resources.base import IResourceProvider

# Register resource from dict
advanced_fsm.register_resource("database", {
    "type": "database",
    "backend": "postgresql",
    "connection_string": "postgresql://..."
})

# Register resource instance
class CustomResource(IResourceProvider):
    async def acquire(self):
        # Acquire resource
        pass

    async def release(self):
        # Release resource
        pass

resource = CustomResource()
advanced_fsm.register_resource("custom", resource)
```

## Transaction Management

Configure transaction strategies:

```python
from dataknobs_fsm.core.transactions import TransactionStrategy

# Configure transactions
advanced_fsm.configure_transactions(
    strategy=TransactionStrategy.BATCH,
    batch_size=100,
    commit_interval=10
)
```

## FSM Inspection

Inspect FSM structure and state:

### Get Available Transitions

```python
# Get transitions from a state
transitions = advanced_fsm.get_available_transitions("process_data")

for trans in transitions:
    print(f"Arc: {trans['name']} -> {trans['target']}")
    print(f"Has pre-test: {trans['has_pre_test']}")
    print(f"Has transform: {trans['has_transform']}")
```

### Inspect State

```python
# Inspect state configuration
state_info = advanced_fsm.inspect_state("validate")

print(f"Name: {state_info['name']}")
print(f"Is start: {state_info['is_start']}")
print(f"Is end: {state_info['is_end']}")
print(f"Resources: {state_info['resources']}")
print(f"Arcs: {state_info['arcs']}")
```

### Visualize FSM

```python
# Generate GraphViz DOT format
dot = advanced_fsm.visualize_fsm()

# Save to file
with open("fsm.dot", "w") as f:
    f.write(dot)

# Convert to image
# dot -Tpng fsm.dot -o fsm.png
```

### Validate Network

```python
# Validate FSM consistency
validation = await advanced_fsm.validate_network()

if validation['valid']:
    print("FSM is valid")
else:
    for issue in validation['issues']:
        print(f"Issue: {issue['type']}")
        if issue['type'] == 'unreachable_states':
            print(f"  States: {issue['states']}")

print(f"Stats: {validation['stats']}")
```

## FSMDebugger

Interactive debugger for FSM execution:

### Initialize Debugger

```python
from dataknobs_fsm import FSMDebugger

# Create debugger
debugger = FSMDebugger(advanced_fsm)

# Start debugging session
debugger.start(
    data={'input': 'value'},
    initial_state='start'  # Optional
)
```

### Debugging Commands

```python
# Step through execution
result = debugger.step()

# Continue to breakpoint
state = debugger.continue_to_breakpoint()

# Inspect data
value = debugger.inspect("field.subfield")  # Dot notation
all_data = debugger.inspect()  # All data

# Watch variables
debugger.watch("status", "data.status")
debugger.unwatch("status")
debugger.print_watches()

# Print state information
debugger.print_state()

# Get detailed state info
info = debugger.inspect_current_state()

# Get execution history
history = debugger.get_history(limit=10)

# Reset debugger
debugger.reset(new_data)  # Optional new data
```

### Interactive Debugging Example

```python
def debug_fsm(config, test_data):
    # Create advanced FSM
    fsm = create_advanced_fsm(config)

    # Set breakpoints
    fsm.add_breakpoint("validation")
    fsm.add_breakpoint("error_handler")

    # Create debugger
    debugger = FSMDebugger(fsm)
    debugger.start(test_data)

    # Interactive debugging loop
    while True:
        debugger.print_state()
        cmd = input("debug> ")

        if cmd == "step" or cmd == "s":
            result = debugger.step()
            if result.is_complete:
                print("Execution complete")
                break

        elif cmd == "continue" or cmd == "c":
            state = debugger.continue_to_breakpoint()
            if not state:
                print("Execution complete")
                break

        elif cmd.startswith("inspect "):
            path = cmd[8:]
            value = debugger.inspect(path)
            print(f"{path} = {value}")

        elif cmd.startswith("watch "):
            parts = cmd[6:].split()
            if len(parts) == 2:
                debugger.watch(parts[0], parts[1])

        elif cmd == "watches":
            debugger.print_watches()

        elif cmd == "history":
            for step in debugger.get_history(5):
                print(f"  {step.from_state} -> {step.to_state}")

        elif cmd == "quit" or cmd == "q":
            break

        else:
            print("Commands: step, continue, inspect <path>, watch <name> <path>, watches, history, quit")
```

## Custom Execution Strategies

Set custom execution strategies:

```python
from dataknobs_fsm.execution.engine import TraversalStrategy

# Set execution strategy
advanced_fsm.set_execution_strategy(TraversalStrategy.DEPTH_FIRST)
# Options: DEPTH_FIRST, BREADTH_FIRST, PRIORITY_BASED, etc.
```

## Data Handlers

Configure custom data handlers:

```python
from dataknobs_fsm.core.data_modes import DataHandler

class CustomDataHandler(DataHandler):
    def handle(self, data):
        # Custom data handling
        return processed_data

handler = CustomDataHandler()
advanced_fsm.set_data_handler(handler)
```

## Complete Example

Here's a complete example using the Advanced API:

```python
import asyncio
from dataknobs_fsm import create_advanced_fsm, ExecutionMode, ExecutionHook

async def debug_workflow():
    # Create FSM with custom functions
    fsm = create_advanced_fsm(
        "workflow.yaml",
        custom_functions={
            'validate': lambda data: data.get('valid', False),
            'transform': lambda data: {'processed': True, **data}
        }
    )

    # Enable history tracking
    fsm.enable_history(max_depth=100)

    # Set breakpoints
    fsm.add_breakpoint("validation")
    fsm.add_breakpoint("error_state")

    # Configure hooks
    fsm.set_hooks(ExecutionHook(
        on_state_enter=lambda s: print(f"-> {s}"),
        on_error=lambda e: print(f"Error: {e}")
    ))

    # Test data
    test_data = {
        'id': 123,
        'valid': True,
        'data': 'test'
    }

    # Profile execution
    print("Profiling execution...")
    profile = await fsm.profile_execution(test_data)
    print(f"Total time: {profile['total_time']}s")

    # Trace execution
    print("\nTracing execution...")
    trace = await fsm.trace_execution(test_data)
    for entry in trace:
        print(f"  {entry['from_state']} -> {entry['to_state']}")

    # Step-by-step execution
    print("\nStep-by-step execution...")
    async with fsm.execution_context(test_data) as context:
        while True:
            # Check available transitions
            state_name = context.current_state
            transitions = fsm.get_available_transitions(state_name)

            if not transitions:
                print("No more transitions")
                break

            # Execute step
            new_state = await fsm.step(context)

            if not new_state:
                print("Execution complete")
                break

            # Check if at breakpoint
            if state_name in fsm.breakpoints:
                print(f"Breakpoint hit at {state_name}")
                # Could add interactive debugging here

    # Get history
    history = fsm.get_history()
    if history:
        print(f"\nExecution history: {len(history.steps)} steps")

    # Validate network
    validation = await fsm.validate_network()
    print(f"\nFSM valid: {validation['valid']}")

# Run the example
asyncio.run(debug_workflow())
```

## Best Practices

### 1. Resource Cleanup

Always ensure proper cleanup:

```python
async with advanced_fsm.execution_context(data) as context:
    # Resources automatically cleaned up
    await advanced_fsm.step(context)
```

### 2. Error Handling

Handle errors appropriately:

```python
result = advanced_fsm.execute_step_sync(context)
if not result.success:
    print(f"Error: {result.error}")
    # Handle error condition
```

### 3. Performance Monitoring

Use profiling for optimization:

```python
# Profile to find bottlenecks
profile = advanced_fsm.profile_execution_sync(data)

# Find slowest states
slowest = max(profile['state_times'].items(),
              key=lambda x: x[1]['avg'])
print(f"Slowest state: {slowest[0]} ({slowest[1]['avg']}s avg)")
```

### 4. Debugging Production Issues

Use tracing for production debugging:

```python
# Enable tracing for specific execution
advanced_fsm.execution_mode = ExecutionMode.TRACE
trace = await advanced_fsm.trace_execution(problematic_data)

# Analyze trace
for entry in trace:
    if 'error' in entry.get('to_state', ''):
        print(f"Error transition: {entry}")
```

## API Reference

### AdvancedFSM Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `create_context(data, data_mode, initial_state)` | Create execution context | ExecutionContext |
| `execute_step_sync(context, arc_name)` | Execute single step (sync) | StepResult |
| `step(context, arc_name)` | Execute single step (async) | StateInstance |
| `run_until_breakpoint(context, max_steps)` | Run to breakpoint (async) | StateInstance |
| `run_until_breakpoint_sync(context, max_steps)` | Run to breakpoint (sync) | StateInstance |
| `trace_execution(data, initial_state)` | Trace execution (async) | List[Dict] |
| `trace_execution_sync(data, initial_state, max_steps)` | Trace execution (sync) | List[Dict] |
| `profile_execution(data, initial_state)` | Profile execution (async) | Dict |
| `profile_execution_sync(data, initial_state, max_steps)` | Profile execution (sync) | Dict |
| `add_breakpoint(state_name)` | Add breakpoint | None |
| `remove_breakpoint(state_name)` | Remove breakpoint | None |
| `clear_breakpoints()` | Clear all breakpoints | None |
| `set_hooks(hooks)` | Set execution hooks | None |
| `enable_history(storage, max_depth)` | Enable history tracking | None |
| `disable_history()` | Disable history tracking | None |
| `register_resource(name, resource)` | Register resource | None |
| `configure_transactions(strategy, **config)` | Configure transactions | None |
| `get_available_transitions(state_name)` | Get available transitions | List[Dict] |
| `inspect_state(state_name)` | Inspect state config | Dict |
| `visualize_fsm()` | Generate DOT visualization | str |
| `validate_network()` | Validate FSM network | Dict |

### FSMDebugger Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `start(data, initial_state)` | Start debug session | None |
| `step()` | Execute single step | StepResult |
| `continue_to_breakpoint()` | Continue to breakpoint | StateInstance |
| `inspect(path)` | Inspect data at path | Any |
| `watch(name, path)` | Add watch expression | None |
| `unwatch(name)` | Remove watch | None |
| `print_watches()` | Print all watches | None |
| `print_state()` | Print state info | None |
| `inspect_current_state()` | Get state details | Dict |
| `get_history(limit)` | Get execution history | List[StepResult] |
| `reset(data)` | Reset debugger | None |

## Next Steps

- [Simple API](simple.md) - Simpler API for basic use cases
- [CLI Guide](../guides/cli.md) - Command-line interface and debugging
- [Examples](../examples/index.md) - Complete examples using Advanced API


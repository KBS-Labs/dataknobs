---
title: fsm (curated)
---

# FSM API Reference

The FSM package provides two main APIs for different use cases, plus core components for direct use:

## SimpleFSM API

The [SimpleFSM](simple.md) class provides a high-level, configuration-driven interface for creating and running finite state machines synchronously. It's perfect for:

- Configuration-based FSM creation
- Simple to moderate complexity workflows
- Synchronous processing contexts
- Batch and stream operations

**Key Features:**
- Configuration-driven (YAML/JSON or dict)
- Custom function registration
- Automatic resource management
- Built-in batch and stream processing
- Support for all three data handling modes
- Synchronous-only operation

**Import:**
```python
from dataknobs_fsm.api.simple import SimpleFSM
```

[View SimpleFSM Documentation →](simple.md)

## AsyncSimpleFSM API

The [AsyncSimpleFSM](async_simple.md) class provides a native async/await interface for creating and running finite state machines asynchronously. It's designed for:

- Async/await contexts
- High-concurrency applications
- I/O-bound operations
- Integration with async frameworks

**Key Features:**
- Native async/await support
- Configuration-driven (YAML/JSON or dict)
- Async custom function support
- Efficient async batch and stream processing
- Full async resource management

**Import:**
```python
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
```

[View AsyncSimpleFSM Documentation →](async_simple.md)

## AdvancedFSM API

The [AdvancedFSM](advanced.md) class offers fine-grained control over FSM execution with debugging and monitoring capabilities. Use it for:

- Step-by-step execution and debugging
- Production monitoring with hooks
- Performance profiling
- Complex state machine inspection

**Key Features:**
- Step-by-step execution control
- Breakpoint debugging
- Execution tracing and profiling
- Custom execution hooks
- History tracking and persistence
- Network visualization
- Transaction configuration

**Import:**
```python
from dataknobs_fsm import AdvancedFSM, ExecutionMode, ExecutionHook
```

[View AdvancedFSM Documentation →](advanced.md)

## Quick Comparison

| Feature | SimpleFSM | AsyncSimpleFSM | AdvancedFSM |
|---------|-----------|----------------|--------------|
| **Primary Use Case** | Sync processing | Async processing | Debugging & monitoring |
| **Context** | Synchronous | Asynchronous | Both |
| **Configuration Support** | Yes (required) | Yes (required) | Yes |
| **Native Async** | No | Yes | Yes (multiple methods) |
| **Batch Processing** | Yes (`process_batch`) | Yes (`process_batch`) | Via execution strategies |
| **Stream Processing** | Yes (`process_stream`) | Yes (`process_stream`) | Via execution strategies |
| **Step-by-step Execution** | No | No | Yes |
| **Debugging Features** | No | No | Breakpoints, tracing |
| **Profiling** | No | No | Yes |
| **Execution Hooks** | No | No | Yes |
| **History Management** | No | No | Yes |
| **Custom Functions** | Yes (registered) | Yes (async preferred) | Yes |

## Core Components

Both APIs build on these core components that can also be used directly:

### FSM Core
```python
from dataknobs_fsm import FSM, StateDefinition, StateInstance, ArcDefinition
```

- **FSM**: Core finite state machine class
- **StateDefinition**: Template for states with schemas and validation
- **StateInstance**: Runtime state instances with data and context
- **ArcDefinition**: Transition definitions with optional transforms

### Execution Components
```python
from dataknobs_fsm import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
```

- **ExecutionContext**: Manages execution state, history, and resources
- **ExecutionEngine**: Synchronous execution with strategy support
- **AsyncExecutionEngine**: Asynchronous execution with concurrency

### Configuration & Building
```python
from dataknobs_fsm import ConfigLoader, FSMBuilder
```

- **ConfigLoader**: Load FSM configurations from YAML/JSON
- **FSMBuilder**: Build FSM instances from configurations

### Data Handling Modes
```python
from dataknobs_fsm.core.data_modes import DataHandlingMode
```

- **DataHandlingMode.COPY**: Safe concurrent processing (default)
- **DataHandlingMode.REFERENCE**: Memory-efficient with locking
- **DataHandlingMode.DIRECT**: High-performance in-place operations

### Processing Modes
```python
from dataknobs_fsm.core.modes import ProcessingMode
```

- **ProcessingMode.SINGLE**: Process one record at a time
- **ProcessingMode.BATCH**: Process multiple records in batches
- **ProcessingMode.STREAM**: Process continuous streams of data

## Usage Examples

### Configuration-Driven Workflow (SimpleFSM)

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

config = {
    "name": "simple_workflow",
    "states": [
        {"name": "start", "is_start": True},
        {"name": "process"},
        {"name": "end", "is_end": True}
    ],
    "arcs": [
        {
            "from": "start",
            "to": "process",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: {**data, 'processed': True}"
            }
        },
        {"from": "process", "to": "end"}
    ]
}

fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)
result = fsm.process({"value": 5})
print(result["data"])  # {'value': 5, 'processed': True}
```

### Debugging Workflow (AdvancedFSM)

```python
from dataknobs_fsm import AdvancedFSM, ExecutionMode, ExecutionHook

class DebugHook(ExecutionHook):
    def on_state_enter(self, state, data):
        print(f"Entering: {state.name}")

    def on_arc_traverse(self, arc, data):
        print(f"Traversing: {arc.source} -> {arc.target}")

# Create FSM with debugging
fsm = AdvancedFSM(
    "workflow.yaml",
    execution_mode=ExecutionMode.DEBUG
)

# Set debugging hooks
fsm.set_hooks([DebugHook()])

# Add breakpoint
fsm.add_breakpoint("process")

# Step through execution
import asyncio

async def debug_run():
    context = fsm.create_context({"input": "data"})

    # Run until breakpoint
    await fsm.run_until_breakpoint(context)
    print(f"Stopped at: {context.current_state}")

    # Continue execution
    await fsm.step(context)

asyncio.run(debug_run())
```

## API Methods Summary

### SimpleFSM Key Methods

- `process(data, initial_state=None, timeout=None)` - Process single record synchronously
- `process_batch(data, batch_size=10, max_workers=4)` - Batch processing
- `process_stream(source, sink=None, chunk_size=100)` - Stream processing
- `validate(data)` - Validate data against schema
- `get_states()` - Get all state names
- `get_resources()` - Get resource names

### AdvancedFSM Key Methods

**Execution Control:**
- `create_context(data, data_mode, initial_state)` - Create execution context
- `step(context, arc_name=None)` - Execute single transition
- `run_until_breakpoint(context, max_steps=1000)` - Run to breakpoint

**Debugging:**
- `add_breakpoint(state_name)` - Add debugging breakpoint
- `trace_execution(data, initial_state=None)` - Full execution trace
- `profile_execution(data, initial_state=None)` - Performance profiling

**Inspection:**
- `get_available_transitions(state_name)` - Get valid transitions
- `inspect_state(state_name)` - Inspect state configuration
- `visualize_fsm()` - Generate visualization
- `validate_network()` - Validate FSM consistency

## API Stability

The FSM package follows semantic versioning:

- **SimpleFSM**: Stable synchronous API, backward compatible within major versions
- **AsyncSimpleFSM**: Stable async API, backward compatible within major versions
- **AdvancedFSM**: Stable API, exported at package level
- **Core Components**: Very stable, minimal changes expected
- **Data/Processing Modes**: Stable enums, but require direct import

## Import Reference

```python
# Core components (exported at package level)
from dataknobs_fsm import (
    FSM, StateDefinition, StateInstance, ArcDefinition,
    ExecutionContext, ConfigLoader, FSMBuilder
)

# Advanced API (exported at package level)
from dataknobs_fsm import (
    AdvancedFSM, ExecutionMode, ExecutionHook,
    StepResult, FSMDebugger, create_advanced_fsm
)

# Simple APIs (direct import required)
from dataknobs_fsm.api.simple import SimpleFSM  # For synchronous contexts
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM  # For async contexts

# Data modes (direct import required)
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import ProcessingMode

# Execution engines (if needed directly)
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
```

## Getting Help

- Check the [Examples](../examples/index.md) for real-world usage
- Read the [Patterns Guide](../patterns/index.md) for common scenarios
- Review the [Guides](../guides/index.md) for specific topics:
  - [Data Modes Guide](../guides/data-modes.md) - Understanding data handling modes
  - [Resource Management](../guides/resources.md) - Managing external resources
  - [Streaming Guide](../guides/streaming.md) - Stream processing
- See the [FAQ](../faq.md) for common questions
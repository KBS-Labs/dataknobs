# Quick Start Guide

Get started with the FSM package in just a few minutes! This guide will walk you through the basics of creating and running finite state machines.

## Installation

```bash
pip install dataknobs-fsm
```

Or with optional dependencies:

```bash
# With database support
pip install dataknobs-fsm[database]

# With LLM provider support
pip install dataknobs-fsm[llm]
```

## Your First FSM

Let's create a simple FSM that processes data through multiple stages:

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Define FSM configuration
config = {
    "name": "simple_processor",
    "states": [
        {"name": "start", "is_start": True},
        {"name": "validate"},
        {"name": "process"},
        {"name": "complete", "is_end": True}
    ],
    "arcs": [
        {
            "from": "start",
            "to": "validate",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: {**data, 'validated': False}"
            }
        },
        {
            "from": "validate",
            "to": "process",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: {**data, 'validated': True}"
            },
            "pre_test": {
                "type": "inline",
                "code": "lambda data, ctx: data.get('value', 0) > 0"
            }
        },
        {
            "from": "process",
            "to": "complete",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: {**data, 'result': data['value'] * 2}"
            }
        }
    ]
}

# Create FSM instance
fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)

# Process data through the FSM
result = fsm.process({"value": 10})
print(result)
# Output: {'final_state': 'complete', 'data': {'value': 10, 'validated': True, 'result': 20}, 'path': ['start', 'validate', 'process', 'complete'], 'success': True, 'error': None, 'metadata': {'arc_start_validate_usage': 1, 'arc_validate_process_usage': 1, 'arc_process_complete_usage': 1}}
```

## Using Configuration Files

For more complex FSMs, use YAML configuration:

```yaml
# workflow.yaml
name: data_workflow
description: A simple data processing workflow

states:
  - name: start
    is_start: true
    
  - name: fetch_data
    metadata:
      timeout: 30
      
  - name: transform
    
  - name: save
    
  - name: complete
    is_end: true

arcs:
  - from: start
    to: fetch_data
    
  - from: fetch_data
    to: transform
    transform:
      type: python
      module: myapp.transformers
      name: clean_data
      
  - from: transform
    to: save
    transform:
      type: lambda
      code: |
        lambda data: {
            **data,
            "saved": True,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
        
  - from: save
    to: complete
```

Load and run the configuration:

```python
from dataknobs_fsm.api.simple import SimpleFSM

# Load FSM from configuration file
fsm = SimpleFSM("workflow.yaml")

# Process input data
result = fsm.process({"source": "api", "records": 100})
print(f"Final state: {result['final_state']}")
print(f"Success: {result['success']}")
print(f"Processed data: {result['data']}")
```

## Async Execution

For I/O-bound operations, use async execution:

```python
import asyncio
from dataknobs_fsm.api.simple import SimpleFSM

# Define async processing functions
async def fetch_data(state):
    """Simulate async API call."""
    await asyncio.sleep(1)
    data = state.data.copy()
    data["fetched"] = True
    return data

async def save_data(state):
    """Simulate async database save."""
    await asyncio.sleep(0.5)
    data = state.data.copy()
    data["saved"] = True
    return data

# Configuration with registered functions
config = {
    "name": "async_workflow",
    "states": [
        {"name": "start", "initial": True},
        {"name": "fetch"},
        {"name": "save"},
        {"name": "done", "terminal": True}
    ],
    "arcs": [
        {
            "from": "start",
            "to": "fetch",
            "transform": {"type": "registered", "name": "fetch_data"}
        },
        {
            "from": "fetch",
            "to": "save",
            "transform": {"type": "registered", "name": "save_data"}
        },
        {
            "from": "save",
            "to": "done"
        }
    ]
}

# Create FSM with async functions
fsm = SimpleFSM(
    config,
    custom_functions={
        "fetch_data": fetch_data,
        "save_data": save_data
    }
)

# Run asynchronously
async def main():
    result = await fsm.process_async({"id": 123})
    print(f"Success: {result['success']}")
    print(f"Data: {result['data']}")

asyncio.run(main())
```

## Using Resources

Manage external resources like databases and APIs:

```python
from dataknobs_fsm.api.simple import SimpleFSM

# Configuration with resources
config = {
    "name": "resource_workflow",
    "resources": [
        {
            "name": "db",
            "type": "database",
            "provider": "sqlite",
            "config": {"database": "myapp.db"}
        }
    ],
    "states": [
        {"name": "start", "initial": True},
        {
            "name": "query",
            "resources": ["db"]  # This state requires the database resource
        },
        {"name": "done", "terminal": True}
    ],
    "arcs": [
        {
            "from": "start",
            "to": "query"
        },
        {
            "from": "query",
            "to": "done",
            "transform": {
                "type": "registered",
                "name": "process_query_result"
            }
        }
    ]
}

# Custom function that uses resources
def process_query_result(state):
    """Process database query results."""
    # In actual implementation, resources are accessed via context
    data = state.data.copy()
    data["processed"] = True
    return data

# Create FSM with resources
fsm = SimpleFSM(
    config,
    custom_functions={"process_query_result": process_query_result}
)

# Process with resource management
result = fsm.process({"user_id": 1})
print(f"Result: {result}")
```

## Batch Processing

Process multiple items efficiently:

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.modes import ProcessingMode

# Configuration for batch processing
config = {
    "name": "batch_processor",
    "states": [
        {"name": "start", "initial": True},
        {"name": "process"},
        {"name": "done", "terminal": True}
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
        {"from": "process", "to": "done"}
    ]
}

# Create FSM for batch processing
fsm = SimpleFSM(config)

# Process batch of items
items = [
    {"id": 1, "value": 10},
    {"id": 2, "value": 20},
    {"id": 3, "value": 30}
]

# Use process_batch method
results = fsm.process_batch(
    items,
    max_workers=3,
    processing_mode=ProcessingMode.BATCH
)

for result in results:
    print(f"Item {result['data']['id']}: Success={result['success']}")
```

## Error Handling

Implement error handling with the AdvancedFSM:

```python
from dataknobs_fsm import AdvancedFSM, ExecutionMode
import random

def risky_operation(state):
    """Operation that might fail."""
    if random.random() < 0.5:
        raise ValueError("Random failure")
    data = state.data.copy()
    data["processed"] = True
    return data

# Configuration with error handling
config = {
    "name": "error_handler",
    "states": [
        {"name": "start", "initial": True},
        {"name": "process"},
        {"name": "error"},
        {"name": "done", "terminal": True}
    ],
    "arcs": [
        {
            "from": "start",
            "to": "process",
            "transform": {
                "type": "registered",
                "name": "risky_operation"
            },
            "error_handler": {
                "target_state": "error",
                "max_retries": 3
            }
        },
        {
            "from": "error",
            "to": "process",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: {**data, 'retry': True}"
            }
        },
        {"from": "process", "to": "done"}
    ]
}

# Create FSM with error handling
fsm = AdvancedFSM(
    config,
    execution_mode=ExecutionMode.DEBUG,
    custom_functions={"risky_operation": risky_operation}
)

# Run with automatic error recovery
result = fsm.run({"input": "data"})
print(f"Success after retries: {result['success']}")
print(f"Final data: {result['data']}")
```

## Advanced Features with AdvancedFSM

For debugging and step-by-step execution:

```python
from dataknobs_fsm import AdvancedFSM, ExecutionMode, ExecutionHook

class DebugHook(ExecutionHook):
    """Custom hook for debugging."""

    def on_state_enter(self, state, data):
        print(f"→ Entering: {state.name}")
        print(f"  Data: {data}")

    def on_state_exit(self, state, data):
        print(f"← Exiting: {state.name}")

# Create FSM with debugging
fsm = AdvancedFSM(
    "workflow.yaml",
    execution_mode=ExecutionMode.STEP_BY_STEP,
    hooks=[DebugHook()]
)

# Step through execution
for step in fsm.step_through({"input": "data"}):
    print(f"\nCurrent State: {step.current_state}")
    if input("Continue? (y/n): ").lower() != 'y':
        break

# Or run with profiling
result, profile = fsm.run_with_profile({"input": "data"})
print(f"\nExecution took {profile.total_time:.2f}s")
print(f"States visited: {profile.states_visited}")
```

## Next Steps

Now that you understand the basics:

1. Explore [Integration Patterns](patterns/index.md) for pre-built solutions
2. Read the [API Documentation](api/index.md) for detailed reference
3. Check out [Examples](examples/index.md) for real-world use cases
4. Learn about [Data Modes](guides/data-modes.md) for efficient data handling
5. Understand [Resource Management](guides/resources.md) for external integrations

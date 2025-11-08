# dataknobs-fsm API Reference

## Overview

The `dataknobs-fsm` package provides finite state machine implementations for building complex workflows with states, transitions (arcs), and data transformations. It offers three API tiers for different use cases, from simple scripts to complex production systems.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/fsm.md) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/fsm/src/dataknobs_fsm) - Browse on GitHub
> - [Package Guide](../packages/fsm/index.md) - Detailed documentation

## API Tiers

The package provides three API levels for different needs:

### SimpleFSM - Synchronous API

**Source:** [`api/simple.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/src/dataknobs_fsm/api/simple.py)

High-level synchronous interface for scripts and simple workflows.

```python
from dataknobs_fsm import SimpleFSM

# Define FSM configuration
config = {
    'name': 'data_pipeline',
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {'name': 'start', 'is_start': True},
            {
                'name': 'transform',
                'functions': {
                    'transform': 'lambda state: {"result": state.data["value"] * 2}'
                }
            },
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'transform', 'name': 'process'},
            {'from': 'transform', 'to': 'end', 'name': 'complete'}
        ]
    }]
}

# Create and run FSM
fsm = SimpleFSM(config)
result = fsm.process({'value': 21})
print(result)  # {'final_state': 'end', 'data': {'result': 42}, 'success': True}
```

**Use SimpleFSM for:**
- Scripts and prototypes
- Simple pipelines
- Synchronous code
- Quick development

### AsyncSimpleFSM - Asynchronous API

**Source:** [`api/async_simple.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/src/dataknobs_fsm/api/async_simple.py)

Asynchronous version for production services and concurrent processing.

```python
from dataknobs_fsm import AsyncSimpleFSM

fsm = AsyncSimpleFSM(config)
result = await fsm.process({'value': 21})
```

**Use AsyncSimpleFSM for:**
- Web services and APIs
- Concurrent processing
- Async applications
- Production systems

### AdvancedFSM - Debugging API

**Source:** [`api/advanced.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/src/dataknobs_fsm/api/advanced.py)

Full control with debugging, step-by-step execution, and profiling.

```python
from dataknobs_fsm import AdvancedFSM, FSMDebugger

fsm = AdvancedFSM(config)
debugger = FSMDebugger(fsm)

# Set breakpoint
debugger.add_breakpoint('transform')

# Step through execution
for step in debugger.step_through({'value': 21}):
    print(f"State: {step.state_name}, Data: {step.data}")
```

**Use AdvancedFSM for:**
- Complex workflows
- Debugging and testing
- Profiling and tracing
- Custom execution strategies

## Data Handling Modes

The FSM supports three data handling modes that control how data flows through states:

### COPY Mode (Default)

**Source:** [`core/data_modes.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/src/dataknobs_fsm/core/data_modes.py)

Creates a deep copy of data for each state. Safe for concurrent processing but higher memory usage.

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

config = {
    'name': 'pipeline',
    'data_mode': DataHandlingMode.COPY,  # or 'copy'
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {'name': 'start', 'is_start': True},
            {
                'name': 'step1',
                'functions': {
                    'transform': 'lambda state: {**state.data, "field1": "value1"}'
                }
            },
            {
                'name': 'step2',
                'functions': {
                    'transform': 'lambda state: {**state.data, "field2": "value2"}'
                }
            },
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'step1'},
            {'from': 'step1', 'to': 'step2'},
            {'from': 'step2', 'to': 'end'}
        ]
    }]
}

fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)
```

**Characteristics:**
- Deep copy on state entry
- Isolated modifications
- Thread-safe
- Higher memory usage
- **Best for:** Production systems, parallel processing

### REFERENCE Mode

Uses references with optimistic locking. Memory-efficient with moderate performance.

```python
fsm = SimpleFSM(config, data_mode=DataHandlingMode.REFERENCE)
```

**Characteristics:**
- Lazy loading
- Optimistic locking
- Version tracking
- Memory-efficient
- **Best for:** Large datasets, memory-constrained environments

### DIRECT Mode

In-place modification for maximum performance. Not thread-safe.

```python
fsm = SimpleFSM(config, data_mode=DataHandlingMode.DIRECT)
```

**Characteristics:**
- In-place modifications
- Fastest performance
- Not thread-safe
- Single state access at a time
- **Best for:** Single-threaded pipelines, performance-critical paths

## Core Methods

### process()

Process a single data item through the FSM.

```python
from dataknobs_fsm import SimpleFSM

fsm = SimpleFSM(config)

# Process single item
result = fsm.process(
    data={'input': 'hello'},
    initial_state=None,  # Optional: override start state
    timeout=30.0  # Optional: timeout in seconds
)

# Result structure
print(result['success'])      # True/False
print(result['final_state'])  # Name of final state reached
print(result['data'])         # Transformed data
print(result['path'])         # List of states traversed
print(result['error'])        # Error message if failed
```

### process_batch()

Process multiple items in parallel batches.

```python
# Process batch of items
items = [
    {'id': 1, 'value': 10},
    {'id': 2, 'value': 20},
    {'id': 3, 'value': 30}
]

results = fsm.process_batch(
    data=items,
    batch_size=10,      # Items per batch
    max_workers=4,      # Parallel workers
    on_progress=None    # Optional progress callback
)

# Results is a list of result dicts
for result in results:
    print(f"ID {result['data']['id']}: Success={result['success']}")
```

### process_stream()

Process data from a stream source with memory efficiency.

```python
# Stream from file
stats = fsm.process_stream(
    source='input.jsonl',        # File path or iterable
    sink='output.jsonl',         # Output path (optional)
    chunk_size=100,              # Records per chunk
    on_progress=None,            # Progress callback
    input_format='auto',         # auto/json/jsonl/csv
    output_format='jsonl'        # json/jsonl/csv
)

print(f"Processed: {stats['total_processed']}")
print(f"Succeeded: {stats['total_succeeded']}")
print(f"Failed: {stats['total_failed']}")
print(f"Duration: {stats['duration']:.2f}s")

# Stream from generator
def data_generator():
    for i in range(1000):
        yield {'id': i, 'value': i * 10}

results = fsm.process_stream(source=data_generator())
```

## Configuration Structure

FSM configurations define networks, states, and arcs (transitions).

### Basic Configuration

```python
config = {
    'name': 'my_fsm',
    'main_network': 'main',
    'data_mode': 'copy',  # copy/reference/direct

    'networks': [{
        'name': 'main',

        'states': [
            {
                'name': 'start',
                'is_start': True,
                'schema': {  # Optional JSON schema validation
                    'type': 'object',
                    'properties': {
                        'input': {'type': 'string'}
                    },
                    'required': ['input']
                }
            },
            {
                'name': 'process',
                'functions': {
                    'transform': 'process_data',  # Function name
                    'validate': 'check_valid'     # Optional validation
                }
            },
            {
                'name': 'end',
                'is_end': True
            }
        ],

        'arcs': [
            {
                'from': 'start',
                'to': 'process',
                'name': 'begin',
                'pre_test': None,     # Optional condition function
                'transform': None,    # Optional data transformation
                'priority': 0         # Higher priority evaluated first
            },
            {
                'from': 'process',
                'to': 'end',
                'name': 'complete'
            }
        ]
    }]
}
```

### Conditional Transitions

Arcs can have conditional logic to determine which path to take.

```python
config = {
    'name': 'conditional_fsm',
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {'name': 'start', 'is_start': True},
            {'name': 'validate', 'functions': {'transform': 'validate_data'}},
            {'name': 'success', 'is_end': True},
            {'name': 'error', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'validate'},
            {
                'from': 'validate',
                'to': 'success',
                'pre_test': 'lambda state: state.data.get("valid", False)',
                'priority': 1  # Checked first
            },
            {
                'from': 'validate',
                'to': 'error',
                'pre_test': 'lambda state: not state.data.get("valid", False)',
                'priority': 0  # Checked second
            }
        ]
    }]
}
```

### Multiple Networks

FSMs can contain multiple state networks for complex workflows.

```python
config = {
    'name': 'multi_network_fsm',
    'main_network': 'main',
    'networks': [
        {
            'name': 'main',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'process_sub', 'push_to': 'validation'},  # Push to sub-network
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'process_sub'},
                {'from': 'process_sub', 'to': 'end'}
            ]
        },
        {
            'name': 'validation',
            'states': [
                {'name': 'val_start', 'is_start': True},
                {'name': 'check', 'functions': {'transform': 'validate'}},
                {'name': 'val_end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'val_start', 'to': 'check'},
                {'from': 'check', 'to': 'val_end'}
            ]
        }
    ]
}
```

## State Functions

States can have functions for validation and transformation.

### Transform Functions

Transform functions modify state data.

```python
# Using lambda in config
config = {
    'states': [{
        'name': 'process',
        'functions': {
            'transform': 'lambda state: {"result": state.data["value"] * 2}'
        }
    }]
}

# Using registered function
def double_value(state):
    """Double the input value."""
    return {'result': state.data['value'] * 2}

fsm = SimpleFSM(config)
fsm.register_function('double_value', double_value)
```

### Validation Functions

Validation functions check data validity.

```python
def validate_positive(state):
    """Validate that value is positive."""
    if state.data.get('value', 0) <= 0:
        raise ValueError("Value must be positive")
    return True

config = {
    'states': [{
        'name': 'validate',
        'functions': {
            'validate': 'validate_positive'
        }
    }]
}

fsm = SimpleFSM(config)
fsm.register_function('validate_positive', validate_positive)
```

## Resource Management

Resources (databases, connections, etc.) can be managed by the FSM.

```python
config = {
    'name': 'resource_fsm',
    'resources': {
        'database': {
            'type': 'postgres',
            'connection_string': 'postgresql://localhost/mydb'
        },
        'api_client': {
            'type': 'http',
            'base_url': 'https://api.example.com'
        }
    },
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {'name': 'start', 'is_start': True},
            {
                'name': 'fetch_data',
                'functions': {'transform': 'fetch_from_db'},
                'required_resources': ['database']
            },
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'fetch_data'},
            {'from': 'fetch_data', 'to': 'end'}
        ]
    }]
}

def fetch_from_db(state, resources):
    """Fetch data using database resource."""
    db = resources['database']
    # Use db connection
    return {'data': 'fetched from db'}

fsm = SimpleFSM(config)
fsm.register_function('fetch_from_db', fetch_from_db)
```

## Full Example

Complete ETL pipeline example:

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Define workflow
config = {
    'name': 'data_etl',
    'data_mode': DataHandlingMode.COPY,
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {
                'name': 'extract',
                'is_start': True,
                'functions': {'transform': 'extract_data'}
            },
            {
                'name': 'transform',
                'functions': {'transform': 'transform_data'}
            },
            {
                'name': 'validate',
                'functions': {
                    'validate': 'check_valid',
                    'transform': 'validate_data'
                }
            },
            {
                'name': 'load',
                'functions': {'transform': 'load_data'}
            },
            {'name': 'success', 'is_end': True},
            {'name': 'error', 'is_end': True}
        ],
        'arcs': [
            {'from': 'extract', 'to': 'transform'},
            {'from': 'transform', 'to': 'validate'},
            {
                'from': 'validate',
                'to': 'load',
                'pre_test': 'lambda state: state.data.get("valid", False)',
                'priority': 1
            },
            {
                'from': 'validate',
                'to': 'error',
                'pre_test': 'lambda state: not state.data.get("valid", False)',
                'priority': 0
            },
            {'from': 'load', 'to': 'success'}
        ]
    }]
}

# Define functions
def extract_data(state):
    """Extract data from source."""
    return {
        'source': 'file.csv',
        'records': ['record1', 'record2', 'record3']
    }

def transform_data(state):
    """Transform extracted data."""
    records = state.data.get('records', [])
    transformed = [r.upper() for r in records]
    return {
        **state.data,
        'transformed': transformed
    }

def check_valid(state):
    """Validate data."""
    return len(state.data.get('transformed', [])) > 0

def validate_data(state):
    """Mark data as validated."""
    return {
        **state.data,
        'valid': len(state.data.get('transformed', [])) > 0,
        'validated_at': '2025-01-01'
    }

def load_data(state):
    """Load data to destination."""
    count = len(state.data.get('transformed', []))
    return {
        **state.data,
        'loaded': count,
        'status': 'complete'
    }

# Create FSM and register functions
fsm = SimpleFSM(config)
fsm.register_function('extract_data', extract_data)
fsm.register_function('transform_data', transform_data)
fsm.register_function('check_valid', check_valid)
fsm.register_function('validate_data', validate_data)
fsm.register_function('load_data', load_data)

# Process single record
result = fsm.process({})
print(f"Success: {result['success']}")
print(f"Final state: {result['final_state']}")
print(f"Data: {result['data']}")

# Process batch
batch = [{'id': i} for i in range(10)]
results = fsm.process_batch(batch, batch_size=5, max_workers=2)
print(f"Processed {len(results)} items")

# Process stream
stats = fsm.process_stream(
    source='input.jsonl',
    sink='output.jsonl',
    chunk_size=100
)
print(f"Stream processed: {stats['total_processed']} items in {stats['duration']:.2f}s")
```

## Advanced Features

### FSMBuilder

Programmatic FSM construction:

```python
from dataknobs_fsm import FSMBuilder
from dataknobs_fsm.core.data_modes import DataHandlingMode

builder = FSMBuilder(name='programmatic_fsm')
builder.set_data_mode(DataHandlingMode.COPY)

# Add network
network = builder.add_network('main', is_main=True)

# Add states
network.add_state('start', is_start=True)
network.add_state('process', transform='process_func')
network.add_state('end', is_end=True)

# Add arcs
network.add_arc('start', 'process')
network.add_arc('process', 'end')

# Build FSM
fsm_config = builder.build()
fsm = SimpleFSM(fsm_config)
```

### Configuration Loader

Load FSM from files:

```python
from dataknobs_fsm import ConfigLoader, SimpleFSM

# Load from YAML
loader = ConfigLoader()
config = loader.load('workflow.yaml')
fsm = SimpleFSM(config)

# Load from JSON
config = loader.load('workflow.json')
fsm = SimpleFSM(config)
```

### Execution Context

Access execution context in functions:

```python
def my_transform(state, context):
    """Transform with access to execution context."""
    # Access execution metadata
    current_state = context.current_state
    history = context.history

    # Modify data
    return {
        **state.data,
        'processed_by': current_state.name,
        'step_count': len(history)
    }
```

### Debugging

Use AdvancedFSM for debugging:

```python
from dataknobs_fsm import AdvancedFSM, FSMDebugger

fsm = AdvancedFSM(config)
debugger = FSMDebugger(fsm)

# Set breakpoints
debugger.add_breakpoint('transform')
debugger.add_breakpoint('validate')

# Step through execution
for step in debugger.step_through({'value': 42}):
    print(f"State: {step.state_name}")
    print(f"Data: {step.data}")
    print(f"Timing: {step.duration}ms")

    # Inspect state
    if step.state_name == 'transform':
        print(f"Before: {step.before_data}")
        print(f"After: {step.after_data}")
```

## Error Handling

FSM provides comprehensive error handling:

```python
from dataknobs_fsm import SimpleFSM

fsm = SimpleFSM(config)

# Process with error handling
result = fsm.process({'value': 'invalid'})

if not result['success']:
    print(f"Error: {result['error']}")
    print(f"Failed at state: {result['final_state']}")
    print(f"Path taken: {result['path']}")

    # Access error details
    if 'error_details' in result:
        print(f"Details: {result['error_details']}")
```

## Async Usage

Use AsyncSimpleFSM for async workflows:

```python
from dataknobs_fsm import AsyncSimpleFSM

async def main():
    fsm = AsyncSimpleFSM(config)

    # Async process
    result = await fsm.process({'value': 42})

    # Async batch
    results = await fsm.process_batch(items, max_workers=10)

    # Async stream
    async for chunk in fsm.process_stream(source):
        print(f"Processed chunk: {chunk}")

# Run
import asyncio
asyncio.run(main())
```

## Best Practices

1. **Choose the Right Data Mode**
   - Use COPY for production systems
   - Use REFERENCE for large datasets
   - Use DIRECT only for single-threaded, performance-critical paths

2. **Use Validation**
   - Add JSON schemas to start states
   - Use validation functions for business logic
   - Handle validation errors gracefully

3. **Batch Processing**
   - Use `process_batch()` for high throughput
   - Tune `batch_size` and `max_workers` for your workload
   - Use `on_progress` callback for long-running batches

4. **Stream Processing**
   - Use `process_stream()` for large files
   - Set appropriate `chunk_size` for memory efficiency
   - Handle partial failures in stream processing

5. **Testing**
   - Use SimpleFSM for quick prototyping
   - Use AdvancedFSM with debugger for testing
   - Test error paths and edge cases

6. **Resource Management**
   - Define resources in configuration
   - Specify required resources per state
   - Clean up resources in error handlers

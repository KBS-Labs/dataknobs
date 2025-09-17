# FSM Configuration Guide

This guide provides comprehensive documentation for creating FSM configurations in the DataKnobs FSM framework.

## Table of Contents
1. [Basic Structure](#basic-structure)
2. [States](#states)
3. [Arcs (Transitions)](#arcs-transitions)
4. [Networks](#networks)
5. [Functions and Transforms](#functions-and-transforms)
6. [Data Modes](#data-modes)
7. [Resources](#resources)
8. [Complete Examples](#complete-examples)

## Basic Structure

Every FSM configuration must have the following top-level structure:

```python
config = {
    "name": "MyFSM",                    # Required: Name of the FSM
    "main_network": "main",              # Required: Name of the main network to execute
    "networks": [                        # Required: List of network definitions
        {
            "name": "main",              # Network name (must match main_network)
            "states": [...],             # List of state definitions
            "arcs": [...]                # List of arc (transition) definitions
        }
    ]
}
```

## States

States are the nodes in your FSM graph. Each state has a name and metadata that defines its behavior.

### State Properties

```python
{
    "name": "state_name",           # Required: Unique name within the network
    "is_start": True,               # Optional: Marks this as an initial state (default: False)
    "is_end": True,                 # Optional: Marks this as a final state (default: False)
    "functions": {                  # Optional: Functions to execute in this state
        "transform": ...,           # Transform function (see Functions section)
        "validate": ...             # Validation function
    },
    "schema": {                     # Optional: JSON schema for data validation
        "type": "object",
        "properties": {...},
        "required": [...]
    },
    "metadata": {...}               # Optional: Additional metadata
}
```

### State Types

1. **Initial States** (`is_start: True`)
   - Entry points for the FSM
   - At least one required per network
   - Multiple initial states allowed (for different entry scenarios)

2. **Final States** (`is_end: True`)
   - Terminal states where execution ends
   - No outgoing arcs allowed
   - Can have transform functions for final data processing

3. **Normal States** (neither start nor end)
   - Intermediate processing states
   - Must have at least one incoming and one outgoing arc

### Example States

```python
"states": [
    {
        "name": "input",
        "is_start": True,
        "schema": {
            "type": "object",
            "properties": {
                "data": {"type": "string"}
            },
            "required": ["data"]
        }
    },
    {
        "name": "process",
        "functions": {
            "transform": {
                "type": "inline",
                "code": "lambda state: {'processed': state.data['data'].upper()}"
            }
        }
    },
    {
        "name": "output",
        "is_end": True
    }
]
```

## Arcs (Transitions)

Arcs define the transitions between states and the conditions under which they occur.

### Arc Properties

```python
{
    "from": "source_state",         # Required: Source state name
    "to": "target_state",           # Required: Target state name
    "name": "arc_name",             # Optional: Name for the arc
    "condition": {...},             # Optional: Condition function
    "transform": {...},             # Optional: Transform function
    "priority": 0,                  # Optional: Priority for arc selection (higher = higher priority)
    "metadata": {...}               # Optional: Additional metadata
}
```

### Conditional Transitions

Arcs can have conditions that determine whether the transition should be taken:

```python
{
    "from": "validate",
    "to": "process",
    "condition": {
        "type": "inline",
        "code": "lambda state: state.data.get('valid', False)"
    }
}
```

### Transform on Transition

Data can be transformed during transitions:

```python
{
    "from": "extract",
    "to": "transform",
    "transform": {
        "type": "inline",
        "code": "lambda state: {'records': [r.upper() for r in state.data['records']]}"
    }
}
```

## Networks

Networks are collections of states and arcs that define a complete FSM or sub-FSM.

### Network Properties

```python
{
    "name": "network_name",         # Required: Unique network name
    "states": [...],                # Required: List of states
    "arcs": [...],                  # Required: List of arcs
    "resources": [...],             # Optional: Resource names used by this network
    "streaming": {...},             # Optional: Streaming configuration
    "metadata": {...}               # Optional: Additional metadata
}
```

### Multi-Network FSMs

FSMs can have multiple networks for different processing paths:

```python
{
    "name": "MultiNetworkFSM",
    "main_network": "batch",
    "networks": [
        {
            "name": "batch",
            "states": [...],
            "arcs": [...]
        },
        {
            "name": "stream",
            "states": [...],
            "arcs": [...]
        }
    ]
}
```

## Functions and Transforms

Functions define the processing logic for states and transitions.

### Function Types

1. **Inline Functions** - Lambda expressions or simple Python code
```python
{
    "type": "inline",
    "code": "lambda state: {'result': state.data['value'] * 2}"
}
```

2. **Registered Functions** - Pre-registered Python functions
```python
{
    "type": "registered",
    "name": "process_data"
}
```

3. **Built-in Functions** - Framework-provided functions
```python
{
    "type": "builtin",
    "name": "validate_json",
    "params": {"schema": {...}}
}
```

4. **Custom Functions** - Functions from external modules
```python
{
    "type": "custom",
    "module": "my_module",
    "name": "my_function"
}
```

### State Functions

States can have multiple function types:

```python
{
    "name": "process",
    "functions": {
        "validate": {                # Runs first to validate data
            "type": "inline",
            "code": "lambda state: state.data.get('value') > 0"
        },
        "transform": {               # Transforms the data
            "type": "registered",
            "name": "process_record"
        }
    }
}
```

## Data Modes

Data modes control how data is handled during state transitions.

### Available Modes

1. **COPY** (default) - Creates copies of data for each state
   - Safe for rollback
   - Higher memory usage
   - Best for transactional workflows

2. **REFERENCE** - Passes references to the same data object
   - Memory efficient
   - Changes affect all states
   - Best for read-only or streaming workflows

3. **DIRECT** - Direct manipulation without copying
   - Most efficient
   - No rollback capability
   - Best for simple transformations

### Configuration

```python
{
    "name": "MyFSM",
    "data_mode": {
        "default": "copy",           # Default mode for all states
        "state_overrides": {         # Override for specific states
            "stream_state": "reference"
        }
    },
    "networks": [...]
}
```

## Resources

Resources are external dependencies that states can use (databases, APIs, file systems, etc.).

### Resource Configuration

```python
{
    "name": "MyFSM",
    "resources": [
        {
            "name": "database",
            "type": "database",
            "config": {
                "connection_string": "sqlite:///data.db",
                "pool_size": 10
            }
        },
        {
            "name": "api",
            "type": "http",
            "config": {
                "base_url": "https://api.example.com",
                "timeout": 30
            }
        }
    ],
    "networks": [...]
}
```

## Complete Examples

### Example 1: Simple Data Processing Pipeline

```python
simple_pipeline = {
    "name": "SimpleDataPipeline",
    "main_network": "main",
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "input",
                "is_start": True
            },
            {
                "name": "validate",
                "functions": {
                    "validate": {
                        "type": "inline",
                        "code": "lambda state: 'data' in state.data"
                    }
                }
            },
            {
                "name": "transform",
                "functions": {
                    "transform": {
                        "type": "inline",
                        "code": "lambda state: {'result': state.data['data'].upper()}"
                    }
                }
            },
            {
                "name": "output",
                "is_end": True
            },
            {
                "name": "error",
                "is_end": True
            }
        ],
        "arcs": [
            {"from": "input", "to": "validate"},
            {
                "from": "validate",
                "to": "transform",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: state.data.get('valid', True)"
                }
            },
            {
                "from": "validate",
                "to": "error",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: not state.data.get('valid', True)"
                }
            },
            {"from": "transform", "to": "output"}
        ]
    }]
}
```

### Example 2: ETL Pipeline with Error Handling

```python
etl_pipeline = {
    "name": "ETLPipeline",
    "main_network": "main",
    "data_mode": {
        "default": "copy"  # Use COPY for transaction safety
    },
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "start",
                "is_start": True
            },
            {
                "name": "extract",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "extract_from_source"
                    }
                }
            },
            {
                "name": "validate",
                "functions": {
                    "validate": {
                        "type": "registered",
                        "name": "validate_records"
                    }
                }
            },
            {
                "name": "transform",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "transform_records"
                    }
                }
            },
            {
                "name": "load",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "load_to_target"
                    }
                }
            },
            {
                "name": "success",
                "is_end": True
            },
            {
                "name": "failure",
                "is_end": True,
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "rollback_changes"
                    }
                }
            }
        ],
        "arcs": [
            {"from": "start", "to": "extract"},
            {"from": "extract", "to": "validate"},
            {
                "from": "validate",
                "to": "transform",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: state.data.get('validation_passed', False)"
                }
            },
            {
                "from": "validate",
                "to": "failure",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: not state.data.get('validation_passed', False)"
                }
            },
            {"from": "transform", "to": "load"},
            {
                "from": "load",
                "to": "success",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: state.data.get('load_successful', False)"
                }
            },
            {
                "from": "load",
                "to": "failure",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: not state.data.get('load_successful', False)"
                }
            }
        ]
    }]
}
```

### Example 3: Streaming Pipeline

```python
streaming_pipeline = {
    "name": "StreamingPipeline",
    "main_network": "main",
    "data_mode": {
        "default": "reference"  # Use REFERENCE for memory efficiency
    },
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "input",
                "is_start": True
            },
            {
                "name": "filter",
                "functions": {
                    "transform": {
                        "type": "inline",
                        "code": "lambda state: {'records': [r for r in state.data.get('records', []) if r.get('active')]}"
                    }
                }
            },
            {
                "name": "enrich",
                "functions": {
                    "transform": {
                        "type": "inline",
                        "code": "lambda state: {'records': [{**r, 'timestamp': __import__('time').time()} for r in state.data.get('records', [])]}"
                    }
                }
            },
            {
                "name": "output",
                "is_end": True
            }
        ],
        "arcs": [
            {"from": "input", "to": "filter"},
            {"from": "filter", "to": "enrich"},
            {"from": "enrich", "to": "output"}
        ],
        "streaming": {
            "enabled": True,
            "chunk_size": 100,
            "parallelism": 4
        }
    }]
}
```

## Best Practices

1. **Always define at least one initial state** (`is_start: True`)
2. **Always define at least one final state** (`is_end: True`)
3. **Use meaningful state and arc names** for better debugging
4. **Validate data early** in the pipeline to catch errors sooner
5. **Use COPY mode** for transactional workflows that need rollback
6. **Use REFERENCE mode** for streaming or read-only workflows
7. **Register functions** instead of using inline code for complex logic
8. **Add error states** to handle failures gracefully
9. **Use conditions** to control flow based on data
10. **Document your FSM** with metadata and descriptions

## Common Patterns

### Pattern 1: Validation → Process → Output
```python
"arcs": [
    {"from": "input", "to": "validate"},
    {"from": "validate", "to": "process", "condition": "valid"},
    {"from": "validate", "to": "error", "condition": "not valid"},
    {"from": "process", "to": "output"}
]
```

### Pattern 2: Retry with Backoff
```python
"states": [
    {"name": "attempt", ...},
    {"name": "retry_wait", ...},
    {"name": "success", "is_end": True},
    {"name": "failure", "is_end": True}
],
"arcs": [
    {"from": "attempt", "to": "success", "condition": "succeeded"},
    {"from": "attempt", "to": "retry_wait", "condition": "failed and retries < max"},
    {"from": "attempt", "to": "failure", "condition": "failed and retries >= max"},
    {"from": "retry_wait", "to": "attempt"}
]
```

### Pattern 3: Parallel Processing (Multiple Networks)
```python
{
    "networks": [
        {"name": "batch", "states": [...], "arcs": [...]},
        {"name": "stream", "states": [...], "arcs": [...]},
        {"name": "real_time", "states": [...], "arcs": [...]}
    ]
}
```

## Troubleshooting

### Common Errors

1. **"Network must have at least one start state"**
   - Solution: Add `"is_start": True` to at least one state

2. **"Arc target 'X' not found in network"**
   - Solution: Ensure all arc targets reference existing state names

3. **"Main network 'X' not found"**
   - Solution: Ensure `main_network` matches a network name in `networks`

4. **Function execution errors**
   - Check lambda syntax
   - Ensure registered functions are properly registered before FSM creation
   - Verify state.data structure matches what functions expect

### Debugging Tips

1. Use the FSM debugger to step through execution
2. Add logging in transform functions
3. Use metadata to track state transitions
4. Start with simple configurations and gradually add complexity
5. Test individual functions before integrating into FSM

## Migration from Other Formats

If you're migrating from other FSM formats, note these key differences:

1. States and arcs are defined separately (not nested)
2. Use `is_start` and `is_end` flags instead of special state types
3. Functions are defined with type specifications
4. Networks allow for modular FSM composition
5. Data modes provide fine-grained control over data handling
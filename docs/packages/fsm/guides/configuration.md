# FSM Configuration Guide

This guide provides comprehensive documentation for creating and understanding FSM configurations in the DataKnobs FSM framework.

!!! note "Complete Configuration Reference"
    For the full configuration guide with all options and examples, see the FSM package documentation.

## Quick Reference

### Basic Structure

Every FSM configuration requires:

```python
config = {
    "name": "MyFSM",                    # FSM name
    "main_network": "main",              # Main network to execute
    "networks": [                        # List of networks
        {
            "name": "main",
            "states": [...],             # State definitions
            "arcs": [...]                # Transition definitions
        }
    ]
}
```

### State Definition

States are the nodes in your FSM:

```python
{
    "name": "state_name",
    "is_start": True,        # Initial state flag
    "is_end": True,         # Final state flag
    "functions": {          # State functions
        "transform": {...},  # Data transformation
        "validate": {...}   # Data validation
    },
    "schema": {...}         # JSON schema for validation
}
```

### Arc (Transition) Definition

Arcs define transitions between states:

```python
{
    "from": "source_state",
    "to": "target_state",
    "condition": {...},     # Optional condition
    "transform": {...},     # Optional transformation
    "priority": 0          # Arc priority
}
```

## Essential Concepts

### 1. States and Types

**Initial States** (`is_start: True`)
- Entry points for FSM execution
- At least one required per network

**Final States** (`is_end: True`)
- Terminal states where execution ends
- No outgoing arcs allowed

**Normal States**
- Intermediate processing states
- Must have incoming and outgoing arcs

### 2. Functions

Functions define processing logic:

```python
# Inline function
{
    "type": "inline",
    "code": "lambda state: {'result': state.data['value'] * 2}"
}

# Registered function
{
    "type": "registered",
    "name": "process_data"
}

# Built-in function
{
    "type": "builtin",
    "name": "validate_json",
    "params": {"schema": {...}}
}
```

### 3. Data Modes

Control how data flows through states:

- **COPY** (default): Safe for transactions, higher memory
- **REFERENCE**: Memory efficient, shared data
- **DIRECT**: Most efficient, no rollback

```python
{
    "data_mode": {
        "default": "copy",
        "state_overrides": {
            "stream_state": "reference"
        }
    }
}
```

## Common Patterns

### Validation → Process → Output

```python
{
    "states": [
        {"name": "input", "is_start": True},
        {"name": "validate", "functions": {"validate": {...}}},
        {"name": "process", "functions": {"transform": {...}}},
        {"name": "output", "is_end": True},
        {"name": "error", "is_end": True}
    ],
    "arcs": [
        {"from": "input", "to": "validate"},
        {"from": "validate", "to": "process", "condition": "valid"},
        {"from": "validate", "to": "error", "condition": "not valid"},
        {"from": "process", "to": "output"}
    ]
}
```

### Retry with Backoff

```python
{
    "states": [
        {"name": "attempt"},
        {"name": "retry_wait"},
        {"name": "success", "is_end": True},
        {"name": "failure", "is_end": True}
    ],
    "arcs": [
        {"from": "attempt", "to": "success", "condition": "succeeded"},
        {"from": "attempt", "to": "retry_wait", "condition": "retry_needed"},
        {"from": "attempt", "to": "failure", "condition": "max_retries"},
        {"from": "retry_wait", "to": "attempt"}
    ]
}
```

## Examples

### Simple Data Pipeline

```python
simple_pipeline = {
    "name": "SimpleDataPipeline",
    "main_network": "main",
    "networks": [{
        "name": "main",
        "states": [
            {"name": "input", "is_start": True},
            {
                "name": "transform",
                "functions": {
                    "transform": {
                        "type": "inline",
                        "code": "lambda state: {'result': state.data['data'].upper()}"
                    }
                }
            },
            {"name": "output", "is_end": True}
        ],
        "arcs": [
            {"from": "input", "to": "transform"},
            {"from": "transform", "to": "output"}
        ]
    }]
}
```

### ETL Pipeline

```python
etl_pipeline = {
    "name": "ETLPipeline",
    "main_network": "main",
    "data_mode": {"default": "copy"},  # Transaction safety
    "networks": [{
        "name": "main",
        "states": [
            {"name": "start", "is_start": True},
            {"name": "extract", "functions": {"transform": {...}}},
            {"name": "validate", "functions": {"validate": {...}}},
            {"name": "transform", "functions": {"transform": {...}}},
            {"name": "load", "functions": {"transform": {...}}},
            {"name": "success", "is_end": True},
            {"name": "failure", "is_end": True}
        ],
        "arcs": [
            {"from": "start", "to": "extract"},
            {"from": "extract", "to": "validate"},
            {"from": "validate", "to": "transform", "condition": "valid"},
            {"from": "validate", "to": "failure", "condition": "not valid"},
            {"from": "transform", "to": "load"},
            {"from": "load", "to": "success", "condition": "success"},
            {"from": "load", "to": "failure", "condition": "failed"}
        ]
    }]
}
```

## Best Practices

1. **Always define initial and final states**
2. **Use meaningful state and arc names**
3. **Validate data early in the pipeline**
4. **Choose appropriate data modes**:
   - COPY for transactional workflows
   - REFERENCE for streaming/read-only
   - DIRECT for simple transformations
5. **Register complex functions** instead of inline code
6. **Add error states** for graceful failure handling
7. **Use conditions** to control flow
8. **Document with metadata**

## Troubleshooting

### Common Errors

| Error | Solution |
|-------|----------|
| "Network must have at least one start state" | Add `"is_start": True` to a state |
| "Arc target 'X' not found in network" | Ensure arc targets exist |
| "Main network 'X' not found" | Check `main_network` name matches |
| Function execution errors | Verify lambda syntax and data structure |

### Debugging Tips

1. Use FSM debugger to step through execution
2. Add logging in transform functions
3. Start simple and add complexity gradually
4. Test functions independently first

## Related Examples

- [End-to-End Streaming](../examples/end-to-end-streaming.md) - Streaming data through FSM
- [Database ETL](../examples/database-etl.md) - ETL pipeline pattern
- [File Processing](../examples/file-processor.md) - File transformation workflows
- [LLM Conversation](../examples/llm-conversation.md) - Conversational AI patterns

## API References

- [SimpleFSM API](../api/simple.md) - Simple synchronous API
- [AsyncSimpleFSM API](../api/async_simple.md) - Async API for streaming
- [AdvancedFSM API](../api/advanced.md) - Advanced features and debugging

## Full Documentation

For complete details including:
- All configuration options
- Advanced patterns
- Migration guides
- Network composition
- Resource management
- Streaming configuration

See the [API documentation](../../../api/dataknobs-fsm.md) for complete details.
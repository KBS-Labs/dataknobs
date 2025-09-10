# SimpleFSM API Reference

The `SimpleFSM` class provides a high-level, user-friendly interface for creating and executing finite state machines.

## Class Definition

```python
from dataknobs_fsm import SimpleFSM

class SimpleFSM:
    """High-level API for creating and running finite state machines."""
```

## Constructor

```python
SimpleFSM(
    name: str = "simple_fsm",
    description: str = "",
    metadata: Dict[str, Any] = None
)
```

**Parameters:**
- `name`: Name of the FSM (default: "simple_fsm")
- `description`: Optional description
- `metadata`: Additional metadata dictionary

**Example:**
```python
fsm = SimpleFSM(
    name="order_processor",
    description="Processes customer orders",
    metadata={"version": "1.0", "author": "team"}
)
```

## Core Methods

### add_state

```python
add_state(
    name: str,
    initial: bool = False,
    terminal: bool = False,
    metadata: Dict[str, Any] = None
) -> None
```

Add a state to the FSM.

**Parameters:**
- `name`: Unique state name
- `initial`: Whether this is the initial state (only one allowed)
- `terminal`: Whether this is a terminal state (multiple allowed)
- `metadata`: Additional state metadata

**Example:**
```python
fsm.add_state("start", initial=True)
fsm.add_state("process", metadata={"timeout": 30})
fsm.add_state("complete", terminal=True)
```

### add_transition

```python
add_transition(
    from_state: str,
    to_state: str,
    function: Optional[Callable] = None,
    condition: Optional[Callable] = None,
    on_error: Optional[str] = None,
    metadata: Dict[str, Any] = None
) -> None
```

Add a transition between states.

**Parameters:**
- `from_state`: Source state name
- `to_state`: Target state name
- `function`: Processing function (sync or async)
- `condition`: Guard condition function
- `on_error`: Error state to transition to on failure
- `metadata`: Additional transition metadata

**Example:**
```python
def process_data(data):
    return {**data, "processed": True}

def is_valid(data):
    return data.get("value", 0) > 0

fsm.add_transition(
    "start", "process",
    function=process_data,
    condition=is_valid,
    on_error="error_state"
)
```

### add_resource

```python
add_resource(
    name: str,
    config: Dict[str, Any],
    shared: bool = False
) -> None
```

Add a resource to the FSM.

**Parameters:**
- `name`: Resource identifier
- `config`: Resource configuration including `type` and provider details
- `shared`: Whether the resource is shared across executions

**Supported Resource Types:**
- `database`: Database connections
- `filesystem`: File system access
- `http`: HTTP client
- `llm`: LLM providers
- `cache`: Caching systems

**Example:**
```python
fsm.add_resource("db", {
    "type": "database",
    "provider": "sqlite",
    "config": {"database": "app.db"}
})

fsm.add_resource("api", {
    "type": "http",
    "config": {
        "base_url": "https://api.example.com",
        "timeout": 30
    }
})
```

## Execution Methods

### run

```python
run(
    input_data: Dict[str, Any] = None,
    max_steps: int = 100,
    max_retries: int = 0,
    timeout: Optional[float] = None
) -> Dict[str, Any]
```

Execute the FSM synchronously.

**Parameters:**
- `input_data`: Initial data dictionary
- `max_steps`: Maximum execution steps (prevents infinite loops)
- `max_retries`: Number of retries on error
- `timeout`: Overall execution timeout in seconds

**Returns:** Final data dictionary after execution

**Example:**
```python
result = fsm.run(
    {"order_id": 123, "amount": 99.99},
    max_steps=50,
    max_retries=3,
    timeout=60.0
)
```

### run_async

```python
async run_async(
    input_data: Dict[str, Any] = None,
    max_steps: int = 100,
    max_retries: int = 0,
    timeout: Optional[float] = None
) -> Dict[str, Any]
```

Execute the FSM asynchronously.

**Parameters:** Same as `run()`

**Example:**
```python
import asyncio

async def main():
    result = await fsm.run_async({"data": "value"})
    print(result)

asyncio.run(main())
```

### run_batch

```python
run_batch(
    items: List[Dict[str, Any]],
    max_workers: int = 4,
    max_retries: int = 0,
    stop_on_error: bool = False
) -> List[Dict[str, Any]]
```

Process multiple items in parallel.

**Parameters:**
- `items`: List of input data dictionaries
- `max_workers`: Number of parallel workers
- `max_retries`: Retries per item
- `stop_on_error`: Stop processing on first error

**Returns:** List of results in the same order as inputs

**Example:**
```python
items = [
    {"id": 1, "value": 10},
    {"id": 2, "value": 20},
    {"id": 3, "value": 30}
]

results = fsm.run_batch(items, max_workers=3)
```

## Configuration Methods

### from_config

```python
@classmethod
from_config(
    config_path: str,
    overrides: Dict[str, Any] = None
) -> SimpleFSM
```

Create an FSM from a configuration file.

**Parameters:**
- `config_path`: Path to YAML or JSON configuration
- `overrides`: Override configuration values

**Example:**
```python
fsm = SimpleFSM.from_config("workflow.yaml")

# With overrides
fsm = SimpleFSM.from_config(
    "workflow.yaml",
    overrides={"timeout": 120}
)
```

### to_config

```python
to_config() -> Dict[str, Any]
```

Export FSM as configuration dictionary.

**Example:**
```python
config = fsm.to_config()
print(config)
```

### save_config

```python
save_config(path: str) -> None
```

Save FSM configuration to file.

**Example:**
```python
fsm.save_config("my_fsm.yaml")
```

## Utility Methods

### validate

```python
validate() -> List[str]
```

Validate FSM structure and return any issues.

**Returns:** List of validation errors (empty if valid)

**Example:**
```python
errors = fsm.validate()
if errors:
    print("Validation errors:", errors)
else:
    print("FSM is valid")
```

### visualize

```python
visualize(
    output_path: str = None,
    format: str = "png"
) -> str
```

Generate a visual representation of the FSM.

**Parameters:**
- `output_path`: Save location (optional)
- `format`: Output format (png, svg, pdf, dot)

**Returns:** Path to generated file or DOT string

**Example:**
```python
# Save as PNG
fsm.visualize("workflow.png")

# Get DOT format
dot_string = fsm.visualize(format="dot")
```

### get_state

```python
get_state(name: str) -> State
```

Get a state by name.

**Example:**
```python
state = fsm.get_state("process")
print(state.metadata)
```

### get_transitions

```python
get_transitions(from_state: str = None) -> List[Arc]
```

Get transitions, optionally filtered by source state.

**Example:**
```python
# All transitions
all_transitions = fsm.get_transitions()

# From specific state
from_start = fsm.get_transitions("start")
```

## Properties

### states

```python
@property
states() -> List[str]
```

Get list of all state names.

### initial_state

```python
@property
initial_state() -> str
```

Get the initial state name.

### terminal_states

```python
@property
terminal_states() -> List[str]
```

Get list of terminal state names.

### resources

```python
@property
resources() -> Dict[str, ResourceProvider]
```

Get configured resources.

## Complete Example

```python
from dataknobs_fsm import SimpleFSM

# Create FSM
fsm = SimpleFSM(name="order_workflow")

# Define states
fsm.add_state("received", initial=True)
fsm.add_state("validated")
fsm.add_state("processed")
fsm.add_state("shipped", terminal=True)
fsm.add_state("cancelled", terminal=True)

# Add resource
fsm.add_resource("db", {
    "type": "database",
    "provider": "sqlite",
    "config": {"database": "orders.db"}
})

# Define processing functions
def validate_order(data):
    if data.get("amount", 0) <= 0:
        raise ValueError("Invalid amount")
    return {**data, "valid": True}

async def process_payment(data, resources):
    db = resources["db"]
    # Process payment logic
    return {**data, "paid": True}

def ship_order(data):
    return {**data, "tracking_number": "TRK123456"}

# Add transitions
fsm.add_transition(
    "received", "validated",
    function=validate_order,
    on_error="cancelled"
)

fsm.add_transition(
    "validated", "processed",
    function=process_payment
)

fsm.add_transition(
    "processed", "shipped",
    function=ship_order
)

# Validate and run
errors = fsm.validate()
if not errors:
    result = fsm.run({
        "order_id": "ORD-001",
        "amount": 99.99,
        "customer": "john@example.com"
    })
    print(f"Order shipped: {result['tracking_number']}")
```

## Error Handling

SimpleFSM provides automatic error handling:

1. **Transition Errors**: Use `on_error` parameter to specify error states
2. **Retries**: Use `max_retries` in execution methods
3. **Timeouts**: Set `timeout` to prevent hanging
4. **Validation**: Call `validate()` before execution

## Best Practices

1. **Always validate** FSMs before execution
2. **Use meaningful state names** that describe the workflow stage
3. **Keep functions pure** when possible (no side effects)
4. **Handle errors explicitly** with error states
5. **Use resources** for external dependencies
6. **Test thoroughly** with different input scenarios
7. **Document complex transitions** with metadata

## See Also

- [AdvancedFSM API](advanced.md) for more control
- [Examples](../examples/index.md) for real-world usage
- [Patterns](../patterns/index.md) for common scenarios
- [Configuration Guide](../guides/configuration.md) for YAML/JSON format
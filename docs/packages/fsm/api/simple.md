# SimpleFSM API Reference

The `SimpleFSM` class provides a simplified, configuration-driven interface for executing finite state machines. It abstracts away the complexity of configuration, resource management, and execution strategies.

> **📖 Also see:** [Auto-generated API Reference](../../../api/reference/fsm.md) - Complete documentation from source code docstrings

This page provides curated examples and usage patterns. The auto-generated reference provides exhaustive technical documentation with all methods, parameters, and type annotations.

---

## Class Definition

```python
from dataknobs_fsm.api.simple import SimpleFSM

class SimpleFSM:
    """Simplified FSM interface for common operations."""
```

## Constructor

```python
SimpleFSM(
    config: Union[str, Path, Dict[str, Any]],
    data_mode: DataHandlingMode = DataHandlingMode.COPY,
    resources: Dict[str, Any] | None = None,
    custom_functions: Dict[str, Callable] | None = None
)
```

**Parameters:**
- `config`: Path to YAML/JSON config file or config dictionary
- `data_mode`: Default data handling mode (COPY, REFERENCE, or DIRECT)
- `resources`: Optional resource configurations
- `custom_functions`: Optional custom functions to register

**Example:**
```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# From configuration file
fsm = SimpleFSM("workflow.yaml")

# From dictionary with custom functions
config = {
    "name": "order_processor",
    "states": [...],
    "arcs": [...]
}

fsm = SimpleFSM(
    config,
    data_mode=DataHandlingMode.COPY,
    custom_functions={"validate": validate_order}
)
```

## Processing Methods

### process

```python
process(
    data: Union[Dict[str, Any], Record],
    initial_state: str | None = None,
    timeout: float | None = None
) -> Dict[str, Any]
```

Process a single data record through the FSM synchronously.

**Parameters:**
- `data`: Input data to process
- `initial_state`: Optional starting state (defaults to FSM start state)
- `timeout`: Optional timeout in seconds

**Returns:** Dict containing:
- `final_state`: Name of the final state reached
- `data`: Processed data from final state
- `path`: List of states traversed
- `success`: Whether processing completed successfully
- `error`: Error message if processing failed (optional)

**Example:**
```python
result = fsm.process(
    {"order_id": 123, "amount": 99.99},
    timeout=60.0
)

if result["success"]:
    print(f"Final state: {result['final_state']}")
    print(f"Processed data: {result['data']}")
else:
    print(f"Error: {result['error']}")
```


### process_batch

```python
process_batch(
    data: List[Union[Dict[str, Any], Record]],
    batch_size: int = 10,
    max_workers: int = 4,
    on_progress: Union[Callable, None] = None
) -> List[Dict[str, Any]]
```

Process multiple records in parallel batches.

**Parameters:**
- `data`: List of input records to process
- `batch_size`: Number of records per batch
- `max_workers`: Maximum parallel workers
- `on_progress`: Optional callback for progress updates

**Returns:** List of results for each input record

**Example:**
```python
items = [
    {"id": 1, "value": 10},
    {"id": 2, "value": 20},
    {"id": 3, "value": 30}
]

results = fsm.process_batch(
    items,
    batch_size=10,
    max_workers=3
)

for result in results:
    print(f"ID {result['data']['id']}: {result['success']}")
```

## Streaming Methods

### process_stream

```python
async process_stream(
    source: AsyncIterator[Union[Dict[str, Any], Record]],
    sink: Optional[Callable] = None,
    buffer_size: int = 100
) -> AsyncIterator[Dict[str, Any]]
```

Process a stream of records asynchronously.

**Parameters:**
- `source`: Async iterator providing input records
- `sink`: Optional callback to handle each result
- `buffer_size`: Size of internal buffer for backpressure

**Returns:** Async iterator of results

**Example:**
```python
async def record_generator():
    for i in range(100):
        yield {"id": i, "value": i * 10}

async def main():
    async for result in fsm.process_stream(record_generator()):
        print(f"Processed: {result['data']}")

asyncio.run(main())
```

## Configuration Format

### YAML Configuration

```yaml
name: order_processor
states:
  - name: start
    is_start: true
  - name: validate
  - name: process
  - name: complete
    is_end: true

arcs:
  - from: start
    to: validate
    transform:
      type: registered
      name: validate_order
  - from: validate
    to: process
    pre_test:
      type: inline
      code: "lambda data, ctx: data.get('valid', False)"
  - from: process
    to: complete

resources:
  - name: db
    type: database
    provider: postgresql
    config:
      connection_string: ${DATABASE_URL}
```

### Dictionary Configuration

```python
config = {
    "name": "order_processor",
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
                "type": "registered",
                "name": "validate_order"
            }
        },
        {
            "from": "validate",
            "to": "process",
            "pre_test": {
                "type": "inline",
                "code": "lambda data, ctx: data.get('valid', False)"
            }
        },
        {"from": "process", "to": "complete"}
    ]
}
```

## Custom Functions

### Registering Custom Functions

Custom functions can be registered when creating the FSM:

```python
def validate_order(state):
    """Custom validation function."""
    data = state.data
    if data.get("amount", 0) <= 0:
        raise ValueError("Invalid amount")
    return {**data, "valid": True}

def calculate_tax(state):
    """Calculate tax on order."""
    data = state.data
    amount = data.get("amount", 0)
    tax = amount * 0.08  # 8% tax
    return {**data, "tax": tax, "total": amount + tax}

# Register functions
fsm = SimpleFSM(
    "order_workflow.yaml",
    custom_functions={
        "validate_order": validate_order,
        "calculate_tax": calculate_tax
    }
)
```

### Using Custom Functions in Configuration

```yaml
arcs:
  - from: start
    to: validate
    transform:
      type: registered  # Use registered function
      name: validate_order
  - from: validate
    to: calculate
    transform:
      type: registered
      name: calculate_tax
```

## Complete Example

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Define custom functions
def validate_order(state):
    """Validate order data."""
    data = state.data
    if data.get("amount", 0) <= 0:
        raise ValueError("Invalid amount")
    return {**data, "valid": True}

def process_payment(state):
    """Process payment."""
    data = state.data
    # Simulate payment processing
    return {**data, "paid": True, "transaction_id": "TXN123"}

def ship_order(state):
    """Ship the order."""
    data = state.data
    return {**data, "tracking_number": "TRK123456", "shipped": True}

# Define configuration
config = {
    "name": "order_workflow",
    "states": [
        {"name": "received", "initial": True},
        {"name": "validated"},
        {"name": "processed"},
        {"name": "shipped", "is_end": True},
        {"name": "cancelled", "is_end": True}
    ],
    "arcs": [
        {
            "from": "received",
            "to": "validated",
            "transform": {
                "type": "registered",
                "name": "validate_order"
            }
        },
        {
            "from": "validated",
            "to": "processed",
            "transform": {
                "type": "registered",
                "name": "process_payment"
            }
        },
        {
            "from": "processed",
            "to": "shipped",
            "transform": {
                "type": "registered",
                "name": "ship_order"
            }
        },
        {
            "from": "received",
            "to": "cancelled",
            "pre_test": {
                "type": "inline",
                "code": "lambda data, ctx: data.get('cancel_requested', False)"
            }
        }
    ],
    "resources": [
        {
            "name": "db",
            "type": "database",
            "provider": "sqlite",
            "config": {"database": "orders.db"}
        }
    ]
}

# Create FSM with custom functions
fsm = SimpleFSM(
    config,
    data_mode=DataHandlingMode.COPY,
    custom_functions={
        "validate_order": validate_order,
        "process_payment": process_payment,
        "ship_order": ship_order
    }
)

# Process an order
result = fsm.process({
    "order_id": "ORD-001",
    "amount": 99.99,
    "customer": "john@example.com"
})

if result["success"]:
    print(f"Order shipped: {result['data']['tracking_number']}")
    print(f"Transaction ID: {result['data']['transaction_id']}")
else:
    print(f"Order processing failed: {result.get('error')}")
```

## Error Handling

SimpleFSM provides error handling through:

1. **Try-Catch in Functions**: Handle errors within custom functions
2. **Result Status**: Check `success` field in results
3. **Timeouts**: Set `timeout` parameter in process methods
4. **Configuration Validation**: Errors during config loading are reported

## Best Practices

1. **Configuration-First**: Define FSMs in YAML/JSON for maintainability
2. **Use meaningful state names** that describe the workflow stage
3. **Keep functions pure** when possible (no side effects)
4. **Handle errors explicitly** in custom functions
5. **Use resources** for external dependencies
6. **Choose appropriate data mode**: COPY for safety, REFERENCE for large data, DIRECT for performance
7. **Test thoroughly** with different input scenarios

## See Also

- [AdvancedFSM API](advanced.md) for debugging and monitoring features
- [Examples](../examples/index.md) for real-world usage
- [Patterns](../patterns/index.md) for pre-built workflows
- [Data Modes Guide](../guides/data-modes.md) for understanding DataHandlingMode vs ProcessingMode
- [Quick Start](../quickstart.md) for getting started
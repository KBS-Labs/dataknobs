# AsyncSimpleFSM API Reference

The `AsyncSimpleFSM` class provides a native async/await interface for executing finite state machines. It is designed for use in async contexts and provides full asynchronous support for all operations.

## Class Definition

```python
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

class AsyncSimpleFSM:
    """Async-first FSM interface for processing data."""
```

## Constructor

```python
AsyncSimpleFSM(
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
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
import asyncio

# From configuration file
async def main():
    fsm = AsyncSimpleFSM("workflow.yaml")
    result = await fsm.process({"input": "data"})
    await fsm.close()

asyncio.run(main())

# From dictionary with custom functions
async def validate_order(state):
    """Async validation function."""
    await asyncio.sleep(0.1)  # Simulate async work
    data = state.data
    if data.get("amount", 0) <= 0:
        raise ValueError("Invalid amount")
    return {**data, "valid": True}

config = {
    "name": "order_processor",
    "states": [...],
    "arcs": [...]
}

async def main():
    fsm = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={"validate": validate_order}
    )
    result = await fsm.process({"amount": 100})
    await fsm.close()

asyncio.run(main())
```

## Async Processing Methods

### process

```python
async process(
    data: Union[Dict[str, Any], Record],
    initial_state: str | None = None,
    timeout: float | None = None
) -> Dict[str, Any]
```

Process a single data record through the FSM asynchronously.

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
async def main():
    fsm = AsyncSimpleFSM("config.yaml")

    result = await fsm.process(
        {"order_id": 123, "amount": 99.99},
        timeout=60.0
    )

    if result["success"]:
        print(f"Final state: {result['final_state']}")
        print(f"Processed data: {result['data']}")
    else:
        print(f"Error: {result['error']}")

    await fsm.close()

asyncio.run(main())
```

### process_batch

```python
async process_batch(
    data: List[Union[Dict[str, Any], Record]],
    batch_size: int = 10,
    max_workers: int = 4,
    on_progress: Union[Callable, None] = None
) -> List[Dict[str, Any]]
```

Process multiple records in parallel batches asynchronously.

**Parameters:**
- `data`: List of input records to process
- `batch_size`: Number of records per batch
- `max_workers`: Maximum parallel workers
- `on_progress`: Optional async callback for progress updates

**Returns:** List of results for each input record

**Example:**
```python
async def progress_callback(completed, total):
    print(f"Progress: {completed}/{total}")

async def main():
    fsm = AsyncSimpleFSM("config.yaml")

    items = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30}
    ]

    results = await fsm.process_batch(
        items,
        batch_size=10,
        max_workers=3,
        on_progress=progress_callback
    )

    for result in results:
        print(f"ID {result['data']['id']}: {result['success']}")

    await fsm.close()

asyncio.run(main())
```

### process_stream

```python
async process_stream(
    source: Union[str, Path, AsyncIterator[Union[Dict[str, Any], Record]]],
    sink: Optional[Union[str, Path, Callable]] = None,
    chunk_size: int = 100
) -> Dict[str, Any]
```

Process a stream of records asynchronously.

**Parameters:**
- `source`: File path or async iterator providing input records
- `sink`: Optional file path or async callback to handle each result
- `chunk_size`: Size of internal buffer for processing

**Returns:** Dict with processing statistics

**Example:**
```python
async def record_generator():
    for i in range(100):
        yield {"id": i, "value": i * 10}

async def main():
    fsm = AsyncSimpleFSM("config.yaml")

    # From async generator
    stats = await fsm.process_stream(record_generator())
    print(f"Processed: {stats['processed']}, Errors: {stats['errors']}")

    # From file to file
    stats = await fsm.process_stream(
        source="input.jsonl",
        sink="output.jsonl",
        chunk_size=50
    )

    await fsm.close()

asyncio.run(main())
```

### validate

```python
async validate(
    data: Union[Dict[str, Any], Record]
) -> bool
```

Validate data against the FSM's schema asynchronously.

**Parameters:**
- `data`: Data to validate

**Returns:** True if valid, False otherwise

**Example:**
```python
async def main():
    fsm = AsyncSimpleFSM("config.yaml")

    is_valid = await fsm.validate({"required_field": "value"})
    print(f"Data is valid: {is_valid}")

    await fsm.close()

asyncio.run(main())
```

### close

```python
async close() -> None
```

Close the FSM and clean up resources.

**Example:**
```python
async def main():
    fsm = AsyncSimpleFSM("config.yaml")
    try:
        result = await fsm.process({"data": "input"})
        print(result)
    finally:
        await fsm.close()  # Always close to clean up resources

asyncio.run(main())
```

## Complete Example with Async Functions

```python
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
import asyncio
import aiohttp

# Define async custom functions
async def fetch_data(state):
    """Fetch data from external API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/data/{state.data['id']}") as response:
            api_data = await response.json()
            return {**state.data, "fetched": api_data}

async def process_data(state):
    """Process the fetched data."""
    await asyncio.sleep(0.5)  # Simulate processing
    data = state.data.copy()
    data["processed"] = True
    data["result"] = data.get("fetched", {}).get("value", 0) * 2
    return data

async def save_data(state):
    """Save processed data to database."""
    # Simulate async database save
    await asyncio.sleep(0.2)
    data = state.data.copy()
    data["saved"] = True
    return data

# Define configuration
config = {
    "name": "async_workflow",
    "states": [
        {"name": "start", "is_start": True},
        {"name": "fetch"},
        {"name": "process"},
        {"name": "save"},
        {"name": "complete", "is_end": True}
    ],
    "arcs": [
        {
            "from": "start",
            "to": "fetch",
            "transform": {
                "type": "registered",
                "name": "fetch_data"
            }
        },
        {
            "from": "fetch",
            "to": "process",
            "transform": {
                "type": "registered",
                "name": "process_data"
            }
        },
        {
            "from": "process",
            "to": "save",
            "transform": {
                "type": "registered",
                "name": "save_data"
            }
        },
        {"from": "save", "to": "complete"}
    ]
}

async def main():
    # Create FSM with async custom functions
    fsm = AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={
            "fetch_data": fetch_data,
            "process_data": process_data,
            "save_data": save_data
        }
    )

    # Process single item
    result = await fsm.process({"id": "123"})
    if result["success"]:
        print(f"Successfully processed: {result['data']}")

    # Process batch
    items = [{"id": str(i)} for i in range(10)]
    results = await fsm.process_batch(items, max_workers=5)
    print(f"Batch processing: {sum(1 for r in results if r['success'])}/10 successful")

    # Clean up
    await fsm.close()

# Run the async workflow
asyncio.run(main())
```

## Context Managers

AsyncSimpleFSM supports async context managers for automatic resource cleanup:

```python
async def main():
    async with AsyncSimpleFSM("config.yaml") as fsm:
        result = await fsm.process({"input": "data"})
        print(result)
    # Resources automatically cleaned up

asyncio.run(main())
```

## Error Handling

AsyncSimpleFSM provides comprehensive error handling:

```python
async def main():
    fsm = AsyncSimpleFSM("config.yaml")

    try:
        result = await fsm.process({"input": "data"}, timeout=5.0)
        if not result["success"]:
            print(f"Processing failed: {result.get('error')}")
    except asyncio.TimeoutError:
        print("Processing timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await fsm.close()

asyncio.run(main())
```

## Integration with Async Frameworks

AsyncSimpleFSM works seamlessly with async frameworks like FastAPI:

```python
from fastapi import FastAPI
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

app = FastAPI()
fsm = AsyncSimpleFSM("workflow.yaml")

@app.on_event("startup")
async def startup():
    # FSM is ready to use
    pass

@app.on_event("shutdown")
async def shutdown():
    await fsm.close()

@app.post("/process")
async def process_data(data: dict):
    result = await fsm.process(data)
    return result
```

## Best Practices

1. **Always use async/await**: AsyncSimpleFSM is designed for async contexts
2. **Close resources**: Always call `close()` or use context managers
3. **Handle timeouts**: Set appropriate timeouts for long-running operations
4. **Use batch processing**: For multiple items, use `process_batch` for efficiency
5. **Stream large datasets**: Use `process_stream` for memory-efficient processing
6. **Async custom functions**: Make custom functions async when they perform I/O

## Differences from SimpleFSM

| Feature | SimpleFSM | AsyncSimpleFSM |
|---------|-----------|----------------|
| **Context** | Synchronous | Asynchronous |
| **Methods** | Regular functions | Async functions |
| **Custom Functions** | Sync or async | Preferably async |
| **Event Loop** | Manages internal loop | Uses existing loop |
| **Use With** | Regular Python code | async/await code |
| **Frameworks** | Flask, Django | FastAPI, aiohttp |

## See Also

- [SimpleFSM API](simple.md) for synchronous usage
- [AdvancedFSM API](advanced.md) for debugging and monitoring features
- [Examples](../examples/index.md) for real-world usage
- [Quick Start](../quickstart.md) for getting started
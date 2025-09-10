# Quick Start Guide

Get started with the FSM package in just a few minutes! This guide will walk you through the basics of creating and running finite state machines.

## Installation

Install the FSM package as part of dataknobs:

```bash
pip install dataknobs
```

## Your First FSM

Let's create a simple FSM that processes data through multiple stages:

```python
from dataknobs_fsm import SimpleFSM

# Create an FSM instance
fsm = SimpleFSM()

# Define states
fsm.add_state("start", initial=True)
fsm.add_state("validate")
fsm.add_state("process")
fsm.add_state("complete", terminal=True)

# Define transitions with processing functions
fsm.add_transition(
    "start", "validate",
    function=lambda data: {**data, "validated": False}
)

fsm.add_transition(
    "validate", "process",
    function=lambda data: {**data, "validated": True},
    condition=lambda data: data.get("value", 0) > 0
)

fsm.add_transition(
    "process", "complete",
    function=lambda data: {**data, "result": data["value"] * 2}
)

# Run the FSM
result = fsm.run({"value": 10})
print(result)
# Output: {"value": 10, "validated": True, "result": 20}
```

## Using Configuration Files

For more complex FSMs, use YAML configuration:

```yaml
# workflow.yaml
name: data_workflow
description: A simple data processing workflow

states:
  - name: start
    initial: true
    
  - name: fetch_data
    metadata:
      timeout: 30
      
  - name: transform
    
  - name: save
    
  - name: complete
    terminal: true

arcs:
  - from: start
    to: fetch_data
    
  - from: fetch_data
    to: transform
    function:
      type: python
      module: myapp.transformers
      name: clean_data
      
  - from: transform
    to: save
    function:
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
from dataknobs_fsm import SimpleFSM

# Load FSM from configuration
fsm = SimpleFSM.from_config("workflow.yaml")

# Run with input data
result = fsm.run({"source": "api", "records": 100})
```

## Async Execution

For I/O-bound operations, use async execution:

```python
import asyncio
from dataknobs_fsm import SimpleFSM

async def fetch_data(data):
    # Simulate async API call
    await asyncio.sleep(1)
    return {**data, "fetched": True}

async def save_data(data):
    # Simulate async database save
    await asyncio.sleep(0.5)
    return {**data, "saved": True}

# Create FSM with async functions
fsm = SimpleFSM()
fsm.add_state("start", initial=True)
fsm.add_state("fetch")
fsm.add_state("save")
fsm.add_state("done", terminal=True)

fsm.add_transition("start", "fetch", function=fetch_data)
fsm.add_transition("fetch", "save", function=save_data)
fsm.add_transition("save", "done")

# Run asynchronously
async def main():
    result = await fsm.run_async({"id": 123})
    print(result)

asyncio.run(main())
```

## Using Resources

Manage external resources like databases and APIs:

```python
from dataknobs_fsm import SimpleFSM

# Create FSM with resources
fsm = SimpleFSM()

# Add a database resource
fsm.add_resource("db", {
    "type": "database",
    "provider": "sqlite",
    "config": {"database": "myapp.db"}
})

# Add states
fsm.add_state("start", initial=True)
fsm.add_state("query")
fsm.add_state("done", terminal=True)

# Use resource in transition
async def query_database(data, resources):
    db = resources["db"]
    async with db.connect() as conn:
        result = await conn.execute("SELECT * FROM users WHERE id = ?", [data["user_id"]])
        return {**data, "user": await result.fetchone()}

fsm.add_transition("start", "query", function=query_database)
fsm.add_transition("query", "done")

# Run with resource management
result = fsm.run({"user_id": 1})
```

## Batch Processing

Process multiple items in parallel:

```python
from dataknobs_fsm import SimpleFSM

fsm = SimpleFSM()

# Define a simple processing pipeline
fsm.add_state("start", initial=True)
fsm.add_state("process")
fsm.add_state("done", terminal=True)

fsm.add_transition(
    "start", "process",
    function=lambda data: {**data, "processed": True}
)
fsm.add_transition("process", "done")

# Process batch of items
items = [
    {"id": 1, "value": 10},
    {"id": 2, "value": 20},
    {"id": 3, "value": 30}
]

results = fsm.run_batch(items, max_workers=3)
for result in results:
    print(result)
```

## Error Handling

Add error handling and retries:

```python
from dataknobs_fsm import SimpleFSM

fsm = SimpleFSM()

# Configure with retry logic
fsm.add_state("start", initial=True)
fsm.add_state("process")
fsm.add_state("error")
fsm.add_state("done", terminal=True)

def risky_operation(data):
    import random
    if random.random() < 0.5:
        raise ValueError("Random failure")
    return {**data, "processed": True}

# Add transitions with error handling
fsm.add_transition(
    "start", "process",
    function=risky_operation,
    on_error="error"
)

fsm.add_transition(
    "error", "process",
    function=lambda data: {**data, "retry": True}
)

fsm.add_transition("process", "done")

# Run with automatic retries
result = fsm.run({"input": "data"}, max_retries=3)
```

## Using the CLI

The FSM package includes a CLI tool for interactive use:

```bash
# List available FSMs
fsm list

# Run an FSM from configuration
fsm run workflow.yaml --input '{"key": "value"}'

# Validate configuration
fsm validate workflow.yaml

# Visualize FSM structure
fsm visualize workflow.yaml --output workflow.png
```

## Next Steps

Now that you understand the basics:

1. Explore [Integration Patterns](patterns/index.md) for pre-built solutions
2. Read the [API Documentation](api/index.md) for detailed reference
3. Check out [Examples](examples/index.md) for real-world use cases
4. Learn about [Data Modes](guides/data-modes.md) for efficient data handling
5. Understand [Resource Management](guides/resources.md) for external integrations
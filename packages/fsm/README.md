# DataKnobs FSM

Finite State Machine framework with data modes, resource management, and streaming support.

## Features

- **Data Modes**: COPY, REFERENCE, and DIRECT modes for flexible data handling
- **Transaction Management**: Single, Batch, and Manual transaction strategies
- **Resource Management**: Built-in support for databases, files, HTTP services, LLMs, and vector stores
- **Streaming Support**: Process large datasets with chunking and backpressure handling
- **Flexible Configuration**: YAML/JSON configuration with schema validation
- **Built-in Functions**: Library of common validation and transformation functions

## Installation

```bash
pip install dataknobs-fsm
```

## Quick Start

```python
from dataknobs_fsm import FSM, StateDefinition, DataMode

# Define states
start = StateDefinition(name="start", type=StateType.START)
process = StateDefinition(name="process", data_mode=DataMode.COPY)
end = StateDefinition(name="end", type=StateType.END)

# Create FSM
fsm = FSM()
fsm.add_state(start)
fsm.add_state(process)
fsm.add_state(end)

# Process data
result = fsm.process({"input": "data"})
```

## Documentation

See the [docs](docs/) directory for detailed documentation.

## Development

This package is part of the DataKnobs ecosystem and follows the project's development guidelines.
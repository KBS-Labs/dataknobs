# DataKnobs FSM

Finite State Machine framework with data modes, resource management, and streaming support.

## Features

- **Multiple APIs**: SimpleFSM, AsyncSimpleFSM, and AdvancedFSM for different use cases
- **Data Handling Modes**: COPY, REFERENCE, and DIRECT modes for flexible data management
- **Resource Management**: Built-in support for databases, files, HTTP services, and vector stores
- **Streaming Support**: Process large datasets with chunking and backpressure handling
- **Advanced Debugging**: Step-by-step execution, breakpoints, and execution hooks
- **Flexible Configuration**: YAML/JSON configuration with schema validation
- **Built-in Functions**: Library of common validation and transformation functions

## Installation

```bash
pip install dataknobs-fsm
```

## Quick Start

### Simple FSM

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Define configuration
config = {
    "name": "data_pipeline",
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

# Create and run FSM
fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)
result = fsm.process({"input": "data"})
print(f"Result: {result['data']}")
```

### Advanced FSM with Debugging

```python
from dataknobs_fsm import AdvancedFSM, ExecutionMode
import asyncio

async def debug_example():
    # Create FSM with debug mode
    fsm = AdvancedFSM(
        "config.yaml",
        execution_mode=ExecutionMode.DEBUG
    )

    # Add breakpoint
    fsm.add_breakpoint("process")

    # Create context and run
    context = fsm.create_context({"input": "data"})
    await fsm.run_until_breakpoint(context)
    print(f"Stopped at: {context.current_state}")

    # Continue execution
    await fsm.step(context)

asyncio.run(debug_example())
```

## Examples

The `examples/` directory contains comprehensive examples:

### Data Processing Examples
- **data_pipeline_example.py** - Data validation and transformation pipeline
- **data_validation_pipeline.py** - Data quality validation workflow
- **database_etl.py** - Complete ETL pipeline with transaction management
- **large_file_processor.py** - Memory-efficient large file processing
- **end_to_end_streaming.py** - Streaming pipeline demonstration

### Advanced Features
- **advanced_debugging.py** - Full debugging features demonstration
- **advanced_debugging_simple.py** - Simplified debugging example

### Text Processing
- **normalize_file_example.py** - Text file normalization with streaming
- **normalize_file_with_regex.py** - Advanced regex transformations
- **test_regex_yaml.py** - Testing script for YAML regex configurations

### Configuration Examples
- **regex_transforms.yaml** - Field transformation workflows
- **regex_workflow.yaml** - Pattern extraction and masking configurations

## Running Examples

```bash
# Navigate to the FSM package
cd packages/fsm

# Run the database ETL example
uv run python examples/database_etl.py

# Run the data processing pipeline
uv run python examples/data_pipeline_example.py

# Run streaming example
uv run python examples/end_to_end_streaming.py

# Run with custom parameters
uv run python examples/database_etl.py --batch-size 500
```

## Data Handling Modes

The FSM framework provides three data handling modes:

- **COPY Mode**: Creates deep copies of data for each state, ensuring isolation
- **REFERENCE Mode**: Uses lazy loading with optimistic locking for memory efficiency
- **DIRECT Mode**: In-place modifications for maximum performance (single-threaded only)

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Use COPY mode for safety
fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)

# Use REFERENCE mode for large datasets
fsm = SimpleFSM(config, data_mode=DataHandlingMode.REFERENCE)

# Use DIRECT mode for performance
fsm = SimpleFSM(config, data_mode=DataHandlingMode.DIRECT)
```

## LLM Integration

For LLM-specific integrations, workflows, and examples, please see the **dataknobs-llm** package:

- **FSM Integration Module**: `dataknobs_llm.fsm_integration`
- **LLM Workflow Patterns**: RAG pipelines, chain-of-thought, multi-agent systems
- **Conversation Examples**: FSM-based conversational AI systems
- **Documentation**: See `packages/llm/README.md` for FSM integration guide

The LLM package provides comprehensive LLM abstractions, providers, and FSM integration capabilities.

## Documentation

For detailed documentation, see:
- [Configuration Guide](docs/FSM_CONFIG_GUIDE.md) - Complete configuration reference
- [Processing Flow](docs/FSM_PROCESSING_FLOW.md) - Understanding FSM execution
- [Examples Documentation](docs/README.md) - Detailed example descriptions

## Testing

Run the tests with:

```bash
cd packages/fsm
uv run pytest tests/ -v
```

## Development

This package is part of the DataKnobs ecosystem. For development setup and guidelines, see the main repository README.

## License

Licensed under the same terms as the DataKnobs project.
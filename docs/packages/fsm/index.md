# FSM (Finite State Machine) Package

The FSM package provides a powerful and flexible framework for building state machines in Python. It enables you to create complex workflows, orchestrate API calls, manage resources, and build sophisticated data processing pipelines.

## Features

- **Core FSM Engine**: Robust state machine implementation with support for complex transitions and data flows
- **Multiple Execution Strategies**: Synchronous, asynchronous, batch, and streaming execution modes
- **Resource Management**: Built-in resource lifecycle management for databases, files, HTTP clients, and more
- **Integration Patterns**: Pre-built patterns for common use cases like ETL, API orchestration, and LLM workflows
- **Data Modes**: Flexible data handling with COPY, REFERENCE, and DIRECT modes
- **Configuration-Driven**: Define FSMs using YAML/JSON configuration or programmatically
- **CLI Tool**: Interactive command-line interface for managing and executing FSMs

## Quick Links

- [Quick Start Guide](quickstart.md) - Get started with FSM in 5 minutes
- [API Documentation](api/index.md) - Complete API reference
- [Patterns Guide](patterns/index.md) - Pre-built integration patterns
- [Examples](examples/index.md) - Real-world usage examples

## Installation

The FSM package is included as part of the dataknobs installation:

```bash
pip install dataknobs
```

Or install just the FSM package:

```bash
pip install dataknobs-fsm
```

## Basic Usage

### Using SimpleFSM API

```python
from dataknobs_fsm import SimpleFSM

# Create an FSM
fsm = SimpleFSM()

# Add states
fsm.add_state("start", initial=True)
fsm.add_state("process")
fsm.add_state("end", terminal=True)

# Add transitions with functions
fsm.add_transition(
    "start", "process",
    function=lambda data: {"processed": True, **data}
)
fsm.add_transition("process", "end")

# Execute
result = fsm.run({"input": "data"})
print(result)  # {"processed": True, "input": "data"}
```

### Using Configuration

```yaml
# fsm_config.yaml
name: data_processor
states:
  - name: start
    initial: true
  - name: process
  - name: end
    terminal: true

arcs:
  - from: start
    to: process
    function:
      type: lambda
      code: "lambda data: {'processed': True, **data}"
  - from: process
    to: end
```

```python
from dataknobs_fsm import SimpleFSM

fsm = SimpleFSM.from_config("fsm_config.yaml")
result = fsm.run({"input": "data"})
```

## Architecture

The FSM package is built with a layered architecture:

1. **Core Layer**: State machines, states, arcs, and execution contexts
2. **Execution Layer**: Various execution strategies and engines
3. **Resource Layer**: Resource providers and lifecycle management
4. **Configuration Layer**: YAML/JSON configuration support
5. **API Layer**: Simple and Advanced APIs for different use cases
6. **Pattern Layer**: Pre-built patterns for common scenarios

## Use Cases

- **Data Processing Pipelines**: ETL workflows, data validation, transformation
- **API Orchestration**: Managing complex API workflows with rate limiting and retries
- **LLM Workflows**: Building conversational AI systems with multiple LLM providers
- **File Processing**: Batch processing of files with different formats
- **Stream Processing**: Real-time data processing with backpressure handling
- **Error Recovery**: Implementing retry logic, circuit breakers, and fallback strategies

## Next Steps

- Explore the [Quick Start Guide](quickstart.md) for a hands-on introduction
- Check out [Examples](examples/index.md) for real-world use cases
- Read the [API Documentation](api/index.md) for detailed reference
- Learn about [Integration Patterns](patterns/index.md) for common scenarios
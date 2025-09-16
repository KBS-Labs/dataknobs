# FSM (Finite State Machine) Package

The FSM package provides a powerful and flexible framework for building state machines in Python. It enables you to create complex workflows, orchestrate API calls, manage resources, and build sophisticated data processing pipelines.

## Features

### Core Capabilities

- **Core FSM Engine**: Robust state machine implementation with support for complex transitions and data flows
- **Multiple Execution Strategies**: Depth-first, breadth-first, resource-optimized, and stream-optimized traversal strategies
- **Execution Modes**: Synchronous, asynchronous, batch, and streaming execution engines
- **Configuration-Driven**: Define FSMs using YAML/JSON configuration or programmatically

### Data Handling Modes

- **COPY Mode** (default): Safe concurrent processing with data isolation
- **REFERENCE Mode**: Memory-efficient processing with optimistic locking
- **DIRECT Mode**: High-performance in-place modifications

### Resource Management

Comprehensive lifecycle management with pooling for:

- Database connections with transaction support
- HTTP clients with retry logic
- LLM providers with rate limiting
- File system resources
- Vector stores for embeddings

### Transform Functions

Interface-based transform system with:

- Validation functions for data quality
- Transform functions for data manipulation
- State test functions for conditional logic
- Extensive function library

### Integration Patterns

Production-ready patterns for:

- ETL workflows (FULL_REFRESH, INCREMENTAL, UPSERT, APPEND)
- API orchestration with multi-service coordination
- Error recovery with retry and circuit breaker patterns
- File processing pipelines
- LLM/AI workflow orchestration

### Advanced Features

- Hierarchical state machines with PushArc support
- Transaction management with commit/rollback
- Execution history tracking and auditing
- Configurable data isolation for sub-networks
- Health monitoring and metrics

### API Design

- **SimpleFSM**: For straightforward workflows
- **AdvancedFSM**: For debugging, stepping, and hooks

## Quick Links

- [Quick Start Guide](quickstart.md) - Get started with FSM in 5 minutes
- [API Documentation](api/index.md) - Complete API reference
- [Patterns Guide](patterns/index.md) - Pre-built integration patterns
- [Examples](examples/index.md) - Real-world usage examples

## Installation

```bash
pip install dataknobs-fsm
```

Or with optional dependencies:

```bash
# With database support
pip install dataknobs-fsm[database]

# With LLM provider support
pip install dataknobs-fsm[llm]

# With all extras
pip install dataknobs-fsm[all]
```

## Basic Usage

### Using SimpleFSM API

```python
from dataknobs_fsm.api.simple import SimpleFSM

# Create an FSM with custom functions
def validate_data(state):
    """Custom validation function."""
    if 'required_field' not in state.data:
        raise ValueError("Missing required field")
    return state.data

def process_data(state):
    """Transform function for processing."""
    data = state.data.copy()
    data['processed'] = True
    data['timestamp'] = datetime.now().isoformat()
    return data

# Initialize FSM with custom functions
fsm = SimpleFSM(
    custom_functions={
        'validate': validate_data,
        'process': process_data
    }
)

# Add states with validation
fsm.add_state("start", initial=True)
fsm.add_state("validate")
fsm.add_state("process")
fsm.add_state("end", terminal=True)

# Add transitions with custom functions
fsm.add_transition(
    "start", "validate",
    function={"type": "registered", "name": "validate"}
)
fsm.add_transition(
    "validate", "process",
    function={"type": "registered", "name": "process"}
)
fsm.add_transition("process", "end")

# Execute with data mode selection
result = fsm.run(
    {"required_field": "value", "input": "data"},
    data_mode="COPY"  # or "REFERENCE" or "DIRECT"
)
print(result)  # {"required_field": "value", "input": "data", "processed": True, "timestamp": "..."}
```

### Using AdvancedFSM API

```python
from dataknobs_fsm import AdvancedFSM
from dataknobs_fsm.api.advanced import ExecutionMode, ExecutionHook

class LoggingHook(ExecutionHook):
    """Custom execution hook for logging."""

    def on_state_enter(self, state, data):
        print(f"Entering state: {state.name}")

    def on_state_exit(self, state, data):
        print(f"Exiting state: {state.name}")

    def on_error(self, error, state):
        print(f"Error in state {state.name}: {error}")

# Create advanced FSM with debugging features
fsm = AdvancedFSM(
    config="fsm_config.yaml",
    execution_mode=ExecutionMode.DEBUG,
    hooks=[LoggingHook()]
)

# Step-by-step execution
for step in fsm.step_through({"input": "data"}):
    print(f"Current state: {step.current_state}")
    print(f"Data: {step.data}")
    if step.can_continue:
        # Optionally modify data or add breakpoints
        continue

# Or run with profiling
result, profile = fsm.run_with_profile({"input": "data"})
print(f"Execution time: {profile.total_time}s")
print(f"States visited: {profile.states_visited}")
```

### Using Configuration

```yaml
# fsm_config.yaml
name: data_processor
data_mode: COPY  # or REFERENCE or DIRECT
execution_strategy: DEPTH_FIRST  # or BREADTH_FIRST, RESOURCE_OPTIMIZED, STREAM_OPTIMIZED

states:
  - name: start
    initial: true
  - name: validate
    functions:
      state_test:
        type: builtin
        name: has_required_fields
        params:
          fields: ["user_id", "data"]
  - name: process
    functions:
      transform:
        type: inline
        code: |
          lambda state: {
              **state.data,
              'processed': True,
              'timestamp': datetime.now().isoformat()
          }
    resources:
      - type: database
        name: main_db
  - name: end
    terminal: true

arcs:
  - from: start
    to: validate
  - from: validate
    to: process
    pre_test:
      type: builtin
      name: data_valid
  - from: process
    to: end
    transform:
      type: builtin
      name: add_metadata

resources:
  database:
    main_db:
      provider: postgresql
      connection_string: ${DATABASE_URL}
      pool_size: 10
```

```python
from dataknobs_fsm.api.simple import SimpleFSM

fsm = SimpleFSM.from_config("fsm_config.yaml")
result = fsm.run({"user_id": "123", "data": "input"})
```

## Architecture

The FSM package is built with a modular, layered architecture:

### Core Components

#### State Management
- `StateDefinition`: Template for states with schemas and validation
- `StateInstance`: Runtime state instances with data and context
- `StateTest`, `ValidityTest`: Condition checking functions

#### Arc System
- `ArcDefinition`: Transition definitions with optional transforms
- `ArcExecution`: Runtime arc execution with resource management
- `PushArc`: Hierarchical composition for sub-networks

#### Execution Engines
- `ExecutionEngine`: Synchronous execution with strategy support
- `AsyncExecutionEngine`: Asynchronous execution with concurrency
- `BatchExecutor`: Optimized batch processing
- `StreamExecutor`: Stream processing with backpressure

#### Data Handling
- `DataModeHandler`: Abstract interface for data operations
- `CopyModeHandler`: Safe concurrent processing
- `ReferenceModeHandler`: Memory-efficient with locking
- `DirectModeHandler`: High-performance in-place operations

#### Resource Management
- `ResourceManager`: Central resource lifecycle control
- `ResourcePool`: Connection pooling with health checks
- Specialized providers for databases, HTTP, LLM, filesystem

#### Function System
- Interface-based design with `ITransformFunction`, `IValidationFunction`
- Extensive function library for common operations
- Custom function registration support

## Use Cases

### Data Processing Pipelines

- ETL workflows with multiple data sources and targets
- Data validation and quality checks
- Format transformation and normalization
- Batch and stream processing modes

### API Orchestration

- Multi-service API coordination
- Rate limiting and quota management
- Retry logic with exponential backoff
- Circuit breaker patterns for fault tolerance

### LLM/AI Workflows

- Multi-provider LLM orchestration (OpenAI, Anthropic, HuggingFace)
- Prompt chaining and response validation
- Fallback strategies for provider failures
- Token usage optimization

### File Processing

- Batch file processing with multiple formats
- Parallel processing with resource constraints
- Progress tracking and resumption
- Error handling and partial failure recovery

### Stream Processing

- Real-time data ingestion and transformation
- Backpressure handling and flow control
- Window-based aggregations
- Event-driven architectures

### Error Recovery Patterns

- Configurable retry strategies
- Circuit breakers with health monitoring
- Fallback and compensation logic
- Dead letter queue handling

## Key Concepts

### Understanding the Two Mode Types

The FSM package has two distinct mode concepts that are often confused:

#### DataHandlingMode - HOW data is managed
Controls how individual states handle data internally:

| Mode | Description | Use Case |
|------|-------------|----------|
| **COPY** | Creates deep copies of data for safe concurrent processing | Default mode, best for multi-threaded environments |
| **REFERENCE** | Works with data references using optimistic locking | Large datasets, database-backed workflows |
| **DIRECT** | Operates directly on source data | Single-threaded, high-performance scenarios |

#### ProcessingMode - HOW MANY records to process
Controls the execution strategy for record volume:

| Mode | Description | Use Case |
|------|-------------|----------|
| **SINGLE** | Process one record at a time | Simple record-by-record operations |
| **BATCH** | Process multiple records in batches | Optimizing database operations, transactions |
| **STREAM** | Process continuous streams of data | Large files, real-time data |

**Key Difference**: DataHandlingMode is about memory safety and concurrency *within* states, while ProcessingMode is about throughput and how many records to handle at once. They work together but address different concerns.

#### Common Combinations

| Use Case | ProcessingMode | DataHandlingMode | Why |
|----------|---------------|------------------|-----|
| Web API requests | SINGLE | COPY | Each request isolated, concurrent safety |
| ETL pipeline | BATCH | COPY | Transaction boundaries, rollback support |
| Large file streaming | STREAM | REFERENCE | Memory efficiency for continuous data |
| High-speed validation | SINGLE | DIRECT | Maximum performance, simple operations |
| Database bulk updates | BATCH | COPY | Transaction safety for batches |

### Execution Strategies

Choose from multiple traversal strategies:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **DEPTH_FIRST** | Explores deeply before backtracking | Linear workflows, sequential processing |
| **BREADTH_FIRST** | Explores all branches at same level | Parallel exploration, comparison workflows |
| **RESOURCE_OPTIMIZED** | Minimizes resource usage | Resource-constrained environments |
| **STREAM_OPTIMIZED** | Optimized for streaming data | Real-time processing, event streams |

### Function Types

Functions can be registered and used throughout the FSM:

| Type | Description | Example |
|------|-------------|---------|
| **inline** | Lambda expressions or code strings | `"lambda state: state.data.upper()"` |
| **builtin** | Pre-registered library functions | `{"type": "builtin", "name": "validate_email"}` |
| **custom** | Functions from Python modules | `{"type": "custom", "module": "myapp.transforms"}` |
| **registered** | Runtime-registered functions | `{"type": "registered", "name": "process_data"}` |

## Next Steps

- **Getting Started**: Follow the [Quick Start Guide](quickstart.md) for a hands-on introduction
- **Examples**: Explore [real-world examples](examples/index.md) with complete code
- **API Reference**: Read the [API Documentation](api/index.md) for detailed reference
- **Patterns**: Learn about [Integration Patterns](patterns/index.md) for common scenarios

### [Guides](guides/index.md): Deep dive into specific features:
- [Data Modes Guide](guides/data-modes.md) - Choosing the right data operation mode
- [Resource Management](guides/resources.md) - Managing external resources
- [Streaming Guide](guides/streaming.md) - Building stream processing workflows
- [CLI Usage](guides/cli.md) - Using the FSM command-line interface

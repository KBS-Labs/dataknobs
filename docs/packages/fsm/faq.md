# Frequently Asked Questions (FAQ)

## General Questions

### What is DataKnobs FSM?

DataKnobs FSM is a flexible Finite State Machine framework for building data processing pipelines. It provides:
- State-based workflow management with hierarchical composition
- Two types of modes:
  - **DataHandlingMode**: COPY, REFERENCE, DIRECT for memory management
  - **ProcessingMode**: SINGLE, BATCH, STREAM for throughput control
- Resource management with pooling for databases, HTTP, LLM providers
- Streaming capabilities for large datasets
- Two main APIs: `SimpleFSM` (configuration-driven) and `AdvancedFSM` (debugging/monitoring)
- CLI tool (`fsm` command) for development and operations

### When should I use FSM?

FSM is ideal for:
- **ETL pipelines** - Extract, transform, and load data workflows
- **Data validation workflows** - Multi-stage data quality checks
- **API orchestration** - Coordinating multiple API calls
- **File processing** - Batch and stream file processing
- **Event-driven workflows** - State-based event processing

### How does FSM differ from other workflow tools?

FSM focuses on:
- **State-based design** - Clear state transitions with pre-tests and transforms
- **Dual mode system** - Separate control of memory safety (DataHandlingMode) and throughput (ProcessingMode)
- **Resource management** - Built-in pooling and lifecycle management
- **Pattern library** - Pre-built patterns for ETL, API orchestration, LLM workflows, error recovery
- **Debugging focus** - AdvancedFSM with breakpoints, stepping, profiling

## Installation and Setup

### How do I install FSM?

```bash
# Using pip
pip install dataknobs-fsm

# Using uv
uv pip install dataknobs-fsm

# From source
git clone https://github.com/dataknobs/fsm
cd fsm
pip install -e .
```

### What are the system requirements?

- Python 3.10 or higher (per pyproject.toml)
- Operating System: Linux, macOS, or Windows
- Memory: Depends on data mode and dataset size
- Core dependencies: pydantic>=2.0.0, dataknobs-data>=0.2.0, click>=8.1.0, rich>=13.0.0
- Optional: LLM providers (openai, anthropic), HTTP clients (httpx, aiohttp)

### How do I verify the installation?

```bash
# Check CLI installation
fsm --version  # Shows 0.1.0

# Test Python imports (note different import paths)
python -c "from dataknobs_fsm.api.simple import SimpleFSM; print('SimpleFSM OK')"
python -c "from dataknobs_fsm import AdvancedFSM; print('AdvancedFSM OK')"
```

## Configuration

### What configuration formats are supported?

FSM supports:
- **YAML** (recommended) - Human-readable, comments supported
- **JSON** - Machine-readable, programmatic generation
- **Python dictionaries** - Direct API usage

### How do I validate my configuration?

```bash
# Using CLI
fsm config validate my_config.yaml

# Using Python
from dataknobs_fsm.config.loader import ConfigLoader
loader = ConfigLoader()
config = loader.load_from_file("my_config.yaml")
```

### Can I use environment variables in configuration?

Yes, use the `${VAR_NAME}` syntax:

```yaml
resources:
  - name: database
    connection_string: ${DATABASE_URL}
```

## Data Handling

### What's the difference between DataHandlingMode and ProcessingMode?

**DataHandlingMode** (HOW data is managed in memory):
- **COPY** - Creates deep copies for safety (default)
- **REFERENCE** - Uses references with optimistic locking
- **DIRECT** - In-place modifications (single-threaded only)

**ProcessingMode** (HOW MANY records to process):
- **SINGLE** - One record at a time
- **BATCH** - Multiple records in groups
- **STREAM** - Continuous data flow

They work together but solve different problems. See [Data Modes Guide](guides/data-modes.md).

### How do I choose the right combination?

| Use Case | ProcessingMode | DataHandlingMode | Why |
|----------|---------------|------------------|-----|
| Web API | SINGLE | COPY | Isolation between requests |
| ETL Pipeline | BATCH | COPY | Transaction boundaries |
| Large Files | STREAM | REFERENCE | Memory efficiency |
| Real-time | SINGLE | DIRECT | Minimum latency |

## Resources

### What are resources in FSM?

Resources are external dependencies like:
- Database connections
- File systems
- HTTP services
- LLM providers
- Custom services

### How do I manage resources?

Resources are typically configured in the FSM config:

```yaml
resources:
  - name: database
    type: database
    provider: postgresql
    config:
      connection_string: ${DATABASE_URL}
      pool_size: 10
```

Or programmatically:

```python
from dataknobs_fsm.api.simple import SimpleFSM

fsm = SimpleFSM(
    config,
    resources={
        "db": {"type": "database", "provider": "postgresql", "connection": "..."}
    }
)
```

See the [Resources Guide](guides/resources.md) for details.

## Streaming

### When should I use streaming?

Use streaming for:
- Files larger than available memory
- Real-time data processing
- Continuous data sources
- Pipeline architectures

### How do I implement streaming?

Use the FileProcessor pattern or configure ProcessingMode.STREAM:

```python
from dataknobs_fsm.patterns.file_processing import create_csv_processor
from dataknobs_fsm.core.modes import ProcessingMode

# Using pattern
processor = create_csv_processor(
    input_file="large.csv",
    output_file="processed.json",
    transformations=[...]
)

# Or with SimpleFSM
fsm = SimpleFSM(config)  # config specifies ProcessingMode.STREAM
result = await fsm.process_stream(source, sink)
```

See the [Streaming Guide](guides/streaming.md) for details.

## Debugging

### How do I debug FSM execution?

Using the CLI:
```bash
# Enable tracing
fsm debug trace config.yaml --data data.json

# Profile execution
fsm debug profile config.yaml --data data.json
```

Using AdvancedFSM:
```python
from dataknobs_fsm import AdvancedFSM, ExecutionMode

fsm = AdvancedFSM(config, execution_mode=ExecutionMode.DEBUG)
fsm.add_breakpoint("process_state")

# Step through execution
context = fsm.create_context(data)
await fsm.run_until_breakpoint(context)
```

### How do I view execution history?

```bash
# List recent executions
fsm history list

# Show specific execution
fsm history show execution_id

# Query by criteria
fsm history list --fsm-name MyFSM --limit 10
```

## Performance

### How can I improve FSM performance?

1. **Choose appropriate data mode** - DIRECT for large datasets
2. **Use streaming** - For files larger than memory
3. **Enable batching** - Process multiple records together
4. **Pool resources** - Reuse connections
5. **Optimize state functions** - Profile and optimize bottlenecks

### What are typical performance metrics?

Performance depends on:
- Data size and complexity
- Number of states and transitions
- Resource operations (I/O, network)
- Data mode and processing mode

Benchmark with your specific use case.

## Troubleshooting

### FSM CLI not found after installation

```bash
# Check installation
pip show dataknobs-fsm

# Reinstall with entry points
pip install --force-reinstall dataknobs-fsm

# Check PATH
which fsm
```

### Configuration validation fails

Common issues:
- Invalid YAML/JSON syntax
- Missing required fields
- Circular state dependencies
- Invalid arc conditions

### Resource acquisition timeout

```python
# Increase timeout
manager.acquire("database", owner_id="state", timeout=60)

# Check resource health
health = manager.health_check("database")
```

### Memory issues with large datasets

- Switch to REFERENCE or DIRECT mode
- Use streaming instead of batch processing
- Increase chunk size for streaming
- Monitor memory usage with profiling

## Best Practices

### Configuration Management

- Keep configurations in version control
- Use environment variables for secrets
- Validate configurations before deployment
- Document custom functions

### Error Handling

- Implement retry logic for transient failures
- Use dead letter queues for failed records
- Log errors with context
- Monitor execution history

### Testing

- Test with small datasets first
- Validate configurations in CI/CD
- Use mock resources for unit tests
- Benchmark performance regularly

## Common Import Errors and Solutions

### ImportError: cannot import name 'SimpleFSM' from 'dataknobs_fsm'

```python
# Wrong:
from dataknobs_fsm import SimpleFSM  # SimpleFSM not exported at package level

# Correct:
from dataknobs_fsm.api.simple import SimpleFSM
```

### ImportError: cannot import name 'DataMode'

```python
# Wrong:
from dataknobs_fsm import DataMode  # Incorrect name

# Correct:
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import ProcessingMode
```

## Getting Help

### Where can I find more documentation?

- [Quick Start Guide](quickstart.md) - Get started quickly
- [Guides](guides/index.md) - In-depth topic guides
- [API Reference](api/index.md) - SimpleFSM and AdvancedFSM documentation
- [Examples](examples/index.md) - Working examples in `packages/fsm/examples/`
- [Pattern Catalog](patterns/index.md) - Pre-built patterns (ETL, API, LLM, etc.)

### How do I report issues?

1. Check existing issues on GitHub
2. Provide minimal reproducible example
3. Include configuration and error messages
4. Specify FSM version and environment

### How can I contribute?

See our [Contributing Guide](../../development/contributing.md) for:
- Code style guidelines
- Testing requirements
- Pull request process
- Development setup
# FSM Guides

## Overview

This section contains comprehensive guides for using the DataKnobs FSM package effectively. Each guide focuses on a specific aspect of the FSM system with practical examples and best practices based on the actual implementation.

## Available Guides

### Core Concepts

- **[Configuration Guide](configuration.md)** - Complete guide to FSM configuration:
  - State and arc definitions
  - Function types and transforms
  - Data modes and networks
  - Common patterns and best practices

- **[Data Modes Guide](data-modes.md)** - Understanding the two types of modes:
  - **DataHandlingMode**: How data is managed within states (COPY, REFERENCE, DIRECT)
  - **ProcessingMode**: How many records to process (SINGLE, BATCH, STREAM)
  - When to use each mode and common combinations

- **[Resources Guide](resources.md)** - Managing external dependencies:
  - ResourceManager system for connection pooling
  - Built-in providers: Database, HTTP, LLM, FileSystem
  - Resource lifecycle and health monitoring

- **[Streaming Guide](streaming.md)** - Efficient large dataset processing:
  - Stream processing with backpressure handling
  - Memory-efficient data pipelines
  - Chunk-based processing strategies

### Tools and Usage

- **[CLI Guide](cli.md)** - Command-line interface (`fsm` command):
  - Configuration management (`fsm config`)
  - Running FSMs (`fsm run`)
  - Debugging and profiling (`fsm debug`)
  - History management (`fsm history`)
  - Pattern execution (`fsm pattern`)

## Quick Start

If you're new to DataKnobs FSM, we recommend reading the guides in this order:

1. **[Configuration Guide](configuration.md)** - Learn how to define states, arcs, and functions
2. **[Data Modes Guide](data-modes.md)** - Understand the critical distinction between DataHandlingMode (memory safety) and ProcessingMode (throughput)
3. **[Resources Guide](resources.md)** - Learn how to manage databases, APIs, and other external dependencies
4. **[CLI Guide](cli.md)** - Use the `fsm` command-line tool for development and debugging
5. **[Streaming Guide](streaming.md)** - Handle large datasets that don't fit in memory

## Key Topics Covered

### API Usage
- Using `SimpleFSM` from `dataknobs_fsm.api.simple`
- Using `AdvancedFSM` for debugging and monitoring
- Registering custom functions
- Configuration-driven FSM creation

### Data Management
- Choosing between COPY, REFERENCE, and DIRECT modes
- Understanding when to use SINGLE, BATCH, or STREAM processing
- Memory optimization strategies
- Transaction support and rollback

### Resource Management
- Connection pooling for databases
- HTTP client management with retry logic
- LLM provider integration
- Resource lifecycle and cleanup

### Performance & Debugging
- Using the CLI for profiling
- Step-by-step debugging with AdvancedFSM
- Execution history tracking
- Performance optimization techniques

## Related Documentation

- **[API Reference](../../../api/dataknobs-fsm.md)** - Complete API documentation for SimpleFSM and AdvancedFSM
- **[Pattern Catalog](../patterns/index.md)** - Pre-built patterns (ETL, File Processing, API Orchestration, LLM, Error Recovery)
- **[Examples](../examples/index.md)** - Working examples including `database_etl.py`, `large_file_processor.py`, and more
- **[Quick Start](../quickstart.md)** - Get started with FSM in minutes

## Import Quick Reference

```python
# Core APIs
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm import AdvancedFSM, ExecutionMode, ExecutionHook

# Data and Processing Modes
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import ProcessingMode

# Configuration
from dataknobs_fsm import ConfigLoader, FSMBuilder

# Patterns (direct imports required)
from dataknobs_fsm.patterns.etl import DatabaseETL, create_etl_pipeline
from dataknobs_fsm.patterns.file_processing import FileProcessor
from dataknobs_fsm.patterns.api_orchestration import APIOrchestrator
from dataknobs_fsm.patterns.llm_workflow import LLMWorkflow
from dataknobs_fsm.patterns.error_recovery import ErrorRecoveryWorkflow
```

## Common Use Cases

The guides cover these common scenarios:

1. **ETL Pipeline**: Use `DatabaseETL` with `DataHandlingMode.COPY` for transaction safety
2. **File Streaming**: Use `FileProcessor` with `ProcessingMode.STREAM` for large files
3. **API Orchestration**: Use `APIOrchestrator` with parallel execution modes
4. **Debugging**: Use `AdvancedFSM` with breakpoints and step-by-step execution
5. **CLI Workflow**: Use `fsm run` commands for development and testing

## Support

For additional help:

- Check the [FAQ](../faq.md) if it exists
- Review troubleshooting sections in each guide
- Consult the [API documentation](../../../api/dataknobs-fsm.md) for detailed method signatures
- See [Examples](../examples/index.md) for complete working code
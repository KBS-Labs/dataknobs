# CLI Guide

## Overview

The DataKnobs FSM package includes a comprehensive command-line interface (CLI) for managing, executing, and debugging FSM workflows. The CLI provides rich terminal output with tables, progress bars, and syntax highlighting using the Rich library.

## Installation

The FSM CLI is automatically installed with the package:

```bash
# Install with pip
pip install dataknobs-fsm

# Or with uv
uv pip install dataknobs-fsm

# Verify installation
fsm --version
```

The CLI is registered as the `fsm` command via the package's entry point.

## Command Structure

The FSM CLI uses a hierarchical command structure:

```
fsm [OPTIONS] COMMAND [ARGS]...

Commands:
  config   - FSM configuration management commands
  run      - Execute FSM operations
  debug    - Debug and profile FSM operations
  history  - Manage FSM execution history
  pattern  - Run pre-configured FSM patterns
```

## Global Options

```bash
# Show help
fsm --help

# Show version (0.1.0)
fsm --version
```

## Configuration Commands

### Create Configuration

Generate FSM configuration from templates:

```bash
# Create from template (available: basic, etl, workflow, processing)
fsm config create basic --output fsm_config.yaml
fsm config create etl --output etl_config.yaml --format yaml
fsm config create workflow --output workflow.json --format json

# Templates:
# - basic: Simple linear FSM with start->process->end states
# - etl: Database ETL pipeline with resources
# - workflow: Approval workflow with branching
# - processing: File processing pipeline
```

### Validate Configuration

Check configuration for errors:

```bash
# Validate configuration file
fsm config validate config.yaml

# Validate with verbose output
fsm config validate config.yaml --verbose
```

### Show Configuration

Display FSM structure:

```bash
# Show as tree (default)
fsm config show config.yaml
fsm config show config.yaml --format tree

# Show as table
fsm config show config.yaml --format table

# Show as Mermaid graph
fsm config show config.yaml --format graph
```

## Execution Commands

### Execute FSM

Execute FSM with data:

```bash
# Run with input file
fsm run execute config.yaml --data data.json

# Run with inline JSON
fsm run execute config.yaml --data '{"key": "value"}'

# Specify initial state
fsm run execute config.yaml --data data.json --initial-state validate

# Run with timeout (seconds)
fsm run execute config.yaml --data data.json --timeout 30

# Save output
fsm run execute config.yaml --data data.json --output results.json

# Verbose execution
fsm run execute config.yaml --data data.json --verbose
```

### Batch Processing

Process multiple records:

```bash
# Process batch data (JSON array or JSONL)
fsm run batch config.yaml data.json
fsm run batch config.yaml data.jsonl

# Set batch size
fsm run batch config.yaml data.json --batch-size 10

# Parallel workers
fsm run batch config.yaml data.json --workers 4

# Show progress bar
fsm run batch config.yaml data.json --progress

# Save results
fsm run batch config.yaml data.json --output results.json
```

### Stream Processing

Process streaming data:

```bash
# Stream from source
fsm run stream config.yaml source_file.json

# Stream with sink
fsm run stream config.yaml source_file.csv --sink output.csv

# Set chunk size
fsm run stream config.yaml large_file.json --chunk-size 100

# Specify format
fsm run stream config.yaml data.csv --format csv
fsm run stream config.yaml data.json --format json
```

## Debug Commands

### Debug Execution

Debug FSM execution with breakpoints and tracing:

```bash
# Run with debugging
fsm debug run config.yaml

# With input data
fsm debug run config.yaml --data data.json
fsm debug run config.yaml --data '{"test": true}'

# Set breakpoints at states
fsm debug run config.yaml --breakpoint validate --breakpoint process

# Enable tracing
fsm debug run config.yaml --data data.json --trace

# Enable profiling
fsm debug run config.yaml --data data.json --profile
```

The debug command creates an AdvancedFSM instance and can:
- Set breakpoints at specific states
- Trace execution flow
- Profile performance
- Start interactive debugging sessions

## History Commands

### List History

View execution history:

```bash
# List recent executions (default limit: 10)
fsm history list

# Filter by FSM name
fsm history list --fsm-name MyFSM

# Limit results
fsm history list --limit 20

# Output as JSON
fsm history list --format json
```

### Show Execution

View specific execution:

```bash
# Show execution by ID
fsm history show abc123def456

# Show with verbose details
fsm history show abc123def456 --verbose
```

History is stored in `~/.fsm/history/` using FileStorage backend.

## Pattern Commands

### ETL Pattern

Run ETL pipeline pattern:

```bash
# Basic ETL
fsm pattern etl \
  --source "sqlite:///source.db" \
  --target "sqlite:///target.db"

# ETL modes (full, incremental, upsert)
fsm pattern etl \
  --source db_connection \
  --target warehouse \
  --mode full

# With batch size
fsm pattern etl \
  --source source.db \
  --target target.db \
  --batch-size 1000

# Resume from checkpoint
fsm pattern etl \
  --source source.db \
  --target target.db \
  --checkpoint checkpoint_id
```

### File Processing Pattern

Process files using FSM pattern:

```bash
# Process file (CSV or JSON)
fsm pattern process-file input.csv

# With output file
fsm pattern process-file input.csv --output output.csv

# Specify format
fsm pattern process-file data.json --format json --output processed.json

# With transformations
fsm pattern process-file data.csv \
  --transform "uppercase,trim" \
  --filter "status==active"
```

## Advanced Usage

### Using SimpleFSM

The CLI uses SimpleFSM for basic execution:

```python
# Internally, the CLI does:
from dataknobs_fsm.api.simple import SimpleFSM

fsm = SimpleFSM(config_file)
result = fsm.process(
    data=input_data,
    initial_state=initial_state,
    timeout=timeout
)
```

### Using AdvancedFSM

For debugging, the CLI uses AdvancedFSM:

```python
# Debug mode uses:
from dataknobs_fsm.api.advanced import AdvancedFSM, FSMDebugger

fsm = AdvancedFSM(config)
fsm.set_breakpoint(state_name)

# Interactive debugging
debugger = FSMDebugger(fsm, config)
await debugger.start_session(input_data)
```

### Async Operations

Many operations run asynchronously:

```python
# Stream processing
async def run_stream():
    result = await fsm.process_stream(
        source=source,
        sink=sink,
        chunk_size=chunk_size
    )

asyncio.run(run_stream())
```

## Output Formatting

### Rich Terminal Output

The CLI uses Rich library for beautiful output:

```bash
# Tree view (config show)
FSM_Name
├── Network: main
│   ├── States
│   │   ├── start (start)
│   │   ├── process
│   │   └── end (end)
│   └── Arcs
│       ├── start → process [begin]
│       └── process → end [complete]
└── Resources
    └── database: database
```

### Table Output

```bash
# Table format (config show --format table)
╔════════════════════════════════════════╗
║ FSM_Name - States                      ║
╠═════════╦══════════╦══════════════════╣
║ Network ║ Name     ║ Type             ║
╠═════════╬══════════╬══════════════════╣
║ main    ║ start    ║ Start            ║
║ main    ║ process  ║ Normal           ║
║ main    ║ end      ║ End              ║
╚═════════╩══════════╩══════════════════╝
```

### Mermaid Graph Output

```bash
# Graph format (config show --format graph)
```mermaid
graph TD
    start([start])
    process[process]
    end((end))
    start -->|begin| process
    process -->|complete| end
```

## Progress and Status

### Progress Indicators

The CLI uses Rich Progress for visual feedback:

```bash
# Spinner for indeterminate progress
⠋ Executing FSM...
⠙ Loading configuration...
⠹ Processing stream...

# Progress bar for batch processing
fsm run batch config.yaml data.json --progress
# Processing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
```

### Console Output

Colored and formatted output:

```bash
# Success indicators
✓ Configuration is valid!
✓ Execution completed successfully!

# Error indicators
✗ Configuration validation failed!
✗ Execution failed!

# Information
  Name: MyFSM
  States: 5
  Arcs: 4
  Data Mode: copy
```

## Troubleshooting

### Common Errors

```bash
# Configuration errors
[red]Error loading configuration: Invalid YAML[/red]
# Fix: Validate YAML syntax

# JSON decode errors
[red]Invalid JSON data[/red]
# Fix: Check JSON format or use file path

# File not found
[red]Error loading configuration: File not found[/red]
# Fix: Check file path exists
```

### Debug Execution

```bash
# Enable tracing to see execution flow
fsm debug run config.yaml --data data.json --trace

# Enable profiling to find bottlenecks
fsm debug run config.yaml --data data.json --profile

# Set breakpoints for step-by-step debugging
fsm debug run config.yaml --breakpoint state_name
```

## Examples

### Example 1: Basic Configuration and Execution

```bash
# Create basic configuration
fsm config create basic --output my_fsm.yaml

# Validate configuration
fsm config validate my_fsm.yaml --verbose

# Show configuration structure
fsm config show my_fsm.yaml --format tree

# Execute with data
fsm run execute my_fsm.yaml --data '{"input": "test"}' --verbose
```

### Example 2: Batch Processing

```bash
# Create batch data file
cat > batch_data.json <<EOF
[
  {"id": 1, "value": "first"},
  {"id": 2, "value": "second"},
  {"id": 3, "value": "third"}
]
EOF

# Process batch
fsm run batch my_fsm.yaml batch_data.json \
  --batch-size 2 \
  --workers 2 \
  --progress \
  --output results.json

# Check results
cat results.json | jq '.[] | select(.success==true)'
```

### Example 3: ETL Pattern

```bash
# Create ETL configuration
fsm config create etl --output etl_pipeline.yaml

# Run ETL pipeline
fsm pattern etl \
  --source "sqlite:///source.db" \
  --target "sqlite:///target.db" \
  --mode full \
  --batch-size 1000

# Check execution history
fsm history list --fsm-name ETL_Pipeline --limit 5
```

### Example 4: Debug with Profiling

```bash
# Create test data
echo '{"test": true, "count": 100}' > test_data.json

# Run with profiling
fsm debug run my_fsm.yaml --data test_data.json --profile

# Output:
# Performance Profile:
#   Total time: 0.245s
#   Transitions: 3
#
# State Execution Times:
#   start: 0.010s
#   process: 0.200s
#   end: 0.035s
```

## Best Practices

### 1. Validate Before Running

Always validate configurations:

```bash
# Validate first
fsm config validate config.yaml

# Then run
fsm run execute config.yaml --data data.json
```

### 2. Use Verbose Mode for Debugging

```bash
# See detailed execution flow
fsm run execute config.yaml --data data.json --verbose

# Debug with tracing
fsm debug run config.yaml --data data.json --trace
```

### 3. Handle Errors Gracefully

The CLI uses sys.exit(1) on errors:

```bash
#!/bin/bash
if fsm run execute config.yaml --data data.json; then
    echo "Success"
else
    echo "Failed with exit code $?"
    # Handle error...
fi
```

### 4. Use Progress for Long Operations

```bash
# Show progress for batch processing
fsm run batch config.yaml large_data.json --progress

# Stream processing with visual feedback
fsm run stream config.yaml source.json --chunk-size 1000
```

## Implementation Details

### Key Components

The CLI is built with:
- **Click**: Command-line interface framework
- **Rich**: Terminal formatting and progress bars
- **SimpleFSM**: Basic FSM execution
- **AdvancedFSM**: Debug and profiling capabilities
- **ConfigLoader**: Configuration file loading
- **ExecutionHistory**: History tracking with FileStorage

### Template System

Built-in templates include:
- **basic**: Simple 3-state linear FSM
- **etl**: ETL pipeline with database resources
- **workflow**: Approval workflow with branching
- **processing**: File processing pipeline

### Async Execution

Many operations use asyncio:
- Stream processing
- Debug sessions
- ETL patterns
- File processing patterns

## Conclusion

The FSM CLI provides a comprehensive command-line interface for:

- **Configuration Management**: Create, validate, and visualize FSM configurations
- **Execution**: Run single, batch, or streaming FSM processes
- **Debugging**: Debug with breakpoints, tracing, and profiling
- **History**: Track and query execution history
- **Patterns**: Execute pre-built ETL and file processing patterns

The CLI uses Rich for beautiful terminal output and integrates with the SimpleFSM and AdvancedFSM APIs.

## Next Steps

- [SimpleFSM API](../api/simple.md) - Programmatic FSM usage
- [AdvancedFSM API](../api/advanced.md) - Advanced debugging and profiling
- [Pattern Catalog](../patterns/index.md) - ETL and file processing patterns
- [Examples](../examples/index.md) - Complete example workflows
# FSM Examples

This section contains practical examples showing how to use the FSM package for real-world data processing tasks. All examples use the actual FSM implementation with proper import paths and API usage.

## Available Examples

### Core Examples (In Repository)

These complete, runnable examples are located in `packages/fsm/examples/`:

#### 1. [Database ETL Pipeline](database-etl.md) (`database_etl.py`)
A comprehensive example showing how to build a production-ready ETL pipeline using the FSM framework.

**Key Features:**
- Uses `SimpleFSM` with `DataHandlingMode.COPY` for transaction safety
- Multi-stage data extraction, transformation, and loading
- Custom function registration for ETL operations
- Error handling with rollback states
- Batch processing with configurable size
- Data validation and quality checks
- Business metrics calculation (revenue, customer segments)

#### 2. Data Validation Pipeline (`data_validation_pipeline.py`)
Demonstrates a validation pipeline for data quality assurance.

**Key Features:**
- Schema validation
- Data type checking
- Business rule validation
- Error collection and reporting
- Configurable validation rules

#### 3. [End-to-End Streaming](end-to-end-streaming.md) (`end_to_end_streaming.py`)
Comprehensive demonstration of FSM's streaming capabilities for processing large datasets.

**Key Features:**
- True end-to-end streaming through FSM states
- File-to-file streaming with transformations
- Real-time stream processing from generators
- Multi-stage streaming pipelines
- Memory-efficient chunk-based processing
- Automatic backpressure management
- Progress tracking and monitoring

#### 4. Large File Processor (`large_file_processor.py`)
Shows how to process large files efficiently using streaming.

**Key Features:**
- Streaming mode for memory efficiency
- Chunk-based processing
- Progress tracking
- Error recovery for partial failures

#### 5. Advanced Debugging Examples (`advanced_debugging.py`, `advanced_debugging_simple.py`)
Demonstrates the `AdvancedFSM` debugging capabilities.

**Key Features:**
- Step-by-step execution
- Breakpoint debugging
- Execution hooks and monitoring
- State inspection
- Performance profiling

#### 6. LLM Conversation System (`llm_conversation.py`)
An FSM-based conversational AI system.

**Key Features:**
- Conversation state management
- Context handling
- Multi-turn dialogue support
- Intent recognition states
- Response generation workflow

#### 7. [Text Normalization with Regex](regex-transformations.md) (`normalize_file_example.py`, `normalize_file_with_regex.py`)
Powerful text processing using regular expressions directly in YAML configurations.

**Key Features:**
- Regular expressions in inline transforms
- Field preservation pattern
- Whitespace and punctuation normalization
- Email and URL pattern matching
- Phone number and SSN masking
- Duplicate word removal
- Multiple format conversions (snake_case, kebab-case, CamelCase)
- Pattern extraction and detection

#### 8. Regex Transform Configurations (`regex_transforms.yaml`, `regex_workflow.yaml`)
Ready-to-use YAML configurations for text processing.

**Key Features:**
- Field-by-field transformations
- All-in-one transform patterns
- Sensitive data masking
- Pattern extraction (emails, URLs, hashtags, mentions)
- Format standardization

### Documentation Examples (Guides)

#### [File Processing Workflow](file-processor.md)
Detailed guide on file processing patterns.

#### [API Orchestration](api-workflow.md)
Guide for coordinating multiple API calls.

#### [LLM Conversation](llm-conversation.md)
Building conversation systems with FSM.

#### [LLM Chain Processing](llm-chain.md)
Multi-step LLM processing chains (to be implemented).

## Quick Start Examples

### Basic FSM with SimpleFSM

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Define configuration
config = {
    "name": "simple_example",
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

### Debugging with AdvancedFSM

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

## Example Features

Each example demonstrates:

- **Correct API Usage**: Using `SimpleFSM` or `AdvancedFSM` properly
- **Data Handling Modes**: When to use `COPY`, `REFERENCE`, or `DIRECT`
- **Custom Functions**: Registering and using custom transform functions
- **Error Handling**: Proper error states and recovery
- **Real-world Patterns**: Practical solutions to common problems

## Running Examples

Examples are located in the `packages/fsm/examples/` directory:

```bash
# Navigate to the FSM package
cd packages/fsm

# Run the database ETL example
uv run python examples/database_etl.py

# Run the data validation pipeline
uv run python examples/data_validation_pipeline.py

# Run the large file processor
uv run python examples/large_file_processor.py

# Run advanced debugging example
uv run python examples/advanced_debugging.py

# Run LLM conversation example
uv run python examples/llm_conversation.py

# Run text normalization example
uv run python examples/normalize_file_example.py

# Run regex transformation examples
uv run python examples/normalize_file_with_regex.py

# Test regex YAML configurations
uv run python examples/test_regex_yaml.py
```

### Running with Custom Parameters

Most examples accept command-line arguments:

```bash
# Database ETL with custom batch size
uv run python examples/database_etl.py --batch-size 500

# File processor with specific input
uv run python examples/large_file_processor.py --input data.csv --output processed.json

# Debugging with specific config
uv run python examples/advanced_debugging.py --config custom_fsm.yaml

# Process file with regex transformations
uv run python examples/normalize_file_example.py
```

## Prerequisites

Before running the examples, ensure you have:

1. **FSM package installed**:
   ```bash
   pip install dataknobs-fsm
   # Or for development
   pip install -e packages/fsm
   ```

2. **Required dependencies for specific examples**:
   ```bash
   # For database examples
   pip install sqlite3

   # For LLM examples
   pip install openai anthropic

   # For file processing
   pip install pandas pyarrow
   ```

3. **Environment setup**:
   - For LLM examples: Set API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
   - For database examples: SQLite is used by default (no setup needed)

## Key Concepts Demonstrated

The examples showcase important FSM concepts:

### Data Handling Modes
- **COPY Mode**: Used in `database_etl.py` for transaction safety
- **REFERENCE Mode**: Used in `large_file_processor.py` for memory efficiency
- **DIRECT Mode**: Shown in performance-critical sections

### Custom Functions
```python
# Register custom functions with SimpleFSM
def transform_data(state):
    data = state.data.copy()
    # Transform logic
    return data

fsm = SimpleFSM(
    config,
    custom_functions={"transform": transform_data}
)
```

### Error Handling
```python
# Configuration with error states
config = {
    "states": [
        {"name": "process"},
        {"name": "error"},
        {"name": "rollback"}
    ],
    "arcs": [
        {
            "from": "process",
            "to": "error",
            "pre_test": {
                "type": "inline",
                "code": "lambda data, ctx: 'error' in data"
            }
        }
    ]
}
```

### Regular Expressions in YAML
```yaml
# Use regex directly in inline transforms
arcs:
  - from: start
    to: normalize
    transform:
      type: inline
      code: |
        lambda data, ctx: {
            **data,
            'normalized': __import__('re').sub(
                r'\\s+',           # Pattern: multiple spaces
                ' ',               # Replacement: single space
                data.get('text', '')
            ).strip()
        }
```

## Learn More

- [API Reference](../api/index.md) - Complete API documentation
- [Patterns Guide](../patterns/index.md) - Pre-built integration patterns
- [Data Modes Guide](../guides/data-modes.md) - Understanding data handling
- [Contributing Guide](../../../development/contributing.md) - Submit your examples
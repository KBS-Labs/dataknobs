# FSM Examples Documentation

This directory contains comprehensive documentation and examples for the FSM (Finite State Machine) package.

## Example Files in Repository

### Python Examples (`packages/fsm/examples/`)

1. **database_etl.py** - Complete ETL pipeline with transaction management
2. **data_validation_pipeline.py** - Data quality validation workflow
3. **large_file_processor.py** - Memory-efficient large file processing
4. **advanced_debugging.py** - Full debugging features demonstration
5. **advanced_debugging_simple.py** - Simplified debugging example
6. **llm_conversation.py** - Conversational AI system using FSM
7. **normalize_file_example.py** - Text file normalization with streaming
8. **normalize_file_with_regex.py** - Advanced regex transformations
9. **test_regex_yaml.py** - Testing script for YAML regex configurations

### YAML Configuration Examples (`packages/fsm/examples/`)

1. **regex_transforms.yaml** - Field transformation workflows
2. **regex_workflow.yaml** - Pattern extraction and masking configurations

## Documentation Files

### Main Documentation
- **index.md** - Complete overview of all examples with quick start guides
- **regex-transformations.md** - Comprehensive guide to using regex in FSM

### Specific Workflow Guides
- **database-etl.md** - Detailed ETL pipeline documentation
- **file-processor.md** - File processing patterns and best practices
- **api-workflow.md** - API orchestration patterns
- **llm-conversation.md** - Building conversational systems
- **llm-chain.md** - LLM processing chains (placeholder)

## Running Examples

All examples should be run from the FSM package directory using `uv`:

```bash
cd packages/fsm
uv run python examples/<example_name>.py
```

## Testing

Unit tests for examples are located in:
- `packages/fsm/tests/test_regex_examples.py` - Tests for regex transformations

Run tests with:
```bash
cd packages/fsm
uv run pytest tests/test_regex_examples.py -v
```

## Key Features Demonstrated

### Core FSM Concepts
- State machines with start and end states
- Arc transitions with conditions and transforms
- Custom function registration
- Error handling and recovery

### Data Processing
- **COPY mode** - Safe concurrent processing
- **REFERENCE mode** - Memory-efficient processing
- **DIRECT mode** - High-performance operations
- Batch processing
- Stream processing

### Advanced Features
- Regular expressions in YAML configurations
- Field preservation patterns
- Pattern extraction and masking
- Step-by-step debugging
- Performance profiling
- Execution hooks

### Text Processing with Regex
- Direct use of `__import__('re')` in YAML
- Multiple transformation pipelines
- Sensitive data masking
- Format conversions (snake_case, CamelCase, etc.)
- Pattern detection and extraction

## Documentation Standards

Each example follows these standards:
1. Clear docstrings explaining purpose
2. Runnable code with proper imports
3. Example output shown
4. Error handling demonstrated
5. Best practices highlighted

## Contributing

To add new examples:
1. Create the example file in `packages/fsm/examples/`
2. Add unit tests in `packages/fsm/tests/`
3. Create documentation in `docs/packages/fsm/examples/`
4. Update `index.md` with the new example
5. Ensure example runs with `uv run python`
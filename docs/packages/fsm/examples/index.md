# FSM Examples

This section contains practical examples showing how to use the FSM package for real-world data processing tasks.

## Available Examples

### [Database ETL Pipeline](database-etl.md)
A comprehensive example showing how to build a production-ready ETL pipeline using the FSM framework. This example demonstrates:

- Data extraction from source databases
- Multi-stage data transformations
- Data validation and quality checks
- Loading data into target systems
- Error handling and progress monitoring
- Business metrics calculation

### [File Processing Workflow](file-processor.md) 
*(Coming Soon)*

### [API Orchestration](api-workflow.md)
*(Coming Soon)*

### [LLM Chain Processing](llm-chain.md)
*(Coming Soon)*

## Getting Started

Each example includes:

- **Complete source code** with detailed comments
- **Step-by-step explanation** of the implementation
- **Configuration examples** for different scenarios
- **Unit tests** to verify functionality
- **Performance considerations** and best practices

## Example Structure

All examples follow a consistent structure:

```
example_name/
├── README.md          # Overview and quick start
├── example.py         # Main implementation
├── config.yaml        # Example configuration
├── tests/            # Unit tests
│   └── test_example.py
└── docs/             # Detailed documentation
    └── guide.md
```

## Running Examples

Examples are located in the `packages/fsm/examples/` directory. To run an example:

```bash
# Navigate to the FSM package
cd packages/fsm

# Run an example
python examples/database_etl.py

# Run with custom configuration
python examples/database_etl.py --config my_config.yaml

# Run tests for an example
uv run pytest tests/test_database_etl_example.py
```

## Prerequisites

Before running the examples, ensure you have:

1. **FSM package installed**: `pip install dataknobs-fsm`
2. **Required dependencies**: Check each example's requirements
3. **Test data or databases**: Some examples may require setup

## Contributing Examples

We welcome contributions of new examples! When submitting an example:

1. Follow the established structure above
2. Include comprehensive tests
3. Document all configuration options
4. Provide performance benchmarks where applicable
5. Include error handling and edge cases

See our [Contributing Guide](../../../development/contributing.md) for more details.
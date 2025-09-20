# Data Processing Pipeline

This example demonstrates how to build a robust data processing pipeline using the FSM framework with real-world features including data validation, enrichment, aggregation, and resource management.

## Overview

The data pipeline example shows how to:
- Create a simple FSM for data processing workflows
- Implement custom transform functions for validation, enrichment, and aggregation
- Use resource management for tracking processing statistics
- Handle errors gracefully during pipeline execution
- Process records through a complete transformation pipeline

## Key Components

### Custom Transform Functions

The example implements three transform functions that work together in a pipeline:

#### DataValidator
Validates incoming data records to ensure they have required fields and correct data types:
```python
class DataValidator(ITransformFunction):
    def transform(self, data: Any, context: FunctionContext) -> Any:
        # Check required fields
        required_fields = ['id', 'timestamp', 'value']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate data types
        if not isinstance(data['value'], (int, float)):
            raise ValueError("Value must be numeric")

        # Add validation flag
        data['validated'] = True
        return data
```

#### DataEnricher
Enriches data with computed fields and metadata:
```python
class DataEnricher(ITransformFunction):
    def transform(self, data: Any, context: FunctionContext) -> Any:
        # Add computed fields
        data['value_squared'] = data['value'] ** 2
        data['value_multiplied'] = data['value'] * self.multiplier
        data['value_category'] = self._categorize_value(data['value'])

        # Track processing stats using resources
        if 'properties' in context.resources:
            props = context.resources['properties']
            count = props.get('processed_count', 0)
            props.set('processed_count', count + 1)

        return data
```

#### DataAggregator
Aggregates data into summary statistics:
```python
class DataAggregator(ITransformFunction):
    def transform(self, data: Any, context: FunctionContext) -> Any:
        # Calculate aggregations
        total = sum(r.get('value', 0) for r in records)
        count = len(records)
        avg = total / count if count > 0 else 0

        return {
            'type': 'aggregation',
            'count': count,
            'total': total,
            'average': avg,
            'min': min_val,
            'max': max_val
        }
```

### FSM Configuration

The pipeline FSM uses a simple three-state network:

```python
def create_simple_pipeline_fsm() -> FSM:
    # Create FSM
    fsm = FSM(name="data_pipeline")

    # Register custom functions
    func_manager = FunctionManager()
    func_manager.register_function('validate', DataValidator())
    func_manager.register_function('enrich', DataEnricher(multiplier=3))
    func_manager.register_function('aggregate', DataAggregator())

    # Create network with states
    network = StateNetwork(name="main")
    network.add_state(State(name="start", type="start"), initial=True)
    network.add_state(State(name="process", type="normal"))
    network.add_state(State(name="end", type="end"), final=True)

    # Connect states with arcs
    network.add_arc("start", "process")
    network.add_arc("process", "end")

    fsm.add_network(network)
    return fsm
```

### Resource Management

The example uses the PropertiesResource to track processing statistics:

```python
# Create resources
resource_manager = ResourceManager()

# Add properties resource for tracking
props_resource = PropertiesResource(
    name='properties',
    initial_properties={
        'source': 'example_pipeline',
        'processed_count': 0
    }
)
resource_manager.register_provider('properties', props_resource)
fsm.resource_manager = resource_manager

# Use resources during processing
props_handle = resource_manager.acquire('properties', f'record_{record["id"]}')
context.resources = {'properties': props_handle}
```

### Processing Records

The pipeline processes records individually through the FSM:

```python
# Test data
test_records = [
    {'id': 1, 'timestamp': '2024-01-01', 'value': 25.0},
    {'id': 2, 'timestamp': '2024-01-02', 'value': 50.0},
    {'id': 3, 'timestamp': '2024-01-03', 'value': 75.0}
]

# Process each record
engine = ExecutionEngine(fsm)
results = []

for record in test_records:
    context = ExecutionContext(data_mode=ProcessingMode.SINGLE)

    # Acquire resources
    props_handle = resource_manager.acquire('properties', f'record_{record["id"]}')
    context.resources = {'properties': props_handle}

    try:
        success, result = engine.execute(context, record)
        if success:
            results.append(result)
    finally:
        # Release resources
        resource_manager.release('properties', props_handle)
```

## Running the Example

To run the data pipeline example:

```bash
cd packages/fsm
python examples/data_pipeline_example.py
```

Expected output:
```
=== Simple Data Pipeline FSM Example ===

1. Building FSM...
   FSM 'data_pipeline' created

2. Setting up resources...
   Resources configured

3. Processing sample data...
   Processed record 1: value=25.0
   Processed record 2: value=50.0
   Processed record 3: value=75.0

   Successfully processed 3 records

4. Sample Result:
   Original value: 25.0
   Validated: True
   Category: medium

=== Example Complete ===
```

## Key Features Demonstrated

### 1. Transform Function Interface
The example shows how to implement the `ITransformFunction` interface with proper method signatures:
- `transform()` method for data processing
- `get_transform_description()` for documentation
- Access to `FunctionContext` for resources and metadata

### 2. Resource Management Pattern
Demonstrates proper resource lifecycle:
- Register resource providers with ResourceManager
- Acquire resources before processing
- Access resources through context during processing
- Release resources after processing (even on error)

### 3. Error Handling
Shows robust error handling patterns:
- Validation errors for missing fields
- Type checking and data validation
- Try/finally blocks for resource cleanup
- Graceful error reporting

### 4. Data Enrichment Patterns
Illustrates common data processing patterns:
- Adding computed fields (squared, multiplied values)
- Categorizing data based on ranges
- Adding processing timestamps
- Tracking processing statistics

## Integration with Testing

The example includes comprehensive tests in `test_data_pipeline_example.py`:

```python
def test_pipeline_with_valid_data():
    """Test pipeline with valid input data."""
    fsm = create_simple_pipeline_fsm()
    engine = ExecutionEngine(fsm)

    test_data = {'id': 1, 'timestamp': '2024-01-01', 'value': 42.0}
    context = ExecutionContext(data_mode=ProcessingMode.SINGLE)

    success, result = engine.execute(context, test_data)

    assert success
    assert result['validated'] is True
    assert result['value_squared'] == 1764.0
    assert result['value_category'] == 'medium'
```

## Use Cases

This example is ideal for:
- ETL (Extract, Transform, Load) pipelines
- Data quality validation workflows
- Real-time data processing streams
- Batch data transformation jobs
- IoT sensor data processing
- Log processing and analysis

## Next Steps

To extend this example for your use case:

1. **Add Custom Functions**: Create additional transform functions for your specific data processing needs
2. **Configure State Networks**: Design more complex state networks with conditional transitions
3. **Integrate Storage**: Add database or file storage sinks for processed data
4. **Add Monitoring**: Integrate metrics collection and alerting
5. **Scale Processing**: Use stream processing for larger datasets
6. **Add Error Recovery**: Implement retry logic and dead letter queues

## Related Examples

- [End-to-End Streaming](end-to-end-streaming.md) - For high-volume stream processing
- [Database ETL](database-etl.md) - For database-specific ETL patterns
- [File Processor](file-processor.md) - For file-based data processing

## API References

- [ITransformFunction](../api/functions.md#itransformfunction) - Transform function interface
- [FunctionManager](../api/functions.md#functionmanager) - Function registration and management
- [ResourceManager](../api/resources.md#resourcemanager) - Resource lifecycle management
- [ExecutionEngine](../api/execution.md#executionengine) - FSM execution engine
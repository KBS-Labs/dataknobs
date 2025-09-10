# FSM Integration Patterns

The FSM package includes pre-built patterns for common integration scenarios. These patterns provide tested, production-ready solutions that you can use directly or customize for your needs.

## Available Patterns

### 1. [ETL Pattern](etl.md)
Extract, Transform, and Load workflows for data processing pipelines.

**Use Cases:**
- Database migrations
- Data warehouse loading
- Batch data processing
- Data synchronization

**Features:**
- Configurable extractors and loaders
- Built-in transformations
- Error handling and retries
- Progress tracking

[Learn more →](etl.md)

### 2. [File Processing Pattern](file-processing.md)
Process files of various formats with streaming support.

**Use Cases:**
- Batch file conversion
- Log file analysis
- Document processing
- Media file handling

**Features:**
- Multiple format support (CSV, JSON, XML, Parquet)
- Streaming for large files
- Parallel processing
- Format conversion

[Learn more →](file-processing.md)

### 3. [API Orchestration Pattern](api-orchestration.md)
Coordinate multiple API calls with advanced features.

**Use Cases:**
- Microservice orchestration
- Third-party API integration
- Data aggregation from multiple sources
- Complex API workflows

**Features:**
- Rate limiting
- Circuit breakers
- Retry strategies
- Request/response transformation

[Learn more →](api-orchestration.md)

### 4. [LLM Workflow Pattern](llm-workflow.md)
Build sophisticated LLM-powered workflows.

**Use Cases:**
- Conversational AI
- Document analysis
- Content generation
- Chain-of-thought reasoning

**Features:**
- Multiple LLM provider support
- RAG (Retrieval Augmented Generation)
- Chain workflows
- Token management

[Learn more →](llm-workflow.md)

### 5. [Error Recovery Pattern](error-recovery.md)
Implement robust error handling and recovery strategies.

**Use Cases:**
- Fault-tolerant systems
- Distributed transactions
- Critical workflows
- High-availability services

**Features:**
- Retry mechanisms
- Circuit breakers
- Fallback strategies
- Compensation logic

[Learn more →](error-recovery.md)

## Quick Start

Each pattern can be used in two ways:

### 1. Direct Usage

```python
from dataknobs_fsm.patterns import ETLPattern

# Create pattern instance
etl = ETLPattern(
    name="data_migration",
    source_config={
        "type": "database",
        "connection": "postgresql://source_db"
    },
    target_config={
        "type": "database", 
        "connection": "postgresql://target_db"
    }
)

# Configure transformations
etl.add_transformation(lambda row: {
    **row,
    "processed_at": datetime.now()
})

# Execute
result = etl.run({
    "source_table": "users",
    "target_table": "users_archive"
})
```

### 2. Configuration-Based

```yaml
# etl_workflow.yaml
pattern: etl
name: data_migration

source:
  type: database
  provider: postgresql
  config:
    host: source.example.com
    database: app_db
    
target:
  type: database
  provider: postgresql
  config:
    host: target.example.com
    database: warehouse

transformations:
  - type: rename_fields
    mapping:
      old_field: new_field
  - type: add_timestamp
    field: processed_at

options:
  batch_size: 1000
  parallel: true
  on_error: continue
```

```python
from dataknobs_fsm.patterns import load_pattern

pattern = load_pattern("etl_workflow.yaml")
result = pattern.run()
```

## Pattern Composition

Patterns can be combined to create complex workflows:

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_fsm.patterns import (
    FileProcessingPattern,
    ETLPattern,
    ErrorRecoveryPattern
)

# Create main FSM
fsm = SimpleFSM(name="complex_workflow")

# Add patterns as components
file_processor = FileProcessingPattern(...)
etl = ETLPattern(...)
error_handler = ErrorRecoveryPattern(...)

# Compose workflow
fsm.add_state("start", initial=True)
fsm.add_state("process_files")
fsm.add_state("load_data")
fsm.add_state("complete", terminal=True)

fsm.add_transition(
    "start", "process_files",
    function=file_processor.process
)

fsm.add_transition(
    "process_files", "load_data",
    function=etl.run,
    on_error=error_handler.handle
)

fsm.add_transition("load_data", "complete")

# Execute composed workflow
result = fsm.run({"input_dir": "/data/incoming"})
```

## Customizing Patterns

All patterns are designed to be extensible:

```python
from dataknobs_fsm.patterns import APIOrchestrationPattern

class CustomAPIPattern(APIOrchestrationPattern):
    """Extended API pattern with custom authentication."""
    
    def authenticate(self, credentials):
        """Custom authentication logic."""
        token = self.get_oauth_token(credentials)
        self.set_header("Authorization", f"Bearer {token}")
        
    def transform_response(self, response):
        """Custom response transformation."""
        data = super().transform_response(response)
        return self.apply_business_rules(data)
```

## Pattern Selection Guide

Choose the right pattern based on your needs:

| Pattern | Best For | Key Features | Performance |
|---------|----------|--------------|-------------|
| **ETL** | Data pipelines | Batch processing, transformations | High throughput |
| **File Processing** | File operations | Format conversion, streaming | Memory efficient |
| **API Orchestration** | Service integration | Rate limiting, resilience | Concurrent requests |
| **LLM Workflow** | AI applications | Multi-provider, chaining | Token optimized |
| **Error Recovery** | Critical systems | Retry, fallback, compensation | High availability |

## Common Configurations

### Batch Processing
```yaml
options:
  batch_size: 1000
  parallel_workers: 4
  memory_limit: "1GB"
```

### Streaming
```yaml
options:
  stream: true
  buffer_size: 8192
  backpressure: true
```

### Error Handling
```yaml
error_handling:
  strategy: exponential_backoff
  max_retries: 3
  timeout: 30
  on_failure: compensate
```

## Performance Considerations

1. **Batch Size**: Larger batches improve throughput but use more memory
2. **Parallelism**: More workers increase speed but may overwhelm resources
3. **Streaming**: Use for large datasets to maintain constant memory usage
4. **Caching**: Enable for repeated operations on same data
5. **Connection Pooling**: Reuse connections for database/API patterns

## Monitoring and Observability

All patterns include built-in monitoring:

```python
pattern = ETLPattern(
    name="monitored_etl",
    monitoring={
        "metrics": True,
        "logging": "INFO",
        "tracing": True
    }
)

# Access metrics
metrics = pattern.get_metrics()
print(f"Records processed: {metrics['records_processed']}")
print(f"Error rate: {metrics['error_rate']}")
```

## Testing Patterns

Test patterns with mock data:

```python
from dataknobs_fsm.patterns.testing import PatternTestCase

class TestETLPattern(PatternTestCase):
    def test_transformation(self):
        pattern = ETLPattern(...)
        result = self.run_pattern(
            pattern,
            mock_data=[
                {"id": 1, "value": 10},
                {"id": 2, "value": 20}
            ]
        )
        self.assert_transformed(result, expected_count=2)
```

## Next Steps

- Explore individual [pattern documentation](etl.md)
- Check out [examples](../examples/index.md) using patterns
- Learn about [pattern composition](../guides/composition.md)
- Read about [performance tuning](../guides/performance.md)
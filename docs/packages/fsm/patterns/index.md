# FSM Integration Patterns

The FSM package includes pre-built patterns for common integration scenarios. These patterns provide tested, production-ready solutions that you can use directly or customize for your needs.

**Note:** Pattern classes must be imported directly from their respective modules as they are not exported at the package level.

## Available Patterns

### 1. [Database ETL Pattern](etl.md)
Database-focused Extract, Transform, and Load workflows for data processing pipelines.

**Class:** `DatabaseETL`
**Import:** `from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode`

**Use Cases:**
- Database migrations
- Data warehouse loading
- Database synchronization
- Incremental data updates

**Features:**
- Multiple ETL modes: `FULL_REFRESH`, `INCREMENTAL`, `UPSERT`, `APPEND`
- Checkpoint support for resumable operations
- Configurable batch sizes and parallelism
- Built-in transformations and field mappings

[Learn more →](etl.md)

### 2. [File Processing Pattern](file-processing.md)
Process files of various formats with streaming support.

**Class:** `FileProcessor`
**Import:** `from dataknobs_fsm.patterns.file_processing import FileProcessor, FileProcessingConfig`

**Use Cases:**
- CSV data processing
- JSON stream processing
- Log file analysis
- Batch file operations

**Features:**
- Format support: CSV, JSON, XML, Parquet, TXT/LOG
- Processing modes: `STREAM`, `BATCH`, `WHOLE`
- Automatic format detection
- Configurable transformations and filters
- Memory-efficient streaming for large files

[Learn more →](file-processing.md)

### 3. [API Orchestration Pattern](api-orchestration.md)
Coordinate multiple API calls with advanced features.

**Class:** `APIOrchestrator`
**Import:** `from dataknobs_fsm.patterns.api_orchestration import APIOrchestrator, APIOrchestrationConfig`

**Use Cases:**
- REST API orchestration
- GraphQL query coordination
- Microservice workflows
- Multi-API data aggregation

**Features:**
- Orchestration modes: `SEQUENTIAL`, `PARALLEL`, `FANOUT`, `PIPELINE`, `CONDITIONAL`, `HYBRID`
- Built-in rate limiting and throttling
- Automatic retry with backoff
- Request/response transformation
- Authentication handling

[Learn more →](api-orchestration.md)

### 4. [LLM Workflow Pattern](llm-workflow.md)
Build sophisticated LLM-powered workflows.

**Class:** `LLMWorkflow`
**Import:** `from dataknobs_fsm.patterns.llm_workflow import LLMWorkflow, LLMWorkflowConfig`

**Use Cases:**
- Simple prompt-response workflows
- Chain-of-thought reasoning
- RAG (Retrieval Augmented Generation)
- Multi-agent systems

**Features:**
- Workflow types: `SIMPLE`, `CHAIN`, `RAG`, `COT`, `TREE`, `AGENT`, `MULTI_AGENT`
- Multiple LLM provider support (OpenAI, Anthropic, HuggingFace)
- Document indexing for RAG
- Token usage tracking and optimization
- Prompt templating and chaining

[Learn more →](llm-workflow.md)

### 5. [Error Recovery Pattern](error-recovery.md)
Implement robust error handling and recovery strategies.

**Class:** `ErrorRecoveryWorkflow`
**Import:** `from dataknobs_fsm.patterns.error_recovery import ErrorRecoveryWorkflow, ErrorRecoveryConfig`

**Use Cases:**
- Fault-tolerant systems
- Resilient API calls
- Critical workflows
- High-availability services

**Features:**
- Recovery strategies: `RETRY`, `CIRCUIT_BREAKER`, `FALLBACK`, `COMPENSATE`, `DEADLINE`, `BULKHEAD`, `CACHE`
- Backoff strategies: `FIXED`, `LINEAR`, `EXPONENTIAL`, `RANDOM`
- Metrics tracking and monitoring
- Configurable failure thresholds
- Compensation and rollback support

[Learn more →](error-recovery.md)

## Quick Start

Each pattern can be used in two ways:

### 1. Direct Usage with Factory Functions

```python
from dataknobs_fsm.patterns.etl import create_etl_pipeline, ETLMode

# Create ETL pipeline using factory function
etl = create_etl_pipeline(
    source={
        "type": "database",
        "provider": "postgresql",
        "connection": "postgresql://source_db"
    },
    target={
        "type": "database",
        "provider": "postgresql",
        "connection": "postgresql://target_db"
    },
    mode=ETLMode.INCREMENTAL,
    transformations=[
        lambda row: {**row, "processed_at": datetime.now()}
    ]
)

# Execute asynchronously
import asyncio
result = asyncio.run(etl.run())
```

### 2. Direct Class Instantiation

```python
from dataknobs_fsm.patterns.file_processing import FileProcessor, FileProcessingConfig, ProcessingMode

# Create configuration
config = FileProcessingConfig(
    input_path="data.csv",
    output_path="processed.json",
    mode=ProcessingMode.STREAM,
    transformations=[
        lambda record: {**record, "processed": True}
    ]
)

# Create and execute processor
processor = FileProcessor(config)
result = asyncio.run(processor.process())
```

### 3. Using Multiple Factory Functions

```python
# File Processing
from dataknobs_fsm.patterns.file_processing import create_csv_processor

csv_processor = create_csv_processor(
    input_file="data.csv",
    output_file="output.json",
    transformations=[lambda row: {**row, "status": "processed"}],
    filters=[lambda row: row.get("active") == True]
)

# API Orchestration
from dataknobs_fsm.patterns.api_orchestration import create_rest_api_orchestrator
from dataknobs_fsm.patterns.api_orchestration import OrchestrationMode

api_orchestrator = create_rest_api_orchestrator(
    base_url="https://api.example.com",
    endpoints=[
        {"name": "users", "path": "/users", "method": "GET"},
        {"name": "posts", "path": "/posts", "method": "GET"}
    ],
    auth_token="your-api-token",
    rate_limit=100,  # requests per second
    mode=OrchestrationMode.PARALLEL
)

# LLM Workflow
from dataknobs_fsm.patterns.llm_workflow import create_simple_llm_workflow

llm_workflow = create_simple_llm_workflow(
    prompt_template="Summarize this text: {text}",
    model="gpt-4",
    provider="openai",
    temperature=0.7
)

# Error Recovery
from dataknobs_fsm.patterns.error_recovery import create_retry_workflow
from dataknobs_fsm.patterns.error_recovery import BackoffStrategy

retry_workflow = create_retry_workflow(
    max_attempts=3,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    initial_delay=1.0,
    max_delay=60.0
)
```

## Pattern Composition

Patterns can be combined using async orchestration:

```python
import asyncio
from dataknobs_fsm.patterns.file_processing import create_csv_processor
from dataknobs_fsm.patterns.etl import create_etl_pipeline, ETLMode
from dataknobs_fsm.patterns.error_recovery import create_retry_workflow

async def complex_workflow():
    """Compose multiple patterns in a workflow."""

    # Step 1: Process CSV file with retry logic
    csv_processor = create_csv_processor(
        input_file="raw_data.csv",
        output_file="processed.json",
        transformations=[lambda row: {**row, "validated": True}]
    )

    retry_wrapper = create_retry_workflow(
        max_attempts=3,
        backoff_strategy="exponential"
    )

    # Execute with retry
    file_result = await retry_wrapper.execute(
        csv_processor.process
    )

    # Step 2: Load to database
    etl = create_etl_pipeline(
        source={"type": "file", "path": "processed.json"},
        target={"type": "database", "connection": "postgresql://db"},
        mode=ETLMode.UPSERT
    )

    etl_result = await etl.run()

    return {
        "file_processing": file_result,
        "etl": etl_result
    }

# Run the composed workflow
result = asyncio.run(complex_workflow())
```

## Customizing Patterns

All patterns are designed to be extensible through configuration:

```python
from dataknobs_fsm.patterns.api_orchestration import APIOrchestrator, APIOrchestrationConfig
from dataknobs_fsm.patterns.api_orchestration import OrchestrationMode, APIEndpoint

# Create custom configuration
config = APIOrchestrationConfig(
    name="custom_api_workflow",
    mode=OrchestrationMode.HYBRID,
    endpoints=[
        APIEndpoint(
            name="auth",
            url="https://api.example.com/auth",
            method="POST",
            headers={"Content-Type": "application/json"},
            retry_config={"max_attempts": 5}
        ),
        APIEndpoint(
            name="data",
            url="https://api.example.com/data",
            method="GET",
            depends_on=["auth"],  # Sequential dependency
            transform=lambda resp: resp.get("data", [])
        )
    ],
    rate_limit=100,
    timeout=30.0
)

# Create orchestrator with custom config
orchestrator = APIOrchestrator(config)
result = await orchestrator.orchestrate({"user": "test"})
```

## Pattern Selection Guide

Choose the right pattern based on your needs:

| Pattern | Class Name | Best For | Key Modes/Strategies |
|---------|------------|----------|---------------------|
| **Database ETL** | `DatabaseETL` | Database operations | FULL_REFRESH, INCREMENTAL, UPSERT, APPEND |
| **File Processing** | `FileProcessor` | File operations | STREAM, BATCH, WHOLE |
| **API Orchestration** | `APIOrchestrator` | API workflows | SEQUENTIAL, PARALLEL, FANOUT, PIPELINE |
| **LLM Workflow** | `LLMWorkflow` | AI applications | SIMPLE, CHAIN, RAG, AGENT |
| **Error Recovery** | `ErrorRecoveryWorkflow` | Resilience | RETRY, CIRCUIT_BREAKER, FALLBACK |

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

## Monitoring and Metrics

Patterns with built-in metrics support:

```python
from dataknobs_fsm.patterns.error_recovery import ErrorRecoveryWorkflow, ErrorRecoveryConfig

# Create workflow with monitoring
config = ErrorRecoveryConfig(
    name="monitored_workflow",
    primary_strategy="retry",
    retry_config={
        "max_attempts": 3,
        "backoff_strategy": "exponential"
    }
)

workflow = ErrorRecoveryWorkflow(config)

# Execute and get metrics
await workflow.execute(some_function, arg1, arg2)
metrics = workflow.get_metrics()

print(f"Total attempts: {metrics['total_attempts']}")
print(f"Success rate: {metrics['success_rate']}")
print(f"Average retry count: {metrics['avg_retry_count']}")
```

## Example: Complete ETL Pipeline

```python
import asyncio
from dataknobs_fsm.patterns.etl import create_data_migration

async def run_migration():
    """Example of a complete data migration."""

    # Create migration with field mappings and transformations
    migration = create_data_migration(
        source={
            "provider": "postgresql",
            "connection": "postgresql://source/db",
            "table": "users"
        },
        target={
            "provider": "postgresql",
            "connection": "postgresql://target/db",
            "table": "users_v2"
        },
        field_mappings={
            "user_id": "id",
            "user_name": "name",
            "user_email": "email"
        },
        transformations=[
            lambda record: {
                **record,
                "migrated_at": datetime.now(),
                "version": "2.0"
            }
        ]
    )

    # Run with checkpoint for resumability
    result = await migration.run(checkpoint_id="migration_2024")

    print(f"Migrated {result['records_processed']} records")
    print(f"Errors: {result['errors']}")

    return result

# Execute migration
if __name__ == "__main__":
    asyncio.run(run_migration())
```

## Import Reference

```python
# ETL Pattern
from dataknobs_fsm.patterns.etl import (
    DatabaseETL, ETLConfig, ETLMode,
    create_etl_pipeline, create_database_sync,
    create_data_migration, create_data_warehouse_load
)

# File Processing Pattern
from dataknobs_fsm.patterns.file_processing import (
    FileProcessor, FileProcessingConfig, ProcessingMode, FileFormat,
    create_csv_processor, create_json_processor,
    create_log_analyzer, create_batch_file_processor
)

# API Orchestration Pattern
from dataknobs_fsm.patterns.api_orchestration import (
    APIOrchestrator, APIOrchestrationConfig, OrchestrationMode,
    APIEndpoint, create_rest_api_orchestrator, create_graphql_orchestrator
)

# LLM Workflow Pattern
from dataknobs_fsm.patterns.llm_workflow import (
    LLMWorkflow, LLMWorkflowConfig, WorkflowType,
    create_simple_llm_workflow, create_rag_workflow, create_chain_workflow
)

# Error Recovery Pattern
from dataknobs_fsm.patterns.error_recovery import (
    ErrorRecoveryWorkflow, ErrorRecoveryConfig, RecoveryStrategy,
    BackoffStrategy, create_retry_workflow, create_circuit_breaker_workflow,
    create_resilient_workflow
)
```

## Next Steps

- Explore individual pattern documentation:
  - [Database ETL](etl.md)
  - [File Processing](file-processing.md)
  - [API Orchestration](api-orchestration.md)
  - [LLM Workflows](llm-workflow.md)
  - [Error Recovery](error-recovery.md)
- Check out [examples](../examples/index.md) using patterns
- Review [guides](../guides/index.md) for detailed usage


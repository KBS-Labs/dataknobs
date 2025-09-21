# Database ETL Pipeline Example

This example demonstrates how to build a production-ready ETL (Extract, Transform, Load) pipeline using the DataKnobs FSM framework with SimpleFSM and COPY mode for transaction safety. The pipeline extracts data from a source database, applies multiple transformation stages, and loads the processed data into a target database.

## Overview

The example showcases:

- **COPY mode** for transaction safety and rollback capability
- **Multi-stage data extraction, transformation, and loading**
- **Custom function registration** for ETL operations
- **Error handling** with proper routing to rollback states
- **Batch processing** with configurable size
- **Data validation and quality checks**

## Source Code

The complete example is available at: [`packages/fsm/examples/database_etl.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/examples/database_etl.py)

## Implementation Details

### FSM Configuration

The example uses SimpleFSM with custom functions registered for each ETL stage:

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

def create_etl_fsm() -> SimpleFSM:
    """Create and configure the ETL FSM."""
    config = {
        "name": "database_etl_pipeline",
        "states": [
            {"name": "start", "initial": True},
            {"name": "initialize"},
            {"name": "extract"},
            {"name": "validate"},
            {"name": "transform"},
            {"name": "enrich"},
            {"name": "load"},
            {"name": "complete", "terminal": True},
            {"name": "rollback"},
            {"name": "error", "terminal": True}
        ],
        "arcs": [...]
    }

    # Create FSM with COPY mode for transaction safety
    fsm = SimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={
            "initialize_etl": initialize_etl,
            "extract_data": extract_data,
            "validate_data": validate_data,
            "transform_data": transform_data,
            "enrich_data": enrich_data,
            "load_data": load_data,
            "finalize_etl": finalize_etl,
            "rollback_transaction": rollback_transaction
        }
    )

    return fsm
```

### ETL Functions

The pipeline implements several key functions:

#### 1. Initialize ETL (`initialize_etl`)
Sets up ETL metadata and statistics tracking:

```python
def initialize_etl(state) -> Dict[str, Any]:
    """Initialize ETL pipeline with configuration and statistics."""
    data = state.data.copy()

    # Initialize ETL metadata
    data['etl_metadata'] = {
        'start_time': datetime.now().isoformat(),
        'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'source_table': data.get('source_table', 'sales_raw'),
        'target_table': data.get('target_table', 'sales_fact'),
        'batch_size': data.get('batch_size', 100),
        'mode': data.get('mode', 'incremental')
    }

    # Initialize statistics
    data['statistics'] = {
        'records_extracted': 0,
        'records_transformed': 0,
        'records_validated': 0,
        'records_loaded': 0,
        'records_failed': 0
    }

    return data
```

#### 2. Extract Data (`extract_data`)
Extracts data from the source database:

```python
def extract_data(state) -> Dict[str, Any]:
    """Extract data from source database."""
    data = state.data.copy()
    source_db = data['source_db']

    conn = sqlite3.connect(source_db)
    cursor = conn.cursor()

    # Build and execute query
    query = f"SELECT * FROM {data['etl_metadata']['source_table']}"
    if data['etl_metadata']['mode'] == 'incremental':
        query += f" WHERE updated_at > '{data['last_sync_time']}'"

    cursor.execute(query)
    records = cursor.fetchall()

    # Convert to list of dictionaries
    columns = [desc[0] for desc in cursor.description]
    data['extracted_records'] = [
        dict(zip(columns, row)) for row in records
    ]

    data['statistics']['records_extracted'] = len(data['extracted_records'])
    conn.close()

    return data
```

#### 3. Transform Data (`transform_data`)
Applies business transformations to the data:

```python
def transform_data(state) -> Dict[str, Any]:
    """Apply business transformations to the data."""
    data = state.data.copy()

    transformed_records = []
    for record in data['validated_records']:
        # Clean string fields
        for key, value in record.items():
            if isinstance(value, str):
                record[key] = value.strip().upper()

        # Calculate derived fields
        record['total_amount'] = record['amount'] * (1 + record.get('tax_rate', 0.1))
        record['profit'] = record['total_amount'] * 0.2  # 20% margin

        # Categorize order size
        if record['amount'] > 1000:
            record['order_category'] = 'LARGE'
        elif record['amount'] > 100:
            record['order_category'] = 'MEDIUM'
        else:
            record['order_category'] = 'SMALL'

        transformed_records.append(record)

    data['transformed_records'] = transformed_records
    data['statistics']['records_transformed'] = len(transformed_records)

    return data
```

#### 4. Load Data (`load_data`)
Loads transformed data into the target database:

```python
def load_data(state) -> Dict[str, Any]:
    """Load transformed data into target database."""
    data = state.data.copy()
    target_db = data['target_db']

    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()

    # Prepare for bulk insert
    records = data['enriched_records']
    if records:
        # Get columns from first record
        columns = list(records[0].keys())
        placeholders = ','.join(['?' for _ in columns])

        # Build insert query
        insert_query = f"""
            INSERT INTO {data['etl_metadata']['target_table']}
            ({','.join(columns)})
            VALUES ({placeholders})
        """

        # Execute bulk insert
        for record in records:
            values = [record.get(col) for col in columns]
            cursor.execute(insert_query, values)

        conn.commit()
        data['statistics']['records_loaded'] = len(records)

    conn.close()
    return data
```
        row['season'] = 'Summer'
    elif month in [3, 4, 5]:
        row['season'] = 'Spring'
    elif month in [9, 10]:
        row['season'] = 'Fall'
    else:
        row['season'] = 'Winter'
    
    return row
```

### Pipeline Execution

The pipeline supports asynchronous execution with progress monitoring:

```python
async def run_async(self, start_date: datetime = None) -> Dict[str, Any]:
    """Run the ETL pipeline asynchronously."""
    if not self.etl:
        self.build_pipeline()
    
    self.stats['start_time'] = datetime.now()
    
    try:
        result = await self.etl.run()
        
        self.stats['end_time'] = datetime.now()
        self.stats['records_extracted'] = result.get('records_extracted', 0)
        self.stats['records_transformed'] = result.get('records_transformed', 0)
        self.stats['records_loaded'] = result.get('records_loaded', 0)
        
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        logger.info(f"ETL completed successfully in {duration:.2f} seconds")
        
        return self.stats
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        self.stats['errors'].append(str(e))
        raise
```

## Configuration Options

The pipeline supports extensive configuration:

### Database Configuration

```python
# Source database
source_config = {
    "provider": "postgresql",
    "config": {
        "host": "localhost",
        "port": 5432,
        "database": "sales_db",
        "user": "etl_user",
        "password": "secure_password"
    }
}

# Target database  
target_config = {
    "provider": "postgresql",
    "config": {
        "host": "warehouse.company.com",
        "port": 5432,
        "database": "warehouse",
        "user": "warehouse_user", 
        "password": "warehouse_password"
    }
}
```

### Pipeline Configuration

```python
pipeline = SalesETLPipeline(source_config, target_config)
etl = pipeline.build_pipeline()

# Configuration is handled via ETLConfig:
# - mode: ETLMode.INCREMENTAL
# - batch_size: 5000
# - parallel_workers: 4
# - checkpoint_interval: 10000
# - error_threshold: 0.01 (1% errors allowed)
```

## Simple FSM Alternative

The example also includes a simpler FSM-based approach for basic ETL workflows:

```python
def create_simple_etl_fsm() -> SimpleFSM:
    """Create a simple ETL FSM for demonstration."""
    
    config = {
        'name': 'simple_etl_workflow',
        'main_network': 'main',
        'networks': [{
            'name': 'main',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'extract'},
                {'name': 'transform'},
                {'name': 'load'},
                {'name': 'complete', 'is_end': True},
                {'name': 'error', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'extract'},
                {'from': 'extract', 'to': 'transform'},
                {'from': 'transform', 'to': 'load'},
                {'from': 'load', 'to': 'complete'}
            ]
        }]
    }
    
    fsm = SimpleFSM(config)
    
    # Add database resources
    fsm.add_resource("source_db", {
        "type": "database",
        "provider": "sqlite",
        "config": {"database": "source.db"}
    })
    
    fsm.add_resource("target_db", {
        "type": "database", 
        "provider": "sqlite",
        "config": {"database": "warehouse.db"}
    })
    
    return fsm
```

## Running the Example

### Prerequisites

1. Install required dependencies:
   ```bash
   pip install dataknobs-fsm
   ```

2. Set up test databases (or use SQLite for testing)

### Basic Usage

```python
import asyncio
from database_etl import SalesETLPipeline

async def main():
    # Configure pipeline
    pipeline = SalesETLPipeline(
        source_config={
            "provider": "sqlite",
            "config": {"database": "sales.db"}
        },
        target_config={
            "provider": "sqlite", 
            "config": {"database": "warehouse.db"}
        }
    )
    
    # Run ETL
    results = await pipeline.run_async()
    print(f"Processed {results['records_loaded']} records")

# Run the pipeline
asyncio.run(main())
```

### Command Line Usage

```bash
# Run with default configuration
python packages/fsm/examples/database_etl.py

# The example includes both ETL Pattern and SimpleFSM demonstrations
```

## Performance Considerations

### Batch Processing
- Default batch size: 5000 records
- Configurable via `batch_size` parameter
- Larger batches reduce overhead but increase memory usage

### Parallel Processing
- Default: 4 parallel workers
- Configurable via `parallel_workers` parameter
- Optimal count depends on database connection limits

### Error Handling
- Default error threshold: 1% (0.01)
- Pipeline continues if error rate is below threshold
- Failed records are logged for manual review

### Checkpointing
- Automatic checkpoints every 10,000 records
- Enables resume from failure point
- Checkpoint interval configurable via `checkpoint_interval`

## Testing

The example includes comprehensive unit tests:

```bash
# Run all tests
uv run pytest packages/fsm/tests/test_database_etl_example.py -v

# Run specific test categories
uv run pytest packages/fsm/tests/test_database_etl_example.py::TestSalesETLPipeline -v
```

Test coverage includes:
- ✅ Transformation function testing
- ✅ Data validation edge cases
- ✅ Error handling scenarios
- ✅ Configuration validation
- ✅ SimpleFSM workflow testing

## Production Deployment

### Database Connections
- Use connection pooling for high throughput
- Configure appropriate timeout values
- Monitor connection usage and limits

### Monitoring
- Implement progress callbacks for real-time monitoring
- Log key metrics (throughput, error rates, processing time)
- Set up alerts for error thresholds

### Security
- Use environment variables for database credentials
- Implement proper access controls
- Audit data access and modifications

### Scaling
- Consider partitioning large datasets by date or key
- Use multiple pipeline instances for different data segments
- Implement queue-based processing for high volumes

## Troubleshooting

### Common Issues

**Connection Timeouts**
```python
# Increase timeout in database config
config = {
    "provider": "postgresql",
    "config": {
        # ... other config ...
        "connect_timeout": 30,
        "command_timeout": 300
    }
}
```

**Memory Issues with Large Datasets**
```python
# Reduce batch size
etl_config = ETLConfig(
    # ... other config ...
    batch_size=1000,  # Reduced from default 5000
    checkpoint_interval=5000  # More frequent checkpoints
)
```

**High Error Rates**
```python
# Check data quality in source
# Adjust error threshold if needed
etl_config = ETLConfig(
    # ... other config ...
    error_threshold=0.05  # Allow 5% errors instead of 1%
)
```

## Next Steps

- Explore the [ETL Pattern Guide](../patterns/etl.md) for more advanced features
- Learn about [Resource Management](../guides/resources.md) for database connections
- Check out [Error Recovery Patterns](../patterns/error-recovery.md) for robust error handling
- See [API Documentation](../api/simple.md) for SimpleFSM usage details
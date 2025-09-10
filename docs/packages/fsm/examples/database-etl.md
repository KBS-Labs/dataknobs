# Database ETL Pipeline Example

This example demonstrates how to build a production-ready ETL (Extract, Transform, Load) pipeline using the DataKnobs FSM framework. The pipeline extracts data from a source database, applies multiple transformation stages, and loads the processed data into a target database.

## Overview

The example showcases:

- **Multi-stage data transformations** including cleaning, validation, enrichment, and metrics calculation
- **Error handling and recovery** with configurable error thresholds
- **Progress monitoring** with real-time feedback
- **Flexible configuration** supporting different database types
- **Production-ready features** like batch processing and checkpointing

## Source Code

The complete example is available at: [`packages/fsm/examples/database_etl.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/examples/database_etl.py)

## Implementation Details

### SalesETLPipeline Class

The main pipeline class encapsulates all ETL functionality:

```python
class SalesETLPipeline:
    """Production-ready ETL pipeline for sales data."""
    
    def __init__(self, source_config: Dict[str, Any], target_config: Dict[str, Any]):
        self.source_config = source_config
        self.target_config = target_config
        self.etl = None
        self.stats = {
            "start_time": None,
            "end_time": None, 
            "records_extracted": 0,
            "records_transformed": 0,
            "records_loaded": 0,
            "errors": []
        }
```

### Transformation Stages

The pipeline implements four transformation stages:

#### 1. Data Cleaning (`clean_data`)
- Removes whitespace from string fields
- Standardizes date formats
- Handles missing values with defaults

```python
def clean_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
    # Remove whitespace
    for key, value in row.items():
        if isinstance(value, str):
            row[key] = value.strip()
    
    # Standardize date formats
    if 'order_date' in row and row['order_date']:
        if isinstance(row['order_date'], str):
            row['order_date'] = datetime.fromisoformat(row['order_date'])
    
    # Handle missing values
    row['customer_segment'] = row.get('customer_segment', 'Unknown')
    row['region'] = row.get('region', 'Unknown')
    
    return row
```

#### 2. Data Validation (`validate_data`)
- Checks for required fields
- Validates data types and ranges
- Enforces business rules

```python
def validate_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
    # Check required fields
    required_fields = ['order_id', 'customer_id', 'order_date', 'amount']
    for field in required_fields:
        if field not in row or row[field] is None:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate ranges
    if row['amount'] < 0:
        raise ValueError(f"Invalid amount: {row['amount']}")
    
    # Business rules
    if row['order_date'] > datetime.now():
        raise ValueError(f"Future order date: {row['order_date']}")
    
    row['_validated'] = True
    return row
```

#### 3. Data Enrichment (`enrich_data`)
- Adds time-based dimensions (year, quarter, month, week)
- Calculates derived financial metrics
- Adds processing metadata

```python
def enrich_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
    # Add time dimensions
    order_date = row['order_date']
    row['year'] = order_date.year
    row['quarter'] = f"Q{(order_date.month - 1) // 3 + 1}"
    row['month'] = order_date.month
    row['week'] = order_date.isocalendar()[1]
    row['day_of_week'] = order_date.strftime('%A')
    
    # Calculate financial metrics
    row['revenue'] = row['amount'] * 1.1  # Add tax
    row['discount_amount'] = row.get('discount_pct', 0) * row['amount'] / 100
    row['net_amount'] = row['amount'] - row['discount_amount']
    
    # Add metadata
    row['etl_timestamp'] = datetime.now()
    row['etl_batch_id'] = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return row
```

#### 4. Metrics Calculation (`calculate_metrics`)
- Classifies customer value based on order history
- Categorizes order sizes
- Adds seasonality indicators

```python
def calculate_metrics(self, row: Dict[str, Any]) -> Dict[str, Any]:
    # Customer value classification
    if row.get('total_orders', 0) > 10:
        row['customer_value'] = 'High'
    elif row.get('total_orders', 0) > 5:
        row['customer_value'] = 'Medium'
    else:
        row['customer_value'] = 'Low'
    
    # Order size classification
    if row['amount'] > 1000:
        row['order_size'] = 'Large'
    elif row['amount'] > 100:
        row['order_size'] = 'Medium'
    else:
        row['order_size'] = 'Small'
    
    # Seasonality
    month = row['month']
    if month in [11, 12]:
        row['season'] = 'Holiday'
    elif month in [6, 7, 8]:
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
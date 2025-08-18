# Sensor Dashboard Example

A practical mini-project demonstrating the DataKnobs data package capabilities through a sensor monitoring system.

## What This Example Demonstrates

### ✅ Successfully Implemented
- **CRUD Operations**: Register sensors, ingest readings
- **Batch Processing**: Bulk import of historical data with error handling
- **Streaming**: Configurable batch sizes for data ingestion
- **Database Abstraction**: Works with any database backend (memory, file, etc.)
- **Pandas Integration**: Convert to DataFrames, calculate hourly averages
- **Async Support**: Full async implementation for concurrent operations
- **Error Handling**: Graceful degradation with partial batch failures
- **Data Processing**: Anomaly detection, duplicate removal, time-range queries

### 📊 Test Coverage Results
- **14 tests passing** covering core functionality
- Successfully exercises database operations, streaming, and pandas integration
- Validates error recovery and batch processing

## Key Learnings & API Discoveries

### 🐛 Critical Issues Found

1. **Nested Field Queries Don't Work**
   - `Filter("metadata.type", ...)` doesn't actually query metadata
   - Workaround: Store queryable data as top-level fields (see `_type` field)

2. **Missing Query Capabilities**
   - No OR operator for filters
   - No date/time range operators
   - Sorting doesn't work reliably across backends

3. **API Ergonomics**
   - Field access is awkward (`record.fields[name].value`)
   - No simple `database.list()` method
   - BatchConfig parameters are confusing

### 💡 Positive Findings

- Database abstraction allows seamless backend switching
- Streaming API with batch configuration works well
- Error handling with skip/continue is effective
- Connection pooling integration is clean

## Usage Example

```python
from dataknobs_data.backends.memory import SyncMemoryDatabase
from examples.sensor_dashboard import SensorDashboard, SensorDataGenerator

# Initialize dashboard with any database backend
db = SyncMemoryDatabase()
dashboard = SensorDashboard(db)

# Generate test data
generator = SensorDataGenerator(seed=42)
sensors = generator.create_sensors(3)

# Register sensors
for sensor in sensors:
    dashboard.register_sensor(sensor)

# Generate and ingest readings
readings = generator.generate_normal_readings(
    sensor_id=sensors[0].sensor_id,
    location=sensors[0].location,
    start_time=datetime.now(),
    count=100
)

# Stream readings with batching
result = dashboard.stream_readings(readings, batch_size=10)
print(f"Ingested {result.successful} readings")

# Query recent data
recent = dashboard.get_recent_readings(sensors[0].sensor_id, limit=10)

# Analyze with pandas
df = dashboard.to_dataframe(recent)
hourly_avg = dashboard.calculate_hourly_averages(readings)
```

## Files in This Example

- `models.py` - Data models for sensors and readings
- `sensor_dashboard.py` - Main dashboard implementation (sync and async)
- `data_generator.py` - Test data generator with various scenarios
- `test_sensor_dashboard_example.py` - Comprehensive test suite

## Recommendations for Data Package

Based on this real-world example, we recommend:

1. **Fix nested field queries** (Critical)
2. **Add time-range query operators**
3. **Improve field value access API**
4. **Add OR operator for filters**
5. **Provide database.list() method**
6. **Document BatchConfig parameters clearly**

See `docs/API_IMPROVEMENTS.md` for detailed findings.

## Running the Tests

```bash
cd packages/data
uv run pytest tests/test_sensor_dashboard_example.py -v
```

## Next Steps

This example provides a foundation for:
- Building tutorials showing real-world usage
- Creating integration tests for the data package
- Identifying and fixing API limitations
- Demonstrating best practices for database abstraction
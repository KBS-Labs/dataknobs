# Sensor Dashboard Example

A comprehensive example demonstrating real-world usage of the DataKnobs data package through a sensor monitoring system.

## Overview

The Sensor Dashboard example showcases:

- **CRUD operations** for sensor management
- **Batch processing** of historical data
- **Streaming ingestion** of real-time readings
- **Advanced querying** with boolean logic and range operators
- **Error handling** and recovery strategies
- **Pandas integration** for data analysis
- **Ergonomic field access** using new Priority 5 features

## Quick Start

```python
from dataknobs_data import SyncMemoryDatabase
from examples.sensor_dashboard import (
    SensorDashboard, 
    SensorDataGenerator,
    SensorReading,
    SensorInfo
)
from datetime import datetime, timedelta

# Initialize dashboard with in-memory database
db = SyncMemoryDatabase()
dashboard = SensorDashboard(db)

# Create test data generator
generator = SensorDataGenerator(seed=42)

# Register sensors
sensors = generator.create_sensors(3)
for sensor in sensors:
    dashboard.register_sensor(sensor)

# Generate and ingest readings
start_time = datetime.now()
for sensor in sensors:
    readings = generator.generate_normal_readings(
        sensor.sensor_id,
        sensor.location,
        start_time,
        count=100
    )
    dashboard.ingest_readings_batch(readings)

# Query recent readings
recent = dashboard.get_recent_readings(sensors[0].sensor_id, limit=10)
for reading in recent:
    print(f"{reading.timestamp}: {reading.temperature}°C, {reading.humidity}%")
```

## Core Components

### Data Models

The example uses two main data models:

```python
@dataclass
class SensorReading:
    """A single sensor reading with environmental data."""
    sensor_id: str
    timestamp: datetime
    temperature: float  # Celsius
    humidity: float    # Percentage 0-100
    battery: Optional[int] = None  # Battery percentage
    location: Optional[str] = None
    
    def to_record(self) -> Record:
        """Convert to DataKnobs Record using new ergonomic features."""
        return Record(
            id=f"{self.sensor_id}_{self.timestamp.isoformat()}",
            data={
                "sensor_id": self.sensor_id,
                "timestamp": self.timestamp.isoformat(),
                "temperature": self.temperature,
                "humidity": self.humidity,
                "battery": self.battery,
                "location": self.location
            },
            metadata={
                "type": "sensor_reading",
                "sensor_id": self.sensor_id
            }
        )
    
    @classmethod
    def from_record(cls, record: Record) -> "SensorReading":
        """Create from Record using new dict-like access."""
        # Using new to_dict() for simple value extraction
        data = record.to_dict()
        return cls(
            sensor_id=data["sensor_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            temperature=data["temperature"],
            humidity=data["humidity"],
            battery=data.get("battery"),
            location=data.get("location")
        )

@dataclass
class SensorInfo:
    """Sensor device information."""
    sensor_id: str
    sensor_type: str
    location: str
    installed: datetime
    status: str = "active"  # active, inactive, maintenance
```

### Dashboard Operations

The `SensorDashboard` class provides high-level operations:

#### Basic CRUD Operations

```python
# Register a sensor
sensor = SensorInfo(
    sensor_id="sensor_001",
    sensor_type="TH100",
    location="warehouse",
    installed=datetime.now(),
    status="active"
)
sensor_id = dashboard.register_sensor(sensor)

# Ingest a reading
reading = SensorReading(
    sensor_id="sensor_001",
    timestamp=datetime.now(),
    temperature=22.5,
    humidity=45.0,
    battery=87,
    location="warehouse"
)
reading_id = dashboard.ingest_reading(reading)

# Query readings
recent = dashboard.get_recent_readings("sensor_001", limit=10)
```

#### Batch Processing

```python
# Generate batch of readings
readings = generator.generate_normal_readings(
    sensor_id="sensor_001",
    location="warehouse",
    start_time=datetime.now(),
    count=1000,
    interval_minutes=5
)

# Batch import
ids = dashboard.ingest_readings_batch(readings)
print(f"Imported {len(ids)} readings")
```

#### Streaming with Error Handling

```python
# Stream readings with batch processing and error recovery
result = dashboard.stream_readings(readings, batch_size=100)

print(f"Processed: {result.total_processed}")
print(f"Successful: {result.successful}")
print(f"Failed: {result.failed}")
print(f"Success rate: {result.success_rate:.1%}")

# Enhanced streaming with failure tracking
result = dashboard.stream_readings_with_tracking(
    readings,
    batch_size=100,
    track_failures=True
)

if result.failed_indices:
    print(f"Failed at indices: {result.failed_indices}")
    # Handle failed records...
```

## Advanced Querying

### Boolean Logic Queries

Using the new boolean operators (AND, OR, NOT):

```python
# Find readings from multiple sensors
query = Query().or_(
    Filter("sensor_id", Operator.EQ, "sensor_001"),
    Filter("sensor_id", Operator.EQ, "sensor_002")
).filter("metadata.type", Operator.EQ, "sensor_reading")

records = db.search(query)
```

### Range Queries

Using the new BETWEEN operator:

```python
# Find readings in a time range
start = datetime(2024, 1, 1, 9, 0)
end = datetime(2024, 1, 1, 17, 0)

readings = dashboard.get_readings_by_timerange(start, end)

# Find optimal conditions
optimal = dashboard.get_optimal_conditions(
    temp_range=(18, 25),      # Comfortable temperature
    humidity_range=(40, 60),   # Comfortable humidity
    min_battery=50.0          # Good battery level
)
```

### Complex Queries

Combining multiple conditions:

```python
# Find critical sensors
critical = dashboard.get_critical_sensors(
    battery_threshold=20.0,
    locations=["server_room", "warehouse"]
)

# Find anomalous readings
anomalies = dashboard.get_anomalous_readings(
    temp_range=(10, 35),     # Normal range
    humidity_range=(30, 70)   # Normal range
)
```

## Ergonomic Field Access Demonstration

The example showcases the new Priority 5 field access improvements:

```python
def demonstrate_ergonomic_field_access():
    """Show new convenient field access methods."""
    
    # Get a sensor reading record
    record = readings[0].to_record()
    
    # 1. Dict-like access (NEW)
    print(f"Temperature: {record['temperature']}°C")
    print(f"Humidity: {record['humidity']}%")
    
    # 2. Attribute access (NEW)
    print(f"Sensor: {record.sensor_id}")
    print(f"Location: {record.location}")
    
    # 3. Modify values easily
    record["temperature"] = 25.5  # Dict-like
    record.humidity = 55.0        # Attribute
    
    # 4. Simple to_dict() conversion
    data = record.to_dict()  # Flat dict by default
    df = pd.DataFrame([data])  # Direct pandas conversion
    
    # 5. Access Field objects when needed
    if "temperature" in record:
        temp_field = record.get_field_object("temperature")
        print(f"Field type: {temp_field.type}")
```

## Data Analysis with Pandas

```python
# Convert readings to DataFrame
readings = dashboard.get_recent_readings("sensor_001", limit=1000)
df = dashboard.to_dataframe(readings)

# Analyze data
print(df.describe())
print(f"Average temperature: {df['temperature'].mean():.1f}°C")
print(f"Max humidity: {df['humidity'].max():.1f}%")

# Find anomalies
anomalies = df[
    (df['temperature'] > 30) | 
    (df['humidity'] > 80) |
    (df['battery'] < 20)
]

# Time-based aggregation
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
hourly = df.resample('1H').mean()
```

## Test Data Generation

The example includes a comprehensive data generator:

```python
generator = SensorDataGenerator(seed=42)

# Normal readings with realistic variations
normal = generator.generate_normal_readings(
    sensor_id="sensor_001",
    location="warehouse",
    start_time=datetime.now(),
    count=100
)

# Anomalous readings for testing
anomalies = generator.generate_anomalous_readings(
    sensor_id="sensor_001",
    location="warehouse",
    start_time=datetime.now(),
    count=20
)

# Simulate missing data
with_gaps = generator.generate_missing_data_readings(
    sensor_id="sensor_001",
    location="warehouse",
    start_time=datetime.now(),
    count=100,
    missing_rate=0.1  # 10% missing
)

# Test batch failure scenarios
readings, failed_indices = generator.generate_batch_failure_scenario(
    sensor_id="sensor_001",
    location="warehouse",
    start_time=datetime.now(),
    count=50
)
```

## Running the Example

### Complete Example

```bash
# Run the main sensor dashboard demo
python examples/sensor_dashboard/sensor_dashboard.py
```

### Advanced Query Demo

```bash
# Demonstrate boolean logic and range operators
python examples/sensor_dashboard/demo_advanced_queries.py
```

### Streaming Improvements Demo

```bash
# Show batch processing and error handling
python examples/sensor_dashboard/demo_streaming_improvements.py
```

## Testing

The example includes comprehensive test coverage:

```python
# Run all sensor dashboard tests
pytest tests/test_sensor_dashboard_example.py -v

# Run advanced query tests
pytest tests/test_sensor_dashboard_advanced.py -v

# Run streaming tests
pytest tests/test_sensor_dashboard_streaming.py -v
```

## Key Learnings

This example demonstrates:

1. **Real-world patterns** for time-series data management
2. **Error handling strategies** for production systems
3. **Performance optimization** through batch processing
4. **Query flexibility** with boolean logic and ranges
5. **Ergonomic API design** for developer productivity
6. **Testing approaches** for data-intensive applications

## Next Steps

- Extend the example with real sensor hardware integration
- Add visualization dashboards using Plotly or Streamlit
- Implement alerting based on threshold violations
- Add machine learning for anomaly detection
- Scale to multiple databases (PostgreSQL, Elasticsearch)
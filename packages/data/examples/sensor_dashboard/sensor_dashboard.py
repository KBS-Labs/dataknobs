"""
Mini Sensor Dashboard Example

Demonstrates key features of the dataknobs data package through
a simple sensor monitoring system.
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path

from dataknobs_data import (
    Record, Query, Filter, Operator, SortSpec, SortOrder,
    ComplexQuery, QueryBuilder, StreamProcessor
)
from dataknobs_data.database import SyncDatabase, AsyncDatabase
from dataknobs_data.streaming import StreamConfig, StreamResult

try:
    from .models import SensorReading, SensorInfo
except ImportError:
    # When run as script, use absolute import
    from models import SensorReading, SensorInfo


class SensorDashboard:
    """
    A simple sensor monitoring dashboard using dataknobs data package.
    
    This example demonstrates:
    - CRUD operations for sensor management
    - Batch import of historical data
    - Streaming ingestion of real-time readings
    - Pandas integration for analysis
    - Error handling and recovery
    """
    
    def __init__(self, database: SyncDatabase):
        """Initialize with a database backend."""
        self.db = database
        self.sensors: Dict[str, SensorInfo] = {}
        
    def register_sensor(self, sensor: SensorInfo) -> str:
        """Register a new sensor in the system."""
        record = sensor.to_record()
        sensor_id = self.db.create(record)
        self.sensors[sensor.sensor_id] = sensor
        return sensor_id
    
    def register_sensors_batch(self, sensors: List[SensorInfo]) -> List[str]:
        """Register multiple sensors at once."""
        records = [sensor.to_record() for sensor in sensors]
        ids = self.db.create_batch(records)
        for sensor in sensors:
            self.sensors[sensor.sensor_id] = sensor
        return ids
    
    def ingest_reading(self, reading: SensorReading) -> str:
        """Store a single sensor reading."""
        record = reading.to_record()
        return self.db.create(record)
    
    def ingest_readings_batch(self, readings: List[SensorReading]) -> List[str]:
        """Batch import historical sensor readings."""
        records = [reading.to_record() for reading in readings]
        return self.db.create_batch(records)
    
    def stream_readings(self, readings: List[SensorReading], 
                       batch_size: int = 10) -> StreamResult:
        """Stream readings with configurable batch size.
        
        Demonstrates basic streaming with error handling and validation.
        """
        import math
        
        # Convert all readings to records - validate BEFORE creating
        records = []
        for i, reading in enumerate(readings):
            try:
                # Check for invalid data BEFORE creating the record
                if not reading.sensor_id:
                    # Empty sensor_id - create a dummy record that will fail
                    record = Record(data={"error": "Empty sensor_id"})
                    record._invalid = True
                    record._invalid_reason = "Empty sensor_id"
                elif math.isnan(reading.temperature) or math.isnan(reading.humidity):
                    # NaN values - create a dummy record that will fail
                    record = Record(data={"error": "NaN values"})
                    record._invalid = True
                    record._invalid_reason = "NaN values detected"
                else:
                    # Valid reading - create the actual record
                    record = reading.to_record()
                    
                records.append(record)
            except Exception as e:
                # If anything fails, create a dummy record that will be rejected
                record = Record(data={"error": str(e)})
                record._invalid = True
                record._invalid_reason = str(e)
                records.append(record)
        
        # Define error handler that continues on error
        def continue_on_error(exc: Exception, record: Record) -> bool:
            # Log the error and continue processing
            return True  # Continue processing
        
        # Monkey-patch create methods to reject invalid records
        original_create = self.db.create
        original_create_batch = self.db.create_batch
        
        def validating_create(record):
            if hasattr(record, '_invalid') and record._invalid:
                raise ValueError("Invalid record")
            return original_create(record)
        
        def validating_create_batch(batch_records):
            # Check if any record is invalid
            for r in batch_records:
                if hasattr(r, '_invalid') and r._invalid:
                    # Batch fails, will fall back to individual processing
                    raise ValueError("Batch contains invalid records")
            return original_create_batch(batch_records)
        
        self.db.create = validating_create
        self.db.create_batch = validating_create_batch
        
        try:
            config = StreamConfig(
                batch_size=batch_size,
                on_error=continue_on_error  # Continue on error
            )
            
            result = self.db.stream_write(records, config)
        finally:
            # Restore original methods
            self.db.create = original_create
            self.db.create_batch = original_create_batch
        
        return result
    
    def stream_readings_with_tracking(self, readings: List[SensorReading],
                                     batch_size: int = 10,
                                     track_failures: bool = True) -> StreamResult:
        """Stream readings with enhanced tracking of batches and failures.
        
        Demonstrates new StreamResult features:
        - total_batches: Number of batches processed
        - failed_indices: Indices of failed records
        """
        records = [reading.to_record() for reading in readings]
        
        # Track errors for analysis
        error_details = []
        
        def error_handler(exc: Exception, record: Record) -> bool:
            if track_failures:
                error_details.append({
                    'error': str(exc),
                    'record_id': record.id if record and hasattr(record, 'id') else None
                })
            return True  # Continue processing
        
        config = StreamConfig(
            batch_size=batch_size,
            on_error=error_handler if track_failures else None
        )
        
        # Convert list to iterator using StreamProcessor
        record_iter = StreamProcessor.list_to_iterator(records)
        result = self.db.stream_write(record_iter, config)
        
        # Enhanced result information
        if result.total_batches > 0:
            print(f"Processed {result.total_batches} batches")
        if result.failed_indices:
            print(f"Failed at indices: {result.failed_indices}")
        
        return result
    
    def stream_with_validation(self, readings: List[SensorReading],
                              temp_range: tuple = (-50, 100),
                              humidity_range: tuple = (0, 100)) -> StreamResult:
        """Stream readings with validation, demonstrating error handling.
        
        Readings outside valid ranges will fail, demonstrating:
        - Graceful batch fallback
        - Individual error tracking
        - Failed indices collection
        """
        validated_records = []
        
        for i, reading in enumerate(readings):
            record = reading.to_record()
            
            # Add validation that might fail
            if not (temp_range[0] <= reading.temperature <= temp_range[1]):
                # This will cause an error during processing
                record.set_field("_invalid", f"Temperature {reading.temperature} out of range")
            if not (humidity_range[0] <= reading.humidity <= humidity_range[1]):
                record.set_field("_invalid", f"Humidity {reading.humidity} out of range")
            
            validated_records.append(record)
        
        # Monkey-patch the database methods to validate
        original_create = self.db.create
        original_create_batch = self.db.create_batch
        
        def validating_create(record):
            if record.get_value("_invalid"):
                raise ValueError(record.get_value("_invalid"))
            return original_create(record)
        
        def validating_create_batch(records):
            # Check if any record is invalid - this forces fallback
            for r in records:
                # Using new dict-like access
                if "_invalid" in r and r["_invalid"]:
                    raise ValueError("Batch contains invalid records")
            return original_create_batch(records)
        
        # Temporarily replace methods
        self.db.create = validating_create
        self.db.create_batch = validating_create_batch
        
        config = StreamConfig(
            batch_size=10,
            on_error=lambda exc, rec: True  # Continue on error
        )
        
        # This will demonstrate batch fallback behavior
        record_iter = StreamProcessor.list_to_iterator(validated_records)
        
        try:
            result = self.db.stream_write(record_iter, config)
        finally:
            # Restore original methods
            self.db.create = original_create
            self.db.create_batch = original_create_batch
        
        return result
    
    def get_recent_readings(self, sensor_id: str, limit: int = 10) -> List[SensorReading]:
        """Get the most recent readings for a sensor."""
        # Query for readings of this specific sensor using nested field queries
        query = Query(
            filters=[
                Filter("metadata.type", Operator.EQ, "sensor_reading"),
                Filter("metadata.sensor_id", Operator.EQ, sensor_id)
            ],
            limit_value=limit * 2  # Get more to account for sorting
        )
        
        records = self.db.search(query)
        readings = [SensorReading.from_record(r) for r in records]
        
        # Sort by timestamp descending and limit
        readings.sort(key=lambda x: x.timestamp, reverse=True)
        return readings[:limit]
    
    def get_readings_by_timerange(self, start: datetime, end: datetime) -> List[SensorReading]:
        """Get all readings within a time range."""
        # Use the new BETWEEN operator for efficient time range queries
        query = Query(
            filters=[
                Filter("metadata.type", Operator.EQ, "sensor_reading"),
                Filter("timestamp", Operator.BETWEEN, (start.isoformat(), end.isoformat()))
            ]
        )
        
        records = self.db.search(query)
        readings = [SensorReading.from_record(r) for r in records]
        
        # Sort by timestamp
        readings.sort(key=lambda x: x.timestamp)
        return readings
    
    def get_all_sensors(self) -> List[SensorInfo]:
        """Get all registered sensors."""
        query = Query(
            filters=[Filter("metadata.type", Operator.EQ, "sensor_info")]
        )
        
        records = self.db.search(query)
        return [SensorInfo.from_record(r) for r in records]
    
    def to_dataframe(self, readings: List[SensorReading]) -> pd.DataFrame:
        """Convert readings to a pandas DataFrame for analysis."""
        if not readings:
            return pd.DataFrame()
        
        data = []
        for reading in readings:
            data.append({
                "sensor_id": reading.sensor_id,
                "timestamp": reading.timestamp,
                "temperature": reading.temperature,
                "humidity": reading.humidity,
                "battery": reading.battery,
                "location": reading.location
            })
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df
    
    def calculate_hourly_averages(self, readings: List[SensorReading]) -> pd.DataFrame:
        """Calculate hourly average temperature and humidity."""
        df = self.to_dataframe(readings)
        
        if df.empty:
            return df
        
        # Resample to hourly frequency and calculate mean
        hourly = df.groupby("sensor_id").resample("1h")[["temperature", "humidity"]].mean()
        hourly = hourly.round(2)
        
        # Add count of readings per hour
        counts = df.groupby("sensor_id").resample("1h", include_groups=False).size()
        hourly["reading_count"] = counts
        
        return hourly
    
    def detect_anomalies(self, readings: List[SensorReading], 
                         temp_threshold: tuple = (10, 35),
                         humidity_threshold: tuple = (20, 80)) -> List[SensorReading]:
        """Detect readings outside normal ranges."""
        anomalies = []
        
        for reading in readings:
            if not (temp_threshold[0] <= reading.temperature <= temp_threshold[1]):
                anomalies.append(reading)
            elif not (humidity_threshold[0] <= reading.humidity <= humidity_threshold[1]):
                anomalies.append(reading)
        
        return anomalies
    
    def get_anomalous_readings_query(self, 
                                    temp_range: tuple = (10, 35),
                                    humidity_range: tuple = (20, 80)) -> List[SensorReading]:
        """Get anomalous readings using NOT_BETWEEN operator.
        
        Demonstrates the new NOT_BETWEEN operator for finding outliers.
        """
        # Find readings with temperature OR humidity outside normal ranges
        query = Query().filter("metadata.type", Operator.EQ, "sensor_reading").or_(
            Filter("temperature", Operator.NOT_BETWEEN, temp_range),
            Filter("humidity", Operator.NOT_BETWEEN, humidity_range)
        )
        
        records = self.db.search(query)
        return [SensorReading.from_record(r) for r in records]
    
    def get_critical_sensors(self, 
                            battery_threshold: float = 20.0,
                            locations: List[str] = None) -> List[SensorReading]:
        """Get readings from critical sensors using boolean logic.
        
        Demonstrates complex boolean queries:
        - Low battery sensors in specific locations
        - OR sensors with very high/low temperatures
        """
        builder = QueryBuilder()
        
        # Base condition: sensor readings only
        builder.where("metadata.type", Operator.EQ, "sensor_reading")
        
        # Critical condition 1: Low battery in important locations
        if locations:
            critical_location = (
                QueryBuilder()
                .where("battery", Operator.LT, battery_threshold)
                .where("location", Operator.IN, locations)
            )
        else:
            critical_location = QueryBuilder().where("battery", Operator.LT, battery_threshold)
        
        # Critical condition 2: Extreme temperatures
        extreme_temp = QueryBuilder().or_(
            Filter("temperature", Operator.LT, 5),  # Too cold
            Filter("temperature", Operator.GT, 40)  # Too hot
        )
        
        # Combine: (low battery AND important location) OR extreme temperatures
        builder.and_(
            QueryBuilder().or_(critical_location, extreme_temp)
        )
        
        query = builder.build()
        records = self.db.search(query)
        return [SensorReading.from_record(r) for r in records]
    
    def get_sensors_by_multiple_locations(self, locations: List[str]) -> List[SensorReading]:
        """Get sensors from multiple locations using OR operator.
        
        Demonstrates simple OR queries for multiple values.
        """
        if not locations:
            return []
        
        # Build OR query for multiple locations
        location_filters = [Filter("location", Operator.EQ, loc) for loc in locations]
        
        query = Query().filter("metadata.type", Operator.EQ, "sensor_reading").or_(
            *location_filters
        )
        
        records = self.db.search(query)
        return [SensorReading.from_record(r) for r in records]
    
    def get_optimal_conditions(self,
                              temp_range: tuple = (18, 25),
                              humidity_range: tuple = (40, 60),
                              min_battery: float = 50.0) -> List[SensorReading]:
        """Get readings with optimal conditions using BETWEEN and AND.
        
        Demonstrates combining BETWEEN operators with AND logic.
        """
        query = Query(
            filters=[
                Filter("metadata.type", Operator.EQ, "sensor_reading"),
                Filter("temperature", Operator.BETWEEN, temp_range),
                Filter("humidity", Operator.BETWEEN, humidity_range),
                Filter("battery", Operator.GTE, min_battery)
            ]
        )
        
        records = self.db.search(query)
        return [SensorReading.from_record(r) for r in records]
    
    def demonstrate_ergonomic_field_access(self) -> None:
        """Demonstrate the new ergonomic field access features.
        
        Shows off:
        - Dict-like access with record["field"]
        - Attribute access with record.field
        - Simplified to_dict() method
        - get_field_object() for accessing Field metadata
        """
        # Get a sample reading
        query = Query(filters=[Filter("metadata.type", Operator.EQ, "sensor_reading")])
        records = self.db.search(query, limit=1)
        
        if not records:
            print("No sensor readings found")
            return
        
        record = records[0]
        
        print("=== Demonstrating Ergonomic Field Access ===\n")
        
        # 1. Dict-like access (NEW)
        print("1. Dict-like access:")
        print(f"   Temperature: {record['temperature']}°C")
        print(f"   Humidity: {record['humidity']}%")
        if "battery" in record:
            print(f"   Battery: {record['battery']}%")
        print()
        
        # 2. Attribute access (NEW)
        print("2. Attribute access:")
        print(f"   Sensor ID: {record.sensor_id}")
        print(f"   Temperature: {record.temperature}°C")
        print(f"   Humidity: {record.humidity}%")
        print()
        
        # 3. Simplified to_dict() (NEW default behavior)
        print("3. Simplified to_dict():")
        simple_dict = record.to_dict()  # Now returns flat dict by default
        print(f"   {simple_dict}")
        print()
        
        # 4. Setting values is also easier
        print("4. Setting field values:")
        # Can set via dict-like access
        record["temperature"] = 25.5
        print(f"   Set via dict: record['temperature'] = 25.5")
        
        # Or via attribute access
        record.humidity = 55.0
        print(f"   Set via attr: record.humidity = 55.0")
        print(f"   New values: temp={record.temperature}, humidity={record.humidity}")
        print()
        
        # 5. Access Field object when needed for metadata
        print("5. Accessing Field metadata when needed:")
        if "temperature" in record.fields:
            temp_field = record.get_field_object("temperature")
            print(f"   Temperature field type: {temp_field.type}")
            print(f"   Temperature field metadata: {temp_field.metadata}")
        
        print("\n=== End of Demonstration ===")
    
    def search_maintenance_needed(self) -> List[SensorReading]:
        """Find sensors needing maintenance using complex logic.
        
        Demonstrates NOT operator and complex conditions:
        - Battery < 30% AND NOT recently checked (>7 days old)
        - OR no readings in last 24 hours
        """
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        day_ago = now - timedelta(days=1)
        
        # Sensors with low battery that haven't been checked recently
        low_battery_old = (
            QueryBuilder()
            .where("metadata.type", Operator.EQ, "sensor_reading")
            .where("battery", Operator.LT, 30)
            .not_(Filter("timestamp", Operator.GT, week_ago.isoformat()))
        )
        
        # Sensors with no recent readings
        no_recent = (
            QueryBuilder()
            .where("metadata.type", Operator.EQ, "sensor_reading")
            .not_(Filter("timestamp", Operator.GT, day_ago.isoformat()))
        )
        
        # Combine conditions
        query = QueryBuilder().or_(low_battery_old, no_recent).build()
        
        records = self.db.search(query)
        return [SensorReading.from_record(r) for r in records]
    
    def handle_duplicate_readings(self, readings: List[SensorReading]) -> List[SensorReading]:
        """Remove duplicate readings (same sensor and timestamp)."""
        seen = set()
        unique = []
        
        for reading in readings:
            key = (reading.sensor_id, reading.timestamp.isoformat())
            if key not in seen:
                seen.add(key)
                unique.append(reading)
        
        return unique
    
    def interpolate_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing sensor data in DataFrame."""
        # Infer object types to proper dtypes before interpolating
        df = df.infer_objects(copy=False)
        
        # Get numeric columns for interpolation
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Check if we have sensor_id column
        if 'sensor_id' in df.columns:
            # Group by sensor and interpolate only numeric columns
            for sensor_id in df['sensor_id'].unique():
                mask = df['sensor_id'] == sensor_id
                # Only interpolate numeric columns to avoid object dtype warning
                if numeric_cols:
                    df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].interpolate(method="time")
        else:
            # Single sensor, interpolate only numeric columns
            if numeric_cols:
                df[numeric_cols] = df[numeric_cols].interpolate(method="time")
        
        return df
    

class AsyncSensorDashboard:
    """Async version of the sensor dashboard."""
    
    def __init__(self, database: AsyncDatabase):
        """Initialize with an async database backend."""
        self.db = database
        self.sensors: Dict[str, SensorInfo] = {}
    
    async def register_sensor(self, sensor: SensorInfo) -> str:
        """Register a new sensor in the system."""
        record = sensor.to_record()
        sensor_id = await self.db.create(record)
        self.sensors[sensor.sensor_id] = sensor
        return sensor_id
    
    async def register_sensors_batch(self, sensors: List[SensorInfo]) -> List[str]:
        """Register multiple sensors at once."""
        records = [sensor.to_record() for sensor in sensors]
        ids = await self.db.create_batch(records)
        for sensor in sensors:
            self.sensors[sensor.sensor_id] = sensor
        return ids
    
    async def ingest_reading(self, reading: SensorReading) -> str:
        """Store a single sensor reading."""
        record = reading.to_record()
        return await self.db.create(record)
    
    async def ingest_readings_batch(self, readings: List[SensorReading]) -> List[str]:
        """Batch import historical sensor readings."""
        records = [reading.to_record() for reading in readings]
        return await self.db.create_batch(records)
    
    async def stream_readings(self, readings: List[SensorReading], 
                            batch_size: int = 10) -> StreamResult:
        """Stream readings with configurable batch size."""
        # Convert readings to records
        records = [reading.to_record() for reading in readings]
        
        # Define error handler that continues on error
        def continue_on_error(exc: Exception, record: Record) -> bool:
            # Log the error and continue processing
            return True  # Continue processing
        
        config = StreamConfig(
            batch_size=batch_size,
            on_error=continue_on_error  # Continue on error
        )
        
        # Use StreamProcessor adapter for cleaner code
        async_iter = StreamProcessor.list_to_async_iterator(records)
        
        result = await self.db.stream_write(async_iter, config)
        return result
    
    async def stream_readings_with_tracking(self, readings: List[SensorReading],
                                           batch_size: int = 10) -> StreamResult:
        """Stream readings with enhanced tracking (async version).
        
        Demonstrates StreamProcessor adapters and enhanced StreamResult.
        """
        records = [reading.to_record() for reading in readings]
        
        config = StreamConfig(
            batch_size=batch_size,
            on_error=lambda exc, rec: True  # Continue on error
        )
        
        # Use StreamProcessor to convert list to async iterator
        async_iter = StreamProcessor.list_to_async_iterator(records)
        result = await self.db.stream_write(async_iter, config)
        
        # Enhanced result provides detailed batch information
        return result
    
    async def get_recent_readings(self, sensor_id: str, limit: int = 10) -> List[SensorReading]:
        """Get the most recent readings for a sensor."""
        # Query for readings of this specific sensor using nested field queries
        query = Query(
            filters=[
                Filter("metadata.type", Operator.EQ, "sensor_reading"),
                Filter("metadata.sensor_id", Operator.EQ, sensor_id)
            ],
            limit_value=limit * 2
        )
        
        records = await self.db.search(query)
        readings = [SensorReading.from_record(r) for r in records]
        
        # Sort by timestamp descending and limit
        readings.sort(key=lambda x: x.timestamp, reverse=True)
        return readings[:limit]
    
    async def process_sensor_stream(self, sensor_readings_generator):
        """Process a stream of sensor readings asynchronously."""
        tasks = []
        async for reading in sensor_readings_generator:
            task = asyncio.create_task(self.ingest_reading(reading))
            tasks.append(task)
            
            # Process in batches to avoid too many concurrent tasks
            if len(tasks) >= 100:
                await asyncio.gather(*tasks)
                tasks = []
        
        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)


# Observations on data package usage:
# 
# 1. ✅ FIXED: Nested field queries now work! Filter("metadata.type", ...) properly
#    queries nested fields after our improvements.
#
# 2. No OR operator for filters - cannot query for "sensor_1 OR sensor_2" efficiently.
#    Must query broader set and filter in memory.
#
# 3. Sorting by timestamp requires post-processing. The Query sort_specs exist but
#    don't reliably work across all field types/backends.
#
# 4. Time-range queries are common but require custom filtering after fetch.
#    Would benefit from BETWEEN, DATE_GT, DATE_LT operators.
#
# 5. No simple database.list() or database.all() method - must use Query with filter.
#
# 6. Field value access could be improved - currently must use record.get_value() or 
#    navigate through record.fields[name].value instead of simple record[name].
#
# The package provides good abstractions for:
# - Database backend switching
# - Batch operations with error handling  
# - Streaming with configurable batch sizes
# - Connection pooling
# - ✅ Nested field queries (after our fix)
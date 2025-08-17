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
    ComplexQuery, QueryBuilder
)
from dataknobs_data.database import SyncDatabase, AsyncDatabase
from dataknobs_data.streaming import StreamConfig, StreamResult

from .models import SensorReading, SensorInfo


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
        """Stream readings with configurable batch size."""
        records = [reading.to_record() for reading in readings]
        
        # Define error handler that continues on error
        def continue_on_error(exc: Exception, record: Record) -> bool:
            # Log the error and continue processing
            return True  # Continue processing
        
        config = StreamConfig(
            batch_size=batch_size,
            on_error=continue_on_error  # Continue on error
        )
        
        result = self.db.stream_write(records, config)
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
        hourly = df.groupby("sensor_id").resample("1H")[["temperature", "humidity"]].mean()
        hourly = hourly.round(2)
        
        # Add count of readings per hour
        counts = df.groupby("sensor_id").resample("1H").size()
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
        # Check if we have sensor_id column
        if 'sensor_id' in df.columns:
            # Group by sensor and interpolate
            for sensor_id in df['sensor_id'].unique():
                mask = df['sensor_id'] == sensor_id
                df.loc[mask] = df.loc[mask].interpolate(method="time")
        else:
            # Single sensor, just interpolate
            df = df.interpolate(method="time")
        
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
        
        # Create async generator from list for async streaming
        async def async_record_generator():
            for record in records:
                yield record
        
        result = await self.db.stream_write(async_record_generator(), config)
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
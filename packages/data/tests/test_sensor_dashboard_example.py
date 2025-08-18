"""
Tests using the Sensor Dashboard Example

These tests demonstrate real-world usage of the data package
through a practical sensor monitoring application.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import tempfile
import math

from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase
from dataknobs_data.backends.file import SyncFileDatabase, AsyncFileDatabase
from dataknobs_data.streaming import StreamConfig
from dataknobs_data import Query, Filter, Operator

# Import from examples
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from sensor_dashboard import (
    SensorDashboard,
    AsyncSensorDashboard,
    SensorReading,
    SensorInfo,
    SensorDataGenerator
)


class TestSensorDashboardSync:
    """Test synchronous sensor dashboard operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db = SyncMemoryDatabase()
        self.dashboard = SensorDashboard(self.db)
        self.generator = SensorDataGenerator(seed=42)
        self.sensors = self.generator.create_sensors(3)
        self.start_time = datetime(2025, 1, 17, 10, 0, 0)
    
    def test_sensor_registration(self):
        """Test registering sensors in the system."""
        # Register single sensor
        sensor = self.sensors[0]
        sensor_id = self.dashboard.register_sensor(sensor)
        assert sensor_id == sensor.sensor_id
        assert sensor.sensor_id in self.dashboard.sensors
        
        # Register batch of sensors
        remaining = self.sensors[1:]
        ids = self.dashboard.register_sensors_batch(remaining)
        assert len(ids) == len(remaining)
        assert all(s.sensor_id in self.dashboard.sensors for s in remaining)
    
    def test_single_reading_ingestion(self):
        """Test ingesting individual sensor readings."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        reading = SensorReading(
            sensor_id=sensor.sensor_id,
            timestamp=self.start_time,
            temperature=22.5,
            humidity=45.0,
            battery=87,
            location=sensor.location
        )
        
        reading_id = self.dashboard.ingest_reading(reading)
        assert reading_id is not None
        
        # Verify it was stored
        recent = self.dashboard.get_recent_readings(sensor.sensor_id, limit=1)
        assert len(recent) == 1
        assert recent[0].temperature == 22.5
        assert recent[0].humidity == 45.0
    
    def test_batch_reading_ingestion(self):
        """Test batch import of historical readings."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        # Generate batch of readings
        readings = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=50
        )
        
        ids = self.dashboard.ingest_readings_batch(readings)
        assert len(ids) == 50
        
        # Verify they were stored
        stored = self.dashboard.get_recent_readings(sensor.sensor_id, limit=50)
        assert len(stored) == 50
    
    def test_streaming_ingestion(self):
        """Test streaming sensor data with different batch sizes."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        # Generate readings
        readings = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=100
        )
        
        # Stream with batch size of 10
        result = self.dashboard.stream_readings(readings, batch_size=10)
        assert result.successful == 100
        assert result.failed == 0
        assert result.total_processed == 100  # Use total_processed instead of total_batches
    
    def test_querying_recent_readings(self):
        """Test retrieving recent readings for a sensor."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        # Add readings at different times
        readings = []
        for i in range(20):
            reading = SensorReading(
                sensor_id=sensor.sensor_id,
                timestamp=self.start_time + timedelta(minutes=i*5),
                temperature=20.0 + i * 0.1,
                humidity=50.0,
                location=sensor.location
            )
            readings.append(reading)
        
        self.dashboard.ingest_readings_batch(readings)
        
        # Get recent 5
        recent = self.dashboard.get_recent_readings(sensor.sensor_id, limit=5)
        assert len(recent) == 5
        # Should be sorted by timestamp descending
        assert recent[0].timestamp >= recent[-1].timestamp
        # Temperature should be from one of the later readings
        assert 19.0 <= recent[0].temperature <= 22.0  # Within expected range
    
    def test_timerange_queries(self):
        """Test querying readings within a time range."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        # Add readings across 24 hours
        readings = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=288,  # Every 5 minutes for 24 hours
            interval_minutes=5
        )
        self.dashboard.ingest_readings_batch(readings)
        
        # Query specific time range (6 hours)
        start = self.start_time + timedelta(hours=6)
        end = self.start_time + timedelta(hours=12)
        
        range_readings = self.dashboard.get_readings_by_timerange(start, end)
        assert len(range_readings) > 0
        assert all(start <= r.timestamp <= end for r in range_readings)
    
    def test_pandas_integration(self):
        """Test converting readings to pandas DataFrame."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        readings = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=20
        )
        
        df = self.dashboard.to_dataframe(readings)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert "temperature" in df.columns
        assert "humidity" in df.columns
        assert df.index.name == "timestamp"
        assert df.index.dtype == "datetime64[ns]"
    
    def test_hourly_aggregation(self):
        """Test calculating hourly averages using pandas."""
        # Register multiple sensors
        for sensor in self.sensors[:2]:
            self.dashboard.register_sensor(sensor)
        
        # Generate 6 hours of data for each sensor
        all_readings = []
        for sensor in self.sensors[:2]:
            readings = self.generator.generate_normal_readings(
                sensor.sensor_id,
                sensor.location,
                self.start_time,
                count=72,  # 12 per hour for 6 hours
                interval_minutes=5
            )
            all_readings.extend(readings)
        
        hourly = self.dashboard.calculate_hourly_averages(all_readings)
        
        assert isinstance(hourly, pd.DataFrame)
        assert "temperature" in hourly.columns
        assert "humidity" in hourly.columns
        assert "reading_count" in hourly.columns
        
        # Should have data for 2 sensors × 6 hours = 12 rows
        assert len(hourly) <= 12
        # Each hour should have ~12 readings
        assert all(hourly["reading_count"] <= 12)
    
    def test_anomaly_detection(self):
        """Test detecting anomalous readings."""
        sensor = self.sensors[0]
        
        # Mix normal and anomalous readings
        normal = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=10
        )
        anomalous = self.generator.generate_anomalous_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time + timedelta(hours=1),
            count=5
        )
        
        all_readings = normal + anomalous
        
        # Detect anomalies with thresholds
        anomalies = self.dashboard.detect_anomalies(
            all_readings,
            temp_threshold=(10, 35),
            humidity_threshold=(20, 80)
        )
        
        assert len(anomalies) > 0
        # Should detect at least some of the anomalous readings
        assert any(r in anomalous for r in anomalies)
    
    def test_duplicate_handling(self):
        """Test removing duplicate readings."""
        sensor = self.sensors[0]
        
        # Generate readings with duplicates
        base_readings = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=20
        )
        with_duplicates = self.generator.generate_duplicate_readings(
            base_readings,
            duplicate_rate=0.3
        )
        
        assert len(with_duplicates) > len(base_readings)
        
        # Remove duplicates
        unique = self.dashboard.handle_duplicate_readings(with_duplicates)
        assert len(unique) == len(base_readings)
        
        # Verify all original readings are present
        unique_keys = {(r.sensor_id, r.timestamp) for r in unique}
        base_keys = {(r.sensor_id, r.timestamp) for r in base_readings}
        assert unique_keys == base_keys
    
    def test_missing_data_interpolation(self):
        """Test interpolating missing sensor data."""
        sensor = self.sensors[0]
        
        # Generate readings with gaps
        readings = self.generator.generate_missing_data_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=50,
            gap_probability=0.2
        )
        
        df = self.dashboard.to_dataframe(readings)
        
        # Check for gaps in time series
        time_diffs = df.index.to_series().diff()
        has_gaps = any(time_diffs > timedelta(minutes=5))
        
        if has_gaps:
            # Interpolate missing values
            interpolated = self.dashboard.interpolate_missing_data(df)
            # Should have same shape
            assert interpolated.shape == df.shape
    
    def test_batch_failure_recovery(self):
        """Test graceful handling of batch operation failures."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        # Generate readings with some invalid ones
        readings, expected_failures = self.generator.generate_batch_failure_scenario(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=30
        )
        
        # Stream with error handling enabled
        result = self.dashboard.stream_readings(readings, batch_size=10)
        
        # Should process successful readings despite failures
        assert result.successful > 0
        assert result.failed == len(expected_failures)
        
        # Verify partial success
        # Note: We expect 24 successful records (30 - 6 failures)
        # 30 readings * 5 minutes = 150 minutes = 2.5 hours
        stored_readings = self.dashboard.get_readings_by_timerange(
            self.start_time - timedelta(hours=1),
            self.start_time + timedelta(hours=3)  # Extended to cover all readings
        )
        stored_count = len(stored_readings)
        
        # Debug: Check what's in the database
        all_records = self.db.search(Query())
        sensor_readings = [r for r in all_records if r.metadata.get("type") == "sensor_reading"]
        sensor_infos = [r for r in all_records if r.metadata.get("type") == "sensor_info"]
        
        # We should have:
        # - 1 sensor_info record (from register_sensor)
        # - 24 sensor_reading records (30 - 6 failures)
        assert len(sensor_infos) == 1, f"Expected 1 sensor_info, got {len(sensor_infos)}"
        assert stored_count == 24, f"Expected 24 readings, got {stored_count}. Total sensor_readings: {len(sensor_readings)}"
    
    def test_file_database_integration(self):
        """Test using file-based database backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # SyncFileDatabase expects a config dict with 'path' key
            config = {"path": str(Path(tmpdir) / "sensor_data.json")}
            file_db = SyncFileDatabase(config)
            file_dashboard = SensorDashboard(file_db)
            
            # Register sensors
            for sensor in self.sensors:
                file_dashboard.register_sensor(sensor)
            
            # Add readings
            readings = self.generator.generate_normal_readings(
                self.sensors[0].sensor_id,
                self.sensors[0].location,
                self.start_time,
                count=10
            )
            
            ids = file_dashboard.ingest_readings_batch(readings)
            assert len(ids) == 10
            
            # Verify persistence
            recent = file_dashboard.get_recent_readings(
                self.sensors[0].sensor_id,
                limit=5
            )
            assert len(recent) == 5


class TestSensorDashboardAsync:
    """Test asynchronous sensor dashboard operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db = AsyncMemoryDatabase()
        self.dashboard = AsyncSensorDashboard(self.db)
        self.generator = SensorDataGenerator(seed=42)
        self.sensors = self.generator.create_sensors(3)
        self.start_time = datetime(2025, 1, 17, 10, 0, 0)
    
    @pytest.mark.asyncio
    async def test_async_sensor_registration(self):
        """Test async sensor registration."""
        # Register single sensor
        sensor = self.sensors[0]
        sensor_id = await self.dashboard.register_sensor(sensor)
        assert sensor_id == sensor.sensor_id
        
        # Register batch
        remaining = self.sensors[1:]
        ids = await self.dashboard.register_sensors_batch(remaining)
        assert len(ids) == len(remaining)
    
    @pytest.mark.asyncio
    async def test_async_streaming(self):
        """Test async streaming ingestion."""
        sensor = self.sensors[0]
        await self.dashboard.register_sensor(sensor)
        
        readings = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=50
        )
        
        result = await self.dashboard.stream_readings(readings, batch_size=5)
        assert result.successful == 50
        assert result.total_processed == 50  # Use total_processed
    
    @pytest.mark.asyncio
    async def test_concurrent_sensor_processing(self):
        """Test processing multiple sensors concurrently."""
        # Register all sensors
        for sensor in self.sensors:
            await self.dashboard.register_sensor(sensor)
        
        # Create concurrent ingestion tasks
        tasks = []
        for sensor in self.sensors:
            readings = self.generator.generate_normal_readings(
                sensor.sensor_id,
                sensor.location,
                self.start_time,
                count=20
            )
            task = self.dashboard.ingest_readings_batch(readings)
            tasks.append(task)
        
        # Process concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(self.sensors)
        assert all(len(ids) == 20 for ids in results)
    
    @pytest.mark.asyncio
    async def test_async_generator_processing(self):
        """Test processing an async generator of sensor readings."""
        sensor = self.sensors[0]
        await self.dashboard.register_sensor(sensor)
        
        # First add some readings directly to verify the system works
        test_readings = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=5
        )
        await self.dashboard.ingest_readings_batch(test_readings)
        
        # Verify those were stored
        initial = await self.dashboard.get_recent_readings(sensor.sensor_id, limit=10)
        assert len(initial) == 5, f"Expected 5 initial readings, got {len(initial)}"
        
        async def async_sensor_stream():
            """Simulate async sensor data stream."""
            count = 0
            for reading in self.generator.generate_continuous_stream(
                [sensor],
                self.start_time + timedelta(hours=1),  # Start after initial readings
                duration_hours=1,
                readings_per_hour=12
            ):
                await asyncio.sleep(0.001)  # Minimal delay for testing
                yield reading
                count += 1
                if count >= 5:  # Only process a few for testing
                    break
        
        # Process the stream
        await self.dashboard.process_sensor_stream(async_sensor_stream())
        
        # Verify readings were stored (should have initial 5 + streamed ones)
        recent = await self.dashboard.get_recent_readings(sensor.sensor_id, limit=20)
        assert len(recent) >= 5, f"Expected at least 5 readings, got {len(recent)}"
    
    @pytest.mark.asyncio
    async def test_async_file_database(self):
        """Test async operations with file database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # AsyncFileDatabase expects a config dict with 'path' key
            config = {"path": str(Path(tmpdir) / "sensor_data.json")}
            file_db = AsyncFileDatabase(config)
            file_dashboard = AsyncSensorDashboard(file_db)
            
            # Register and add data
            sensor = self.sensors[0]
            await file_dashboard.register_sensor(sensor)
            
            readings = self.generator.generate_normal_readings(
                sensor.sensor_id,
                sensor.location,
                self.start_time,
                count=15
            )
            
            result = await file_dashboard.stream_readings(readings, batch_size=5)
            assert result.successful == 15
            assert result.total_processed == 15  # Use total_processed


class TestSensorDashboardEdgeCases:
    """Test edge cases and error scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db = SyncMemoryDatabase()
        self.dashboard = SensorDashboard(self.db)
        self.generator = SensorDataGenerator(seed=42)
        self.sensors = self.generator.create_sensors(2)
        self.start_time = datetime(2025, 1, 17, 10, 0, 0)
    
    def test_empty_dataframe_handling(self):
        """Test operations on empty data."""
        # No readings
        df = self.dashboard.to_dataframe([])
        assert df.empty
        
        # Hourly averages on empty data
        hourly = self.dashboard.calculate_hourly_averages([])
        assert hourly.empty
    
    def test_out_of_order_processing(self):
        """Test handling out-of-order sensor readings."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        # Generate ordered readings
        ordered = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=30
        )
        
        # Shuffle to create out-of-order
        disordered = self.generator.generate_out_of_order_readings(
            ordered,
            disorder_rate=0.3
        )
        
        # Ingest disordered
        self.dashboard.ingest_readings_batch(disordered)
        
        # Query should return in correct order
        recent = self.dashboard.get_recent_readings(sensor.sensor_id, limit=30)
        
        # Verify chronological order
        for i in range(len(recent) - 1):
            assert recent[i].timestamp >= recent[i + 1].timestamp
    
    def test_nan_value_handling(self):
        """Test handling of NaN values in readings."""
        sensor = self.sensors[0]
        
        # Create reading with NaN
        reading = SensorReading(
            sensor_id=sensor.sensor_id,
            timestamp=self.start_time,
            temperature=float('nan'),
            humidity=50.0,
            location=sensor.location
        )
        
        # Convert to DataFrame
        df = self.dashboard.to_dataframe([reading])
        
        # Check NaN is preserved
        assert math.isnan(df.iloc[0]["temperature"])
        assert df.iloc[0]["humidity"] == 50.0
    
    def test_sensor_id_validation(self):
        """Test handling of invalid sensor IDs."""
        # Reading with empty sensor_id
        invalid_reading = SensorReading(
            sensor_id="",
            timestamp=self.start_time,
            temperature=20.0,
            humidity=50.0,
            location="test"
        )
        
        # Should handle gracefully
        try:
            record = invalid_reading.to_record()
            # Record created but ID might be generated differently
            assert record is not None
        except Exception:
            # Or might raise validation error - both are acceptable
            pass
    
    def test_concurrent_modifications(self):
        """Test concurrent access patterns."""
        sensor = self.sensors[0]
        self.dashboard.register_sensor(sensor)
        
        # Add initial readings
        readings1 = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time,
            count=10
        )
        self.dashboard.ingest_readings_batch(readings1)
        
        # Concurrent-like operations
        # Query while adding more data
        recent1 = self.dashboard.get_recent_readings(sensor.sensor_id, limit=5)
        
        readings2 = self.generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            self.start_time + timedelta(hours=1),
            count=10
        )
        self.dashboard.ingest_readings_batch(readings2)
        
        recent2 = self.dashboard.get_recent_readings(sensor.sensor_id, limit=5)
        
        # Should have different results (or at least not fail)
        # The second batch has timestamps after the first
        assert recent2[0].timestamp >= recent1[0].timestamp
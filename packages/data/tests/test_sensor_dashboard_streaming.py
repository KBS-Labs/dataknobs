"""
Test sensor dashboard streaming improvements.

These tests exercise the new batch/streaming features through
practical sensor monitoring scenarios.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from dataknobs_data import StreamConfig, StreamProcessor
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase
import sys
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from sensor_dashboard.sensor_dashboard import SensorDashboard, AsyncSensorDashboard
from sensor_dashboard.models import SensorInfo, SensorReading
from sensor_dashboard.data_generator import SensorDataGenerator


class TestSensorDashboardStreamingEnhancements:
    """Test enhanced streaming features through sensor dashboard."""
    
    @pytest.fixture
    def dashboard(self):
        """Create dashboard with in-memory database."""
        db = SyncMemoryDatabase()
        return SensorDashboard(db)
    
    @pytest.fixture
    def test_sensors(self):
        """Create test sensors."""
        now = datetime.now()
        return [
            SensorInfo("sensor_1", "DHT22", "lab_a", now - timedelta(days=30)),
            SensorInfo("sensor_2", "BME280", "lab_b", now - timedelta(days=20)),
        ]
    
    @pytest.fixture
    def test_readings(self):
        """Create test readings."""
        readings = []
        now = datetime.now()
        
        for i in range(25):
            reading = SensorReading(
                sensor_id=f"sensor_{(i % 2) + 1}",
                timestamp=now - timedelta(hours=25-i),
                temperature=20.0 + (i % 10),
                humidity=40.0 + (i % 20),
                battery=100 - (i * 2) if i < 40 else 20,
                location=f"lab_{'a' if i % 2 == 0 else 'b'}"
            )
            readings.append(reading)
        
        return readings
    
    def test_stream_readings_with_tracking(self, dashboard, test_sensors, test_readings):
        """Test streaming with batch and failure tracking."""
        # Register sensors
        dashboard.register_sensors_batch(test_sensors)
        
        # Stream readings with tracking
        result = dashboard.stream_readings_with_tracking(
            test_readings,
            batch_size=10,
            track_failures=True
        )
        
        # Verify enhanced StreamResult properties
        assert hasattr(result, 'total_batches')
        assert hasattr(result, 'failed_indices')
        
        # Check batch count (25 readings / 10 batch_size = 3 batches)
        assert result.total_batches == 3
        assert result.total_processed == 25
        assert result.successful == 25
        assert result.failed == 0
        assert result.failed_indices == []
    
    def test_stream_processor_adapter_usage(self, dashboard, test_sensors):
        """Test that StreamProcessor adapters are used correctly."""
        dashboard.register_sensors_batch(test_sensors)
        
        # Create a small set of readings
        readings = []
        now = datetime.now()
        for i in range(5):
            reading = SensorReading(
                sensor_id=test_sensors[i % 2].sensor_id,
                timestamp=now - timedelta(hours=i),
                temperature=22.0,
                humidity=50.0,
                battery=80,
                location=test_sensors[i % 2].location
            )
            readings.append(reading)
        
        # Stream using the method that explicitly uses StreamProcessor
        result = dashboard.stream_readings_with_tracking(readings, batch_size=3)
        
        assert result.total_processed == 5
        assert result.successful == 5
        assert result.total_batches == 2  # 3 + 2 records
    
    def test_streaming_with_validation_errors(self, dashboard, test_sensors):
        """Test streaming with validation that causes errors."""
        dashboard.register_sensors_batch(test_sensors)
        
        # Create readings with some invalid values
        readings = []
        now = datetime.now()
        
        # Mix valid and invalid readings
        for i in range(10):
            if i in [2, 5, 8]:  # These will be invalid
                reading = SensorReading(
                    sensor_id=test_sensors[0].sensor_id,
                    timestamp=now - timedelta(hours=i),
                    temperature=150.0,  # Out of range
                    humidity=50.0,
                    battery=80,
                    location=test_sensors[0].location
                )
            else:
                reading = SensorReading(
                    sensor_id=test_sensors[i % 2].sensor_id,
                    timestamp=now - timedelta(hours=i),
                    temperature=22.0,
                    humidity=50.0,
                    battery=80,
                    location=test_sensors[i % 2].location
                )
            readings.append(reading)
        
        # Stream with validation
        result = dashboard.stream_with_validation(
            readings,
            temp_range=(-50, 100),
            humidity_range=(0, 100)
        )
        
        # Should have processed all but failed on invalid ones
        assert result.total_processed == 10
        assert result.successful == 7  # Valid readings
        assert result.failed == 3  # Invalid readings
        
        # Check that failed indices are tracked
        assert len(result.failed_indices) == 3
    
    def test_batch_fallback_behavior(self, dashboard, test_sensors):
        """Test that batch operations fall back gracefully."""
        dashboard.register_sensors_batch(test_sensors)
        
        # Create a batch where batch operation will fail
        # but individual operations will partially succeed
        readings = []
        now = datetime.now()
        
        for i in range(8):
            if i % 3 == 0:  # Every 3rd is invalid
                reading = SensorReading(
                    sensor_id=test_sensors[0].sensor_id,
                    timestamp=now - timedelta(minutes=i*5),
                    temperature=-200.0,  # Way out of range
                    humidity=50.0,
                    battery=80,
                    location=test_sensors[0].location
                )
            else:
                reading = SensorReading(
                    sensor_id=test_sensors[0].sensor_id,
                    timestamp=now - timedelta(minutes=i*5),
                    temperature=23.0,
                    humidity=50.0,
                    battery=80,
                    location=test_sensors[0].location
                )
            readings.append(reading)
        
        # Stream with small batch size
        result = dashboard.stream_with_validation(
            readings,
            temp_range=(-50, 100),
            humidity_range=(0, 100)
        )
        
        # Verify fallback occurred
        assert result.total_batches > 0
        assert result.total_processed == 8
        # 3 invalid readings (indices 0, 3, 6)
        assert result.failed == 3
        assert result.successful == 5
    
    def test_different_batch_sizes(self, dashboard, test_sensors, test_readings):
        """Test streaming with different batch sizes."""
        dashboard.register_sensors_batch(test_sensors)
        
        # Test small batch size
        result_small = dashboard.stream_readings_with_tracking(
            test_readings[:15],
            batch_size=4
        )
        assert result_small.total_batches == 4  # 4+4+4+3
        
        # Test large batch size
        result_large = dashboard.stream_readings_with_tracking(
            test_readings[:15],
            batch_size=20
        )
        assert result_large.total_batches == 1  # All in one batch
        
        # Test exact batch size
        result_exact = dashboard.stream_readings_with_tracking(
            test_readings[:15],
            batch_size=5
        )
        assert result_exact.total_batches == 3  # 5+5+5


@pytest.mark.asyncio
class TestAsyncSensorDashboardStreaming:
    """Test async streaming improvements."""
    
    @pytest_asyncio.fixture
    async def async_dashboard(self):
        """Create async dashboard."""
        db = AsyncMemoryDatabase()
        await db.connect()  # Ensure database is connected
        return AsyncSensorDashboard(db)
    
    @pytest.fixture
    def test_sensors(self):
        """Create test sensors."""
        now = datetime.now()
        return [
            SensorInfo("async_sensor_1", "DHT22", "room_1", now - timedelta(days=10)),
            SensorInfo("async_sensor_2", "BME280", "room_2", now - timedelta(days=5)),
        ]
    
    async def test_async_stream_with_tracking(self, async_dashboard, test_sensors):
        """Test async streaming with enhanced tracking."""
        await async_dashboard.register_sensors_batch(test_sensors)
        
        # Create readings
        readings = []
        now = datetime.now()
        for i in range(20):
            reading = SensorReading(
                sensor_id=test_sensors[i % 2].sensor_id,
                timestamp=now - timedelta(hours=i),
                temperature=21.0 + (i % 5),
                humidity=45.0 + (i % 10),
                battery=100 - (i * 4),
                location=test_sensors[i % 2].location
            )
            readings.append(reading)
        
        # Stream with tracking
        result = await async_dashboard.stream_readings_with_tracking(
            readings,
            batch_size=7
        )
        
        # Verify results
        assert result.total_processed == 20
        assert result.successful == 20
        assert result.total_batches == 3  # 7+7+6
        assert result.failed_indices == []
    
    async def test_async_stream_processor_adapters(self, async_dashboard, test_sensors):
        """Test that async uses StreamProcessor adapters."""
        await async_dashboard.register_sensors_batch(test_sensors)
        
        readings = []
        now = datetime.now()
        for i in range(10):
            reading = SensorReading(
                sensor_id=test_sensors[i % 2].sensor_id,
                timestamp=now - timedelta(hours=i),
                temperature=22.0,
                humidity=50.0,
                battery=90,
                location=test_sensors[i % 2].location
            )
            readings.append(reading)
        
        # The async version should use StreamProcessor.list_to_async_iterator
        result = await async_dashboard.stream_readings(readings, batch_size=4)
        
        assert result.total_processed == 10
        assert result.successful == 10
        assert result.total_batches == 3  # 4+4+2


class TestStreamingStatistics:
    """Test that streaming provides useful statistics."""
    
    def test_success_rate_calculation(self):
        """Test success rate is calculated correctly."""
        db = SyncMemoryDatabase()
        dashboard = SensorDashboard(db)
        
        # Register a sensor
        sensor = SensorInfo("test_sensor", "DHT22", "test_lab", datetime.now())
        dashboard.register_sensor(sensor)
        
        # Create readings with controlled validation
        readings = []
        now = datetime.now()
        
        # 7 valid, 3 invalid for 70% success rate
        for i in range(10):
            if i in [3, 6, 9]:  # 30% invalid
                temp = 200.0  # Invalid
            else:
                temp = 22.0  # Valid
            
            reading = SensorReading(
                sensor_id=sensor.sensor_id,
                timestamp=now - timedelta(hours=i),
                temperature=temp,
                humidity=50.0,
                battery=80,
                location=sensor.location
            )
            readings.append(reading)
        
        result = dashboard.stream_with_validation(
            readings,
            temp_range=(-50, 100),
            humidity_range=(0, 100)
        )
        
        # Check statistics
        assert result.total_processed == 10
        assert result.successful == 7
        assert result.failed == 3
        assert result.success_rate == 70.0
    
    def test_empty_stream_statistics(self):
        """Test statistics with empty stream."""
        db = SyncMemoryDatabase()
        dashboard = SensorDashboard(db)
        
        # Stream empty list
        result = dashboard.stream_readings([], batch_size=10)
        
        assert result.total_processed == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.total_batches == 0
        assert result.success_rate == 0.0
        assert result.failed_indices == []
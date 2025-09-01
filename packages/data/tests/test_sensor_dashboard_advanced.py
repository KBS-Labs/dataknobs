"""
Test cases for sensor dashboard using advanced query features.

These tests exercise the new boolean logic and range operators through
practical sensor monitoring scenarios.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from dataknobs_data import (
    Query, Filter, Operator, QueryBuilder,
    ComplexQuery, Record
)
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase
import sys
from pathlib import Path
# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from sensor_dashboard.sensor_dashboard import SensorDashboard, AsyncSensorDashboard
from sensor_dashboard.models import SensorInfo, SensorReading
from sensor_dashboard.data_generator import SensorDataGenerator as DataGenerator


class TestSensorDashboardRangeOperators:
    """Test BETWEEN and NOT_BETWEEN operators through sensor dashboard."""
    
    @pytest.fixture
    def dashboard_with_data(self):
        """Create dashboard with test data."""
        db = SyncMemoryDatabase()
        dashboard = SensorDashboard(db)
        
        # Create sensors with specific readings
        now = datetime.now()
        sensors = [
            SensorInfo("sensor_1", "DHT22", "room_a", now - timedelta(days=30)),
            SensorInfo("sensor_2", "BME280", "room_b", now - timedelta(days=20)),
            SensorInfo("sensor_3", "DHT22", "room_c", now - timedelta(days=10)),
        ]
        
        dashboard.register_sensors_batch(sensors)
        
        # Create readings with varied conditions
        now = datetime.now()
        readings = [
            # Normal readings
            SensorReading("sensor_1", now - timedelta(hours=1), 22.0, 45.0, 80.0, "room_a"),
            SensorReading("sensor_1", now - timedelta(hours=2), 23.0, 50.0, 75.0, "room_a"),
            # Extreme temperature
            SensorReading("sensor_2", now - timedelta(hours=1), 38.0, 60.0, 60.0, "room_b"),
            # Low battery
            SensorReading("sensor_2", now - timedelta(hours=3), 25.0, 55.0, 15.0, "room_b"),
            # Extreme humidity
            SensorReading("sensor_3", now - timedelta(hours=1), 20.0, 85.0, 90.0, "room_c"),
            # Old reading
            SensorReading("sensor_3", now - timedelta(days=8), 21.0, 45.0, 100.0, "room_c"),
        ]
        
        dashboard.ingest_readings_batch(readings)
        
        return dashboard
    
    def test_time_range_query_with_between(self, dashboard_with_data):
        """Test time range queries using BETWEEN operator."""
        dashboard = dashboard_with_data
        now = datetime.now()
        
        # Get readings from last 4 hours
        start = now - timedelta(hours=4)
        end = now
        
        readings = dashboard.get_readings_by_timerange(start, end)
        
        assert len(readings) == 5  # All except the 8-day old reading
        
        # Verify all timestamps are within range
        for reading in readings:
            assert start <= reading.timestamp <= end
    
    def test_optimal_conditions_with_between(self, dashboard_with_data):
        """Test finding optimal conditions using multiple BETWEEN operators."""
        dashboard = dashboard_with_data
        
        # Define optimal ranges
        optimal = dashboard.get_optimal_conditions(
            temp_range=(20, 25),  # 20-25°C
            humidity_range=(40, 60),  # 40-60% humidity
            min_battery=50.0  # Battery >= 50%
        )
        
        # Should find readings that meet ALL criteria
        assert len(optimal) >= 2
        
        for reading in optimal:
            assert 20 <= reading.temperature <= 25
            assert 40 <= reading.humidity <= 60
            assert reading.battery >= 50
    
    def test_anomaly_detection_with_not_between(self, dashboard_with_data):
        """Test anomaly detection using NOT_BETWEEN operator."""
        dashboard = dashboard_with_data
        
        # Find readings outside normal ranges
        anomalies = dashboard.get_anomalous_readings_query(
            temp_range=(15, 30),  # Normal: 15-30°C
            humidity_range=(30, 70)  # Normal: 30-70% humidity
        )
        
        # Should find sensor_2 (38°C) and sensor_3 (85% humidity)
        assert len(anomalies) >= 2
        
        # Verify at least one temperature anomaly
        temp_anomaly = any(
            r.temperature < 15 or r.temperature > 30 
            for r in anomalies
        )
        assert temp_anomaly
        
        # Verify at least one humidity anomaly
        humidity_anomaly = any(
            r.humidity < 30 or r.humidity > 70
            for r in anomalies
        )
        assert humidity_anomaly
    
    def test_combined_range_queries(self, dashboard_with_data):
        """Test combining BETWEEN with other operators."""
        dashboard = dashboard_with_data
        
        # Query: Temperature in range AND battery not low
        query = Query(
            filters=[
                Filter("metadata.type", Operator.EQ, "sensor_reading"),
                Filter("temperature", Operator.BETWEEN, (20, 25)),
                Filter("battery", Operator.NOT_BETWEEN, (0, 30))  # Battery NOT low
            ]
        )
        
        records = dashboard.db.search(query)
        
        # Convert to readings
        readings = [SensorReading.from_record(r) for r in records]
        
        # Should find readings with good temp and good battery
        for reading in readings:
            assert 20 <= reading.temperature <= 25
            assert reading.battery > 30


class TestSensorDashboardBooleanLogic:
    """Test OR, AND, NOT operators through sensor dashboard."""
    
    @pytest.fixture
    def dashboard_with_critical_data(self):
        """Create dashboard with critical sensor scenarios."""
        db = SyncMemoryDatabase()
        dashboard = SensorDashboard(db)
        
        now = datetime.now()
        sensors = [
            SensorInfo("critical_1", "DHT22", "server_room", now - timedelta(days=30)),
            SensorInfo("critical_2", "BME280", "warehouse", now - timedelta(days=20)),
            SensorInfo("normal_1", "DHT22", "office", now - timedelta(days=10)),
        ]
        
        dashboard.register_sensors_batch(sensors)
        
        now = datetime.now()
        readings = [
            # Critical: Low battery in server room
            SensorReading("critical_1", now - timedelta(hours=1), 
                         25.0, 50.0, 10.0, "server_room"),
            # Critical: Extreme temperature
            SensorReading("critical_2", now - timedelta(hours=1), 
                         42.0, 60.0, 80.0, "warehouse"),
            # Critical: Very cold
            SensorReading("critical_2", now - timedelta(hours=2), 
                         3.0, 40.0, 75.0, "warehouse"),
            # Normal reading
            SensorReading("normal_1", now - timedelta(hours=1), 
                         22.0, 45.0, 90.0, "office"),
            # Old reading for maintenance check
            SensorReading("critical_1", now - timedelta(days=8), 
                         24.0, 48.0, 25.0, "server_room"),
        ]
        
        dashboard.ingest_readings_batch(readings)
        
        return dashboard
    
    def test_or_operator_multiple_locations(self, dashboard_with_critical_data):
        """Test OR operator for multiple location queries."""
        dashboard = dashboard_with_critical_data
        
        # Get readings from server_room OR warehouse
        readings = dashboard.get_sensors_by_multiple_locations(
            ["server_room", "warehouse"]
        )
        
        assert len(readings) == 4  # All except office
        
        # Verify locations
        locations = {r.location for r in readings}
        assert "server_room" in locations
        assert "warehouse" in locations
        assert "office" not in locations
    
    def test_complex_critical_conditions(self, dashboard_with_critical_data):
        """Test complex boolean logic for critical sensor detection."""
        dashboard = dashboard_with_critical_data
        
        # Find critical sensors: low battery OR extreme temps
        critical = dashboard.get_critical_sensors(
            battery_threshold=20.0,
            locations=["server_room", "warehouse"]
        )
        
        assert len(critical) >= 3  # Low battery + extreme temps
        
        # Verify we found the critical conditions
        has_low_battery = any(r.battery < 20 for r in critical)
        has_extreme_temp = any(
            r.temperature < 5 or r.temperature > 40 
            for r in critical
        )
        
        assert has_low_battery
        assert has_extreme_temp
    
    def test_not_operator_maintenance(self, dashboard_with_critical_data):
        """Test NOT operator for maintenance detection."""
        dashboard = dashboard_with_critical_data
        
        # Find sensors needing maintenance
        maintenance = dashboard.search_maintenance_needed()
        
        # Should find old readings and low battery old readings
        assert len(maintenance) >= 1
        
        # The 8-day old reading should be included
        old_reading = any(
            (datetime.now() - r.timestamp).days >= 7
            for r in maintenance
        )
        assert old_reading
    
    def test_combined_and_or_logic(self, dashboard_with_critical_data):
        """Test combining AND and OR operators."""
        dashboard = dashboard_with_critical_data
        
        # Query: (server_room OR warehouse) AND (battery < 50 OR temp > 30)
        query = Query().filter("metadata.type", Operator.EQ, "sensor_reading").or_(
            Filter("location", Operator.EQ, "server_room"),
            Filter("location", Operator.EQ, "warehouse")
        )
        
        # This creates a ComplexQuery
        # Now search with it
        results = dashboard.db.search(query)
        
        # Should get server_room and warehouse readings
        assert len(results) == 4
        
        # Further filter for critical conditions
        critical_query = Query().filter("metadata.type", Operator.EQ, "sensor_reading").or_(
            Filter("battery", Operator.LT, 50),
            Filter("temperature", Operator.GT, 30)
        )
        
        critical_results = dashboard.db.search(critical_query)
        assert len(critical_results) >= 2  # Low battery and high temp readings


class TestQueryBuilderIntegration:
    """Test QueryBuilder through sensor dashboard scenarios."""
    
    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with diverse sensor data."""
        db = SyncMemoryDatabase()
        dashboard = SensorDashboard(db)
        
        # Use data generator for variety
        generator = DataGenerator()
        sensors = generator.create_sensors(5)
        
        # Generate readings for each sensor
        readings = []
        now = datetime.now()
        for sensor in sensors:
            sensor_readings = generator.generate_normal_readings(
                sensor.sensor_id,
                sensor.location,
                now - timedelta(hours=24),
                count=48  # 24 hours * 2 readings per hour
            )
            readings.extend(sensor_readings)
        
        dashboard.register_sensors_batch(sensors)
        dashboard.ingest_readings_batch(readings)
        
        return dashboard
    
    def test_query_builder_step_by_step(self, populated_dashboard):
        """Test building complex queries step by step."""
        dashboard = populated_dashboard
        
        # Build: sensor readings with (temp > 25 OR humidity < 40) AND battery > 50
        builder = QueryBuilder()
        
        # Base condition
        builder.where("metadata.type", Operator.EQ, "sensor_reading")
        
        # Temperature or humidity condition
        temp_humidity = QueryBuilder().or_(
            Filter("temperature", Operator.GT, 25),
            Filter("humidity", Operator.LT, 40)
        )
        
        # Combine with battery condition
        builder.and_(temp_humidity)
        builder.where("battery", Operator.GT, 50)
        
        query = builder.build()
        results = dashboard.db.search(query)
        
        # Verify results match conditions
        for record in results:
            temp = record.get_value("temperature")
            humidity = record.get_value("humidity")
            battery = record.get_value("battery")
            
            assert (temp > 25 or humidity < 40)
            assert battery > 50
    
    def test_nested_not_conditions(self, populated_dashboard):
        """Test nested NOT conditions with QueryBuilder."""
        dashboard = populated_dashboard
        
        # Find sensors NOT in good condition
        # Good = (temp 18-28 AND humidity 30-70 AND battery > 50)
        builder = QueryBuilder()
        builder.where("metadata.type", Operator.EQ, "sensor_reading")
        
        good_conditions = (
            QueryBuilder()
            .where("temperature", Operator.BETWEEN, (18, 28))
            .where("humidity", Operator.BETWEEN, (30, 70))
            .where("battery", Operator.GT, 50)
        )
        
        builder.not_(good_conditions)
        
        query = builder.build()
        problematic = dashboard.db.search(query)
        
        # Each result should violate at least one "good" condition
        for record in problematic:
            temp = record.get_value("temperature")
            humidity = record.get_value("humidity")
            battery = record.get_value("battery")
            
            is_problematic = (
                temp < 18 or temp > 28 or
                humidity < 30 or humidity > 70 or
                battery is None or battery <= 50
            )
            assert is_problematic
    
    def test_query_builder_with_sorting_and_limits(self, populated_dashboard):
        """Test QueryBuilder with sorting and pagination."""
        dashboard = populated_dashboard
        
        # Get top 5 most recent high-temperature readings
        query = (
            QueryBuilder()
            .where("metadata.type", Operator.EQ, "sensor_reading")
            .where("temperature", Operator.GT, 25)
            .sort_by("timestamp", "desc")
            .limit(5)
            .build()
        )
        
        results = dashboard.db.search(query)
        
        assert len(results) <= 5
        
        # Verify temperature condition
        for record in results:
            assert record.get_value("temperature") > 25


@pytest.mark.asyncio
class TestAsyncSensorDashboardAdvanced:
    """Test advanced features with async dashboard."""
    
    @pytest_asyncio.fixture
    async def async_dashboard(self):
        """Create async dashboard with test data."""
        db = AsyncMemoryDatabase()
        await db.connect()  # Ensure database is connected
        dashboard = AsyncSensorDashboard(db)
        
        now = datetime.now()
        sensors = [
            SensorInfo("async_1", "DHT22", "lab", now - timedelta(days=15)),
            SensorInfo("async_2", "BME280", "storage", now - timedelta(days=10)),
        ]
        
        await dashboard.register_sensors_batch(sensors)
        
        now = datetime.now()
        readings = [
            SensorReading("async_1", now - timedelta(hours=1), 24.0, 50.0, 75.0, "lab"),
            SensorReading("async_1", now - timedelta(hours=2), 26.0, 48.0, 70.0, "lab"),
            SensorReading("async_2", now - timedelta(hours=1), 18.0, 65.0, 80.0, "storage"),
            SensorReading("async_2", now - timedelta(hours=3), 35.0, 30.0, 60.0, "storage"),
        ]
        
        await dashboard.ingest_readings_batch(readings)
        
        return dashboard
    
    async def test_async_complex_query(self, async_dashboard):
        """Test complex queries with async database."""
        dashboard = async_dashboard
        
        # Query: (location="lab" AND temp > 23) OR (location="storage" AND humidity < 40)
        builder = QueryBuilder()
        
        lab_condition = (
            QueryBuilder()
            .where("location", Operator.EQ, "lab")
            .where("temperature", Operator.GT, 23)
        )
        
        storage_condition = (
            QueryBuilder()
            .where("location", Operator.EQ, "storage")
            .where("humidity", Operator.LT, 40)
        )
        
        builder.where("metadata.type", Operator.EQ, "sensor_reading")
        builder.and_(QueryBuilder().or_(lab_condition, storage_condition))
        
        query = builder.build()
        results = await dashboard.db.search(query)
        
        assert len(results) >= 2  # At least one from each condition
        
        # Verify conditions
        for record in results:
            location = record.get_value("location")
            temp = record.get_value("temperature")
            humidity = record.get_value("humidity")
            
            if location == "lab":
                assert temp > 23
            elif location == "storage":
                assert humidity < 40


class TestRangeAndBooleanCombination:
    """Test combining range operators with boolean logic."""
    
    @pytest.fixture
    def dashboard(self):
        """Create dashboard for combination tests."""
        db = SyncMemoryDatabase()
        return SensorDashboard(db)
    
    def test_between_with_or_logic(self, dashboard):
        """Test BETWEEN combined with OR."""
        # Generate test data
        generator = DataGenerator()
        sensors = generator.create_sensors(3)
        
        # Generate readings for each sensor
        readings = []
        now = datetime.now()
        for sensor in sensors:
            sensor_readings = generator.generate_normal_readings(
                sensor.sensor_id,
                sensor.location,
                now - timedelta(hours=12),
                count=12  # 12 hours * 1 reading per hour
            )
            readings.extend(sensor_readings)
        
        dashboard.register_sensors_batch(sensors)
        dashboard.ingest_readings_batch(readings)
        
        # Query: temp BETWEEN 20-25 OR humidity NOT_BETWEEN 40-60
        query = Query().filter("metadata.type", Operator.EQ, "sensor_reading").or_(
            Filter("temperature", Operator.BETWEEN, (20, 25)),
            Filter("humidity", Operator.NOT_BETWEEN, (40, 60))
        )
        
        results = dashboard.db.search(query)
        
        for record in results:
            temp = record.get_value("temperature")
            humidity = record.get_value("humidity")
            
            # Should match at least one condition
            assert (20 <= temp <= 25) or (humidity < 40 or humidity > 60)
    
    def test_complex_range_with_not(self, dashboard):
        """Test NOT with range operators."""
        # Generate test data
        generator = DataGenerator()
        sensors = generator.create_sensors(2)
        
        # Generate readings for each sensor
        readings = []
        now = datetime.now()
        for sensor in sensors:
            sensor_readings = generator.generate_normal_readings(
                sensor.sensor_id,
                sensor.location,
                now - timedelta(hours=6),
                count=6  # 6 hours * 1 reading per hour
            )
            readings.extend(sensor_readings)
        
        dashboard.register_sensors_batch(sensors)
        dashboard.ingest_readings_batch(readings)
        
        # Query: NOT (temp BETWEEN 18-28 AND battery > 50)
        # This finds problematic sensors
        builder = QueryBuilder()
        builder.where("metadata.type", Operator.EQ, "sensor_reading")
        
        good_condition = (
            QueryBuilder()
            .where("temperature", Operator.BETWEEN, (18, 28))
            .where("battery", Operator.GT, 50)
        )
        
        builder.not_(good_condition)
        
        query = builder.build()
        results = dashboard.db.search(query)
        
        for record in results:
            temp = record.get_value("temperature")
            battery = record.get_value("battery")
            
            # Should NOT satisfy both conditions
            # Handle None battery values
            assert not (18 <= temp <= 28 and battery is not None and battery > 50)
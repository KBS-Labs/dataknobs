#!/usr/bin/env python3
"""
Demonstration of Advanced Query Features in Sensor Dashboard

This script showcases:
1. BETWEEN and NOT_BETWEEN operators for range queries
2. Boolean logic operators (OR, AND, NOT) for complex queries
3. QueryBuilder for fluent query construction
4. Real-world sensor monitoring scenarios
"""

import random
from datetime import datetime, timedelta
from dataknobs_data.backends.memory import SyncMemoryDatabase
from sensor_dashboard import SensorDashboard
from models import SensorInfo, SensorReading
from data_generator import SensorDataGenerator as DataGenerator


def demo_range_operators(dashboard: SensorDashboard):
    """Demonstrate BETWEEN and NOT_BETWEEN operators."""
    print("\n" + "="*60)
    print("DEMO: Range Operators (BETWEEN and NOT_BETWEEN)")
    print("="*60)
    
    # Generate sample data
    generator = DataGenerator()
    sensors = generator.create_sensors(3)
    
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
    
    # Register sensors and ingest readings
    dashboard.register_sensors_batch(sensors)
    dashboard.ingest_readings_batch(readings)
    
    # Demo 1: Get readings in a specific time range using BETWEEN
    print("\n1. Readings from last 6 hours using BETWEEN:")
    now = datetime.now()
    six_hours_ago = now - timedelta(hours=6)
    recent_readings = dashboard.get_readings_by_timerange(six_hours_ago, now)
    print(f"   Found {len(recent_readings)} readings in the last 6 hours")
    
    # Demo 2: Find optimal conditions using multiple BETWEEN
    print("\n2. Readings with optimal conditions (18-25°C, 40-60% humidity):")
    optimal = dashboard.get_optimal_conditions(
        temp_range=(18, 25),
        humidity_range=(40, 60),
        min_battery=50.0
    )
    print(f"   Found {len(optimal)} readings with optimal conditions")
    if optimal:
        sample = optimal[0]
        print(f"   Sample: Temp={sample.temperature}°C, Humidity={sample.humidity}%")
    
    # Demo 3: Find anomalies using NOT_BETWEEN
    print("\n3. Anomalous readings (outside normal ranges):")
    anomalies = dashboard.get_anomalous_readings_query(
        temp_range=(10, 35),
        humidity_range=(20, 80)
    )
    print(f"   Found {len(anomalies)} anomalous readings")
    for reading in anomalies[:3]:  # Show first 3
        print(f"   - Sensor {reading.sensor_id}: Temp={reading.temperature}°C, "
              f"Humidity={reading.humidity}%")


def demo_boolean_logic(dashboard: SensorDashboard):
    """Demonstrate OR, AND, NOT operators."""
    print("\n" + "="*60)
    print("DEMO: Boolean Logic Operators (OR, AND, NOT)")
    print("="*60)
    
    # Generate more diverse data
    generator = DataGenerator()
    sensors = generator.create_sensors(5)
    readings = []
    
    # Create specific test scenarios
    for i, sensor in enumerate(sensors):
        if i == 0:
            # Sensor with low battery in critical location
            for j in range(5):
                reading = SensorReading(
                    sensor_id=sensor.sensor_id,
                    timestamp=datetime.now() - timedelta(hours=j),
                    temperature=22.0 + random.uniform(-2, 2),
                    humidity=45.0 + random.uniform(-5, 5),
                    battery=15.0,  # Low battery
                    location="server_room"  # Critical location
                )
                readings.append(reading)
        elif i == 1:
            # Sensor with extreme temperatures
            for j in range(5):
                reading = SensorReading(
                    sensor_id=sensor.sensor_id,
                    timestamp=datetime.now() - timedelta(hours=j),
                    temperature=45.0 if j % 2 == 0 else 3.0,  # Extreme temps
                    humidity=50.0 + random.uniform(-10, 10),
                    battery=random.uniform(20, 100),
                    location=sensor.location
                )
                readings.append(reading)
        else:
            # Normal sensors
            sensor_readings = generator.generate_normal_readings(
                sensor.sensor_id,
                sensor.location,
                datetime.now() - timedelta(hours=1),
                count=12  # 1 hour of readings
            )
            readings.extend(sensor_readings)
    
    dashboard.register_sensors_batch(sensors)
    dashboard.ingest_readings_batch(readings)
    
    # Demo 1: OR operator for multiple locations
    print("\n1. Sensors from multiple locations using OR:")
    locations = ["server_room", "warehouse", "office"]
    multi_location = dashboard.get_sensors_by_multiple_locations(locations)
    print(f"   Found {len(multi_location)} readings from {locations}")
    
    # Demo 2: Complex boolean logic for critical sensors
    print("\n2. Critical sensors (low battery OR extreme temps):")
    critical = dashboard.get_critical_sensors(
        battery_threshold=20.0,
        locations=["server_room", "warehouse"]
    )
    print(f"   Found {len(critical)} critical readings")
    
    # Demo 3: NOT operator for maintenance detection
    print("\n3. Sensors needing maintenance (using NOT):")
    maintenance = dashboard.search_maintenance_needed()
    print(f"   Found {len(maintenance)} sensors needing attention")


def demo_query_builder(dashboard: SensorDashboard):
    """Demonstrate QueryBuilder for complex queries."""
    print("\n" + "="*60)
    print("DEMO: QueryBuilder for Complex Queries")
    print("="*60)
    
    from dataknobs_data import QueryBuilder, Filter, Operator
    
    # Generate test data
    generator = DataGenerator()
    sensors = generator.create_sensors(4)
    
    # Generate readings for each sensor
    readings = []
    now = datetime.now()
    for sensor in sensors:
        sensor_readings = generator.generate_normal_readings(
            sensor.sensor_id,
            sensor.location,
            now - timedelta(hours=48),
            count=48  # 48 hours * 1 reading per hour
        )
        readings.extend(sensor_readings)
    
    dashboard.register_sensors_batch(sensors)
    dashboard.ingest_readings_batch(readings)
    
    # Demo 1: Build a complex query step by step
    print("\n1. Building complex query with QueryBuilder:")
    
    # Find: (temp > 25 AND humidity < 50) OR (battery < 20 AND location = "warehouse")
    builder = QueryBuilder()
    
    # Base filter
    builder.where("metadata.type", Operator.EQ, "sensor_reading")
    
    # Hot and dry conditions
    hot_dry = (
        QueryBuilder()
        .where("temperature", Operator.GT, 25)
        .where("humidity", Operator.LT, 50)
    )
    
    # Low battery in warehouse
    low_battery_warehouse = (
        QueryBuilder()
        .where("battery", Operator.LT, 20)
        .where("location", Operator.EQ, "warehouse")
    )
    
    # Combine with OR
    builder.and_(
        QueryBuilder().or_(hot_dry, low_battery_warehouse)
    )
    
    # Add sorting and limit
    query = builder.sort_by("timestamp", "desc").limit(10).build()
    
    results = dashboard.db.search(query)
    print(f"   Found {len(results)} matching records")
    
    # Demo 2: Nested NOT conditions
    print("\n2. Complex NOT conditions:")
    
    # Find sensors that are NOT (in optimal range AND have good battery)
    # i.e., sensors with problems
    builder = QueryBuilder()
    builder.where("metadata.type", Operator.EQ, "sensor_reading")
    
    # Define "good" conditions
    good_conditions = (
        QueryBuilder()
        .where("temperature", Operator.BETWEEN, (18, 28))
        .where("humidity", Operator.BETWEEN, (30, 70))
        .where("battery", Operator.GT, 50)
    )
    
    # Negate to find problematic sensors
    builder.not_(good_conditions)
    
    query = builder.build()
    problematic = dashboard.db.search(query)
    print(f"   Found {len(problematic)} sensors with issues")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("SENSOR DASHBOARD - ADVANCED QUERY FEATURES DEMO")
    print("="*60)
    
    # Initialize in-memory database and dashboard
    db = SyncMemoryDatabase()
    dashboard = SensorDashboard(db)
    
    # Run demonstrations
    demo_range_operators(dashboard)
    demo_boolean_logic(dashboard)
    demo_query_builder(dashboard)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. BETWEEN/NOT_BETWEEN operators simplify range queries")
    print("2. Boolean logic (OR, AND, NOT) enables complex conditions")
    print("3. QueryBuilder provides a fluent API for query construction")
    print("4. These features make real-world queries intuitive and efficient")


if __name__ == "__main__":
    main()
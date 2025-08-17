"""Debug test for async generator processing."""
import pytest
import asyncio
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data import Query, Filter, Operator
from sensor_dashboard import AsyncSensorDashboard, SensorDataGenerator, SensorInfo


@pytest.mark.asyncio
async def test_async_generator_debug():
    """Debug async generator processing."""
    db = AsyncMemoryDatabase()
    dashboard = AsyncSensorDashboard(db)
    generator = SensorDataGenerator(seed=42)
    
    # Create a sensor
    sensor = SensorInfo(
        sensor_id="test_sensor",
        sensor_type="TH100",
        location="test_room",
        installed=datetime.now(),
        status="active"
    )
    
    # Register sensor
    sensor_id = await dashboard.register_sensor(sensor)
    print(f"Registered sensor: {sensor_id}")
    
    # Create some readings manually
    readings = generator.generate_normal_readings(
        sensor.sensor_id,
        sensor.location,
        datetime.now(),
        count=5
    )
    
    # Ingest them
    ids = await dashboard.ingest_readings_batch(readings)
    print(f"Ingested {len(ids)} readings: {ids[:2]}...")
    
    # Check what's in storage
    print(f"Storage has {len(db._storage)} records")
    for key in list(db._storage.keys())[:3]:
        record = db._storage[key]
        print(f"  - {key}: type={record.fields.get('_type')}, sensor_id={record.fields.get('sensor_id')}")
    
    # Try different queries
    # Query 1: Get all records
    query1 = Query()
    results1 = await db.search(query1)
    print(f"Query all: {len(results1)} results")
    
    # Query 2: Filter by _type
    query2 = Query(filters=[Filter("_type", Operator.EQ, "sensor_reading")])
    results2 = await db.search(query2)
    print(f"Query _type=sensor_reading: {len(results2)} results")
    
    # Query 3: Filter by sensor_id
    query3 = Query(filters=[Filter("sensor_id", Operator.EQ, sensor.sensor_id)])
    results3 = await db.search(query3)
    print(f"Query sensor_id={sensor.sensor_id}: {len(results3)} results")
    
    # Query 4: Combined filters (what dashboard uses)
    query4 = Query(filters=[
        Filter("_type", Operator.EQ, "sensor_reading"),
        Filter("sensor_id", Operator.EQ, sensor.sensor_id)
    ])
    results4 = await db.search(query4)
    print(f"Query combined: {len(results4)} results")
    
    # Use dashboard method
    recent = await dashboard.get_recent_readings(sensor.sensor_id, limit=10)
    print(f"Dashboard get_recent_readings: {len(recent)} results")
    
    # Now test async generator
    async def async_sensor_stream():
        """Simulate async sensor data stream."""
        for reading in generator.generate_continuous_stream(
            [sensor],
            datetime.now(),
            duration_hours=1,
            readings_per_hour=12
        ):
            await asyncio.sleep(0.001)  # Minimal delay
            yield reading
            # Only yield a few for testing
            break
    
    # Process the stream
    await dashboard.process_sensor_stream(async_sensor_stream())
    
    # Check results after generator
    print(f"After generator, storage has {len(db._storage)} records")
    recent_after = await dashboard.get_recent_readings(sensor.sensor_id, limit=20)
    print(f"After generator, get_recent_readings: {len(recent_after)} results")


if __name__ == "__main__":
    asyncio.run(test_async_generator_debug())
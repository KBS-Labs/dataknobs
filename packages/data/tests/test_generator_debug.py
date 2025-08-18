"""Debug the data generator."""
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from sensor_dashboard import SensorDataGenerator, SensorInfo


def test_generator():
    """Test the continuous stream generator."""
    generator = SensorDataGenerator(seed=42)
    
    sensor = SensorInfo(
        sensor_id="sensor_000",
        sensor_type="TH100", 
        location="room_a",
        installed=datetime.now(),
        status="active"
    )
    
    print(f"Sensor status: {sensor.status}")
    
    count = 0
    for reading in generator.generate_continuous_stream(
        [sensor],
        datetime(2025, 1, 17, 10, 0, 0),
        duration_hours=1,
        readings_per_hour=12
    ):
        count += 1
        if count <= 3:
            print(f"Reading {count}: sensor={reading.sensor_id}, time={reading.timestamp}, temp={reading.temperature}")
        if count >= 12:
            break
    
    print(f"Total readings generated: {count}")


if __name__ == "__main__":
    test_generator()
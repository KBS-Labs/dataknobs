"""
Test Data Generator for Sensor Dashboard

Generates realistic sensor data with various edge cases for testing.
"""
import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
try:
    from .models import SensorReading, SensorInfo
except ImportError:
    # When run as script, use absolute import
    from models import SensorReading, SensorInfo


class SensorDataGenerator:
    """Generate synthetic sensor data for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed:
            random.seed(seed)
        
        self.sensor_types = ["TH100", "TH200", "ENV-SENSOR-X"]
        self.locations = ["room_a", "room_b", "warehouse", "lobby", "server_room"]
    
    def create_sensors(self, count: int = 5) -> List[SensorInfo]:
        """Create a set of sensor devices."""
        sensors = []
        base_date = datetime.now() - timedelta(days=90)
        
        for i in range(count):
            sensor = SensorInfo(
                sensor_id=f"sensor_{i:03d}",
                sensor_type=random.choice(self.sensor_types),
                location=self.locations[i % len(self.locations)],
                installed=base_date + timedelta(days=random.randint(0, 30)),
                status="active" if random.random() > 0.1 else "maintenance"
            )
            sensors.append(sensor)
        
        return sensors
    
    def generate_normal_readings(self, 
                                sensor_id: str,
                                location: str,
                                start_time: datetime,
                                count: int = 100,
                                interval_minutes: int = 5) -> List[SensorReading]:
        """Generate normal sensor readings with realistic variations."""
        readings = []
        
        # Base values for location
        base_temp, base_humidity = self._get_location_baseline(location)
        current_time = start_time
        
        for i in range(count):
            # Add realistic variations
            temp_variation = random.gauss(0, 1.5)  # ±1.5°C variation
            humidity_variation = random.gauss(0, 5)  # ±5% variation
            
            # Simulate time-of-day effects
            hour = current_time.hour
            if 9 <= hour <= 17:  # Daytime - slightly warmer
                temp_variation += 1.5
            elif hour < 6 or hour > 22:  # Night - cooler
                temp_variation -= 2.0
            
            reading = SensorReading(
                sensor_id=sensor_id,
                timestamp=current_time,
                temperature=round(base_temp + temp_variation, 1),
                humidity=round(max(0, min(100, base_humidity + humidity_variation)), 1),
                battery=max(10, 100 - i // 10) if random.random() > 0.1 else None,  # Slow drain
                location=location
            )
            readings.append(reading)
            current_time += timedelta(minutes=interval_minutes)
        
        return readings
    
    def generate_anomalous_readings(self,
                                   sensor_id: str,
                                   location: str,
                                   start_time: datetime,
                                   count: int = 20) -> List[SensorReading]:
        """Generate readings with anomalies."""
        readings = []
        current_time = start_time
        
        for i in range(count):
            anomaly_type = random.choice(["extreme_temp", "extreme_humidity", "sensor_fault"])
            
            if anomaly_type == "extreme_temp":
                temperature = random.choice([random.uniform(-10, 5), random.uniform(40, 60)])
                humidity = random.uniform(30, 70)
            elif anomaly_type == "extreme_humidity":
                temperature = random.uniform(18, 25)
                humidity = random.choice([random.uniform(0, 10), random.uniform(90, 100)])
            else:  # sensor_fault
                temperature = random.choice([0.0, -999.0, 999.0])  # Fault values
                humidity = random.choice([0.0, -1.0, 200.0])
            
            reading = SensorReading(
                sensor_id=sensor_id,
                timestamp=current_time,
                temperature=round(temperature, 1),
                humidity=round(humidity, 1),
                battery=random.randint(0, 100) if random.random() > 0.3 else None,
                location=location
            )
            readings.append(reading)
            current_time += timedelta(minutes=random.randint(10, 60))
        
        return readings
    
    def generate_duplicate_readings(self,
                                   base_readings: List[SensorReading],
                                   duplicate_rate: float = 0.1) -> List[SensorReading]:
        """Add duplicate readings to simulate retry/network issues."""
        with_duplicates = []
        
        for reading in base_readings:
            with_duplicates.append(reading)
            if random.random() < duplicate_rate:
                # Add duplicate
                with_duplicates.append(reading)
                if random.random() < 0.3:  # Sometimes triple
                    with_duplicates.append(reading)
        
        return with_duplicates
    
    def generate_out_of_order_readings(self,
                                      readings: List[SensorReading],
                                      disorder_rate: float = 0.1) -> List[SensorReading]:
        """Shuffle some readings to simulate out-of-order delivery."""
        result = readings.copy()
        
        for i in range(len(result) - 1):
            if random.random() < disorder_rate:
                # Swap with next reading
                result[i], result[i + 1] = result[i + 1], result[i]
        
        return result
    
    def generate_missing_data_readings(self,
                                      sensor_id: str,
                                      location: str,
                                      start_time: datetime,
                                      count: int = 100,
                                      gap_probability: float = 0.15) -> List[SensorReading]:
        """Generate readings with missing data gaps."""
        readings = []
        current_time = start_time
        base_temp, base_humidity = self._get_location_baseline(location)
        
        for i in range(count):
            if random.random() > gap_probability:
                reading = SensorReading(
                    sensor_id=sensor_id,
                    timestamp=current_time,
                    temperature=round(base_temp + random.gauss(0, 1.5), 1),
                    humidity=round(base_humidity + random.gauss(0, 5), 1),
                    battery=max(10, 100 - i // 10),
                    location=location
                )
                readings.append(reading)
            # Skip this timestamp if gap
            current_time += timedelta(minutes=5)
        
        return readings
    
    def generate_batch_failure_scenario(self,
                                       sensor_id: str,
                                       location: str,
                                       start_time: datetime,
                                       count: int = 50) -> Tuple[List[SensorReading], List[int]]:
        """Generate readings where some will fail validation."""
        readings = []
        expected_failures = []
        current_time = start_time
        
        for i in range(count):
            if i % 10 == 5:  # Every 10th reading at position 5 fails
                # Invalid reading
                reading = SensorReading(
                    sensor_id="",  # Invalid: empty sensor_id
                    timestamp=current_time,
                    temperature=20.0,
                    humidity=50.0,
                    location=location
                )
                expected_failures.append(i)
            elif i % 10 == 7:
                # Another type of invalid reading
                reading = SensorReading(
                    sensor_id=sensor_id,
                    timestamp=current_time,
                    temperature=float('nan'),  # Invalid: NaN
                    humidity=50.0,
                    location=location
                )
                expected_failures.append(i)
            else:
                # Valid reading
                reading = SensorReading(
                    sensor_id=sensor_id,
                    timestamp=current_time,
                    temperature=round(20.0 + random.gauss(0, 2), 1),
                    humidity=round(50.0 + random.gauss(0, 5), 1),
                    battery=random.randint(50, 100),
                    location=location
                )
            
            readings.append(reading)
            current_time += timedelta(minutes=5)
        
        return readings, expected_failures
    
    def generate_continuous_stream(self,
                                  sensors: List[SensorInfo],
                                  start_time: datetime,
                                  duration_hours: int = 24,
                                  readings_per_hour: int = 12):
        """Generate a continuous stream of readings from multiple sensors."""
        end_time = start_time + timedelta(hours=duration_hours)
        current_time = start_time
        interval = timedelta(minutes=60 // readings_per_hour)
        
        while current_time < end_time:
            for sensor in sensors:
                if sensor.status == "active":
                    base_temp, base_humidity = self._get_location_baseline(sensor.location)
                    
                    # Occasionally skip a reading (sensor offline)
                    if random.random() > 0.02:
                        yield SensorReading(
                            sensor_id=sensor.sensor_id,
                            timestamp=current_time,
                            temperature=round(base_temp + random.gauss(0, 1.5), 1),
                            humidity=round(base_humidity + random.gauss(0, 5), 1),
                            battery=random.randint(20, 100) if random.random() > 0.1 else None,
                            location=sensor.location
                        )
            
            current_time += interval
    
    def _get_location_baseline(self, location: str) -> Tuple[float, float]:
        """Get baseline temperature and humidity for a location."""
        baselines = {
            "room_a": (22.0, 45.0),
            "room_b": (21.0, 50.0),
            "warehouse": (18.0, 60.0),
            "lobby": (23.0, 40.0),
            "server_room": (19.0, 35.0)
        }
        return baselines.get(location, (20.0, 50.0))
"""
Sensor Dashboard Data Models

Simple data models for our sensor monitoring example.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from dataknobs_data import Record


@dataclass
class SensorReading:
    """A single sensor reading with temperature and humidity."""
    sensor_id: str
    timestamp: datetime
    temperature: float  # Celsius
    humidity: float    # Percentage 0-100
    battery: Optional[int] = None  # Battery percentage 0-100
    location: Optional[str] = None
    
    def to_record(self) -> Record:
        """Convert to a dataknobs Record."""
        return Record(
            id=f"{self.sensor_id}_{self.timestamp.isoformat()}",
            data={
                # Store type as top-level field for querying (workaround for nested query limitation)
                "_type": "sensor_reading",
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
        """Create from a dataknobs Record."""
        data = {f.name: f.value for f in record.fields.values()}
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
    """Sensor device information and metadata."""
    sensor_id: str
    sensor_type: str
    location: str
    installed: datetime
    status: str = "active"  # active, inactive, maintenance
    
    def to_record(self) -> Record:
        """Convert to a dataknobs Record."""
        return Record(
            id=self.sensor_id,
            data={
                # Store type as top-level field for querying (workaround for nested query limitation)
                "_type": "sensor_info",
                "sensor_id": self.sensor_id,
                "sensor_type": self.sensor_type,
                "location": self.location,
                "installed": self.installed.isoformat(),
                "status": self.status
            },
            metadata={
                "type": "sensor_info"
            }
        )
    
    @classmethod
    def from_record(cls, record: Record) -> "SensorInfo":
        """Create from a dataknobs Record."""
        data = {f.name: f.value for f in record.fields.values()}
        return cls(
            sensor_id=data["sensor_id"],
            sensor_type=data["sensor_type"],
            location=data["location"],
            installed=datetime.fromisoformat(data["installed"]),
            status=data.get("status", "active")
        )
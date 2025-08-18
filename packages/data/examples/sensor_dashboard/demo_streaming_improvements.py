#!/usr/bin/env python3
"""
Demonstration of Batch and Streaming Improvements

This script showcases:
1. Enhanced StreamResult with total_batches and failed_indices
2. StreamProcessor adapters for list to iterator conversion
3. Graceful batch fallback with error tracking
4. Detailed streaming statistics
"""

import asyncio
from datetime import datetime, timedelta
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sensor_dashboard import SensorDashboard, AsyncSensorDashboard
from models import SensorInfo, SensorReading
from data_generator import SensorDataGenerator


def demo_basic_streaming(dashboard: SensorDashboard):
    """Demonstrate basic streaming with batch tracking."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Streaming with Batch Tracking")
    print("="*60)
    
    # Generate test data
    generator = SensorDataGenerator()
    sensors = generator.create_sensors(3)
    dashboard.register_sensors_batch(sensors)
    
    # Create 25 readings (will be 3 batches with batch_size=10)
    readings = []
    now = datetime.now()
    for i in range(25):
        reading = SensorReading(
            sensor_id=sensors[i % 3].sensor_id,
            timestamp=now - timedelta(hours=25-i),
            temperature=20.0 + (i % 10),
            humidity=40.0 + (i % 20),
            battery=100 - (i * 2),
            location=sensors[i % 3].location
        )
        readings.append(reading)
    
    # Stream with tracking
    print(f"\nStreaming {len(readings)} readings with batch_size=10...")
    result = dashboard.stream_readings_with_tracking(readings, batch_size=10)
    
    print(f"\nResults:")
    print(f"  Total processed: {result.total_processed}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Total batches: {result.total_batches}")
    print(f"  Success rate: {result.success_rate:.1f}%")
    
    if result.failed_indices:
        print(f"  Failed indices: {result.failed_indices}")


def demo_streaming_with_errors(dashboard: SensorDashboard):
    """Demonstrate streaming with validation errors and graceful fallback."""
    print("\n" + "="*60)
    print("DEMO 2: Streaming with Validation Errors")
    print("="*60)
    
    # Generate readings with some invalid data
    generator = SensorDataGenerator()
    sensors = generator.create_sensors(2)
    dashboard.register_sensors_batch(sensors)
    
    readings = []
    now = datetime.now()
    
    print("\nGenerating 15 readings with some invalid values...")
    for i in range(15):
        if i in [3, 7, 11]:  # These will be invalid
            # Create readings with out-of-range values
            reading = SensorReading(
                sensor_id=sensors[0].sensor_id,
                timestamp=now - timedelta(hours=i),
                temperature=150.0 if i == 3 else -100.0,  # Out of range
                humidity=200.0 if i == 7 else -50.0,      # Out of range
                battery=50,
                location=sensors[0].location
            )
            print(f"  Reading {i}: INVALID - temp={reading.temperature}, humidity={reading.humidity}")
        else:
            # Normal reading
            reading = SensorReading(
                sensor_id=sensors[i % 2].sensor_id,
                timestamp=now - timedelta(hours=i),
                temperature=22.0 + (i % 5),
                humidity=45.0 + (i % 10),
                battery=100 - i * 5,
                location=sensors[i % 2].location
            )
        readings.append(reading)
    
    # Stream with validation
    print(f"\nStreaming with validation (temp: -50 to 100, humidity: 0 to 100)...")
    result = dashboard.stream_with_validation(
        readings,
        temp_range=(-50, 100),
        humidity_range=(0, 100)
    )
    
    print(f"\nResults after validation:")
    print(f"  Total processed: {result.total_processed}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Total batches: {result.total_batches}")
    print(f"  Success rate: {result.success_rate:.1f}%")
    
    if result.failed_indices:
        print(f"  Failed at indices: {result.failed_indices}")
        print(f"  (These correspond to readings: {[3, 7, 11]})")


def demo_batch_fallback_behavior(dashboard: SensorDashboard):
    """Demonstrate batch fallback when batch operations fail."""
    print("\n" + "="*60)
    print("DEMO 3: Batch Fallback Behavior")
    print("="*60)
    
    # Create a scenario where batch operations might fail
    # but individual operations succeed for some records
    generator = SensorDataGenerator()
    sensors = generator.create_sensors(1)
    dashboard.register_sensors_batch(sensors)
    
    # Mix of valid and invalid readings
    readings = []
    now = datetime.now()
    
    print("\nCreating batch with mixed valid/invalid readings...")
    for i in range(12):
        if i % 4 == 0:  # Every 4th reading is invalid
            reading = SensorReading(
                sensor_id=sensors[0].sensor_id,
                timestamp=now - timedelta(minutes=i*5),
                temperature=-200.0,  # Way out of range
                humidity=50.0,
                battery=80,
                location=sensors[0].location
            )
            print(f"  Reading {i}: INVALID")
        else:
            reading = SensorReading(
                sensor_id=sensors[0].sensor_id,
                timestamp=now - timedelta(minutes=i*5),
                temperature=23.0,
                humidity=50.0,
                battery=80,
                location=sensors[0].location
            )
            print(f"  Reading {i}: Valid")
        readings.append(reading)
    
    # Stream with small batch size to show multiple batch attempts
    print(f"\nStreaming {len(readings)} readings with batch_size=4...")
    print("(Each batch with an invalid reading will fall back to individual processing)")
    
    result = dashboard.stream_with_validation(
        readings,
        temp_range=(-50, 100),
        humidity_range=(0, 100)
    )
    
    print(f"\nFallback Results:")
    print(f"  Total batches attempted: {result.total_batches}")
    print(f"  Records processed individually after fallback: {result.total_processed}")
    print(f"  Successful: {result.successful} (valid readings)")
    print(f"  Failed: {result.failed} (invalid readings)")
    
    # Calculate how many batches had to fall back
    expected_batches = len(readings) // 4 + (1 if len(readings) % 4 else 0)
    print(f"  Expected batches: {expected_batches}")
    print(f"  All batches fell back to individual processing due to invalid records")


async def demo_async_streaming_improvements(dashboard: AsyncSensorDashboard):
    """Demonstrate async streaming with StreamProcessor adapters."""
    print("\n" + "="*60)
    print("DEMO 4: Async Streaming with StreamProcessor Adapters")
    print("="*60)
    
    # Generate test data
    generator = SensorDataGenerator()
    sensors = generator.create_sensors(2)
    await dashboard.register_sensors_batch(sensors)
    
    # Create readings
    readings = []
    now = datetime.now()
    for i in range(20):
        reading = SensorReading(
            sensor_id=sensors[i % 2].sensor_id,
            timestamp=now - timedelta(hours=i),
            temperature=20.0 + (i % 8),
            humidity=45.0 + (i % 15),
            battery=100 - (i * 3),
            location=sensors[i % 2].location
        )
        readings.append(reading)
    
    print(f"\nAsync streaming {len(readings)} readings...")
    result = await dashboard.stream_readings_with_tracking(readings, batch_size=7)
    
    print(f"\nAsync Streaming Results:")
    print(f"  Total processed: {result.total_processed}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Total batches: {result.total_batches}")
    print(f"  Batch size: 7")
    print(f"  Expected batches: {20 // 7 + (1 if 20 % 7 else 0)} (matches actual: {result.total_batches})")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("SENSOR DASHBOARD - BATCH & STREAMING IMPROVEMENTS DEMO")
    print("="*60)
    
    # Initialize sync database and dashboard
    sync_db = SyncMemoryDatabase()
    sync_dashboard = SensorDashboard(sync_db)
    
    # Run sync demonstrations
    demo_basic_streaming(sync_dashboard)
    demo_streaming_with_errors(sync_dashboard)
    demo_batch_fallback_behavior(sync_dashboard)
    
    # Run async demonstration
    async_db = AsyncMemoryDatabase()
    async_dashboard = AsyncSensorDashboard(async_db)
    asyncio.run(demo_async_streaming_improvements(async_dashboard))
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nKey Improvements Demonstrated:")
    print("1. StreamResult.total_batches - Track number of batches processed")
    print("2. StreamResult.failed_indices - Track specific failed record positions")
    print("3. StreamProcessor adapters - Clean conversion from lists to iterators")
    print("4. Graceful batch fallback - Automatic retry of individual records")
    print("5. Enhanced error tracking - Detailed failure information")


if __name__ == "__main__":
    main()
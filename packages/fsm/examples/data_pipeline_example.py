"""
Data Pipeline Example using FSM - Fixed Version

This example demonstrates how to build a robust data processing pipeline
using the FSM framework with real-world features:
- Data validation
- Stream processing
- Error recovery
- Resource management
- Progress tracking
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dataknobs_fsm.config.validator import ConfigValidator
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.state import State
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import ITransformFunction, FunctionContext
from dataknobs_fsm.functions.manager import FunctionManager
from dataknobs_fsm.resources.properties import PropertiesResource
from dataknobs_fsm.resources.manager import ResourceManager


class DataValidator(ITransformFunction):
    """Validates incoming data records."""

    def transform(self, data: Any, context: FunctionContext) -> Any:
        """Validate data structure and content."""
        # Check required fields
        required_fields = ['id', 'timestamp', 'value']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate data types
        if not isinstance(data['id'], (int, str)):
            raise ValueError("ID must be int or string")

        if not isinstance(data['value'], (int, float)):
            raise ValueError("Value must be numeric")

        # Add validation flag
        data['validated'] = True
        if context and 'timestamp' in context.metadata:
            data['validation_timestamp'] = context.metadata['timestamp']
        else:
            import time
            data['validation_timestamp'] = time.time()

        return data

    def get_transform_description(self) -> str:
        """Get description of this transformation."""
        return "Validates data structure and required fields"


class DataEnricher(ITransformFunction):
    """Enriches data with additional information."""

    def __init__(self, multiplier: int = 2):
        """Initialize with multiplier."""
        self.multiplier = multiplier

    def transform(self, data: Any, context: FunctionContext) -> Any:
        """Add computed fields and metadata."""
        # Access properties resource if available
        if 'properties' in context.resources:
            props = context.resources['properties']
            data['enrichment_source'] = props.get('source', 'unknown')

            # Track processing stats
            count = props.get('processed_count', 0)
            props.set('processed_count', count + 1)

        # Add computed fields
        data['value_squared'] = data['value'] ** 2
        data['value_multiplied'] = data['value'] * self.multiplier
        data['value_category'] = self._categorize_value(data['value'])

        # Add processing metadata
        data['enriched'] = True
        if context and 'timestamp' in context.metadata:
            data['enrichment_timestamp'] = context.metadata['timestamp']
        else:
            import time
            data['enrichment_timestamp'] = time.time()

        return data

    def _categorize_value(self, value: float) -> str:
        """Categorize value into ranges."""
        if value < 0:
            return 'negative'
        elif value < 10:
            return 'low'
        elif value < 100:
            return 'medium'
        else:
            return 'high'

    def get_transform_description(self) -> str:
        """Get description of this transformation."""
        return f"Enriches data with computed fields (multiplier={self.multiplier})"


class DataAggregator(ITransformFunction):
    """Aggregates data in batches."""

    def transform(self, data: Any, context: FunctionContext) -> Any:
        """Aggregate batch of records."""
        # Handle both single and batch modes
        if isinstance(data, list):
            records = data
        else:
            records = [data]

        # Calculate aggregations
        total = sum(r.get('value', 0) for r in records)
        count = len(records)
        avg = total / count if count > 0 else 0

        # Find min/max
        values = [r.get('value', 0) for r in records]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0

        # Create aggregation result
        result = {
            'type': 'aggregation',
            'count': count,
            'total': total,
            'average': avg,
            'min': min_val,
            'max': max_val,
            'records': records
        }

        # Add timestamp
        if context and 'timestamp' in context.metadata:
            result['aggregation_timestamp'] = context.metadata['timestamp']
        else:
            import time
            result['aggregation_timestamp'] = time.time()

        return result

    def get_transform_description(self) -> str:
        """Get description of this transformation."""
        return "Aggregates data into summary statistics"


def create_simple_pipeline_fsm() -> FSM:
    """Create a simple FSM for data pipeline processing."""

    # Create FSM
    fsm = FSM(name="data_pipeline")

    # Register custom functions
    func_manager = FunctionManager()
    func_manager.register_function('validate', DataValidator())
    func_manager.register_function('enrich', DataEnricher(multiplier=3))
    func_manager.register_function('aggregate', DataAggregator())

    # Set function manager
    fsm.function_manager = func_manager

    # Create network
    network = StateNetwork(name="main")

    # Create states
    start_state = State(name="start", type="start")
    process_state = State(name="process", type="normal")
    end_state = State(name="end", type="end")

    # Add states to network
    network.add_state(start_state, initial=True)
    network.add_state(process_state)
    network.add_state(end_state, final=True)

    # Add arcs
    network.add_arc("start", "process")
    network.add_arc("process", "end")

    # Add network to FSM
    fsm.add_network(network)

    return fsm


def create_sample_data(file_path: Path, num_records: int = 20):
    """Create sample data file for testing."""
    import random
    from datetime import datetime, timedelta

    records = []
    base_time = datetime.now()

    for i in range(num_records):
        record = {
            'id': i,
            'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
            'value': random.uniform(-10, 150),
            'source': random.choice(['sensor_a', 'sensor_b', 'sensor_c'])
        }
        records.append(record)

    with open(file_path, 'w') as f:
        json.dump(records, f, indent=2)

    return records


def run_simple_pipeline_example():
    """Run a simplified pipeline example."""
    print("=== Simple Data Pipeline FSM Example ===\n")

    # Create FSM
    print("1. Building FSM...")
    fsm = create_simple_pipeline_fsm()
    print(f"   FSM '{fsm.name}' created\n")

    # Create resources
    print("2. Setting up resources...")
    resource_manager = ResourceManager()

    # Add properties resource for tracking
    props_resource = PropertiesResource(
        name='properties',
        initial_properties={
            'source': 'example_pipeline',
            'processed_count': 0
        }
    )
    resource_manager.register_provider('properties', props_resource)
    fsm.resource_manager = resource_manager
    print("   Resources configured\n")

    # Test with sample data
    print("3. Processing sample data...")
    engine = ExecutionEngine(fsm)

    # Test data
    test_records = [
        {'id': 1, 'timestamp': '2024-01-01', 'value': 25.0},
        {'id': 2, 'timestamp': '2024-01-02', 'value': 50.0},
        {'id': 3, 'timestamp': '2024-01-03', 'value': 75.0}
    ]

    results = []
    for record in test_records:
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)

        # Acquire resources
        props_handle = resource_manager.acquire('properties', f'record_{record["id"]}')
        context.resources = {'properties': props_handle}

        try:
            success, result = engine.execute(context, record)
            if success:
                results.append(result)
                print(f"   Processed record {record['id']}: value={record['value']}")
        except Exception as e:
            print(f"   Error processing record {record['id']}: {e}")
        finally:
            # Release resources
            resource_manager.release('properties', props_handle)

    print(f"\n   Successfully processed {len(results)} records")

    # Show sample result
    if results:
        sample = results[0]
        print(f"\n4. Sample Result:")
        print(f"   Original value: {test_records[0]['value']}")
        if 'validated' in sample:
            print(f"   Validated: {sample.get('validated')}")
        if 'value_category' in sample:
            print(f"   Category: {sample.get('value_category')}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    run_simple_pipeline_example()
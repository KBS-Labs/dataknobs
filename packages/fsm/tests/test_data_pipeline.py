"""Test the fixed data pipeline example."""

import sys
import os
import pytest

# Add examples to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from data_pipeline_example import (
    DataValidator,
    DataEnricher,
    DataAggregator,
    create_simple_pipeline_fsm,
    run_simple_pipeline_example
)

from dataknobs_fsm.functions.base import FunctionContext


class TestDataValidatorFixed:
    """Test the DataValidator function."""

    def test_validate_valid_data(self):
        """Test validation of valid data."""
        validator = DataValidator()
        context = FunctionContext(state_name="test", function_name="validate")
        data = {'id': 1, 'timestamp': '2024-01-01', 'value': 42.0}

        result = validator.transform(data, context)

        assert result['validated'] is True
        assert 'validation_timestamp' in result

    def test_validate_missing_field(self):
        """Test validation with missing required field."""
        validator = DataValidator()
        context = FunctionContext(state_name="test", function_name="validate")
        data = {'id': 1, 'timestamp': '2024-01-01'}  # Missing 'value'

        with pytest.raises(ValueError, match="Missing required field: value"):
            validator.transform(data, context)

    def test_validate_invalid_type(self):
        """Test validation with invalid data type."""
        validator = DataValidator()
        context = FunctionContext(state_name="test", function_name="validate")
        data = {'id': 1, 'timestamp': '2024-01-01', 'value': 'not_a_number'}

        with pytest.raises(ValueError, match="Value must be numeric"):
            validator.transform(data, context)


class TestDataEnricherFixed:
    """Test the DataEnricher function."""

    def test_enrich_basic(self):
        """Test basic data enrichment."""
        enricher = DataEnricher(multiplier=2)
        context = FunctionContext(state_name="test", function_name="validate")
        data = {'id': 1, 'value': 5.0}

        result = enricher.transform(data, context)

        assert result['enriched'] is True
        assert result['value_squared'] == 25.0
        assert result['value_multiplied'] == 10.0
        assert result['value_category'] == 'low'
        assert 'enrichment_timestamp' in result

    def test_categorization(self):
        """Test value categorization."""
        enricher = DataEnricher()
        context = FunctionContext(state_name="test", function_name="validate")

        test_cases = [
            (-5, 'negative'),
            (5, 'low'),
            (50, 'medium'),
            (150, 'high')
        ]

        for value, expected_category in test_cases:
            data = {'value': value}
            result = enricher.transform(data, context)
            assert result['value_category'] == expected_category


class TestDataAggregatorFixed:
    """Test the DataAggregator function."""

    def test_aggregate_single_record(self):
        """Test aggregation of single record."""
        aggregator = DataAggregator()
        context = FunctionContext(state_name="test", function_name="validate")
        data = {'id': 1, 'value': 10.0}

        result = aggregator.transform(data, context)

        assert result['type'] == 'aggregation'
        assert result['count'] == 1
        assert result['total'] == 10.0
        assert result['average'] == 10.0
        assert result['min'] == 10.0
        assert result['max'] == 10.0

    def test_aggregate_multiple_records(self):
        """Test aggregation of multiple records."""
        aggregator = DataAggregator()
        context = FunctionContext(state_name="test", function_name="validate")
        records = [
            {'id': 1, 'value': 10.0},
            {'id': 2, 'value': 20.0},
            {'id': 3, 'value': 30.0}
        ]

        result = aggregator.transform(records, context)

        assert result['count'] == 3
        assert result['total'] == 60.0
        assert result['average'] == 20.0
        assert result['min'] == 10.0
        assert result['max'] == 30.0


class TestPipelineFSMFixed:
    """Test the complete pipeline FSM."""

    def test_create_pipeline_fsm(self):
        """Test creating the pipeline FSM."""
        fsm = create_simple_pipeline_fsm()

        assert fsm.name == 'data_pipeline'
        assert 'main' in fsm.networks

        network = fsm.networks['main']
        assert 'start' in network._states
        assert 'process' in network._states
        assert 'end' in network._states

        # Check function manager
        assert fsm.function_manager is not None

    def test_run_pipeline_example(self):
        """Test running the simplified pipeline example."""
        # This test ensures the example runs without errors
        try:
            run_simple_pipeline_example()
        except Exception as e:
            pytest.fail(f"Pipeline example failed: {e}")

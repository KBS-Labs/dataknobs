"""Tests for file processing function implementations."""

import pytest
from dataknobs_fsm.patterns.file_processing import FileProcessingConfig, FileProcessor


class TestFileProcessingFunctions:
    """Test file processing function code generation and execution."""

    def test_validation_schema_code_generation(self):
        """Test validation schema code generation."""
        # Test empty schema
        config = FileProcessingConfig(input_path="test.json")
        processor = FileProcessor(config)
        assert processor._get_validator_code() == "True"
        
        # Test complex schema
        config = FileProcessingConfig(
            input_path="test.json",
            validation_schema={
                'name': {'required': True, 'type': 'str'},
                'age': {'required': True, 'type': 'int', 'min': 0, 'max': 120},
                'email': {'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
                'active': True  # Simple required field
            }
        )
        processor = FileProcessor(config)
        validator_code = processor._get_validator_code()
        
        # Check that all validations are included
        assert "'name' in data" in validator_code
        assert "isinstance(data.get('name'), str)" in validator_code
        assert "'age' in data" in validator_code
        assert "isinstance(data.get('age'), int)" in validator_code
        assert "data.get('age', 0) >= 0" in validator_code
        assert "data.get('age', 0) <= 120" in validator_code
        assert "re.match(r'^[^@]+@[^@]+\\.[^@]+$'" in validator_code
        assert "'active' in data" in validator_code
        assert " and " in validator_code  # Conditions joined with 'and'

    def test_filter_code_generation(self):
        """Test filter code generation."""
        # Test no filters
        config = FileProcessingConfig(input_path="test.json")
        processor = FileProcessor(config)
        assert processor._get_filter_code() == "True"
        
        # Test with filters
        def filter1(data): return data.get('value', 0) > 10
        def filter2(data): return data.get('status') == 'active'
        
        config = FileProcessingConfig(
            input_path="test.json",
            filters=[filter1, filter2]
        )
        processor = FileProcessor(config)
        filter_code = processor._get_filter_code()
        
        assert filter_code == "filter_0(data) and filter_1(data)"

    def test_transformer_code_generation(self):
        """Test transformer code generation."""
        # Test no transformations
        config = FileProcessingConfig(input_path="test.json")
        processor = FileProcessor(config)
        assert processor._get_transformer_code() == "data"
        
        # Test with transformations
        def transform1(data): return {**data, 'step1': True}
        def transform2(data): return {**data, 'step2': True}
        
        config = FileProcessingConfig(
            input_path="test.json",
            transformations=[transform1, transform2]
        )
        processor = FileProcessor(config)
        transformer_code = processor._get_transformer_code()
        
        assert transformer_code == "transform_1(transform_0(data))"

    def test_aggregator_code_generation(self):
        """Test aggregator code generation."""
        # Test no aggregations
        config = FileProcessingConfig(input_path="test.json")
        processor = FileProcessor(config)
        assert processor._get_aggregator_code() == "data"
        
        # Test with aggregations
        def sum_values(data): return sum(data.get('values', []))
        def count_items(data): return len(data.get('items', []))
        
        config = FileProcessingConfig(
            input_path="test.json",
            aggregations={
                'total': sum_values,
                'count': count_items
            }
        )
        processor = FileProcessor(config)
        aggregator_code = processor._get_aggregator_code()
        
        # Should generate a dictionary with aggregation results
        assert "'total': agg_total(data)" in aggregator_code
        assert "'count': agg_count(data)" in aggregator_code
        assert aggregator_code.startswith("{")
        assert aggregator_code.endswith("}")

    def test_functions_registry_building(self):
        """Test that functions are properly registered."""
        def test_filter(data): return data.get('value', 0) > 10
        def test_transform(data): return {**data, 'processed': True}
        def test_sum(data): return sum(data.get('values', []))
        def test_count(data): return len(data.get('items', []))
        
        config = FileProcessingConfig(
            input_path="test.json",
            filters=[test_filter],
            transformations=[test_transform],
            aggregations={
                'total': test_sum,
                'count': test_count
            }
        )
        processor = FileProcessor(config)
        functions = processor._build_functions()
        
        # Check all functions are registered with correct names
        assert 'filter_0' in functions
        assert 'transform_0' in functions
        assert 'agg_total' in functions
        assert 'agg_count' in functions
        assert 'parser' in functions
        
        # Check functions are callable
        assert callable(functions['filter_0'])
        assert callable(functions['transform_0'])
        assert callable(functions['agg_total'])
        assert callable(functions['agg_count'])
        assert callable(functions['parser'])

    def test_filter_function_execution(self):
        """Test that registered filter functions work correctly."""
        def positive_values(data): 
            return data.get('value', 0) > 0
        
        def active_status(data): 
            return data.get('status') == 'active'
        
        config = FileProcessingConfig(
            input_path="test.json",
            filters=[positive_values, active_status]
        )
        processor = FileProcessor(config)
        functions = processor._build_functions()
        
        # Test individual filters
        test_data1 = {'value': 5, 'status': 'active'}
        assert functions['filter_0'](test_data1) == True  # value > 0
        assert functions['filter_1'](test_data1) == True  # status == 'active'
        
        test_data2 = {'value': -1, 'status': 'inactive'}
        assert functions['filter_0'](test_data2) == False  # value <= 0
        assert functions['filter_1'](test_data2) == False  # status != 'active'

    def test_transformation_function_execution(self):
        """Test that registered transformation functions work correctly."""
        def add_timestamp(data):
            return {**data, 'timestamp': '2024-01-01'}
        
        def add_processed_flag(data):
            return {**data, 'processed': True}
        
        config = FileProcessingConfig(
            input_path="test.json",
            transformations=[add_timestamp, add_processed_flag]
        )
        processor = FileProcessor(config)
        functions = processor._build_functions()
        
        # Test transformation chain
        original_data = {'value': 42}
        
        # Apply first transformation
        step1_result = functions['transform_0'](original_data)
        assert step1_result['value'] == 42
        assert step1_result['timestamp'] == '2024-01-01'
        
        # Apply second transformation
        step2_result = functions['transform_1'](step1_result)
        assert step2_result['value'] == 42
        assert step2_result['timestamp'] == '2024-01-01'
        assert step2_result['processed'] == True

    def test_aggregation_function_execution(self):
        """Test that registered aggregation functions work correctly."""
        def sum_values(data):
            return sum(data.get('values', []))
        
        def count_items(data):
            return len(data.get('items', []))
        
        def average_score(data):
            scores = data.get('scores', [])
            return sum(scores) / len(scores) if scores else 0
        
        config = FileProcessingConfig(
            input_path="test.json",
            aggregations={
                'total': sum_values,
                'count': count_items,
                'avg_score': average_score
            }
        )
        processor = FileProcessor(config)
        functions = processor._build_functions()
        
        # Test aggregation functions
        test_data = {
            'values': [1, 2, 3, 4, 5],
            'items': ['a', 'b', 'c'],
            'scores': [85, 90, 78, 92]
        }
        
        assert functions['agg_total'](test_data) == 15  # sum of values
        assert functions['agg_count'](test_data) == 3   # count of items
        assert functions['agg_avg_score'](test_data) == 86.25  # average score

    def test_complex_validation_schema(self):
        """Test complex validation scenarios."""
        config = FileProcessingConfig(
            input_path="test.json",
            validation_schema={
                'user_id': {'required': True, 'type': 'int', 'min': 1},
                'username': {'required': True, 'type': 'str'},
                'email': {'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
                'age': {'type': 'int', 'min': 13, 'max': 99},
                'is_premium': True  # Simple required boolean
            }
        )
        processor = FileProcessor(config)
        validator_code = processor._get_validator_code()
        
        # Verify all constraints are included
        expected_parts = [
            "'user_id' in data",
            "isinstance(data.get('user_id'), int)",
            "data.get('user_id', 0) >= 1",
            "'username' in data", 
            "isinstance(data.get('username'), str)",
            "re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'",
            "isinstance(data.get('age'), int)",
            "data.get('age', 0) >= 13",
            "data.get('age', 0) <= 99",
            "'is_premium' in data"
        ]
        
        for part in expected_parts:
            assert part in validator_code

    def test_empty_configurations(self):
        """Test that empty configurations generate appropriate default code."""
        config = FileProcessingConfig(input_path="test.json")
        processor = FileProcessor(config)
        
        # All should return safe defaults
        assert processor._get_validator_code() == "True"
        assert processor._get_filter_code() == "True"
        assert processor._get_transformer_code() == "data"
        assert processor._get_aggregator_code() == "data"
        
        # Functions registry should only contain parser
        functions = processor._build_functions()
        assert len(functions) == 1
        assert 'parser' in functions

    def test_multiple_filters_and_transformations(self):
        """Test scenarios with multiple filters and transformations."""
        def filter_positive(data): return data.get('value', 0) > 0
        def filter_even(data): return data.get('value', 0) % 2 == 0
        def filter_small(data): return data.get('value', 0) < 100
        
        def transform_double(data): return {**data, 'value': data.get('value', 0) * 2}
        def transform_add_flag(data): return {**data, 'doubled': True}
        def transform_add_category(data): 
            value = data.get('value', 0)
            category = 'large' if value > 50 else 'small'
            return {**data, 'category': category}
        
        config = FileProcessingConfig(
            input_path="test.json",
            filters=[filter_positive, filter_even, filter_small],
            transformations=[transform_double, transform_add_flag, transform_add_category]
        )
        processor = FileProcessor(config)
        
        # Test filter code generation
        filter_code = processor._get_filter_code()
        expected_filter = "filter_0(data) and filter_1(data) and filter_2(data)"
        assert filter_code == expected_filter
        
        # Test transformer code generation  
        transformer_code = processor._get_transformer_code()
        expected_transformer = "transform_2(transform_1(transform_0(data)))"
        assert transformer_code == expected_transformer
        
        # Test functions are registered
        functions = processor._build_functions()
        assert all(f'filter_{i}' in functions for i in range(3))
        assert all(f'transform_{i}' in functions for i in range(3))

    def test_fsm_configuration_integration(self):
        """Test that the complete FSM configuration is built correctly."""
        def simple_filter(data): return data.get('valid', True)
        def simple_transform(data): return {**data, 'processed': True}
        def simple_aggregator(data): return len(data.get('items', []))
        
        config = FileProcessingConfig(
            input_path="test.json",
            validation_schema={'name': True},
            filters=[simple_filter],
            transformations=[simple_transform],
            aggregations={'count': simple_aggregator}
        )
        
        processor = FileProcessor(config)
        
        # Test that FSM can be built without errors
        fsm = processor._fsm
        assert fsm is not None
        
        # Test that functions section exists in config
        functions = processor._build_functions()
        assert len(functions) == 4  # filter_0, transform_0, agg_count, parser
        assert all(key in functions for key in ['filter_0', 'transform_0', 'agg_count', 'parser'])
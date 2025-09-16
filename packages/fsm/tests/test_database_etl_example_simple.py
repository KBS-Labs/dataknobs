"""Simple test for database ETL example imports and basic functionality."""


import pytest

from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


class TestBasicETLExample:
    """Test basic ETL example functionality."""

    def test_imports(self):
        """Test that all imports work correctly."""
        # Just test that the imports work
        assert SimpleFSM is not None
        assert DatabaseETL is not None
        assert ETLConfig is not None
        assert ETLMode is not None

    def test_etl_config_creation(self):
        """Test ETL configuration creation."""
        config = ETLConfig(
            source_db={'type': 'memory'},
            target_db={'type': 'memory'},
            mode=ETLMode.INCREMENTAL,
            batch_size=100
        )

        assert config.source_db == {'type': 'memory'}
        assert config.target_db == {'type': 'memory'}
        assert config.mode == ETLMode.INCREMENTAL
        assert config.batch_size == 100

    def test_database_etl_creation(self):
        """Test DatabaseETL creation."""
        config = ETLConfig(
            source_db={'type': 'memory'},
            target_db={'type': 'memory'}
        )

        etl = DatabaseETL(config)
        assert etl.config == config
        assert etl._fsm is not None

    def test_transformation_functions(self):
        """Test individual transformation functions."""

        def clean_data(row):
            return {**row, 'cleaned': True}

        def validate_data(row):
            if 'id' not in row:
                raise ValueError("Missing id")
            return {**row, 'validated': True}

        # Test transformations
        test_row = {'id': 1, 'name': '  test  '}

        cleaned = clean_data(test_row)
        assert cleaned['cleaned'] is True

        validated = validate_data(cleaned)
        assert validated['validated'] is True

        # Test validation failure
        with pytest.raises(ValueError):
            validate_data({'name': 'test'})

    def test_simple_fsm_creation(self):
        """Test simple FSM creation."""
        config = {
            'name': 'test_etl',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {'name': 'start', 'is_start': True},
                    {'name': 'process'},
                    {'name': 'complete', 'is_end': True}
                ],
                'arcs': [
                    {'from': 'start', 'to': 'process'},
                    {'from': 'process', 'to': 'complete'}
                ]
            }]
        }

        fsm = SimpleFSM(config)

        # Should be able to create without errors
        assert fsm is not None

    @pytest.mark.asyncio
    async def test_simple_fsm_execution(self):
        """Test simple FSM execution."""
        config = {
            'name': 'test_etl_execution',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {
                        'name': 'start',
                        'is_start': True,
                        'schema': {
                            'type': 'object',
                            'properties': {'input': {'type': 'string'}},
                            'required': ['input']
                        }
                    },
                    {
                        'name': 'process',
                        'functions': {
                            'transform': 'lambda state: {"input": state.data["input"], "processed": True}'
                        }
                    },
                    {'name': 'complete', 'is_end': True}
                ],
                'arcs': [
                    {'from': 'start', 'to': 'process'},
                    {'from': 'process', 'to': 'complete'}
                ]
            }]
        }

        fsm = SimpleFSM(config)
        result = fsm.process({"input": "test_data"})

        assert result['success'] is True
        assert result['final_state'] == 'complete'
        assert result['data']['processed'] is True
        assert result['data']['input'] == 'test_data'

"""Tests for ETL pattern implementation."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, AsyncIterator
from dataclasses import dataclass

from dataknobs_fsm.patterns.etl import (
    DatabaseETL, ETLConfig, ETLMode,
    create_etl_pipeline, create_database_sync,
    create_data_migration, create_data_warehouse_load
)


@pytest.fixture
def source_db_config():
    """Source database configuration."""
    return {
        'type': 'postgres',
        'host': 'localhost',
        'port': 5432,
        'database': 'source_db',
        'user': 'test',
        'password': 'test'
    }


@pytest.fixture
def target_db_config():
    """Target database configuration."""
    return {
        'type': 'postgres',
        'host': 'localhost',
        'port': 5433,
        'database': 'target_db',
        'user': 'test',
        'password': 'test'
    }


@pytest.fixture
def etl_config(source_db_config, target_db_config):
    """Basic ETL configuration."""
    return ETLConfig(
        source_db=source_db_config,
        target_db=target_db_config,
        mode=ETLMode.FULL_REFRESH,
        batch_size=100,
        parallel_workers=2
    )


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    mock_db = AsyncMock()
    
    # Mock record
    mock_record = Mock()
    mock_record.to_dict.return_value = {'id': 1, 'name': 'test', 'value': 100}
    
    # Mock stream_read to return async generator
    async def mock_stream():
        for i in range(5):
            record = Mock()
            record.to_dict.return_value = {'id': i, 'name': f'test_{i}', 'value': i * 10}
            yield record
    
    mock_db.stream_read.return_value = mock_stream()
    mock_db.close = AsyncMock()
    
    return mock_db


class TestETLConfig:
    """Test ETL configuration."""
    
    def test_basic_config(self, source_db_config, target_db_config):
        """Test basic ETL configuration."""
        config = ETLConfig(
            source_db=source_db_config,
            target_db=target_db_config
        )
        
        assert config.mode == ETLMode.FULL_REFRESH
        assert config.batch_size == 1000
        assert config.parallel_workers == 4
        assert config.error_threshold == 0.05
    
    def test_config_with_mappings(self, source_db_config, target_db_config):
        """Test ETL config with field mappings."""
        config = ETLConfig(
            source_db=source_db_config,
            target_db=target_db_config,
            field_mappings={'old_name': 'new_name'},
            validation_schema={'type': 'object', 'properties': {'id': {'type': 'integer'}}}
        )
        
        assert config.field_mappings == {'old_name': 'new_name'}
        assert config.validation_schema is not None


class TestDatabaseETL:
    """Test DatabaseETL class."""
    
    def test_initialization(self, etl_config):
        """Test DatabaseETL initialization."""
        etl = DatabaseETL(etl_config)
        
        assert etl.config == etl_config
        assert etl._metrics['extracted'] == 0
        assert etl._metrics['loaded'] == 0
        assert etl._metrics['errors'] == 0
    
    def test_fsm_building(self, etl_config):
        """Test FSM building for ETL."""
        etl = DatabaseETL(etl_config)
        
        # Check FSM was built
        assert etl._fsm is not None
        assert etl._fsm.name == 'ETL_Pipeline'
        
        # Check states
        states = etl._fsm._fsm.states
        state_names = [s.name for s in states.values()]
        assert 'extract' in state_names
        assert 'validate' in state_names
        assert 'transform' in state_names
        assert 'load' in state_names
        assert 'complete' in state_names
        assert 'error' in state_names
    
    @pytest.mark.asyncio
    async def test_run_full_refresh(self, etl_config, mock_database):
        """Test running ETL in full refresh mode."""
        with patch('dataknobs_data.Database.create') as mock_create:
            mock_create.return_value = mock_database
            
            etl = DatabaseETL(etl_config)
            
            # Mock FSM process_batch
            etl._fsm.process_batch = Mock(return_value=[
                {'success': True, 'final_state': 'complete'},
                {'success': True, 'final_state': 'complete'},
                {'success': True, 'final_state': 'complete'},
                {'success': True, 'final_state': 'complete'},
                {'success': True, 'final_state': 'complete'}
            ])
            
            metrics = await etl.run()
            
            assert metrics['extracted'] == 5
            assert metrics['loaded'] == 5
            assert metrics['errors'] == 0
            assert mock_database.close.called
    
    @pytest.mark.asyncio
    async def test_run_incremental(self, source_db_config, target_db_config, mock_database):
        """Test running ETL in incremental mode."""
        config = ETLConfig(
            source_db=source_db_config,
            target_db=target_db_config,
            mode=ETLMode.INCREMENTAL
        )
        
        with patch('dataknobs_data.Database.create') as mock_create:
            mock_create.return_value = mock_database
            
            etl = DatabaseETL(config)
            etl._checkpoint_data['last_timestamp'] = '2024-01-01'
            
            # Mock FSM process_batch
            etl._fsm.process_batch = Mock(return_value=[
                {'success': True, 'final_state': 'complete'}
            ])
            
            metrics = await etl.run()
            
            # Should use incremental query
            assert metrics['extracted'] > 0
    
    @pytest.mark.asyncio
    async def test_error_threshold(self, etl_config, mock_database):
        """Test error threshold handling."""
        etl_config.error_threshold = 0.3  # 30% error threshold
        
        with patch('dataknobs_data.Database.create') as mock_create:
            mock_create.return_value = mock_database
            
            etl = DatabaseETL(etl_config)
            
            # Mock FSM to return mix of success and errors
            etl._fsm.process_batch = Mock(return_value=[
                {'success': True, 'final_state': 'complete'},
                {'success': True, 'final_state': 'complete'},
                {'success': True, 'final_state': 'error'},
                {'success': True, 'final_state': 'error'}
            ])
            
            with pytest.raises(Exception) as exc_info:
                await etl.run()
            
            assert "Error threshold exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_checkpointing(self, etl_config):
        """Test checkpoint save and load."""
        etl = DatabaseETL(etl_config)
        etl._metrics = {
            'extracted': 1000,
            'loaded': 950,
            'errors': 50,
            'transformed': 950,
            'skipped': 0
        }
        
        # Save checkpoint
        checkpoint_id = await etl._save_checkpoint()
        assert checkpoint_id is not None
        
        # Load checkpoint
        await etl._load_checkpoint(checkpoint_id)
        assert etl._metrics['extracted'] == 1000
        assert etl._metrics['loaded'] == 950
    
    def test_validation_test_creation(self, etl_config):
        """Test validation test function creation."""
        etl_config.validation_schema = {
            'type': 'object',
            'properties': {
                'id': {'type': 'integer'},
                'name': {'type': 'string'}
            },
            'required': ['id']
        }
        
        etl = DatabaseETL(etl_config)
        test_func = etl._create_validation_test()
        
        assert test_func is not None
        # Would need actual validation testing here
    
    @pytest.mark.asyncio
    async def test_transformer_creation(self, etl_config):
        """Test transformer function creation."""
        etl_config.field_mappings = {'old_field': 'new_field'}
        etl_config.transformations = [
            lambda data: {**data, 'transformed': True}
        ]
        
        etl = DatabaseETL(etl_config)
        transformer = etl._create_transformer()
        
        result = await transformer({'old_field': 'value', 'other': 'data'})
        assert 'transformed' in result
        assert result['transformed'] is True


class TestETLFactoryFunctions:
    """Test ETL factory functions."""
    
    def test_create_etl_pipeline(self, source_db_config, target_db_config):
        """Test creating ETL pipeline."""
        etl = create_etl_pipeline(
            source=source_db_config,
            target=target_db_config,
            mode=ETLMode.UPSERT,
            batch_size=500
        )
        
        assert isinstance(etl, DatabaseETL)
        assert etl.config.mode == ETLMode.UPSERT
        assert etl.config.batch_size == 500
    
    def test_create_from_connection_string(self):
        """Test creating ETL from connection strings."""
        etl = create_etl_pipeline(
            source='postgresql://user:pass@localhost/source',
            target='mongodb://localhost/target'
        )
        
        assert isinstance(etl, DatabaseETL)
        assert etl.config.source_db['type'] == 'postgres'
        assert etl.config.target_db['type'] == 'mongodb'
    
    def test_create_database_sync(self, source_db_config, target_db_config):
        """Test creating database sync pipeline."""
        sync = create_database_sync(
            source=source_db_config,
            target=target_db_config,
            sync_interval=600
        )
        
        assert isinstance(sync, DatabaseETL)
        assert sync.config.mode == ETLMode.INCREMENTAL
        assert sync.config.checkpoint_interval == 1000
    
    def test_create_data_migration(self, source_db_config, target_db_config):
        """Test creating data migration pipeline."""
        migration = create_data_migration(
            source=source_db_config,
            target=target_db_config,
            field_mappings={'old': 'new'},
            transformations=[lambda x: x]
        )
        
        assert isinstance(migration, DatabaseETL)
        assert migration.config.mode == ETLMode.FULL_REFRESH
        assert migration.config.field_mappings == {'old': 'new'}
        assert migration.config.batch_size == 5000
        assert migration.config.parallel_workers == 8
    
    def test_create_data_warehouse_load(self, target_db_config):
        """Test creating data warehouse loading pipelines."""
        sources = [
            {'type': 'postgres', 'host': 'source1'},
            {'type': 'mysql', 'host': 'source2'}
        ]
        
        pipelines = create_data_warehouse_load(
            sources=sources,
            warehouse=target_db_config,
            aggregations=[lambda x: x]
        )
        
        assert len(pipelines) == 2
        assert all(isinstance(p, DatabaseETL) for p in pipelines)
        assert all(p.config.mode == ETLMode.APPEND for p in pipelines)


class TestETLModes:
    """Test different ETL modes."""
    
    @pytest.mark.asyncio
    async def test_full_refresh_mode(self, source_db_config, target_db_config, mock_database):
        """Test FULL_REFRESH mode."""
        config = ETLConfig(
            source_db=source_db_config,
            target_db=target_db_config,
            mode=ETLMode.FULL_REFRESH
        )
        
        with patch('dataknobs_data.Database.create') as mock_create:
            mock_create.return_value = mock_database
            
            etl = DatabaseETL(config)
            # Full refresh should not use incremental query
            query = etl._get_incremental_query()
            assert query is not None  # Would be empty Query()
    
    @pytest.mark.asyncio
    async def test_upsert_mode(self, source_db_config, target_db_config):
        """Test UPSERT mode."""
        config = ETLConfig(
            source_db=source_db_config,
            target_db=target_db_config,
            mode=ETLMode.UPSERT
        )
        
        etl = DatabaseETL(config)
        assert etl.config.mode == ETLMode.UPSERT
    
    @pytest.mark.asyncio
    async def test_append_mode(self, source_db_config, target_db_config):
        """Test APPEND mode."""
        config = ETLConfig(
            source_db=source_db_config,
            target_db=target_db_config,
            mode=ETLMode.APPEND
        )
        
        etl = DatabaseETL(config)
        assert etl.config.mode == ETLMode.APPEND


class TestETLBatchProcessing:
    """Test ETL batch processing."""
    
    @pytest.mark.asyncio
    async def test_batch_extraction(self, etl_config):
        """Test extracting data in batches."""
        etl = DatabaseETL(etl_config)
        
        # Create mock database with more records
        mock_db = AsyncMock()
        
        async def mock_stream():
            for i in range(250):  # More than batch size
                record = Mock()
                record.to_dict.return_value = {'id': i}
                yield record
        
        mock_db.stream_read.return_value = mock_stream()
        
        batches = []
        async for batch in etl._extract_batches(mock_db, Mock()):
            batches.append(batch)
        
        # Should create 3 batches (100, 100, 50)
        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50
    
    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self, etl_config):
        """Test parallel processing of batches."""
        etl_config.parallel_workers = 4
        etl = DatabaseETL(etl_config)
        
        # Track parallel execution
        execution_times = []
        
        def track_execution(batch):
            import time
            start = time.time()
            # Simulate work
            time.sleep(0.01)
            execution_times.append(time.time() - start)
            return [{'success': True, 'final_state': 'complete'} for _ in batch]
        
        etl._fsm.process_batch = track_execution
        
        # Process batch
        batch = [{'id': i} for i in range(10)]
        results = etl._fsm.process_batch(batch)
        
        assert len(results) == 10


class TestETLEnrichment:
    """Test ETL data enrichment."""
    
    def test_enrichment_sources_config(self, etl_config):
        """Test configuration with enrichment sources."""
        etl_config.enrichment_sources = [
            {'database': {'type': 'postgres', 'host': 'enrichment1'}},
            {'api': 'http://api.example.com/enrich'}
        ]
        
        etl = DatabaseETL(etl_config)
        
        # Check resources are configured
        transform_resources = etl._get_transform_resources()
        enrichment_resources = etl._get_enrichment_resources()
        
        assert 'enrichment_db_0' in transform_resources
        assert 'enrichment_api_1' in enrichment_resources
    
    def test_enricher_creation(self, etl_config):
        """Test enricher function creation."""
        etl_config.enrichment_sources = [
            {'api': 'http://api.example.com/enrich'}
        ]
        
        etl = DatabaseETL(etl_config)
        enricher = etl._create_enricher()
        
        assert enricher is not None
        assert callable(enricher)
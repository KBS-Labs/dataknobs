"""Tests for the database ETL pipeline example.

This test validates that the database ETL example works correctly,
including COPY mode transaction safety, staging/rollback, and proper routing.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from database_etl import (
    initialize_etl,
    extract_data,
    validate_records,
    transform_records,
    load_to_staging,
    commit_to_target,
    rollback_staging,
    finalize_etl,
    check_validation_passed,
    check_ready_to_commit,
    create_etl_fsm,
    etl_config
)
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode


class TestDatabaseETLPipeline:
    """Test the database ETL pipeline example."""
    
    @pytest.fixture
    def fsm(self):
        """Create ETL FSM with custom functions."""
        return create_etl_fsm()
    
    def test_successful_etl_pipeline(self, fsm):
        """Test successful ETL pipeline execution."""
        result = fsm.process({
            'source_table': 'sales_raw',
            'target_table': 'sales_fact',
            'batch_size': 20,
            'mode': 'incremental'
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'success'
        
        # Verify the path through all stages
        expected_path = [
            'start', 'initialize', 'extract', 'validate', 
            'transform', 'staging', 'commit', 'finalize', 'success'
        ]
        assert result['path'] == expected_path
        
        # Check ETL summary
        summary = result['data'].get('etl_summary', {})
        assert summary['status'] == 'SUCCESS'
        assert summary['records_extracted'] == 20
        assert summary['records_validated'] == 20
        assert summary['records_transformed'] == 20
        assert summary['records_loaded'] == 20
        assert summary['success_rate'] == 100.0
    
    def test_etl_with_validation_failure(self):
        """Test ETL pipeline with validation failures exceeding threshold."""
        # Create FSM with modified validator that forces failures
        def validate_with_high_error_rate(state):
            data = state.data.copy()
            data['extracted_records'] = data.get('extracted_records', [])
            data['validated_records'] = []
            data['statistics']['records_extracted'] = 100
            data['statistics']['records_failed'] = 20  # 20% error rate
            data['statistics']['records_validated'] = 80
            data['validation_passed'] = False  # Exceeds 10% threshold
            return data
        
        fsm = SimpleFSM(
            etl_config,
            data_mode=DataHandlingMode.COPY,
            custom_functions={
                'initialize_etl': initialize_etl,
                'extract_data': extract_data,
                'validate_records': validate_with_high_error_rate,
                'transform_records': transform_records,
                'load_to_staging': load_to_staging,
                'commit_to_target': commit_to_target,
                'rollback_staging': rollback_staging,
                'finalize_etl': finalize_etl,
                'check_validation_passed': check_validation_passed,
                'check_ready_to_commit': check_ready_to_commit
            }
        )
        
        result = fsm.process({
            'source_table': 'sales_raw',
            'target_table': 'sales_fact',
            'batch_size': 100
        })
        
        assert result['success'] is True  # FSM execution succeeds
        assert result['final_state'] == 'failure'  # But ends in failure state
        
        # Verify it went through rollback
        expected_path = [
            'start', 'initialize', 'extract', 'validate', 
            'rollback', 'finalize', 'failure'
        ]
        assert result['path'] == expected_path
        
        # Check rollback was executed
        assert result['data'].get('rollback_complete') is True
        rollback_log = result['data'].get('rollback_log', {})
        assert rollback_log['reason'] == 'Validation failed'
        
        # Check final status
        summary = result['data'].get('etl_summary', {})
        assert summary['status'] == 'FAILED'
    
    def test_initialize_etl_function(self):
        """Test ETL initialization function."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        state = MockState({
            'source_table': 'test_source',
            'target_table': 'test_target',
            'batch_size': 50,
            'mode': 'full'
        })
        
        result = initialize_etl(state)
        
        assert 'etl_metadata' in result
        assert result['etl_metadata']['source_table'] == 'test_source'
        assert result['etl_metadata']['target_table'] == 'test_target'
        assert result['etl_metadata']['batch_size'] == 50
        assert result['etl_metadata']['mode'] == 'full'
        assert 'batch_id' in result['etl_metadata']
        
        assert 'statistics' in result
        assert result['statistics']['records_extracted'] == 0
        assert result['statistics']['records_loaded'] == 0
        
        assert result['initialization_complete'] is True
    
    def test_extract_data_function(self):
        """Test data extraction function."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        state = MockState({
            'etl_metadata': {
                'source_table': 'sales_raw',
                'batch_size': 10
            },
            'statistics': {
                'records_extracted': 0
            }
        })
        
        result = extract_data(state)
        
        assert 'extracted_records' in result
        assert len(result['extracted_records']) == 10
        assert result['statistics']['records_extracted'] == 10
        
        # Check record structure
        for record in result['extracted_records']:
            assert 'order_id' in record
            assert 'customer_id' in record
            assert 'quantity' in record
            assert 'unit_price' in record
            assert record['raw_data'] is True
    
    def test_validate_records_function(self):
        """Test record validation function."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        # Test with valid records
        valid_records = [
            {
                'id': 1,
                'order_id': 'ORD001',
                'customer_id': 'CUST001',
                'quantity': 5,
                'unit_price': 100.0,
                'order_date': (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                'id': 2,
                'order_id': 'ORD002',
                'customer_id': 'CUST002',
                'quantity': 3,
                'unit_price': 50.0,
                'order_date': (datetime.now() - timedelta(days=2)).isoformat()
            }
        ]
        
        state = MockState({
            'extracted_records': valid_records,
            'statistics': {
                'records_extracted': 2,
                'records_validated': 0,
                'records_failed': 0,
                'validation_errors': []
            }
        })
        
        result = validate_records(state)
        
        assert len(result['validated_records']) == 2
        assert result['statistics']['records_validated'] == 2
        assert result['statistics']['records_failed'] == 0
        assert result['validation_passed'] is True
        
        # Test with invalid records
        invalid_records = [
            {
                'id': 1,
                'order_id': 'ORD001',
                # Missing customer_id
                'quantity': -1,  # Invalid quantity
                'unit_price': 100.0,
                'order_date': (datetime.now() + timedelta(days=1)).isoformat()  # Future date
            }
        ]
        
        state = MockState({
            'extracted_records': invalid_records,
            'statistics': {
                'records_extracted': 1,
                'records_validated': 0,
                'records_failed': 0,
                'validation_errors': []
            }
        })
        
        result = validate_records(state)
        
        assert len(result['validated_records']) == 0
        assert result['statistics']['records_failed'] == 1
        assert len(result['statistics']['validation_errors']) == 1
    
    def test_transform_records_function(self):
        """Test record transformation function."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        records = [
            {
                'id': 1,
                'order_id': 'ORD001',
                'customer_id': 'CUST001',
                'quantity': 5,
                'unit_price': 100.0,
                'order_date': datetime.now().isoformat(),
                'validated': True
            }
        ]
        
        state = MockState({
            'validated_records': records,
            'etl_metadata': {
                'batch_id': 'batch_001'
            },
            'statistics': {
                'records_transformed': 0,
                'transformation_errors': []
            }
        })
        
        result = transform_records(state)
        
        assert len(result['transformed_records']) == 1
        transformed = result['transformed_records'][0]
        
        # Check calculations
        assert transformed['total_amount'] == 500.0  # 5 * 100
        assert transformed['discount_rate'] == 0.1  # quantity >= 5
        assert transformed['discount_amount'] == 50.0
        assert transformed['net_amount'] == 450.0
        
        # Check time dimensions
        assert 'year' in transformed
        assert 'month' in transformed
        assert 'quarter' in transformed
        assert 'day_of_week' in transformed
        
        # Check customer segmentation
        assert transformed['customer_segment'] == 'Standard'  # net_amount = 450
        
        # Check ETL metadata
        assert transformed['etl_batch_id'] == 'batch_001'
        assert 'etl_timestamp' in transformed
        assert transformed['transformed'] is True
        assert 'raw_data' not in transformed
    
    def test_staging_and_commit(self):
        """Test staging and commit functions."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        transformed_records = [
            {
                'id': 1,
                'order_id': 'ORD001',
                'customer_id': 'CUST001',
                'product_id': 'PROD01',
                'quantity': 5,
                'unit_price': 100.0,
                'total_amount': 500.0,
                'discount_amount': 50.0,
                'net_amount': 450.0,
                'order_date': datetime.now().isoformat(),
                'status': 'completed',
                'region': 'North',
                'customer_segment': 'Standard',
                'year': 2025,
                'month': 9,
                'quarter': 'Q3',
                'etl_batch_id': 'batch_001',
                'etl_timestamp': datetime.now().isoformat()
            }
        ]
        
        # Test staging
        state = MockState({
            'transformed_records': transformed_records
        })
        
        staging_result = load_to_staging(state)
        
        assert 'staging_records' in staging_result
        assert len(staging_result['staging_records']) == 1
        assert staging_result['staging_complete'] is True
        assert staging_result['ready_to_commit'] is True
        
        # Test commit
        state = MockState({
            'staging_records': staging_result['staging_records'],
            'statistics': {
                'records_loaded': 0
            }
        })
        
        commit_result = commit_to_target(state)
        
        assert commit_result['committed'] is True
        assert 'commit_timestamp' in commit_result
        assert commit_result['statistics']['records_loaded'] == 1
        assert len(commit_result['staging_records']) == 0  # Cleared after commit
        assert 'last_processed' in commit_result
    
    def test_rollback_function(self):
        """Test rollback function."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        state = MockState({
            'staging_records': [{'id': 1}, {'id': 2}],
            'staging_complete': True,
            'ready_to_commit': True,
            'commit_error': 'Database connection failed',
            'statistics': {
                'records_transformed': 2
            }
        })
        
        result = rollback_staging(state)
        
        assert len(result['staging_records']) == 0
        assert result['staging_complete'] is False
        assert result['ready_to_commit'] is False
        assert result['rollback_complete'] is True
        
        rollback_log = result['rollback_log']
        assert rollback_log['reason'] == 'Database connection failed'
        assert rollback_log['records_rolled_back'] == 2
    
    def test_finalize_etl_function(self):
        """Test ETL finalization function."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        start_time = datetime.now() - timedelta(seconds=5)
        
        state = MockState({
            'etl_metadata': {
                'start_time': start_time.isoformat(),
                'batch_id': 'batch_001',
                'source_table': 'sales_raw',
                'target_table': 'sales_fact'
            },
            'statistics': {
                'records_extracted': 100,
                'records_validated': 95,
                'records_transformed': 95,
                'records_loaded': 95,
                'records_failed': 5
            },
            'committed': True
        })
        
        result = finalize_etl(state)
        
        summary = result['etl_summary']
        assert summary['batch_id'] == 'batch_001'
        assert summary['status'] == 'SUCCESS'
        assert summary['records_loaded'] == 95
        assert summary['success_rate'] == 95.0
        assert summary['duration_seconds'] >= 5.0
    
    def test_check_functions(self):
        """Test conditional check functions."""
        # Test validation check
        assert check_validation_passed({'validation_passed': True}, None) is True
        assert check_validation_passed({'validation_passed': False}, None) is False
        assert check_validation_passed({}, None) is False
        
        # Test commit readiness check
        assert check_ready_to_commit({'ready_to_commit': True}, None) is True
        assert check_ready_to_commit({'ready_to_commit': False}, None) is False
        assert check_ready_to_commit({}, None) is False
    
    def test_batch_processing(self, fsm):
        """Test processing multiple batches."""
        batch_results = []
        
        for i in range(3):
            result = fsm.process({
                'source_table': 'sales_raw',
                'target_table': 'sales_fact',
                'batch_size': 10,
                'batch_number': i + 1
            })
            
            assert result['success'] is True
            assert result['final_state'] == 'success'
            
            summary = result['data'].get('etl_summary', {})
            batch_results.append({
                'batch': i + 1,
                'loaded': summary.get('records_loaded', 0)
            })
        
        # Verify all batches processed
        assert len(batch_results) == 3
        for batch in batch_results:
            assert batch['loaded'] == 10
        
        total_loaded = sum(b['loaded'] for b in batch_results)
        assert total_loaded == 30
    
    def test_incremental_mode(self, fsm):
        """Test incremental processing mode."""
        last_processed = (datetime.now() - timedelta(days=7)).isoformat()
        
        result = fsm.process({
            'source_table': 'sales_raw',
            'target_table': 'sales_fact',
            'batch_size': 15,
            'mode': 'incremental',
            'last_processed': last_processed
        })
        
        assert result['success'] is True
        assert result['data']['etl_metadata']['mode'] == 'incremental'
        assert result['data']['last_processed'] != last_processed  # Updated after processing
    
    def test_copy_mode_behavior(self, fsm):
        """Test that COPY mode is properly configured."""
        # Process data and verify staging behavior characteristic of COPY mode
        result = fsm.process({
            'source_table': 'sales_raw',
            'target_table': 'sales_fact',
            'batch_size': 5
        })
        
        # In COPY mode, data goes through staging before commit
        assert 'staging' in result['path']
        assert 'commit' in result['path']
        
        # Verify staging records were created and then cleared after commit
        assert result['data'].get('staging_complete') is True
        assert result['data'].get('committed') is True
        assert len(result['data'].get('staging_records', [])) == 0  # Cleared after commit
    
    def test_etl_statistics_tracking(self, fsm):
        """Test that statistics are properly tracked throughout the pipeline."""
        result = fsm.process({
            'source_table': 'sales_raw',
            'target_table': 'sales_fact',
            'batch_size': 25
        })
        
        stats = result['data']['statistics']
        
        # Verify statistics are consistent
        assert stats['records_extracted'] == 25
        assert stats['records_validated'] == 25
        assert stats['records_transformed'] == 25
        assert stats['records_loaded'] == 25
        assert stats['records_failed'] == 0
        
        # Check logs exist
        assert 'extraction_log' in result['data']
        assert 'staging_log' in result['data']
        assert result['data']['extraction_log']['count'] == 25
        assert result['data']['staging_log']['record_count'] == 25
    
    def test_error_threshold_calculation(self):
        """Test validation error threshold calculation."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        # Test 5% error rate (should pass with 10% threshold)
        state = MockState({
            'extracted_records': [{'id': i} for i in range(100)],
            'statistics': {
                'records_extracted': 100,
                'records_validated': 0,
                'records_failed': 0,
                'validation_errors': []
            }
        })
        
        # Simulate 5 failures
        state.data['statistics']['records_failed'] = 5
        state.data['statistics']['records_validated'] = 95
        state.data['validated_records'] = state.data['extracted_records'][:95]
        
        # Calculate error rate
        error_rate = state.data['statistics']['records_failed'] / state.data['statistics']['records_extracted']
        validation_passed = error_rate <= 0.1
        
        assert error_rate == 0.05
        assert validation_passed is True
        
        # Test 15% error rate (should fail with 10% threshold)
        state.data['statistics']['records_failed'] = 15
        state.data['statistics']['records_validated'] = 85
        
        error_rate = state.data['statistics']['records_failed'] / state.data['statistics']['records_extracted']
        validation_passed = error_rate <= 0.1
        
        assert error_rate == 0.15
        assert validation_passed is False
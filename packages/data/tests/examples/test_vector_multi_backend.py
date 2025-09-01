"""Tests for the vector multi-backend example."""

import pytest
import asyncio
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any

import numpy as np

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from vector_multi_backend import MultiBackendVectorExample
from dataknobs_data import Record, VectorField, Field
from dataknobs_data.query import Query, Operator


@pytest.fixture
def vector_example():
    """Create a MultiBackendVectorExample instance."""
    return MultiBackendVectorExample(verbose=False)


@pytest.fixture
def vector_example_verbose():
    """Create a verbose MultiBackendVectorExample instance for testing output."""
    return MultiBackendVectorExample(verbose=True)


class TestMultiBackendVectorExample:
    """Test cases for MultiBackendVectorExample class."""
    
    def test_initialization(self, vector_example):
        """Test MultiBackendVectorExample initialization."""
        assert vector_example.verbose is False
        assert len(vector_example.test_data) == 6
        
        # Check test data structure
        for item in vector_example.test_data:
            assert 'name' in item
            assert 'description' in item
            assert 'vector' in item
            assert 'category' in item
            assert isinstance(item['vector'], np.ndarray)
            assert item['vector'].shape == (3,)
    
    def test_log_verbose(self, vector_example_verbose, capsys):
        """Test logging in verbose mode."""
        vector_example_verbose.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
    
    def test_log_silent(self, vector_example, capsys):
        """Test logging in silent mode."""
        vector_example.log("Test message")
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_create_records(self, vector_example):
        """Test record creation with vectors."""
        records = vector_example.create_records()
        
        assert len(records) == 6
        
        for i, record in enumerate(records):
            # Check fields
            assert 'name' in record.fields
            assert 'description' in record.fields
            assert 'category' in record.fields
            assert 'embedding' in record.fields
            
            # Check embedding field
            embedding_field = record.fields['embedding']
            assert isinstance(embedding_field, VectorField)
            assert embedding_field.dimensions == 3
            assert isinstance(embedding_field.value, np.ndarray)
            assert embedding_field.value.shape == (3,)
            
            # Check normalization
            norm = np.linalg.norm(embedding_field.value)
            assert abs(norm - 1.0) < 1e-6  # Should be unit vector
            
            # Check metadata
            assert record.metadata['index'] == i
    
    def test_sync_memory_backend(self, vector_example):
        """Test sync Memory backend vector operations."""
        results = vector_example.run_sync_memory_example()
        
        # Should return top 3 results
        assert len(results) == 3
        
        # Check result structure
        for result in results:
            assert hasattr(result, 'record')
            assert hasattr(result, 'score')
            assert isinstance(result.record, Record)
            assert isinstance(result.score, float)
            assert -0.01 <= result.score <= 1.01  # Cosine similarity range (with tolerance)
        
        # First result should be XY-diagonal (closest to query [0.7, 0.7, 0.0])
        first_result = results[0]
        assert first_result.record.get_field('name').value == 'XY-diagonal'
        assert first_result.score > 0.99  # Should be very close to 1.0
        
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_sync_file_backend(self, vector_example):
        """Test sync File backend vector operations."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            filepath = tmp.name
        
        try:
            results = vector_example.run_sync_file_example(filepath=filepath)
            
            # Should return top 3 results
            assert len(results) == 3
            
            # Check that file was created
            assert os.path.exists(filepath)
            
            # First result should be Z-axis (query was [0, 0, 1])
            first_result = results[0]
            assert first_result.record.get_field('name').value == 'Z-axis'
            
            # Using Euclidean metric, score represents inverse distance
            assert first_result.score > 0.99
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_sync_file_backend_temp_file(self, vector_example):
        """Test sync File backend with automatic temp file creation."""
        # Test without providing filepath
        results = vector_example.run_sync_file_example()
        
        assert len(results) == 3
        # File should be automatically cleaned up
    
    @pytest.mark.asyncio
    async def test_async_memory_backend(self, vector_example):
        """Test async Memory backend vector operations."""
        results = await vector_example.run_async_memory_example()
        
        # Should return top 3 results
        assert len(results) == 3
        
        # First result should be X-axis (query was [1, 0, 0])
        first_result = results[0]
        assert first_result.record.get_field('name').value == 'X-axis'
        
        # Using dot product metric
        assert first_result.score > 0.99
        
        # Check all results have valid structure
        for result in results:
            assert hasattr(result, 'record')
            assert hasattr(result, 'score')
    
    @pytest.mark.asyncio
    async def test_async_file_with_filter(self, vector_example):
        """Test async File backend with filtering."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            filepath = tmp.name
        
        try:
            results = await vector_example.run_async_file_with_filter(filepath=filepath)
            
            # Should return at most 2 results (k=2)
            assert len(results) <= 2
            
            # All results should be from "diagonal" category
            for result in results:
                category = result.record.get_field('category').value
                assert category == 'diagonal'
            
            # First result should be XY-diagonal (closest to query)
            if results:
                first_result = results[0]
                assert first_result.record.get_field('name').value == 'XY-diagonal'
                
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    @pytest.mark.asyncio
    async def test_async_file_no_filepath(self, vector_example):
        """Test async File backend with automatic temp file."""
        results = await vector_example.run_async_file_with_filter()
        
        # Should work with automatically created temp file
        assert len(results) <= 2
        
        for result in results:
            category = result.record.get_field('category').value
            assert category == 'diagonal'
    
    def test_s3_example_info(self, vector_example, capsys):
        """Test S3 information display."""
        # Should not crash even in non-verbose mode
        vector_example.run_s3_example_info()
        
        # In verbose mode, check output
        verbose_example = MultiBackendVectorExample(verbose=True)
        verbose_example.run_s3_example_info()
        
        captured = capsys.readouterr()
        assert "S3 Backend Vector Support" in captured.out
        assert "backend='s3'" in captured.out
        assert "vector_enabled=True" in captured.out
    
    def test_vector_metrics(self, vector_example):
        """Test different vector similarity metrics."""
        # Create simple test vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([0.707, 0.707, 0.0])
        
        # Test cosine similarity (used in memory sync)
        # Perpendicular vectors should have 0 similarity
        dot = np.dot(vec1, vec2)
        assert abs(dot) < 1e-6
        
        # Test that diagonal is equally similar to both axes
        sim_x = np.dot(vec3 / np.linalg.norm(vec3), vec1)
        sim_y = np.dot(vec3 / np.linalg.norm(vec3), vec2)
        assert abs(sim_x - sim_y) < 1e-6
        assert abs(sim_x - 0.707) < 1e-2
    
    def test_record_serialization(self, vector_example):
        """Test that records with vectors can be serialized."""
        records = vector_example.create_records()
        
        for record in records:
            # Should be able to convert to dict
            record_dict = record.to_dict(include_metadata=True, flatten=False)
            
            # Check vector field is properly serialized
            embedding_data = record_dict['fields']['embedding']
            assert 'value' in embedding_data
            assert isinstance(embedding_data['value'], list)
            assert len(embedding_data['value']) == 3
            assert all(isinstance(x, float) for x in embedding_data['value'])


class TestIntegrationScenarios:
    """Integration test scenarios across backends."""
    
    @pytest.mark.asyncio
    async def test_cross_backend_consistency(self):
        """Test that all backends return consistent results for same data."""
        example = MultiBackendVectorExample(verbose=False)
        
        # Query vector
        query = np.array([0.5, 0.5, 0.5])
        query = query / np.linalg.norm(query)
        
        # Collect results from different backends
        sync_memory_results = example.run_sync_memory_example()
        async_memory_results = await example.run_async_memory_example()
        
        # Both memory backends should return same number of results
        assert len(sync_memory_results) == len(async_memory_results)
        
        # Check that top result names are consistent (allowing for metric differences)
        # Note: Different metrics may give different orderings
        sync_names = {r.record.get_field('name').value for r in sync_memory_results}
        async_names = {r.record.get_field('name').value for r in async_memory_results}
        
        # Should have significant overlap
        overlap = sync_names & async_names
        assert len(overlap) >= 2  # At least 2 common results in top 3
    
    def test_edge_cases(self, vector_example):
        """Test edge cases in vector operations."""
        # Test with zero vector (should be normalized)
        zero_vec = np.array([0.0, 0.0, 0.0])
        
        # NumPy division by zero results in NaN, not exception
        norm = np.linalg.norm(zero_vec)
        assert norm == 0.0
        
        # Division results in NaN values
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = zero_vec / norm
            assert np.all(np.isnan(normalized))
        
        # Test that very small vectors are handled
        small_vec = np.array([1e-10, 1e-10, 1e-10])
        norm = np.linalg.norm(small_vec)
        assert norm > 0  # Should be non-zero
        normalized = small_vec / norm
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6
    
    def test_full_workflow(self):
        """Test complete workflow across multiple backends."""
        example = MultiBackendVectorExample(verbose=False)
        
        # Run all sync examples
        memory_results = example.run_sync_memory_example()
        file_results = example.run_sync_file_example()
        
        # All should complete successfully
        assert memory_results is not None
        assert file_results is not None
        
        # Both should return results
        assert len(memory_results) > 0
        assert len(file_results) > 0


def test_main_function():
    """Test that the main function runs without errors."""
    # We need to test the individual components since main() uses asyncio.run()
    # which can't be called from within a test's event loop
    example = MultiBackendVectorExample(verbose=False)
    
    # Test sync operations
    memory_results = example.run_sync_memory_example()
    assert memory_results is not None
    assert len(memory_results) == 3
    
    file_results = example.run_sync_file_example()
    assert file_results is not None
    assert len(file_results) == 3
    
    # Test info display
    example.run_s3_example_info()  # Should not raise


@pytest.mark.asyncio
async def test_async_operations():
    """Test async operations separately."""
    example = MultiBackendVectorExample(verbose=False)
    
    # Test async operations
    async_memory_results = await example.run_async_memory_example()
    assert async_memory_results is not None
    assert len(async_memory_results) == 3
    
    async_file_results = await example.run_async_file_with_filter()
    assert async_file_results is not None
    assert len(async_file_results) <= 2


class TestS3Backend:
    """Tests specific to S3 backend (requires TEST_S3 environment variable)."""
    
    @pytest.mark.skipif(
        os.getenv("TEST_S3") != "true",
        reason="S3 tests require TEST_S3=true and localstack/AWS setup"
    )
    def test_s3_sync_backend(self):
        """Test S3 backend with vector support."""
        from dataknobs_data import DatabaseFactory
        
        # Detect if we're running in Docker container
        if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
            localstack_host = 'localstack'
        else:
            localstack_host = os.getenv('LOCALSTACK_HOST', 'localhost')
        
        factory = DatabaseFactory()
        db = factory.create(
            backend="s3",
            bucket="test-bucket",
            prefix="test-vectors/",
            endpoint_url=f"http://{localstack_host}:4566",  # LocalStack
            vector_enabled=True,
            vector_metric="cosine"
        )
        
        try:
            db.connect()
            
            # Create a test record with vector
            record = Record(
                data={
                    "name": Field(name="name", value="test"),
                    "embedding": VectorField(
                        name="embedding",
                        value=np.array([1.0, 0.0, 0.0]),
                        dimensions=3
                    )
                }
            )
            
            record_id = db.create(record)
            assert record_id is not None
            
            # Perform vector search
            results = db.vector_search(
                query_vector=np.array([1.0, 0.0, 0.0]),
                vector_field="embedding",
                k=1
            )
            
            assert len(results) == 1
            assert results[0].record.get_field('name').value == 'test'
            
        finally:
            db.clear()
    
    @pytest.mark.skipif(
        os.getenv("TEST_S3") != "true",
        reason="S3 tests require TEST_S3=true and localstack/AWS setup"
    )
    @pytest.mark.asyncio
    async def test_s3_async_backend(self):
        """Test async S3 backend with vector support."""
        from dataknobs_data import AsyncDatabaseFactory
        
        # Detect if we're running in Docker container
        if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
            localstack_host = 'localstack'
        else:
            localstack_host = os.getenv('LOCALSTACK_HOST', 'localhost')
        
        factory = AsyncDatabaseFactory()
        db = factory.create(
            backend="s3",
            bucket="test-bucket",
            prefix="test-vectors-async/",
            endpoint_url=f"http://{localstack_host}:4566",  # LocalStack
            vector_enabled=True,
            vector_metric="euclidean"
        )
        
        try:
            await db.connect()
            
            # Create test records
            example = MultiBackendVectorExample(verbose=False)
            records = example.create_records()
            
            for record in records:
                await db.create(record)
            
            # Perform vector search
            results = await db.vector_search(
                query_vector=np.array([0.0, 1.0, 0.0]),  # Y-axis
                vector_field="embedding",
                k=2
            )
            
            assert len(results) <= 2
            # First result should be Y-axis
            assert results[0].record.get_field('name').value == 'Y-axis'
            
        finally:
            await db.clear()
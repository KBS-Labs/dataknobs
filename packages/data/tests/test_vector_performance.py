"""Performance tests for vector store operations."""

import asyncio
import os
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="Vector performance tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

from dataknobs_data.vector.benchmarks import (
    BenchmarkResult,
    ComparativeBenchmark,
    VectorStoreBenchmark,
)
from dataknobs_data.vector.optimizations import (
    BatchConfig,
    BatchProcessor,
    ConnectionPool,
    ConnectionPoolConfig,
    QueryOptimizer,
    VectorOptimizer,
)
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.types import DistanceMetric


class TestVectorOptimizer:
    """Test vector optimization utilities."""
    
    def test_optimize_batch_size(self):
        """Test batch size optimization."""
        # Small dataset
        batch_size = VectorOptimizer.optimize_batch_size(
            num_vectors=100,
            vector_dim=128,
            available_memory=1024 * 1024 * 100  # 100MB
        )
        assert batch_size == 100  # Should fit all vectors
        
        # Large dataset with limited memory
        batch_size = VectorOptimizer.optimize_batch_size(
            num_vectors=1000000,
            vector_dim=768,
            available_memory=1024 * 1024 * 100  # 100MB
        )
        assert 10 <= batch_size <= 10000  # Should be within reasonable range
        
        # Edge case: very high dimensions
        batch_size = VectorOptimizer.optimize_batch_size(
            num_vectors=10000,
            vector_dim=4096,
            available_memory=1024 * 1024 * 50  # 50MB
        )
        assert batch_size >= 10  # Should have minimum batch size
    
    def test_select_index_type(self):
        """Test index type selection."""
        # Small dataset - should use flat
        config = VectorOptimizer.select_index_type(
            num_vectors=1000,
            vector_dim=128,
            metric=DistanceMetric.COSINE
        )
        assert config["type"] == "flat"
        
        # Medium dataset - should use IVF
        config = VectorOptimizer.select_index_type(
            num_vectors=100000,
            vector_dim=256,
            metric=DistanceMetric.EUCLIDEAN
        )
        assert config["type"] == "ivfflat"
        assert "nlist" in config
        assert 100 <= config["nlist"] <= 4096
        
        # Large dataset - should use HNSW
        config = VectorOptimizer.select_index_type(
            num_vectors=10000000,
            vector_dim=512,
            metric=DistanceMetric.DOT_PRODUCT
        )
        assert config["type"] == "hnsw"
        assert config["m"] == 16
        assert config["ef_construction"] == 200
    
    def test_optimize_search_params(self):
        """Test search parameter optimization."""
        # Flat index - no params needed
        params = VectorOptimizer.optimize_search_params("flat", 0.95)
        assert params == {}
        
        # IVF with high recall
        params = VectorOptimizer.optimize_search_params("ivfflat", 0.99)
        assert params["nprobe"] == 128
        
        # IVF with moderate recall
        params = VectorOptimizer.optimize_search_params("ivfflat", 0.90)
        assert params["nprobe"] == 32
        
        # HNSW with high recall
        params = VectorOptimizer.optimize_search_params("hnsw", 0.99)
        assert params["ef_search"] == 200
        
        # HNSW with low recall
        params = VectorOptimizer.optimize_search_params("hnsw", 0.85)
        assert params["ef_search"] == 32


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_add_and_flush(self):
        """Test adding items and flushing batches."""
        config = BatchConfig(size=3, max_queue_size=10)
        processor = BatchProcessor(config)
        
        processed_items = []
        
        async def callback(item):
            processed_items.append(item)
        
        # Add items (less than batch size)
        await processor.add(1, callback)
        await processor.add(2, callback)
        assert len(processed_items) == 0  # Not flushed yet
        
        # Add one more to trigger auto-flush
        await processor.add(3, callback)
        # Auto-flush should have happened
        assert len(processed_items) == 3
        
        # Add more items
        await processor.add(4, callback)
        await processor.add(5, callback)
        
        # Manual flush
        count = await processor.flush()
        assert count == 2
        assert len(processed_items) == 5
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self):
        """Test parallel batch processing."""
        config = BatchConfig(size=4, parallel_workers=2)
        processor = BatchProcessor(config)
        
        processed_items = []
        lock = asyncio.Lock()
        
        async def callback(item):
            async with lock:
                processed_items.append(item)
            await asyncio.sleep(0.01)  # Simulate work
        
        # Add items
        for i in range(8):
            await processor.add(i, callback)
        
        # Process all
        await processor.flush()
        await processor.flush()
        
        assert len(processed_items) == 8
        assert set(processed_items) == set(range(8))
    
    @pytest.mark.asyncio
    async def test_auto_flush(self):
        """Test automatic flushing."""
        config = BatchConfig(size=10, flush_interval=0.1)
        processor = BatchProcessor(config)
        
        processed_items = []
        
        async def callback(item):
            processed_items.append(item)
        
        # Start auto-flush
        await processor.start_auto_flush()
        
        # Add items
        await processor.add(1, callback)
        await processor.add(2, callback)
        
        # Wait for auto-flush
        await asyncio.sleep(0.2)
        assert len(processed_items) == 2
        
        # Stop auto-flush
        await processor.stop_auto_flush()
    
    @pytest.mark.asyncio
    async def test_error_handling_with_retry(self):
        """Test error handling and retry logic."""
        config = BatchConfig(size=2, retry_on_failure=True, max_retries=3)
        processor = BatchProcessor(config)
        
        attempt_count = {}
        processed_items = []
        
        async def callback(item):
            if item not in attempt_count:
                attempt_count[item] = 0
            attempt_count[item] += 1
            
            # Fail first attempt
            if attempt_count[item] == 1:
                raise ValueError("Simulated error")
            
            processed_items.append(item)
        
        # Add items
        await processor.add(1, callback)
        await processor.add(2, callback)
        
        # Process with retry
        await processor.flush()
        await processor.flush()  # Retry failed items
        
        assert len(processed_items) == 2
        assert attempt_count[1] == 2
        assert attempt_count[2] == 2


class TestConnectionPool:
    """Test connection pooling."""
    
    @pytest.mark.asyncio
    async def test_connection_acquire_release(self):
        """Test acquiring and releasing connections."""
        connection_count = 0
        
        async def factory():
            nonlocal connection_count
            connection_count += 1
            return f"conn_{connection_count}"
        
        config = ConnectionPoolConfig(min_connections=1, max_connections=3)
        pool = ConnectionPool(factory, config)
        
        # Acquire first connection
        conn1 = await pool.acquire()
        assert conn1 == "conn_1"
        assert len(pool.in_use) == 1
        
        # Acquire second connection
        conn2 = await pool.acquire()
        assert conn2 == "conn_2"
        assert len(pool.in_use) == 2
        
        # Release first connection
        await pool.release(conn1)
        assert len(pool.in_use) == 1
        assert len(pool.available) == 1
        
        # Acquire again - should reuse
        conn3 = await pool.acquire()
        assert conn3 == "conn_1"  # Reused
        assert connection_count == 2  # Only 2 created total
        
        # Cleanup
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_connection_limit(self):
        """Test connection pool limits."""
        async def factory():
            return Mock()
        
        config = ConnectionPoolConfig(max_connections=2)
        pool = ConnectionPool(factory, config)
        
        # Acquire max connections
        conn1 = await pool.acquire()
        conn2 = await pool.acquire()
        
        # Try to acquire one more (should timeout quickly in test)
        with pytest.raises(asyncio.TimeoutError):
            # This will retry for ~10 seconds then timeout
            await asyncio.wait_for(pool.acquire(), timeout=0.5)
        
        # Release one and try again
        await pool.release(conn1)
        conn3 = await pool.acquire()
        assert conn3 is not None
        
        # Cleanup
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_pool_close(self):
        """Test closing the connection pool."""
        closed_connections = []
        
        class MockConnection:
            async def close(self):
                closed_connections.append(self)
        
        async def factory():
            return MockConnection()
        
        pool = ConnectionPool(factory)
        
        # Create some connections
        conn1 = await pool.acquire()
        conn2 = await pool.acquire()
        await pool.release(conn1)
        
        # Close pool
        await pool.close()
        
        # Should have closed all connections
        assert len(closed_connections) == 2
        
        # Should not be able to acquire after close
        with pytest.raises(RuntimeError):
            await pool.acquire()


class TestQueryOptimizer:
    """Test query optimization logic."""
    
    def test_should_use_index(self):
        """Test index usage decision."""
        # Small k, should use index
        assert QueryOptimizer.should_use_index(10000, 10, 1.0) is True
        
        # Large k relative to dataset, should scan
        assert QueryOptimizer.should_use_index(10000, 2000, 1.0) is False
        
        # Very selective filter, should scan filtered
        assert QueryOptimizer.should_use_index(10000, 10, 0.005) is False
        
        # Normal case
        assert QueryOptimizer.should_use_index(100000, 100, 0.5) is True
    
    def test_optimize_reranking(self):
        """Test reranking optimization."""
        # Basic reranking
        candidates = QueryOptimizer.optimize_reranking(1000, 10, 3.0)
        assert candidates == 30
        
        # Ensure minimum candidates
        candidates = QueryOptimizer.optimize_reranking(1000, 10, 1.0)
        assert candidates >= 20  # At least 2x final_k
        
        # Ensure maximum candidates
        candidates = QueryOptimizer.optimize_reranking(50, 10, 10.0)
        assert candidates <= 50  # Can't exceed initial_k


class TestVectorStoreBenchmark:
    """Test benchmarking functionality."""
    
    @pytest.fixture
    async def store(self):
        """Create a memory vector store for testing."""
        config = {"dimensions": 128, "metric": "cosine"}
        store = MemoryVectorStore(config)
        await store.initialize()
        return store
    
    @pytest.mark.asyncio
    async def test_benchmark_indexing(self, store):
        """Test indexing benchmark."""
        benchmark = VectorStoreBenchmark(store)
        
        result = await benchmark.benchmark_indexing(
            num_vectors=100,
            vector_dim=128,
            batch_size=10
        )
        
        assert result.operation == "indexing"
        assert result.num_vectors == 100
        assert result.throughput > 0
        assert result.duration > 0
        assert await store.count() == 100
    
    @pytest.mark.asyncio
    async def test_benchmark_search(self, store):
        """Test search benchmark."""
        benchmark = VectorStoreBenchmark(store)
        
        # Add some vectors first
        await benchmark.benchmark_indexing(100, 128)
        
        result = await benchmark.benchmark_search(
            num_queries=10,
            k=5,
            vector_dim=128
        )
        
        assert result.operation == "search"
        assert result.throughput > 0
        assert result.latency_p50 is not None
        assert result.latency_p95 is not None
        assert result.latency_p99 is not None
    
    @pytest.mark.asyncio
    async def test_benchmark_update(self, store):
        """Test update benchmark."""
        benchmark = VectorStoreBenchmark(store)
        
        # Add vectors first
        await benchmark.benchmark_indexing(50, 128)
        
        result = await benchmark.benchmark_update(
            num_updates=20,
            vector_dim=128
        )
        
        assert result.operation == "update"
        assert result.num_vectors == 20
        assert result.throughput > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_delete(self, store):
        """Test deletion benchmark."""
        benchmark = VectorStoreBenchmark(store)
        
        # Add vectors first
        await benchmark.benchmark_indexing(50, 128)
        initial_count = await store.count()
        
        result = await benchmark.benchmark_delete(num_deletes=20)
        
        assert result.operation == "delete"
        assert result.throughput > 0
        assert await store.count() < initial_count
    
    @pytest.mark.asyncio
    async def test_benchmark_concurrent(self, store):
        """Test concurrent operations benchmark."""
        benchmark = VectorStoreBenchmark(store)
        
        result = await benchmark.benchmark_concurrent_operations(
            num_workers=3,
            operations_per_worker=10,
            vector_dim=128
        )
        
        assert result.operation == "concurrent"
        assert result.throughput > 0
        assert result.metadata["num_workers"] == 3
        assert "avg_worker_time" in result.metadata
    
    @pytest.mark.asyncio
    async def test_benchmark_report(self, store):
        """Test report generation."""
        benchmark = VectorStoreBenchmark(store)
        
        # Run some benchmarks
        await benchmark.benchmark_indexing(50, 128)
        await benchmark.benchmark_search(10, 5, 128)
        
        report = benchmark.generate_report()
        
        assert "Vector Store Benchmark Report" in report
        assert "INDEXING Operations:" in report
        assert "SEARCH Operations:" in report
        assert "Throughput:" in report
        assert "Latency P50:" in report


class TestComparativeBenchmark:
    """Test comparative benchmarking."""
    
    @pytest.mark.asyncio
    async def test_compare_stores(self):
        """Test comparing multiple stores."""
        # Create two stores
        store1 = MemoryVectorStore({"dimensions": 64, "metric": "cosine"})
        store2 = MemoryVectorStore({"dimensions": 64, "metric": "euclidean"})
        
        await store1.initialize()
        await store2.initialize()
        
        stores = {"cosine": store1, "euclidean": store2}
        comparison = ComparativeBenchmark(stores)
        
        # Compare indexing
        results = await comparison.compare_indexing(
            num_vectors=100,
            vector_dim=64
        )
        
        assert "cosine" in results
        assert "euclidean" in results
        assert results["cosine"].operation == "indexing"
        assert results["euclidean"].operation == "indexing"
        
        # Generate report
        report = comparison.generate_comparison_report()
        assert "Vector Store Comparison Report" in report
        assert "INDEXING Comparison:" in report
        assert "cosine" in report
        assert "euclidean" in report
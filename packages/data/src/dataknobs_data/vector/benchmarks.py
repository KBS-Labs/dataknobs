"""Vector store performance benchmarks."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .stores.base import VectorStore


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    operation: str
    num_vectors: int
    vector_dim: int
    duration: float
    throughput: float
    memory_used: int | None = None
    latency_p50: float | None = None
    latency_p95: float | None = None
    latency_p99: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __str__(self) -> str:
        """String representation of results."""
        lines = [
            f"Operation: {self.operation}",
            f"Vectors: {self.num_vectors:,}",
            f"Dimensions: {self.vector_dim}",
            f"Duration: {self.duration:.3f}s",
            f"Throughput: {self.throughput:.0f} vectors/s"
        ]

        if self.latency_p50 is not None:
            lines.append(f"Latency P50: {self.latency_p50*1000:.2f}ms")
        if self.latency_p95 is not None:
            lines.append(f"Latency P95: {self.latency_p95*1000:.2f}ms")
        if self.latency_p99 is not None:
            lines.append(f"Latency P99: {self.latency_p99*1000:.2f}ms")
        if self.memory_used is not None:
            lines.append(f"Memory: {self.memory_used / (1024*1024):.1f}MB")

        return "\n".join(lines)


class VectorStoreBenchmark:
    """Benchmarks for vector store operations."""

    def __init__(self, store: VectorStore):
        """Initialize benchmark with a vector store.
        
        Args:
            store: Vector store to benchmark
        """
        self.store = store
        self.results: list[BenchmarkResult] = []
        self.rng = np.random.default_rng()  # Create RNG once for all benchmarks

    async def benchmark_indexing(
        self,
        num_vectors: int = 10000,
        vector_dim: int = 128,
        batch_size: int = 100
    ) -> BenchmarkResult:
        """Benchmark vector indexing performance.
        
        Args:
            num_vectors: Number of vectors to index
            vector_dim: Dimension of vectors
            batch_size: Batch size for indexing
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking indexing: {num_vectors} vectors of dim {vector_dim}")

        # Generate random vectors
        vectors = self.rng.random((num_vectors, vector_dim), dtype=np.float32)
        ids = [str(i) for i in range(num_vectors)]
        metadata = [{"index": i} for i in range(num_vectors)]

        # Measure indexing time
        start_time = time.time()

        # Index in batches
        for i in range(0, num_vectors, batch_size):
            batch_end = min(i + batch_size, num_vectors)
            await self.store.add_vectors(
                vectors[i:batch_end],
                ids=ids[i:batch_end],
                metadata=metadata[i:batch_end]
            )

        duration = time.time() - start_time
        throughput = num_vectors / duration if duration > 0 else 0

        result = BenchmarkResult(
            operation="indexing",
            num_vectors=num_vectors,
            vector_dim=vector_dim,
            duration=duration,
            throughput=throughput,
            metadata={"batch_size": batch_size}
        )

        self.results.append(result)
        return result

    async def benchmark_search(
        self,
        num_queries: int = 1000,
        k: int = 10,
        vector_dim: int = 128
    ) -> BenchmarkResult:
        """Benchmark vector search performance.
        
        Args:
            num_queries: Number of search queries
            k: Number of results per query
            vector_dim: Dimension of query vectors
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking search: {num_queries} queries, k={k}")

        # Generate random query vectors
        queries = self.rng.random((num_queries, vector_dim), dtype=np.float32)

        # Measure search latencies
        latencies = []
        start_time = time.time()

        for i in range(num_queries):
            query_start = time.time()
            await self.store.search(queries[i], k=k)
            latencies.append(time.time() - query_start)

        duration = time.time() - start_time
        throughput = num_queries / duration if duration > 0 else 0

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        result = BenchmarkResult(
            operation="search",
            num_vectors=await self.store.count(),
            vector_dim=vector_dim,
            duration=duration,
            throughput=throughput,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            metadata={"num_queries": num_queries, "k": k}
        )

        self.results.append(result)
        return result

    async def benchmark_update(
        self,
        num_updates: int = 1000,
        vector_dim: int = 128
    ) -> BenchmarkResult:
        """Benchmark vector update performance.
        
        Args:
            num_updates: Number of vectors to update
            vector_dim: Dimension of vectors
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking updates: {num_updates} vectors")

        # Get existing vector IDs
        count = await self.store.count()
        if count == 0:
            # Add some vectors first
            await self.benchmark_indexing(num_updates, vector_dim)

        # Generate new vectors for updates
        vectors = self.rng.random((num_updates, vector_dim), dtype=np.float32)
        ids = [str(i) for i in range(num_updates)]
        metadata = [{"updated": True, "index": i} for i in range(num_updates)]

        # Measure update time (vectors and metadata)
        start_time = time.time()

        # Update vectors and metadata
        await self.store.update_vectors(vectors, ids, metadata)

        duration = time.time() - start_time
        throughput = num_updates / duration if duration > 0 else 0

        result = BenchmarkResult(
            operation="update",
            num_vectors=num_updates,
            vector_dim=vector_dim,
            duration=duration,
            throughput=throughput
        )

        self.results.append(result)
        return result

    async def benchmark_delete(
        self,
        num_deletes: int = 1000
    ) -> BenchmarkResult:
        """Benchmark vector deletion performance.
        
        Args:
            num_deletes: Number of vectors to delete
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking deletion: {num_deletes} vectors")

        # Get vector count
        initial_count = await self.store.count()

        # Generate IDs to delete
        ids = [str(i) for i in range(min(num_deletes, initial_count))]

        # Measure deletion time
        start_time = time.time()
        deleted = await self.store.delete_vectors(ids)
        duration = time.time() - start_time

        throughput = deleted / duration if duration > 0 else 0

        result = BenchmarkResult(
            operation="delete",
            num_vectors=deleted,
            vector_dim=0,
            duration=duration,
            throughput=throughput
        )

        self.results.append(result)
        return result

    async def benchmark_concurrent_operations(
        self,
        num_workers: int = 10,
        operations_per_worker: int = 100,
        vector_dim: int = 128
    ) -> BenchmarkResult:
        """Benchmark concurrent operations.
        
        Args:
            num_workers: Number of concurrent workers
            operations_per_worker: Operations per worker
            vector_dim: Dimension of vectors
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking concurrency: {num_workers} workers")

        async def worker(worker_id: int) -> float:
            """Worker function for concurrent operations."""
            start = time.time()

            for i in range(operations_per_worker):
                # Mix of operations
                if i % 4 == 0:
                    # Add vector
                    vector = self.rng.random(vector_dim, dtype=np.float32)
                    await self.store.add_vectors(
                        vector,
                        ids=[f"w{worker_id}_v{i}"],
                        metadata=[{"worker": worker_id}]
                    )
                else:
                    # Search
                    query = self.rng.random(vector_dim, dtype=np.float32)
                    await self.store.search(query, k=5)

            return time.time() - start

        # Run workers concurrently
        start_time = time.time()
        tasks = [worker(i) for i in range(num_workers)]
        worker_times = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        total_ops = num_workers * operations_per_worker
        throughput = total_ops / duration if duration > 0 else 0

        # Calculate worker time statistics
        avg_worker_time = sum(worker_times) / len(worker_times)

        result = BenchmarkResult(
            operation="concurrent",
            num_vectors=total_ops,
            vector_dim=vector_dim,
            duration=duration,
            throughput=throughput,
            metadata={
                "num_workers": num_workers,
                "ops_per_worker": operations_per_worker,
                "avg_worker_time": avg_worker_time
            }
        )

        self.results.append(result)
        return result

    async def run_full_benchmark(
        self,
        vector_dims: list[int] | None = None,
        num_vectors_list: list[int] | None = None
    ) -> list[BenchmarkResult]:
        """Run a complete benchmark suite.
        
        Args:
            vector_dims: List of vector dimensions to test
            num_vectors_list: List of vector counts to test
            
        Returns:
            List of all benchmark results
        """
        if vector_dims is None:
            vector_dims = [128, 256, 512]
        if num_vectors_list is None:
            num_vectors_list = [1000, 10000, 50000]

        logger.info("Starting full benchmark suite")

        for dim in vector_dims:
            for num_vectors in num_vectors_list:
                # Clear store
                await self.store.clear()

                # Run benchmarks
                await self.benchmark_indexing(num_vectors, dim)
                await self.benchmark_search(min(1000, num_vectors // 10), 10, dim)
                await self.benchmark_update(min(1000, num_vectors // 10), dim)
                await self.benchmark_delete(min(1000, num_vectors // 10))

        # Test concurrency
        await self.store.clear()
        await self.benchmark_concurrent_operations()

        return self.results

    def generate_report(self) -> str:
        """Generate a benchmark report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No benchmark results available"

        lines = ["=" * 60, "Vector Store Benchmark Report", "=" * 60, ""]

        # Group by operation
        by_operation = {}
        for result in self.results:
            if result.operation not in by_operation:
                by_operation[result.operation] = []
            by_operation[result.operation].append(result)

        for operation, results in by_operation.items():
            lines.append(f"\n{operation.upper()} Operations:")
            lines.append("-" * 40)

            for result in results:
                lines.append(str(result))
                lines.append("")

        return "\n".join(lines)


class ComparativeBenchmark:
    """Compare performance across different vector stores."""

    def __init__(self, stores: dict[str, VectorStore]):
        """Initialize with multiple stores to compare.
        
        Args:
            stores: Dictionary of store name to store instance
        """
        self.stores = stores
        self.results: dict[str, list[BenchmarkResult]] = {}

    async def compare_indexing(
        self,
        num_vectors: int = 10000,
        vector_dim: int = 128
    ) -> dict[str, BenchmarkResult]:
        """Compare indexing performance across stores.
        
        Args:
            num_vectors: Number of vectors to index
            vector_dim: Dimension of vectors
            
        Returns:
            Dictionary of store name to results
        """
        comparison = {}

        for name, store in self.stores.items():
            logger.info(f"Benchmarking {name}")
            benchmark = VectorStoreBenchmark(store)
            result = await benchmark.benchmark_indexing(num_vectors, vector_dim)
            comparison[name] = result

            if name not in self.results:
                self.results[name] = []
            self.results[name].append(result)

        return comparison

    async def compare_search(
        self,
        num_queries: int = 1000,
        k: int = 10,
        vector_dim: int = 128
    ) -> dict[str, BenchmarkResult]:
        """Compare search performance across stores.
        
        Args:
            num_queries: Number of queries
            k: Results per query
            vector_dim: Query vector dimension
            
        Returns:
            Dictionary of store name to results
        """
        comparison = {}

        for name, store in self.stores.items():
            logger.info(f"Benchmarking {name} search")
            benchmark = VectorStoreBenchmark(store)
            result = await benchmark.benchmark_search(num_queries, k, vector_dim)
            comparison[name] = result

            if name not in self.results:
                self.results[name] = []
            self.results[name].append(result)

        return comparison

    def generate_comparison_report(self) -> str:
        """Generate a comparison report.
        
        Returns:
            Formatted comparison report
        """
        if not self.results:
            return "No comparison results available"

        lines = ["=" * 80, "Vector Store Comparison Report", "=" * 80, ""]

        # Find all operations
        all_operations = set()
        for store_results in self.results.values():
            for result in store_results:
                all_operations.add(result.operation)

        # Compare by operation
        for operation in all_operations:
            lines.append(f"\n{operation.upper()} Comparison:")
            lines.append("-" * 60)

            # Create comparison table
            table_data = []
            for store_name, store_results in self.results.items():
                for result in store_results:
                    if result.operation == operation:
                        table_data.append([
                            store_name,
                            f"{result.throughput:.0f} vec/s",
                            f"{result.duration:.3f}s",
                            f"{result.latency_p50*1000:.1f}ms" if result.latency_p50 else "N/A"
                        ])

            # Format table
            if table_data:
                headers = ["Store", "Throughput", "Duration", "P50 Latency"]
                col_widths = [
                    max(len(h), max(len(row[i]) for row in table_data))
                    for i, h in enumerate(headers)
                ]

                # Header
                header_line = " | ".join(
                    h.ljust(w) for h, w in zip(headers, col_widths, strict=False)
                )
                lines.append(header_line)
                lines.append("-" * len(header_line))

                # Data rows
                for row in table_data:
                    row_line = " | ".join(
                        cell.ljust(w) for cell, w in zip(row, col_widths, strict=False)
                    )
                    lines.append(row_line)

            lines.append("")

        return "\n".join(lines)


# Export main classes
__all__ = [
    "BenchmarkResult",
    "ComparativeBenchmark",
    "VectorStoreBenchmark",
]

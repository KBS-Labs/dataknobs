"""Benchmarks for RAG search operations."""

import time
import asyncio
from typing import List

from dataknobs_llm.prompts.adapters.dict_adapter import AsyncDictResourceAdapter

try:
    from .benchmark_result import BenchmarkResult
except ImportError:
    from benchmark_result import BenchmarkResult


class RAGBenchmark:
    """Benchmark RAG search performance.

    This class provides benchmarks for:
    - Dict adapter search on various dataset sizes
    - Search with different k values
    - Parallel RAG searches
    """

    def __init__(self, iterations: int = 100):
        """Initialize RAG benchmark.

        Args:
            iterations: Number of iterations to run for each benchmark
        """
        self.iterations = iterations

    def _create_dataset(self, size: int) -> dict:
        """Create a test dataset of specified size.

        Args:
            size: Number of items in dataset

        Returns:
            Dictionary with test data
        """
        return {
            f"item_{i}": {
                "title": f"Item {i}",
                "description": f"This is a description for item number {i}",
                "content": f"Content for item {i}. " * 20  # ~20 words per item
            }
            for i in range(size)
        }

    async def benchmark_dict_adapter_small(self) -> BenchmarkResult:
        """Benchmark DictResourceAdapter search on small dataset (100 items)."""
        data = self._create_dataset(100)
        adapter = AsyncDictResourceAdapter(data)

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            await adapter.search("item 50", k=5)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("RAG Search: Small Dataset (100 items)", times)

    async def benchmark_dict_adapter_medium(self) -> BenchmarkResult:
        """Benchmark DictResourceAdapter search on medium dataset (1000 items)."""
        data = self._create_dataset(1000)
        adapter = AsyncDictResourceAdapter(data)

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            await adapter.search("item 500", k=5)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("RAG Search: Medium Dataset (1K items)", times)

    async def benchmark_dict_adapter_large(self) -> BenchmarkResult:
        """Benchmark DictResourceAdapter search on large dataset (10000 items)."""
        data = self._create_dataset(10000)
        adapter = AsyncDictResourceAdapter(data)

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            await adapter.search("item 5000", k=5)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("RAG Search: Large Dataset (10K items)", times)

    async def benchmark_different_k_values(self) -> List[BenchmarkResult]:
        """Benchmark search with different k values (result counts).

        Returns:
            List of BenchmarkResults for k=1, 5, 10, 20
        """
        data = self._create_dataset(1000)
        adapter = AsyncDictResourceAdapter(data)

        results = []
        for k in [1, 5, 10, 20]:
            times = []
            for _ in range(self.iterations):
                start = time.perf_counter()
                await adapter.search("item 500", k=k)
                end = time.perf_counter()
                times.append(end - start)

            result = BenchmarkResult.from_times(f"RAG Search: k={k}", times)
            results.append(result)

        return results

    async def benchmark_parallel_searches(self) -> BenchmarkResult:
        """Benchmark parallel RAG searches (4 concurrent queries)."""
        data = self._create_dataset(1000)
        adapter = AsyncDictResourceAdapter(data)

        queries = [
            "item 250",
            "item 500",
            "item 750",
            "item 900"
        ]

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            # Execute searches in parallel
            await asyncio.gather(*[
                adapter.search(query, k=5)
                for query in queries
            ])
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("RAG Search: 4 Parallel Queries", times)

    async def run_all_async(self) -> List[BenchmarkResult]:
        """Run all RAG search benchmarks asynchronously.

        Returns:
            List of BenchmarkResult objects
        """
        print(f"Running RAG search benchmarks ({self.iterations} iterations)...")
        print()

        results = []

        # Dataset size benchmarks
        print("Running Small Dataset...")
        result = await self.benchmark_dict_adapter_small()
        results.append(result)
        print(f"  {result.operations_per_second:.0f} ops/sec "
              f"({result.mean_time * 1000:.3f}ms mean)")

        print("Running Medium Dataset...")
        result = await self.benchmark_dict_adapter_medium()
        results.append(result)
        print(f"  {result.operations_per_second:.0f} ops/sec "
              f"({result.mean_time * 1000:.3f}ms mean)")

        print("Running Large Dataset...")
        result = await self.benchmark_dict_adapter_large()
        results.append(result)
        print(f"  {result.operations_per_second:.0f} ops/sec "
              f"({result.mean_time * 1000:.3f}ms mean)")

        # K value benchmarks
        print("Running Different K Values...")
        k_results = await self.benchmark_different_k_values()
        results.extend(k_results)
        for result in k_results:
            print(f"  {result.name}: {result.operations_per_second:.0f} ops/sec "
                  f"({result.mean_time * 1000:.3f}ms mean)")

        # Parallel search benchmark
        print("Running Parallel Searches...")
        result = await self.benchmark_parallel_searches()
        results.append(result)
        print(f"  {result.operations_per_second:.0f} ops/sec "
              f"({result.mean_time * 1000:.3f}ms mean)")

        print()
        print("RAG search benchmarks complete!")
        return results

    def run_all(self) -> List[BenchmarkResult]:
        """Run all RAG search benchmarks (synchronous wrapper).

        Returns:
            List of BenchmarkResult objects
        """
        return asyncio.run(self.run_all_async())

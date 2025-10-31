"""Benchmark result data structures and utilities."""

import statistics
from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        name: Benchmark name/description
        iterations: Number of iterations run
        total_time: Total time for all iterations (seconds)
        mean_time: Mean time per iteration (seconds)
        median_time: Median time per iteration (seconds)
        std_dev: Standard deviation (seconds)
        min_time: Minimum time (seconds)
        max_time: Maximum time (seconds)
        operations_per_second: Throughput (ops/sec)
    """
    name: str
    iterations: int
    total_time: float
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    operations_per_second: float

    @classmethod
    def from_times(cls, name: str, times: List[float]) -> "BenchmarkResult":
        """Create BenchmarkResult from a list of execution times.

        Args:
            name: Benchmark name
            times: List of execution times in seconds

        Returns:
            BenchmarkResult with computed statistics
        """
        iterations = len(times)
        total_time = sum(times)
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        ops_per_sec = iterations / total_time if total_time > 0 else 0.0

        return cls(
            name=name,
            iterations=iterations,
            total_time=total_time,
            mean_time=mean_time,
            median_time=median_time,
            std_dev=std_dev,
            min_time=min_time,
            max_time=max_time,
            operations_per_second=ops_per_sec
        )

    def format_summary(self) -> str:
        """Format benchmark result as a summary string.

        Returns:
            Formatted summary with key metrics
        """
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_time * 1000:.3f}ms\n"
            f"  Median: {self.median_time * 1000:.3f}ms\n"
            f"  Std Dev: {self.std_dev * 1000:.3f}ms\n"
            f"  Min: {self.min_time * 1000:.3f}ms\n"
            f"  Max: {self.max_time * 1000:.3f}ms\n"
            f"  Throughput: {self.operations_per_second:.0f} ops/sec"
        )

    def format_table_row(self) -> str:
        """Format benchmark result as a markdown table row.

        Returns:
            Markdown table row string
        """
        return (
            f"| {self.name} "
            f"| {self.operations_per_second:.0f} "
            f"| {self.mean_time * 1000:.2f} "
            f"| {self.median_time * 1000:.2f} "
            f"| {self.std_dev * 1000:.2f} |"
        )

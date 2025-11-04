"""Tests for benchmark infrastructure and result handling."""

import sys
from pathlib import Path
import pytest

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent.parent / "benchmarks"
sys.path.insert(0, str(benchmarks_dir))

from benchmark_result import BenchmarkResult


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass and methods."""

    def test_from_times_basic(self):
        """Test creating BenchmarkResult from times list."""
        times = [0.001, 0.002, 0.001, 0.003, 0.001]

        result = BenchmarkResult.from_times("Test Benchmark", times)

        assert result.name == "Test Benchmark"
        assert result.iterations == 5
        assert result.total_time == sum(times)
        assert result.mean_time == pytest.approx(0.0016, abs=0.0001)
        assert result.min_time == 0.001
        assert result.max_time == 0.003
        assert result.operations_per_second > 0

    def test_from_times_single_iteration(self):
        """Test with single iteration."""
        times = [0.005]

        result = BenchmarkResult.from_times("Single", times)

        assert result.iterations == 1
        assert result.mean_time == 0.005
        assert result.median_time == 0.005
        assert result.std_dev == 0.0  # No std dev with single value
        assert result.operations_per_second == pytest.approx(200.0, abs=1.0)

    def test_format_summary(self):
        """Test summary formatting."""
        times = [0.001, 0.002, 0.001]
        result = BenchmarkResult.from_times("Test", times)

        summary = result.format_summary()

        assert "Test:" in summary
        assert "Iterations: 3" in summary
        assert "Mean:" in summary
        assert "ms" in summary
        assert "Throughput:" in summary
        assert "ops/sec" in summary

    def test_format_table_row(self):
        """Test markdown table row formatting."""
        times = [0.001, 0.002, 0.001]
        result = BenchmarkResult.from_times("Test", times)

        row = result.format_table_row()

        assert row.startswith("|")
        assert row.endswith("|")
        assert "Test" in row
        assert "|" in row  # Multiple columns

    def test_operations_per_second_calculation(self):
        """Test throughput calculation."""
        # 1000 iterations at 0.001s each = 1000 ops/sec
        times = [0.001] * 1000

        result = BenchmarkResult.from_times("Throughput Test", times)

        # Total time = 1.0s, iterations = 1000
        # ops/sec = 1000 / 1.0 = 1000
        assert result.operations_per_second == pytest.approx(1000.0, abs=10.0)

    def test_zero_time_edge_case(self):
        """Test handling of zero times (shouldn't happen but defensive)."""
        times = [0.0, 0.0, 0.0]

        result = BenchmarkResult.from_times("Zero Time", times)

        # Should handle gracefully without division by zero
        assert result.total_time == 0.0
        assert result.operations_per_second == 0.0

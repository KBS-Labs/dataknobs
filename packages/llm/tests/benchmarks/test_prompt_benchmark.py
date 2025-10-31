"""Tests for PromptBenchmark to ensure all benchmarks run correctly."""

import sys
from pathlib import Path
import pytest

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent.parent / "benchmarks"
sys.path.insert(0, str(benchmarks_dir))

from prompt_benchmark import PromptBenchmark


class TestPromptBenchmark:
    """Test PromptBenchmark class methods."""

    @pytest.fixture
    def prompt_bench(self):
        """Create benchmark with small iteration count for tests."""
        return PromptBenchmark(iterations=10)

    def test_benchmark_simple_rendering(self, prompt_bench):
        """Test simple variable substitution benchmark."""
        result = prompt_bench.benchmark_simple_rendering()

        assert result.name == "Simple Variable Substitution"
        assert result.iterations == 10
        assert result.mean_time > 0
        assert result.operations_per_second > 0
        assert result.std_dev >= 0

    def test_benchmark_conditional_rendering(self, prompt_bench):
        """Test conditional rendering benchmark."""
        result = prompt_bench.benchmark_conditional_rendering()

        assert result.name == "Conditional Rendering (( ))"
        assert result.iterations == 10
        assert result.mean_time > 0
        assert result.operations_per_second > 0

    def test_benchmark_jinja2_filters(self, prompt_bench):
        """Test Jinja2 filters benchmark."""
        result = prompt_bench.benchmark_jinja2_filters()

        assert result.name == "Jinja2 Filters"
        assert result.iterations == 10
        assert result.mean_time > 0

    def test_benchmark_jinja2_conditionals(self, prompt_bench):
        """Test Jinja2 conditionals benchmark."""
        result = prompt_bench.benchmark_jinja2_conditionals()

        assert result.name == "Jinja2 Conditionals"
        assert result.iterations == 10
        assert result.mean_time > 0

    def test_benchmark_jinja2_loops(self, prompt_bench):
        """Test Jinja2 loops benchmark."""
        result = prompt_bench.benchmark_jinja2_loops()

        assert result.name == "Jinja2 Loops (10 items)"
        assert result.iterations == 10
        assert result.mean_time > 0

    def test_benchmark_complex_template(self, prompt_bench):
        """Test complex template benchmark."""
        result = prompt_bench.benchmark_complex_template()

        assert result.name == "Complex Template (Mixed)"
        assert result.iterations == 10
        assert result.mean_time > 0

    def test_benchmark_mixed_vs_jinja2_mode(self, prompt_bench):
        """Test mode comparison benchmark."""
        results = prompt_bench.benchmark_mixed_vs_jinja2_mode()

        assert len(results) == 2
        assert results[0].name == "Mode Comparison: MIXED"
        assert results[1].name == "Mode Comparison: JINJA2"
        assert results[0].iterations == 10
        assert results[1].iterations == 10

    def test_benchmark_nested_conditionals(self, prompt_bench):
        """Test nested conditionals benchmark."""
        result = prompt_bench.benchmark_nested_conditionals()

        assert result.name == "Nested Conditionals (5 levels)"
        assert result.iterations == 10
        assert result.mean_time > 0

    def test_run_all(self, prompt_bench):
        """Test running all benchmarks."""
        results = prompt_bench.run_all()

        # Should have at least 8 results (7 individual + 2 from comparison)
        assert len(results) >= 9

        # All results should have valid data
        for result in results:
            assert result.iterations == 10
            assert result.mean_time > 0
            assert result.operations_per_second > 0
            assert result.std_dev >= 0

    def test_performance_consistency(self, prompt_bench):
        """Test that benchmarks produce consistent results across runs."""
        # Run same benchmark twice
        result1 = prompt_bench.benchmark_simple_rendering()
        result2 = prompt_bench.benchmark_simple_rendering()

        # Results should be similar (within 50% variance is reasonable for small samples)
        mean_ratio = result1.mean_time / result2.mean_time
        assert 0.5 < mean_ratio < 2.0, "Benchmark results are too inconsistent"

    def test_benchmark_iterations_respected(self):
        """Test that iteration count is respected."""
        benchmark = PromptBenchmark(iterations=5)
        result = benchmark.benchmark_simple_rendering()

        assert result.iterations == 5

    def test_benchmarks_dont_error(self, prompt_bench):
        """Test that all benchmarks complete without errors."""
        # This test ensures none of the benchmarks raise exceptions
        try:
            prompt_bench.benchmark_simple_rendering()
            prompt_bench.benchmark_conditional_rendering()
            prompt_bench.benchmark_jinja2_filters()
            prompt_bench.benchmark_jinja2_conditionals()
            prompt_bench.benchmark_jinja2_loops()
            prompt_bench.benchmark_complex_template()
            prompt_bench.benchmark_nested_conditionals()
            prompt_bench.benchmark_mixed_vs_jinja2_mode()
        except Exception as e:
            pytest.fail(f"Benchmark raised unexpected exception: {e}")

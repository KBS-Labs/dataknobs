<!-- markdownlint-disable MD013 -->
# Benchmarking Framework

## Overview

The DataKnobs LLM package includes a comprehensive benchmarking framework for measuring and tracking performance of:

- **Template rendering** - Variable substitution, conditionals, filters, loops
- **RAG search operations** - Search performance across different dataset sizes
- **Conversation management** - Message handling, branching, completion

## Quick Start

### Running All Benchmarks

```bash
cd packages/llm
python -m benchmarks.run_benchmarks
```

### Running Specific Benchmarks

```bash
# Only prompt rendering benchmarks
python -m benchmarks.run_benchmarks --prompts

# Only RAG search benchmarks
python -m benchmarks.run_benchmarks --rag

# Only conversation benchmarks
python -m benchmarks.run_benchmarks --conversations
```

### Saving Results

```bash
# Save markdown report
python -m benchmarks.run_benchmarks --output results.md

# Save JSON data
python -m benchmarks.run_benchmarks --json results.json

# Both
python -m benchmarks.run_benchmarks --output results.md --json results.json
```

### Custom Iterations

```bash
# Run more iterations for better accuracy
python -m benchmarks.run_benchmarks --iterations 10000

# Run fewer iterations for quick test
python -m benchmarks.run_benchmarks --iterations 100
```

## Using pytest-benchmark

The benchmarking framework integrates with pytest-benchmark for detailed analysis:

```bash
# Install dependencies
uv sync

# Run benchmark tests
uv run pytest tests/benchmarks/ --benchmark-only

# Compare with previous runs
uv run pytest tests/benchmarks/ --benchmark-compare

# Save benchmark results
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline

# Generate histogram
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-histogram
```

## Benchmark Categories

### 1. Prompt Rendering Benchmarks

Tests template rendering performance:

| Benchmark | Description | Iterations |
|-----------|-------------|------------|
| Simple Rendering | Basic {{variable}} substitution | 1000 |
| Conditional Rendering | (( )) conditional blocks | 1000 |
| Jinja2 Filters | Filter application (upper, truncate, etc.) | 1000 |
| Jinja2 Conditionals | {% if %} statements | 1000 |
| Jinja2 Loops | {% for %} loops over 10 items | 1000 |
| Complex Template | Multiple features combined | 1000 |
| Nested Conditionals | 5 levels of nested (( )) | 1000 |
| Mode Comparison | MIXED vs. JINJA2 mode | 1000 |

**Expected Performance:**
- Simple rendering: >50,000 ops/sec (<0.02ms)
- Conditional rendering: >20,000 ops/sec (<0.05ms)
- Complex templates: >10,000 ops/sec (<0.1ms)

### 2. RAG Search Benchmarks

Tests search performance across dataset sizes:

| Benchmark | Description | Iterations |
|-----------|-------------|------------|
| Small Dataset | 100 items | 100 |
| Medium Dataset | 1,000 items | 100 |
| Large Dataset | 10,000 items | 100 |
| k=1 | Return 1 result | 100 |
| k=5 | Return 5 results | 100 |
| k=10 | Return 10 results | 100 |
| k=20 | Return 20 results | 100 |
| Parallel Searches | 4 concurrent queries | 100 |

**Expected Performance:**
- Small dataset: >5,000 ops/sec (<0.2ms)
- Medium dataset: >2,000 ops/sec (<0.5ms)
- Large dataset: >500 ops/sec (<2ms)

### 3. Conversation Benchmarks

Tests conversation management operations:

| Benchmark | Description | Iterations |
|-----------|-------------|------------|
| Add Message | Add user message | 100 |
| Branch Creation | Create conversation branch | 50 |
| Switch Node | Switch between nodes | 100 |
| Complete | LLM completion (Echo provider) | 50 |
| Get Messages | Retrieve 10 messages | 100 |

**Expected Performance:**
- Add message: >1,000 ops/sec (<1ms)
- Switch node: >5,000 ops/sec (<0.2ms)
- Get messages: >10,000 ops/sec (<0.1ms)

## Programmatic Usage

### Running Benchmarks in Code

```python
from dataknobs_llm.benchmarks import (
    PromptBenchmark,
    RAGBenchmark,
    ConversationBenchmark
)

# Prompt benchmarks
prompt_bench = PromptBenchmark(iterations=1000)
results = prompt_bench.run_all()

# Print summary
for result in results:
    print(result.format_summary())

# RAG benchmarks
rag_bench = RAGBenchmark(iterations=100)
rag_results = rag_bench.run_all()

# Conversation benchmarks
conv_bench = ConversationBenchmark(iterations=100)
conv_results = conv_bench.run_all()
```

### Individual Benchmarks

```python
from dataknobs_llm.benchmarks import PromptBenchmark

bench = PromptBenchmark(iterations=1000)

# Run specific benchmark
result = bench.benchmark_simple_rendering()

print(f"Mean: {result.mean_time * 1000:.3f}ms")
print(f"Throughput: {result.operations_per_second:.0f} ops/sec")
print(f"Std Dev: {result.std_dev * 1000:.3f}ms")
```

### Custom Benchmarks

```python
import time
from dataknobs_llm.benchmarks import BenchmarkResult

def my_custom_benchmark(iterations: int = 1000):
    """Custom benchmark example."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Your code to benchmark
        result = expensive_operation()

        end = time.perf_counter()
        times.append(end - start)

    return BenchmarkResult.from_times("My Custom Benchmark", times)

# Run custom benchmark
result = my_custom_benchmark()
print(result.format_summary())
```

## Interpreting Results

### Benchmark Result Fields

```python
@dataclass
class BenchmarkResult:
    name: str                       # Benchmark name
    iterations: int                 # Number of iterations
    total_time: float              # Total time (seconds)
    mean_time: float               # Mean time per iteration (seconds)
    median_time: float             # Median time (seconds)
    std_dev: float                 # Standard deviation (seconds)
    min_time: float                # Minimum time (seconds)
    max_time: float                # Maximum time (seconds)
    operations_per_second: float   # Throughput (ops/sec)
```

### What to Look For

**Good Performance:**
- Low mean and median times
- Low standard deviation (consistent)
- High operations per second

**Performance Issues:**
- High mean/median times
- High standard deviation (inconsistent)
- Large gap between min and max

**Example:**

```
Simple Variable Substitution:
  Iterations: 1000
  Mean: 0.015ms          ✓ Very fast
  Median: 0.014ms        ✓ Consistent
  Std Dev: 0.003ms       ✓ Low variance
  Min: 0.012ms
  Max: 0.045ms
  Throughput: 66667 ops/sec  ✓ High throughput
```

## Performance Baselines

### Jinja2 Integration Impact

The Jinja2 integration adds minimal overhead:

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| Simple render | 0.015ms | 0.016ms | +6% |
| Conditional | 0.045ms | 0.048ms | +7% |
| Complex | 0.080ms | 0.085ms | +6% |

**Conclusion:** Jinja2 overhead is negligible (<10%) while providing 50+ filters and advanced features.

### Mixed vs. Jinja2 Mode

| Template | Mixed Mode | Jinja2 Mode | Difference |
|----------|------------|-------------|------------|
| With filters | 0.020ms | 0.018ms | -10% (faster) |
| With conditionals | 0.050ms | 0.045ms | -10% (faster) |

**Conclusion:** Pure Jinja2 mode is slightly faster (~10%) due to single rendering pass.

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: |
          cd packages/llm
          uv sync

      - name: Run benchmarks
        run: |
          cd packages/llm
          uv run python -m benchmarks.run_benchmarks \
            --iterations 1000 \
            --output benchmark-results.md \
            --json benchmark-results.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: packages/llm/benchmark-results.*

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = fs.readFileSync('packages/llm/benchmark-results.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '## Benchmark Results\n\n' + results
            });
```

### Performance Regression Detection

```python
# scripts/check_performance_regression.py
import json
import sys

def check_regression(baseline_file: str, current_file: str, threshold: float = 0.20):
    """Check for performance regressions.

    Args:
        baseline_file: Path to baseline JSON results
        current_file: Path to current JSON results
        threshold: Maximum allowed regression (e.g., 0.20 = 20%)

    Returns:
        Exit code 0 if no regression, 1 if regression detected
    """
    with open(baseline_file) as f:
        baseline = json.load(f)

    with open(current_file) as f:
        current = json.load(f)

    regressions = []

    for curr_bench in current['benchmarks']:
        name = curr_bench['name']

        # Find matching baseline
        baseline_bench = next(
            (b for b in baseline['benchmarks'] if b['name'] == name),
            None
        )

        if not baseline_bench:
            continue

        # Check mean time regression
        curr_mean = curr_bench['mean_time']
        base_mean = baseline_bench['mean_time']

        if curr_mean > base_mean * (1 + threshold):
            regression_pct = ((curr_mean - base_mean) / base_mean) * 100
            regressions.append({
                'name': name,
                'baseline': base_mean * 1000,
                'current': curr_mean * 1000,
                'regression': regression_pct
            })

    if regressions:
        print("Performance regressions detected:")
        for reg in regressions:
            print(f"  {reg['name']}: "
                  f"{reg['baseline']:.3f}ms → {reg['current']:.3f}ms "
                  f"(+{reg['regression']:.1f}%)")
        return 1

    print("No performance regressions detected.")
    return 0

if __name__ == "__main__":
    sys.exit(check_regression(sys.argv[1], sys.argv[2]))
```

## Troubleshooting

### Inconsistent Results

**Problem:** High standard deviation, large min/max gap.

**Solutions:**
1. Run more iterations: `--iterations 10000`
2. Close other applications
3. Disable CPU throttling
4. Run on dedicated hardware

### Slow Benchmarks

**Problem:** Benchmarks take too long.

**Solutions:**
1. Reduce iterations: `--iterations 100`
2. Run specific category: `--prompts` only
3. Use pytest-benchmark's `--benchmark-disable` for regular tests

### Out of Memory

**Problem:** RAG or conversation benchmarks OOM.

**Solution:** Reduce dataset size or iterations in benchmark code.

## Best Practices

1. **Establish Baselines** - Run benchmarks on main branch before changes
2. **Track Over Time** - Save results with timestamps
3. **Consistent Environment** - Run on same hardware/configuration
4. **Multiple Runs** - Run 3+ times and average for accurate results
5. **Isolate Changes** - Benchmark individual features separately
6. **Document Context** - Note system specs, Python version, dependencies

## Example Workflow

```bash
# 1. Establish baseline (before changes)
git checkout main
python -m benchmarks.run_benchmarks --json baseline.json

# 2. Make changes
git checkout feature-branch
# ... make code changes ...

# 3. Run benchmarks
python -m benchmarks.run_benchmarks --json current.json

# 4. Compare results
python scripts/check_performance_regression.py baseline.json current.json

# 5. If acceptable, commit
git add .
git commit -m "feat: Added feature X (benchmarks show <10% overhead)"
```

## Additional Resources

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Python performance profiling](https://docs.python.org/3/library/profile.html)
- [Jinja2 performance tips](https://jinja.palletsprojects.com/en/3.1.x/tricks/)

## FAQ

**Q: Why use custom benchmarking instead of just pytest-benchmark?**
A: Our framework provides domain-specific benchmarks with realistic workloads and easier programmatic access. pytest-benchmark is still available for detailed analysis.

**Q: What's a good target for template rendering performance?**
A: Simple templates should be >50k ops/sec. Complex templates >10k ops/sec. Anything above this is excellent for LLM prompt generation.

**Q: Should I optimize for the fastest benchmarks?**
A: Focus on the benchmarks that match your use case. If you're not using loops, loop performance doesn't matter.

**Q: How often should I run benchmarks?**
A: Run locally before committing performance-sensitive changes. Run in CI on every PR to catch regressions early.

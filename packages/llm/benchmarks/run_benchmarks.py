#!/usr/bin/env python
"""Benchmark runner script.

Usage:
    python -m benchmarks.run_benchmarks [OPTIONS]

Options:
    --iterations N      Number of iterations per benchmark (default: 1000)
    --prompts          Run only prompt rendering benchmarks
    --rag              Run only RAG search benchmarks
    --conversations    Run only conversation benchmarks
    --output FILE      Save results to markdown file
    --json FILE        Save results to JSON file
"""

import argparse
import json
import sys
from datetime import datetime
from typing import List

from .benchmark_result import BenchmarkResult
from .prompt_benchmark import PromptBenchmark
from .rag_benchmark import RAGBenchmark

try:
    from .conversation_benchmark import ConversationBenchmark
    CONVERSATION_AVAILABLE = True
except ImportError:
    ConversationBenchmark = None
    CONVERSATION_AVAILABLE = False


def format_markdown_report(results: List[BenchmarkResult], title: str = "Benchmark Results") -> str:
    """Format benchmark results as a markdown report.

    Args:
        results: List of BenchmarkResult objects
        title: Report title

    Returns:
        Formatted markdown string
    """
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Benchmark | Ops/Sec | Mean (ms) | Median (ms) | Std Dev (ms) |",
        "|-----------|---------|-----------|-------------|--------------|",
    ]

    for result in results:
        lines.append(result.format_table_row())

    lines.extend([
        "",
        "## Detailed Results",
        ""
    ])

    for result in results:
        lines.extend([
            f"### {result.name}",
            "",
            f"- **Iterations**: {result.iterations}",
            f"- **Throughput**: {result.operations_per_second:.0f} ops/sec",
            f"- **Mean**: {result.mean_time * 1000:.3f}ms",
            f"- **Median**: {result.median_time * 1000:.3f}ms",
            f"- **Std Dev**: {result.std_dev * 1000:.3f}ms",
            f"- **Min**: {result.min_time * 1000:.3f}ms",
            f"- **Max**: {result.max_time * 1000:.3f}ms",
            ""
        ])

    return "\n".join(lines)


def save_json_results(results: List[BenchmarkResult], filepath: str):
    """Save benchmark results as JSON.

    Args:
        results: List of BenchmarkResult objects
        filepath: Path to output JSON file
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": [
            {
                "name": r.name,
                "iterations": r.iterations,
                "total_time": r.total_time,
                "mean_time": r.mean_time,
                "median_time": r.median_time,
                "std_dev": r.std_dev,
                "min_time": r.min_time,
                "max_time": r.max_time,
                "operations_per_second": r.operations_per_second
            }
            for r in results
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filepath}")


def main():
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(description="Run DataKnobs LLM benchmarks")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations per benchmark (default: 1000)"
    )
    parser.add_argument(
        "--prompts",
        action="store_true",
        help="Run only prompt rendering benchmarks"
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Run only RAG search benchmarks"
    )
    parser.add_argument(
        "--conversations",
        action="store_true",
        help="Run only conversation benchmarks"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to markdown file"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Determine which benchmarks to run
    run_all = not (args.prompts or args.rag or args.conversations)

    all_results = []

    print("=" * 70)
    print("DataKnobs LLM Benchmark Suite")
    print("=" * 70)
    print()

    # Run prompt benchmarks
    if run_all or args.prompts:
        try:
            prompt_bench = PromptBenchmark(iterations=args.iterations)
            prompt_results = prompt_bench.run_all()
            all_results.extend(prompt_results)
            print()
        except Exception as e:
            print(f"Error running prompt benchmarks: {e}")
            sys.exit(1)

    # Run RAG benchmarks
    if run_all or args.rag:
        try:
            # Use fewer iterations for RAG (more expensive)
            rag_iterations = min(args.iterations // 10, 100)
            rag_bench = RAGBenchmark(iterations=rag_iterations)
            rag_results = rag_bench.run_all()
            all_results.extend(rag_results)
            print()
        except Exception as e:
            print(f"Error running RAG benchmarks: {e}")
            sys.exit(1)

    # Run conversation benchmarks (if available)
    if run_all or args.conversations:
        if not CONVERSATION_AVAILABLE:
            print("⚠️  Conversation benchmarks not available (requires dataknobs-common)")
            print()
        else:
            try:
                # Use fewer iterations for conversations (more expensive)
                conv_iterations = min(args.iterations // 10, 100)
                conv_bench = ConversationBenchmark(iterations=conv_iterations)
                conv_results = conv_bench.run_all()
                all_results.extend(conv_results)
                print()
            except Exception as e:
                print(f"Error running conversation benchmarks: {e}")
                sys.exit(1)

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print(f"{'Benchmark':<50} {'Ops/Sec':>10} {'Mean (ms)':>10}")
    print("-" * 70)
    for result in all_results:
        print(f"{result.name:<50} {result.operations_per_second:>10.0f} {result.mean_time * 1000:>10.2f}")

    print()

    # Save results if requested
    if args.output:
        report = format_markdown_report(all_results)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Markdown report saved to {args.output}")

    if args.json:
        save_json_results(all_results, args.json)

    print()
    print("Benchmarks complete!")


if __name__ == "__main__":
    main()

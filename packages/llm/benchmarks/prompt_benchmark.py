"""Benchmarks for prompt template rendering performance."""

import time
from typing import List

from dataknobs_llm.prompts.rendering.template_renderer import TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode

try:
    from .benchmark_result import BenchmarkResult
except ImportError:
    from benchmark_result import BenchmarkResult


class PromptBenchmark:
    """Benchmark prompt rendering performance.

    This class provides benchmarks for:
    - Simple variable substitution
    - Conditional rendering with (( ))
    - Jinja2 filters
    - Jinja2 conditionals and loops
    - Complex templates with multiple features
    - Mixed mode vs. pure Jinja2 mode
    """

    def __init__(self, iterations: int = 1000):
        """Initialize prompt benchmark.

        Args:
            iterations: Number of iterations to run for each benchmark
        """
        self.iterations = iterations
        self.renderer = TemplateRenderer()

    def benchmark_simple_rendering(self) -> BenchmarkResult:
        """Benchmark simple variable substitution.

        Template: "Hello {{name}}, you are {{age}} years old"
        """
        template = "Hello {{name}}, you are {{age}} years old"
        params = {"name": "Alice", "age": 30}

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template, params)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Simple Variable Substitution", times)

    def benchmark_conditional_rendering(self) -> BenchmarkResult:
        """Benchmark conditional template rendering with (( )).

        Template: "Hello {{name}}((, age {{age}}))((, from {{city}}))"
        """
        template = "Hello {{name}}((, age {{age}}))((, from {{city}}))"
        params = {"name": "Alice", "age": 30, "city": "NYC"}

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template, params, mode=TemplateMode.MIXED)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Conditional Rendering (( ))", times)

    def benchmark_jinja2_filters(self) -> BenchmarkResult:
        """Benchmark Jinja2 filter application.

        Template: "{{name|upper}} - {{text|truncate(50)}}"
        """
        template = "{{name|upper}} - {{text|truncate(50)}}"
        params = {
            "name": "alice",
            "text": "This is a longer text that needs to be truncated to fit"
        }

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template, params, mode=TemplateMode.JINJA2)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Jinja2 Filters", times)

    def benchmark_jinja2_conditionals(self) -> BenchmarkResult:
        """Benchmark Jinja2 {% if %} conditionals.

        Template: "{% if age >= 18 %}Adult{% else %}Minor{% endif %}"
        """
        template = "{% if age >= 18 %}Adult{% else %}Minor{% endif %}"
        params = {"age": 25}

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template, params, mode=TemplateMode.JINJA2)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Jinja2 Conditionals", times)

    def benchmark_jinja2_loops(self) -> BenchmarkResult:
        """Benchmark Jinja2 {% for %} loops.

        Template with loop over 10 items.
        """
        template = """
        {% for item in items %}
        {{ loop.index }}. {{ item.name }} - {{ item.value }}
        {% endfor %}
        """
        params = {
            "items": [
                {"name": f"Item {i}", "value": i * 10}
                for i in range(10)
            ]
        }

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template, params, mode=TemplateMode.JINJA2)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Jinja2 Loops (10 items)", times)

    def benchmark_complex_template(self) -> BenchmarkResult:
        """Benchmark complex template with many variables and features.

        Template with nested objects, multiple conditionals, and filters.
        """
        template = """
        User: {{user.name|upper}}
        Email: {{user.email}}
        Location: {{user.city}}, {{user.country}}
        ((Preferences: Theme={{prefs.theme}}, Language={{prefs.language}}))
        {% if user.premium %}⭐ Premium Member{% endif %}
        Settings: {{settings.notifications}}, {{settings.privacy}}
        """
        params = {
            "user": {
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "city": "New York",
                "country": "USA",
                "premium": True
            },
            "prefs": {"theme": "dark", "language": "en"},
            "settings": {"notifications": "on", "privacy": "strict"}
        }

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template, params, mode=TemplateMode.MIXED)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Complex Template (Mixed)", times)

    def benchmark_mixed_vs_jinja2_mode(self) -> List[BenchmarkResult]:
        """Compare performance of mixed vs. jinja2 mode.

        Returns:
            List of two BenchmarkResults for comparison
        """
        template_mixed = "{{name|upper}}((, age {{age}}))"
        template_jinja2 = "{{name|upper}}{% if age %}, age {{age}}{% endif %}"
        params = {"name": "alice", "age": 30}

        # Mixed mode
        times_mixed = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template_mixed, params, mode=TemplateMode.MIXED)
            end = time.perf_counter()
            times_mixed.append(end - start)

        # Jinja2 mode
        times_jinja2 = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template_jinja2, params, mode=TemplateMode.JINJA2)
            end = time.perf_counter()
            times_jinja2.append(end - start)

        return [
            BenchmarkResult.from_times("Mode Comparison: MIXED", times_mixed),
            BenchmarkResult.from_times("Mode Comparison: JINJA2", times_jinja2)
        ]

    def benchmark_nested_conditionals(self) -> BenchmarkResult:
        """Benchmark deeply nested conditional blocks.

        Template: "{{a}}(({{b}}(({{c}}(({{d}}(({{e}})))))))))"
        """
        template = "{{a}}(({{b}}(({{c}}(({{d}}(({{e}})))))))))"
        params = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            self.renderer.render(template, params, mode=TemplateMode.MIXED)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Nested Conditionals (5 levels)", times)

    def run_all(self) -> List[BenchmarkResult]:
        """Run all prompt rendering benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        print(f"Running prompt rendering benchmarks ({self.iterations} iterations)...")
        print()

        benchmarks = [
            ("Simple Rendering", self.benchmark_simple_rendering),
            ("Conditional Rendering", self.benchmark_conditional_rendering),
            ("Jinja2 Filters", self.benchmark_jinja2_filters),
            ("Jinja2 Conditionals", self.benchmark_jinja2_conditionals),
            ("Jinja2 Loops", self.benchmark_jinja2_loops),
            ("Complex Template", self.benchmark_complex_template),
            ("Nested Conditionals", self.benchmark_nested_conditionals),
        ]

        results = []
        for name, benchmark_func in benchmarks:
            print(f"Running {name}...")
            result = benchmark_func()
            results.append(result)
            print(f"  {result.operations_per_second:.0f} ops/sec "
                  f"({result.mean_time * 1000:.3f}ms mean)")

        # Mode comparison
        print("Running Mode Comparison...")
        comparison_results = self.benchmark_mixed_vs_jinja2_mode()
        results.extend(comparison_results)
        for result in comparison_results:
            print(f"  {result.name}: {result.operations_per_second:.0f} ops/sec "
                  f"({result.mean_time * 1000:.3f}ms mean)")

        print()
        print("Prompt rendering benchmarks complete!")
        return results

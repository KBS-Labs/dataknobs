"""Pytest-benchmark integration for prompt rendering benchmarks.

Run with:
    pytest tests/benchmarks/test_prompt_benchmarks.py --benchmark-only
    pytest tests/benchmarks/test_prompt_benchmarks.py --benchmark-compare
"""

import pytest
from dataknobs_llm.prompts.rendering.template_renderer import TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode


@pytest.fixture
def renderer():
    """Create template renderer for benchmarks."""
    return TemplateRenderer()


def test_benchmark_simple_rendering(benchmark, renderer):
    """Benchmark simple variable substitution."""
    template = "Hello {{name}}, you are {{age}} years old"
    params = {"name": "Alice", "age": 30}

    result = benchmark(renderer.render, template, params)
    assert "Alice" in result.content


def test_benchmark_conditional_rendering(benchmark, renderer):
    """Benchmark conditional template rendering with (( ))."""
    template = "Hello {{name}}((, age {{age}}))((, from {{city}}))"
    params = {"name": "Alice", "age": 30, "city": "NYC"}

    result = benchmark(renderer.render, template, params, mode=TemplateMode.MIXED)
    assert "Alice" in result.content


def test_benchmark_jinja2_filters(benchmark, renderer):
    """Benchmark Jinja2 filter application."""
    template = "{{name|upper}} - {{text|truncate(50)}}"
    params = {
        "name": "alice",
        "text": "This is a longer text that needs to be truncated to fit"
    }

    result = benchmark(renderer.render, template, params, mode=TemplateMode.JINJA2)
    assert "ALICE" in result.content


def test_benchmark_jinja2_conditionals(benchmark, renderer):
    """Benchmark Jinja2 {% if %} conditionals."""
    template = "{% if age >= 18 %}Adult{% else %}Minor{% endif %}"
    params = {"age": 25}

    result = benchmark(renderer.render, template, params, mode=TemplateMode.JINJA2)
    assert "Adult" in result.content


def test_benchmark_jinja2_loops(benchmark, renderer):
    """Benchmark Jinja2 {% for %} loops."""
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

    result = benchmark(renderer.render, template, params, mode=TemplateMode.JINJA2)
    assert "Item 0" in result.content


def test_benchmark_complex_template(benchmark, renderer):
    """Benchmark complex template with many variables and features."""
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

    result = benchmark(renderer.render, template, params, mode=TemplateMode.MIXED)
    assert "ALICE JOHNSON" in result.content


def test_benchmark_mixed_mode(benchmark, renderer):
    """Benchmark mixed mode rendering."""
    template = "{{name|upper}}((, age {{age}}))"
    params = {"name": "alice", "age": 30}

    result = benchmark(renderer.render, template, params, mode=TemplateMode.MIXED)
    assert "ALICE" in result.content


def test_benchmark_jinja2_mode(benchmark, renderer):
    """Benchmark pure Jinja2 mode rendering."""
    template = "{{name|upper}}{% if age %}, age {{age}}{% endif %}"
    params = {"name": "alice", "age": 30}

    result = benchmark(renderer.render, template, params, mode=TemplateMode.JINJA2)
    assert "ALICE" in result.content


def test_benchmark_nested_conditionals(benchmark, renderer):
    """Benchmark deeply nested conditional blocks."""
    template = "{{a}}(({{b}}(({{c}}(({{d}}(({{e}})))))))))"
    params = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}

    result = benchmark(renderer.render, template, params, mode=TemplateMode.MIXED)
    assert "ABCDE" in result.content

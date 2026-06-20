"""Behavior tests for JinjaInputsProjector."""

from __future__ import annotations

import pytest

# Skip the entire module if jinja2 is unavailable.
jinja2 = pytest.importorskip("jinja2")

from dataknobs_bots.prompts.scope import JinjaInputsProjector  # noqa: E402


def test_evaluates_single_jinja_expression() -> None:
    proj = JinjaInputsProjector(
        inputs={"upper": "x | upper"},
        base_context={"x": "hello"},
    )
    assert proj.project(None) == {"upper": "HELLO"}


def test_evaluates_multiple_expressions() -> None:
    proj = JinjaInputsProjector(
        inputs={
            "upper": "x | upper",
            "length": "x | length",
        },
        base_context={"x": "abc"},
    )
    result = proj.project(None)
    assert result == {"upper": "ABC", "length": 3}


def test_empty_inputs_returns_empty() -> None:
    proj = JinjaInputsProjector(
        inputs={},
        base_context={"x": "anything"},
    )
    assert proj.project(None) == {}


def test_uses_custom_env_when_supplied() -> None:
    env = jinja2.Environment(autoescape=False)
    env.filters["double"] = lambda x: x * 2
    proj = JinjaInputsProjector(
        inputs={"d": "n | double"},
        base_context={"n": 5},
        env=env,
    )
    assert proj.project(None) == {"d": 10}


def test_constructor_captures_inputs_and_context() -> None:
    """Mutations to the inputs / base_context after construction do not
    affect projections.
    """
    inputs = {"k": "x"}
    base_context = {"x": "original"}
    proj = JinjaInputsProjector(
        inputs=inputs,
        base_context=base_context,
    )
    inputs["k2"] = "y"  # not seen by projector
    base_context["x"] = "modified"  # not seen by projector
    result = proj.project(None)
    assert result == {"k": "original"}


def test_source_argument_is_ignored() -> None:
    proj = JinjaInputsProjector(
        inputs={"k": "x"},
        base_context={"x": 42},
    )
    # source argument doesn't affect output
    assert proj.project("ignored") == {"k": 42}
    assert proj.project(None) == {"k": 42}
    assert proj.project({"different": "source"}) == {"k": 42}


def test_conforms_to_scope_projector_protocol() -> None:
    from dataknobs_common.scope import ScopeProjector

    proj = JinjaInputsProjector(inputs={}, base_context={})
    assert isinstance(proj, ScopeProjector)

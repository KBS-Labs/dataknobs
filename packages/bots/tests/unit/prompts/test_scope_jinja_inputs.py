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


# --- Security: default environment is sandboxed ----------------------- #

def test_default_env_blocks_sandbox_escape() -> None:
    """With no env= supplied the default is a SandboxedEnvironment, so an
    attribute-traversal SSTI payload raises instead of leaking interpreter
    internals.
    """
    proj = JinjaInputsProjector(
        inputs={"escape": "().__class__.__bases__"},
        base_context={},
    )
    with pytest.raises(jinja2.exceptions.TemplateError):
        proj.project(None)


def test_explicit_unsandboxed_env_is_caller_opt_in() -> None:
    """A caller who explicitly passes a plain (unsandboxed) Environment
    gets unsandboxed behavior — the unsafe choice is opt-in, not the
    default. This documents the contrast with the sandboxed default above.
    """
    unsafe_env = jinja2.Environment(autoescape=False)
    proj = JinjaInputsProjector(
        inputs={"bases": "().__class__.__bases__"},
        base_context={},
        env=unsafe_env,
    )
    # Plain env leaks; the point is the default does NOT (test above).
    assert proj.project(None)["bases"] == (object,)


# --- Error handling: strict vs graceful degradation ------------------- #

def test_strict_true_propagates_failing_expression() -> None:
    """Default strict=True: a failing expression propagates."""
    proj = JinjaInputsProjector(
        inputs={"bad": "n | length"},  # length of an int -> TypeError
        base_context={"n": 5},
    )
    with pytest.raises(TypeError):
        proj.project(None)


def test_strict_false_skips_failing_expression() -> None:
    """strict=False: a failing expression is skipped (omitted) while the
    other inputs still evaluate.
    """
    proj = JinjaInputsProjector(
        inputs={
            "good": "n | string",
            "bad": "n | length",  # TypeError on an int
        },
        base_context={"n": 5},
        strict=False,
    )
    result = proj.project(None)
    assert result == {"good": "5"}  # "bad" omitted, "good" survives


def test_strict_false_skips_sandbox_escape() -> None:
    """strict=False degrades on a sandbox violation too — the escape is
    both blocked and skipped, not raised.
    """
    proj = JinjaInputsProjector(
        inputs={"escape": "().__class__.__bases__"},
        base_context={},
        strict=False,
    )
    assert proj.project(None) == {}

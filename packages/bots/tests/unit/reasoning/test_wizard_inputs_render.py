"""End-to-end render-wiring tests for declarative stage ``inputs:``.

A wizard stage may declare an ``inputs:`` mapping of
``name -> Jinja expression``. The renderer evaluates each expression
against the assembled template context and merges the derived variables
into the template scope (later-wins), so response templates can reference
computed values without a consumer subclassing the renderer.

These tests drive the real ``WizardRenderer`` with a minimal state stub
(matching the fields the renderer reads) — a legitimate unit test of the
renderer's context-construction logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_renderer import WizardRenderer


@dataclass
class _StubState:
    current_stage: str = "gather"
    data: dict[str, Any] = field(default_factory=dict)
    transient: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=lambda: ["gather"])
    completed: bool = False


@pytest.fixture()
def renderer() -> WizardRenderer:
    return WizardRenderer()


def test_inputs_populate_context(renderer: WizardRenderer) -> None:
    stage = {
        "name": "gather",
        "inputs": {"shout": "topic | upper"},
    }
    state = _StubState(data={"topic": "python"})
    ctx = renderer.build_context(stage, state)
    assert ctx["shout"] == "PYTHON"


def test_inputs_available_in_response_template(
    renderer: WizardRenderer,
) -> None:
    stage = {
        "name": "gather",
        "inputs": {
            "sanitized": "raw | lower | trim",
            "length": "raw | length",
        },
    }
    state = _StubState(data={"raw": "  HELLO  "})
    rendered = renderer.render(
        "{{ sanitized }} has {{ length }} chars",
        stage,
        state,
    )
    assert rendered == "hello has 9 chars"


def test_inputs_can_reference_multiple_context_vars(
    renderer: WizardRenderer,
) -> None:
    stage = {
        "name": "gather",
        "inputs": {"full": "first ~ ' ' ~ last"},
    }
    state = _StubState(data={"first": "Ada", "last": "Lovelace"})
    ctx = renderer.build_context(stage, state)
    assert ctx["full"] == "Ada Lovelace"


def test_inputs_override_collected_data_on_collision(
    renderer: WizardRenderer,
) -> None:
    """Declared inputs win over base context on key collision
    (ChainedProjector later-wins).
    """
    stage = {
        "name": "gather",
        "inputs": {"topic": "topic | upper"},
    }
    state = _StubState(data={"topic": "python"})
    ctx = renderer.build_context(stage, state)
    assert ctx["topic"] == "PYTHON"


def test_no_inputs_leaves_context_unchanged(
    renderer: WizardRenderer,
) -> None:
    stage = {"name": "gather"}
    state = _StubState(data={"topic": "python"})
    ctx = renderer.build_context(stage, state)
    assert ctx["topic"] == "python"
    assert "shout" not in ctx


def test_empty_inputs_mapping_is_noop(renderer: WizardRenderer) -> None:
    stage = {"name": "gather", "inputs": {}}
    state = _StubState(data={"topic": "python"})
    ctx = renderer.build_context(stage, state)
    assert ctx["topic"] == "python"


def test_inputs_see_extra_context(renderer: WizardRenderer) -> None:
    """Inputs are evaluated after extra_context merges, so they can
    reference bank/artifact/LLM-supplied values.
    """
    stage = {
        "name": "gather",
        "inputs": {"greeting": "'Hi ' ~ name"},
    }
    state = _StubState(data={})
    ctx = renderer.build_context(
        stage, state, extra_context={"name": "Sam"},
    )
    assert ctx["greeting"] == "Hi Sam"

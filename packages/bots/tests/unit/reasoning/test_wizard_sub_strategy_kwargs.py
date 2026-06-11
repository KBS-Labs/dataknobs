"""Tests for kwargs forwarding from WizardReasoning to per-stage
sub-strategies.

Verifies that construction kwargs supplied to
``WizardReasoning.from_config`` land on the mixin's ``self.components``
mapping and are forwarded to per-stage sub-strategy ``from_config``
via ``_resolve_stage_strategy``.

Pinned contracts:

- ``knowledge_base`` (and any other ``**kwargs``) supplied to
  ``WizardReasoning.from_config`` reach the sub-strategy.
- The wizard's own consumed collaborator ``wizard_fsm`` is NEVER
  forwarded to sub-strategies (declared in ``INTERNAL_COMPONENTS``).
- A strict-signature sub-strategy (no ``**kwargs`` absorber) raises a
  clear error documenting the per-stage-safe convention.
"""
from __future__ import annotations

from typing import Any, ClassVar

import pytest

from dataknobs_bots.reasoning import WizardReasoning
from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.reasoning.registry import (
    get_registry,
    register_strategy,
)


class _CaptureStrategy(ReasoningStrategy):
    """Test strategy: records the kwargs ``from_config`` received."""

    last_kwargs: ClassVar[dict[str, Any]] = {}

    @classmethod
    def from_config(
        cls, config: dict[str, Any], **kwargs: Any
    ) -> "_CaptureStrategy":
        cls.last_kwargs = dict(kwargs)
        return cls(greeting_template=None)

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        return None


class _StrictStrategy(ReasoningStrategy):
    """Test strategy: strict ``from_config`` signature with no ``**kwargs``.

    Any kwarg forwarded by the wizard surfaces as a clear ``TypeError``
    (documenting the per-stage-safe convention: strategies that may be
    used as wizard sub-strategies should declare ``**kwargs``).
    """

    @classmethod
    def from_config(  # type: ignore[override]
        cls, config: dict[str, Any]
    ) -> "_StrictStrategy":
        return cls(greeting_template=None)

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        return None


@pytest.fixture(autouse=True)
def _register_test_strategies() -> None:
    register_strategy("capture", _CaptureStrategy, override=True)
    register_strategy("strict", _StrictStrategy, override=True)
    _CaptureStrategy.last_kwargs = {}


def _build_wizard_with_capture_stage(
    **from_config_kwargs: Any,
) -> WizardReasoning:
    config = {
        "wizard_config": {
            "name": "kwargs-forwarding-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "only",
                    "is_start": True,
                    "is_end": True,
                    "reasoning": "capture",
                    "prompt": "noop",
                },
            ],
        },
    }
    return WizardReasoning.from_config(config, **from_config_kwargs)


# ---------------------------------------------------------------------------
# Reproducing pin: knowledge_base reaches the sub-strategy
# ---------------------------------------------------------------------------


def test_wizard_forwards_knowledge_base_kwarg_to_sub_strategy() -> None:
    """RED before fix, GREEN after: knowledge_base kwarg threads through."""
    sentinel = object()
    wizard = _build_wizard_with_capture_stage(knowledge_base=sentinel)

    strategy = wizard._resolve_stage_strategy(
        {"reasoning": "capture", "name": "only"}
    )

    assert isinstance(strategy, _CaptureStrategy)
    assert _CaptureStrategy.last_kwargs.get("knowledge_base") is sentinel


# ---------------------------------------------------------------------------
# Back-compat: no extras → empty forwarding (no-op)
# ---------------------------------------------------------------------------


def test_no_kwargs_forwards_empty_no_op() -> None:
    wizard = _build_wizard_with_capture_stage()

    wizard._resolve_stage_strategy(
        {"reasoning": "capture", "name": "only"}
    )

    assert _CaptureStrategy.last_kwargs == {}


# ---------------------------------------------------------------------------
# Mixin surface: self.components exposes forwarded collaborators
# ---------------------------------------------------------------------------


def test_self_components_exposes_forwarded_collaborators() -> None:
    """Pins the documented pass-through surface that consumer composing
    strategies are expected to leverage.

    After ``from_config(config, knowledge_base=kb, prompt_resolver=pr)``,
    the wizard's ``self.components`` MUST contain both keys (plus
    ``wizard_fsm`` for the wizard's own consumption).
    """
    kb = object()
    pr = object()
    wizard = _build_wizard_with_capture_stage(
        knowledge_base=kb,
        prompt_resolver=pr,
    )

    assert wizard.components["knowledge_base"] is kb
    assert wizard.components["prompt_resolver"] is pr
    assert "wizard_fsm" in wizard.components


def test_forwardable_components_excludes_wizard_internal() -> None:
    """The wizard does NOT forward its own consumed collaborator
    (``wizard_fsm``) to sub-strategies — sub-strategies must never
    receive an outer wizard's FSM handle.
    """
    kb = object()
    wizard = _build_wizard_with_capture_stage(knowledge_base=kb)

    wizard._resolve_stage_strategy(
        {"reasoning": "capture", "name": "only"}
    )

    assert _CaptureStrategy.last_kwargs.get("knowledge_base") is kb
    assert "wizard_fsm" not in _CaptureStrategy.last_kwargs


# ---------------------------------------------------------------------------
# Multi-kwarg simultaneous forwarding
# ---------------------------------------------------------------------------


def test_forwards_multiple_kwargs_simultaneously() -> None:
    kb = object()
    pr = object()
    extra = object()
    wizard = _build_wizard_with_capture_stage(
        knowledge_base=kb,
        prompt_resolver=pr,
        custom_collaborator=extra,
    )

    wizard._resolve_stage_strategy(
        {"reasoning": "capture", "name": "only"}
    )

    assert _CaptureStrategy.last_kwargs.get("knowledge_base") is kb
    assert _CaptureStrategy.last_kwargs.get("prompt_resolver") is pr
    assert _CaptureStrategy.last_kwargs.get("custom_collaborator") is extra


# ---------------------------------------------------------------------------
# Strict-signature sub-strategy: clear error documents contract
# ---------------------------------------------------------------------------


def test_strict_from_config_raises_when_forwarded_unknown_kwarg() -> None:
    """Per-stage-safe strategies should absorb ``**kwargs``. A strict
    signature surfaces a clear error when the wizard forwards a key
    the strategy doesn't accept. The wizard wraps the underlying
    ``TypeError`` as a ``ConfigurationError`` (see
    ``_resolve_stage_strategy``'s except branch).
    """
    wizard = _build_wizard_with_capture_stage(knowledge_base=object())

    with pytest.raises(Exception) as exc_info:
        wizard._resolve_stage_strategy(
            {"reasoning": "strict", "name": "only"}
        )

    msg = str(exc_info.value).lower()
    assert "knowledge_base" in msg or "unexpected" in msg


# ---------------------------------------------------------------------------
# Sanity: registry still exposes _CaptureStrategy and _StrictStrategy
# ---------------------------------------------------------------------------


def test_test_strategies_are_registered() -> None:
    """Sanity check: autouse fixture registered both strategies."""
    assert get_registry().get_factory("capture") is _CaptureStrategy
    assert get_registry().get_factory("strict") is _StrictStrategy

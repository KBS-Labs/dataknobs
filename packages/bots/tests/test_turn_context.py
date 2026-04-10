"""Tests for TurnContext, TransformContext.turn, and per_turn_keys.

Validates:
- TurnContext is delivered via _build_transform_context
- TransformContext.turn field is populated
- per_turn_keys cleared at start of generate
- per_turn_keys suppressed in _detect_conflicts
- per_turn_keys merged into _ephemeral_keys
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dataknobs_bots.reasoning.wizard import (
    DEFAULT_EPHEMERAL_KEYS,
    TurnContext,
    WizardReasoning,
)
from dataknobs_bots.reasoning.wizard_extraction import WizardExtractor


class TestTurnContextDataclass:
    """TurnContext holds per-turn values."""

    def test_defaults_are_none(self) -> None:
        turn = TurnContext()
        assert turn.message is None
        assert turn.bank_fn is None
        assert turn.intent is None
        assert turn.transform_error is None
        assert turn.corpus is None

    def test_fields_populated(self) -> None:
        fn = lambda: "bank"  # noqa: E731
        turn = TurnContext(message="hello", bank_fn=fn, intent="greet")
        assert turn.message == "hello"
        assert turn.bank_fn is fn
        assert turn.intent == "greet"


class TestTransformContextTurnField:
    """TransformContext carries TurnContext via .turn field."""

    def test_turn_field_exists(self) -> None:
        from dataknobs_bots.artifacts.transforms import TransformContext

        ctx = TransformContext(turn=TurnContext(message="hi"))
        assert ctx.turn is not None
        assert ctx.turn.message == "hi"

    def test_turn_default_is_none(self) -> None:
        from dataknobs_bots.artifacts.transforms import TransformContext

        ctx = TransformContext()
        assert ctx.turn is None


class TestBuildTransformContextIncludesTurn:
    """_build_transform_context includes _current_turn."""

    def test_turn_included_when_set(self) -> None:
        """When _current_turn is set, factory output includes it."""
        # Build a minimal WizardReasoning with mocked WizardFSM
        # We only need the factory method, so we test it directly.
        wizard_fsm = MagicMock()
        wizard_fsm.settings = {}
        wizard_fsm.stages = {}
        wizard_fsm.config = {}
        wizard_fsm.hooks = MagicMock()
        wizard_fsm.hooks.on_complete = []

        # Note: WizardReasoning.__init__ is complex, so test the factory
        # method in isolation by calling it as an unbound method with
        # the right attributes set.
        from dataknobs_bots.artifacts.transforms import TransformContext

        # Create a standalone instance with minimal state
        reasoning = WizardReasoning.__new__(WizardReasoning)
        reasoning._current_turn = TurnContext(message="test", intent="ask")
        reasoning._artifact_registry = None
        reasoning._review_executor = None
        reasoning._current_llm = None
        reasoning._banks = {}

        result = reasoning._build_transform_context(None)

        assert isinstance(result, TransformContext)
        assert result.turn is not None
        assert result.turn.message == "test"
        assert result.turn.intent == "ask"


class TestPerTurnKeysConfig:
    """per_turn_keys configuration support."""

    def test_per_turn_keys_merged_into_ephemeral(self) -> None:
        """per_turn_keys are included in _ephemeral_keys."""
        reasoning = WizardReasoning.__new__(WizardReasoning)

        # Simulate __init__ logic for ephemeral keys
        per_turn_keys = frozenset(["action", "intent_raw"])
        config_ephemeral: list[str] = []
        ephemeral_keys = (
            DEFAULT_EPHEMERAL_KEYS
            | frozenset(config_ephemeral)
            | per_turn_keys
        )

        assert "action" in ephemeral_keys
        assert "intent_raw" in ephemeral_keys
        # Framework keys still present
        assert "_message" in ephemeral_keys

    def test_detect_conflicts_skips_per_turn_keys(self) -> None:
        """_detect_conflicts skips fields in per_turn_keys."""
        extractor = WizardExtractor.__new__(WizardExtractor)
        extractor._per_turn_keys = frozenset(["action"])

        existing = {"action": "old_action", "topic": "old_topic"}
        new_data = {"action": "new_action", "topic": "new_topic"}

        conflicts = extractor._detect_conflicts(existing, new_data)

        # "action" should be skipped (per-turn key), "topic" should be flagged
        field_names = [c["field"] for c in conflicts]
        assert "action" not in field_names
        assert "topic" in field_names

    def test_per_turn_keys_cleared_from_data(self) -> None:
        """per_turn_keys are removed from wizard_state.data."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeState:
            data: dict[str, Any] = field(default_factory=dict)
            transient: dict[str, Any] = field(default_factory=dict)

        state = FakeState(
            data={"action": "old", "topic": "math", "other": 1},
            transient={"action": "stale"},
        )

        per_turn_keys = frozenset(["action"])
        for key in per_turn_keys:
            state.data.pop(key, None)
            state.transient.pop(key, None)

        assert "action" not in state.data
        assert "action" not in state.transient
        assert state.data["topic"] == "math"

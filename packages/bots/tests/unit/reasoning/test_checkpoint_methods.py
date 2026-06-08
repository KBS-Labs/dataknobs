"""Tests for ``ReasoningStrategy.restore_from_checkpoint`` and
``ReasoningStrategy.undo_to_checkpoint`` (161-B and 161-C).

Strategy-class layer coverage:
- The base ``ReasoningStrategy`` default implementations are no-ops.
- ``WizardReasoning`` overrides both — the bucket-restore writes and
  the per-bank undo loop live here now (moved from ``DynaBot``).

End-to-end coverage for the bot's ``undo_last_turn`` integration is
preserved in ``tests/unit/test_dynabot_undo.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


# =====================================================================
# Helpers
# =====================================================================


class _NoOpStrategy(ReasoningStrategy):
    """Minimal concrete strategy exercising the base no-op defaults."""

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError


class _StubManager:
    """Minimal manager exposing ``metadata`` for restore tests."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}


def _wizard_config(
    *, with_banks: bool = False, with_history: bool = False
) -> dict[str, Any]:
    """Build a minimal valid wizard config, optionally with banks."""
    config: dict[str, Any] = {
        "name": "checkpoint-test-wizard",
        "version": "1.0",
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "prompt": "Hello",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "transitions": [{"target": "done"}],
            },
            {"name": "done", "is_end": True, "prompt": "Finished"},
        ],
    }
    if with_banks:
        config["settings"] = {
            "banks": {
                "ingredients": {
                    "schema": {"required": ["name"]},
                    "max_records": 10,
                },
            },
        }
    if with_history:  # placeholder for future history-related cases
        pass
    return config


def _build_wizard(*, with_banks: bool = False) -> WizardReasoning:
    """Build a real ``WizardReasoning`` (no mocks) for unit-level tests."""
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(_wizard_config(with_banks=with_banks))
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# =====================================================================
# Base no-op defaults
# =====================================================================


class TestBaseNoOpDefaults:
    """The base ``ReasoningStrategy`` defaults must be safe no-ops."""

    @pytest.mark.asyncio
    async def test_base_restore_from_checkpoint_is_noop(self) -> None:
        strategy = _NoOpStrategy()
        manager = _StubManager()
        # Should not raise, even with foreign metadata keys.
        strategy.restore_from_checkpoint(
            manager, {"foreign_key": "value"}
        )
        assert manager.metadata == {}

    @pytest.mark.asyncio
    async def test_base_undo_to_checkpoint_is_noop(self) -> None:
        strategy = _NoOpStrategy()
        # Should not raise on any node id.
        strategy.undo_to_checkpoint("0.0.0")


# =====================================================================
# WizardReasoning.restore_from_checkpoint
# =====================================================================


class TestWizardRestoreFromCheckpoint:
    """``WizardReasoning`` owns the per-bucket restore logic."""

    def test_writes_expected_buckets_from_fsm_state(self) -> None:
        strategy = _build_wizard()
        manager = _StubManager()

        node_metadata = {
            "wizard_fsm_state": {
                "current_stage": "collect",
                "data": {"name": "Alice"},
                "completed": False,
                "history": ["start"],
                "transitions": [],
            },
        }

        strategy.restore_from_checkpoint(manager, node_metadata)

        wizard_meta = manager.metadata["wizard"]
        # Nested fsm_state is restored verbatim.
        assert wizard_meta["fsm_state"] == node_metadata["wizard_fsm_state"]
        # Flat top-level keys mirror the snapshot — ``normalize_wizard_state``
        # reads these with higher priority than nested fsm_state.
        assert wizard_meta["current_stage"] == "collect"
        assert wizard_meta["data"] == {"name": "Alice"}
        assert wizard_meta["completed"] is False
        assert wizard_meta["history"] == ["start"]

    def test_preserves_other_wizard_meta_keys(self) -> None:
        """Restore writes its keys without wiping out pre-existing ones."""
        strategy = _build_wizard()
        manager = _StubManager()
        manager.metadata["wizard"] = {"some_other_key": "keep_me"}

        strategy.restore_from_checkpoint(
            manager,
            {
                "wizard_fsm_state": {
                    "current_stage": "collect",
                    "data": {},
                    "completed": False,
                    "history": [],
                },
            },
        )

        wizard_meta = manager.metadata["wizard"]
        assert wizard_meta["some_other_key"] == "keep_me"
        assert wizard_meta["current_stage"] == "collect"

    def test_noop_when_wizard_fsm_state_absent(self) -> None:
        strategy = _build_wizard()
        manager = _StubManager()

        # Empty metadata: no-op.
        strategy.restore_from_checkpoint(manager, {})
        assert manager.metadata.get("wizard") is None

        # Unrelated key: still no-op.
        manager.metadata.clear()
        strategy.restore_from_checkpoint(manager, {"unrelated_key": "x"})
        assert manager.metadata.get("wizard") is None

    def test_noop_when_wizard_fsm_state_empty_dict(self) -> None:
        """Empty ``wizard_fsm_state`` dict is falsy and skipped (matches
        the original bot-side behaviour exactly)."""
        strategy = _build_wizard()
        manager = _StubManager()
        strategy.restore_from_checkpoint(manager, {"wizard_fsm_state": {}})
        assert manager.metadata.get("wizard") is None


# =====================================================================
# WizardReasoning.undo_to_checkpoint
# =====================================================================


class _SpyBank:
    """Records ``undo_to_checkpoint`` calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def undo_to_checkpoint(self, checkpoint_node_id: str) -> int:
        self.calls.append(checkpoint_node_id)
        return 0


class TestWizardUndoToCheckpoint:
    """``WizardReasoning`` iterates its ``_banks`` and forwards the id."""

    def test_iterates_all_banks_with_checkpoint_id(self) -> None:
        strategy = _build_wizard()
        spy_a = _SpyBank()
        spy_b = _SpyBank()
        # Replace the wizard's auto-built bank dict with spies so we can
        # observe forwarding deterministically without depending on the
        # real MemoryBank.
        strategy._banks = {"alpha": spy_a, "beta": spy_b}

        strategy.undo_to_checkpoint("node-7")

        assert spy_a.calls == ["node-7"]
        assert spy_b.calls == ["node-7"]

    def test_noop_when_banks_empty(self) -> None:
        """A wizard with no configured banks has nothing to undo."""
        strategy = _build_wizard(with_banks=False)
        assert strategy._banks == {}
        # No bank means nothing to forward to — call must not raise.
        strategy.undo_to_checkpoint("node-42")

    def test_with_real_memory_bank_does_not_raise(self) -> None:
        """A wizard with a real ``MemoryBank`` undoes cleanly even when the
        bank has no records yet — guards against the loop breaking on
        zero-record edge cases."""
        strategy = _build_wizard(with_banks=True)
        # The wizard's auto-built bank is keyed by config name.
        assert "ingredients" in strategy._banks
        bank = strategy._banks["ingredients"]
        assert isinstance(bank, MemoryBank)
        # Real ``MemoryBank.undo_to_checkpoint`` returns 0 on empty bank.
        strategy.undo_to_checkpoint("0.0.0")
        # Bank still has zero records — the call was a no-op.
        assert list(bank.all()) == []

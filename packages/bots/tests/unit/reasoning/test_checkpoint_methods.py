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

from dataknobs_bots.bot.base import normalize_wizard_state
from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.reasoning.wizard_types import SubflowContext, WizardState

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
    *,
    bank_names: tuple[str, ...] = (),
    with_history: bool = False,
) -> dict[str, Any]:
    """Build a minimal valid wizard config, optionally with named banks."""
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
    if bank_names:
        config["settings"] = {
            "banks": {
                name: {
                    "schema": {"required": ["name"]},
                    "max_records": 10,
                }
                for name in bank_names
            },
        }
    if with_history:  # placeholder for future history-related cases
        pass
    return config


def _build_wizard(*, bank_names: tuple[str, ...] = ()) -> WizardReasoning:
    """Build a real ``WizardReasoning`` (no mocks) for unit-level tests."""
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(_wizard_config(bank_names=bank_names))
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


def _subflow_wizard_config() -> dict[str, Any]:
    """A wizard whose main flow can push into a subflow.

    Kept separate from :func:`_wizard_config` so the main-flow cases above
    keep exercising a wizard with no subflow registry at all.
    """
    return {
        "name": "checkpoint-subflow-wizard",
        "version": "1.0",
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "prompt": "Hello",
                "transitions": [{"target": "done"}],
            },
            {"name": "done", "is_end": True, "prompt": "Finished"},
        ],
        "subflows": {
            "detail": {
                "stages": [
                    {
                        "name": "detail_start",
                        "is_start": True,
                        "prompt": "Details",
                        # Differs from every main-flow stage, so a reading
                        # taken off the wrong FSM is visible rather than
                        # coinciding.
                        "can_skip": True,
                        "transitions": [{"target": "detail_done"}],
                    },
                    {"name": "detail_done", "is_end": True, "prompt": "Done"},
                ]
            }
        },
    }


def _build_subflow_wizard() -> WizardReasoning:
    """Build a real ``WizardReasoning`` whose main flow has a subflow."""
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(_subflow_wizard_config())
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


def _main_flow_state() -> WizardState:
    """A state on the main flow, outside any subflow."""
    return WizardState(
        current_stage="collect",
        data={"name": "Alice"},
        history=["collect"],
    )


def _in_subflow_state() -> WizardState:
    """A state inside the ``detail`` subflow, pushed from ``collect``."""
    return WizardState(
        current_stage="detail_start",
        data={"name": "Alice"},
        history=["detail_start"],
        subflow_stack=[
            SubflowContext(
                parent_stage="collect",
                parent_data={},
                parent_history=["collect"],
                return_stage="done",
                result_mapping={},
                subflow_network="detail",
            )
        ],
    )


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
        strategy.restore_from_checkpoint(manager, {"foreign_key": "value"})
        assert manager.metadata == {}

    @pytest.mark.asyncio
    async def test_base_undo_to_checkpoint_is_noop(self) -> None:
        strategy = _NoOpStrategy()
        manager = _StubManager()
        # Should not raise on any node id.
        strategy.undo_to_checkpoint(manager, "0.0.0")


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

    def test_restore_moves_every_field_that_depends_on_the_stage(self) -> None:
        """Undo moves the stage, so it must move what the stage decides.

        ``restore_from_checkpoint`` used to hand-copy four keys out of the
        snapshot, and ``normalize_wizard_state`` reads the flat keys ahead
        of nested ``fsm_state``.  ``stage_index`` / ``total_stages`` /
        ``progress`` are derived from the stage and were in neither list, so
        they survived the undo describing the stage the wizard had just left.

        The reader looked like it covered this — it fell back to
        ``fsm_state.get("stage_index", 0)`` — but no writer anywhere puts
        ``stage_index`` inside ``fsm_state``, so the fallback could not fire
        and the stale flat value won uncontested.

        This is the narrow case that surfaced it; the guard below is the
        one that generalizes.
        """
        strategy = _build_wizard()
        manager = _StubManager()
        # Metadata as ``_build_wizard_metadata`` leaves it on the last stage.
        manager.metadata["wizard"] = {
            "current_stage": "done",
            "stage_index": 1,
            "total_stages": 2,
            "progress": 1.0,
            "completed": True,
            "data": {"name": "Alice"},
            "history": ["collect", "done"],
        }

        strategy.restore_from_checkpoint(
            manager,
            {
                "wizard_fsm_state": {
                    "current_stage": "collect",
                    "data": {"name": "Alice"},
                    "completed": False,
                    "history": ["collect"],
                },
            },
        )

        state = normalize_wizard_state(manager.metadata["wizard"])

        assert state["current_stage"] == "collect"
        assert state["stage_index"] == 0, (
            "undo restored the stage but left the index on the stage it "
            f"undid: {state['stage_index']}"
        )
        assert state["progress"] == 0.0, (
            f"progress still reports the undone stage: {state['progress']}"
        )

    def test_restore_agrees_with_the_metadata_builder_at_the_same_stage(self) -> None:
        """Guard on the class, by comparing the two writers instead of listing keys.

        ``_build_wizard_metadata`` and ``restore_from_checkpoint`` both write
        ``manager.metadata["wizard"]``; they now derive the stage-dependent
        fields from one shared method, and this is what holds them to it.

        A hand-written list of "fields the restore must also refresh" would
        be a third place to forget one — the same shape as the defect — so
        this compares everything the normalized view exposes instead.  That
        generality is not theoretical: written against the first, narrower
        fix, it caught two further stale fields (``can_go_back``, ``stages``)
        that the hand-written list had missed.
        """
        strategy = _build_wizard()

        # What the builder produces for the checkpoint's stage.
        state = WizardState(
            current_stage="collect",
            data={"name": "Alice"},
            history=["collect"],
        )
        strategy._restore_fsm_state(state)
        built = normalize_wizard_state(strategy._build_wizard_metadata(state))

        # What the restore path produces for the same stage, starting from
        # metadata left on a later one.
        manager = _StubManager()
        manager.metadata["wizard"] = strategy._build_wizard_metadata(
            WizardState(
                current_stage="done",
                data={"name": "Alice"},
                history=["collect", "done"],
            )
        )
        strategy.restore_from_checkpoint(
            manager,
            {"wizard_fsm_state": state.to_dict()},
        )
        restored = normalize_wizard_state(manager.metadata["wizard"])

        assert restored == built

    @pytest.mark.parametrize(
        ("undone_from", "restored_to", "case"),
        [
            (_in_subflow_state, _main_flow_state, "undo out of a subflow"),
            (_main_flow_state, _in_subflow_state, "undo into a subflow"),
        ],
        ids=["out_of_subflow", "into_subflow"],
    )
    def test_restore_agrees_with_the_builder_across_a_subflow_boundary(
        self,
        undone_from: Any,
        restored_to: Any,
        case: str,
    ) -> None:
        """The same guard, at the one boundary its fixtures did not cross.

        ``subflow_stage`` follows the stage like every other derived field,
        but it was built by the metadata builder alone and so was not in
        what restore refreshes.  Restore ``update()``s onto the *existing*
        dict, so the pre-undo value survives in both directions: undoing
        out of a subflow leaves a ``subflow_stage`` naming the subflow just
        left, and undoing into one leaves the key absent — which
        ``normalize_wizard_state`` reads as main flow at depth 0.

        Parametrized over both directions because a fix that only stops
        writing the stale key closes one of them and leaves the other.
        """
        strategy = _build_subflow_wizard()

        target = restored_to()
        strategy._restore_fsm_state(target)
        built = normalize_wizard_state(strategy._build_wizard_metadata(target))

        # Metadata left behind by the turn being undone, on the other side
        # of the boundary.
        manager = _StubManager()
        manager.metadata["wizard"] = strategy._build_wizard_metadata(undone_from())

        strategy.restore_from_checkpoint(
            manager,
            {"wizard_fsm_state": target.to_dict()},
        )
        restored = normalize_wizard_state(manager.metadata["wizard"])

        assert restored == built, case

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


class TestWizardUndoToCheckpoint:
    """``WizardReasoning`` iterates its banks and forwards the id."""

    def test_iterates_all_banks_with_checkpoint_id(self) -> None:
        """Two configured banks both have their records past the
        checkpoint removed — proving the forwarding loop covers every
        bank, not just the first.

        Records added at ``"0.0.1"`` are not ancestors of the checkpoint
        ``"0.0"`` and should be removed; records added at ``"0.0"`` (the
        checkpoint itself) survive. Asserting the observable bank state
        rather than the call sequence keeps the test honest against the
        real ``MemoryBank.undo_to_checkpoint`` contract.
        """
        strategy = _build_wizard(bank_names=("alpha", "beta"))
        banks = strategy.banks
        assert set(banks) == {"alpha", "beta"}

        # Each bank gets one ancestor record (kept) + one descendant
        # record (removed by undo).
        for name in ("alpha", "beta"):
            bank = banks[name]
            assert isinstance(bank, MemoryBank)
            bank.add({"name": f"{name}-keep"}, source_node_id="0.0")
            bank.add({"name": f"{name}-drop"}, source_node_id="0.0.1")
            assert bank.count() == 2

        # A stub manager with no ``conversation_id`` keys the default
        # (construction-time) slot — where ``_build_wizard`` built these banks.
        strategy.undo_to_checkpoint(_StubManager(), "0.0")

        for name in ("alpha", "beta"):
            survivors = [r.data["name"] for r in banks[name].all()]
            assert survivors == [f"{name}-keep"]

    def test_noop_when_banks_empty(self) -> None:
        """A wizard with no configured banks has nothing to undo."""
        strategy = _build_wizard()
        assert dict(strategy.banks) == {}
        # No bank means nothing to forward to — call must not raise.
        strategy.undo_to_checkpoint(_StubManager(), "node-42")

    def test_with_real_memory_bank_does_not_raise(self) -> None:
        """A wizard with a real ``MemoryBank`` undoes cleanly even when the
        bank has no records yet — guards against the loop breaking on
        zero-record edge cases."""
        strategy = _build_wizard(bank_names=("ingredients",))
        banks = strategy.banks
        # The wizard's auto-built bank is keyed by config name.
        assert "ingredients" in banks
        bank = banks["ingredients"]
        assert isinstance(bank, MemoryBank)
        # Real ``MemoryBank.undo_to_checkpoint`` returns 0 on empty bank.
        strategy.undo_to_checkpoint(_StubManager(), "0.0.0")
        # Bank still has zero records — the call was a no-op.
        assert list(bank.all()) == []

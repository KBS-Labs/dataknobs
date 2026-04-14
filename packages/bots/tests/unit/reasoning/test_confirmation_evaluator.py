"""Unit tests for ConfirmationEvaluator (Item 87).

Mostly pure synchronous tests — no LLM, no BotTestHarness.  Construct
stage dicts and ``WizardState`` objects, call ``evaluate()``, assert on
the returned ``ConfirmationEvaluation``.

The ``TestAdvancePathSnapshotSave`` class at the end is async and
exercises snapshot saving through the ``advance()`` API.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_confirmation import ConfirmationEvaluator
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.reasoning.wizard_types import WizardState

# ── Helpers ──────────────────────────────────────────────────────────

def _make_stage(
    *,
    name: str = "gather",
    response_template: str | None = "{{ value }}",
    confirm_first_render: bool | None = None,
    confirm_on_new_data: bool = False,
    fields: dict[str, str] | None = None,
) -> dict:
    """Build a minimal stage dict for evaluator testing."""
    stage: dict = {"name": name}
    if response_template is not None:
        stage["response_template"] = response_template
    if confirm_first_render is not None:
        stage["confirm_first_render"] = confirm_first_render
    if confirm_on_new_data:
        stage["confirm_on_new_data"] = True
    # Build schema from fields
    if fields is None:
        fields = {"value": "string"}
    stage["schema"] = {
        "properties": {k: {"type": v} for k, v in fields.items()},
    }
    return stage


def _make_state(
    data: dict | None = None,
    current_stage: str = "gather",
) -> WizardState:
    """Build a WizardState with optional initial data."""
    state = WizardState(current_stage=current_stage)
    if data:
        state.data.update(data)
    return state


# ── Tests ────────────────────────────────────────────────────────────

class TestEvaluateNoResponseTemplate:
    """Without response_template, confirmation never fires."""

    def test_no_template_never_confirms(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(response_template=None)
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is False
        assert result.confirm_keys == set()
        assert result.should_save_snapshot is False


class TestFirstRender:
    """First-render confirmation (render_count == 0)."""

    def test_new_data_with_default_confirm_first_render(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage()
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is True
        assert result.confirm_keys == {"value"}

    def test_new_data_with_confirm_first_render_true(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(confirm_first_render=True)
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is True
        assert result.confirm_keys == {"value"}

    def test_confirm_first_render_false_skips(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(confirm_first_render=False)
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is False
        assert result.confirm_keys == set()

    def test_confirm_first_render_false_without_confirm_on_new_data_no_save(
        self,
    ) -> None:
        """When both confirm_first_render=False and confirm_on_new_data
        is not set, no snapshot is saved."""
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(
            confirm_first_render=False, confirm_on_new_data=False,
        )
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is False
        assert result.should_save_snapshot is False

    def test_confirm_first_render_false_with_confirm_on_new_data_saves_snapshot(
        self,
    ) -> None:
        """When confirm_first_render=False skips confirmation but
        confirm_on_new_data is set, should_save_snapshot is True
        so a baseline exists for future diffs."""
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(
            confirm_first_render=False, confirm_on_new_data=True,
        )
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is False
        assert result.should_save_snapshot is True

    def test_empty_new_data_keys_skips(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage()
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, set())

        assert result.should_confirm is False

    def test_first_render_confirm_saves_snapshot_when_confirm_on_new_data(
        self,
    ) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(confirm_on_new_data=True)
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is True
        assert result.should_save_snapshot is True

    def test_first_render_confirm_no_save_without_confirm_on_new_data(
        self,
    ) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(confirm_on_new_data=False)
        state = _make_state({"value": "hello"})

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is True
        assert result.should_save_snapshot is False


class TestReconfirmOnNewData:
    """Re-confirmation via confirm_on_new_data (render_count > 0)."""

    def test_snapshot_diff_triggers_reconfirm(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(confirm_on_new_data=True)
        state = _make_state({"value": "new"})
        # Simulate prior render: render_count > 0, snapshot has old value
        state.increment_render_count("gather")
        state.save_stage_snapshot("gather", {"value"})
        state.data["value"] = "changed"

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is True
        assert "value" in result.confirm_keys
        assert "value" in result.snapshot_diff_keys

    def test_snapshot_same_skips_reconfirm(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(confirm_on_new_data=True)
        state = _make_state({"value": "same"})
        state.increment_render_count("gather")
        state.save_stage_snapshot("gather", {"value"})

        result = evaluator.evaluate(stage, state, set())

        assert result.should_confirm is False
        assert result.should_save_snapshot is True

    def test_no_confirm_on_new_data_skips_regardless(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(confirm_on_new_data=False)
        state = _make_state({"value": "changed"})
        state.increment_render_count("gather")

        result = evaluator.evaluate(stage, state, {"value"})

        assert result.should_confirm is False

    def test_bug_a_confirm_keys_is_union(self) -> None:
        """Bug A regression: confirm_keys must include BOTH extraction
        keys AND snapshot diff keys."""
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(
            confirm_on_new_data=True,
            fields={"domain": "string", "kb_name": "string"},
        )
        state = _make_state({"domain": "Math", "kb_name": "science-kb"})
        state.increment_render_count("gather")
        # Prior snapshot has old domain but no kb_name
        state.set_stage_snapshot("gather", {"domain": "Science"})

        # Extraction only changed domain; kb_name was set by tool
        result = evaluator.evaluate(stage, state, {"domain"})

        assert result.should_confirm is True
        assert "domain" in result.confirm_keys, "extraction key missing"
        assert "kb_name" in result.confirm_keys, "snapshot diff key missing"

    def test_bug_b_empty_extraction_with_diff_triggers_reconfirm(self) -> None:
        """Bug B regression: empty new_data_keys must NOT prevent
        re-confirmation when the snapshot diff is non-empty."""
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(
            confirm_on_new_data=True,
            fields={"domain": "string", "kb_name": "string"},
        )
        state = _make_state({"domain": "Science", "kb_name": "science-kb"})
        state.increment_render_count("gather")
        # Prior snapshot doesn't include kb_name
        state.set_stage_snapshot("gather", {"domain": "Science"})

        result = evaluator.evaluate(stage, state, set())

        assert result.should_confirm is True
        assert "kb_name" in result.confirm_keys


class TestComputeSnapshotDiff:
    """Tests for compute_snapshot_diff()."""

    def test_empty_prior_snapshot(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string", "b": "string"})
        state = _make_state({"a": "1", "b": "2"})

        diff = evaluator.compute_snapshot_diff(stage, state)

        assert diff == {"a", "b"}

    def test_no_diff(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string"})
        state = _make_state({"a": "1"})
        state.save_stage_snapshot("gather", {"a"})

        diff = evaluator.compute_snapshot_diff(stage, state)

        assert diff == set()

    def test_none_values_excluded(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string", "b": "string"})
        state = _make_state({"a": "1", "b": None})

        diff = evaluator.compute_snapshot_diff(stage, state)

        assert "a" in diff
        assert "b" not in diff

    def test_missing_keys_excluded(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string", "b": "string"})
        state = _make_state({"a": "1"})
        # b is not in state.data at all

        diff = evaluator.compute_snapshot_diff(stage, state)

        assert "a" in diff
        assert "b" not in diff

    def test_cleared_field_detected(self) -> None:
        """A field present in the prior snapshot but now None/absent
        should appear in the diff (field deletion)."""
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string", "b": "string"})
        state = _make_state({"a": "1"})
        # Prior snapshot had both a and b
        state.set_stage_snapshot("gather", {"a": "1", "b": "2"})

        diff = evaluator.compute_snapshot_diff(stage, state)

        assert "a" not in diff, "a is unchanged"
        assert "b" in diff, "b was cleared and should appear in diff"

    def test_cleared_field_set_to_none_detected(self) -> None:
        """A field set to None (rather than absent) should also be
        detected as cleared."""
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string", "b": "string"})
        state = _make_state({"a": "1", "b": None})
        state.set_stage_snapshot("gather", {"a": "1", "b": "old_value"})

        diff = evaluator.compute_snapshot_diff(stage, state)

        assert "a" not in diff
        assert "b" in diff


class TestSaveSnapshot:
    """Tests for save_snapshot() round-trip."""

    def test_roundtrip(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string", "b": "string"})
        state = _make_state({"a": "1", "b": "2"})

        evaluator.save_snapshot(stage, state)
        snapshot = state.get_stage_snapshot("gather")

        assert snapshot == {"a": "1", "b": "2"}

    def test_none_excluded_from_snapshot(self) -> None:
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string", "b": "string"})
        state = _make_state({"a": "1", "b": None})

        evaluator.save_snapshot(stage, state)
        snapshot = state.get_stage_snapshot("gather")

        assert snapshot == {"a": "1"}

    def test_save_then_no_diff(self) -> None:
        """After saving, compute_snapshot_diff returns empty."""
        evaluator = ConfirmationEvaluator()
        stage = _make_stage(fields={"a": "string"})
        state = _make_state({"a": "1"})

        evaluator.save_snapshot(stage, state)
        diff = evaluator.compute_snapshot_diff(stage, state)

        assert diff == set()


# ── advance() path snapshot test ─────────────────────────────────────

def _make_reasoning(config: dict[str, Any]) -> WizardReasoning:
    """Create a WizardReasoning from a config dict."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=wizard_fsm, strict_validation=False)


class TestAdvancePathSnapshotSave:
    """Verify that advance() saves the stage snapshot via _prepare_transition.

    If _prepare_transition were changed to skip the snapshot save, or
    if advance() bypassed _prepare_transition, subsequent generate()
    flows on the same state would see stale snapshot baselines.
    """

    @pytest.mark.asyncio
    async def test_advance_saves_snapshot_for_confirm_on_new_data_stage(
        self,
    ) -> None:
        config = {
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Enter value.",
                    "response_template": "Value: {{ value }}",
                    "confirm_on_new_data": True,
                    "schema": {
                        "properties": {"value": {"type": "string"}},
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('done')"},
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Done."},
            ],
        }
        reasoning = _make_reasoning(config)
        state = WizardState(
            current_stage="gather",
            data={},
            stage_entry_time=time.time(),
        )

        # advance() with dict input (no LLM needed)
        await reasoning.advance(
            user_input={"value": "hello"},
            state=state,
        )

        # _prepare_transition should have saved a snapshot
        snapshot = state.get_stage_snapshot("gather")
        assert snapshot.get("value") == "hello", (
            "advance() should save stage snapshot via _prepare_transition "
            "so that subsequent generate() flows have an accurate baseline"
        )

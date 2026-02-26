"""Tests for wizard stage render count management.

Covers:
- WizardState render count helper methods
- Render count incremented after restart navigation
- Render count incremented after back navigation
- No spurious first-render confirmation after restart
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider


# ---------------------------------------------------------------------------
# WizardState render count method tests (pure unit)
# ---------------------------------------------------------------------------


class TestWizardStateRenderCounts:
    """Tests for render count helpers on WizardState."""

    def test_get_render_count_default_zero(self) -> None:
        """Untracked stage returns 0."""
        state = WizardState(current_stage="start")
        assert state.get_render_count("start") == 0

    def test_increment_render_count(self) -> None:
        """increment_render_count advances count and returns new value."""
        state = WizardState(current_stage="start")
        assert state.increment_render_count("start") == 1
        assert state.increment_render_count("start") == 2
        assert state.get_render_count("start") == 2

    def test_render_counts_per_stage(self) -> None:
        """Different stages have independent counts."""
        state = WizardState(current_stage="a")
        state.increment_render_count("a")
        state.increment_render_count("a")
        state.increment_render_count("b")
        assert state.get_render_count("a") == 2
        assert state.get_render_count("b") == 1

    def test_render_count_stored_in_data(self) -> None:
        """Counts are stored in state.data so they persist across turns."""
        state = WizardState(current_stage="start")
        state.increment_render_count("start")
        assert state.data["_stage_render_counts"]["start"] == 1


class TestWizardStateSnapshots:
    """Tests for stage snapshot helpers on WizardState."""

    def test_get_snapshot_default_empty(self) -> None:
        """Untracked stage returns empty dict."""
        state = WizardState(current_stage="start")
        assert state.get_stage_snapshot("start") == {}

    def test_save_and_get_snapshot(self) -> None:
        """save_stage_snapshot captures current values; get retrieves them."""
        state = WizardState(
            current_stage="config",
            data={"color": "blue", "size": "large", "_internal": "skip"},
        )
        state.save_stage_snapshot("config", {"color", "size"})
        snap = state.get_stage_snapshot("config")
        assert snap == {"color": "blue", "size": "large"}

    def test_snapshot_excludes_none_values(self) -> None:
        """None values are not included in snapshots."""
        state = WizardState(
            current_stage="config",
            data={"color": "blue", "size": None},
        )
        state.save_stage_snapshot("config", {"color", "size"})
        assert state.get_stage_snapshot("config") == {"color": "blue"}

    def test_snapshot_excludes_missing_keys(self) -> None:
        """Keys not present in data are excluded from snapshot."""
        state = WizardState(
            current_stage="config",
            data={"color": "blue"},
        )
        state.save_stage_snapshot("config", {"color", "size"})
        assert state.get_stage_snapshot("config") == {"color": "blue"}


# ---------------------------------------------------------------------------
# Restart render count fix (integration)
# ---------------------------------------------------------------------------


@pytest.fixture
def restart_wizard_config() -> dict[str, Any]:
    """Wizard with response_template on start stage for restart testing."""
    return {
        "name": "restart-render-test",
        "version": "1.0",
        "stages": [
            {
                "name": "collect_name",
                "is_start": True,
                "prompt": "Ask the user for their name.",
                "response_template": "Welcome! What is your name?",
                "schema": {
                    "type": "object",
                    "properties": {"user_name": {"type": "string"}},
                    "required": ["user_name"],
                },
                "transitions": [
                    {
                        "target": "done",
                        "condition": "data.get('user_name')",
                    },
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "All done!",
            },
        ],
    }


class TestRestartRenderCount:
    """After restart, the start stage render count must be incremented.

    Without the fix, render_counts[stage_name] == 0 after restart, so the
    next user input with extracted data would trigger first-render
    confirmation logic instead of evaluating transitions — an unnecessary
    extra round-trip.
    """

    @pytest.mark.asyncio
    async def test_restart_increments_render_count(
        self,
        restart_wizard_config: dict[str, Any],
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After restart, the start stage render count is > 0."""
        manager, provider = conversation_manager_pair
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(restart_wizard_config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Step 1: greet — renders the start stage template
        await reasoning.greet(manager, llm=None)

        # Verify render count is 1 after greet
        fsm_state = manager.metadata["wizard"]["fsm_state"]
        counts = fsm_state.get("data", {}).get("_stage_render_counts", {})
        assert counts.get("collect_name", 0) == 1

        # Step 2: user says "restart"
        await manager.add_message(role="user", content="restart")
        provider.set_responses(["Welcome! What is your name?"])
        await reasoning.generate(manager, llm=None)

        # Verify render count is 1 for the fresh start stage after restart
        # (data was cleared, so this is a new count from the restart handler)
        fsm_state = manager.metadata["wizard"]["fsm_state"]
        counts = fsm_state.get("data", {}).get("_stage_render_counts", {})
        assert counts.get("collect_name", 0) == 1, (
            "Restart handler must increment render count so the next user "
            "message with data doesn't trigger first-render confirmation"
        )

    @pytest.mark.asyncio
    async def test_back_increments_render_count(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After going back, the previous stage render count is > 0."""
        config: dict[str, Any] = {
            "name": "back-render-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "step_one",
                    "is_start": True,
                    "prompt": "Step one",
                    "response_template": "Welcome to step one!",
                    "transitions": [{"target": "step_two"}],
                },
                {
                    "name": "step_two",
                    "prompt": "Step two",
                    "response_template": "Now at step two!",
                    "can_go_back": True,
                    "transitions": [{"target": "done"}],
                },
                {
                    "name": "done",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        manager, provider = conversation_manager_pair
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Greet at step_one
        await reasoning.greet(manager, llm=None)

        # Advance to step_two
        await manager.add_message(role="user", content="proceed")
        provider.set_responses(["Now at step two!"])
        await reasoning.generate(manager, llm=None)

        # Navigate back to step_one
        await manager.add_message(role="user", content="go back")
        provider.set_responses(["Welcome to step one!"])
        await reasoning.generate(manager, llm=None)

        # Verify step_one render count is > 0 after back navigation
        fsm_state = manager.metadata["wizard"]["fsm_state"]
        counts = fsm_state.get("data", {}).get("_stage_render_counts", {})
        assert counts.get("step_one", 0) >= 1, (
            "Back handler must increment render count so the next user "
            "message doesn't trigger first-render confirmation"
        )

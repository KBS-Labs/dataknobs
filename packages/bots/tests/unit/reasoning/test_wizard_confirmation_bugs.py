"""Integration tests for confirm_on_new_data refactor (Item 87).

Verifies the ConfirmationEvaluator is correctly wired into wizard.py
and produces the expected confirmation behavior through BotTestHarness.

The pre-refactor code had two bugs (verified to fail before changes):

Bug A — Incomplete confirmation message: the confirmation message only
used extraction keys, not snapshot diff keys.

Bug B — Missed re-confirmation: empty ``new_data_keys`` short-circuited
the entire confirmation block, even when the snapshot diff was non-empty.

Both bugs are fixed by:
1. The ConfirmationEvaluator decoupling the re-confirm gate from
   ``new_data_keys`` (using snapshot diff as the trigger instead).
2. The finalize_preamble snapshot save ensuring the snapshot
   accurately reflects post-finalize-turn state.

These tests verify the integrated behavior is correct.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_llm.testing import text_response
from dataknobs_llm.tools.base import Tool


# ── Helper: minimal tool for tool_result_mapping ─────────────────────

class LookupKbTool(Tool):
    """Returns a deterministic kb_name based on the domain parameter."""

    def __init__(self) -> None:
        super().__init__(
            name="lookup_kb",
            description="Look up knowledge base for a domain",
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
            },
            "required": ["domain"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        domain = kwargs.get("domain", "unknown")
        return {"kb_name": f"{domain.lower()}-kb"}


class TestConfirmOnNewDataWithToolResultMapping:
    """Verifies confirm_on_new_data works correctly with tool_result_mapping."""

    @pytest.mark.asyncio
    async def test_tool_mapped_field_in_snapshot_after_finalize(self) -> None:
        """After tool_result_mapping writes kb_name in finalize_turn,
        the snapshot includes it.  On the next turn, only fields that
        ACTUALLY changed trigger re-confirmation.

        Flow:
          Turn 1: extract {domain: "Science"}, confirm_first_render=False
                  → no confirmation → tool writes kb_name → snapshot
                  saved in finalize_preamble includes kb_name.
          Turn 2: extract {domain: "Math"} → snapshot diff = {domain}
                  (kb_name didn't change) → re-confirmation correctly
                  shows domain only.
        """
        config = (
            WizardConfigBuilder("tool-mapping-test")
            .stage(
                "gather",
                is_start=True,
                prompt="Tell me the domain.",
                response_template="Domain: {{ domain }}{% if kb_name %}, KB: {{ kb_name }}{% endif %}",
                confirm_first_render=False,
                confirm_on_new_data=True,
                tool_result_mapping=[{
                    "tool": "lookup_kb",
                    "params": {"domain": "domain"},
                    "mapping": {"kb_name": "kb_name"},
                }],
            )
            .field("domain", field_type="string", required=True)
            .field("kb_name", field_type="string")
            .transition("done", "data.get('_user_confirmed')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Domain: Science, KB: science-kb"),
                text_response("Updated!"),
            ],
            extraction_results=[
                [{"domain": "Science"}],
                [{"domain": "Math"}],
            ],
            tools=[LookupKbTool()],
        ) as harness:
            # Turn 1: extract domain, tool writes kb_name
            r1 = await harness.chat("Science domain")
            assert r1.wizard_stage == "gather"
            assert r1.wizard_data.get("kb_name") == "science-kb"

            # Turn 2: change domain → re-confirmation fires
            r2 = await harness.chat("change to Math")
            assert r2.wizard_stage == "gather"
            # Confirmation auto-generated with changed field
            assert r2.response.startswith("Here's what I got:")
            assert "Math" in r2.response

    @pytest.mark.asyncio
    async def test_no_spurious_reconfirm_when_nothing_changed(self) -> None:
        """When extraction produces empty new_data_keys and the snapshot
        accurately reflects post-finalize state (including tool-written
        fields), no unnecessary re-confirmation fires.

        This verifies Bug B is fixed holistically: the evaluator checks
        the snapshot diff (not new_data_keys), and the finalize_preamble
        snapshot save ensures accuracy.
        """
        config = (
            WizardConfigBuilder("no-spurious-test")
            .stage(
                "gather",
                is_start=True,
                prompt="Tell me the domain.",
                response_template="Domain: {{ domain }}{% if kb_name %}, KB: {{ kb_name }}{% endif %}",
                confirm_first_render=False,
                confirm_on_new_data=True,
                tool_result_mapping=[{
                    "tool": "lookup_kb",
                    "params": {"domain": "domain"},
                    "mapping": {"kb_name": "kb_name"},
                }],
            )
            .field("domain", field_type="string", required=True)
            .field("kb_name", field_type="string")
            .transition("done", "data.get('_user_confirmed')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Domain: Science, KB: science-kb"),
                text_response("Domain: Science, KB: science-kb"),
            ],
            extraction_results=[
                [{"domain": "Science"}],
                [{}],  # Empty extraction
            ],
            tools=[LookupKbTool()],
        ) as harness:
            # Turn 1: extract domain, tool writes kb_name
            r1 = await harness.chat("Science domain")
            assert r1.wizard_stage == "gather"
            assert r1.wizard_data.get("kb_name") == "science-kb"

            # Turn 2: empty extraction, nothing changed → no confirmation
            r2 = await harness.chat("looks good")
            assert r2.wizard_stage == "gather"
            # Should NOT be an auto-generated confirmation —
            # it should be a regular stage response.
            assert not r2.response.startswith("Here's what I got:"), (
                "Should not re-confirm when nothing changed"
            )


class TestConfirmOnNewDataReconfirmFlow:
    """Verifies the re-confirmation flow works for value changes."""

    @pytest.mark.asyncio
    async def test_value_change_triggers_reconfirm(self) -> None:
        """When a field value changes on a subsequent turn at a stage
        with confirm_on_new_data=True, re-confirmation fires.

        This is the core confirm_on_new_data use case (not specific to
        tool_result_mapping).
        """
        config = (
            WizardConfigBuilder("reconfirm-test")
            .stage(
                "gather",
                is_start=True,
                prompt="What is your name?",
                response_template="Got it, {{ user_name }}!",
                confirm_first_render=False,
                confirm_on_new_data=True,
            )
            .field("user_name", field_type="string", required=True)
            .transition("done", "data.get('_user_confirmed')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Got it, Alice!"),
                text_response("Updated!"),
            ],
            extraction_results=[
                [{"user_name": "Alice"}],
                [{"user_name": "Bob"}],
            ],
        ) as harness:
            # Turn 1: extract name, skip first-render confirmation
            r1 = await harness.chat("My name is Alice")
            assert r1.wizard_stage == "gather"
            assert r1.wizard_data["user_name"] == "Alice"

            # Turn 2: change name → re-confirmation fires
            r2 = await harness.chat("Actually, call me Bob")
            assert r2.wizard_stage == "gather"
            assert r2.response.startswith("Here's what I got:")
            assert "Bob" in r2.response

    @pytest.mark.asyncio
    async def test_baseline_snapshot_saved_on_skip_first_render(self) -> None:
        """When confirm_first_render=False skips first-render confirmation,
        a baseline snapshot is still saved (via should_save_snapshot) so
        that the next turn's diff compares against actual values, not an
        empty dict.

        Without the baseline save, the second turn would diff against {}
        and treat ALL current values as "new" — even if unchanged.
        """
        config = (
            WizardConfigBuilder("baseline-snapshot-test")
            .stage(
                "gather",
                is_start=True,
                prompt="What is your name?",
                response_template="Got it, {{ user_name }}!",
                confirm_first_render=False,
                confirm_on_new_data=True,
            )
            .field("user_name", field_type="string", required=True)
            .transition("done", "data.get('_user_confirmed')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Got it, Alice!"),
                # Turn 2: same value → should NOT re-confirm
                text_response("Got it, Alice!"),
            ],
            extraction_results=[
                [{"user_name": "Alice"}],
                [{"user_name": "Alice"}],  # Same value
            ],
        ) as harness:
            # Turn 1: extract name, skip confirmation
            r1 = await harness.chat("My name is Alice")
            assert r1.wizard_stage == "gather"

            # Turn 2: same value extracted → no diff → no confirmation
            r2 = await harness.chat("My name is Alice")
            assert r2.wizard_stage == "gather"
            assert not r2.response.startswith("Here's what I got:"), (
                "Should not re-confirm when value is unchanged"
            )

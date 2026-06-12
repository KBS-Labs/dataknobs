"""Tests for the wizard ``intent_confirm:`` stage primitive.

The primitive is purely declarative: at load time, the synthesizer
expands ``intent_confirm:`` into ``mode: conversation`` +
``response_template`` + ``intent_detection`` + ``schema`` +
``transitions``. There is NO new runtime dispatch — all runtime
behavior goes through existing wizard paths.
"""
from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.testing import BotTestHarness


def _wizard_with_intent_confirm(**extra: Any) -> dict[str, Any]:
    return {
        "name": "intent-confirm-test",
        "version": "1.0",
        "stages": [
            {
                "name": "propose",
                "is_start": True,
                "intent_confirm": {
                    # v5 (Finding E): wizard YAML has no `inputs:` field for
                    # populating Jinja vars. Authors thread values in via
                    # prior-stage signals / data; for the test we literalize
                    # the proposal_template to avoid taking a dependency on
                    # a non-existent contract.
                    "proposal_template": "Use ASRM?",
                    "intents": {
                        "accept":  {"target": "accepted"},
                        "decline": {"target": "declined"},
                    },
                    **extra,
                },
            },
            {
                "name": "accepted", "is_end": True,
                "response_template": "Activated.",
            },
            {
                "name": "declined", "is_end": True,
                "response_template": "Skipped.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# End-to-end runtime behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_render_emits_proposal_no_llm() -> None:
    async with await BotTestHarness.create(
        wizard_config=_wizard_with_intent_confirm(),
    ) as harness:
        result = await harness.greet()
        assert "Use ASRM?" in result.response
        # v5 (Finding D): BotTestHarness exposes provider; call_count lives there.
        assert harness.provider.call_count == 0


@pytest.mark.asyncio
async def test_accept_keyword_routes_to_accept_target() -> None:
    async with await BotTestHarness.create(
        wizard_config=_wizard_with_intent_confirm(),
    ) as harness:
        await harness.greet()
        await harness.chat("yes")
        assert harness.wizard_stage == "accepted"
        # Per-intent boolean (the synthesizer's per_intent_booleans flag)
        assert harness.wizard_data.get("accept") is True
        # Back-compat: the existing _intent key is ALSO written
        assert harness.wizard_data.get("_intent") == "accept"


@pytest.mark.asyncio
async def test_decline_keyword_routes_to_decline_target() -> None:
    async with await BotTestHarness.create(
        wizard_config=_wizard_with_intent_confirm(),
    ) as harness:
        await harness.greet()
        await harness.chat("no")
        assert harness.wizard_stage == "declined"


# ---------------------------------------------------------------------------
# Per-intent keyword override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_intent_keyword_override_used_over_default() -> None:
    config = _wizard_with_intent_confirm()
    config["stages"][0]["intent_confirm"]["intents"]["accept"]["keywords"] = (
        ["affirm"]
    )
    async with await BotTestHarness.create(wizard_config=config) as harness:
        await harness.greet()
        await harness.chat("affirm")
        assert harness.wizard_stage == "accepted"


# ---------------------------------------------------------------------------
# Extract field + LLM fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_field_captures_payload_via_llm_fallback() -> None:
    from dataknobs_llm.testing import text_response

    config = _wizard_with_intent_confirm(
        intents={
            "accept":      {"target": "accepted"},
            "alternative": {
                "target": "declined",
                "extract": "framework_name",
                "llm_fallback": True,
            },
        },
        llm_fallback=True,
    )
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=[
            text_response('{"intent": "alternative", "extracted": "AIAM"}'),
        ],
    ) as harness:
        await harness.greet()
        await harness.chat("Actually use AIAM instead")
        assert harness.wizard_data.get("framework_name") == "AIAM"


# ---------------------------------------------------------------------------
# On-no-match handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_no_match_routes_to_configured_target() -> None:
    config = _wizard_with_intent_confirm()
    config["stages"][0]["intent_confirm"]["on_no_match"] = {
        "target": "declined",
    }
    async with await BotTestHarness.create(wizard_config=config) as harness:
        await harness.greet()
        await harness.chat("xyz123 ambiguous")
        assert harness.wizard_stage == "declined"


@pytest.mark.asyncio
async def test_on_no_match_reprompts_with_clarification_template() -> None:
    config = _wizard_with_intent_confirm()
    config["stages"][0]["intent_confirm"]["on_no_match"] = {
        "clarification_template": "Was that a yes or no?",
    }
    async with await BotTestHarness.create(wizard_config=config) as harness:
        await harness.greet()
        result = await harness.chat("xyz123")
        assert harness.wizard_stage == "propose"
        assert "Was that a yes or no?" in result.response


# ---------------------------------------------------------------------------
# Validation — load-time errors via shared helper
# ---------------------------------------------------------------------------


def test_load_rejects_intent_confirm_plus_schema() -> None:
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["schema"] = {"type": "object"}
    with pytest.raises(ConfigurationError, match="intent_confirm"):
        WizardConfigLoader().load_from_dict(bad)


def test_load_rejects_intent_confirm_plus_response_template() -> None:
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["response_template"] = "redundant"
    with pytest.raises(ConfigurationError, match="intent_confirm"):
        WizardConfigLoader().load_from_dict(bad)


def test_load_rejects_intent_confirm_plus_transitions() -> None:
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["transitions"] = [{"target": "accepted"}]
    with pytest.raises(ConfigurationError, match="intent_confirm"):
        WizardConfigLoader().load_from_dict(bad)


def test_load_rejects_empty_intents_dict() -> None:
    """At least one intent is required for a useful primitive."""
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["intent_confirm"]["intents"] = {}
    with pytest.raises(ConfigurationError, match="declares no"):
        WizardConfigLoader().load_from_dict(bad)


def test_load_rejects_intent_without_target() -> None:
    """Without ``target:``, the synthesizer cannot emit a transition.

    Pre-fix this would crash at synthesize-time with a bare
    ``KeyError('target')`` carrying no stage context.
    """
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["intent_confirm"]["intents"] = {
        "accept": {},  # missing target
    }
    with pytest.raises(ConfigurationError, match="missing required 'target'"):
        WizardConfigLoader().load_from_dict(bad)


def test_load_rejects_non_mapping_intents_value() -> None:
    """``intents:`` must be a dict (mapping intent name → spec)."""
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["intent_confirm"]["intents"] = ["accept", "decline"]
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        WizardConfigLoader().load_from_dict(bad)


def test_load_rejects_intent_spec_not_a_dict() -> None:
    """Each intent value must be a mapping (with at least ``target:``)."""
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["intent_confirm"]["intents"] = {
        "accept": "accepted",  # bare string instead of dict
    }
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        WizardConfigLoader().load_from_dict(bad)


def test_load_rejects_reserved_intent_name() -> None:
    """``_intent`` is already used by the intent-detection runtime to
    record the matched intent name in ``state.data``. Letting a user
    name an intent ``_intent`` would silently overwrite that bookkeeping
    key during transition-condition evaluation. Reject at load time
    with a clear message rather than letting the collision surface as
    a downstream surprise.
    """
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
    from dataknobs_common.exceptions import ConfigurationError

    bad = _wizard_with_intent_confirm()
    bad["stages"][0]["intent_confirm"]["intents"] = {
        "_intent": {"target": "accepted"},
    }
    with pytest.raises(ConfigurationError, match="reserved"):
        WizardConfigLoader().load_from_dict(bad)


# ---------------------------------------------------------------------------
# Per-intent llm_fallback wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_intent_llm_fallback_promotes_classifier_to_composite(
) -> None:
    """Setting ``llm_fallback: true`` on a single intent (no
    block-level flag) MUST promote the whole stage's classifier to a
    composite chain — the documented USER_GUIDE behavior. Pre-fix the
    per-intent flag was a silent no-op.
    """
    from dataknobs_llm.testing import text_response

    config = _wizard_with_intent_confirm(
        intents={
            "accept":      {"target": "accepted"},
            "alternative": {
                "target": "declined",
                "extract": "framework_name",
                "llm_fallback": True,   # ← per-intent only
            },
        },
        # NO block-level llm_fallback
    )
    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=[
            text_response('{"intent": "alternative", "extracted": "AIAM"}'),
        ],
    ) as harness:
        await harness.greet()
        await harness.chat("Actually use AIAM instead")
        # If per-intent llm_fallback had not promoted to composite,
        # this message would no-match (no keyword) and the wizard
        # would stay on `propose`.
        assert harness.wizard_data.get("framework_name") == "AIAM"


def test_per_intent_llm_fallback_visible_in_synthesized_classifier() -> None:
    """Direct synthesis snapshot: per-intent ``llm_fallback`` produces
    a composite classifier shape even with no block-level flag.
    """
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    config = _wizard_with_intent_confirm(
        intents={
            "accept":      {"target": "accepted"},
            "alternative": {
                "target": "declined",
                "extract": "framework_name",
                "llm_fallback": True,
            },
        },
    )
    fsm = WizardConfigLoader().load_from_dict(config)
    assert (
        fsm.stages["propose"]["intent_detection"]["classifier"]
        == "composite"
    )


# ---------------------------------------------------------------------------
# Synthesizer strips the original block
# ---------------------------------------------------------------------------


def test_intent_confirm_block_removed_after_synthesis() -> None:
    """After synthesis the FSM-metadata layer must NOT carry the raw
    ``intent_confirm`` block as a parallel source of truth — only the
    expanded primitives.
    """
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    fsm = WizardConfigLoader().load_from_dict(
        _wizard_with_intent_confirm(),
    )
    propose_meta = fsm.stages["propose"]
    assert propose_meta.get("intent_confirm") is None


# ---------------------------------------------------------------------------
# Synthesis snapshot — downstream sees normalized shape
# ---------------------------------------------------------------------------


def test_synthesis_expands_to_conversation_mode_template_intent_detection(
) -> None:
    """The synthesizer expands intent_confirm to existing wizard
    primitives — zero new runtime branches. Pinning the synthesis
    documents the expansion shape.
    """
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(_wizard_with_intent_confirm())
    # v5 (Finding C): WizardFSM exposes `stages`, not `stage_metadata`.
    propose_meta = fsm.stages["propose"]

    assert propose_meta["mode"] == "conversation"
    # v5 (Finding E): literalized — see fixture above.
    assert propose_meta["response_template"] == "Use ASRM?"
    # v5 (Finding B): synthesizer emits `classifier:`, not `method:`.
    # `detect_intent` accepts both for back-compat, but new YAML uses
    # the v3 shape.
    assert propose_meta["intent_detection"]["classifier"] == "keyword"
    assert propose_meta["intent_detection"]["per_intent_booleans"] is True
    assert propose_meta["schema"] == {
        "type": "object",
        "properties": {
            "accept":  {"type": "boolean"},
            "decline": {"type": "boolean"},
        },
    }
    targets = {
        t["target"]: t["condition"] for t in propose_meta["transitions"]
    }
    assert targets["accepted"] == "data.get('accept') == True"
    assert targets["declined"] == "data.get('decline') == True"


# ---------------------------------------------------------------------------
# Consumer-extensible synthesizer registry
# ---------------------------------------------------------------------------


def test_consumer_can_register_own_stage_synthesizer() -> None:
    """Pins the synthesizer registry contract: a consumer registers
    their own primitive synthesizer and the loader picks it up.
    """
    from dataknobs_bots.reasoning.wizard_loader import (
        WizardConfigLoader,
        register_stage_synthesizer,
        unregister_stage_synthesizer,
    )

    class CustomPrimitiveSynthesizer:
        field = "my_primitive"

        def synthesize(self, stage: dict[str, Any]) -> None:
            block = stage[self.field]
            stage["response_template"] = f"CUSTOM: {block['message']}"

    register_stage_synthesizer(CustomPrimitiveSynthesizer())
    try:
        config = {
            "name": "custom-primitive-test",
            "version": "1.0",
            "stages": [
                {
                    "name": "only",
                    "is_start": True,
                    "is_end": True,
                    "my_primitive": {"message": "hello"},
                    "prompt": "noop",
                },
            ],
        }
        fsm = WizardConfigLoader().load_from_dict(config)
        # v5 (Finding C): WizardFSM exposes `stages`, not `stage_metadata`.
        assert fsm.stages["only"]["response_template"] == "CUSTOM: hello"
    finally:
        unregister_stage_synthesizer("my_primitive")

"""Tests for the manager-metadata inbox bridge.

The bridge is implemented as an auto-registered ``on_turn_start``
hook when ``WizardReasoningConfig.manager_metadata_inbox_key`` is
set. These tests pin the consumer-visible contract:

- Consume-on-read (pop, not get).
- None-as-eviction.
- Empty-dict no-op (no log churn).
- Per-turn clear runs BEFORE the inbox merge (an inbox carrying an
  ephemeral key survives the per-turn clear).
- Multi-key support.
- Custom merge_fn override.
- ``greet`` symmetrically consumes the inbox.
- Writer helper round-trips end-to-end.
"""
from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_config import WizardReasoningConfig
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm import EchoProvider


def _minimal_wizard_dict() -> dict[str, Any]:
    return {
        "name": "inbox-test",
        "version": "1.0",
        "stages": [
            {
                "name": "only",
                "is_start": True,
                "is_end": True,
                "prompt": "noop",
            },
        ],
    }


def _build_wizard(
    *,
    inbox_key: str | list[str] | None = None,
    inbox_merge_fn: Any = None,
) -> WizardReasoning:
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(_minimal_wizard_dict())
    config = WizardReasoningConfig(
        wizard_config=_minimal_wizard_dict(),
        manager_metadata_inbox_key=inbox_key,
        inbox_merge_fn=inbox_merge_fn,
    )
    return WizardReasoning(wizard_fsm=fsm, config=config)


class _FakeManager:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self._messages: list[dict[str, Any]] = []

    def get_messages(self) -> list[dict[str, Any]]:
        return self._messages


def _dummy_llm() -> Any:
    return EchoProvider({"provider": "echo", "model": "test"})


# ---------------------------------------------------------------------------
# Reproducing pin: inbox consumed into state.data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbox_consumed_into_state_data() -> None:
    """RED before fix, GREEN after: configured inbox key is consumed."""
    wizard = _build_wizard(inbox_key="_inbox")
    manager = _FakeManager()
    manager.metadata["_inbox"] = {"proposal_id": "asrm", "_proposal_queued": True}

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    assert handle.wizard_state.data["proposal_id"] == "asrm"
    assert handle.wizard_state.data["_proposal_queued"] is True


# ---------------------------------------------------------------------------
# Back-compat: default None = no auto-registration = no behaviour change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_inbox_key_none_is_no_op() -> None:
    wizard = _build_wizard(inbox_key=None)
    manager = _FakeManager()
    manager.metadata["_inbox"] = {"proposal_id": "asrm"}

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    assert "proposal_id" not in handle.wizard_state.data
    assert manager.metadata["_inbox"] == {"proposal_id": "asrm"}


# ---------------------------------------------------------------------------
# Consume-on-read: key is popped, not get'd
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_on_read_pops_inbox_key() -> None:
    wizard = _build_wizard(inbox_key="_inbox")
    manager = _FakeManager()
    manager.metadata["_inbox"] = {"x": "v"}

    await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    assert "_inbox" not in manager.metadata


# ---------------------------------------------------------------------------
# None-as-eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_value_evicts_prior_state_data_key() -> None:
    wizard = _build_wizard(inbox_key="_inbox")
    manager = _FakeManager()

    manager.metadata["_inbox"] = {"x": "v1"}
    h1 = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert h1.wizard_state.data["x"] == "v1"

    # Persist state for the next turn so the merge sees prior x.
    await wizard._save_wizard_state(manager, h1.wizard_state)

    manager.metadata["_inbox"] = {"x": None}
    h2 = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert h2.wizard_state.data["x"] is None


# ---------------------------------------------------------------------------
# Empty-dict no-op (silent — no log churn, no pop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_dict_inbox_is_silent_no_op(caplog) -> None:
    wizard = _build_wizard(inbox_key="_inbox")
    manager = _FakeManager()
    manager.metadata["_inbox"] = {}

    caplog.set_level("DEBUG", logger="dataknobs_bots.reasoning.wizard_inbox")
    await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    # Empty payload should pop the key (consume-on-read still applies)
    # but NOT log "consumed" — it's a no-op merge.
    assert "_inbox" not in manager.metadata
    assert not any(
        "consumed manager.metadata" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Per-turn clear runs BEFORE the inbox merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbox_merge_runs_after_per_turn_clear() -> None:
    """Inbox value for an ephemeral (per-turn) key survives the per-turn
    clear because the merge happens after."""
    wizard_dict = _minimal_wizard_dict()
    wizard_dict["settings"] = {"per_turn_keys": ["_intent"]}
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(wizard_dict)

    config = WizardReasoningConfig(
        wizard_config=wizard_dict,
        manager_metadata_inbox_key="_inbox",
    )
    wizard = WizardReasoning(wizard_fsm=fsm, config=config)

    manager = _FakeManager()
    manager.metadata["_inbox"] = {"_intent": "new"}

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    assert handle.wizard_state.data.get("_intent") == "new"


# ---------------------------------------------------------------------------
# Multi-key inbox: drains all listed keys per turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_key_inbox_drains_all_keys() -> None:
    wizard = _build_wizard(inbox_key=["_signals", "_audit"])
    manager = _FakeManager()
    manager.metadata["_signals"] = {"x": 1}
    manager.metadata["_audit"] = {"y": 2}

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    assert handle.wizard_state.data["x"] == 1
    assert handle.wizard_state.data["y"] == 2
    assert "_signals" not in manager.metadata
    assert "_audit" not in manager.metadata


# ---------------------------------------------------------------------------
# Custom merge_fn: consumer-supplied deep-merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_merge_fn_overrides_default_dict_update() -> None:
    def deep_merge(target: dict, source: dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                deep_merge(target[key], value)
            else:
                target[key] = value

    wizard = _build_wizard(inbox_key="_inbox", inbox_merge_fn=deep_merge)
    manager = _FakeManager()

    # Seed prior state via the inbox itself.
    manager.metadata["_inbox"] = {"nested": {"existing": "v"}}
    h1 = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert h1.wizard_state.data["nested"] == {"existing": "v"}
    await wizard._save_wizard_state(manager, h1.wizard_state)

    # Second turn: deep_merge should MERGE nested dicts, not overwrite.
    manager.metadata["_inbox"] = {"nested": {"new": "w"}}
    h2 = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert h2.wizard_state.data["nested"] == {"existing": "v", "new": "w"}


# ---------------------------------------------------------------------------
# greet symmetrically consumes the inbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_greet_consumes_inbox_too() -> None:
    """Bot-initiated greet fires on_turn_start, so the inbox-bridge
    auto-registered hook applies on the greeting turn too.

    The hook fires upstream of ``generate_stage_response``; the minimal
    ``_FakeManager`` raises further down. The bridge effect (state
    populated, inbox key popped) lives upstream of that raise — we
    catch the downstream error and verify the merged state via
    ``wizard._last_wizard_state`` (set by ``_get_wizard_state`` before
    the hook fires).
    """
    wizard = _build_wizard(inbox_key="_inbox")
    manager = _FakeManager()
    manager.metadata["_inbox"] = {"greeting_signal": "from_writer"}

    try:
        await wizard.greet(manager, llm=_dummy_llm())
    except AttributeError:
        pass

    state = wizard._last_wizard_state
    assert state is not None
    assert state.data.get("greeting_signal") == "from_writer"
    assert "_inbox" not in manager.metadata


# ---------------------------------------------------------------------------
# Concurrency: distinct managers don't bleed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_wizards_distinct_managers_isolated() -> None:
    w1 = _build_wizard(inbox_key="_inbox")
    w2 = _build_wizard(inbox_key="_inbox")
    m1 = _FakeManager()
    m2 = _FakeManager()
    m1.metadata["_inbox"] = {"x": "from_m1"}
    m2.metadata["_inbox"] = {"x": "from_m2"}

    h1 = await w1.begin_turn(m1, llm=_dummy_llm(), tools=None)
    h2 = await w2.begin_turn(m2, llm=_dummy_llm(), tools=None)

    assert h1.wizard_state.data["x"] == "from_m1"
    assert h2.wizard_state.data["x"] == "from_m2"


# ---------------------------------------------------------------------------
# Writer helper round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_to_inbox_helper_round_trips() -> None:
    """End-to-end: writer helper publishes → next-turn begin_turn merges."""
    from dataknobs_bots.reasoning.wizard_inbox import write_to_inbox

    wizard = _build_wizard(inbox_key="_inbox")
    manager = _FakeManager()

    write_to_inbox(manager, "_inbox", {"signal_from_writer": True})

    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)
    assert handle.wizard_state.data["signal_from_writer"] is True


@pytest.mark.asyncio
async def test_write_to_inbox_helper_rejects_manager_without_metadata() -> None:
    from dataknobs_bots.reasoning.wizard_inbox import write_to_inbox

    class _NoMetadata:
        pass

    with pytest.raises(AttributeError, match="metadata"):
        write_to_inbox(_NoMetadata(), "_inbox", {"x": 1})


# ---------------------------------------------------------------------------
# Non-dict payload: WARNING + skip (no crash)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_mapping_payload_warns_and_skips(caplog) -> None:
    wizard = _build_wizard(inbox_key="_inbox")
    manager = _FakeManager()
    manager.metadata["_inbox"] = "not a dict"  # writer bug

    caplog.set_level("WARNING", logger="dataknobs_bots.reasoning.wizard_inbox")
    handle = await wizard.begin_turn(manager, llm=_dummy_llm(), tools=None)

    # Did not crash; key still popped (consume-on-read); state unaffected.
    assert "_inbox" not in manager.metadata
    assert handle.wizard_state.data == {} or all(
        k.startswith("_") for k in handle.wizard_state.data
    ) or "not a dict" not in str(handle.wizard_state.data)
    assert any(
        "not a mapping" in rec.message.lower()
        for rec in caplog.records
    )

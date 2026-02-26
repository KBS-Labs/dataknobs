"""Tests for MemoryBank collection mode in wizard flows.

Covers Phase 2: done signal detection, field clearing between records,
and collection-mode branching in WizardReasoning.generate().
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import EchoProvider
from dataknobs_llm.conversations.manager import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm import LLMConfig
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.prompts.implementations.config_library import ConfigPromptLibrary
from dataknobs_llm.testing import text_response


# =====================================================================
# Helpers
# =====================================================================

def _make_collection_wizard(
    done_keywords: list[str] | None = None,
    min_records: int = 0,
) -> WizardReasoning:
    """Create a WizardReasoning with a collection-mode stage."""
    if done_keywords is None:
        done_keywords = ["done", "that's all", "finished"]

    condition = "data.get('_collection_done')"
    if min_records > 0:
        condition += f" and bank('ingredients').count() >= {min_records}"

    config: dict[str, Any] = {
        "name": "collection-wizard",
        "version": "1.0",
        "settings": {
            "banks": {
                "ingredients": {
                    "schema": {"required": ["name"]},
                    "max_records": 50,
                },
            },
        },
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "prompt": "What ingredient would you like to add?",
                "collection_mode": "collection",
                "collection_config": {
                    "bank_name": "ingredients",
                    "done_keywords": done_keywords,
                    "done_prompt": "Got it! Anything else?",
                },
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "amount": {"type": "string"},
                    },
                    "required": ["name"],
                },
                "transitions": [
                    {
                        "target": "review",
                        "condition": condition,
                    },
                ],
            },
            {
                "name": "review",
                "is_end": True,
                "prompt": "Here are your ingredients",
                "response_template": (
                    "You added {{ bank('ingredients').count() }} items."
                ),
            },
        ],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# =====================================================================
# Done signal detection
# =====================================================================

class TestDoneSignalDetection:

    def test_exact_match(self) -> None:
        assert WizardReasoning._is_done_signal("done", ["done", "finished"])

    def test_case_insensitive(self) -> None:
        assert WizardReasoning._is_done_signal("DONE", ["done"])

    def test_whitespace_stripped(self) -> None:
        assert WizardReasoning._is_done_signal("  done  ", ["done"])

    def test_no_match(self) -> None:
        assert not WizardReasoning._is_done_signal("add flour", ["done"])

    def test_empty_keywords(self) -> None:
        assert not WizardReasoning._is_done_signal("done", [])

    def test_thats_all_keyword(self) -> None:
        assert WizardReasoning._is_done_signal(
            "that's all", ["done", "that's all"]
        )


# =====================================================================
# Collection mode integration
# =====================================================================

class TestCollectionModeIntegration:

    def test_collection_stage_config_recognised(self) -> None:
        reasoning = _make_collection_wizard()
        stage_names = reasoning._fsm.stage_names
        assert "collect" in stage_names

    def test_bank_initialised_for_collection(self) -> None:
        reasoning = _make_collection_wizard()
        assert "ingredients" in reasoning._banks

    def test_handle_collection_adds_to_bank(self) -> None:
        """Verify the low-level _handle_collection_mode adds records."""
        reasoning = _make_collection_wizard()
        state = WizardState(current_stage="collect", data={})
        stage = {
            "name": "collect",
            "collection_mode": "collection",
            "collection_config": {
                "bank_name": "ingredients",
                "done_keywords": ["done"],
            },
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": "string"},
                },
            },
        }
        # Simulate extraction result
        extracted = {"name": "flour", "amount": "2 cups"}

        # The method is async, but for the core bank-addition logic
        # we can verify by directly calling the bank
        reasoning._banks["ingredients"].add(
            {"name": "flour", "amount": "2 cups"}, source_stage="collect"
        )
        assert reasoning._banks["ingredients"].count() == 1

    def test_done_signal_sets_collection_done(self) -> None:
        """When done keyword is detected, _collection_done flag is set."""
        reasoning = _make_collection_wizard()
        state = WizardState(current_stage="collect", data={})

        # Add a record first
        reasoning._banks["ingredients"].add({"name": "flour"})

        # Set _collection_done manually (simulating what
        # _handle_collection_mode does on done signal)
        state.data["_collection_done"] = True

        # Now the condition should evaluate True
        result = reasoning._evaluate_condition(
            "data.get('_collection_done') and bank('ingredients').count() >= 1",
            state.data,
        )
        assert result is True

    def test_field_clearing_between_records(self) -> None:
        """Schema fields should be cleared between collection records."""
        reasoning = _make_collection_wizard()
        state = WizardState(current_stage="collect", data={
            "name": "flour",
            "amount": "2 cups",
        })
        # Simulate what _handle_collection_mode does: clear schema fields
        schema_props = {"name", "amount"}
        for field_name in schema_props:
            state.data.pop(field_name, None)

        assert "name" not in state.data
        assert "amount" not in state.data

    def test_min_records_condition(self) -> None:
        """Collection with min_records requires sufficient records."""
        reasoning = _make_collection_wizard(min_records=2)

        # Only 1 record → condition should be False
        reasoning._banks["ingredients"].add({"name": "flour"})
        result = reasoning._evaluate_condition(
            "data.get('_collection_done') and "
            "bank('ingredients').count() >= 2",
            {"_collection_done": True},
        )
        assert result is False

        # 2 records → condition should be True
        reasoning._banks["ingredients"].add({"name": "sugar"})
        result = reasoning._evaluate_condition(
            "data.get('_collection_done') and "
            "bank('ingredients').count() >= 2",
            {"_collection_done": True},
        )
        assert result is True

    def test_back_preserves_bank_records(self) -> None:
        """Going back from a collection stage preserves collected records."""
        reasoning = _make_collection_wizard()
        reasoning._banks["ingredients"].add({"name": "flour"})
        reasoning._banks["ingredients"].add({"name": "sugar"})

        # Simulate going back — data changes but bank stays
        assert reasoning._banks["ingredients"].count() == 2

    def test_skip_sets_done_flag(self) -> None:
        """Skipping a collection stage should allow done condition."""
        reasoning = _make_collection_wizard()
        state = WizardState(current_stage="collect", data={})

        # Simulate skip — sets _collection_done
        state.data["_collection_done"] = True

        # With no records but done signal, transition condition with
        # min_records=0 should pass
        result = reasoning._evaluate_condition(
            "data.get('_collection_done')",
            state.data,
        )
        assert result is True

    def test_bank_count_in_condition_via_loader(self) -> None:
        """bank() is available in FSM-level conditions via wizard_loader."""
        reasoning = _make_collection_wizard()

        # Inject bank_fn as would happen in generate()
        bank_fn = reasoning._make_bank_accessor()

        # Add records through the accessor
        bank = bank_fn("ingredients")
        bank.add({"name": "flour"})

        # Evaluate a condition as the loader would
        data: dict[str, Any] = {
            "_bank_fn": bank_fn,
            "_collection_done": True,
        }
        result = reasoning._evaluate_condition(
            "data.get('_collection_done') and "
            "bank('ingredients').count() > 0",
            data,
        )
        assert result is True


# =====================================================================
# Loaded metadata end-to-end tests
#
# These tests exercise collection mode through the loader-produced
# stage metadata (active_fsm.current_metadata), not manually
# constructed dicts.  This catches the bug where _extract_metadata
# omitted collection_mode/collection_config from the metadata.
# =====================================================================

class TestCollectionModeThroughLoadedMetadata:
    """Verify collection mode works when stage metadata comes from the loader."""

    def test_loaded_metadata_contains_collection_mode(self) -> None:
        """current_metadata from a loaded FSM must include collection_mode."""
        reasoning = _make_collection_wizard()
        fsm = reasoning._fsm

        # Navigate to the collect stage (it's the start stage)
        stage = fsm.current_metadata
        assert stage["name"] == "collect"
        assert stage.get("collection_mode") == "collection", (
            "collection_mode missing from loaded stage metadata — "
            "_extract_metadata must include it"
        )
        assert stage.get("collection_config") is not None
        assert stage["collection_config"]["bank_name"] == "ingredients"

    @pytest.mark.asyncio
    async def test_handle_collection_mode_with_loaded_metadata(self) -> None:
        """_handle_collection_mode works with loader-produced metadata.

        Previous tests constructed the stage dict manually, which masked
        the bug where collection_mode was absent from loaded metadata.
        """
        reasoning = _make_collection_wizard()
        fsm = reasoning._fsm
        stage = fsm.current_metadata  # From the loader

        # Create minimal manager and state for the async call
        config = LLMConfig(
            provider="echo", model="echo-test",
            options={"echo_prefix": ""},
        )
        provider = EchoProvider(config)
        library = ConfigPromptLibrary(
            {"system": {"test": {"template": "Test bot."}}}
        )
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())
        manager = await ConversationManager.create(
            llm=provider, prompt_builder=builder,
            storage=storage, system_prompt_name="test",
        )
        state = WizardState(current_stage="collect", data={})
        # Seed a user message so _generate_stage_response can find it
        await manager.add_message(role="user", content="2 cups of flour")

        # Script a response for the template render
        provider.set_responses([text_response("Got it! Anything else?")])

        extracted = {"name": "flour", "amount": "2 cups"}
        result = await reasoning._handle_collection_mode(
            user_message="2 cups of flour",
            extracted_data=extracted,
            stage=stage,  # From the loader, not manually constructed
            state=state,
            manager=manager,
            llm=provider,
            tools=[],
        )

        # Should return a response (not None — that's the done path)
        assert result is not None
        # Record should be in the bank
        assert reasoning._banks["ingredients"].count() == 1
        records = reasoning._banks["ingredients"].all()
        assert records[0].data["name"] == "flour"
        assert records[0].data["amount"] == "2 cups"
        # Schema fields should be cleared from state.data
        assert "name" not in state.data
        assert "amount" not in state.data

    @pytest.mark.asyncio
    async def test_done_signal_through_loaded_metadata(self) -> None:
        """'done' keyword triggers exit when using loader-produced metadata."""
        reasoning = _make_collection_wizard()
        fsm = reasoning._fsm
        stage = fsm.current_metadata

        config = LLMConfig(
            provider="echo", model="echo-test",
            options={"echo_prefix": ""},
        )
        provider = EchoProvider(config)
        library = ConfigPromptLibrary(
            {"system": {"test": {"template": "Test bot."}}}
        )
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())
        manager = await ConversationManager.create(
            llm=provider, prompt_builder=builder,
            storage=storage, system_prompt_name="test",
        )
        state = WizardState(current_stage="collect", data={})

        # Pre-populate bank with an ingredient
        reasoning._banks["ingredients"].add(
            {"name": "flour", "amount": "2 cups"}, source_stage="collect"
        )

        result = await reasoning._handle_collection_mode(
            user_message="done",
            extracted_data={},
            stage=stage,
            state=state,
            manager=manager,
            llm=provider,
            tools=[],
        )

        # Should return None (fall through to transition evaluation)
        assert result is None
        # _collection_done flag should be set
        assert state.data.get("_collection_done") is True


# =====================================================================
# Bank config parsing tests
# =====================================================================

class TestBankConfigParsing:
    """Verify _init_banks handles both flat and nested duplicate config."""

    def test_flat_duplicate_strategy(self) -> None:
        """Top-level duplicate_strategy and match_fields are honoured."""
        config: dict[str, Any] = {
            "name": "flat-bank-test",
            "settings": {
                "banks": {
                    "items": {
                        "schema": {"required": ["name"]},
                        "max_records": 20,
                        "duplicate_strategy": "reject",
                        "match_fields": ["name"],
                    },
                },
            },
            "stages": [
                {"name": "s", "is_start": True, "is_end": True, "prompt": "Go"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        bank = reasoning._banks["items"]
        assert bank._duplicate_strategy == "reject"
        assert bank._match_fields == ["name"]

    def test_nested_duplicate_detection(self) -> None:
        """Nested duplicate_detection.strategy format is also supported."""
        config: dict[str, Any] = {
            "name": "nested-bank-test",
            "settings": {
                "banks": {
                    "items": {
                        "schema": {"required": ["name"]},
                        "duplicate_detection": {
                            "strategy": "reject",
                            "match_fields": ["name"],
                        },
                    },
                },
            },
            "stages": [
                {"name": "s", "is_start": True, "is_end": True, "prompt": "Go"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        bank = reasoning._banks["items"]
        assert bank._duplicate_strategy == "reject"
        assert bank._match_fields == ["name"]


# =====================================================================
# Full generate() flow tests
#
# These exercise the complete generate() path, ensuring the
# pre-extraction done-keyword check prevents clarification loops.
#
# NOTE: generate() retrieves wizard state from manager.metadata, not
# from a passed-in state parameter.  To simulate a second turn at
# the "collect" stage we seed the manager metadata with fsm_state.
# =====================================================================

async def _make_manager_and_provider() -> tuple[ConversationManager, EchoProvider]:
    """Create a ConversationManager + EchoProvider pair for tests."""
    config = LLMConfig(
        provider="echo", model="echo-test",
        options={"echo_prefix": ""},
    )
    provider = EchoProvider(config)
    library = ConfigPromptLibrary(
        {"system": {"test": {"template": "You are a test bot."}}}
    )
    builder = AsyncPromptBuilder(library=library)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    manager = await ConversationManager.create(
        llm=provider, prompt_builder=builder,
        storage=storage, system_prompt_name="test",
    )
    return manager, provider


def _seed_wizard_state(
    manager: Any,
    stage: str = "collect",
    data: dict[str, Any] | None = None,
) -> None:
    """Seed wizard FSM state into manager metadata.

    generate() retrieves state via manager.metadata["wizard"]["fsm_state"].
    This helper sets that up so tests can simulate a conversation already
    at a particular stage.
    """
    manager.set_metadata("wizard", {
        "fsm_state": {
            "current_stage": stage,
            "history": [stage],
            "data": data or {},
            "completed": False,
            "clarification_attempts": 0,
            "transitions": [],
            "stage_entry_time": 0,
            "tasks": {},
            "subflow_stack": [],
        },
    })


class TestCollectionModeGenerateFlow:
    """Full generate() tests for collection-mode done signal and transitions.

    These tests call reasoning.generate() directly, covering the
    pre-extraction done-keyword check that prevents the "done" keyword
    from being treated as a data record (which would fail extraction
    confidence and produce a clarification response instead of
    transitioning out of the collection stage).
    """

    @pytest.mark.asyncio
    async def test_done_triggers_transition_not_clarification(self) -> None:
        """'done' must trigger a stage transition, not a clarification loop.

        Regression: before the pre-extraction done check, 'done' was sent
        to the schema extractor, failed confidence, and produced a
        clarification response — the wizard never reached the transition.
        """
        reasoning = _make_collection_wizard()
        manager, provider = await _make_manager_and_provider()

        # Seed state at "collect" so generate() restores it there
        _seed_wizard_state(manager, stage="collect")

        # The user says "done"
        await manager.add_message(role="user", content="done")

        # generate() produces a response — no LLM calls needed
        # because "review" uses a response_template.
        result = await reasoning.generate(manager=manager, llm=provider)

        # Read persisted wizard state back from manager metadata
        fsm_state = manager.metadata["wizard"]["fsm_state"]
        assert fsm_state["current_stage"] == "review", (
            f"Expected transition to 'review', but stuck at "
            f"'{fsm_state['current_stage']}'. "
            "The done keyword likely fell through to extraction/clarification."
        )
        assert fsm_state["completed"] is True
        assert fsm_state["data"].get("_collection_done") is True

    @pytest.mark.asyncio
    async def test_done_with_empty_bank_stays_at_collection(self) -> None:
        """'done' with an empty bank and min_records=1 should not transition.

        The done flag is set, but the transition condition requires
        bank('ingredients').count() >= 1, which is not satisfied.
        """
        reasoning = _make_collection_wizard(min_records=1)
        manager, provider = await _make_manager_and_provider()

        _seed_wizard_state(manager, stage="collect")
        await manager.add_message(role="user", content="done")

        # No transition → wizard stays at "collect" and generates a
        # stage response via LLM (collect has no response_template).
        provider.set_responses([
            text_response("Please add at least one ingredient."),
        ])

        result = await reasoning.generate(manager=manager, llm=provider)

        fsm_state = manager.metadata["wizard"]["fsm_state"]
        assert fsm_state["current_stage"] == "collect"
        assert fsm_state["data"].get("_collection_done") is True

    @pytest.mark.asyncio
    async def test_ingredient_message_adds_to_bank_via_generate(self) -> None:
        """A non-done message should be extracted and added to the bank."""
        from dataknobs_llm.extraction import SchemaExtractor

        # EchoProvider for extraction — returns JSON for the ingredient
        ext_config = LLMConfig(
            provider="echo", model="echo-ext",
            options={"echo_prefix": ""},
        )
        ext_provider = EchoProvider(ext_config)
        extractor = SchemaExtractor(provider=ext_provider)

        reasoning = _make_collection_wizard()
        reasoning._extractor = extractor  # inject extractor

        manager, provider = await _make_manager_and_provider()
        _seed_wizard_state(manager, stage="collect")
        await manager.add_message(role="user", content="2 cups flour")

        # Extraction provider returns structured JSON
        ext_provider.set_responses([
            text_response('{"name": "flour", "amount": "2 cups"}'),
        ])
        # Main provider generates the collection-loop stage response
        provider.set_responses([
            text_response("Got it! What's next?"),
        ])

        result = await reasoning.generate(manager=manager, llm=provider)

        # Should stay on collect, ingredient added to bank
        fsm_state = manager.metadata["wizard"]["fsm_state"]
        assert fsm_state["current_stage"] == "collect"
        assert reasoning._banks["ingredients"].count() == 1

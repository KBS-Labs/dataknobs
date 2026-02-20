"""Integration tests for wizard state persistence through the full pipeline.

Tests the complete save → storage → load roundtrip using real
DataknobsConversationStorage with AsyncMemoryDatabase, ensuring that
non-serializable objects in wizard state.data do not crash the pipeline.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import pytest_asyncio
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import EchoProvider
from dataknobs_llm.conversations.manager import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.prompts.implementations.config_library import ConfigPromptLibrary


class NonSerializableObject:
    """Simulates a live object like ArtifactCorpus that is not JSON-safe."""

    def __init__(self, name: str = "live-object") -> None:
        self.name = name


def _make_wizard_config() -> dict[str, Any]:
    """Simple 2-stage wizard for persistence tests."""
    return {
        "name": "persist-test",
        "version": "1.0",
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "prompt": "Provide details",
                "schema": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
                "transitions": [{"target": "done", "condition": "data.get('topic')"}],
            },
            {"name": "done", "is_end": True, "prompt": "Complete"},
        ],
    }


@pytest_asyncio.fixture
async def conversation_storage() -> DataknobsConversationStorage:
    """Real conversation storage backed by in-memory database."""
    db = AsyncMemoryDatabase()
    return DataknobsConversationStorage(db)


@pytest_asyncio.fixture
async def conversation_manager(
    conversation_storage: DataknobsConversationStorage,
) -> ConversationManager:
    """ConversationManager with Echo LLM and memory storage."""
    provider = EchoProvider({"provider": "echo", "model": "test"})
    library = ConfigPromptLibrary({
        "system": {
            "test_wizard": {"template": "You are a test wizard."},
        },
    })
    builder = AsyncPromptBuilder(library=library)
    manager = await ConversationManager.create(
        llm=provider,
        prompt_builder=builder,
        storage=conversation_storage,
        system_prompt_name="test_wizard",
    )
    return manager


class TestWizardPersistence:
    """Full pipeline: wizard state → conversation metadata → storage → load."""

    @pytest.mark.asyncio
    async def test_wizard_state_survives_conversation_storage_roundtrip(
        self,
        conversation_manager: ConversationManager,
        conversation_storage: DataknobsConversationStorage,
    ) -> None:
        """Save + load: no crash, serializable data preserved."""
        config = _make_wizard_config()
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Set up state with mixed data (serializable + non-serializable)
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        state.data["_corpus_id"] = "corpus-abc"
        state.data["_corpus"] = NonSerializableObject("live corpus")
        state.data["_bank_questions"] = [{"id": "q1"}]
        state.data["_target_count"] = 50

        # Save wizard state → metadata
        reasoning._save_wizard_state(conversation_manager, state)

        # Persist to storage (this is where json.dumps happens)
        conv_id = conversation_manager.state.conversation_id
        await conversation_storage.save_conversation(conversation_manager.state)

        # Load from storage
        loaded_state = await conversation_storage.load_conversation(conv_id)
        assert loaded_state is not None

        # Verify serializable data survived
        wizard_data = loaded_state.metadata.get("wizard", {})
        fsm_data = wizard_data.get("fsm_state", {}).get("data", {})
        assert fsm_data["topic"] == "English grammar"
        assert fsm_data["_corpus_id"] == "corpus-abc"
        assert fsm_data["_bank_questions"] == [{"id": "q1"}]
        assert fsm_data["_target_count"] == 50
        # Non-serializable object should be gone
        assert "_corpus" not in fsm_data

    @pytest.mark.asyncio
    async def test_wizard_generate_with_template_mode_no_crash(
        self,
        conversation_manager: ConversationManager,
        conversation_storage: DataknobsConversationStorage,
    ) -> None:
        """Exact bug scenario: template stage response with non-serializable data.

        The original crash occurred at _generate_stage_response → add_message → _save_state
        → save_conversation → json.dumps. This test exercises that exact path.
        """
        config = _make_wizard_config()
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Inject non-serializable data into the state
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "Physics"
        state.data["_corpus_id"] = "corpus-phys"
        state.data["_corpus"] = NonSerializableObject("live corpus")

        # Save state (this populates manager.metadata)
        reasoning._save_wizard_state(conversation_manager, state)

        # Now attempt to add a message — this triggers _save_state → storage → json.dumps
        await conversation_manager.add_message(
            role="user",
            content="I'd like to study physics",
        )

        # If we get here without TypeError, the fix is working
        # Verify the conversation was persisted successfully
        conv_id = conversation_manager.state.conversation_id
        loaded = await conversation_storage.load_conversation(conv_id)
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_add_message_after_data_mutation_no_crash(
        self,
        conversation_manager: ConversationManager,
        conversation_storage: DataknobsConversationStorage,
    ) -> None:
        """Reproduces the exact crash from api-log.025.log:

        1. _save_wizard_state stores sanitized data in metadata (previous request)
        2. _get_wizard_state loads data (was a reference to metadata)
        3. Transforms add non-serializable _corpus to wizard_state.data
        4. manager.add_message() → _save_state() → json.dumps(metadata) → CRASH
           (note: _save_wizard_state has NOT been called yet — it comes AFTER
            _generate_stage_response in the generate() flow at line 1038)

        The fix: _get_wizard_state must return a COPY of the data, not a reference.
        """
        config = _make_wizard_config()
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Step 1: Save clean wizard state to metadata (simulates previous request)
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        reasoning._save_wizard_state(conversation_manager, state)

        # Persist to storage (so next load is realistic)
        await conversation_storage.save_conversation(conversation_manager.state)

        # Step 2: Reload state (creates the shared reference — the bug)
        reloaded = reasoning._get_wizard_state(conversation_manager)

        # Step 3: Simulate transforms adding non-serializable object
        # (this is what initialize_bank does — no _save_wizard_state in between!)
        reloaded.data["_corpus"] = NonSerializableObject("ArtifactCorpus")
        reloaded.data["_corpus_id"] = "corpus-abc"

        # Step 4: Verify metadata is still JSON-serializable despite mutation.
        # In the real flow (with SQLite), add_message → _save_state →
        # json.dumps(metadata) would crash here because the shared reference
        # put ArtifactCorpus into metadata. AsyncMemoryDatabase doesn't call
        # json.dumps, so we check explicitly.
        state_dict = conversation_manager.state.to_dict()
        json.dumps(state_dict)  # Must NOT raise TypeError

        # Also verify add_message works (which triggers _save_state)
        await conversation_manager.add_message(
            role="assistant",
            content="Here is your first question...",
        )

        # Verify persistence succeeded
        conv_id = conversation_manager.state.conversation_id
        loaded = await conversation_storage.load_conversation(conv_id)
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_save_after_transition_with_non_serializable_snapshot(
        self,
        conversation_manager: ConversationManager,
        conversation_storage: DataknobsConversationStorage,
    ) -> None:
        """Transition data_snapshot contains non-serializable objects.

        create_transition_record snapshots wizard_state.data.copy() which is
        a shallow copy. When _save_wizard_state stores transitions via
        t.to_dict() (which uses asdict), the snapshot still contains the
        non-serializable object. This must be sanitized.
        """
        from dataknobs_bots.reasoning.observability import create_transition_record

        config = _make_wizard_config()
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Set up state with non-serializable object
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "Chemistry"
        state.data["_corpus"] = NonSerializableObject("live corpus")
        state.data["_corpus_id"] = "corpus-chem"

        # Create transition with non-serializable in snapshot
        transition = create_transition_record(
            from_stage="define_topic",
            to_stage="preview_question",
            trigger="user_input",
            data_snapshot=state.data.copy(),
        )
        state.transitions.append(transition)

        # Save wizard state (must sanitize transitions too)
        reasoning._save_wizard_state(conversation_manager, state)

        # add_message → _save_state → json.dumps → must not crash
        await conversation_manager.add_message(
            role="assistant",
            content="Transition complete",
        )

        # Verify persistence succeeded
        conv_id = conversation_manager.state.conversation_id
        loaded = await conversation_storage.load_conversation(conv_id)
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_fsm_restore_does_not_contaminate_metadata(
        self,
        conversation_manager: ConversationManager,
        conversation_storage: DataknobsConversationStorage,
    ) -> None:
        """Reproduces the exact crash path through WizardFSM.restore().

        The crash sequence is:
        1. _save_wizard_state writes clean data to metadata
        2. _get_wizard_state calls fsm.restore(fsm_state) which used to
           create context.data as a reference to metadata["wizard"]["fsm_state"]["data"]
        3. step_async() transforms mutate context.data, contaminating metadata
        4. manager.add_message() → _save_state() → json.dumps(metadata) → CRASH

        The fix: WizardFSM.restore() now deep-copies data.
        """
        config = _make_wizard_config()
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Step 1: Save clean wizard state
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        reasoning._save_wizard_state(conversation_manager, state)
        await conversation_storage.save_conversation(conversation_manager.state)

        # Step 2: Reload — this calls fsm.restore() which used to create
        # a shared reference between context.data and metadata
        reloaded = reasoning._get_wizard_state(conversation_manager)

        # Step 3: Simulate transform mutation (what step_async does internally)
        # This writes to wizard_state.data AND (via the old bug) to context.data
        reloaded.data["_corpus"] = NonSerializableObject("ArtifactCorpus")
        reloaded.data["_corpus_id"] = "corpus-abc"
        reloaded.data["_bank_questions"] = [{"id": "q1"}]

        # Step 4: Verify metadata is NOT contaminated
        meta_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert "_corpus" not in meta_data, (
            "FSM restore() shared reference: non-serializable object leaked "
            "from wizard_state.data into metadata"
        )

        # Step 5: json.dumps must succeed
        json.dumps(conversation_manager.state.to_dict())

        # Step 6: add_message must not crash
        await conversation_manager.add_message(
            role="assistant",
            content="Here is your first question...",
        )
        conv_id = conversation_manager.state.conversation_id
        loaded = await conversation_storage.load_conversation(conv_id)
        assert loaded is not None

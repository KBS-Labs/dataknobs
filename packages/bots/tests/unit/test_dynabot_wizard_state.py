"""Tests for DynaBot wizard state API (enhancement 2g).

Tests the get_wizard_state(), _normalize_wizard_state(), and
normalize_wizard_state() methods for public wizard state access.

TestNormalizeWizardState and TestCanonicalSchema test the pure
normalize_wizard_state() function.

TestGetWizardState tests the DynaBot.get_wizard_state() method, which
looks up state from cache then falls back to storage. Edge cases
(None metadata, empty metadata, nested format) use lightweight
SimpleNamespace objects injected into the cache since they test
get_wizard_state's lookup logic, not wizard flow behavior.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from dataknobs_bots import DynaBot
from dataknobs_bots.bot.base import normalize_wizard_state
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationNode,
    ConversationState,
    DataknobsConversationStorage,
)
from dataknobs_llm.llm.base import LLMMessage
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary
from dataknobs_structures.tree import Tree


class TestNormalizeWizardState:
    """Tests for the module-level normalize_wizard_state() function."""

    def test_normalize_flat_format(self) -> None:
        """Verify normalization of flat (new) wizard state format."""
        wizard_meta = {
            "current_stage": "configure",
            "stage_index": 1,
            "total_stages": 3,
            "progress": 0.33,
            "completed": False,
            "data": {"name": "test"},
            "can_skip": True,
            "can_go_back": True,
            "suggestions": ["Skip", "Continue"],
            "history": ["welcome", "configure"],
        }

        result = normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "configure"
        assert result["stage_index"] == 1
        assert result["total_stages"] == 3
        assert result["progress"] == 0.33
        assert result["completed"] is False
        assert result["data"] == {"name": "test"}
        assert result["can_skip"] is True
        assert result["can_go_back"] is True
        assert result["suggestions"] == ["Skip", "Continue"]
        assert result["history"] == ["welcome", "configure"]

    def test_normalize_nested_fsm_state_format(self) -> None:
        """Verify normalization of nested (legacy) fsm_state format.

        The nested dict is ``WizardState.to_dict()``, so the three keys
        below are what a reader can actually find there.  ``stage_index``
        is not among them — it is derived from the stage by whichever
        writer builds the flat metadata — and asserting it was readable
        here pinned a fallback that no writer could ever satisfy.
        """
        wizard_meta = {
            "fsm_state": {
                "current_stage": "step_2",
                "data": {"field": "value"},
                "history": ["step_1", "step_2"],
            }
        }

        result = normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "step_2"
        assert result["data"] == {"field": "value"}
        assert result["history"] == ["step_1", "step_2"]
        assert result["stage_index"] == 0

    def test_normalize_prefers_direct_over_nested(self) -> None:
        """Verify direct fields take precedence over fsm_state."""
        wizard_meta = {
            "current_stage": "direct_stage",
            "data": {"direct": True},
            "fsm_state": {
                "current_stage": "nested_stage",
                "data": {"nested": True},
            },
        }

        result = normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "direct_stage"
        assert result["data"] == {"direct": True}

    def test_normalize_stage_fallback_to_old_format(self) -> None:
        """Verify 'stage' field is used as fallback for current_stage."""
        wizard_meta = {
            "stage": "old_format_stage",
            "data": {},
        }

        result = normalize_wizard_state(wizard_meta)

        assert result["current_stage"] == "old_format_stage"

    def test_normalize_defaults(self) -> None:
        """Verify default values are applied for missing fields."""
        wizard_meta: dict[str, Any] = {}

        result = normalize_wizard_state(wizard_meta)

        assert result["current_stage"] is None
        assert result["stage_index"] == 0
        assert result["total_stages"] == 0
        assert result["progress"] == 0.0
        assert result["completed"] is False
        assert result["data"] == {}
        assert result["can_skip"] is False
        assert result["can_go_back"] is True
        assert result["suggestions"] == []
        assert result["history"] == []

    def test_normalize_canonical_schema(self) -> None:
        """Verify all canonical fields are present in output."""
        wizard_meta = {"current_stage": "test"}

        result = normalize_wizard_state(wizard_meta)

        expected_keys = {
            "current_stage",
            "stage_index",
            "total_stages",
            "progress",
            "completed",
            "data",
            "can_skip",
            "can_go_back",
            "suggestions",
            "history",
            "stages",
            "subflow_depth",
        }
        assert set(result.keys()) == expected_keys


class TestInstanceNormalizeWizardState:
    """Tests that DynaBot._normalize_wizard_state delegates correctly."""

    @pytest.fixture
    def bot(self) -> DynaBot:
        """Create a minimal DynaBot for testing normalization."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        library = ConfigPromptLibrary(
            {
                "system": {"assistant": {"template": "You are a bot."}},
            }
        )
        builder = AsyncPromptBuilder(library=library)
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())
        return DynaBot(
            llm=provider,
            prompt_builder=builder,
            conversation_storage=storage,
        )

    def test_instance_delegates_to_module_function(self, bot: DynaBot) -> None:
        """Verify _normalize_wizard_state delegates to normalize_wizard_state."""
        wizard_meta = {
            "current_stage": "test_stage",
            "stage_index": 2,
            "total_stages": 5,
        }

        instance_result = bot._normalize_wizard_state(wizard_meta)
        module_result = normalize_wizard_state(wizard_meta)

        assert instance_result == module_result


@pytest.mark.asyncio
class TestGetWizardState:
    """Tests for DynaBot.get_wizard_state() method.

    Edge cases use lightweight SimpleNamespace objects injected into the
    conversation cache to test get_wizard_state's lookup logic without
    exercising full wizard flow.
    """

    @pytest.fixture
    def storage(self) -> DataknobsConversationStorage:
        """Create a real conversation storage for testing."""
        return DataknobsConversationStorage(AsyncMemoryDatabase())

    @pytest.fixture
    def bot(self, storage: DataknobsConversationStorage) -> DynaBot:
        """Create a minimal DynaBot for testing."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        library = ConfigPromptLibrary(
            {
                "system": {"assistant": {"template": "You are a bot."}},
            }
        )
        builder = AsyncPromptBuilder(library=library)
        return DynaBot(
            llm=provider,
            prompt_builder=builder,
            conversation_storage=storage,
        )

    async def test_returns_none_for_unknown_conversation(self, bot: DynaBot) -> None:
        """Verify None is returned for non-existent conversation."""
        result = await bot.get_wizard_state("unknown-conv-id")
        assert result is None

    async def test_returns_none_when_no_wizard_metadata(self, bot: DynaBot) -> None:
        """Verify None is returned when conversation has no wizard metadata."""
        manager = SimpleNamespace(metadata={"other": "data"})
        bot._conversation_managers["conv-123"] = manager  # type: ignore[assignment]

        result = await bot.get_wizard_state("conv-123")
        assert result is None

    async def test_returns_none_when_metadata_empty(self, bot: DynaBot) -> None:
        """Verify None is returned when metadata is empty."""
        manager = SimpleNamespace(metadata={})
        bot._conversation_managers["conv-123"] = manager  # type: ignore[assignment]

        result = await bot.get_wizard_state("conv-123")
        assert result is None

    async def test_returns_none_when_metadata_is_none(self, bot: DynaBot) -> None:
        """Verify None is returned when metadata is None."""
        manager = SimpleNamespace(metadata=None)
        bot._conversation_managers["conv-123"] = manager  # type: ignore[assignment]

        result = await bot.get_wizard_state("conv-123")
        assert result is None

    async def test_returns_normalized_state_via_harness(self) -> None:
        """Verify get_wizard_state returns normalized state via real bot flow.

        Uses 2 required fields to bypass verbatim capture (single-field
        schemas skip LLM extraction and capture the raw user message).
        """
        config = (
            WizardConfigBuilder("state-test")
            .stage("gather", is_start=True, prompt="Tell me your name and topic.")
            .field("name", field_type="string", required=True)
            .field("topic", field_type="string", required=True)
            .transition("done", "data.get('name') and data.get('topic')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice", "topic": "math"}]],
        ) as harness:
            await harness.chat("My name is Alice and I like math")
            state = harness.wizard_state

            assert state is not None
            assert state["data"]["name"] == "Alice"
            assert state["data"]["topic"] == "math"
            assert "current_stage" in state
            assert "stage_index" in state
            assert "progress" in state

    async def test_returns_normalized_state_from_nested_format(self, bot: DynaBot) -> None:
        """Verify legacy nested format is normalized correctly.

        ``fsm_state`` is ``WizardState.to_dict()``, so the fixture carries
        the keys that serializer emits and no others.  It deliberately has
        no ``stage_index``: the position is derived from the stage rather
        than stored beside it, and reading it out of ``fsm_state`` — which
        this test used to assert — described a payload no writer produces.
        """
        manager = SimpleNamespace(
            metadata={
                "wizard": {
                    "fsm_state": {
                        "current_stage": "configure",
                        "data": {"config": "value"},
                        "history": ["welcome", "configure"],
                    }
                }
            }
        )
        bot._conversation_managers["conv-123"] = manager  # type: ignore[assignment]

        result = await bot.get_wizard_state("conv-123")

        assert result is not None
        assert result["current_stage"] == "configure"
        assert result["data"] == {"config": "value"}
        assert result["history"] == ["welcome", "configure"]
        # The flat key is absent here, so the caller gets the documented
        # default rather than a position invented from the nested dict.
        assert result["stage_index"] == 0

    async def test_fallback_to_storage_when_not_cached(
        self,
        bot: DynaBot,
        storage: DataknobsConversationStorage,
    ) -> None:
        """Verify get_wizard_state falls back to storage when not in cache."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Wizard bot"),
            node_id="",
        )
        tree = Tree(root_node)
        state = ConversationState(
            conversation_id="conv-storage",
            message_tree=tree,
            metadata={
                "wizard": {
                    "current_stage": "review",
                    "stage_index": 2,
                    "total_stages": 4,
                    "progress": 0.5,
                    "completed": False,
                    "data": {"name": "stored"},
                }
            },
        )
        await storage.save_conversation(state)

        result = await bot.get_wizard_state("conv-storage")

        assert result is not None
        assert result["current_stage"] == "review"
        assert result["stage_index"] == 2
        assert result["data"] == {"name": "stored"}

    async def test_cache_takes_precedence_over_storage(
        self,
        bot: DynaBot,
        storage: DataknobsConversationStorage,
    ) -> None:
        """Verify in-memory cache wins over storage when both have data."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Wizard bot"),
            node_id="",
        )
        tree = Tree(root_node)
        state = ConversationState(
            conversation_id="conv-both",
            message_tree=tree,
            metadata={
                "wizard": {
                    "current_stage": "old_stage",
                    "data": {"version": "stale"},
                }
            },
        )
        await storage.save_conversation(state)

        manager = SimpleNamespace(
            metadata={
                "wizard": {
                    "current_stage": "new_stage",
                    "data": {"version": "fresh"},
                }
            }
        )
        bot._conversation_managers["conv-both"] = manager  # type: ignore[assignment]

        result = await bot.get_wizard_state("conv-both")

        assert result is not None
        assert result["current_stage"] == "new_stage"
        assert result["data"] == {"version": "fresh"}

    async def test_returns_none_when_not_in_cache_or_storage(self, bot: DynaBot) -> None:
        """Verify None is returned when conversation is nowhere."""
        result = await bot.get_wizard_state("totally-missing")
        assert result is None


class TestCanonicalSchema:
    """Tests verifying the canonical wizard state schema."""

    def test_all_canonical_fields_present(self) -> None:
        """Verify normalized state always has all canonical fields."""
        wizard_meta = {"current_stage": "test"}

        result = normalize_wizard_state(wizard_meta)

        assert "current_stage" in result
        assert "stage_index" in result
        assert "total_stages" in result
        assert "progress" in result
        assert "completed" in result
        assert "data" in result
        assert "can_skip" in result
        assert "can_go_back" in result
        assert "suggestions" in result
        assert "history" in result

    def test_field_types(self) -> None:
        """Verify canonical fields have correct types."""
        wizard_meta = {
            "current_stage": "test",
            "stage_index": 1,
            "total_stages": 3,
            "progress": 0.5,
            "completed": False,
            "data": {"key": "value"},
            "can_skip": True,
            "can_go_back": False,
            "suggestions": ["option1", "option2"],
            "history": ["stage1", "stage2"],
        }

        result = normalize_wizard_state(wizard_meta)

        assert isinstance(result["current_stage"], str)
        assert isinstance(result["stage_index"], int)
        assert isinstance(result["total_stages"], int)
        assert isinstance(result["progress"], float)
        assert isinstance(result["completed"], bool)
        assert isinstance(result["data"], dict)
        assert isinstance(result["can_skip"], bool)
        assert isinstance(result["can_go_back"], bool)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["history"], list)


class TestTheNormalizerDoesNotHandOutWhatItRead:
    """The normalized dict is a hand-out, and it aliased its source.

    ``normalize_wizard_state`` reads ``data``, ``history``, ``suggestions``
    and ``stages`` straight off ``wizard_meta`` and puts the *same objects*
    in its result.  On the fast path ``wizard_meta`` is
    ``manager.metadata["wizard"]`` -- live, persisted conversation state --
    so a caller writing into the dict this returns rewrites it.

    Both consumers are exposed: ``DynaBot.get_wizard_state()``, which is
    public and documented as returning a state dict, and
    ``WizardReasoning.snapshot_from_metadata``, which feeds a snapshot the
    type documents as read-only.  Copying here is what makes both true,
    which is why the fix is on the reader rather than on either caller.
    """

    def test_the_normalized_containers_are_not_the_source_objects(self) -> None:
        """A pure unit on the reader itself."""
        wizard_meta: dict[str, Any] = {
            "current_stage": "gather",
            "data": {"profile": {"city": "Boston"}},
            "history": ["welcome"],
            "suggestions": ["yes", "no"],
            "stages": [{"name": "gather", "label": "Gather", "status": "current"}],
        }

        result = normalize_wizard_state(wizard_meta)

        assert result["data"] is not wizard_meta["data"]
        assert result["history"] is not wizard_meta["history"]
        assert result["suggestions"] is not wizard_meta["suggestions"]
        assert result["stages"] is not wizard_meta["stages"]
        assert result["stages"][0] is not wizard_meta["stages"][0], (
            "the roadmap entries are dicts; copying only the outer list "
            "leaves each entry pointing at the source"
        )
        assert result["data"]["profile"] is not wizard_meta["data"]["profile"], (
            "a nested collected value is where a shallow copy stops"
        )

    def test_writing_through_the_normalized_dict_does_not_reach_the_source(
        self,
    ) -> None:
        """Identity is the mechanism; this is the effect."""
        wizard_meta: dict[str, Any] = {
            "current_stage": "gather",
            "data": {"profile": {"city": "Boston"}},
            "history": ["welcome"],
            "suggestions": ["yes"],
            "stages": [{"name": "gather", "label": "Gather", "status": "current"}],
        }

        result = normalize_wizard_state(wizard_meta)
        result["data"]["profile"]["city"] = "tampered"
        result["history"].append("injected")
        result["suggestions"].append("injected")
        result["stages"][0]["status"] = "tampered"

        assert wizard_meta["data"]["profile"]["city"] == "Boston"
        assert wizard_meta["history"] == ["welcome"]
        assert wizard_meta["suggestions"] == ["yes"]
        assert wizard_meta["stages"][0]["status"] == "current"

    @pytest.mark.asyncio
    async def test_the_public_state_does_not_alias_persisted_metadata(self) -> None:
        """The consumer that made this a public-surface defect.

        ``get_wizard_state`` reads the in-memory manager first, so what it
        normalized was the live metadata dict itself -- not a copy loaded
        from storage.
        """
        config = (
            WizardConfigBuilder("alias-test")
            .stage("gather", is_start=True, prompt="Tell me your name and topic.")
            .field("name", field_type="string", required=True)
            .field("topic", field_type="string", required=True)
            .transition("done", "data.get('name') and data.get('topic')")
            .stage("done", is_end=True, prompt="All done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[[{"name": "Alice", "topic": "math"}]],
        ) as harness:
            await harness.chat("My name is Alice and I like math")
            conversation_id = harness.context.conversation_id
            manager = harness.bot.get_conversation_manager(conversation_id)

            state = await harness.bot.get_wizard_state(conversation_id)
            assert state is not None
            state["data"]["injected"] = True
            if state["stages"]:
                state["stages"][0]["status"] = "tampered"

            wizard_meta = manager.metadata["wizard"]
            persisted_data = wizard_meta.get("data") or wizard_meta["fsm_state"]["data"]
            assert "injected" not in persisted_data, (
                "writing into the public state dict rewrote persisted wizard state"
            )
            if wizard_meta.get("stages"):
                assert wizard_meta["stages"][0]["status"] != "tampered", (
                    "writing into the public roadmap rewrote persisted wizard state"
                )

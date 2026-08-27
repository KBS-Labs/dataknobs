"""Tests for ToolExecutionContext and ToolWizardState.

These tests build a **real** ``ConversationManager`` over a real
``ConversationState``.  That matters here rather than being ceremony: the
context factory reads ``manager.state``, and a stub that answers every
attribute reports a published live-wizard channel on a manager that never
published one.
"""

import dataclasses
import importlib
import warnings

import pytest

import dataknobs_llm.tools
import dataknobs_llm.tools.context as context_module

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
)
from dataknobs_llm.conversations.storage import ConversationNode, ConversationState
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.llm.base import LLMMessage
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary
from dataknobs_llm.tools.context import (
    ToolExecutionContext,
    ToolWizardState,
)
from dataknobs_structures.tree import Tree


WIZARD_METADATA = {
    "wizard": {
        "fsm_state": {
            "current_stage": "configure",
            "data": {"domain_id": "test-bot"},
            "history": ["welcome", "configure"],
            "completed": False,
        }
    }
}


def make_manager(
    metadata: dict | None = None,
    conversation_id: str = "conv-abc",
    with_state: bool = True,
) -> ConversationManager:
    """Build a real ConversationManager over a real ConversationState.

    Args:
        metadata: Conversation metadata to seed onto the state.
        conversation_id: Conversation id for the state.
        with_state: When False, build the manager before its state has
            materialized — the shape a manager has before its first
            message.
    """
    state = None
    if with_state:
        root = ConversationNode(
            message=LLMMessage(role="system", content="You are helpful"),
            node_id="",
        )
        state = ConversationState(
            conversation_id=conversation_id,
            message_tree=Tree(root),
            metadata=dict(metadata or {}),
        )
    return ConversationManager(
        llm=EchoProvider(LLMConfig(provider="echo", model="echo-model")),
        prompt_builder=AsyncPromptBuilder(library=ConfigPromptLibrary({})),
        storage=DataknobsConversationStorage(AsyncMemoryDatabase()),
        state=state,
        conversation_id=conversation_id,
    )


@pytest.fixture
def manager() -> ConversationManager:
    """A manager with no wizard state."""
    return make_manager({"some_key": "some_value"})


@pytest.fixture
def wizard_manager() -> ConversationManager:
    """A manager carrying persisted wizard metadata and nothing live."""
    return make_manager(WIZARD_METADATA, conversation_id="conv-xyz")


class TestToolWizardState:
    """Tests for ToolWizardState."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        snapshot = ToolWizardState()

        assert snapshot.current_stage is None
        assert snapshot.collected_data == {}
        assert snapshot.history == []
        assert snapshot.completed is False
        assert snapshot.stage_metadata == {}

    def test_custom_values(self) -> None:
        """Test creating state with custom values."""
        snapshot = ToolWizardState(
            current_stage="configure",
            collected_data={"name": "test-bot", "template": "tutor"},
            history=["welcome", "select_template", "configure"],
            completed=False,
            stage_metadata={"prompt": "Configure your bot"},
        )

        assert snapshot.current_stage == "configure"
        assert snapshot.collected_data["name"] == "test-bot"
        assert len(snapshot.history) == 3
        assert snapshot.completed is False

    def test_from_manager_metadata_empty(self) -> None:
        """Test creating state from empty metadata."""
        snapshot = ToolWizardState.from_manager_metadata({})

        assert snapshot.current_stage is None
        assert snapshot.collected_data == {}
        assert snapshot.history == []
        assert snapshot.completed is False

    def test_from_manager_metadata_with_wizard_data(self) -> None:
        """Test creating state from metadata with wizard state."""
        metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "review",
                    "data": {"domain_id": "math-tutor", "domain_name": "Math Tutor"},
                    "history": ["welcome", "configure", "review"],
                    "completed": False,
                }
            }
        }

        snapshot = ToolWizardState.from_manager_metadata(metadata)

        assert snapshot.current_stage == "review"
        assert snapshot.collected_data["domain_id"] == "math-tutor"
        assert snapshot.collected_data["domain_name"] == "Math Tutor"
        assert len(snapshot.history) == 3
        assert snapshot.completed is False

    def test_from_manager_metadata_completed(self) -> None:
        """Test creating state from completed wizard."""
        metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "complete",
                    "data": {"domain_id": "my-bot"},
                    "history": ["welcome", "configure", "complete"],
                    "completed": True,
                }
            }
        }

        snapshot = ToolWizardState.from_manager_metadata(metadata)

        assert snapshot.current_stage == "complete"
        assert snapshot.completed is True

    def test_from_manager_metadata_holds_the_persisted_dict_by_reference(self) -> None:
        """The fallback route shares the metadata dict, it does not copy it.

        Two states built from the same metadata therefore see each other's
        writes.  This is what makes tool writes accumulate *within* a turn
        even though they do not survive it.
        """
        metadata = {"wizard": {"fsm_state": {"data": {"a": 1}}}}

        first = ToolWizardState.from_manager_metadata(metadata)
        second = ToolWizardState.from_manager_metadata(metadata)
        first.collected_data["b"] = 2

        assert first.collected_data is metadata["wizard"]["fsm_state"]["data"]
        assert second.collected_data["b"] == 2

    def test_from_manager_metadata_cannot_supply_stage_metadata(self) -> None:
        """Stage metadata is not persisted, so the fallback leaves it empty."""
        metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "configure",
                    "data": {},
                    "stage_metadata": {"prompt": "ignored - not a persisted key"},
                }
            }
        }

        snapshot = ToolWizardState.from_manager_metadata(metadata)

        assert snapshot.stage_metadata == {}


class TestDeprecatedWizardStateSnapshotAlias:
    """``WizardStateSnapshot`` resolves until 1.0.0, and warns."""

    def test_alias_is_the_renamed_class(self) -> None:
        """The alias is the class itself, not a subclass or a copy."""
        with pytest.warns(DeprecationWarning, match="use ToolWizardState"):
            alias = context_module.WizardStateSnapshot

        assert alias is ToolWizardState

    def test_alias_is_still_exported_from_the_package(self) -> None:
        """Existing import sites keep working without a code change."""
        assert "WizardStateSnapshot" in dataknobs_llm.tools.__all__
        assert "ToolWizardState" in dataknobs_llm.tools.__all__

        with pytest.warns(DeprecationWarning, match="use ToolWizardState"):
            alias = dataknobs_llm.tools.WizardStateSnapshot

        assert alias is dataknobs_llm.tools.ToolWizardState

    def test_importing_the_package_does_not_warn(self) -> None:
        """The warning has to name a caller, not us.

        An eager re-export of the deprecated name would fire the warning
        on every ``import dataknobs_llm.tools``, pointing at our own
        ``__init__`` rather than at the consumer who has not migrated --
        which is noise, not a signal. Serving it from ``__getattr__``
        is what keeps the import quiet.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(dataknobs_llm.tools)

        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []

    def test_instances_built_through_the_alias_are_accepted(self) -> None:
        """A context built with the old name is the same context."""
        with pytest.warns(DeprecationWarning):
            alias = context_module.WizardStateSnapshot

        context = ToolExecutionContext(wizard_state=alias(collected_data={"a": 1}))

        assert isinstance(context.wizard_state, ToolWizardState)
        assert context.wizard_data() == {"a": 1}

    def test_alias_warning_names_the_deprecation_period(self) -> None:
        """Both warnings say when the alias arrived and when it goes.

        "One minor version" is not a schedule a caller can act on: it
        names neither the release that started the clock nor the one that
        stops it, so nothing about it is checkable. Both warning sites
        must carry the period, because a caller reaches the alias through
        either module and only ever sees one of them.
        """
        with pytest.warns(DeprecationWarning) as module_warning:
            _ = context_module.WizardStateSnapshot

        with pytest.warns(DeprecationWarning) as package_warning:
            _ = dataknobs_llm.tools.WizardStateSnapshot

        for caught in (module_warning, package_warning):
            message = str(caught[0].message)
            assert "0.8.0" in message, message
            assert "1.0.0" in message, message

    def test_an_unknown_attribute_still_raises(self) -> None:
        """``__getattr__`` must not turn typos into silent successes."""
        with pytest.raises(AttributeError, match="no attribute"):
            _ = context_module.NoSuchName

        with pytest.raises(AttributeError, match="no attribute"):
            _ = dataknobs_llm.tools.NoSuchName


class TestToolExecutionContext:
    """Tests for ToolExecutionContext."""

    def test_empty_context(self) -> None:
        """Test creating empty context."""
        context = ToolExecutionContext.empty()

        assert context.conversation_id is None
        assert context.user_id is None
        assert context.client_id is None
        assert context.conversation_metadata == {}
        assert context.wizard_state is None
        assert context.request_metadata == {}
        assert context.extra == {}

    def test_custom_context(self) -> None:
        """Test creating context with custom values."""
        wizard_state = ToolWizardState(
            current_stage="configure",
            collected_data={"name": "test"},
        )

        context = ToolExecutionContext(
            conversation_id="conv-123",
            user_id="user-456",
            client_id="client-789",
            conversation_metadata={"key": "value"},
            wizard_state=wizard_state,
            request_metadata={"header": "x-custom"},
            extra={"custom_key": "custom_value"},
        )

        assert context.conversation_id == "conv-123"
        assert context.user_id == "user-456"
        assert context.client_id == "client-789"
        assert context.conversation_metadata["key"] == "value"
        assert context.wizard_state is not None
        assert context.wizard_state.current_stage == "configure"
        assert context.request_metadata["header"] == "x-custom"
        assert context.extra["custom_key"] == "custom_value"

    def test_from_manager_basic(self, manager: ConversationManager) -> None:
        """Test creating context from manager without wizard state."""
        context = ToolExecutionContext.from_manager(manager)

        assert context.conversation_id == "conv-abc"
        assert context.conversation_metadata["some_key"] == "some_value"
        assert context.wizard_state is None

    def test_from_manager_with_wizard_state(self, wizard_manager: ConversationManager) -> None:
        """Test creating context from manager with persisted wizard state."""
        context = ToolExecutionContext.from_manager(wizard_manager)

        assert context.conversation_id == "conv-xyz"
        assert context.wizard_state is not None
        assert context.wizard_state.current_stage == "configure"
        assert context.wizard_state.collected_data["domain_id"] == "test-bot"

    def test_from_manager_with_extra(self, manager: ConversationManager) -> None:
        """Test creating context with extra values."""
        context = ToolExecutionContext.from_manager(
            manager,
            request_metadata={"trace_id": "trace-456"},
            extra={"custom": "data"},
        )

        assert context.request_metadata["trace_id"] == "trace-456"
        assert context.extra["custom"] == "data"

    def test_from_manager_bridges_turn_data(self, manager: ConversationManager) -> None:
        """Per-turn plugin data reaches tools through ``extra``."""
        manager.state.turn_data["seeded"] = "by-the-caller"

        context = ToolExecutionContext.from_manager(manager)

        assert context.extra["turn_data"] is manager.state.turn_data

    def test_from_manager_before_state_materializes(self) -> None:
        """A manager with no state yet yields a context, not an error."""
        context = ToolExecutionContext.from_manager(make_manager(with_state=False))

        assert context.wizard_state is None
        assert context.wizard_data() is None
        assert "turn_data" not in context.extra

    def test_get_from_extra(self) -> None:
        """Test dict-like access to extra values."""
        context = ToolExecutionContext(extra={"key1": "value1", "key2": 42})

        assert context.get("key1") == "value1"
        assert context.get("key2") == 42
        assert context.get("missing") is None
        assert context.get("missing", "default") == "default"

    def test_with_extra_creates_new_context(self) -> None:
        """Test that with_extra creates a new context."""
        original = ToolExecutionContext(
            conversation_id="conv-123",
            extra={"key1": "value1"},
        )

        new_context = original.with_extra(key2="value2", key3="value3")

        # Original is unchanged
        assert "key2" not in original.extra
        assert "key3" not in original.extra

        # New context has all values
        assert new_context.extra["key1"] == "value1"
        assert new_context.extra["key2"] == "value2"
        assert new_context.extra["key3"] == "value3"

        # Other fields are preserved
        assert new_context.conversation_id == "conv-123"

    def test_with_extra_keeps_the_same_wizard_state_object(self) -> None:
        """A derived context reads and writes the same wizard data."""
        live = {"a": 1}
        original = ToolExecutionContext.from_wizard_data(live)

        derived = original.with_extra(key="value")

        assert derived.wizard_state is original.wizard_state
        assert derived.wizard_data() is live

    def test_from_manager_handles_missing_attributes(self) -> None:
        """from_manager degrades on an object with none of the attributes."""
        context = ToolExecutionContext.from_manager(object())

        assert context.conversation_id is None
        assert context.conversation_metadata == {}
        assert context.wizard_state is None


class TestWizardDataAccessor:
    """``wizard_data()`` is the supported way for a tool to reach the data."""

    def test_returns_none_without_wizard_state(self) -> None:
        """``None`` rather than ``{}`` — a throwaway dict hides the failure."""
        assert ToolExecutionContext.empty().wizard_data() is None

    def test_returns_none_outside_a_wizard_conversation(self, manager: ConversationManager) -> None:
        """A tool run outside a wizard can tell that it was."""
        context = ToolExecutionContext.from_manager(manager)

        assert context.wizard_data() is None

    def test_returns_the_persisted_dict_on_the_fallback_path(
        self, wizard_manager: ConversationManager
    ) -> None:
        """With no live view published, the accessor reads the metadata dict."""
        context = ToolExecutionContext.from_manager(wizard_manager)
        persisted = wizard_manager.metadata["wizard"]["fsm_state"]["data"]

        assert context.wizard_data() is persisted

    def test_returns_the_live_dict_when_one_is_published(
        self, wizard_manager: ConversationManager
    ) -> None:
        """The accessor hands back the strategy's own dict, by reference."""
        live = {"domain_id": "live-bot"}
        wizard_manager.state.live_wizard_state = ToolWizardState(collected_data=live)

        context = ToolExecutionContext.from_manager(wizard_manager)

        assert context.wizard_data() is live

    def test_a_write_through_the_accessor_reaches_the_published_dict(
        self, wizard_manager: ConversationManager
    ) -> None:
        """This is the whole point of the channel: the write is not discarded."""
        live: dict = {}
        wizard_manager.state.live_wizard_state = ToolWizardState(collected_data=live)

        ToolExecutionContext.from_manager(wizard_manager).wizard_data()["written"] = True

        assert live == {"written": True}

    def test_from_wizard_data_holds_the_callers_dict(self) -> None:
        """The standalone factory is live too — the caller sees the writes."""
        caller_dict: dict = {}
        context = ToolExecutionContext.from_wizard_data(caller_dict)

        context.wizard_data()["written"] = True

        assert caller_dict == {"written": True}


class TestLiveWizardStateChannel:
    """``ConversationState.live_wizard_state`` and how from_manager reads it."""

    def test_defaults_to_none(self) -> None:
        """A fresh state has published nothing."""
        assert make_manager().state.live_wizard_state is None

    def test_published_view_wins_over_persisted_metadata(
        self, wizard_manager: ConversationManager
    ) -> None:
        """The persisted stage is one turn old; the published one is current."""
        wizard_manager.state.live_wizard_state = ToolWizardState(
            current_stage="review",
            collected_data={"domain_id": "live-bot"},
            history=["welcome", "configure", "review"],
        )

        context = ToolExecutionContext.from_manager(wizard_manager)

        assert wizard_manager.metadata["wizard"]["fsm_state"]["current_stage"] == "configure"
        assert context.wizard_state is wizard_manager.state.live_wizard_state
        assert context.wizard_state.current_stage == "review"
        assert context.wizard_data() == {"domain_id": "live-bot"}

    def test_published_view_supplies_stage_metadata(
        self, wizard_manager: ConversationManager
    ) -> None:
        """The field the metadata route cannot fill is filled by the publisher."""
        wizard_manager.state.live_wizard_state = ToolWizardState(
            current_stage="configure",
            stage_metadata={"prompt": "Configure your bot", "schema": {"name": "str"}},
        )

        context = ToolExecutionContext.from_manager(wizard_manager)

        assert context.wizard_state is not None
        assert context.wizard_state.stage_metadata["prompt"] == "Configure your bot"

    def test_published_view_reaches_a_conversation_with_no_wizard_metadata(self) -> None:
        """Turn one: nothing is persisted yet, and the tool still sees state."""
        manager = make_manager({})
        live = {"name": "Alice"}
        manager.state.live_wizard_state = ToolWizardState(
            current_stage="gather", collected_data=live
        )

        context = ToolExecutionContext.from_manager(manager)

        assert "wizard" not in manager.metadata
        assert context.wizard_state is not None
        assert context.wizard_state.current_stage == "gather"
        assert context.wizard_data() is live

    def test_every_context_in_the_turn_sees_the_same_object(
        self, wizard_manager: ConversationManager
    ) -> None:
        """Two tools in one turn read and write one dict, not two copies."""
        wizard_manager.state.live_wizard_state = ToolWizardState(collected_data={})

        first = ToolExecutionContext.from_manager(wizard_manager)
        second = ToolExecutionContext.from_manager(wizard_manager)
        first.wizard_data()["written_by_first"] = True

        assert second.wizard_data()["written_by_first"] is True

    def test_clearing_the_channel_restores_the_fallback(
        self, wizard_manager: ConversationManager
    ) -> None:
        """End of turn: the publisher clears, and the metadata route resumes."""
        wizard_manager.state.live_wizard_state = ToolWizardState(current_stage="review")

        wizard_manager.state.live_wizard_state = None
        context = ToolExecutionContext.from_manager(wizard_manager)

        assert context.wizard_state is not None
        assert context.wizard_state.current_stage == "configure"

    def test_is_transient_and_never_serialized(self, wizard_manager: ConversationManager) -> None:
        """The channel must not reach storage, in either direction."""
        wizard_manager.state.live_wizard_state = ToolWizardState(
            collected_data={"secret": "live-only"}
        )
        state = wizard_manager.state

        serialized = state.to_dict()

        assert "live_wizard_state" not in serialized
        assert "live_wizard_state" not in dataclasses.asdict(state)
        assert not any(f.name == "live_wizard_state" for f in dataclasses.fields(state))
        assert ConversationState.from_dict(serialized).live_wizard_state is None

    def test_publishing_does_not_write_into_conversation_metadata(
        self, wizard_manager: ConversationManager
    ) -> None:
        """The live dict and the persisted dict stay separate objects.

        Wizard data is deep-copied on restore precisely so live state and
        persisted metadata cannot share a reference.  This channel exists
        to give tools the live dict *without* reintroducing that sharing.
        """
        live = {"domain_id": "live-bot"}
        wizard_manager.state.live_wizard_state = ToolWizardState(collected_data=live)

        ToolExecutionContext.from_manager(wizard_manager).wizard_data()["new_key"] = 1

        persisted = wizard_manager.metadata["wizard"]["fsm_state"]["data"]
        assert "new_key" not in persisted
        assert persisted is not live

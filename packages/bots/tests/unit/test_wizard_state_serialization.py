"""Tests for wizard state serialization safety.

Verifies that _save_wizard_state produces JSON-serializable metadata,
even when state.data contains non-serializable objects like ArtifactCorpus
or DedupResult dataclass instances.

These tests reproduce the crash:
    TypeError: Object of type ArtifactCorpus is not JSON serializable
that occurs when _save_wizard_state stores state.data raw into metadata
which later flows to json.dumps() via ConversationState.to_dict().

Also tests the transient/persistent partition system that separates
ephemeral wizard data (live objects, per-step errors) from persistent
data (user-collected fields, artifact IDs).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import (
    SubflowContext,
    WizardReasoning,
    WizardState,
    _is_json_safe,
)
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.reasoning.wizard_types import DEFAULT_EPHEMERAL_KEYS, partition_wizard_data
from dataknobs_common.exceptions import SerializationError
from dataknobs_common.serialization import sanitize_for_json
from dataknobs_llm.conversations import ConversationManager


class NonSerializableObject:
    """Simulates a live object like ArtifactCorpus that is not JSON-safe."""

    def __init__(self, name: str = "live-object") -> None:
        self.name = name


@dataclass
class FakeDedupResult:
    """Simulates DedupResult: a dataclass with primitive fields."""

    is_exact_duplicate: bool
    exact_match_id: str | None = None
    similar_items: list[str] = field(default_factory=list)
    recommendation: str = "unique"
    content_hash: str = ""


class Status(Enum):
    ACTIVE = "active"


class Priced:
    """An object that chose its own JSON shape, counting how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    def to_dict(self) -> dict[str, Any]:
        self.calls += 1
        return {"amount": "9.05"}


class Unpriced:
    def to_dict(self) -> dict[str, Any]:
        raise ValueError("no price list loaded")


def _holds_itself() -> dict[str, Any]:
    loop: dict[str, Any] = {"name": "loop"}
    loop["self"] = loop
    return loop


def _make_wizard(
    config: dict[str, Any] | None = None,
) -> WizardReasoning:
    """Create a WizardReasoning instance for tests."""
    if config is None:
        config = {
            "name": "test-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done"},
            ],
        }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


class TestWizardStateSerialization:
    """Tests for wizard state serialization safety."""

    @pytest.mark.asyncio
    async def test_save_state_strips_non_serializable_objects(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Verifies fix for the crash: non-serializable objects in state.data
        are stripped during save, so json.dumps(metadata) succeeds.

        Before this fix, ArtifactCorpus in state.data caused:
            TypeError: Object of type ArtifactCorpus is not JSON serializable
        """
        reasoning = _make_wizard()

        # Simulate what happens after initialize_bank stores _corpus in data
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        state.data["_corpus_id"] = "corpus-abc"
        state.data["_corpus"] = NonSerializableObject("ArtifactCorpus")

        await reasoning._save_wizard_state(conversation_manager, state)

        # Must NOT raise — non-serializable object stripped during save
        json_str = json.dumps(conversation_manager.metadata)
        parsed = json.loads(json_str)
        data = parsed["wizard"]["fsm_state"]["data"]
        assert data["topic"] == "English grammar"
        assert data["_corpus_id"] == "corpus-abc"
        assert "_corpus" not in data

    @pytest.mark.asyncio
    async def test_save_state_preserves_all_serializable_underscore_keys(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Every quiz-style underscore-prefixed serializable key survives save."""
        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "Biology"
        state.data["_corpus_id"] = "corpus-xyz"
        state.data["_bank_questions"] = [{"id": "q1"}, {"id": "q2"}]
        state.data["_bank_question_ids"] = ["art-1", "art-2"]
        state.data["_bank_difficulty_counts"] = {"easy": 1, "medium": 1, "hard": 0}
        state.data["_bank_batch_count"] = 1
        state.data["_target_count"] = 50
        state.data["_current_question"] = {"question_text": "What is DNA?"}
        state.data["_current_question_number"] = 3
        state.data["_dedup_result"] = None

        await reasoning._save_wizard_state(conversation_manager, state)

        saved_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["topic"] == "Biology"
        assert saved_data["_corpus_id"] == "corpus-xyz"
        assert saved_data["_bank_questions"] == [{"id": "q1"}, {"id": "q2"}]
        assert saved_data["_bank_question_ids"] == ["art-1", "art-2"]
        assert saved_data["_bank_difficulty_counts"] == {"easy": 1, "medium": 1, "hard": 0}
        assert saved_data["_bank_batch_count"] == 1
        assert saved_data["_target_count"] == 50
        assert saved_data["_current_question"] == {"question_text": "What is DNA?"}
        assert saved_data["_current_question_number"] == 3

    @pytest.mark.asyncio
    async def test_save_state_converts_dataclass_in_data(
        self, conversation_manager: ConversationManager
    ) -> None:
        """DedupResult dataclass in data becomes a dict after save."""
        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["_dedup_result"] = FakeDedupResult(
            is_exact_duplicate=False,
            content_hash="abc123",
        )

        await reasoning._save_wizard_state(conversation_manager, state)

        saved_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        dedup = saved_data["_dedup_result"]
        assert isinstance(dedup, dict)
        assert dedup["is_exact_duplicate"] is False
        assert dedup["content_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_save_state_none_values_preserved(
        self, conversation_manager: ConversationManager
    ) -> None:
        """data["_dedup_result"] = None remains None after save."""
        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["_dedup_result"] = None
        state.data["_corpus_id"] = None
        state.data["topic"] = "test"

        await reasoning._save_wizard_state(conversation_manager, state)

        saved_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["_dedup_result"] is None
        assert saved_data["_corpus_id"] is None
        assert saved_data["topic"] == "test"

    @pytest.mark.asyncio
    async def test_saved_state_is_json_serializable(
        self, conversation_manager: ConversationManager
    ) -> None:
        """json.dumps(metadata) succeeds with mixed data including
        non-serializable objects and dataclasses.
        """
        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        state.data["_corpus_id"] = "corpus-abc"
        state.data["_corpus"] = NonSerializableObject("live corpus")
        state.data["_dedup_result"] = FakeDedupResult(
            is_exact_duplicate=True,
            exact_match_id="q-old",
        )
        state.data["_bank_questions"] = [{"id": "q1"}]

        await reasoning._save_wizard_state(conversation_manager, state)

        # Must not raise
        json_str = json.dumps(conversation_manager.metadata)
        parsed = json.loads(json_str)

        data = parsed["wizard"]["fsm_state"]["data"]
        assert data["topic"] == "English grammar"
        assert data["_corpus_id"] == "corpus-abc"
        assert "_corpus" not in data  # Non-serializable stripped
        assert data["_dedup_result"]["is_exact_duplicate"] is True
        assert data["_bank_questions"] == [{"id": "q1"}]

    @pytest.mark.asyncio
    async def test_roundtrip_save_and_restore_preserves_data(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Save → restore via _get_wizard_state → all serializable data intact."""
        reasoning = _make_wizard()

        # Set up state with mixed data
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "Chemistry"
        state.data["_corpus_id"] = "corpus-chem"
        state.data["_corpus"] = NonSerializableObject("live")
        state.data["_bank_questions"] = [{"id": "q1", "topic": "atoms"}]
        state.data["_bank_question_ids"] = ["art-1"]
        state.data["_bank_difficulty_counts"] = {"easy": 0, "medium": 1, "hard": 0}

        # Save
        await reasoning._save_wizard_state(conversation_manager, state)

        # Restore
        restored = reasoning._get_wizard_state(conversation_manager)

        assert restored.data["topic"] == "Chemistry"
        assert restored.data["_corpus_id"] == "corpus-chem"
        assert restored.data["_bank_questions"] == [{"id": "q1", "topic": "atoms"}]
        assert restored.data["_bank_question_ids"] == ["art-1"]
        assert restored.data["_bank_difficulty_counts"] == {
            "easy": 0,
            "medium": 1,
            "hard": 0,
        }

    def test_save_state_subflow_parent_data_sanitized(self) -> None:
        """SubflowContext parent_data also stripped of non-serializable objects."""
        parent_data = {
            "topic": "Math",
            "_corpus_id": "corpus-math",
            "_corpus": NonSerializableObject("live corpus"),
            "_bank_questions": [{"id": "q1"}],
        }

        subflow = SubflowContext(
            parent_stage="define_topic",
            parent_data=parent_data,
            parent_history=["define_topic"],
            return_stage="preview_question",
            result_mapping={},
            subflow_network="question_loop",
        )

        serialized = subflow.to_dict()

        # Must be JSON-serializable
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        assert parsed["parent_data"]["topic"] == "Math"
        assert parsed["parent_data"]["_corpus_id"] == "corpus-math"
        assert "_corpus" not in parsed["parent_data"]
        assert parsed["parent_data"]["_bank_questions"] == [{"id": "q1"}]


class TestWizardFSMRestoreDecoupling:
    """Tests for the WizardFSM.restore() shared reference fix.

    Root cause: WizardFSM.restore() extracted data from the input state dict
    without copying, so context.data was a direct reference to
    manager.metadata["wizard"]["fsm_state"]["data"]. When step_async()
    transforms mutated context.data (e.g. adding _corpus), they contaminated
    the metadata dict, causing json.dumps(metadata) to crash.

    Fix: restore() now deep-copies data before creating the execution context.
    """

    def test_restore_decouples_context_data_from_input_state(self) -> None:
        """After restore(), mutating context.data must NOT modify the
        original state dict that was passed to restore().

        This is the FSM-level root cause: restore() used a direct reference
        to state["data"], so any mutation to context.data (by transforms)
        wrote through to the caller's dict (manager.metadata).
        """
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config = {
            "name": "test-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        # Simulate metadata with persisted wizard state
        original_data = {"topic": "English grammar", "_corpus_id": "corpus-abc"}
        state = {
            "current_stage": "welcome",
            "data": original_data,
        }

        fsm.restore(state)

        # Mutate context.data (simulating what transforms do)
        fsm._context.data["_corpus"] = NonSerializableObject("live corpus")
        fsm._context.data["_injected_key"] = "transform output"

        # Original dict must NOT be contaminated
        assert "_corpus" not in original_data, (
            "restore() created a shared reference: context.data mutation "
            "leaked into the original state dict"
        )
        assert "_injected_key" not in original_data

    def test_restore_preserves_original_data_values(self) -> None:
        """restore() deep-copies data, so context.data has the correct
        values but is independent of the original.
        """
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config = {
            "name": "test-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        original_data = {
            "topic": "English grammar",
            "nested": {"key": "value", "list": [1, 2, 3]},
        }
        fsm.restore({"current_stage": "welcome", "data": original_data})

        # Values are preserved
        assert fsm._context.data["topic"] == "English grammar"
        assert fsm._context.data["nested"]["key"] == "value"
        assert fsm._context.data["nested"]["list"] == [1, 2, 3]

        # But deeply independent — mutating nested structures doesn't leak
        fsm._context.data["nested"]["key"] = "mutated"
        assert original_data["nested"]["key"] == "value"


class TestWizardStateSharedReference:
    """Tests for the shared reference bug between wizard_state.data and metadata.

    These test the full pipeline through _get_wizard_state (which calls
    WizardFSM.restore internally), verifying that the combined deepcopy
    in both _get_wizard_state and restore() fully breaks the reference chain.
    """

    @pytest.mark.asyncio
    async def test_get_wizard_state_data_is_decoupled_from_metadata(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Modifying wizard_state.data after _get_wizard_state must NOT
        modify manager.metadata.

        This is the root cause of the serialization crash: _get_wizard_state
        uses a reference to the same dict stored in metadata. When transforms
        add _corpus to wizard_state.data, it appears in metadata too.
        """
        reasoning = _make_wizard()

        # Initial save with clean data
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        await reasoning._save_wizard_state(conversation_manager, state)

        # Load state back — this is where the shared reference is created
        loaded_state = reasoning._get_wizard_state(conversation_manager)

        # Simulate what transforms do: add non-serializable object
        loaded_state.data["_corpus"] = NonSerializableObject("live corpus")
        loaded_state.data["_new_key"] = "injected"

        # The metadata must NOT be contaminated
        fsm_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert "_corpus" not in fsm_data, (
            "Non-serializable object leaked from wizard_state.data "
            "into manager.metadata via shared reference"
        )
        assert "_new_key" not in fsm_data, (
            "New key leaked from wizard_state.data into manager.metadata"
        )

    @pytest.mark.asyncio
    async def test_mutating_data_after_load_does_not_crash_json_dumps(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Load state → add non-serializable to data → json.dumps(metadata) → no crash.

        This reproduces the exact crash from the api-log: after transforms
        add _corpus to wizard_state.data, the next json.dumps(metadata) fails
        because the shared reference puts the ArtifactCorpus in metadata.
        """
        reasoning = _make_wizard()

        # Save clean state
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        await reasoning._save_wizard_state(conversation_manager, state)

        # Reload state (creates the shared reference)
        reloaded = reasoning._get_wizard_state(conversation_manager)

        # Simulate transform adding non-serializable object
        reloaded.data["_corpus"] = NonSerializableObject("ArtifactCorpus")
        reloaded.data["_corpus_id"] = "corpus-abc"

        # json.dumps on metadata must succeed (the key assertion)
        json_str = json.dumps(conversation_manager.metadata)
        parsed = json.loads(json_str)
        assert "wizard" in parsed

    @pytest.mark.asyncio
    async def test_transition_data_snapshot_is_serializable(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Transition records with non-serializable data in snapshot must
        not crash json.dumps after _save_wizard_state.
        """
        from dataknobs_bots.reasoning.observability import create_transition_record

        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "Physics"
        state.data["_corpus"] = NonSerializableObject("live corpus")
        state.data["_corpus_id"] = "corpus-phys"

        # Create transition with data snapshot containing non-serializable
        transition = create_transition_record(
            from_stage="define_topic",
            to_stage="preview_question",
            trigger="user_input",
            data_snapshot=state.data.copy(),
        )
        state.transitions.append(transition)

        # Save must sanitize transitions too
        await reasoning._save_wizard_state(conversation_manager, state)

        # json.dumps on full metadata must succeed
        json_str = json.dumps(conversation_manager.metadata)
        parsed = json.loads(json_str)
        transitions = parsed["wizard"]["fsm_state"]["transitions"]
        assert len(transitions) == 1
        snapshot = transitions[0]["data_snapshot"]
        assert snapshot["topic"] == "Physics"
        assert snapshot["_corpus_id"] == "corpus-phys"
        assert "_corpus" not in snapshot


class TestIsJsonSafe:
    """Tests for the _is_json_safe helper."""

    def test_primitives_are_safe(self) -> None:
        assert _is_json_safe(None)
        assert _is_json_safe(True)
        assert _is_json_safe(42)
        assert _is_json_safe(3.14)
        assert _is_json_safe("hello")

    def test_nested_structures_are_safe(self) -> None:
        assert _is_json_safe({"a": 1, "b": [1, 2, "three"]})
        assert _is_json_safe([{"nested": {"deep": True}}])

    def test_non_serializable_detected(self) -> None:
        assert not _is_json_safe(NonSerializableObject("test"))
        assert not _is_json_safe({"ok": 1, "bad": NonSerializableObject()})
        assert not _is_json_safe([1, 2, NonSerializableObject()])

    def test_tuple_is_safe(self) -> None:
        """Tuples are JSON-safe (serialized as arrays)."""
        assert _is_json_safe((1, 2, 3))

    def test_dataclass_with_safe_fields_is_safe(self) -> None:
        """Dataclasses with JSON-safe fields are accepted (sanitize_for_json converts them)."""
        assert _is_json_safe(FakeDedupResult(is_exact_duplicate=False))

    def test_dataclass_with_non_safe_fields_is_not_safe(self) -> None:
        """Dataclasses containing non-serializable fields are not safe."""
        from dataclasses import dataclass as dc

        @dc
        class BadDataclass:
            obj: Any = None

        assert not _is_json_safe(BadDataclass(obj=NonSerializableObject()))

    def test_values_the_save_converts_are_safe(self) -> None:
        """An Enum, a to_dict() object and a non-str key are saved converted."""
        assert _is_json_safe(Status.ACTIVE)
        assert _is_json_safe({"price": Priced()})
        assert _is_json_safe({1: "aisle 4"})

    @pytest.mark.parametrize(
        "value",
        [math.nan, [math.inf], {"score": -math.inf}],
        ids=["nan", "inf-in-list", "-inf-in-dict"],
    )
    def test_non_finite_floats_are_safe(self, value: Any) -> None:
        """The save writes them as None, so it keeps the value whole."""
        assert _is_json_safe(value)

    def test_a_non_finite_key_is_not_safe(self) -> None:
        assert not _is_json_safe({math.nan: 1})

    def test_unwritable_key_is_not_safe(self) -> None:
        assert not _is_json_safe({(1, 2): "pair"})

    def test_to_dict_that_raises_is_not_safe(self) -> None:
        assert not _is_json_safe(Unpriced())

    def test_a_value_that_holds_itself_is_not_safe(self) -> None:
        """Answered, not a RecursionError."""
        assert not _is_json_safe(_holds_itself())

    @pytest.mark.parametrize(
        "value",
        [
            {"a": 1},
            Status.ACTIVE,
            Priced(),
            Unpriced(),
            FakeDedupResult(is_exact_duplicate=True),
            {1: "x"},
            {(1, 2): "x"},
            [1.5, math.nan],
            {math.inf: 1},
            NonSerializableObject(),
            {1, 2},
        ],
    )
    def test_agrees_with_what_the_save_keeps(self, value: Any) -> None:
        """Safe exactly when the save would keep the whole value."""
        try:
            sanitize_for_json(value, on_drop="error")
        except SerializationError:
            kept = False
        else:
            kept = True
        assert _is_json_safe(value) is kept


class TestPartitionData:
    """Tests for partition_wizard_data(), with a wizard's ephemeral keys."""

    def test_classifies_ephemeral_keys(self) -> None:
        """Known ephemeral keys end up in transient dict."""
        reasoning = _make_wizard()

        data = {
            "topic": "Math",
            "_corpus": NonSerializableObject("live"),
            "_transform_error": "Something failed",
            "_message": "user input",
        }

        persistent, transient, _saved = partition_wizard_data(data, reasoning.ephemeral_keys)

        assert "topic" in persistent
        assert "_corpus" in transient
        assert "_transform_error" in transient
        assert "_message" in transient
        assert "_corpus" not in persistent
        assert "_transform_error" not in persistent

    def test_classifies_persistent_keys(self) -> None:
        """User data and persistent _-prefixed keys stay in persistent dict."""
        reasoning = _make_wizard()

        data = {
            "topic": "English grammar",
            "difficulty": "medium",
            "_corpus_id": "corpus-abc",
            "_bank_questions": [{"id": "q1"}],
            "_target_count": 50,
        }

        persistent, transient, _saved = partition_wizard_data(data, reasoning.ephemeral_keys)

        assert persistent == data  # All should be persistent
        assert transient == {}

    def test_catches_unknown_non_serializable(self) -> None:
        """Non-serializable objects not in EPHEMERAL_KEYS go to transient."""
        reasoning = _make_wizard()

        data = {
            "topic": "Math",
            "_unknown_live_object": NonSerializableObject("surprise"),
            "_corpus_id": "corpus-abc",
        }

        persistent, transient, _saved = partition_wizard_data(data, reasoning.ephemeral_keys)

        assert "topic" in persistent
        assert "_corpus_id" in persistent
        assert "_unknown_live_object" in transient
        assert "_unknown_live_object" not in persistent

    def test_routes_what_the_save_would_drop(self, caplog: Any) -> None:
        """A live object or a value that holds itself stays live, with the reason logged."""
        reasoning = _make_wizard()
        data = {
            "topic": "Math",
            "handle": {"id": 1, "conn": NonSerializableObject()},
            "loop": _holds_itself(),
        }

        with caplog.at_level(logging.WARNING):
            persistent, transient, saved = partition_wizard_data(data, reasoning.ephemeral_keys)

        assert persistent == {"topic": "Math"}
        assert saved == {"topic": "Math"}
        assert set(transient) == {"handle", "loop"}
        assert "handle" in caplog.text
        assert "conn (type=NonSerializableObject)" in caplog.text
        assert "self (cycle)" in caplog.text

    def test_the_warning_names_a_few_problems_and_counts_the_rest(self, caplog: Any) -> None:
        reasoning = _make_wizard()
        handles = [NonSerializableObject() for _ in range(50)]

        with caplog.at_level(logging.WARNING):
            partition_wizard_data({"handles": handles}, reasoning.ephemeral_keys)

        [record] = [r for r in caplog.records if r.name == "dataknobs_bots.reasoning.wizard_types"]
        message = record.getMessage()
        assert "[0] (type=NonSerializableObject)" in message
        assert "[2] (type=NonSerializableObject)" in message
        assert "[3]" not in message
        assert message.endswith("(+47 more)")

    def test_a_nested_nan_keeps_its_key_persistent(self, caplog: Any) -> None:
        """A sparse record is saved with its other values, the NaN as None."""
        reasoning = _make_wizard()
        readings = {"t1": [20.5, math.nan, 21.0], "t2": {"min": math.nan, "max": 30.0}}

        with caplog.at_level(logging.WARNING):
            persistent, transient, saved = partition_wizard_data(
                {"readings": readings}, reasoning.ephemeral_keys
            )

        assert persistent == {"readings": readings}
        assert transient == {}
        assert saved == {"readings": {"t1": [20.5, None, 21.0], "t2": {"min": None, "max": 30.0}}}
        assert [
            r for r in caplog.records if r.name == "dataknobs_bots.reasoning.wizard_types"
        ] == []

    def test_saved_is_the_persistent_part_converted(self) -> None:
        reasoning = _make_wizard()
        data = {"status": Status.ACTIVE, "price": Priced(), "_corpus": "x", "fn": len}

        persistent, _transient, saved = partition_wizard_data(data, reasoning.ephemeral_keys)

        assert saved == sanitize_for_json(persistent)
        assert saved == {"status": "active", "price": {"amount": "9.05"}}

    def test_keeps_what_the_save_converts(self) -> None:
        reasoning = _make_wizard()
        data = {"status": Status.ACTIVE, "price": Priced()}

        persistent, transient, _saved = partition_wizard_data(data, reasoning.ephemeral_keys)

        assert persistent == data
        assert transient == {}

    def test_merges_config_ephemeral_keys(self) -> None:
        """Keys from settings.ephemeral_keys are treated as ephemeral."""
        config = {
            "name": "test-wizard",
            "version": "1.0",
            "settings": {
                "ephemeral_keys": ["_dedup_result", "_review_summary"],
            },
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        data = {
            "topic": "Biology",
            "_dedup_result": {"is_duplicate": False, "score": 0.2},
            "_review_summary": "All good",
            "_corpus_id": "corpus-bio",
        }

        persistent, transient, _saved = partition_wizard_data(data, reasoning.ephemeral_keys)

        assert "topic" in persistent
        assert "_corpus_id" in persistent
        assert "_dedup_result" in transient
        assert "_review_summary" in transient

    def test_ephemeral_keys_include_defaults_and_config(self) -> None:
        """The merged set includes both DEFAULT_EPHEMERAL_KEYS and config keys."""
        config = {
            "name": "test-wizard",
            "version": "1.0",
            "settings": {
                "ephemeral_keys": ["_custom_key"],
            },
            "stages": [
                {
                    "name": "welcome",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        # Framework default keys should be present
        assert "_corpus" in reasoning._ephemeral_keys
        assert "_transform_error" in reasoning._ephemeral_keys
        # Config key should be present
        assert "_custom_key" in reasoning._ephemeral_keys

    @pytest.mark.asyncio
    async def test_save_wizard_state_excludes_transient(
        self, conversation_manager: ConversationManager
    ) -> None:
        """After partition + save, persisted metadata has no ephemeral keys."""
        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        state.data["_corpus_id"] = "corpus-abc"
        state.data["_corpus"] = NonSerializableObject("ArtifactCorpus")
        state.data["_transform_error"] = "Generation failed"
        state.data["_message"] = "user input"

        await reasoning._save_wizard_state(conversation_manager, state)

        # Verify persisted data excludes ephemeral keys
        saved_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["topic"] == "English grammar"
        assert saved_data["_corpus_id"] == "corpus-abc"
        assert "_corpus" not in saved_data
        assert "_transform_error" not in saved_data
        assert "_message" not in saved_data

        # Verify transient is populated on the state object
        assert state.transient.get("_corpus") is not None
        assert state.transient.get("_transform_error") == "Generation failed"

        # Verify JSON serializable
        json.dumps(conversation_manager.metadata)

    @pytest.mark.asyncio
    async def test_the_partition_and_the_save_agree(
        self, conversation_manager: ConversationManager, caplog: Any
    ) -> None:
        """What the partition keeps is saved whole, and nothing it keeps is dropped.

        An Enum is saved as its value and a NaN as None; the live value stays
        in memory, persistent, for the rest of the turn.
        """
        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["status"] = Status.ACTIVE
        state.data["score"] = math.nan

        with caplog.at_level(logging.WARNING, logger="dataknobs_common.serialization"):
            await reasoning._save_wizard_state(conversation_manager, state)

        saved_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["status"] == "active"
        assert saved_data["score"] is None
        assert math.isnan(state.data["score"])
        assert "score" not in state.transient
        dropped = [r for r in caplog.records if r.name == "dataknobs_common.serialization"]
        assert dropped == []

    @pytest.mark.asyncio
    async def test_a_save_calls_to_dict_once_for_storage(
        self, conversation_manager: ConversationManager
    ) -> None:
        """One call for the response metadata, one for what is stored.

        The partition and the save share one walk, so the stored value is
        not converted twice. The response metadata is built from the state
        before the artifact syncs and the partition, so it walks on its own.
        """
        reasoning = _make_wizard()
        price = Priced()
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["price"] = price

        await reasoning._save_wizard_state(conversation_manager, state)

        assert price.calls == 2
        saved_data = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["price"] == {"amount": "9.05"}

    @pytest.mark.asyncio
    async def test_restored_from_json_storage(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Saved, written as JSON and read back, as SQLite or Postgres storage does.

        The converted forms come back, not the originals: an Enum as its
        value, a NaN as None, and a non-str key as text, since JSON writes
        every key as a string. The in-memory backend keeps the key as it was.
        """
        reasoning = _make_wizard()
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["status"] = Status.ACTIVE
        state.data["readings"] = [1.0, math.nan, 3.0]
        state.data["aisles"] = {1: "produce"}

        await reasoning._save_wizard_state(conversation_manager, state)
        stored = json.loads(json.dumps(conversation_manager.metadata, allow_nan=False))
        conversation_manager.metadata["wizard"] = stored["wizard"]
        restored = reasoning._get_wizard_state(conversation_manager)

        assert restored.data["status"] == "active"
        assert restored.data["readings"] == [1.0, None, 3.0]
        assert restored.data["aisles"] == {"1": "produce"}

    @pytest.mark.asyncio
    async def test_response_metadata_keeps_transient_and_storage_does_not(
        self, conversation_manager: ConversationManager
    ) -> None:
        """The turn's response shows transient keys; the persisted metadata does not.

        The response builds its metadata from the state, transient included,
        for the UI. ``manager.metadata["wizard"]`` is persisted, so both its
        ``data`` and ``fsm_state.data`` hold only what the save keeps.
        """
        reasoning = _make_wizard()

        state = reasoning._get_wizard_state(conversation_manager)
        state.data["topic"] = "English grammar"
        state.data["_transform_error"] = "Generation failed"

        await reasoning._save_wizard_state(conversation_manager, state)

        ui_data = reasoning._build_wizard_metadata(state)["data"]
        assert ui_data["topic"] == "English grammar"
        assert ui_data["_transform_error"] == "Generation failed"

        wizard = conversation_manager.metadata["wizard"]
        for persisted in (wizard["data"], wizard["fsm_state"]["data"]):
            assert persisted["topic"] == "English grammar"
            assert "_transform_error" not in persisted

    def test_template_rendering_sees_transient_data(self) -> None:
        """Templates can reference both persistent and transient keys."""
        config = {
            "name": "test-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "show",
                    "is_start": True,
                    "prompt": "Hello",
                    "response_template": "Topic: {{ topic }}{% if _transform_error %} Error: {{ _transform_error }}{% endif %}",
                    "transitions": [{"target": "done"}],
                },
                {"name": "done", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)

        state = WizardState(
            current_stage="show",
            data={"topic": "Math"},
            transient={"_transform_error": "LLM timeout"},
        )
        stage = fsm.current_metadata

        result = reasoning._render_response_template(stage["response_template"], stage, state)

        assert "Topic: Math" in result
        assert "Error: LLM timeout" in result


_CONFIGURED = {
    "name": "test-wizard",
    "version": "1.0",
    "settings": {
        "ephemeral_keys": ["_dedup_result"],
        "per_turn_keys": ["_turn_note"],
    },
    "stages": [
        {
            "name": "welcome",
            "is_start": True,
            "prompt": "Hello",
            "transitions": [{"target": "done"}],
        },
        {"name": "done", "is_end": True, "prompt": "Done"},
    ],
}


def _copy_of_live_data() -> dict[str, Any]:
    """A working dict as a frame or a snapshot copies it mid-turn."""
    return {
        "topic": "Math",
        "_message": "user input",
        "_intent": "create",
        "_dedup_result": {"is_duplicate": False},
        "_turn_note": "this turn only",
        "record": {"a": 1, "live": NonSerializableObject()},
        "score": math.nan,
    }


def _state_holding_copies(data: dict[str, Any] | None = None) -> WizardState:
    from dataknobs_bots.reasoning.observability import create_transition_record

    state = WizardState(current_stage="welcome", data=dict(data or {"topic": "Math"}))
    state.subflow_stack.append(
        SubflowContext(
            parent_stage="welcome",
            parent_data=_copy_of_live_data(),
            parent_history=["welcome"],
            return_stage="welcome",
            result_mapping={},
            subflow_network="sub",
        )
    )
    state.transitions.append(
        create_transition_record(
            from_stage="welcome",
            to_stage="done",
            trigger="user_input",
            data_snapshot=_copy_of_live_data(),
        )
    )
    return state


_KEPT_COPY = {"topic": "Math", "score": None}


class TestOnePersistenceRule:
    """The main data, each subflow frame and each transition snapshot are
    stored by the same rule, wherever the state is serialized from.
    """

    @pytest.mark.asyncio
    async def test_the_save_applies_the_rule_to_frames_and_snapshots(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Ephemeral keys stay out, and a live object keeps its key out whole."""
        loader = WizardConfigLoader()
        reasoning = WizardReasoning(
            wizard_fsm=loader.load_from_dict(_CONFIGURED), strict_validation=False
        )
        state = _state_holding_copies()

        await reasoning._save_wizard_state(conversation_manager, state)

        stored = json.loads(json.dumps(conversation_manager.metadata, allow_nan=False))
        fsm_state = stored["wizard"]["fsm_state"]
        assert fsm_state["subflow_stack"][0]["parent_data"] == _KEPT_COPY
        assert fsm_state["transitions"][0]["data_snapshot"] == _KEPT_COPY

    @pytest.mark.asyncio
    async def test_a_popped_frame_restored_from_storage_brings_back_no_ephemeral_key(
        self, conversation_manager: ConversationManager
    ) -> None:
        loader = WizardConfigLoader()
        reasoning = WizardReasoning(
            wizard_fsm=loader.load_from_dict(_CONFIGURED), strict_validation=False
        )
        state = _state_holding_copies()

        await reasoning._save_wizard_state(conversation_manager, state)
        stored = json.loads(json.dumps(conversation_manager.metadata, allow_nan=False))
        conversation_manager.metadata["wizard"] = stored["wizard"]
        restored = reasoning._get_wizard_state(conversation_manager)
        restored.data.pop("topic")
        reasoning._subflows.handle_pop(restored)

        assert restored.data == _KEPT_COPY

    def test_to_dict_is_json_writable(self) -> None:
        """A live object, a NaN, and an ephemeral key, in the data and in copies."""
        state = _state_holding_copies(
            {"topic": "Math", "_message": "hi", "score": math.nan, "live": NonSerializableObject()}
        )

        payload = json.loads(json.dumps(state.to_dict(), allow_nan=False))

        assert payload["data"] == {"topic": "Math", "score": None}
        # Without the wizard's settings, only the framework keys are known.
        expected_copy = {**_KEPT_COPY, "_dedup_result": {"is_duplicate": False}}
        expected_copy["_turn_note"] = "this turn only"
        assert payload["subflow_stack"][0]["parent_data"] == expected_copy
        assert payload["transitions"][0]["data_snapshot"] == expected_copy

    def test_to_dict_survives_a_snapshot_value_that_cannot_be_copied(self) -> None:
        """A transition snapshot is not deep-copied on its way out."""
        import threading

        state = _state_holding_copies()
        assert state.transitions[0].data_snapshot is not None
        state.transitions[0].data_snapshot["lock"] = threading.Lock()

        payload = json.loads(json.dumps(state.to_dict(), allow_nan=False))

        assert "lock" not in payload["transitions"][0]["data_snapshot"]

    def test_serialize_state_applies_the_wizards_ephemeral_keys(self) -> None:
        loader = WizardConfigLoader()
        reasoning = WizardReasoning(
            wizard_fsm=loader.load_from_dict(_CONFIGURED), strict_validation=False
        )
        state = _state_holding_copies(_copy_of_live_data())

        payload = json.loads(json.dumps(reasoning.serialize_state(state), allow_nan=False))

        assert payload["data"] == _KEPT_COPY
        assert payload["subflow_stack"][0]["parent_data"] == _KEPT_COPY
        assert payload["transitions"][0]["data_snapshot"] == _KEPT_COPY
        assert "_message" in state.data, "serializing does not change the state"

    @pytest.mark.asyncio
    async def test_the_save_stores_what_serialize_state_returns(
        self, conversation_manager: ConversationManager
    ) -> None:
        loader = WizardConfigLoader()
        reasoning = WizardReasoning(
            wizard_fsm=loader.load_from_dict(_CONFIGURED), strict_validation=False
        )
        state = _state_holding_copies(_copy_of_live_data())

        await reasoning._save_wizard_state(conversation_manager, state)

        stored = conversation_manager.metadata["wizard"]["fsm_state"]
        assert stored == reasoning.serialize_state(state)


class TestWhatTheSaveLeavesInMemory:
    """prepare_save moves what is not stored into transient, in place."""

    def test_moves_ephemeral_and_unstorable_keys_into_transient(self) -> None:
        state = WizardState(
            current_stage="welcome",
            data={"topic": "Math", "_message": "hi", "conn": NonSerializableObject()},
            transient={"_stale": 1},
        )

        state.prepare_save(DEFAULT_EPHEMERAL_KEYS)

        assert state.data == {"topic": "Math"}
        assert set(state.transient) == {"_message", "conn"}

    def test_keeps_the_identity_of_data(self) -> None:
        """A holder of ``state.data`` sees what the save left, not a stranded dict."""
        state = WizardState(current_stage="welcome", data={"topic": "Math", "_message": "hi"})
        held = state.data

        state.prepare_save(DEFAULT_EPHEMERAL_KEYS)

        assert state.data is held
        assert held == {"topic": "Math"}

    def test_the_stored_form_shares_no_list_with_the_state(self) -> None:
        state = _state_holding_copies()

        payload = state.to_dict()
        payload["history"].append("elsewhere")
        payload["subflow_stack"][0]["parent_history"].append("elsewhere")

        assert state.history == ["welcome"]
        assert state.subflow_stack[0].parent_history == ["welcome"]


class TestReadingStoredState:
    """What is restored from a stored dict is not the stored dict."""

    def test_a_restored_frame_does_not_hold_the_stored_parent_data(self) -> None:
        stored = SubflowContext(
            parent_stage="p",
            parent_data={"rec": {"a": 1}},
            parent_history=["p"],
            return_stage="p",
            result_mapping={},
            subflow_network="sub",
        ).to_dict()

        frame = SubflowContext.from_dict(stored)
        frame.parent_data["rec"]["a"] = 2
        frame.parent_history.append("q")

        assert stored["parent_data"] == {"rec": {"a": 1}}
        assert stored["parent_history"] == ["p"]

    def test_a_restored_state_does_not_hold_the_stored_history(self) -> None:
        stored = WizardState(current_stage="a", history=["a"]).to_dict()

        state = WizardState.from_dict(stored)
        state.history.append("b")

        assert stored["history"] == ["a"]


class TestWhatIsLoggedForCopies:
    """A key kept out of storage is reported once per save, wherever it is first found."""

    def test_a_key_reported_in_the_data_is_quiet_in_its_copies(self, caplog: Any) -> None:
        state = _state_holding_copies({"record": {"a": 1, "live": NonSerializableObject()}})

        with caplog.at_level(logging.DEBUG, logger="dataknobs_bots.reasoning.wizard_types"):
            state.prepare_save(DEFAULT_EPHEMERAL_KEYS)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert [r.getMessage().split(" in ")[1].split(" (")[0] for r in warnings] == ["data"]

    def test_a_key_found_only_in_a_copy_is_warned_once(self, caplog: Any) -> None:
        """A subflow pushed in the turn the value appeared: the data never held it at a save."""
        state = _state_holding_copies()

        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.reasoning.wizard_types"):
            state.prepare_save(DEFAULT_EPHEMERAL_KEYS)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Non-serializable key 'record' in " in warnings[0]


class TestTransitionFields:
    def test_a_field_other_than_the_snapshot_is_written_as_json(self) -> None:
        from dataknobs_bots.reasoning.observability import create_transition_record

        state = WizardState(current_stage="welcome")
        state.transitions.append(
            create_transition_record(
                from_stage="welcome",
                to_stage="done",
                trigger=Status.ACTIVE,  # type: ignore[arg-type]
            )
        )

        payload = json.loads(json.dumps(state.to_dict(), allow_nan=False))

        assert payload["transitions"][0]["trigger"] == "active"


class TestPersistedConversationMetadata:
    @pytest.mark.asyncio
    async def test_the_flat_data_is_the_stored_data(
        self, conversation_manager: ConversationManager
    ) -> None:
        """``metadata["wizard"]["data"]`` is persisted too, so it holds no ephemeral key."""
        loader = WizardConfigLoader()
        reasoning = WizardReasoning(
            wizard_fsm=loader.load_from_dict(_CONFIGURED), strict_validation=False
        )
        state = reasoning._get_wizard_state(conversation_manager)
        state.data.update(_copy_of_live_data())

        await reasoning._save_wizard_state(conversation_manager, state)

        wizard = conversation_manager.metadata["wizard"]
        assert wizard["data"] == wizard["fsm_state"]["data"] == _KEPT_COPY
        assert wizard["data"] is not wizard["fsm_state"]["data"]

    @pytest.mark.asyncio
    async def test_a_datetime_is_stored_as_text(
        self, conversation_manager: ConversationManager
    ) -> None:
        import datetime
        import uuid

        reasoning = _make_wizard()
        state = reasoning._get_wizard_state(conversation_manager)
        state.data["when"] = datetime.datetime(2026, 1, 2, 3, 4)
        state.data["id"] = uuid.UUID(int=1)
        state.data["tags"] = {"b", "a"}

        await reasoning._save_wizard_state(conversation_manager, state)

        stored = conversation_manager.metadata["wizard"]["fsm_state"]["data"]
        assert stored["when"] == "2026-01-02T03:04:00"
        assert stored["id"] == "00000000-0000-0000-0000-000000000001"
        assert stored["tags"] == ["a", "b"]
        assert isinstance(state.data["when"], datetime.datetime), "kept live in memory"

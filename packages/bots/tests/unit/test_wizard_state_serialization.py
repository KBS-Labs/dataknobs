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
from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import (
    DEFAULT_EPHEMERAL_KEYS,
    SubflowContext,
    WizardReasoning,
    _is_json_safe,
)
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


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


def _make_wizard_and_manager(
    config: dict[str, Any] | None = None,
) -> tuple[WizardReasoning, Any]:
    """Create a WizardReasoning + WizardTestManager pair."""
    from tests.unit.conftest import WizardTestManager

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
    reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)
    manager = WizardTestManager()
    return reasoning, manager


class TestWizardStateSerialization:
    """Tests for wizard state serialization safety."""

    def test_save_state_strips_non_serializable_objects(self) -> None:
        """Verifies fix for the crash: non-serializable objects in state.data
        are stripped during save, so json.dumps(metadata) succeeds.

        Before this fix, ArtifactCorpus in state.data caused:
            TypeError: Object of type ArtifactCorpus is not JSON serializable
        """
        reasoning, manager = _make_wizard_and_manager()

        # Simulate what happens after initialize_bank stores _corpus in data
        state = reasoning._get_wizard_state(manager)
        state.data["topic"] = "English grammar"
        state.data["_corpus_id"] = "corpus-abc"
        state.data["_corpus"] = NonSerializableObject("ArtifactCorpus")

        reasoning._save_wizard_state(manager, state)

        # Must NOT raise — non-serializable object stripped during save
        json_str = json.dumps(manager.metadata)
        parsed = json.loads(json_str)
        data = parsed["wizard"]["fsm_state"]["data"]
        assert data["topic"] == "English grammar"
        assert data["_corpus_id"] == "corpus-abc"
        assert "_corpus" not in data

    def test_save_state_preserves_all_serializable_underscore_keys(self) -> None:
        """Every quiz-style underscore-prefixed serializable key survives save."""
        reasoning, manager = _make_wizard_and_manager()

        state = reasoning._get_wizard_state(manager)
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

        reasoning._save_wizard_state(manager, state)

        saved_data = manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["topic"] == "Biology"
        assert saved_data["_corpus_id"] == "corpus-xyz"
        assert saved_data["_bank_questions"] == [{"id": "q1"}, {"id": "q2"}]
        assert saved_data["_bank_question_ids"] == ["art-1", "art-2"]
        assert saved_data["_bank_difficulty_counts"] == {"easy": 1, "medium": 1, "hard": 0}
        assert saved_data["_bank_batch_count"] == 1
        assert saved_data["_target_count"] == 50
        assert saved_data["_current_question"] == {"question_text": "What is DNA?"}
        assert saved_data["_current_question_number"] == 3

    def test_save_state_converts_dataclass_in_data(self) -> None:
        """DedupResult dataclass in data becomes a dict after save."""
        reasoning, manager = _make_wizard_and_manager()

        state = reasoning._get_wizard_state(manager)
        state.data["_dedup_result"] = FakeDedupResult(
            is_exact_duplicate=False,
            content_hash="abc123",
        )

        reasoning._save_wizard_state(manager, state)

        saved_data = manager.metadata["wizard"]["fsm_state"]["data"]
        dedup = saved_data["_dedup_result"]
        assert isinstance(dedup, dict)
        assert dedup["is_exact_duplicate"] is False
        assert dedup["content_hash"] == "abc123"

    def test_save_state_none_values_preserved(self) -> None:
        """data["_dedup_result"] = None remains None after save."""
        reasoning, manager = _make_wizard_and_manager()

        state = reasoning._get_wizard_state(manager)
        state.data["_dedup_result"] = None
        state.data["_corpus_id"] = None
        state.data["topic"] = "test"

        reasoning._save_wizard_state(manager, state)

        saved_data = manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["_dedup_result"] is None
        assert saved_data["_corpus_id"] is None
        assert saved_data["topic"] == "test"

    def test_saved_state_is_json_serializable(self) -> None:
        """json.dumps(metadata) succeeds with mixed data including
        non-serializable objects and dataclasses."""
        reasoning, manager = _make_wizard_and_manager()

        state = reasoning._get_wizard_state(manager)
        state.data["topic"] = "English grammar"
        state.data["_corpus_id"] = "corpus-abc"
        state.data["_corpus"] = NonSerializableObject("live corpus")
        state.data["_dedup_result"] = FakeDedupResult(
            is_exact_duplicate=True,
            exact_match_id="q-old",
        )
        state.data["_bank_questions"] = [{"id": "q1"}]

        reasoning._save_wizard_state(manager, state)

        # Must not raise
        json_str = json.dumps(manager.metadata)
        parsed = json.loads(json_str)

        data = parsed["wizard"]["fsm_state"]["data"]
        assert data["topic"] == "English grammar"
        assert data["_corpus_id"] == "corpus-abc"
        assert "_corpus" not in data  # Non-serializable stripped
        assert data["_dedup_result"]["is_exact_duplicate"] is True
        assert data["_bank_questions"] == [{"id": "q1"}]

    def test_roundtrip_save_and_restore_preserves_data(self) -> None:
        """Save → restore via _get_wizard_state → all serializable data intact."""
        reasoning, manager = _make_wizard_and_manager()

        # Set up state with mixed data
        state = reasoning._get_wizard_state(manager)
        state.data["topic"] = "Chemistry"
        state.data["_corpus_id"] = "corpus-chem"
        state.data["_corpus"] = NonSerializableObject("live")
        state.data["_bank_questions"] = [{"id": "q1", "topic": "atoms"}]
        state.data["_bank_question_ids"] = ["art-1"]
        state.data["_bank_difficulty_counts"] = {"easy": 0, "medium": 1, "hard": 0}

        # Save
        reasoning._save_wizard_state(manager, state)

        # Restore
        restored = reasoning._get_wizard_state(manager)

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
        values but is independent of the original."""
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

    def test_get_wizard_state_data_is_decoupled_from_metadata(self) -> None:
        """Modifying wizard_state.data after _get_wizard_state must NOT
        modify manager.metadata.

        This is the root cause of the serialization crash: _get_wizard_state
        uses a reference to the same dict stored in metadata. When transforms
        add _corpus to wizard_state.data, it appears in metadata too.
        """
        reasoning, manager = _make_wizard_and_manager()

        # Initial save with clean data
        state = reasoning._get_wizard_state(manager)
        state.data["topic"] = "English grammar"
        reasoning._save_wizard_state(manager, state)

        # Load state back — this is where the shared reference is created
        loaded_state = reasoning._get_wizard_state(manager)

        # Simulate what transforms do: add non-serializable object
        loaded_state.data["_corpus"] = NonSerializableObject("live corpus")
        loaded_state.data["_new_key"] = "injected"

        # The metadata must NOT be contaminated
        fsm_data = manager.metadata["wizard"]["fsm_state"]["data"]
        assert "_corpus" not in fsm_data, (
            "Non-serializable object leaked from wizard_state.data "
            "into manager.metadata via shared reference"
        )
        assert "_new_key" not in fsm_data, (
            "New key leaked from wizard_state.data into manager.metadata"
        )

    def test_mutating_data_after_load_does_not_crash_json_dumps(self) -> None:
        """Load state → add non-serializable to data → json.dumps(metadata) → no crash.

        This reproduces the exact crash from the api-log: after transforms
        add _corpus to wizard_state.data, the next json.dumps(metadata) fails
        because the shared reference puts the ArtifactCorpus in metadata.
        """
        reasoning, manager = _make_wizard_and_manager()

        # Save clean state
        state = reasoning._get_wizard_state(manager)
        state.data["topic"] = "English grammar"
        reasoning._save_wizard_state(manager, state)

        # Reload state (creates the shared reference)
        reloaded = reasoning._get_wizard_state(manager)

        # Simulate transform adding non-serializable object
        reloaded.data["_corpus"] = NonSerializableObject("ArtifactCorpus")
        reloaded.data["_corpus_id"] = "corpus-abc"

        # json.dumps on metadata must succeed (the key assertion)
        json_str = json.dumps(manager.metadata)
        parsed = json.loads(json_str)
        assert "wizard" in parsed

    def test_transition_data_snapshot_is_serializable(self) -> None:
        """Transition records with non-serializable data in snapshot must
        not crash json.dumps after _save_wizard_state."""
        from dataknobs_bots.reasoning.observability import create_transition_record

        reasoning, manager = _make_wizard_and_manager()

        state = reasoning._get_wizard_state(manager)
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
        reasoning._save_wizard_state(manager, state)

        # json.dumps on full metadata must succeed
        json_str = json.dumps(manager.metadata)
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


class TestPartitionData:
    """Tests for WizardReasoning._partition_data()."""

    def test_classifies_ephemeral_keys(self) -> None:
        """Known ephemeral keys end up in transient dict."""
        reasoning, _ = _make_wizard_and_manager()

        data = {
            "topic": "Math",
            "_corpus": NonSerializableObject("live"),
            "_transform_error": "Something failed",
            "_message": "user input",
        }

        persistent, transient = reasoning._partition_data(data)

        assert "topic" in persistent
        assert "_corpus" in transient
        assert "_transform_error" in transient
        assert "_message" in transient
        assert "_corpus" not in persistent
        assert "_transform_error" not in persistent

    def test_classifies_persistent_keys(self) -> None:
        """User data and persistent _-prefixed keys stay in persistent dict."""
        reasoning, _ = _make_wizard_and_manager()

        data = {
            "topic": "English grammar",
            "difficulty": "medium",
            "_corpus_id": "corpus-abc",
            "_bank_questions": [{"id": "q1"}],
            "_target_count": 50,
        }

        persistent, transient = reasoning._partition_data(data)

        assert persistent == data  # All should be persistent
        assert transient == {}

    def test_catches_unknown_non_serializable(self) -> None:
        """Non-serializable objects not in EPHEMERAL_KEYS go to transient."""
        reasoning, _ = _make_wizard_and_manager()

        data = {
            "topic": "Math",
            "_unknown_live_object": NonSerializableObject("surprise"),
            "_corpus_id": "corpus-abc",
        }

        persistent, transient = reasoning._partition_data(data)

        assert "topic" in persistent
        assert "_corpus_id" in persistent
        assert "_unknown_live_object" in transient
        assert "_unknown_live_object" not in persistent

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

        persistent, transient = reasoning._partition_data(data)

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

    def test_save_wizard_state_excludes_transient(self) -> None:
        """After partition + save, persisted metadata has no ephemeral keys."""
        reasoning, manager = _make_wizard_and_manager()

        state = reasoning._get_wizard_state(manager)
        state.data["topic"] = "English grammar"
        state.data["_corpus_id"] = "corpus-abc"
        state.data["_corpus"] = NonSerializableObject("ArtifactCorpus")
        state.data["_transform_error"] = "Generation failed"
        state.data["_message"] = "user input"

        reasoning._save_wizard_state(manager, state)

        # Verify persisted data excludes ephemeral keys
        saved_data = manager.metadata["wizard"]["fsm_state"]["data"]
        assert saved_data["topic"] == "English grammar"
        assert saved_data["_corpus_id"] == "corpus-abc"
        assert "_corpus" not in saved_data
        assert "_transform_error" not in saved_data
        assert "_message" not in saved_data

        # Verify transient is populated on the state object
        assert state.transient.get("_corpus") is not None
        assert state.transient.get("_transform_error") == "Generation failed"

        # Verify JSON serializable
        json.dumps(manager.metadata)

    def test_metadata_data_includes_transient_for_ui(self) -> None:
        """The 'data' field in response metadata includes both persistent + transient."""
        reasoning, manager = _make_wizard_and_manager()

        state = reasoning._get_wizard_state(manager)
        state.data["topic"] = "English grammar"
        state.data["_transform_error"] = "Generation failed"

        reasoning._save_wizard_state(manager, state)

        # The top-level wizard metadata 'data' should include transient
        # (for UI display), sanitized via sanitize_for_json
        ui_data = manager.metadata["wizard"]["data"]
        assert ui_data["topic"] == "English grammar"
        assert ui_data["_transform_error"] == "Generation failed"

        # But fsm_state.data should NOT have transient
        persisted_data = manager.metadata["wizard"]["fsm_state"]["data"]
        assert persisted_data["topic"] == "English grammar"
        assert "_transform_error" not in persisted_data

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

        from dataknobs_bots.reasoning.wizard import WizardState

        state = WizardState(
            current_stage="show",
            data={"topic": "Math"},
            transient={"_transform_error": "LLM timeout"},
        )
        stage = fsm.current_metadata

        result = reasoning._render_response_template(
            stage["response_template"], stage, state
        )

        assert "Topic: Math" in result
        assert "Error: LLM timeout" in result

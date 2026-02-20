"""Tests for wizard state serialization safety.

Verifies that _save_wizard_state produces JSON-serializable metadata,
even when state.data contains non-serializable objects like ArtifactCorpus
or DedupResult dataclass instances.

These tests reproduce the crash:
    TypeError: Object of type ArtifactCorpus is not JSON serializable
that occurs when _save_wizard_state stores state.data raw into metadata
which later flows to json.dumps() via ConversationState.to_dict().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import SubflowContext, WizardReasoning
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

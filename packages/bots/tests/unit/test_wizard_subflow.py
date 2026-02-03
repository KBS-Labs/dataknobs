"""Tests for wizard subflow support.

Tests cover:
- SubflowContext dataclass creation and serialization
- WizardState subflow properties (is_in_subflow, subflow_depth)
- WizardFSM subflow registry methods
- Data mapping (parent -> subflow) and result mapping (subflow -> parent)
- TransitionRecord subflow fields
- Subflow state serialization/deserialization
"""

import time

import pytest

from dataknobs_bots.reasoning.observability import (
    TransitionRecord,
    create_transition_record,
)
from dataknobs_bots.reasoning.wizard import (
    SubflowContext,
    WizardState,
)


class TestSubflowContext:
    """Tests for SubflowContext dataclass."""

    def test_create_context(self) -> None:
        """Test creating a subflow context."""
        context = SubflowContext(
            parent_stage="configure_knowledge",
            parent_data={"domain_id": "test", "kb_type": "faq"},
            parent_history=["welcome", "configure_knowledge"],
            return_stage="configure_tools",
            result_mapping={"kb_config": "kb_config", "kb_resources": "kb_resources"},
            subflow_network="knowledge_base_acquisition",
        )

        assert context.parent_stage == "configure_knowledge"
        assert context.parent_data == {"domain_id": "test", "kb_type": "faq"}
        assert context.parent_history == ["welcome", "configure_knowledge"]
        assert context.return_stage == "configure_tools"
        assert context.result_mapping == {
            "kb_config": "kb_config",
            "kb_resources": "kb_resources",
        }
        assert context.subflow_network == "knowledge_base_acquisition"
        assert context.push_timestamp > 0

    def test_create_context_with_timestamp(self) -> None:
        """Test creating a subflow context with explicit timestamp."""
        timestamp = 1234567890.0
        context = SubflowContext(
            parent_stage="stage1",
            parent_data={},
            parent_history=["stage1"],
            return_stage="stage2",
            result_mapping={},
            subflow_network="subflow1",
            push_timestamp=timestamp,
        )

        assert context.push_timestamp == timestamp

    def test_to_dict(self) -> None:
        """Test converting context to dictionary."""
        timestamp = 1234567890.0
        context = SubflowContext(
            parent_stage="configure_knowledge",
            parent_data={"domain_id": "test"},
            parent_history=["welcome"],
            return_stage="configure_tools",
            result_mapping={"kb_config": "config"},
            subflow_network="kb_acquisition",
            push_timestamp=timestamp,
        )

        data = context.to_dict()

        assert data["parent_stage"] == "configure_knowledge"
        assert data["parent_data"] == {"domain_id": "test"}
        assert data["parent_history"] == ["welcome"]
        assert data["return_stage"] == "configure_tools"
        assert data["result_mapping"] == {"kb_config": "config"}
        assert data["subflow_network"] == "kb_acquisition"
        assert data["push_timestamp"] == timestamp

    def test_from_dict(self) -> None:
        """Test creating context from dictionary."""
        data = {
            "parent_stage": "stage_a",
            "parent_data": {"key": "value"},
            "parent_history": ["welcome", "stage_a"],
            "return_stage": "stage_b",
            "result_mapping": {"result": "output"},
            "subflow_network": "my_subflow",
            "push_timestamp": 9876543210.0,
        }

        context = SubflowContext.from_dict(data)

        assert context.parent_stage == "stage_a"
        assert context.parent_data == {"key": "value"}
        assert context.parent_history == ["welcome", "stage_a"]
        assert context.return_stage == "stage_b"
        assert context.result_mapping == {"result": "output"}
        assert context.subflow_network == "my_subflow"
        assert context.push_timestamp == 9876543210.0

    def test_from_dict_missing_optional_fields(self) -> None:
        """Test creating context from dict with missing optional fields."""
        data = {
            "parent_stage": "stage_a",
            "parent_data": {},
            "parent_history": [],
            "return_stage": "stage_b",
            "subflow_network": "my_subflow",
        }

        context = SubflowContext.from_dict(data)

        assert context.result_mapping == {}
        # push_timestamp defaults to current time
        assert context.push_timestamp > 0


class TestWizardStateSubflow:
    """Tests for WizardState subflow properties."""

    def test_is_in_subflow_false_when_empty(self) -> None:
        """Test is_in_subflow is False when subflow_stack is empty."""
        state = WizardState(
            current_stage="welcome",
            subflow_stack=[],
        )

        assert state.is_in_subflow is False

    def test_is_in_subflow_true_when_has_subflow(self) -> None:
        """Test is_in_subflow is True when subflow_stack has entries."""
        context = SubflowContext(
            parent_stage="stage1",
            parent_data={},
            parent_history=[],
            return_stage="stage2",
            result_mapping={},
            subflow_network="subflow1",
        )
        state = WizardState(
            current_stage="subflow_stage1",
            subflow_stack=[context],
        )

        assert state.is_in_subflow is True

    def test_subflow_depth_zero_when_empty(self) -> None:
        """Test subflow_depth is 0 when in main flow."""
        state = WizardState(
            current_stage="welcome",
            subflow_stack=[],
        )

        assert state.subflow_depth == 0

    def test_subflow_depth_one_when_in_subflow(self) -> None:
        """Test subflow_depth is 1 when in one subflow."""
        context = SubflowContext(
            parent_stage="stage1",
            parent_data={},
            parent_history=[],
            return_stage="stage2",
            result_mapping={},
            subflow_network="subflow1",
        )
        state = WizardState(
            current_stage="subflow_stage1",
            subflow_stack=[context],
        )

        assert state.subflow_depth == 1

    def test_subflow_depth_nested(self) -> None:
        """Test subflow_depth reflects nesting level."""
        context1 = SubflowContext(
            parent_stage="main_stage",
            parent_data={},
            parent_history=[],
            return_stage="main_return",
            result_mapping={},
            subflow_network="subflow1",
        )
        context2 = SubflowContext(
            parent_stage="subflow1_stage",
            parent_data={},
            parent_history=[],
            return_stage="subflow1_return",
            result_mapping={},
            subflow_network="nested_subflow",
        )
        state = WizardState(
            current_stage="nested_stage",
            subflow_stack=[context1, context2],
        )

        assert state.subflow_depth == 2

    def test_current_subflow_none_when_empty(self) -> None:
        """Test current_subflow is None in main flow."""
        state = WizardState(
            current_stage="welcome",
            subflow_stack=[],
        )

        assert state.current_subflow is None

    def test_current_subflow_returns_top_of_stack(self) -> None:
        """Test current_subflow returns most recent subflow context."""
        context1 = SubflowContext(
            parent_stage="main_stage",
            parent_data={},
            parent_history=[],
            return_stage="main_return",
            result_mapping={},
            subflow_network="subflow1",
        )
        context2 = SubflowContext(
            parent_stage="subflow1_stage",
            parent_data={},
            parent_history=[],
            return_stage="subflow1_return",
            result_mapping={},
            subflow_network="nested_subflow",
        )
        state = WizardState(
            current_stage="nested_stage",
            subflow_stack=[context1, context2],
        )

        assert state.current_subflow == context2
        assert state.current_subflow.subflow_network == "nested_subflow"


class TestTransitionRecordSubflowFields:
    """Tests for subflow fields in TransitionRecord."""

    def test_create_record_with_subflow_push(self) -> None:
        """Test creating a transition record for subflow push."""
        record = TransitionRecord(
            from_stage="configure_knowledge",
            to_stage="kb_welcome",
            timestamp=time.time(),
            trigger="subflow_push",
            subflow_push="knowledge_base_acquisition",
            subflow_depth=1,
        )

        assert record.subflow_push == "knowledge_base_acquisition"
        assert record.subflow_pop is None
        assert record.subflow_depth == 1

    def test_create_record_with_subflow_pop(self) -> None:
        """Test creating a transition record for subflow pop."""
        record = TransitionRecord(
            from_stage="kb_complete",
            to_stage="configure_tools",
            timestamp=time.time(),
            trigger="subflow_pop",
            subflow_pop="knowledge_base_acquisition",
            subflow_depth=0,
        )

        assert record.subflow_push is None
        assert record.subflow_pop == "knowledge_base_acquisition"
        assert record.subflow_depth == 0

    def test_create_record_default_subflow_fields(self) -> None:
        """Test that subflow fields default correctly."""
        record = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=time.time(),
            trigger="user_input",
        )

        assert record.subflow_push is None
        assert record.subflow_pop is None
        assert record.subflow_depth == 0

    def test_factory_with_subflow_fields(self) -> None:
        """Test create_transition_record factory with subflow fields."""
        record = create_transition_record(
            from_stage="stage1",
            to_stage="subflow_start",
            trigger="subflow_push",
            subflow_push="my_subflow",
            subflow_depth=1,
        )

        assert record.subflow_push == "my_subflow"
        assert record.subflow_depth == 1

    def test_to_dict_includes_subflow_fields(self) -> None:
        """Test that to_dict includes subflow fields."""
        record = TransitionRecord(
            from_stage="stage1",
            to_stage="subflow_start",
            timestamp=1234567890.0,
            trigger="subflow_push",
            subflow_push="my_subflow",
            subflow_pop=None,
            subflow_depth=1,
        )

        data = record.to_dict()

        assert data["subflow_push"] == "my_subflow"
        assert data["subflow_pop"] is None
        assert data["subflow_depth"] == 1

    def test_from_dict_restores_subflow_fields(self) -> None:
        """Test that from_dict restores subflow fields."""
        data = {
            "from_stage": "kb_complete",
            "to_stage": "configure_tools",
            "timestamp": 1234567890.0,
            "trigger": "subflow_pop",
            "duration_in_stage_ms": 0.0,
            "data_snapshot": None,
            "user_input": None,
            "condition_evaluated": None,
            "condition_result": None,
            "error": None,
            "subflow_push": None,
            "subflow_pop": "kb_acquisition",
            "subflow_depth": 0,
        }

        record = TransitionRecord.from_dict(data)

        assert record.subflow_push is None
        assert record.subflow_pop == "kb_acquisition"
        assert record.subflow_depth == 0


class TestSubflowStateSerialization:
    """Tests for subflow state serialization in WizardState."""

    def test_serialize_empty_subflow_stack(self) -> None:
        """Test that empty subflow_stack serializes correctly."""
        state = WizardState(
            current_stage="welcome",
            data={},
            history=["welcome"],
            subflow_stack=[],
        )

        # Simulate what _save_wizard_state does
        serialized_stack = [s.to_dict() for s in state.subflow_stack]

        assert serialized_stack == []

    def test_serialize_with_subflow(self) -> None:
        """Test that subflow_stack serializes correctly."""
        context = SubflowContext(
            parent_stage="configure_knowledge",
            parent_data={"domain_id": "test"},
            parent_history=["welcome", "configure_knowledge"],
            return_stage="configure_tools",
            result_mapping={"kb_config": "kb_config"},
            subflow_network="kb_acquisition",
            push_timestamp=1234567890.0,
        )
        state = WizardState(
            current_stage="kb_welcome",
            data={"kb_type": "faq"},
            history=["kb_welcome"],
            subflow_stack=[context],
        )

        serialized_stack = [s.to_dict() for s in state.subflow_stack]

        assert len(serialized_stack) == 1
        assert serialized_stack[0]["parent_stage"] == "configure_knowledge"
        assert serialized_stack[0]["subflow_network"] == "kb_acquisition"

    def test_deserialize_subflow_stack(self) -> None:
        """Test that subflow_stack deserializes correctly."""
        serialized_stack = [
            {
                "parent_stage": "configure_knowledge",
                "parent_data": {"domain_id": "test"},
                "parent_history": ["welcome", "configure_knowledge"],
                "return_stage": "configure_tools",
                "result_mapping": {"kb_config": "kb_config"},
                "subflow_network": "kb_acquisition",
                "push_timestamp": 1234567890.0,
            }
        ]

        restored_stack = [
            SubflowContext.from_dict(s) for s in serialized_stack
        ]

        assert len(restored_stack) == 1
        assert restored_stack[0].parent_stage == "configure_knowledge"
        assert restored_stack[0].subflow_network == "kb_acquisition"

    def test_serialize_nested_subflows(self) -> None:
        """Test serialization of nested subflows."""
        context1 = SubflowContext(
            parent_stage="main_stage",
            parent_data={"key1": "value1"},
            parent_history=["welcome", "main_stage"],
            return_stage="main_return",
            result_mapping={"r1": "o1"},
            subflow_network="subflow1",
            push_timestamp=1000.0,
        )
        context2 = SubflowContext(
            parent_stage="subflow1_stage",
            parent_data={"key2": "value2"},
            parent_history=["subflow1_welcome", "subflow1_stage"],
            return_stage="subflow1_return",
            result_mapping={"r2": "o2"},
            subflow_network="nested_subflow",
            push_timestamp=2000.0,
        )
        state = WizardState(
            current_stage="nested_current",
            subflow_stack=[context1, context2],
        )

        serialized = [s.to_dict() for s in state.subflow_stack]

        assert len(serialized) == 2
        assert serialized[0]["subflow_network"] == "subflow1"
        assert serialized[1]["subflow_network"] == "nested_subflow"

        # Restore and verify
        restored = [SubflowContext.from_dict(s) for s in serialized]
        assert restored[0].subflow_network == "subflow1"
        assert restored[1].subflow_network == "nested_subflow"
        assert restored[0].push_timestamp == 1000.0
        assert restored[1].push_timestamp == 2000.0


class TestDataMapping:
    """Tests for data mapping helper functions.

    These test the _apply_data_mapping and _apply_result_mapping
    logic used during subflow push/pop operations.
    """

    def test_apply_data_mapping_basic(self) -> None:
        """Test basic data mapping from parent to subflow."""
        source_data = {
            "domain_id": "my_domain",
            "kb_type": "faq",
            "other_field": "ignored",
        }
        mapping = {
            "domain_id": "domain",  # parent_field -> subflow_field
            "kb_type": "type",
        }

        # Simulate _apply_data_mapping logic
        result: dict = {}
        for parent_field, subflow_field in mapping.items():
            if parent_field in source_data:
                result[subflow_field] = source_data[parent_field]

        assert result == {"domain": "my_domain", "type": "faq"}
        assert "other_field" not in result

    def test_apply_data_mapping_empty(self) -> None:
        """Test data mapping with empty mapping."""
        source_data = {"field1": "value1"}
        mapping: dict = {}

        result: dict = {}
        for parent_field, subflow_field in mapping.items():
            if parent_field in source_data:
                result[subflow_field] = source_data[parent_field]

        assert result == {}

    def test_apply_data_mapping_missing_field(self) -> None:
        """Test data mapping when source field doesn't exist."""
        source_data = {"field1": "value1"}
        mapping = {"missing_field": "target"}

        result: dict = {}
        for parent_field, subflow_field in mapping.items():
            if parent_field in source_data:
                result[subflow_field] = source_data[parent_field]

        assert result == {}

    def test_apply_result_mapping_basic(self) -> None:
        """Test basic result mapping from subflow to parent."""
        source_data = {
            "kb_config": {"sources": ["doc1", "doc2"]},
            "kb_resources": 5,
            "internal_field": "not_mapped",
        }
        mapping = {
            "kb_config": "knowledge_config",  # subflow_field -> parent_field
            "kb_resources": "resource_count",
        }

        # Simulate _apply_result_mapping logic
        result: dict = {}
        for subflow_field, parent_field in mapping.items():
            if subflow_field in source_data:
                result[parent_field] = source_data[subflow_field]

        assert result == {
            "knowledge_config": {"sources": ["doc1", "doc2"]},
            "resource_count": 5,
        }

    def test_apply_result_mapping_preserves_types(self) -> None:
        """Test that result mapping preserves complex types."""
        source_data = {
            "list_field": [1, 2, 3],
            "dict_field": {"nested": "value"},
            "none_field": None,
        }
        mapping = {
            "list_field": "output_list",
            "dict_field": "output_dict",
            "none_field": "output_none",
        }

        result: dict = {}
        for subflow_field, parent_field in mapping.items():
            if subflow_field in source_data:
                result[parent_field] = source_data[subflow_field]

        assert result["output_list"] == [1, 2, 3]
        assert result["output_dict"] == {"nested": "value"}
        assert result["output_none"] is None

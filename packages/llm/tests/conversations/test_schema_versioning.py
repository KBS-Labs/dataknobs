"""Tests for conversation storage schema versioning."""

import pytest
from datetime import datetime
from dataknobs_llm.conversations import (
    ConversationNode,
    ConversationState,
    SCHEMA_VERSION,
    SchemaVersionError
)
from dataknobs_llm.llm import LLMMessage
from dataknobs_structures.tree import Tree


class TestSchemaVersioning:
    """Test schema version handling and migration."""

    def test_new_conversation_has_current_version(self):
        """Test that new conversations use current schema version."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Test"),
            node_id=""
        )
        tree = Tree(root_node)

        state = ConversationState(
            conversation_id="test-123",
            message_tree=tree
        )

        assert state.schema_version == SCHEMA_VERSION

    def test_to_dict_includes_schema_version(self):
        """Test that serialization includes schema version."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Test"),
            node_id=""
        )
        tree = Tree(root_node)

        state = ConversationState(
            conversation_id="test-123",
            message_tree=tree
        )

        data = state.to_dict()

        assert "schema_version" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_from_dict_with_current_version(self):
        """Test loading data with current schema version."""
        # Create and serialize a conversation
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Test"),
            node_id=""
        )
        tree = Tree(root_node)

        state1 = ConversationState(
            conversation_id="test-123",
            message_tree=tree
        )

        data = state1.to_dict()

        # Deserialize
        state2 = ConversationState.from_dict(data)

        assert state2.schema_version == SCHEMA_VERSION
        assert state2.conversation_id == "test-123"

    def test_from_dict_with_missing_version(self):
        """Test loading data without schema version (legacy data)."""
        # Simulate old data without schema_version field
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Test"),
            node_id=""
        )
        tree = Tree(root_node)

        state = ConversationState(
            conversation_id="test-123",
            message_tree=tree
        )

        data = state.to_dict()
        # Remove schema_version to simulate old data
        del data["schema_version"]

        # Should still load (migrates from 0.0.0 to current)
        state2 = ConversationState.from_dict(data)

        assert state2.schema_version == SCHEMA_VERSION
        assert state2.conversation_id == "test-123"

    def test_migration_from_0_0_0_to_1_0_0(self):
        """Test migration from unversioned to version 1.0.0."""
        data = {
            "conversation_id": "test-123",
            "nodes": [
                {
                    "message": {
                        "role": "system",
                        "content": "Test",
                        "name": None,
                        "metadata": {}
                    },
                    "node_id": "",
                    "timestamp": datetime.now().isoformat(),
                    "prompt_name": None,
                    "branch_name": None,
                    "metadata": {}
                }
            ],
            "edges": [],
            "current_node_id": "",
            "metadata": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
            # No schema_version field - represents legacy data
        }

        # Should migrate successfully
        state = ConversationState.from_dict(data)

        assert state.schema_version == "1.0.0"
        assert state.conversation_id == "test-123"

    def test_migration_logs_version_change(self, caplog):
        """Test that migration logs the version change."""
        import logging

        # Create data with old version
        data = {
            "schema_version": "0.0.0",
            "conversation_id": "test-123",
            "nodes": [
                {
                    "message": {
                        "role": "system",
                        "content": "Test",
                        "name": None,
                        "metadata": {}
                    },
                    "node_id": "",
                    "timestamp": datetime.now().isoformat(),
                    "prompt_name": None,
                    "branch_name": None,
                    "metadata": {}
                }
            ],
            "edges": [],
            "current_node_id": "",
            "metadata": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        with caplog.at_level(logging.INFO):
            state = ConversationState.from_dict(data)

        # Should log migration
        assert "Migrating conversation test-123" in caplog.text
        assert "from schema 0.0.0" in caplog.text

    def test_same_version_no_migration(self):
        """Test that same version doesn't trigger migration."""
        # Create data with current version
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Test"),
            node_id=""
        )
        tree = Tree(root_node)

        state1 = ConversationState(
            conversation_id="test-123",
            message_tree=tree
        )

        data = state1.to_dict()

        # Load it - should not migrate
        state2 = ConversationState.from_dict(data)

        assert state2.schema_version == SCHEMA_VERSION

    def test_downgrade_raises_error(self):
        """Test that downgrading schema version raises error."""
        data = {
            "schema_version": "2.0.0",  # Future version
            "conversation_id": "test-123",
            "nodes": [
                {
                    "message": {
                        "role": "system",
                        "content": "Test",
                        "name": None,
                        "metadata": {}
                    },
                    "node_id": "",
                    "timestamp": datetime.now().isoformat(),
                    "prompt_name": None,
                    "branch_name": None,
                    "metadata": {}
                }
            ],
            "edges": [],
            "current_node_id": "",
            "metadata": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Should raise error when trying to downgrade
        with pytest.raises(SchemaVersionError, match="Cannot downgrade"):
            ConversationState.from_dict(data)

    def test_unknown_minor_version_warns(self, caplog):
        """Test that unknown minor version triggers warning."""
        import logging

        data = {
            "schema_version": "1.5.0",  # Unknown minor version
            "conversation_id": "test-123",
            "nodes": [
                {
                    "message": {
                        "role": "system",
                        "content": "Test",
                        "name": None,
                        "metadata": {}
                    },
                    "node_id": "",
                    "timestamp": datetime.now().isoformat(),
                    "prompt_name": None,
                    "branch_name": None,
                    "metadata": {}
                }
            ],
            "edges": [],
            "current_node_id": "",
            "metadata": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        with caplog.at_level(logging.WARNING):
            state = ConversationState.from_dict(data)

        # Should warn about no migration path
        assert "No migration path defined" in caplog.text

    def test_serialization_roundtrip_preserves_version(self):
        """Test that serialize + deserialize preserves schema version."""
        root_node = ConversationNode(
            message=LLMMessage(role="system", content="Test"),
            node_id=""
        )
        tree = Tree(root_node)

        # Add some child nodes
        user_node = ConversationNode(
            message=LLMMessage(role="user", content="Hello"),
            node_id="0"
        )
        tree.add_child(Tree(user_node))

        state1 = ConversationState(
            conversation_id="test-123",
            message_tree=tree,
            current_node_id="0",
            metadata={"user_id": "alice"}
        )

        # Serialize and deserialize
        data = state1.to_dict()
        state2 = ConversationState.from_dict(data)

        # Everything should match
        assert state2.schema_version == state1.schema_version
        assert state2.conversation_id == state1.conversation_id
        assert state2.current_node_id == state1.current_node_id
        assert state2.metadata == state1.metadata
        assert len(state2.get_current_messages()) == len(state1.get_current_messages())

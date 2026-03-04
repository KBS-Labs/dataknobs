"""Tests for ConversationNode serialization and schema 1.0→1.1 migration.

Verifies that:
- ConversationNode.to_dict()/from_dict() preserves tool_calls and function_call
- Schema migration from 1.0.0 to 1.1.0 reconstructs tool_calls from metadata
- Full ConversationState roundtrip preserves tool_calls
"""

from __future__ import annotations

from datetime import datetime

from dataknobs_llm.conversations import (
    ConversationNode,
    ConversationState,
    SCHEMA_VERSION,
)
from dataknobs_llm.llm.base import LLMMessage, ToolCall
from dataknobs_structures.tree import Tree


class TestConversationNodeToolCallSerialization:
    """ConversationNode roundtrip with tool_calls."""

    def test_roundtrip_with_tool_calls(self) -> None:
        calls = [
            ToolCall(name="search", parameters={"q": "test"}, id="call_1"),
        ]
        msg = LLMMessage(role="assistant", content="", tool_calls=calls)
        node = ConversationNode(message=msg, node_id="0.0")

        data = node.to_dict()
        assert "tool_calls" in data["message"]
        assert data["message"]["tool_calls"][0]["name"] == "search"

        restored = ConversationNode.from_dict(data)
        assert restored.message.tool_calls is not None
        assert len(restored.message.tool_calls) == 1
        assert restored.message.tool_calls[0].name == "search"
        assert restored.message.tool_calls[0].id == "call_1"

    def test_roundtrip_with_function_call(self) -> None:
        fc = {"name": "get_weather", "arguments": '{"city": "NYC"}'}
        msg = LLMMessage(role="assistant", content="", function_call=fc)
        node = ConversationNode(message=msg, node_id="0.0")

        data = node.to_dict()
        assert data["message"]["function_call"] == fc

        restored = ConversationNode.from_dict(data)
        assert restored.message.function_call == fc

    def test_roundtrip_without_tool_calls(self) -> None:
        msg = LLMMessage(role="user", content="hello")
        node = ConversationNode(message=msg, node_id="0")

        data = node.to_dict()
        assert "tool_calls" not in data["message"]

        restored = ConversationNode.from_dict(data)
        assert restored.message.tool_calls is None
        assert restored.message.content == "hello"

    def test_roundtrip_preserves_node_metadata(self) -> None:
        calls = [ToolCall(name="fn", parameters={}, id="c1")]
        msg = LLMMessage(role="assistant", content="", tool_calls=calls)
        node = ConversationNode(
            message=msg,
            node_id="0.0",
            metadata={"usage": {"tokens": 100}, "model": "test"},
        )

        data = node.to_dict()
        restored = ConversationNode.from_dict(data)
        assert restored.metadata["usage"]["tokens"] == 100
        assert restored.message.tool_calls is not None


class TestSchemaMigration1_0To1_1:
    """Tests for schema 1.0.0 → 1.1.0 migration."""

    def _make_v1_0_data(
        self,
        *,
        tool_calls: list[dict] | None = None,
        function_call: dict | None = None,
    ) -> dict:
        """Build a schema 1.0.0 ConversationState dict.

        In schema 1.0, ConversationNode.to_dict() did NOT include tool_calls
        or function_call in the message dict. ConversationManager stored them
        as a backup in node metadata.
        """
        nodes = [
            {
                "node_id": "",
                "message": {
                    "role": "system",
                    "content": "You are helpful",
                    "name": None,
                    "metadata": {},
                },
                "timestamp": "2024-01-01T00:00:00",
                "prompt_name": None,
                "branch_name": None,
                "metadata": {},
            },
            {
                "node_id": "0",
                "message": {
                    "role": "user",
                    "content": "Do something",
                    "name": None,
                    "metadata": {},
                },
                "timestamp": "2024-01-01T00:00:01",
                "prompt_name": None,
                "branch_name": None,
                "metadata": {},
            },
        ]

        # Assistant node — message lacks tool_calls, metadata has backup
        assistant_meta: dict = {"usage": {"completion_tokens": 50}}
        assistant_message: dict = {
            "role": "assistant",
            "content": "",
            "name": None,
            "metadata": {},
        }
        if tool_calls is not None:
            assistant_meta["tool_calls"] = tool_calls
        if function_call is not None:
            assistant_meta["function_call"] = function_call

        nodes.append({
            "node_id": "0.0",
            "message": assistant_message,
            "timestamp": "2024-01-01T00:00:02",
            "prompt_name": None,
            "branch_name": None,
            "metadata": assistant_meta,
        })

        return {
            "schema_version": "1.0.0",
            "conversation_id": "test-migration",
            "nodes": nodes,
            "edges": [["", "0"], ["0", "0.0"]],
            "current_node_id": "0.0",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:02",
        }

    def test_migration_reconstructs_tool_calls(self) -> None:
        tool_calls_backup = [
            {"name": "search", "parameters": {"q": "test"}, "id": "c1"},
        ]
        data = self._make_v1_0_data(tool_calls=tool_calls_backup)

        state = ConversationState.from_dict(data)
        assert state.schema_version == SCHEMA_VERSION

        # Navigate to assistant node and check tool_calls
        nodes = state.get_current_nodes()
        assistant_node = nodes[-1]
        assert assistant_node.message.role == "assistant"
        assert assistant_node.message.tool_calls is not None
        assert len(assistant_node.message.tool_calls) == 1
        assert assistant_node.message.tool_calls[0].name == "search"
        assert assistant_node.message.tool_calls[0].id == "c1"

    def test_migration_reconstructs_function_call(self) -> None:
        fc = {"name": "calc", "arguments": '{"expr": "2+2"}'}
        data = self._make_v1_0_data(function_call=fc)

        state = ConversationState.from_dict(data)
        nodes = state.get_current_nodes()
        assistant_node = nodes[-1]
        assert assistant_node.message.function_call == fc

    def test_migration_no_tool_calls_is_noop(self) -> None:
        """Nodes without tool_calls in metadata are unaffected."""
        data = self._make_v1_0_data()

        state = ConversationState.from_dict(data)
        nodes = state.get_current_nodes()
        assistant_node = nodes[-1]
        assert assistant_node.message.tool_calls is None
        assert assistant_node.message.function_call is None

    def test_migration_does_not_overwrite_existing(self) -> None:
        """If message already has tool_calls, migration does not overwrite."""
        data = self._make_v1_0_data(
            tool_calls=[{"name": "backup", "parameters": {}, "id": "old"}],
        )
        # Simulate a message that somehow already has tool_calls
        data["nodes"][2]["message"]["tool_calls"] = [
            {"name": "existing", "parameters": {}, "id": "new"},
        ]

        state = ConversationState.from_dict(data)
        nodes = state.get_current_nodes()
        assistant_node = nodes[-1]
        assert assistant_node.message.tool_calls is not None
        assert assistant_node.message.tool_calls[0].name == "existing"

    def test_current_version(self) -> None:
        assert SCHEMA_VERSION == "1.1.0"


class TestConversationStateRoundtripWithToolCalls:
    """Full roundtrip: create state with tool_calls → to_dict → from_dict."""

    def test_full_roundtrip(self) -> None:
        calls = [
            ToolCall(name="search", parameters={"q": "python"}, id="call_1"),
            ToolCall(name="calc", parameters={"expr": "1+1"}, id="call_2"),
        ]

        root = ConversationNode(
            message=LLMMessage(role="system", content="You are helpful"),
            node_id="",
        )
        user = ConversationNode(
            message=LLMMessage(role="user", content="Help me"),
            node_id="0",
        )
        assistant = ConversationNode(
            message=LLMMessage(role="assistant", content="", tool_calls=calls),
            node_id="0.0",
            metadata={"usage": {"completion_tokens": 100}},
        )

        tree = Tree(root)
        user_tree = tree.add_child(Tree(user))
        user_tree.add_child(Tree(assistant))

        state = ConversationState(
            conversation_id="roundtrip-test",
            message_tree=tree,
            current_node_id="0.0",
            metadata={"user_id": "test"},
        )

        data = state.to_dict()
        restored = ConversationState.from_dict(data)

        nodes = restored.get_current_nodes()
        assert len(nodes) == 3
        restored_assistant = nodes[-1]
        assert restored_assistant.message.tool_calls is not None
        assert len(restored_assistant.message.tool_calls) == 2
        assert restored_assistant.message.tool_calls[0].name == "search"
        assert restored_assistant.message.tool_calls[0].id == "call_1"
        assert restored_assistant.message.tool_calls[1].name == "calc"

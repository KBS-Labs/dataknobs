"""Tests for ToolCall and LLMMessage canonical serialization.

Verifies that:
- ToolCall.to_dict()/from_dict() roundtrips correctly
- LLMMessage.to_dict()/from_dict() roundtrips correctly with all field combinations
- Optional fields are omitted when absent (clean output)
- Legacy formats (without tool_calls/function_call) still deserialize
"""

from __future__ import annotations

import pytest
from dataknobs_common.exceptions import ValidationError
from dataknobs_llm.llm.base import LLMMessage, ToolCall


class TestToolCallSerialization:
    """Tests for ToolCall.to_dict() and from_dict()."""

    def test_roundtrip_all_fields(self) -> None:
        tc = ToolCall(name="search", parameters={"q": "test"}, id="call_1")
        data = tc.to_dict()
        restored = ToolCall.from_dict(data)
        assert restored.name == tc.name
        assert restored.parameters == tc.parameters
        assert restored.id == tc.id

    def test_roundtrip_without_id(self) -> None:
        tc = ToolCall(name="search", parameters={"q": "test"})
        data = tc.to_dict()
        assert "id" not in data
        restored = ToolCall.from_dict(data)
        assert restored.name == "search"
        assert restored.parameters == {"q": "test"}
        assert restored.id is None

    def test_to_dict_format(self) -> None:
        tc = ToolCall(name="fn", parameters={"a": 1}, id="c1")
        data = tc.to_dict()
        assert data == {"name": "fn", "parameters": {"a": 1}, "id": "c1"}

    def test_from_dict_missing_parameters(self) -> None:
        data = {"name": "fn"}
        tc = ToolCall.from_dict(data)
        assert tc.name == "fn"
        assert tc.parameters == {}

    def test_roundtrip_complex_parameters(self) -> None:
        params = {"nested": {"key": [1, 2, 3]}, "flag": True}
        tc = ToolCall(name="complex", parameters=params, id="c2")
        restored = ToolCall.from_dict(tc.to_dict())
        assert restored.parameters == params


class TestLLMMessageSerialization:
    """Tests for LLMMessage.to_dict() and from_dict()."""

    def test_roundtrip_minimal(self) -> None:
        msg = LLMMessage(role="user", content="hello")
        data = msg.to_dict()
        restored = LLMMessage.from_dict(data)
        assert restored.role == "user"
        assert restored.content == "hello"
        assert restored.name is None
        assert restored.tool_calls is None
        assert restored.function_call is None
        assert restored.metadata == {}

    def test_to_dict_omits_absent_fields(self) -> None:
        msg = LLMMessage(role="user", content="hi")
        data = msg.to_dict()
        assert "name" not in data
        assert "tool_calls" not in data
        assert "function_call" not in data
        assert "metadata" not in data

    def test_roundtrip_with_name(self) -> None:
        msg = LLMMessage(role="tool", content='{"ok": true}', name="search")
        data = msg.to_dict()
        assert data["name"] == "search"
        restored = LLMMessage.from_dict(data)
        assert restored.name == "search"

    def test_roundtrip_with_tool_calls(self) -> None:
        calls = [
            ToolCall(name="search", parameters={"q": "test"}, id="c1"),
            ToolCall(name="calc", parameters={"expr": "2+2"}, id="c2"),
        ]
        msg = LLMMessage(role="assistant", content="", tool_calls=calls)
        data = msg.to_dict()
        assert len(data["tool_calls"]) == 2
        assert data["tool_calls"][0]["name"] == "search"

        restored = LLMMessage.from_dict(data)
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].name == "search"
        assert restored.tool_calls[0].id == "c1"
        assert restored.tool_calls[1].name == "calc"
        assert restored.tool_calls[1].parameters == {"expr": "2+2"}

    def test_roundtrip_with_function_call(self) -> None:
        fc = {"name": "get_weather", "arguments": '{"city": "NYC"}'}
        msg = LLMMessage(role="assistant", content="", function_call=fc)
        data = msg.to_dict()
        assert data["function_call"] == fc
        restored = LLMMessage.from_dict(data)
        assert restored.function_call == fc

    def test_roundtrip_with_metadata(self) -> None:
        msg = LLMMessage(
            role="assistant",
            content="response",
            metadata={"model": "test", "tokens": 42},
        )
        data = msg.to_dict()
        assert data["metadata"]["model"] == "test"
        restored = LLMMessage.from_dict(data)
        assert restored.metadata["tokens"] == 42

    def test_roundtrip_all_fields(self) -> None:
        calls = [ToolCall(name="fn", parameters={"x": 1}, id="c1")]
        fc = {"name": "legacy_fn", "arguments": "{}"}
        msg = LLMMessage(
            role="assistant",
            content="text",
            name="assistant_name",
            tool_calls=calls,
            function_call=fc,
            metadata={"key": "value"},
        )
        data = msg.to_dict()
        restored = LLMMessage.from_dict(data)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.name == msg.name
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].name == "fn"
        assert restored.function_call == fc
        assert restored.metadata == {"key": "value"}

    def test_from_dict_legacy_format(self) -> None:
        """Deserialize a dict without tool_calls or function_call (schema 1.0)."""
        data = {
            "role": "assistant",
            "content": "hello",
            "name": None,
            "metadata": {},
        }
        msg = LLMMessage.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.function_call is None

    def test_from_dict_empty_tool_calls_list(self) -> None:
        """Empty tool_calls list should result in None (not empty list)."""
        data = {"role": "assistant", "content": "", "tool_calls": []}
        msg = LLMMessage.from_dict(data)
        assert msg.tool_calls is None

    def test_from_dict_missing_content(self) -> None:
        """Missing content defaults to empty string."""
        data = {"role": "system"}
        msg = LLMMessage.from_dict(data)
        assert msg.content == ""

    def test_roundtrip_with_tool_call_id(self) -> None:
        """tool_call_id should roundtrip through to_dict/from_dict."""
        msg = LLMMessage(
            role="tool",
            content='{"result": "ok"}',
            name="search",
            tool_call_id="toolu_abc123",
        )
        data = msg.to_dict()
        assert data["tool_call_id"] == "toolu_abc123"
        restored = LLMMessage.from_dict(data)
        assert restored.tool_call_id == "toolu_abc123"

    def test_to_dict_omits_tool_call_id_when_none(self) -> None:
        """tool_call_id should not appear in dict when None."""
        msg = LLMMessage(role="tool", content="result", name="search")
        data = msg.to_dict()
        assert "tool_call_id" not in data

    def test_from_dict_missing_tool_call_id(self) -> None:
        """Missing tool_call_id in dict should default to None (backward compat)."""
        data = {"role": "tool", "content": "result", "name": "search"}
        msg = LLMMessage.from_dict(data)
        assert msg.tool_call_id is None


class TestLoadNormalizesParameters:
    """A ToolCall read back from storage carries the shape its type declares.

    ``from_dict`` built ``parameters`` from the stored value verbatim, so the
    one guarantee ``ToolCall.parameters: Dict[str, Any]`` makes held for calls
    a provider had just parsed and not for calls read back from a conversation
    store or a capture. Consumers splat it either way, so a record persisted
    before the Ollama adapter was fixed -- arguments JSON-encoded as a string,
    which is what a model emitting them that way produces -- reloaded as a
    ``str`` and raised ``TypeError`` at the splat site, a long way from the
    load that produced it.

    Loading now goes through the same normalization as a live parse. Repair is
    what makes those legacy records usable rather than merely loadable: the
    realistic population is JSON strings that decode, because ``to_dict``
    writes ``parameters`` verbatim and only a bad in-memory call could have
    persisted a bad value.

    Failing before the fix: every test here except the two controls.
    """

    def test_a_persisted_json_string_is_repaired(self) -> None:
        restored = ToolCall.from_dict({"name": "search", "parameters": '{"q": "cats"}'})
        assert restored.parameters == {"q": "cats"}

    def test_loaded_parameters_are_splattable(self) -> None:
        """The guarantee consumers actually rely on."""
        restored = ToolCall.from_dict({"name": "search", "parameters": '{"q": "cats"}'})
        assert dict(**restored.parameters) == {"q": "cats"}

    def test_unreadable_parameters_raise_at_the_load(self) -> None:
        """Not at the splat site, and not silently as ``{}``."""
        with pytest.raises(ValidationError, match="search"):
            ToolCall.from_dict({"name": "search", "parameters": "not json at all"})

    def test_a_json_scalar_is_not_an_arguments_object(self) -> None:
        with pytest.raises(ValidationError, match="search"):
            ToolCall.from_dict({"name": "search", "parameters": "5"})

    def test_a_message_repairs_the_calls_it_carries(self) -> None:
        msg = LLMMessage.from_dict(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "search", "parameters": '{"q": "cats"}', "id": "c1"}],
            }
        )
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].parameters == {"q": "cats"}

    def test_a_well_formed_mapping_is_unchanged(self) -> None:
        """Control: the overwhelmingly common case is untouched."""
        data = {"name": "search", "parameters": {"q": "cats"}, "id": "c1"}
        restored = ToolCall.from_dict(data)
        assert restored.parameters == {"q": "cats"}
        assert restored.to_dict() == data

    def test_absent_parameters_load_as_an_empty_mapping(self) -> None:
        """Control: a tool taking no arguments."""
        assert ToolCall.from_dict({"name": "ping"}).parameters == {}


class TestConversationLoadNamesTheBadNode:
    """An unreadable stored call says which node carries it.

    ``ConversationNode.from_dict`` reconstitutes one node of a stored
    conversation, and the loader builds every node before it builds the tree,
    so a call whose arguments cannot be read fails the whole load. That is the
    right outcome -- the alternative is a conversation whose history is quietly
    not what was recorded -- but the bare error names only the tool, and a
    consumer holding a stored conversation needs to know which record to
    repair. Failing before the fix: the error mentions the tool and not the
    node.
    """

    def test_the_error_names_the_node(self) -> None:
        from dataknobs_llm.conversations.storage import ConversationNode

        with pytest.raises(ValidationError, match="n-7") as excinfo:
            ConversationNode.from_dict(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "search", "parameters": "not json"}],
                    },
                    "node_id": "n-7",
                    "timestamp": "2026-01-01T00:00:00",
                }
            )
        # Documented alongside the message, so a caller can find the record
        # without parsing prose.
        assert excinfo.value.context["node_id"] == "n-7"
        assert excinfo.value.context["tool"] == "search"

    def test_the_error_still_names_the_tool(self) -> None:
        from dataknobs_llm.conversations.storage import ConversationNode

        with pytest.raises(ValidationError, match="search"):
            ConversationNode.from_dict(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "search", "parameters": "not json"}],
                    },
                    "node_id": "n-7",
                    "timestamp": "2026-01-01T00:00:00",
                }
            )

    def test_a_readable_node_loads(self) -> None:
        """Control."""
        from dataknobs_llm.conversations.storage import ConversationNode

        node = ConversationNode.from_dict(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "search", "parameters": '{"q": "cats"}'}],
                },
                "node_id": "n-7",
                "timestamp": "2026-01-01T00:00:00",
            }
        )
        assert node.message.tool_calls is not None
        assert node.message.tool_calls[0].parameters == {"q": "cats"}

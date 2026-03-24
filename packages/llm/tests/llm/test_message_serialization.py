"""Tests for ToolCall and LLMMessage canonical serialization.

Verifies that:
- ToolCall.to_dict()/from_dict() roundtrips correctly
- LLMMessage.to_dict()/from_dict() roundtrips correctly with all field combinations
- Optional fields are omitted when absent (clean output)
- Legacy formats (without tool_calls/function_call) still deserialize
"""

from __future__ import annotations

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

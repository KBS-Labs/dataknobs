"""Tests for Ollama tool-call parsing in ``OllamaAdapter.adapt_response``.

Bug: the adapter read ``message.tool_calls[].function.arguments`` straight into
``ToolCall.parameters``, which is declared ``Dict[str, Any]``, with no shape
check at all. Ollama normally sends a JSON object there, but a model that emits
JSON-encoded *arguments* instead produced a ``ToolCall`` whose ``parameters``
was a ``str`` — invisible to the type checker, since the declared type is
``Dict[str, Any]`` and nothing enforces it, and fatal at execution time, where
every consumer splats it (``tool.execute(**tool_call.parameters)``).

Ollama was the only adapter without a guard: OpenAI parses its string-typed
wire field with ``json.loads``, and the two Claude-family adapters check
``isinstance(..., dict)``.

Fix: normalize in the adapter — pass a mapping through, parse a JSON string,
and raise on a string that will not parse, so a tool call that cannot be
executed is reported where it is parsed rather than where it is splatted.

The first three tests FAIL against the unfixed adapter (``parameters`` is a
``str``; the splat raises ``TypeError``; nothing raises on unparseable
arguments). The rest close a standing coverage gap: nothing tested Ollama
tool-call parsing at all, which is why the defect survived.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_llm.llm.providers.ollama import OllamaAdapter


def _chat_response(tool_calls: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Build an Ollama ``/api/chat`` payload carrying the given tool calls."""
    payload: dict[str, Any] = {
        "model": "llama3.2:3b",
        "done": True,
        "done_reason": "stop",
        "message": {"content": "", "tool_calls": tool_calls},
    }
    payload.update(extra)
    return payload


def _call(name: str, arguments: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


def test_string_arguments_are_parsed_into_a_dict() -> None:
    """A JSON-encoded ``arguments`` string becomes the mapping it encodes."""
    parsed = OllamaAdapter().adapt_response(
        _chat_response([_call("get_time", '{"tz": "UTC"}')])
    )

    assert parsed.tool_calls is not None
    (call,) = parsed.tool_calls
    assert isinstance(call.parameters, dict)
    assert call.parameters == {"tz": "UTC"}


def test_parameters_are_always_splattable() -> None:
    """The shape consumers actually depend on.

    ``react.py`` and ``bot/base.py`` both execute a tool as
    ``tool.execute(**tool_call.parameters, ...)``. A ``str`` there raises
    ``TypeError: dict() argument after ** must be a mapping, not str`` at the
    call site — a long way from the adapter that produced it.
    """
    parsed = OllamaAdapter().adapt_response(
        _chat_response(
            [
                _call("get_weather", {"city": "Paris"}),
                _call("get_time", '{"tz": "UTC"}'),
            ]
        )
    )

    assert parsed.tool_calls is not None
    for call in parsed.tool_calls:
        assert dict(**call.parameters) == call.parameters


def test_unparseable_arguments_raise_rather_than_reaching_the_consumer() -> None:
    """A tool call whose arguments cannot be read is not a tool call.

    Passing it through hands the consumer something ``**`` rejects; dropping it
    silently is the failure mode this module's sibling defects were made of.
    """
    with pytest.raises(ValidationError) as excinfo:
        OllamaAdapter().adapt_response(_chat_response([_call("get_time", "not json")]))

    assert "get_time" in str(excinfo.value)


def test_object_arguments_pass_through_unchanged() -> None:
    """The ordinary case is untouched — same object, not a copy round-trip."""
    arguments = {"city": "Paris", "units": "metric"}
    parsed = OllamaAdapter().adapt_response(_chat_response([_call("get_weather", arguments)]))

    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0].parameters == arguments


def test_missing_arguments_become_an_empty_mapping() -> None:
    """A tool taking no arguments is still splattable."""
    parsed = OllamaAdapter().adapt_response(_chat_response([{"function": {"name": "ping"}}]))

    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0].parameters == {}


def test_every_tool_call_survives_the_parse() -> None:
    """Coverage gap: nothing asserted that the adapter keeps them all."""
    parsed = OllamaAdapter().adapt_response(
        _chat_response(
            [
                _call("a", {"i": 1}),
                _call("b", {"i": 2}),
                _call("c", {"i": 3}),
            ]
        )
    )

    assert parsed.tool_calls is not None
    assert [c.name for c in parsed.tool_calls] == ["a", "b", "c"]
    assert parsed.finish_reason == "tool_calls"


def test_a_truncated_tool_call_turn_is_flagged() -> None:
    """``done_reason: "length"`` on a tool-call turn is the dangerous case."""
    parsed = OllamaAdapter().adapt_response(
        _chat_response([_call("a", {"i": 1})], done_reason="length")
    )

    assert parsed.truncated is True

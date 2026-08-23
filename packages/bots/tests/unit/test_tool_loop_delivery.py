"""The tool-loop delivery contract, asked of both modes at once.

``bot/tool_loop.py`` factors the buffered (``chat``) and streaming
(``stream_chat``) tool-execution loops behind one ``_ToolLoopDelivery``
interface, and ``test_monolithic_tool_loop_unification.py`` pins what the
*loop* does with it. These pin what the deliveries themselves promise, which
the loop cannot reach: it asks ``has_pending()`` before ``pending_calls()``
every time, so the two answering differently was invisible from there.

Direct instantiation of the private delivery classes is deliberate -- the
interface is the subject, not a bot flow. Behavioral coverage of the loop that
drives them lives in the unification file and in
``test_tool_execution_loop.py``.
"""

from __future__ import annotations

from typing import Any

from dataknobs_bots.bot.tool_loop import _BufferedDelivery, _StreamingDelivery


class _Response:
    """Minimal stand-in for a provider response carrying ``tool_calls``."""

    def __init__(self, tool_calls: list[Any] | None) -> None:
        self.tool_calls = tool_calls


class _NoToolCalls:
    """A provider response that never grew a ``tool_calls`` attribute."""


def _buffered(response: Any) -> _BufferedDelivery:
    return _BufferedDelivery(response, recall_kwargs={}, turn_timeout=1.0)


def _streaming(pending: list[Any] | None) -> _StreamingDelivery:
    return _StreamingDelivery(pending, provider=None, has_tools=False, recall_kwargs={})


class TestPendingCallsAgreeAcrossModes:
    """Both deliveries must answer "what is pending" the same way.

    Buffered answered the question with two different tolerances: ``getattr(
    self.response, "tool_calls", None)`` for the boolean, a bare
    ``self.response.tool_calls`` for the list. So a response object carrying no
    ``tool_calls`` attribute at all made ``has_pending()`` return ``False`` and
    ``pending_calls()`` raise ``AttributeError``. Streaming was ``None``-safe
    in both.

    The loop's fixed ask-order kept that latent. What it does not protect is a
    delivery added later that copies one method's idiom and not the other's, or
    a caller that reaches for ``pending_calls()`` first -- and the declared
    return of ``list[Any] | None`` invited exactly that, since it promised a
    ``None`` the buffered mode would raise rather than return.
    """

    def test_nothing_pending_is_an_empty_list_in_both_modes(self) -> None:
        """Not ``None`` from one and an exception from the other."""
        for label, delivery in (
            ("buffered, response has no tool_calls", _buffered(_NoToolCalls())),
            ("buffered, tool_calls is None", _buffered(_Response(None))),
            ("streaming, pending is None", _streaming(None)),
        ):
            assert delivery.has_pending() is False, label
            assert delivery.pending_calls() == [], label

    def test_pending_calls_round_trip_in_both_modes(self) -> None:
        """The empty case must not have flattened the populated one."""
        calls: list[Any] = [{"name": "search"}]
        for label, delivery in (
            ("buffered", _buffered(_Response(calls))),
            ("streaming", _streaming(calls)),
        ):
            assert delivery.has_pending() is True, label
            assert delivery.pending_calls() == calls, label

    def test_streaming_clears_pending_to_empty_not_none(self) -> None:
        """``clear_pending_after_execute`` runs before the budget gate.

        A budget-break right after it must find nothing pending rather than
        the ``None`` that used to reach the caller as a declared return value.
        """
        delivery = _streaming([{"name": "search"}])
        delivery.clear_pending_after_execute()
        assert delivery.has_pending() is False
        assert delivery.pending_calls() == []

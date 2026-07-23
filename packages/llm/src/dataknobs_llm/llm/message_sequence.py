"""Shared message-sequence invariants over ``list[LLMMessage]``.

Provider Messages APIs impose structural invariants on the conversation
history a request carries — Anthropic's Messages API in particular rejects a
dangling assistant ``tool_use`` that has no following ``tool_result`` (a 400),
and rejects consecutive same-role messages. These invariants are not specific
to any one reasoning strategy or provider: any code assembling a history to
send to a completion needs the same guarantees.

This module is the shared, provider-agnostic home for those invariants,
expressed as **pure functions over** ``list[LLMMessage]`` (never mutating the
input). It sits in the lower ``dataknobs_llm.llm`` layer — below both the
provider adapters and the ``conversations`` package — so the adapters and any
higher layer (a ``ConversationManager`` wrapper, a reasoning strategy) can
reuse a single implementation instead of re-deriving the same rules. A
manager-coupled helper could not live here (the ``conversations → llm`` import
direction is one-way; importing upward would cycle), which is why the core is
kept free of any ``ConversationManager`` dependency.

Current invariants:

- :func:`pair_orphan_tool_calls` — pair a dangling assistant ``tool_use`` with
  a synthetic ``tool_result`` so the request is structurally valid on every
  backend.
"""

from __future__ import annotations

import json
from typing import Any

from .base import LLMMessage

#: Guidance carried by a synthetic ``tool_result`` when a tool call was
#: never *reached* — the reasoning loop ended before executing it (max
#: iterations or a consumer-level tool-loop timeout).  It both satisfies the
#: provider ``tool_use`` → ``tool_result`` pairing contract *and* informs the
#: synthesis call, subsuming the intent of a mid-conversation ``role="system"``
#: "loop ended, use existing results" notice.
_UNEXECUTED_TOOL_RESULT = (
    "[Tool result unavailable: the reasoning loop ended before this call "
    "was executed. Use the results already in the conversation to respond "
    "to the user.]"
)


def _duplicate_tool_result(name: str) -> str:
    """Guidance for an orphan that repeats an already-answered call.

    Restores the nuance of a duplicate-break ``role="system"`` notice: the
    abandoned ``tool_use`` is the second half of a duplicate tool call — the
    same tool invoked with identical parameters whose result is already in the
    conversation — so the synthesis call is steered to reuse that result rather
    than treating the call as merely unreached.
    """
    return (
        f"[Tool result unavailable: '{name}' was already called with "
        "identical parameters earlier in this conversation. Use those "
        "existing results to respond to the user.]"
    )


def _tool_call_signature(tc: Any) -> tuple[str, str]:
    """Canonical ``(name, params)`` identity of a tool call.

    Mirrors a reasoning loop's own duplicate-detection key
    (``json.dumps(parameters, sort_keys=True)``) so "is this orphan a
    duplicate?" matches "did the loop break on a duplicate?" exactly.
    """
    return (tc.name, json.dumps(tc.parameters, sort_keys=True))


def pair_orphan_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Return synthetic ``tool_result`` messages pairing dangling ``tool_use``.

    Pure over ``messages`` (never mutates the input): scans a conversation
    history and returns a new ``role="tool"`` message for every assistant
    ``tool_use`` that has no following ``tool_result``, in first-encountered
    order.  The caller appends the returned messages to the history before
    re-sending it to a synthesis completion.  Returns an empty list when the
    history is already well-formed (the common case), so it is a safe no-op
    on every path.

    Providers differ in how strictly they enforce the pairing — Anthropic's
    Messages API rejects a dangling ``tool_use`` with a 400, while others
    tolerate it — so the invariant is enforced here, at the message-sequence
    layer, on every backend.

    The pairing key mirrors the adapters' own (``tc.id or tc.name``), so it
    pairs correctly whether or not the provider assigned tool-call ids, and it
    is idempotent — an already-answered ``tool_use`` is skipped, so re-running
    it over a partially-paired history appends nothing new.

    Guidance is route-aware *without* the caller passing its route down: an
    orphan whose canonical ``(name, parameters)`` matches an earlier
    *answered* call is the abandoned half of a duplicate-tool-call break, so
    it carries the "already called with identical parameters — use those
    results" guidance (:func:`_duplicate_tool_result`); any other orphan (a
    call the loop never reached) carries the generic "loop ended before
    execution" guidance (:data:`_UNEXECUTED_TOOL_RESULT`).

    Id-less duplicate names (``tc.id is None`` with a repeated ``name``): the
    ``(tool_call_id or name)`` key collapses same-name calls, so once a first
    same-name call is answered, a later same-name orphan is treated as
    already-answered and skipped rather than paired.  This is intentional and
    structurally safe — the adapters key *every* same-name ``tool_use`` block
    to the same ``name`` when no id is present, so the single existing
    ``tool_result`` pairs them all and no dangling block survives adaptation.
    (Strict providers such as Anthropic always assign ids, so the collapse
    cannot arise there.)

    The pure ``list[LLMMessage]`` signature is deliberate: this same invariant
    is needed by the provider adapters' message-consolidation path, which
    operates on message lists in this ``dataknobs_llm.llm`` layer.  Keeping the
    core free of any ``ConversationManager`` dependency lets it be shared
    rather than reimplemented (a manager-coupled helper could not live in this
    layer without an import cycle).

    Args:
        messages: Conversation history about to be re-sent to a synthesis
            completion call.

    Returns:
        New ``role="tool"`` messages to append (possibly empty).  The input
        list is never mutated.
    """
    answered: set[str] = {
        key
        for m in messages
        if m.role == "tool"
        and (key := (m.tool_call_id or m.name)) is not None
    }
    # Canonical signatures of the calls that DID get answered, so an orphan
    # repeating one is recognized as the abandoned half of a duplicate break.
    answered_signatures: set[tuple[str, str]] = {
        _tool_call_signature(tc)
        for m in messages
        if m.role == "assistant" and m.tool_calls
        for tc in m.tool_calls
        if (tc.id or tc.name) in answered
    }

    results: list[LLMMessage] = []
    for msg in messages:
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            key = tc.id or tc.name
            if key in answered:
                continue
            content = (
                _duplicate_tool_result(tc.name)
                if _tool_call_signature(tc) in answered_signatures
                else _UNEXECUTED_TOOL_RESULT
            )
            results.append(
                LLMMessage(
                    role="tool",
                    content=content,
                    name=tc.name,
                    tool_call_id=tc.id,
                )
            )
            answered.add(key)
    return results

"""Shared ``ConversationManager`` adapter over the orphan-``tool_use`` core.

A dangling assistant ``tool_use`` (a tool call with no following
``tool_result``) left in conversation history is a hard 400 on Anthropic's
Messages API when that history is re-sent — and a latent structural defect on
every other backend.  The provider-agnostic invariant that closes it is the
pure :func:`dataknobs_llm.llm.message_sequence.pair_orphan_tool_calls` core
(``list[LLMMessage] -> list[LLMMessage]``), which lives in the lower
``dataknobs_llm`` layer so it cannot depend on a ``ConversationManager``.

This module is the single thin adapter that binds that pure core to a
``ConversationManager``: read the history, run the core, append whatever
synthetic ``tool_result`` messages it yields.  It is shared by every consumer
that must guarantee a well-formed history before it is persisted or re-sent —
both a strategy that re-completes mid-turn (ReAct's Layer B, *before* its
synthesis call) and the turn-finalize chokepoint (``DynaBot._finalize_turn``'s
Layer A).  Layer A is gated on ``TurnState.tool_loop_left_pending_call`` so it
runs only for the monolithic-loop break/cap routes that can leave an orphan;
phased strategies pair their own orphan via Layer B and so never rely on it
(see that field's CONTRACT note before adding a new orphan-producing path).
Keeping it in one place is deliberate: re-typing the six lines into each caller
is the duplicated orchestration the shared-behavior-extraction mandate rejects.
"""

from __future__ import annotations

from typing import Any

from dataknobs_llm.llm.message_sequence import pair_orphan_tool_calls


async def pair_orphan_tool_calls_on_manager(manager: Any) -> None:
    """Append synthetic ``tool_result``s for any dangling ``tool_use``.

    Thin :class:`~dataknobs_llm.conversations.ConversationManager` adapter over
    the pure :func:`~dataknobs_llm.llm.message_sequence.pair_orphan_tool_calls`
    core.  Reads the manager's history via its public API, runs the pure core,
    and appends whatever ``role="tool"`` results it yields — so a subsequent
    provider request (a strategy's mid-turn synthesis, or the next turn's
    replay of the persisted history) is structurally valid on every backend.

    Idempotent: the pure core returns an empty list for an already-paired
    history (the common case — every happy-path, wizard, and
    already-paired turn), so this is a safe no-op on every path.  The manager
    is duck-typed (``Any``) so this adapter imports only the pure ``dataknobs_llm``
    core and takes no dependency on the ``conversations`` package.

    Args:
        manager: Conversation manager whose history is about to be persisted
            or re-sent to a completion call.
    """
    history = await manager.get_history()
    for result in pair_orphan_tool_calls(history):
        await manager.add_message(
            role="tool",
            content=result.content,
            name=result.name,
            tool_call_id=result.tool_call_id,
        )

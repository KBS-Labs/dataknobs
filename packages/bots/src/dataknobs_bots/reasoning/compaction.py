"""In-loop history-compaction strategies for the ReAct reasoning loop.

The *mechanism* (a pairing-aware re-root of the conversation path) lives on
``dataknobs_llm``'s ``ConversationManager.compact_history``; this module is the
bots-side *policy* seam that decides how the dropped span is handled. A
:class:`CompactionStrategy` is the extension point — the two shipped reference
impls (:class:`WindowCompaction`, :class:`SummarizeCompaction`) cover the common
cases, and a consumer can inject their own via the reasoning ``components``
channel (``compaction_strategy=...``) without subclassing ``ReActReasoning``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dataknobs_llm import LLMSummarizer, Summarizer


@runtime_checkable
class CompactionStrategy(Protocol):
    """How a ReAct loop compacts its history when over budget.

    One async method: given the conversation ``manager`` and how many recent
    tool iterations to keep, perform the compaction and return the number of
    iterations compacted (``0`` == nothing done). Implementations delegate to
    ``manager.compact_history`` (the shared, pairing-safe primitive) — the
    strategy only decides *whether/how* the dropped span is preserved.
    """

    async def compact(
        self, manager: Any, *, keep_recent_iterations: int
    ) -> int:
        """Compact ``manager``'s history, keeping the most recent iterations."""
        ...


class WindowCompaction:
    """Drop the oldest complete tool iterations (LLM-free — the default).

    Cheap, deterministic, no extra provider call: the oldest iteration-pairs are
    simply windowed out of the active path.
    """

    async def compact(
        self, manager: Any, *, keep_recent_iterations: int
    ) -> int:
        return await manager.compact_history(keep_recent_iterations)


class SummarizeCompaction:
    """Fold the oldest tool iterations into a single summary node.

    Information-preserving at the cost of one LLM call per compaction. Wraps any
    :class:`~dataknobs_llm.summarization.Summarizer` (default
    :class:`~dataknobs_llm.summarization.LLMSummarizer` over a provider).
    """

    def __init__(self, summarizer: Summarizer) -> None:
        self._summarizer = summarizer

    async def compact(
        self, manager: Any, *, keep_recent_iterations: int
    ) -> int:
        return await manager.compact_history(
            keep_recent_iterations, summarizer=self._summarizer
        )


def build_compaction_strategy(
    strategy: str, *, summary_provider: Any
) -> CompactionStrategy:
    """Build a reference :class:`CompactionStrategy` for a config ``strategy``.

    Args:
        strategy: ``"window"`` or ``"summarize"`` (validated upstream by
            ``HistoryCompactionConfig``).
        summary_provider: The provider the summarize strategy summarizes with
            (the bot's main provider, or a dedicated one built from
            ``summary_llm``). Unused for ``"window"``.

    Returns:
        A ready :class:`CompactionStrategy`.
    """
    if strategy == "summarize":
        return SummarizeCompaction(LLMSummarizer(summary_provider))
    return WindowCompaction()

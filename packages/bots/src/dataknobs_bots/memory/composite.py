"""Composite memory strategy combining multiple memory implementations."""

import logging
from typing import Any

from .base import Memory

logger = logging.getLogger(__name__)


class CompositeMemory(Memory):
    """Combines multiple memory strategies into one.

    All sub-strategies receive every ``add_message()`` call independently.
    On ``get_context()``, the primary strategy's results appear first,
    followed by deduplicated results from secondary strategies.

    Graceful degradation: if any strategy fails on a read or write,
    the composite logs a warning and continues with the remaining
    strategies.

    Attributes:
        primary: The primary memory strategy (results appear first).
        strategies: All sub-strategies in order.
    """

    def __init__(
        self,
        strategies: list[Memory],
        *,
        primary_index: int = 0,
    ) -> None:
        """Initialize composite memory.

        Args:
            strategies: List of memory strategy instances.
            primary_index: Index of the primary strategy in the list.

        Raises:
            ValueError: If strategies is empty or primary_index is out of range.
        """
        if not strategies:
            raise ValueError("CompositeMemory requires at least one strategy")
        if primary_index < 0 or primary_index >= len(strategies):
            raise ValueError(
                f"primary_index {primary_index} out of range for "
                f"{len(strategies)} strategies"
            )
        self._strategies = strategies
        self._primary_index = primary_index

    @property
    def primary(self) -> Memory:
        """The primary memory strategy."""
        return self._strategies[self._primary_index]

    @property
    def strategies(self) -> list[Memory]:
        """All sub-strategies (defensive copy)."""
        return list(self._strategies)

    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Forward message to all strategies.

        If a strategy raises, the error is logged and remaining strategies
        still receive the message.
        """
        for i, strategy in enumerate(self._strategies):
            try:
                await strategy.add_message(content, role, metadata)
            except Exception:
                logger.warning(
                    "Memory strategy %d (%s) failed on add_message",
                    i,
                    type(strategy).__name__,
                    exc_info=True,
                )

    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        """Collect context from all strategies, primary first.

        Results from the primary strategy appear first. Secondary results
        are deduplicated by ``(role, content)`` — if a message with the same
        role and content already appeared, it is not repeated.
        """
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        # Primary first
        try:
            primary_results = await self.primary.get_context(current_message)
            for msg in primary_results:
                key = (msg.get("role", ""), msg.get("content", ""))
                results.append(msg)
                seen.add(key)
        except Exception:
            logger.warning(
                "Primary memory strategy (%s) failed on get_context",
                type(self.primary).__name__,
                exc_info=True,
            )

        # Secondaries — skip primary, dedup by (role, content)
        for i, strategy in enumerate(self._strategies):
            if i == self._primary_index:
                continue
            try:
                secondary_results = await strategy.get_context(current_message)
                for msg in secondary_results:
                    key = (msg.get("role", ""), msg.get("content", ""))
                    if key not in seen:
                        results.append(msg)
                        seen.add(key)
            except Exception:
                logger.warning(
                    "Memory strategy %d (%s) failed on get_context",
                    i,
                    type(strategy).__name__,
                    exc_info=True,
                )

        return results

    async def clear(self) -> None:
        """Clear all strategies. Log and continue on individual failures."""
        for i, strategy in enumerate(self._strategies):
            try:
                await strategy.clear()
            except Exception:
                logger.warning(
                    "Memory strategy %d (%s) failed on clear",
                    i,
                    type(strategy).__name__,
                    exc_info=True,
                )

    async def pop_messages(self, count: int = 2) -> list[dict[str, Any]]:
        """Delegate to primary strategy only.

        Secondary strategies (especially vector) may not support undo.
        If the primary doesn't support it, NotImplementedError propagates.
        """
        return await self.primary.pop_messages(count)

    async def close(self) -> None:
        """Close all strategies that support it."""
        for i, strategy in enumerate(self._strategies):
            try:
                await strategy.close()
            except Exception:
                logger.warning(
                    "Memory strategy %d (%s) failed on close",
                    i,
                    type(strategy).__name__,
                    exc_info=True,
                )

    def providers(self) -> dict[str, Any]:
        """Aggregate providers from all sub-strategies."""
        result: dict[str, Any] = {}
        for strategy in self._strategies:
            result.update(strategy.providers())
        return result

    def set_provider(self, role: str, provider: Any) -> bool:
        """Forward to all sub-strategies; return True if any accepted."""
        accepted = False
        for strategy in self._strategies:
            if strategy.set_provider(role, provider):
                accepted = True
        return accepted

"""Composite memory strategy combining multiple memory implementations."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.exceptions import DataknobsError
from dataknobs_common.structured_config import StructuredConfigConsumer

from .base import Memory
from .config import CompositeMemoryConfig

if TYPE_CHECKING:
    from dataknobs_bots.prompts.resolver import PromptResolver

logger = logging.getLogger(__name__)

# Transient infrastructure failures that warrant graceful degradation
# (log + continue).  Programming errors (AttributeError, TypeError,
# ValueError, KeyError, NotImplementedError, etc.) are NOT caught — they
# should surface during development.
#
# DataknobsError covers the entire dataknobs exception hierarchy
# (ResourceError, OperationError, RateLimitError, etc.) raised by
# real backends.  asyncio.TimeoutError is included separately because
# on Python 3.10 it does not inherit from builtins.TimeoutError.
_STRATEGY_ERRORS = (
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    DataknobsError,
)


class CompositeMemory(StructuredConfigConsumer[CompositeMemoryConfig], Memory):
    """Combines multiple memory strategies into one.

    All sub-strategies receive every ``add_message()`` call independently.
    On ``get_context()``, the primary strategy's results appear first,
    followed by deduplicated results from secondary strategies.

    Graceful degradation: if any strategy fails on a read or write,
    the composite logs a warning and continues with the remaining
    strategies.

    Construct from config (``await CompositeMemory.from_config({...})`` —
    each child spec in ``strategies`` is built recursively through the
    memory factory) or from pre-built children
    (``CompositeMemory.from_components(strategies=[m1, m2],
    primary_index=0)``).

    Attributes:
        primary: The primary memory strategy (results appear first).
        strategies: All sub-strategies in order.
    """

    CONFIG_CLS: ClassVar[type[CompositeMemoryConfig]] = CompositeMemoryConfig

    @classmethod
    async def from_config(  # type: ignore[override]
        cls, config: Any, **components: Any
    ) -> CompositeMemory:
        """Create CompositeMemory from configuration (async warmup).

        Construction is asynchronous — each child strategy is built
        recursively through the memory factory — so ``from_config``
        delegates to :meth:`from_config_async` to run ``_ainit``. The
        injected ``llm_provider`` / ``prompt_resolver`` collaborators are
        threaded into each child build.
        """
        return await cls.from_config_async(config, **components)

    def _setup(self) -> None:
        """Placeholder until strategies are bound.

        The child strategies are bound by :meth:`_ainit` (config-driven
        recursion) or :meth:`_adopt_components` (pre-built injection).
        """
        self._strategies: list[Memory] = []
        self._primary_index = 0

    def _validate_and_bind(
        self, strategies: list[Memory], primary_index: int
    ) -> None:
        """Validate non-empty + in-range, then bind the strategies.

        Raises:
            ValueError: If ``strategies`` is empty or ``primary_index``
                is out of range.
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

    async def _ainit(
        self,
        *,
        llm_provider: Any = None,
        prompt_resolver: PromptResolver | None = None,
        **_: Any,
    ) -> None:
        """Build child strategies from config, recursing through the factory.

        Each child spec is dispatched through the public factory so it
        gets the same error contract and collaborator threading. On any
        failure the already-built strategies are closed before the error
        propagates.
        """
        if self._prebuilt:
            return
        from .registry import create_memory_from_config

        strategies: list[Memory] = []
        try:
            for child in self.config.strategies:
                strategies.append(
                    await create_memory_from_config(
                        child, llm_provider, prompt_resolver=prompt_resolver,
                    )
                )
            if not strategies:
                raise ValueError(
                    "Composite memory requires at least one strategy "
                    "in 'strategies' list"
                )
            self._validate_and_bind(strategies, self.config.primary_index)
        except Exception:
            # Clean up any already-initialized strategies
            for s in strategies:
                try:
                    await s.close()
                except Exception:
                    logger.warning(
                        "Failed to close strategy during cleanup: %s",
                        type(s).__name__,
                        exc_info=True,
                    )
            raise

    def _adopt_components(
        self,
        *,
        strategies: list[Memory] | None = None,
        primary_index: int = 0,
        **_: Any,
    ) -> None:
        """Adopt pre-built child strategies for ``from_components``.

        An empty (or omitted) ``strategies`` list raises ``ValueError``
        via :meth:`_validate_and_bind` — the same contract the legacy
        ``CompositeMemory([])`` ctor enforced.

        The ``primary_index`` arrives as a collaborator kwarg (live
        ``Memory`` strategies cannot be folded into the frozen
        ``strategies`` config field), so the config snapshot is rebuilt to
        record it — otherwise ``config.primary_index`` would misreport the
        default ``0`` while ``self.primary`` reads the real index.
        """
        self._validate_and_bind(strategies or [], primary_index)
        self._config = replace(self._config, primary_index=primary_index)

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
            except _STRATEGY_ERRORS:
                logger.warning(
                    "Memory strategy %d (%s) failed on add_message",
                    i,
                    type(strategy).__name__,
                    exc_info=True,
                )

    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        """Collect context from all strategies, primary first.

        Results from the primary strategy appear first. All results are
        deduplicated by ``(role, content)`` — if a message with the same
        role and content already appeared, it is not repeated.
        """
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        # Primary first
        try:
            primary_results = await self.primary.get_context(current_message)
            for msg in primary_results:
                key = (msg.get("role", ""), msg.get("content", ""))
                if key not in seen:
                    results.append(msg)
                    seen.add(key)
        except _STRATEGY_ERRORS:
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
            except _STRATEGY_ERRORS:
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
            except _STRATEGY_ERRORS:
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
        """Close all strategies. Log and continue on individual failures."""
        for i, strategy in enumerate(self._strategies):
            try:
                await strategy.close()
            except _STRATEGY_ERRORS:
                logger.warning(
                    "Memory strategy %d (%s) failed on close",
                    i,
                    type(strategy).__name__,
                    exc_info=True,
                )

    def providers(self) -> dict[str, Any]:
        """Aggregate providers from all sub-strategies.

        If multiple strategies expose the same role, the last one wins
        and a warning is logged.
        """
        result: dict[str, Any] = {}
        for i, strategy in enumerate(self._strategies):
            for role, provider in strategy.providers().items():
                if role in result:
                    logger.warning(
                        "Provider role %r already registered by an earlier "
                        "strategy; strategy %d (%s) overwrites it",
                        role,
                        i,
                        type(strategy).__name__,
                    )
                result[role] = provider
        return result

    def set_provider(self, role: str, provider: Any) -> bool:
        """Forward to all sub-strategies; return True if any accepted."""
        accepted = False
        for strategy in self._strategies:
            if strategy.set_provider(role, provider):
                accepted = True
        return accepted

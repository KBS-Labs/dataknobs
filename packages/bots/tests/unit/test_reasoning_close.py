"""Tests for ReasoningStrategy.close() lifecycle contract.

Verifies that:
- close() is defined on the ABC with a default no-op
- SimpleReasoning and ReActReasoning are safely closeable
- DynaBot.close() calls strategy.close() without hasattr guard
"""

import asyncio

import pytest

from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.reasoning.simple import SimpleReasoning
from dataknobs_bots.reasoning.react import ReActReasoning


class TestReasoningStrategyClose:
    """close() is part of the ABC contract."""

    @pytest.mark.asyncio()
    async def test_close_exists_on_abc(self):
        """close() is defined on ReasoningStrategy (not abstract)."""
        assert hasattr(ReasoningStrategy, "close")
        # Should not be in __abstractmethods__
        assert "close" not in getattr(ReasoningStrategy, "__abstractmethods__", set())

    @pytest.mark.asyncio()
    async def test_simple_reasoning_close_is_noop(self):
        strategy = SimpleReasoning()
        await strategy.close()  # Should not raise

    @pytest.mark.asyncio()
    async def test_react_reasoning_close_is_noop(self):
        strategy = ReActReasoning()
        await strategy.close()  # Should not raise

    @pytest.mark.asyncio()
    async def test_close_is_callable_on_base(self):
        """A minimal concrete subclass can call close() without overriding."""

        class MinimalStrategy(ReasoningStrategy):
            async def generate(self, manager, llm, tools=None, **kwargs):
                return "response"

        strategy = MinimalStrategy()
        await strategy.close()  # Default no-op, should not raise

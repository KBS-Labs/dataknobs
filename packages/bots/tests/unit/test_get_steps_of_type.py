"""Tests for ``DynaBot.get_steps_of_type`` (161-A).

Typed helper that filters reasoning-strategy pipeline steps by class.
Uses ``BotTestHarness`` to build a real bot, and a minimal pipeline-
shaped fake strategy to exercise the iteration path without requiring
a concrete pipeline strategy to exist in dataknobs-bots today.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.testing import BotTestHarness


_MIN_BOT_CONFIG: dict[str, Any] = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
    "reasoning": {"strategy": "simple"},
}


class _StepA:
    """Marker test-step type A."""


class _StepB:
    """Marker test-step type B."""


class _SubA(_StepA):
    """Subclass of ``_StepA`` for subclass-match coverage."""


class _FakePipelineStrategy(ReasoningStrategy):
    """Minimal pipeline-shaped strategy exposing a public ``steps`` list.

    Subclasses ``ReasoningStrategy`` so it satisfies the bot's typed
    ``reasoning_strategy`` slot. Only implements the abstract
    ``generate`` method (never called in these tests — they exercise
    only ``get_steps_of_type``).
    """

    def __init__(self, steps: list[Any]) -> None:
        super().__init__()
        self.steps = steps

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError


class _NoStepsStrategy(ReasoningStrategy):
    """Strategy that does NOT expose a ``steps`` attribute."""

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError


class TestGetStepsOfType:
    """Cover the seven cases from the impl-plan §6.1."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_strategy(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_MIN_BOT_CONFIG,
            main_responses=["hi"],
        ) as harness:
            harness.bot.reasoning_strategy = None
            assert harness.bot.get_steps_of_type(_StepA) == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_strategy_has_no_steps_attr(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_MIN_BOT_CONFIG,
            main_responses=["hi"],
        ) as harness:
            harness.bot.reasoning_strategy = _NoStepsStrategy()
            assert harness.bot.get_steps_of_type(_StepA) == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_steps(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_MIN_BOT_CONFIG,
            main_responses=["hi"],
        ) as harness:
            harness.bot.reasoning_strategy = _FakePipelineStrategy(steps=[])
            assert harness.bot.get_steps_of_type(_StepA) == []

    @pytest.mark.asyncio
    async def test_exact_class_filter(self) -> None:
        a1, b, a2 = _StepA(), _StepB(), _StepA()
        async with await BotTestHarness.create(
            bot_config=_MIN_BOT_CONFIG,
            main_responses=["hi"],
        ) as harness:
            harness.bot.reasoning_strategy = _FakePipelineStrategy(
                steps=[a1, b, a2],
            )
            result = harness.bot.get_steps_of_type(_StepA)
            assert result == [a1, a2]

    @pytest.mark.asyncio
    async def test_subclass_matches_base(self) -> None:
        base, sub = _StepA(), _SubA()
        async with await BotTestHarness.create(
            bot_config=_MIN_BOT_CONFIG,
            main_responses=["hi"],
        ) as harness:
            harness.bot.reasoning_strategy = _FakePipelineStrategy(
                steps=[base, sub],
            )
            # Querying for the base returns both; querying for the
            # subclass returns only the subclass instance.
            assert harness.bot.get_steps_of_type(_StepA) == [base, sub]
            assert harness.bot.get_steps_of_type(_SubA) == [sub]

    @pytest.mark.asyncio
    async def test_preserves_insertion_order(self) -> None:
        ordered = [_StepA(), _StepA(), _StepA(), _StepA()]
        async with await BotTestHarness.create(
            bot_config=_MIN_BOT_CONFIG,
            main_responses=["hi"],
        ) as harness:
            harness.bot.reasoning_strategy = _FakePipelineStrategy(
                steps=ordered,
            )
            assert harness.bot.get_steps_of_type(_StepA) == ordered

    @pytest.mark.asyncio
    async def test_return_is_list_not_generator(self) -> None:
        async with await BotTestHarness.create(
            bot_config=_MIN_BOT_CONFIG,
            main_responses=["hi"],
        ) as harness:
            harness.bot.reasoning_strategy = _FakePipelineStrategy(
                steps=[_StepA(), _StepA()],
            )
            result = harness.bot.get_steps_of_type(_StepA)
            assert isinstance(result, list)
            # Iterable twice (a generator would be exhausted after one pass).
            assert list(result) == list(result)

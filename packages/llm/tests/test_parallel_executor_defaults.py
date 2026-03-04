"""Tests for ParallelLLMExecutor.default_config_overrides.

Verifies that:
- No default + no task overrides → provider receives None
- Default overrides alone → provider receives the defaults
- Task overrides alone → provider receives the task overrides
- Both set → merged dict where task-level keys win
- Defaults apply across all tasks in a batch
"""

from __future__ import annotations

import pytest

from dataknobs_llm import (
    EchoProvider,
    LLMMessage,
    LLMTask,
    ParallelLLMExecutor,
)
from dataknobs_llm.testing import text_response


@pytest.fixture()
def provider() -> EchoProvider:
    return EchoProvider({"provider": "echo", "model": "test"})


def _msg(content: str = "hi") -> list[LLMMessage]:
    return [LLMMessage(role="user", content=content)]


@pytest.mark.asyncio
async def test_no_overrides(provider: EchoProvider) -> None:
    """Neither executor nor task set overrides → None passed to provider."""
    provider.set_responses([text_response("ok")])
    executor = ParallelLLMExecutor(provider, max_concurrency=1)

    results = await executor.execute({"t": LLMTask(messages=_msg())})

    assert results["t"].success
    assert provider.get_last_call()["config_overrides"] is None


@pytest.mark.asyncio
async def test_default_overrides_only(provider: EchoProvider) -> None:
    """Executor defaults applied when task has no overrides."""
    provider.set_responses([text_response("ok")])
    executor = ParallelLLMExecutor(
        provider,
        max_concurrency=1,
        default_config_overrides={"temperature": 0.5, "model": "fast"},
    )

    results = await executor.execute({"t": LLMTask(messages=_msg())})

    assert results["t"].success
    overrides = provider.get_last_call()["config_overrides"]
    assert overrides == {"temperature": 0.5, "model": "fast"}


@pytest.mark.asyncio
async def test_task_overrides_only(provider: EchoProvider) -> None:
    """Task overrides used when no executor defaults are set."""
    provider.set_responses([text_response("ok")])
    executor = ParallelLLMExecutor(provider, max_concurrency=1)

    results = await executor.execute({
        "t": LLMTask(messages=_msg(), config_overrides={"temperature": 0.9}),
    })

    assert results["t"].success
    overrides = provider.get_last_call()["config_overrides"]
    assert overrides == {"temperature": 0.9}


@pytest.mark.asyncio
async def test_task_overrides_win(provider: EchoProvider) -> None:
    """Task-level keys take precedence over executor defaults."""
    provider.set_responses([text_response("ok")])
    executor = ParallelLLMExecutor(
        provider,
        max_concurrency=1,
        default_config_overrides={"temperature": 0.5, "model": "fast"},
    )

    results = await executor.execute({
        "t": LLMTask(
            messages=_msg(),
            config_overrides={"temperature": 0.9},
        ),
    })

    assert results["t"].success
    overrides = provider.get_last_call()["config_overrides"]
    assert overrides == {"temperature": 0.9, "model": "fast"}


@pytest.mark.asyncio
async def test_defaults_applied_to_all_tasks(provider: EchoProvider) -> None:
    """Default overrides are applied to every task in a batch."""
    provider.set_responses([
        text_response("r1"),
        text_response("r2"),
        text_response("r3"),
    ])
    executor = ParallelLLMExecutor(
        provider,
        max_concurrency=5,
        default_config_overrides={"model": "fast"},
    )

    results = await executor.execute({
        "a": LLMTask(messages=_msg("q1")),
        "b": LLMTask(messages=_msg("q2")),
        "c": LLMTask(messages=_msg("q3")),
    })

    assert all(r.success for r in results.values())
    for call in provider.calls:
        assert call["config_overrides"] == {"model": "fast"}

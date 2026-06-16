"""Behavior tests for :class:`CallbackRegistry`."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from dataknobs_common.callbacks import (
    BatchedCallbackError,
    CallbackRegistry,
    CapturingCallbackRegistry,
    ErrorPolicy,
    PriorityOrdering,
    RecordingCallbackRegistry,
    StageOrdering,
)


def _raiser(message: str):
    def cb(_: dict[str, Any]) -> None:
        raise RuntimeError(message)

    return cb


def test_register_then_fire_runs_callback() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    seen: list[dict] = []
    registry.register("t", lambda payload: seen.append(payload))
    registry.fire("t", {"k": 1})
    assert seen == [{"k": 1}]


def test_fire_no_callbacks_is_noop() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    registry.fire("absent_topic", {})  # does not raise


def test_unregister_removes_callback() -> None:
    registry: CallbackRegistry = CallbackRegistry()

    def cb(_: dict[str, Any]) -> None: ...

    registry.register("t", cb)
    assert registry.unregister("t", cb) is True
    assert registry.unregister("t", cb) is False


def test_unregister_unknown_callback_returns_false() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    registry.register("t", lambda _: None)
    # Different lambda object — identity miss
    assert registry.unregister("t", lambda _: None) is False


def test_unregister_unknown_topic_returns_false() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    assert registry.unregister("absent", lambda _: None) is False


def test_fire_preserves_fifo_under_default_ordering() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    seen: list[int] = []
    for i in range(5):
        registry.register("t", lambda _, i=i: seen.append(i))
    registry.fire("t", {})
    assert seen == [0, 1, 2, 3, 4]


def test_priority_ordering_fires_lower_first() -> None:
    registry: CallbackRegistry = CallbackRegistry(
        ordering=PriorityOrdering(),
    )
    seen: list[str] = []
    registry.register("t", lambda _: seen.append("default"))
    registry.register("t", lambda _: seen.append("low"), priority=-10)
    registry.register("t", lambda _: seen.append("high"), priority=10)
    registry.fire("t", {})
    assert seen == ["low", "default", "high"]


def test_stage_ordering_fires_in_stage_sequence() -> None:
    registry: CallbackRegistry = CallbackRegistry(
        ordering=StageOrdering(stages=("pre", "main", "post")),
    )
    seen: list[str] = []
    registry.register("t", lambda _: seen.append("main"), stage="main")
    registry.register("t", lambda _: seen.append("post"), stage="post")
    registry.register("t", lambda _: seen.append("pre"), stage="pre")
    registry.fire("t", {})
    assert seen == ["pre", "main", "post"]


def test_unknown_stage_sorts_to_end() -> None:
    registry: CallbackRegistry = CallbackRegistry(
        ordering=StageOrdering(stages=("a", "b")),
    )
    seen: list[str] = []
    registry.register("t", lambda _: seen.append("a"), stage="a")
    registry.register("t", lambda _: seen.append("z"), stage="z")
    registry.register("t", lambda _: seen.append("b"), stage="b")
    registry.fire("t", {})
    assert seen == ["a", "b", "z"]


def test_set_ordering_affects_subsequent_fires() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    seen: list[int] = []
    registry.register("t", lambda _: seen.append(1))
    registry.register("t", lambda _: seen.append(2), priority=-10)
    registry.fire("t", {})  # FIFO: [1, 2]
    registry.set_ordering(PriorityOrdering())
    registry.fire("t", {})  # Priority: [2, 1]
    assert seen == [1, 2, 2, 1]


def test_error_policy_raise_aborts_remaining_callbacks() -> None:
    registry: CallbackRegistry = CallbackRegistry(
        error_policy=ErrorPolicy.RAISE,
    )
    seen: list[int] = []
    registry.register("t", _raiser("first"))
    registry.register("t", lambda _: seen.append(2))
    with pytest.raises(RuntimeError, match="first"):
        registry.fire("t", {})
    assert seen == []  # callback after the raiser did not run


def test_error_policy_log_and_continue_runs_remaining(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry: CallbackRegistry = CallbackRegistry()  # default policy
    seen: list[int] = []
    registry.register("t", _raiser("x"))
    registry.register("t", lambda _: seen.append(2))
    with caplog.at_level(logging.ERROR, logger="dataknobs_common.callbacks"):
        registry.fire("t", {})
    assert seen == [2]
    assert any("continuing dispatch" in r.message for r in caplog.records)


def test_error_policy_log_and_raise_at_end_aggregates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry: CallbackRegistry = CallbackRegistry(
        error_policy=ErrorPolicy.LOG_AND_RAISE_AT_END,
    )
    seen: list[int] = []
    registry.register("t", _raiser("a"))
    registry.register("t", lambda _: seen.append(1))
    registry.register("t", _raiser("b"))
    with (
        caplog.at_level(logging.ERROR, logger="dataknobs_common.callbacks"),
        pytest.raises(BatchedCallbackError) as exc_info,
    ):
        registry.fire("t", {})
    assert len(exc_info.value.failures) == 2
    assert seen == [1]  # non-failing callback still ran
    failure_topics = {entry.topic for entry, _ in exc_info.value.failures}
    assert failure_topics == {"t"}


@pytest.mark.asyncio
async def test_fire_async_awaits_coroutine_callbacks() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    seen: list[str] = []

    async def async_cb(_: dict) -> None:
        await asyncio.sleep(0)
        seen.append("async")

    def sync_cb(_: dict) -> None:
        seen.append("sync")

    registry.register("t", async_cb)
    registry.register("t", sync_cb)
    await registry.fire_async("t", {})
    assert seen == ["async", "sync"]


@pytest.mark.asyncio
async def test_fire_async_error_policy_continues_after_failure() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    seen: list[str] = []

    async def boom(_: dict) -> None:
        raise RuntimeError("boom")

    def trailing(_: dict) -> None:
        seen.append("trailing")

    registry.register("t", boom)
    registry.register("t", trailing)
    await registry.fire_async("t", {})
    assert seen == ["trailing"]


def test_fire_sync_with_async_callback_raises() -> None:
    registry: CallbackRegistry = CallbackRegistry()

    async def async_cb(_: dict) -> None:
        return None

    registry.register("t", async_cb)
    with pytest.raises(TypeError, match="async callback"):
        registry.fire("t", {})


def test_topics_returns_registered_topics() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    registry.register("a", lambda _: None)
    registry.register("b", lambda _: None)
    assert set(registry.topics()) == {"a", "b"}


def test_topics_omits_drained_topic() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    registry.register("a", lambda _: None)
    registry.clear("a")
    assert set(registry.topics()) == set()


def test_callback_count() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    assert registry.callback_count("absent") == 0
    for _ in range(3):
        registry.register("t", lambda _: None)
    assert registry.callback_count("t") == 3


def test_clear_drains_in_place_preserving_ordering_and_policy() -> None:
    registry: CallbackRegistry = CallbackRegistry(
        ordering=PriorityOrdering(),
        error_policy=ErrorPolicy.RAISE,
    )
    registry.register("t", lambda _: None, priority=5)
    registry.clear()
    assert registry.callback_count("t") == 0

    # Re-registering after clear honors the originally-configured
    # ordering and error policy (the registry instance is preserved).
    seen: list[str] = []
    registry.register("t", lambda _: seen.append("default"))
    registry.register("t", lambda _: seen.append("low"), priority=-1)
    registry.fire("t", {})
    assert seen == ["low", "default"]  # PriorityOrdering still in effect

    registry.clear()
    registry.register("t", _raiser("x"))
    registry.register("t", lambda _: seen.append("after"))
    with pytest.raises(RuntimeError, match="x"):
        registry.fire("t", {})
    assert seen == ["low", "default"]  # RAISE policy preserved across clear


def test_clear_specific_topic_leaves_other_topics() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    registry.register("a", lambda _: None)
    registry.register("b", lambda _: None)
    registry.clear("a")
    assert registry.callback_count("a") == 0
    assert registry.callback_count("b") == 1


def test_clear_unknown_topic_is_noop() -> None:
    registry: CallbackRegistry = CallbackRegistry()
    registry.clear("absent")  # does not raise


def test_capturing_registry_records_dispatched_payloads() -> None:
    registry: CapturingCallbackRegistry = CapturingCallbackRegistry()
    seen: list[dict] = []
    registry.register("t", lambda p: seen.append(p))
    registry.fire("t", {"k": 1})
    assert registry.captured == [("t", {"k": 1})]
    assert seen == [{"k": 1}]  # underlying dispatch still ran


@pytest.mark.asyncio
async def test_capturing_registry_records_async_dispatched_payloads() -> None:
    registry: CapturingCallbackRegistry = CapturingCallbackRegistry()

    async def cb(_: dict) -> None:
        return None

    registry.register("t", cb)
    await registry.fire_async("t", {"k": 2})
    assert registry.captured == [("t", {"k": 2})]


def test_recording_registry_does_not_dispatch_to_callbacks() -> None:
    registry = RecordingCallbackRegistry()
    seen: list[dict] = []
    registry.register("t", lambda p: seen.append(p))
    registry.fire("t", {"k": 1})
    assert registry.captured == [("t", {"k": 1})]
    assert seen == []  # registered callback did NOT run


@pytest.mark.asyncio
async def test_recording_registry_async_fire_records_without_dispatch() -> None:
    registry = RecordingCallbackRegistry()
    seen: list[dict] = []

    async def cb(p: dict) -> None:
        seen.append(p)

    registry.register("t", cb)
    await registry.fire_async("t", {"k": 1})
    assert registry.captured == [("t", {"k": 1})]
    assert seen == []


def test_recording_registry_unregister_identity_match() -> None:
    registry = RecordingCallbackRegistry()

    def cb(_: dict) -> None: ...

    registry.register("t", cb)
    assert registry.unregister("t", cb) is True
    assert registry.unregister("t", cb) is False

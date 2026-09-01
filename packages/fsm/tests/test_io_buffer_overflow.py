"""``IOBuffer`` overflow: what it drains, and what happens when the drain fails.

The overflow handler is the only thing standing between a full buffer and
losing its contents --- the items come off the buffer before the handler is
called, and no copy is kept. Three properties follow from that, and none of
them held.

**A handler that raises loses the items.** They are already off the buffer, so
the exception propagates out of ``add`` with the overflow gone and nothing
holding it. The docstring cited the no-copy design as the *reason* for
dispatching carefully, while leaving the window it describes wide open.

**A small buffer never drains at all.** The drain is ``max_size // 2``, which
is ``0`` for ``max_size=1``. The handler is then invoked with an empty list on
every subsequent ``add``, the buffer keeps every item, and it grows without
bound past the maximum it was configured with --- a memory leak in the one
component whose whole job is bounding memory.

**An async handler was refused by the annotation.** The dispatch accepts one;
the published signature said ``Callable[[List[Any]], None]``, so a consumer
writing the async handler the code supports got a type error at their own call
site.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_fsm.io.utils import IOBuffer


class RecordingHandler:
    """A synchronous overflow handler that keeps what it was given."""

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []

    def __call__(self, items: list[Any]) -> None:
        self.batches.append(list(items))


class RecordingAsyncHandler:
    """The async callable-object shape a consumer writes for a real flush."""

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []

    async def __call__(self, items: list[Any]) -> None:
        self.batches.append(list(items))


class FailingHandler:
    """A flush that fails the way a real one does --- disk full, socket refused."""

    def __init__(self) -> None:
        self.attempts: list[list[Any]] = []

    def __call__(self, items: list[Any]) -> None:
        self.attempts.append(list(items))
        raise OSError("no space left on device")


@pytest.mark.asyncio
async def test_a_failing_handler_does_not_swallow_the_overflow() -> None:
    """The items are the caller's until something has accepted them.

    Before the fix they were sliced off the buffer, handed to a handler that
    raised, and gone --- the exception reached the caller with no way to
    recover what it had been holding.
    """
    handler = FailingHandler()
    buffer = IOBuffer(max_size=4, overflow_handler=handler)

    with pytest.raises(OSError, match="no space left"):
        for item in range(4):
            await buffer.add(item)

    assert handler.attempts, "the handler was never called"
    assert sorted(buffer.buffer) == [0, 1, 2, 3], (
        "a failed flush must leave the overflow in the buffer; the items are "
        "off it before the handler runs and no copy is kept anywhere else"
    )


@pytest.mark.asyncio
async def test_a_failing_handler_leaves_the_items_in_order() -> None:
    """Restored to the front, so a retry drains the oldest first."""
    handler = FailingHandler()
    buffer = IOBuffer(max_size=4, overflow_handler=handler)

    with pytest.raises(OSError):
        for item in ["a", "b", "c", "d"]:
            await buffer.add(item)

    assert buffer.buffer == ["a", "b", "c", "d"]


@pytest.mark.parametrize("max_size", [1, 2, 3])
@pytest.mark.asyncio
async def test_a_small_buffer_still_drains(max_size: int) -> None:
    """``max_size // 2`` is zero at one, and the buffer then grows for ever.

    The handler was called with an empty list on every add while the buffer
    kept everything --- unbounded growth in the component whose contract is
    to bound it.
    """
    handler = RecordingHandler()
    buffer = IOBuffer(max_size=max_size, overflow_handler=handler)

    for item in range(10):
        await buffer.add(item)

    assert handler.batches, "the handler never received anything"
    assert all(batch for batch in handler.batches), (
        "the handler was called with an empty list, which is the drain "
        "computing zero rather than the buffer being empty"
    )
    assert len(buffer.buffer) <= max_size, (
        f"buffer holds {len(buffer.buffer)} items with max_size={max_size}"
    )

    drained = [item for batch in handler.batches for item in batch]
    assert sorted(drained + buffer.buffer) == list(range(10)), "an item was lost or duplicated"


@pytest.mark.asyncio
async def test_an_async_callable_object_handler_receives_the_overflow() -> None:
    """The shape ``iscoroutinefunction`` misread, kept working."""
    handler = RecordingAsyncHandler()
    buffer = IOBuffer(max_size=4, overflow_handler=handler)

    for item in range(4):
        await buffer.add(item)

    drained = [item for batch in handler.batches for item in batch]
    assert drained, "the async handler never received the overflow"
    assert sorted(drained + buffer.buffer) == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_a_buffer_with_no_handler_keeps_its_items() -> None:
    """No handler means no drain --- and no silent loss either."""
    buffer = IOBuffer(max_size=2)

    for item in range(5):
        await buffer.add(item)

    assert buffer.buffer == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_flush_returns_and_clears() -> None:
    buffer = IOBuffer(max_size=100)
    for item in range(3):
        await buffer.add(item)

    assert await buffer.flush() == [0, 1, 2]
    assert buffer.buffer == []

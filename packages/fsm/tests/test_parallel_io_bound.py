"""``ParallelIOExecutor`` honours the concurrency bound it takes.

``max_workers`` was stored by ``__init__``, documented by the factory, and
read nowhere. That was inert while synchronous providers were skipped
entirely: there was no fan-out to bound.

It stopped being inert when synchronous providers started being offloaded.
Every one of them now becomes an ``asyncio.to_thread`` submission, issued at
once, into the event loop's **default** executor --- which is shared by the
whole process and sized ``min(32, cpu_count + 4)``. A consumer building
``parallel_io_executor(providers, max_workers=4)`` over a few hundred slow
providers therefore saturates the pool that every other offload in the
application is also using, including the ones this package's own buffer and
router depend on. The knob that was supposed to prevent exactly that was not
connected to anything.

These tests pin the bound rather than the pool. What matters to a caller is
that no more than ``max_workers`` of its providers are in flight at once; how
that is achieved is the class's business.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from dataknobs_fsm.io.utils import ParallelIOExecutor, parallel_io_executor


class ConcurrencyWitness:
    """Shared tally of how many provider calls overlap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)

    def leave(self) -> None:
        with self._lock:
            self.in_flight -= 1


class SlowSyncProvider:
    """A synchronous provider whose read and write take real wall-clock.

    Synchronous on purpose: the offload is what creates the fan-out, so a
    provider that is already async would not exercise the bound this test is
    about.
    """

    def __init__(self, witness: ConcurrencyWitness, duration: float = 0.05) -> None:
        self._witness = witness
        self._duration = duration
        self.writes: list[Any] = []

    def read(self, **kwargs: Any) -> str:
        self._witness.enter()
        try:
            threading.Event().wait(self._duration)
            return "read"
        finally:
            self._witness.leave()

    def write(self, data: Any, **kwargs: Any) -> None:
        self._witness.enter()
        try:
            threading.Event().wait(self._duration)
            self.writes.append(data)
        finally:
            self._witness.leave()


@pytest.mark.parametrize("max_workers", [1, 2, 3])
@pytest.mark.asyncio
async def test_read_all_never_exceeds_max_workers(max_workers: int) -> None:
    """The defect: every provider was submitted at once, whatever the bound."""
    witness = ConcurrencyWitness()
    providers = [SlowSyncProvider(witness) for _ in range(9)]
    executor = ParallelIOExecutor(providers, max_workers=max_workers)

    results = await executor.read_all()

    assert len(results) == 9, "every provider must still be read"
    assert witness.peak <= max_workers, (
        f"{witness.peak} providers ran at once with max_workers={max_workers}"
    )


@pytest.mark.asyncio
async def test_write_all_never_exceeds_max_workers() -> None:
    """The same bound on the write path, which fans out identically."""
    witness = ConcurrencyWitness()
    providers = [SlowSyncProvider(witness) for _ in range(9)]
    executor = ParallelIOExecutor(providers, max_workers=2)

    await executor.write_all({"payload": True})

    assert witness.peak <= 2, f"{witness.peak} providers wrote at once with max_workers=2"
    assert all(provider.writes == [{"payload": True}] for provider in providers)


@pytest.mark.asyncio
async def test_the_bound_still_allows_real_concurrency() -> None:
    """A bound is not a serialisation --- the point is parallel I/O.

    Without this the previous two tests would pass against an implementation
    that ran everything one at a time, which would be a worse bug than the
    one they were written for.
    """
    witness = ConcurrencyWitness()
    providers = [SlowSyncProvider(witness) for _ in range(6)]
    executor = ParallelIOExecutor(providers, max_workers=3)

    await executor.read_all()

    assert witness.peak > 1, "providers were serialised rather than run in parallel"


@pytest.mark.asyncio
async def test_the_factory_passes_the_bound_through() -> None:
    """``parallel_io_executor`` documents ``max_workers``; it must connect."""
    witness = ConcurrencyWitness()
    providers = [SlowSyncProvider(witness) for _ in range(6)]

    await parallel_io_executor(providers, max_workers=2).read_all()

    assert witness.peak <= 2


@pytest.mark.asyncio
async def test_a_provider_with_neither_method_is_still_skipped() -> None:
    """The existing gate survives the bound being wired in."""

    class Inert:
        pass

    witness = ConcurrencyWitness()
    providers: list[Any] = [Inert(), SlowSyncProvider(witness), Inert()]

    results = await ParallelIOExecutor(providers, max_workers=2).read_all()

    assert results == ["read"]


@pytest.mark.asyncio
async def test_a_nonsensical_bound_is_refused() -> None:
    """Zero workers would deadlock; a negative one is a caller's typo."""
    for bound in (0, -1):
        with pytest.raises(ValueError, match="max_workers"):
            ParallelIOExecutor([], max_workers=bound)


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_bound() -> None:
    """Two overlapping ``read_all`` calls on one executor stay inside it.

    A semaphore rebuilt per call would pass every test above and still let
    the fan-out double whenever a consumer drives the executor from two
    tasks, which is the ordinary way to use it.
    """
    witness = ConcurrencyWitness()
    providers = [SlowSyncProvider(witness) for _ in range(4)]
    executor = ParallelIOExecutor(providers, max_workers=2)

    await asyncio.gather(executor.read_all(), executor.read_all())

    assert witness.peak <= 2, f"{witness.peak} ran at once across two concurrent callers"

"""Acquisition behaviour of :class:`ResourcePool`.

A pool exists to hand out a resource promptly. These tests pin *when* it is
allowed to make the caller wait: only when every resource it is permitted to
create is already out on loan. Waiting in any other situation is a stall, not
back-pressure — and because the wait is bounded by ``acquire_timeout``
(default 30 seconds) rather than by anything the caller did wrong, it reads
as a hang.

Timing is the symptom under test, so these assert on elapsed time. The
margins are wide (a configured timeout of seconds against an assertion of
well under one) so that only the difference between "waited for the timeout"
and "did not wait" can trip them, not scheduling noise.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import BaseResourceProvider
from dataknobs_fsm.resources.pool import PoolConfig, ResourcePool
from dataknobs_fsm.resources.properties import PropertiesResource

# Long enough that waiting for it is unmistakable in the elapsed time, short
# enough that a regression fails the suite in seconds rather than stalling it.
_TIMEOUT = 3.0

# What "did not wait" has to beat. Two orders of magnitude below the timeout,
# so no plausible amount of scheduling jitter reaches it.
_PROMPT = 0.5


@pytest.fixture
def provider() -> PropertiesResource:
    """A real resource provider — no fake needed for a pool of dictionaries."""
    return PropertiesResource("props", initial_properties={"k": "v"})


def test_empty_pool_under_capacity_creates_instead_of_waiting(
    provider: PropertiesResource,
) -> None:
    """An empty pool that may still grow must not wait for a release.

    ``min_size=0`` leaves the queue empty, and no resource is out on loan, so
    the pool is free to create one immediately. Blocking on the queue first
    means blocking for the entire ``acquire_timeout`` before reaching the
    branch that creates it — a resource the pool could have produced at once,
    delivered 3 seconds late.
    """
    pool = ResourcePool(
        provider, PoolConfig(min_size=0, max_size=2, acquire_timeout=_TIMEOUT)
    )
    try:
        started = time.monotonic()
        resource = pool.acquire()
        elapsed = time.monotonic() - started

        assert resource is not None
        assert elapsed < _PROMPT, (
            f"acquire() from an empty pool with spare capacity took {elapsed:.2f}s; "
            f"it waited out the {_TIMEOUT}s acquire timeout before creating a "
            "resource it was always allowed to create"
        )
    finally:
        pool.close()


def test_pool_drained_below_capacity_still_creates_promptly(
    provider: PropertiesResource,
) -> None:
    """The same, reached by handing out the pre-created resources.

    Distinct from the ``min_size=0`` case: here the pool *did* start with
    resources and has simply lent them all out while remaining under
    ``max_size``. It is the shape a real workload hits — a burst that drains
    the initial resources — and it must not stall either.
    """
    pool = ResourcePool(
        provider, PoolConfig(min_size=1, max_size=3, acquire_timeout=_TIMEOUT)
    )
    try:
        first = pool.acquire()  # drains the one pre-created resource

        started = time.monotonic()
        second = pool.acquire()
        elapsed = time.monotonic() - started

        assert second is not first
        assert elapsed < _PROMPT, (
            f"acquire() on a drained but under-capacity pool took {elapsed:.2f}s"
        )
    finally:
        pool.close()


def test_acquire_at_capacity_waits_and_then_times_out(
    provider: PropertiesResource,
) -> None:
    """At capacity, waiting is correct — the fix must not turn it into a fail.

    Creating past ``max_size`` is exactly what the limit forbids, so the only
    resource that can satisfy this caller is one another holder gives back.
    Waiting for it is what ``acquire_timeout`` is for, and the guard against
    over-correcting into "never wait" is that this must still wait, and still
    raise when nothing is returned.
    """
    pool = ResourcePool(provider, PoolConfig(min_size=1, max_size=1))
    try:
        pool.acquire()  # the pool's one permitted resource, now out on loan

        started = time.monotonic()
        with pytest.raises(ResourceError, match="Failed to acquire resource"):
            pool.acquire(timeout=0.5)
        elapsed = time.monotonic() - started

        assert elapsed >= 0.4, (
            f"acquire() at capacity gave up after {elapsed:.2f}s without waiting "
            "out its timeout"
        )
    finally:
        pool.close()


def test_release_during_the_wait_satisfies_a_waiting_acquire(
    provider: PropertiesResource,
) -> None:
    """A blocked caller is woken by a release, not left until the timeout."""
    pool = ResourcePool(
        provider, PoolConfig(min_size=1, max_size=1, acquire_timeout=_TIMEOUT)
    )
    try:
        held = pool.acquire()

        def _release_shortly() -> None:
            time.sleep(0.2)
            pool.release(held)

        releaser = threading.Thread(target=_release_shortly)
        releaser.start()
        try:
            started = time.monotonic()
            resource = pool.acquire()
            elapsed = time.monotonic() - started
        finally:
            releaser.join()

        assert resource is not None
        assert elapsed < _PROMPT, (
            f"acquire() blocked at capacity took {elapsed:.2f}s to notice a release"
        )
    finally:
        pool.close()


def test_zero_timeout_does_not_fall_back_to_the_configured_default(
    provider: PropertiesResource,
) -> None:
    """``timeout=0`` means do not wait — it is a value, not an absent argument.

    Read as a truthiness test, ``0`` is indistinguishable from ``None`` and
    silently becomes the configured default. A caller asking for a resource
    only if one is free right now would instead block for the full
    ``acquire_timeout``, which is the opposite of what they asked for.
    """
    pool = ResourcePool(
        provider, PoolConfig(min_size=1, max_size=1, acquire_timeout=_TIMEOUT)
    )
    try:
        pool.acquire()  # at capacity, so a zero-wait acquire cannot succeed

        started = time.monotonic()
        with pytest.raises(ResourceError, match="Failed to acquire resource"):
            pool.acquire(timeout=0)
        elapsed = time.monotonic() - started

        assert elapsed < _PROMPT, (
            f"acquire(timeout=0) waited {elapsed:.2f}s; a zero timeout was treated "
            f"as unset and replaced with the {_TIMEOUT}s configured default"
        )
    finally:
        pool.close()


def test_capacity_freed_while_waiting_is_used_rather_than_timing_out(
    provider: PropertiesResource,
) -> None:
    """A resource retired on release frees capacity without waking the waiter.

    ``release()`` retires a resource past ``max_lifetime`` instead of
    returning it to the queue, so a caller blocked on that queue is never
    notified — even though the pool may now create one. Only re-checking
    capacity after the wait expires turns that silent timeout into a
    resource.

    Unlike the two stall tests above, this pins behaviour the pool already
    had: the original ordering re-checked capacity because the wait came
    first. Moving the check ahead of the wait must not lose the check *after*
    it.
    """
    pool = ResourcePool(
        provider,
        PoolConfig(
            min_size=1, max_size=1, max_lifetime=0.05, acquire_timeout=_TIMEOUT
        ),
    )
    try:
        held = pool.acquire()

        def _retire_shortly() -> None:
            # By now the resource is past the 0.05s max_lifetime, so
            # release() retires it: capacity drops, but nothing is queued.
            time.sleep(0.2)
            pool.release(held)

        retirer = threading.Thread(target=_retire_shortly)
        retirer.start()
        try:
            resource = pool.acquire(timeout=1.0)
        finally:
            retirer.join()

        assert resource is not None
    finally:
        pool.close()


class _NullResourceProvider(BaseResourceProvider):
    """A provider whose resource *is* ``None``.

    Not a stand-in for a real provider — the return value is the whole point.
    ``IResourceProvider.acquire`` is typed ``-> Any``, so ``None`` is a value
    a consumer's provider may hand back, and it has to stay distinguishable
    from the pool's own internal "no capacity" signal.
    """

    def acquire(self, **kwargs: Any) -> Any:
        return None

    def release(self, resource: Any) -> None:
        pass


def test_a_none_resource_is_delivered_rather_than_read_as_no_capacity() -> None:
    """``None`` from the provider must not be read as "the pool is full".

    The pool books a resource into its active set before the caller ever sees
    it, so conflating the two loses it twice: the caller waits the whole
    ``acquire_timeout`` for a resource that had already been created, and is
    then handed a timeout for a pool that was never at capacity.
    """
    pool = ResourcePool(
        _NullResourceProvider("null"),
        PoolConfig(min_size=0, max_size=2, acquire_timeout=_TIMEOUT),
    )
    try:
        started = time.monotonic()
        resource = pool.acquire()
        elapsed = time.monotonic() - started

        assert resource is None
        assert elapsed < _PROMPT
    finally:
        pool.close()

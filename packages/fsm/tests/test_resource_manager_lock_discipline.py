"""``cleanup()`` must agree with ``close()`` about the registry they share.

``ResourceManager`` holds a :class:`threading.RLock` and its synchronous
teardown respects it: ``close()`` iterates ``_providers`` *under* the lock and
clears it in the same critical section. The asynchronous ``cleanup()`` does
neither --- it reads the same dict unlocked, and it awaits three times while
holding nothing. The two halves therefore disagree about the same data
structure, and the disagreement has two distinct costs:

* **A provider registered while teardown runs is silently dropped.** It is not
  in the sweep, so it is never closed; it is not a failure, so it is never
  recorded in ``unclosed_providers``; and the final ``_providers.clear()``
  removes it. The transport stays open with nothing pointing at it. This is
  reproducible on one thread --- the await windows are wide open --- and is
  fixed by refusing the registration, not by locking the sweep.
* **A registry mutated during the classification sweep can abort it.** Only
  another thread can do this now that registration is refused, so the test
  below shortens the interpreter's switch interval to make the interleave
  reliable rather than hoping for it.

The two fixes are independent and neither subsumes the other, which is why
each has its own test: the snapshot alone leaves the leak, and the guard alone
leaves the sweep unlocked against ``unregister_provider``.

Real providers throughout --- the subject is what teardown does to a provider,
so a stand-in for one would be testing the stand-in.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import BaseResourceProvider
from dataknobs_fsm.resources.manager import ResourceManager


class _Provider(BaseResourceProvider):
    """A provider whose synchronous teardown is observable."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.closed = False

    def acquire(self, **kwargs: Any) -> Any:
        return {"name": self.name}

    def release(self, resource: Any) -> None:
        return None

    def close(self) -> None:
        super().close()
        self.closed = True


class _GatedProvider(_Provider):
    """A provider whose awaited teardown holds ``cleanup`` open on request.

    No ``close``: the awaited path is the one under test, and inheriting a
    synchronous teardown would let the sweep finish without ever suspending,
    so the window the tests below aim at would not exist.

    Two events rather than a sleep: that window is the subject, so it is
    opened and closed by handoff and cannot close early on a loaded machine.
    """

    close = None  # type: ignore[assignment]

    def __init__(self, name: str, entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__(name)
        self.aclosed = False
        self._entered = entered
        self._release = release

    async def aclose(self) -> None:
        self._entered.set()
        await self._release.wait()
        self.aclosed = True


# --------------------------------------------------------------------------- #
# Registration during teardown
# --------------------------------------------------------------------------- #


async def test_a_provider_registered_while_cleanup_runs_is_not_dropped() -> None:
    """The leak, stated as the caller can observe it.

    The registration is sequenced *into* the awaited-teardown window rather
    than raced against it: the first provider announces that its ``aclose``
    has begun and holds there until the registrar has run. Landing in that
    window is the whole claim, and a ``sleep`` long enough to usually land
    there would be a claim about this machine instead.

    Every outcome is acceptable except the one that happens today: the
    provider vanishes from the registry having never been closed and never
    been reported, so nothing anywhere names the transport it left open.
    """
    manager = ResourceManager()
    teardown_began = asyncio.Event()
    registrar_done = asyncio.Event()
    manager.register_provider("first", _GatedProvider("first", teardown_began, registrar_done))
    late = _Provider("late")
    refused = False

    async def register_late() -> None:
        nonlocal refused
        await teardown_began.wait()
        try:
            manager.register_provider("late", late)
        except ResourceError:
            refused = True
        registrar_done.set()

    await asyncio.gather(manager.cleanup(), register_late())

    assert refused or late.closed or "late" in manager.unclosed_providers, (
        "a provider registered during teardown was accepted, never closed "
        "and never reported --- its transport is open and nothing names it"
    )
    assert "late" not in manager.get_all_providers(), (
        "a manager that finished teardown is still holding a provider"
    )


async def test_register_provider_refuses_a_manager_that_is_closed() -> None:
    """The mechanism the test above leaves open to implementation.

    ``acquire`` already refuses a closed manager with a ``ResourceError``
    carrying the operation that failed. Registration is the same condition on
    the same object, so it reports it the same way --- a caller asking "did I
    use this after closing it?" should not have to catch two exception types
    depending on which method they reached for.
    """
    manager = ResourceManager()
    await manager.cleanup()

    with pytest.raises(ResourceError) as excinfo:
        manager.register_provider("late", _Provider("late"))

    assert excinfo.value.operation == "register_provider"
    assert excinfo.value.resource_name == "late"


def test_sync_close_refuses_a_later_registration_too() -> None:
    """Both teardown spellings leave the manager equally unusable.

    ``close()`` claims closure at the same point ``cleanup()`` does, so a
    manager torn down synchronously must refuse registration for the same
    reason. Without this the guard would depend on which half ran.
    """
    manager = ResourceManager()
    manager.close()

    with pytest.raises(ResourceError):
        manager.register_provider("late", _Provider("late"))


def test_the_guard_runs_before_the_teardown_convention_check() -> None:
    """A closed manager is closed regardless of the provider offered.

    The convention check raises ``ValueError`` for a misnamed teardown. If it
    ran first, a closed manager would report the wrong problem for a provider
    that has two --- and the caller would fix the name and hit the real one.
    """

    class _Misnamed(_Provider):
        async def close(self) -> None:  # type: ignore[override]
            return None

    manager = ResourceManager()
    manager.close()

    with pytest.raises(ResourceError):
        manager.register_provider("late", _Misnamed("late"))


# --------------------------------------------------------------------------- #
# Mutation during the classification sweep
# --------------------------------------------------------------------------- #


async def test_cleanup_survives_concurrent_unregistration(
    brief_switch_interval: None,
) -> None:
    """The sweep must not abort because another thread touched the registry.

    ``unregister_provider`` takes the lock; the sweep does not, so the two
    are not in fact excluding each other. Aborting here is worse than it
    looks: it leaves ``_providers`` uncleared and every provider after the
    mutation point untorn-down, with the exception surfacing from a
    ``close()`` a caller may have written inside ``__aexit__``.
    """
    manager = ResourceManager()
    for index in range(500):
        manager.register_provider(f"p{index}", _Provider(f"p{index}"))

    start = threading.Event()

    def churn() -> None:
        start.wait()
        for index in range(500):
            try:
                manager.unregister_provider(f"p{index}")
            except Exception:
                # The churn is scaffolding; the sweep's verdict is the subject,
                # so a provider this thread loses a race for is not a failure.
                pass

    thread = threading.Thread(target=churn)
    thread.start()
    start.set()
    try:
        await manager.cleanup()
    finally:
        thread.join()

    assert not manager.get_all_providers(), "teardown did not finish the registry"

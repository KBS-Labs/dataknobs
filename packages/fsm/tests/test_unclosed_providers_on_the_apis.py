"""What teardown could not finish is readable from the object a caller holds.

``ResourceManager`` records it, but a consumer does not hold a
``ResourceManager`` --- it holds a ``SimpleFSM``, an ``AdvancedFSM`` or an
``AsyncSimpleFSM``, and the manager is a private attribute of one of those. A
record only a test reaching into ``_resource_manager`` can read is a fixture,
not a feature, so all three inherit the record --- and the registration that
puts something there to record --- from one shared surface.

That surface is why every test below goes in through ``register_resource``
and out through ``unclosed_providers``, on all three classes, touching no
private attribute at either end. The file could not be written that way
before: two of the three classes had no public way to register a provider,
so proving the record worked meant reaching past the API to set the record
up. A test that has to break the surface to reach the feature is evidence
about the surface, and this file used to be that evidence.

The three are not interchangeable, and the tests here pin the difference
rather than assuming it:

* ``AdvancedFSM.close()`` runs the *synchronous* teardown path, so it is the
  one surface where a caller can reach the skipped-awaited-teardown
  population. That is the case the record exists for.
* ``SimpleFSM.close()`` looks synchronous and is not: it drives the async
  cleanup through the shared bridge, so an ``aclose`` provider is awaited and
  nothing is recorded. Worth a test precisely because the name suggests
  otherwise.
* ``AsyncSimpleFSM`` awaits throughout.

Real providers throughout --- the point is what a provider's own teardown
does, so a stand-in for one would be testing the stand-in.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import pytest

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    NetworkConfig,
    StateConfig,
)
from dataknobs_fsm.resources.base import ResourceHealth, ResourceMetrics, ResourceStatus


def _trivial_dict() -> dict[str, object]:
    """The same FSM in the dict form ``SimpleFSM`` accepts."""
    return {
        "name": "trivial",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end", "name": "go"}],
            }
        ],
    }


def _trivial_config() -> FSMConfig:
    """A minimal start->end FSM (no transforms, no resources)."""
    return FSMConfig(
        name="trivial",
        main_network="main",
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                    StateConfig(name="end", is_end=True),
                ],
            )
        ],
    )


class _Provider:
    """A provider holding one releasable thing, so release is observable."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = ResourceStatus.IDLE
        self.closed = False

    def acquire(self, **kwargs: object) -> object:
        self.status = ResourceStatus.BUSY
        return {"name": self.name}

    def release(self, resource: object) -> None:
        self.status = ResourceStatus.IDLE

    def validate(self, resource: object) -> bool:
        return True

    def health_check(self) -> ResourceHealth:
        return ResourceHealth.HEALTHY

    def get_metrics(self) -> ResourceMetrics:
        return ResourceMetrics(total_acquisitions=0, active_connections=0, failed_acquisitions=0)

    def close(self) -> None:
        self.closed = True
        self.status = ResourceStatus.CLOSED


class _AsyncProvider(_Provider):
    """A provider wrapping an async transport, so its teardown is awaited."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.aclosed = False

    # No `close`: this provider's only teardown must be awaited. Inheriting
    # one would make the synchronous path look successful for the wrong
    # reason, which is the defect rather than the fixture.
    close = None  # type: ignore[assignment]

    async def aclose(self) -> None:
        self.aclosed = True
        self.status = ResourceStatus.CLOSED


class _RaisingProvider(_Provider):
    """A provider whose synchronous teardown fails."""

    def close(self) -> None:
        raise RuntimeError(f"{self.name} refused to close")


class _RaisingAsyncProvider(_Provider):
    """A provider whose awaited teardown fails."""

    close = None  # type: ignore[assignment]

    async def aclose(self) -> None:
        raise RuntimeError(f"{self.name} refused to aclose")


# --------------------------------------------------------------------------- #
# AdvancedFSM --- the one surface whose close() cannot await
# --------------------------------------------------------------------------- #


def test_advanced_fsm_names_the_provider_its_sync_close_could_not_await() -> None:
    """End to end through public API only, which is the whole point.

    ``register_resource`` in, ``unclosed_providers`` out. Before this
    property the only way to learn that the database was left open was to
    read the log, and the only way to *assert* it was to touch a private
    attribute.
    """
    fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))
    provider = _AsyncProvider("db")
    fsm.register_resource("db", provider)

    fsm.close()

    assert "db" in fsm.unclosed_providers, (
        "a provider whose teardown could not be awaited was not reported to the caller"
    )
    assert not provider.aclosed, "the fixture is wrong: the teardown did run"


def test_advanced_fsm_aclose_awaits_it_and_records_nothing() -> None:
    """The other half of the same case: choosing ``aclose`` fixes it.

    Without this, the test above passes just as well against a property that
    reports every async provider unconditionally.
    """

    async def run() -> tuple[AdvancedFSM, _AsyncProvider]:
        fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))
        provider = _AsyncProvider("db")
        fsm.register_resource("db", provider)
        await fsm.aclose()
        return fsm, provider

    fsm, provider = asyncio.run(run())

    assert provider.aclosed
    assert not fsm.unclosed_providers


def test_advanced_fsm_names_a_provider_whose_close_raised() -> None:
    """The second recorded population, from the surface a caller holds."""
    fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))
    fsm.register_resource("bad", _RaisingProvider("bad"))

    fsm.close()

    assert "bad" in fsm.unclosed_providers


def test_advanced_fsm_records_nothing_when_teardown_succeeds() -> None:
    """Empty is the normal answer, and is what a caller asserts."""
    fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))
    provider = _Provider("props")
    fsm.register_resource("props", provider)

    fsm.close()

    assert provider.closed
    assert not fsm.unclosed_providers


def test_advanced_fsm_reports_it_after_a_context_manager_exit() -> None:
    """``with`` is the recommended spelling, so the record must survive it.

    ``__exit__`` calls ``close()``, which does not propagate a teardown
    failure --- it is reachable from a ``with`` body that may itself be
    raising. The record is what is left instead of the exception, so it has
    to be readable once the block is over.
    """
    fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))

    with fsm:
        fsm.register_resource("bad", _RaisingProvider("bad"))

    assert "bad" in fsm.unclosed_providers


# --------------------------------------------------------------------------- #
# AsyncSimpleFSM --- awaits throughout
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_async_simple_fsm_names_a_provider_whose_aclose_raised() -> None:
    fsm = AsyncSimpleFSM(_trivial_dict())
    fsm.register_resource("bad", _RaisingAsyncProvider("bad"))

    await fsm.close()

    assert "bad" in fsm.unclosed_providers


@pytest.mark.asyncio
async def test_async_simple_fsm_records_nothing_when_teardown_succeeds() -> None:
    fsm = AsyncSimpleFSM(_trivial_dict())
    provider = _AsyncProvider("db")
    fsm.register_resource("db", provider)

    await fsm.close()

    assert provider.aclosed
    assert not fsm.unclosed_providers


# --------------------------------------------------------------------------- #
# SimpleFSM --- synchronous in name, awaited in fact
# --------------------------------------------------------------------------- #


def test_simple_fsm_sync_close_still_awaits_an_aclose_provider() -> None:
    """The surprise its docstring exists to remove.

    ``close()`` is synchronous, but drives the *async* cleanup through the
    shared bridge --- so unlike ``AdvancedFSM.close()`` it does not skip
    awaited teardown, and nothing is recorded. A reader who assumes the two
    sync surfaces behave alike would assume the opposite.
    """
    fsm = SimpleFSM(_trivial_dict())
    provider = _AsyncProvider("db")
    fsm.register_resource("db", provider)

    fsm.close()

    assert provider.aclosed, "SimpleFSM.close() no longer drives the awaited path"
    assert not fsm.unclosed_providers


def test_simple_fsm_names_a_provider_whose_teardown_raised() -> None:
    fsm = SimpleFSM(_trivial_dict())
    fsm.register_resource("bad", _RaisingAsyncProvider("bad"))

    fsm.close()

    assert "bad" in fsm.unclosed_providers


def test_simple_fsm_shares_one_manager_with_the_async_class_behind_it() -> None:
    """``SimpleFSM`` borrows ``AsyncSimpleFSM``'s manager rather than owning one.

    The shared surface reads ``self._resource_manager`` on all three classes,
    which is only correct here because ``SimpleFSM.__init__`` caches the
    manager its inner ``AsyncSimpleFSM`` built. If that ever became a second
    manager the mixin would register into one object and the FSM would tear
    down the other, and both halves would look fine in isolation --- so the
    registration goes in through the outer class and the record is read from
    both. Reaching for the inner class is the point of this test and of no
    other in the file.
    """
    fsm = SimpleFSM(_trivial_dict())
    fsm.register_resource("bad", _RaisingProvider("bad"))

    fsm.close()

    assert dict(fsm.unclosed_providers) == dict(fsm._async_fsm.unclosed_providers)


# --------------------------------------------------------------------------- #
# The property is a view, not a handle
# --------------------------------------------------------------------------- #


def test_the_record_cannot_be_edited_through_the_api_property() -> None:
    """A caller must not be able to clear the evidence by accident."""
    fsm = AdvancedFSM(FSMBuilder().build(_trivial_config()))
    fsm.register_resource("bad", _RaisingProvider("bad"))
    fsm.close()

    with pytest.raises(TypeError):
        fsm.unclosed_providers["bad"] = "not my problem"  # type: ignore[index]


# --------------------------------------------------------------------------- #
# One surface, three classes
# --------------------------------------------------------------------------- #


def _close_anyhow(fsm: object) -> None:
    """Tear down whichever of the three this is.

    Only ``AsyncSimpleFSM.close`` is a coroutine. That asymmetry is real and
    is not what these tests are about, so it is absorbed once here rather
    than splitting the parametrisation in two.
    """
    result = fsm.close()  # type: ignore[attr-defined]
    if inspect.isawaitable(result):
        asyncio.run(result)


_ALL_THREE: list[Any] = [
    pytest.param(lambda: SimpleFSM(_trivial_dict()), id="SimpleFSM"),
    pytest.param(lambda: AsyncSimpleFSM(_trivial_dict()), id="AsyncSimpleFSM"),
    pytest.param(lambda: AdvancedFSM(FSMBuilder().build(_trivial_config())), id="AdvancedFSM"),
]


@pytest.mark.parametrize("make_fsm", _ALL_THREE)
def test_each_api_class_can_register_and_then_list_a_provider(
    make_fsm: Callable[[], Any],
) -> None:
    """The claim the shared surface makes, stated as one test over three classes.

    Registering and listing used to be on opposite halves of the API: the two
    simple classes could list and not register, the advanced one could
    register and not list. So a caller picked their class by which half they
    needed, and *this test could not have been written* --- there was no class
    it passed on. That it is now parametrised is the whole result.

    Synchronous on all three, including the async one: registering and listing
    touch only the manager's own dict, so making them awaitable on that class
    alone would be a third shape for no gain.
    """
    fsm = make_fsm()
    try:
        assert fsm.get_resources() == [], "a fresh FSM reported a provider"
        fsm.register_resource("db", _Provider("db"))
        assert fsm.get_resources() == ["db"]
    finally:
        _close_anyhow(fsm)


@pytest.mark.parametrize("make_fsm", _ALL_THREE)
def test_the_dict_form_of_register_resource_works_on_every_class(
    make_fsm: Callable[[], Any],
) -> None:
    """``AdvancedFSM``'s two-argument shape came across whole.

    The mixin took the signature the one class that had this method already
    published, rather than a narrower one for the classes gaining it --- a
    provider *or* a config dict. Narrowing it on two of three would be the
    same divergence again, one level down.
    """
    fsm = make_fsm()
    try:
        fsm.register_resource("props", {"data": {"k": "v"}})
        assert fsm.get_resources() == ["props"]
    finally:
        _close_anyhow(fsm)


def test_a_provider_registered_through_the_simple_api_is_torn_down() -> None:
    """End to end on the class that previously had no way in.

    ``SimpleFSM`` could list providers and could not add one, so the only way
    to get a provider into it was through two private attributes. Registering
    through the public surface must reach the same manager that ``close``
    tears down --- if it reached any other object this would pass every
    assertion except this one.
    """
    fsm = SimpleFSM(_trivial_dict())
    provider = _Provider("db")
    fsm.register_resource("db", provider)

    fsm.close()

    assert provider.closed, "a provider registered through the public surface was not closed"
    assert not fsm.unclosed_providers

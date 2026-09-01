"""Provider teardown is routed by method *name*, and nothing bound anyone to it.

``ResourceManager`` decides how to tear a provider down by asking which method
it has: ``close`` goes to the synchronous path, ``aclose`` / ``cleanup`` to the
awaited one. That is the standard convention --- ``asyncio``,
``contextlib.aclosing``, and the pair ``dataknobs_common.lifecycle`` probes ---
and it is a sound one. What it lacked was anything holding a provider to it.

The convention was stated nowhere: ``IResourceProvider`` declares no teardown
method at all, and ``BaseResourceProvider`` gives *every* provider a sync
``close()``, so ``hasattr(provider, "close")`` is true of all of them and
distinguishes nothing. A provider was therefore free to spell an awaitable
teardown ``close``, and one shipped inside the manager itself did: the
``SimpleResourceProvider`` returned by ``create_provider_from_dict`` had
``async def close``, so the manager's own bucket sort filed it as synchronous,
called it, discarded the coroutine, and logged that it had closed it.

Four consequences, and the quietest is the expensive one:

**A coroutine discarded and logged as a success.** The loud one --- it emits a
``RuntimeWarning`` --- and the cheap one, since that provider's body is ``pass``.

**An ``AttributeError`` raised from a ``finally:``.** The stream executor
probed for ``aclose`` *or* ``close`` and then called ``close`` unconditionally,
so a source offering only the former replaced whatever exception the body was
propagating.

**Teardown abandoned partway.** ``close()``'s provider loop had no ``try``, so
one raising provider stranded every provider after it in iteration order ---
and skipped the registry clear that follows, leaving the manager marked closed
while still holding everything.

**A teardown silently skipped.** ``AsyncDatabaseResourceAdapter`` does not
override ``close()``, so the sync path ran the inherited base close, released
the handles, and never touched the database. No coroutine is created, so
nothing warns; the manager then clears its registry and the object holding the
open connection becomes unreachable. That one is covered in
``test_lifecycle_async_parity.py``, where the sync/async parity claim lives.

Real providers throughout. Where a test needs a provider that fails, it is a
real ``BaseResourceProvider`` subclass whose teardown raises --- the failure is
the condition under test, not a stand-in for one.
"""

from __future__ import annotations

import ast
import gc
import warnings
from pathlib import Path
from typing import Any

import pytest

from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import State
from dataknobs_fsm.execution.stream import StreamExecutor, StreamPipeline
from dataknobs_fsm.resources.base import BaseResourceProvider, ResourceStatus
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.streaming.core import StreamChunk


class RecordingProvider(BaseResourceProvider):
    """A minimal real provider that records whether it was closed."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.closed = False

    def acquire(self, **kwargs: Any) -> Any:
        return {"name": self.name}

    def release(self, resource: Any) -> None:
        pass

    def close(self) -> None:
        super().close()
        self.closed = True


class FailingCloseProvider(RecordingProvider):
    """A provider whose synchronous teardown raises."""

    def close(self) -> None:
        raise RuntimeError(f"{self.name} refused to close")


class FailingAcloseProvider(RecordingProvider):
    """A provider whose awaited teardown raises."""

    async def aclose(self) -> None:
        raise RuntimeError(f"{self.name} refused to aclose")


class FailingCleanupProvider(RecordingProvider):
    """A provider spelling its awaited teardown ``cleanup``, and failing it."""

    async def cleanup(self) -> None:
        raise RuntimeError(f"{self.name} refused to clean up")


class AcloseProvider(RecordingProvider):
    """A provider whose real teardown must be awaited."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.aclosed = False

    async def aclose(self) -> None:
        self.aclosed = True
        self.status = ResourceStatus.CLOSED


# --------------------------------------------------------------------------- #
# The convention is enforced where a provider enters the system
# --------------------------------------------------------------------------- #


def test_registering_a_provider_with_an_async_close_is_refused() -> None:
    """The one moment a developer can still act on it.

    A provider whose ``close`` must be awaited cannot be served correctly by
    *any* caller of this manager, so accepting it buys a smoother start in
    exchange for a leak later. The message has to name the fix, because the
    author's mistake is a naming one and is invisible from the symptom.
    """

    class AsyncCloseProvider(RecordingProvider):
        async def close(self) -> None:  # type: ignore[override]
            pass

    manager = ResourceManager()

    with pytest.raises(ValueError, match="aclose"):
        manager.register_provider("bad", AsyncCloseProvider("bad"))


def test_a_provider_with_a_sync_close_still_registers() -> None:
    """The regression guard for the check above: it must refuse only the defect."""
    manager = ResourceManager()
    manager.register_provider("good", RecordingProvider("good"))
    assert "good" in manager.get_all_providers()


def test_a_provider_with_aclose_still_registers() -> None:
    """``aclose`` is the *correct* spelling and must not be caught by the check."""
    manager = ResourceManager()
    manager.register_provider("good", AcloseProvider("good"))
    assert "good" in manager.get_all_providers()


def test_registering_a_provider_with_a_sync_aclose_is_refused() -> None:
    """The other half of the same naming mistake.

    ``AsyncClosable`` is ``runtime_checkable``, so the routing sees the name
    and not the asyncness --- ``await provider.aclose()`` then runs a
    synchronous body for its side effect and raises ``TypeError`` on the
    ``await``, which the manager records as a teardown that failed. The
    teardown in fact ran. Refusing at registration is what makes the record
    mean what it says, and it is the same "the name is the contract" defect
    as an async ``close``.
    """

    class SyncAcloseProvider(RecordingProvider):
        def aclose(self) -> None:
            pass

    manager = ResourceManager()

    with pytest.raises(ValueError, match="synchronous aclose"):
        manager.register_provider("bad", SyncAcloseProvider("bad"))


def test_registering_a_provider_with_a_sync_cleanup_is_refused() -> None:
    """``cleanup`` is the alternate spelling, so it carries the same obligation.

    Worse than the ``aclose`` case on the synchronous path: the provider has
    no ``close``, so ``_close_provider`` calls nothing at all and records the
    skipped-awaited-teardown reason over a method that was synchronous and
    callable the whole time.
    """

    class SyncCleanupProvider(RecordingProvider):
        def cleanup(self) -> None:
            pass

    manager = ResourceManager()

    with pytest.raises(ValueError, match="synchronous cleanup"):
        manager.register_provider("bad", SyncCleanupProvider("bad"))


def test_a_provider_disclaiming_an_inherited_aclose_still_registers() -> None:
    """The guard tests presence the way the routing does: not ``None``.

    ``AsyncClosable`` is an ``isinstance`` against a ``runtime_checkable``
    Protocol, and setting the attribute to ``None`` fails that check --- so
    such a provider is routed down the synchronous path and must not be
    refused on the way in.
    """

    class DisclaimingProvider(RecordingProvider):
        aclose = None

    manager = ResourceManager()
    manager.register_provider("good", DisclaimingProvider("good"))
    assert "good" in manager.get_all_providers()


def test_the_managers_own_provider_tears_down_without_discarding_a_coroutine() -> None:
    """``create_provider_from_dict`` violated the manager's own routing rule.

    Recorded rather than filtered to an error: the "coroutine was never
    awaited" warning is emitted from the coroutine's deallocator, where an
    exception would be printed and swallowed rather than raised.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        manager = ResourceManager()
        manager.register_from_dict("props", {"data": {"k": "v"}})
        manager.close()
        gc.collect()

    unawaited = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not unawaited, (
        f"a coroutine was created and discarded: {[str(w.message) for w in unawaited]}"
    )


# --------------------------------------------------------------------------- #
# close() must survive a provider that fails
# --------------------------------------------------------------------------- #


def test_one_failing_provider_does_not_strand_the_providers_after_it() -> None:
    """The loop had no ``try``, so failure ended teardown rather than surviving it.

    Three providers, the middle one raising. The assertion that matters is the
    *third*: it is the one an unguarded loop never reaches. The registry clear
    that follows the loop is checked too --- it was skipped for the same
    reason, leaving a manager marked closed while still holding everything it
    had failed to close.
    """
    manager = ResourceManager()
    first = RecordingProvider("first")
    second = FailingCloseProvider("second")
    third = RecordingProvider("third")
    manager.register_provider("first", first)
    manager.register_provider("second", second)
    manager.register_provider("third", third)

    manager.close()

    assert first.closed
    assert third.closed, "teardown stopped at the failing provider"
    assert manager.get_all_providers() == {}, "the registry was never cleared"
    assert "second" in manager.unclosed_providers


def test_a_failure_during_close_is_recorded_rather_than_propagated() -> None:
    """``close()`` is reachable from ``__exit__``, where raising is a worse outcome.

    An exception here replaces whatever the ``with`` body was raising, turning
    a leak into a lost diagnosis. The failure is recorded instead, which is
    what makes swallowing it defensible.
    """
    manager = ResourceManager()
    manager.register_provider("bad", FailingCloseProvider("bad"))

    with manager:
        pass

    assert set(manager.unclosed_providers) == {"bad"}


# --------------------------------------------------------------------------- #
# unclosed_providers: what teardown could not finish
# --------------------------------------------------------------------------- #


def test_a_clean_close_records_nothing() -> None:
    """Empty is the normal answer, and the assertion a caller should write."""
    manager = ResourceManager()
    manager.register_provider("props", RecordingProvider("props"))

    manager.close()

    assert manager.unclosed_providers == {}


async def test_a_clean_cleanup_records_nothing() -> None:
    manager = ResourceManager()
    manager.register_provider("sync", RecordingProvider("sync"))
    manager.register_provider("async", AcloseProvider("async"))

    await manager.cleanup()

    assert manager.unclosed_providers == {}


async def test_cleanup_names_the_provider_that_failed_to_aclose() -> None:
    """The ``gather`` results were reported as ``task {i}`` --- an index into a
    list the reader cannot see. Attributing a failure needs the pairing, which
    is the same change that populates the record.
    """
    manager = ResourceManager()
    manager.register_provider("ok", AcloseProvider("ok"))
    manager.register_provider("bad", FailingAcloseProvider("bad"))

    await manager.cleanup()

    assert set(manager.unclosed_providers) == {"bad"}


async def test_cleanup_names_the_provider_that_failed_to_clean_up() -> None:
    """The second awaited path: a provider spelling its teardown ``cleanup``."""
    manager = ResourceManager()
    manager.register_provider("bad", FailingCleanupProvider("bad"))

    await manager.cleanup()

    assert set(manager.unclosed_providers) == {"bad"}


async def test_cleanup_records_a_failure_on_the_synchronous_path() -> None:
    manager = ResourceManager()
    manager.register_provider("bad", FailingCloseProvider("bad"))

    await manager.cleanup()

    assert set(manager.unclosed_providers) == {"bad"}


def test_a_provider_with_no_teardown_at_all_is_not_recorded() -> None:
    """There was nothing to close, which is a legitimate shape.

    ``lifecycle._report_unclosable`` logs exactly this case at DEBUG rather
    than WARNING for the same reason --- a frozen config or a plain mapping
    needs no teardown, and a record that fires on those is one people learn to
    ignore.
    """

    class NoTeardownProvider:
        def acquire(self, **kwargs: Any) -> Any:
            return {}

        def release(self, resource: Any) -> None:
            pass

    manager = ResourceManager()
    manager.register_provider("plain", NoTeardownProvider())  # type: ignore[arg-type]

    manager.close()

    assert manager.unclosed_providers == {}


def test_the_record_survives_a_second_close() -> None:
    """Monotonic, and the reason is a hazard rather than a preference.

    ``close()`` is terminal and clears the registry, so a second call --- which
    is exactly what a ``with`` block produces after an explicit one --- finds
    nothing to close and would honestly reset the record to empty, erasing the
    first call's evidence. "Did anything leak during this manager's life" is
    the question a caller has, and it has one answer.
    """
    manager = ResourceManager()
    manager.register_provider("bad", FailingCloseProvider("bad"))

    manager.close()
    assert set(manager.unclosed_providers) == {"bad"}

    manager.close()
    assert set(manager.unclosed_providers) == {"bad"}, "the second close erased the record"


def test_the_record_cannot_be_edited_through_the_property() -> None:
    manager = ResourceManager()
    manager.close()

    with pytest.raises(TypeError):
        manager.unclosed_providers["invented"] = "nonsense"  # type: ignore[index]


# --------------------------------------------------------------------------- #
# The stream executor's teardown probe must name the method it calls
# --------------------------------------------------------------------------- #


def _one_state_fsm() -> FSM:
    fsm = FSM(name="teardown_fsm")
    network = StateNetwork(name="main")
    network.add_state(State(name="start", type="start"), initial=True)
    network.add_state(State(name="end", type="end"), final=True)
    network.add_arc("start", "end")
    fsm.add_network(network, is_main=True)
    return fsm


class AcloseOnlySource:
    """A source whose teardown must be awaited --- so it has no ``close``.

    Deliberately not a subclass of ``IStreamSource``: subclassing a Protocol
    explicitly inherits its method bodies, which would hand this class the
    very ``close`` whose absence is the point.
    """

    def __init__(self) -> None:
        self.aclosed = False
        self._sent = False

    def read_chunk(self) -> StreamChunk | None:
        if self._sent:
            return None
        self._sent = True
        return StreamChunk(data=[{"value": 1}], chunk_id=0, is_last=True)

    def __iter__(self) -> Any:
        while (chunk := self.read_chunk()) is not None:
            yield chunk

    async def aclose(self) -> None:
        self.aclosed = True


def test_a_source_offering_only_aclose_does_not_raise_from_the_finally() -> None:
    """The probe admitted either name and then called one of them.

    Raised from a ``finally:``, an ``AttributeError`` here replaces whatever
    the body was propagating --- so the failure a caller sees is not the
    failure that happened.
    """
    executor = StreamExecutor(fsm=_one_state_fsm())
    source = AcloseOnlySource()

    stats = executor.execute_stream(StreamPipeline(source=source))  # type: ignore[arg-type]

    assert stats is not None


class CloseOnlySource(AcloseOnlySource):
    """The ordinary shape: synchronous teardown, no awaited half."""

    aclose = None  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_a_source_offering_close_is_still_closed() -> None:
    """Regression guard for the shape that already worked."""
    executor = StreamExecutor(fsm=_one_state_fsm())
    source = CloseOnlySource()

    executor.execute_stream(StreamPipeline(source=source))  # type: ignore[arg-type]

    assert source.closed


def test_a_source_offering_both_runs_the_half_it_can() -> None:
    """A source with both halves gets the synchronous one run, not skipped.

    Same policy as ``ResourceManager._close_provider``: do what this engine
    can do, then report what it cannot. Skipping ``close()`` because ``aclose``
    exists would discard the half that *was* available.
    """

    class BothSource(AcloseOnlySource):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    executor = StreamExecutor(fsm=_one_state_fsm())
    source = BothSource()

    executor.execute_stream(StreamPipeline(source=source))  # type: ignore[arg-type]

    assert source.closed
    assert not source.aclosed, "the synchronous engine cannot have awaited it"


# --------------------------------------------------------------------------- #
# Recurrence guard
# --------------------------------------------------------------------------- #

_RESOURCES = Path(__file__).resolve().parents[1] / "src" / "dataknobs_fsm" / "resources"


def test_no_provider_in_this_package_spells_an_awaited_teardown_close() -> None:
    """Our own providers, checked at commit time.

    The registration check in ``ResourceManager.register_provider`` covers
    every provider that enters at runtime, including a consumer's. This covers
    ours a step earlier, and in the one directory that holds them --- which is
    where ``SimpleResourceProvider`` would have been caught. It walks the AST
    rather than importing, because the provider that broke the convention was
    a class nested inside a method, and no attribute walk would have found it.

    Scoped to ``resources/`` deliberately: an ``async def close`` is correct
    elsewhere in this package (``AsyncSimpleFSM``, the io adapters, the storage
    backends), where the caller knows the object's shape statically instead of
    routing on a method name.
    """
    offenders: list[str] = []

    for path in sorted(_RESOURCES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for member in node.body:
                if isinstance(member, ast.AsyncFunctionDef) and member.name == "close":
                    offenders.append(f"{path.name}:{member.lineno} {node.name}.close")

    assert not offenders, (
        "teardown is routed by method name: an awaited teardown is spelled "
        f"`aclose`, never `close`. Offenders: {offenders}"
    )

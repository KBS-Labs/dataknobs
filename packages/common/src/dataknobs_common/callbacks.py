"""In-process named-callback dispatch with pluggable ordering and error policy.

The :class:`CallbackRegistry` is the canonical in-process callback
substrate for dataknobs subsystems (bot lifecycle hooks,
knowledge-ingestion events, intent-classification observability,
resource-coordination callbacks). Consumer code registers named
callbacks on string topics; the registry fires them under the configured
ordering.

Sync AND async callbacks are both supported. The dispatch model is
chosen at fire-time: :meth:`CallbackRegistry.fire` runs sync callbacks
(and raises ``TypeError`` if an async callback is registered on the
topic); :meth:`CallbackRegistry.fire_async` awaits both sync and async
callbacks.

Pluggable ordering:

- :class:`FIFOOrdering` — insertion order (default; matches a list
  registry's natural behavior).
- :class:`PriorityOrdering` — numeric priority (lower fires first),
  FIFO tiebreaker.
- :class:`StageOrdering` — named-stage ordering; unknown stages sort to
  the end; FIFO tiebreaker within a stage.
- :class:`CompositeOrdering` — first non-zero compare wins.

Pluggable error policy (per-registry):

- :attr:`ErrorPolicy.RAISE` — re-raise the first failing callback;
  subsequent callbacks do not run.
- :attr:`ErrorPolicy.LOG_AND_CONTINUE` (default) — log the exception,
  continue dispatch.
- :attr:`ErrorPolicy.LOG_AND_RAISE_AT_END` — run all callbacks (later
  callbacks still run after a failure), then raise
  :class:`BatchedCallbackError` carrying every ``(entry, exc)`` pair.

Composition with :class:`~dataknobs_common.events.EventBus`::

    registry.also_publish_to(bus, topic_prefix="wizard:")

Every :meth:`CallbackRegistry.fire` / :meth:`CallbackRegistry.fire_async`
additionally publishes the payload to ``bus`` under
``topic_prefix + topic``. Local callbacks still run; the bus is the
cross-replica observability substrate.

Fan-out is *observability*: a failed bus ``publish`` must never break
or mask the operation being observed. By default
(``isolate_errors=True``) a publish exception is logged and swallowed
so the fire — and the caller driving it — proceeds unaffected. A
consumer who needs publish failures to be fatal (a durable audit
trail, say) opts out per-target via ``isolate_errors=False``.
``asyncio.CancelledError`` is always re-raised regardless, since
cancellation is control-flow, not a publish failure.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cmp_to_key
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from dataknobs_common.events import Event, EventBus, EventType
from dataknobs_common.exceptions import DataknobsError

__all__ = [
    "BatchedCallbackError",
    "CallbackEntry",
    "CallbackOrdering",
    "CallbackRegistry",
    "CapturingCallbackRegistry",
    "CompositeOrdering",
    "ErrorPolicy",
    "FIFOOrdering",
    "PriorityOrdering",
    "RecordingCallbackRegistry",
    "StageOrdering",
]

logger = logging.getLogger(__name__)

CallbackT = TypeVar("CallbackT", bound=Callable[..., Any])


# --------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CallbackEntry(Generic[CallbackT]):
    """Registered callback metadata.

    Implementation detail of :class:`CallbackRegistry`; exposed for
    :class:`CallbackOrdering` implementations to read.

    Attributes:
        topic: The fire-time topic the callback responds to.
        callback: The user-supplied callable.
        priority: Numeric priority. Lower fires first under
            :class:`PriorityOrdering`; ignored by :class:`FIFOOrdering`
            and :class:`StageOrdering` alone.
        stage: Named-stage label. Used by :class:`StageOrdering`;
            defaults to ``"main"``.
        registration_seq: Monotonic per-registry insertion counter,
            stamped by the registry on :meth:`CallbackRegistry.register`.
            Stable FIFO tiebreaker for every ordering.
    """

    topic: str
    callback: CallbackT
    priority: int = 0
    stage: str = "main"
    registration_seq: int = 0


class ErrorPolicy(Enum):
    """Per-registry policy for callback dispatch errors.

    String values (``"raise"`` / ``"log"`` / ``"batch"``) are
    intentionally short for compact serialization and config
    declarations; they do not mirror the member names. Consumers
    deserializing from config use the value form
    (``ErrorPolicy("log")`` ⇒ :attr:`LOG_AND_CONTINUE`), not the
    member name. Treat the values as a stable identifier surface —
    renaming a value is a breaking config-format change.
    """

    RAISE = "raise"
    LOG_AND_CONTINUE = "log"
    LOG_AND_RAISE_AT_END = "batch"


class BatchedCallbackError(DataknobsError):
    """Aggregate error raised by :attr:`ErrorPolicy.LOG_AND_RAISE_AT_END`.

    Raised when one or more callbacks failed during a single
    :meth:`CallbackRegistry.fire` / :meth:`CallbackRegistry.fire_async`
    call. Carries the list of ``(entry, exception)`` pairs for diagnosis.
    """

    def __init__(
        self,
        failures: list[tuple[CallbackEntry[Any], BaseException]],
    ) -> None:
        self.failures = failures
        topics = sorted({entry.topic for entry, _ in failures})
        super().__init__(
            f"{len(failures)} callback(s) failed during dispatch "
            f"on topic(s) {topics}",
        )


# --------------------------------------------------------------------- #
# Ordering protocol + reference implementations
# --------------------------------------------------------------------- #


@runtime_checkable
class CallbackOrdering(Protocol):
    """Compares two registered callbacks for dispatch order.

    Implementations return ``-1`` / ``0`` / ``+1`` like ``cmp()``:
    ``a`` fires first if ``compare(a, b) < 0``. Implementations MUST
    be:

    - deterministic (same inputs → same output across invocations);
    - total (no incomparable pairs that produce inconsistent results);
    - pure (no observable side effects).

    Equal entries (``compare`` returns ``0``) MAY fire in any order;
    consumers needing a stable tiebreaker compose with
    :class:`FIFOOrdering` via :class:`CompositeOrdering`.
    """

    def compare(
        self,
        a: CallbackEntry[Any],
        b: CallbackEntry[Any],
    ) -> int: ...


@dataclass(frozen=True)
class FIFOOrdering:
    """Insertion-order dispatch.

    The default; matches a list registry's natural behavior. Earlier-
    registered callbacks fire first.
    """

    def compare(
        self,
        a: CallbackEntry[Any],
        b: CallbackEntry[Any],
    ) -> int:
        if a.registration_seq < b.registration_seq:
            return -1
        if a.registration_seq > b.registration_seq:
            return 1
        return 0


@dataclass(frozen=True)
class PriorityOrdering:
    """Numeric priority comparator.

    Lower priority fires first. Consumers register guard callbacks
    with ``priority=-100`` so they fire before default-priority
    callbacks regardless of registration order.

    Equal-priority entries compare ``0`` — the registry's stable
    :func:`sorted` preserves their registration order, and
    :class:`CompositeOrdering` passes the tie to the next inner
    ordering. To make the FIFO behavior explicit in a composition,
    append :class:`FIFOOrdering` as the last tier.
    """

    def compare(
        self,
        a: CallbackEntry[Any],
        b: CallbackEntry[Any],
    ) -> int:
        if a.priority < b.priority:
            return -1
        if a.priority > b.priority:
            return 1
        return 0


@dataclass(frozen=True)
class StageOrdering:
    """Named-stage comparator.

    Stages listed earlier fire first. Unknown stages sort to the end.

    Same-stage entries compare ``0`` — the registry's stable
    :func:`sorted` preserves their registration order, and
    :class:`CompositeOrdering` passes the tie to the next inner
    ordering (so the canonical
    ``CompositeOrdering(StageOrdering(...), PriorityOrdering(), FIFOOrdering())``
    composition reaches priority within a stage).

    Construct with the stage labels in fire-order::

        StageOrdering(stages=("pre", "main", "post"))
    """

    stages: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError(
                "StageOrdering requires at least one stage label",
            )

    def _stage_index(self, stage: str) -> int:
        try:
            return self.stages.index(stage)
        except ValueError:
            return len(self.stages)

    def compare(
        self,
        a: CallbackEntry[Any],
        b: CallbackEntry[Any],
    ) -> int:
        ai, bi = self._stage_index(a.stage), self._stage_index(b.stage)
        if ai < bi:
            return -1
        if ai > bi:
            return 1
        return 0


@dataclass(frozen=True)
class CompositeOrdering:
    """Compose multiple orderings.

    The first inner ordering returning a non-zero compare wins; later
    orderings act as tiebreakers. Canonical composition for the common
    stage + priority + FIFO case::

        CompositeOrdering(
            StageOrdering(("pre", "main", "post")),
            PriorityOrdering(),
            FIFOOrdering(),
        )
    """

    orderings: tuple[CallbackOrdering, ...]

    def __init__(self, *orderings: CallbackOrdering) -> None:
        if not orderings:
            raise ValueError(
                "CompositeOrdering requires at least one inner ordering",
            )
        # __init__ on a frozen dataclass needs object.__setattr__
        object.__setattr__(self, "orderings", orderings)

    def compare(
        self,
        a: CallbackEntry[Any],
        b: CallbackEntry[Any],
    ) -> int:
        for ordering in self.orderings:
            result = ordering.compare(a, b)
            if result != 0:
                return result
        return 0


# --------------------------------------------------------------------- #
# CallbackRegistry
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _FanoutTarget:
    """A configured EventBus fan-out target.

    Internal to :class:`CallbackRegistry`. ``isolate_errors`` decides
    whether a failed ``publish`` to this bus is logged and swallowed
    (the default — fan-out is observability and must never break or
    mask the observed operation) or re-raised out of the fire.
    """

    bus: EventBus
    topic_prefix: str
    isolate_errors: bool


class CallbackRegistry(Generic[CallbackT]):
    """In-process named-callback dispatch with pluggable ordering.

    Construct with an ordering (default :class:`FIFOOrdering`) and an
    error policy (default :attr:`ErrorPolicy.LOG_AND_CONTINUE`). Register
    callbacks on string topics; fire them by topic.

    Sync and async dispatch:

    - :meth:`fire` runs sync callbacks in ordering order. If an async
      callback is registered on the topic, ``fire`` raises ``TypeError``
      at dispatch time (not at registration — the registration cost is
      a single insertion).
    - :meth:`fire_async` awaits both sync and async callbacks in
      ordering order. Sync callbacks run inline; async callbacks are
      awaited.

    EventBus fan-out: call :meth:`also_publish_to` to additionally
    publish every fired payload to a shared
    :class:`~dataknobs_common.events.EventBus` under
    ``topic_prefix + topic``. Local callbacks still run; the bus is the
    cross-replica observability substrate.
    """

    def __init__(
        self,
        *,
        ordering: CallbackOrdering | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.LOG_AND_CONTINUE,
    ) -> None:
        self._ordering: CallbackOrdering = ordering or FIFOOrdering()
        self._error_policy = error_policy
        self._entries: dict[str, list[CallbackEntry[CallbackT]]] = {}
        self._seq = 0
        self._fanout_buses: list[_FanoutTarget] = []

    # ----- registration ----- #

    def register(
        self,
        topic: str,
        callback: CallbackT,
        *,
        priority: int = 0,
        stage: str = "main",
    ) -> None:
        """Register ``callback`` on ``topic``.

        Multiple callbacks on the same topic are allowed; the configured
        ordering decides fire order.
        """
        self._seq += 1
        entry = CallbackEntry(
            topic=topic,
            callback=callback,
            priority=priority,
            stage=stage,
            registration_seq=self._seq,
        )
        self._entries.setdefault(topic, []).append(entry)

    def unregister(self, topic: str, callback: CallbackT) -> bool:
        """Remove the first matching ``callback`` from ``topic``.

        Returns ``True`` if found and removed; ``False`` otherwise.
        Identity comparison (``is``) — pass the same callable object
        that was registered.
        """
        entries = self._entries.get(topic)
        if not entries:
            return False
        for i, entry in enumerate(entries):
            if entry.callback is callback:
                del entries[i]
                return True
        return False

    def clear(self, topic: str | None = None) -> None:
        """Drop registered callbacks.

        Drains the entry lists in place — the registry instance, its
        ordering, its error policy, and any configured EventBus fan-out
        targets are preserved. Consumers holding a reference to the
        registry continue to use the same object after clear.

        Args:
            topic: When ``None`` (default), every topic is drained.
                Otherwise, only ``topic`` is drained.
        """
        if topic is None:
            for entries in self._entries.values():
                entries.clear()
            return
        entries = self._entries.get(topic)
        if entries is not None:
            entries.clear()

    def set_ordering(self, ordering: CallbackOrdering) -> None:
        """Replace the ordering. Affects subsequent ``fire`` calls only."""
        self._ordering = ordering

    # ----- dispatch ----- #

    def fire(self, topic: str, payload: dict[str, Any]) -> None:
        """Fire ``topic`` synchronously.

        Raises ``TypeError`` if any callback registered on ``topic``
        is a coroutine function, OR if EventBus fan-out is configured
        via :meth:`also_publish_to` and ``fire`` is called from inside
        a running event loop — both cases require :meth:`fire_async`.
        The fan-out guard prevents fire-and-forget background tasks
        whose exceptions would be silently swallowed and whose
        completion is not guaranteed (the task could be garbage-
        collected before delivery, raising ``Task was destroyed but
        is pending`` in Python 3.12+).

        When no event loop is running, fan-out is driven to completion
        on a fresh loop via :func:`asyncio.run` before this method
        returns.

        .. note::

           The "no running loop" guard reads
           :func:`asyncio.get_running_loop` against the calling thread
           only. If ``fire()`` is called from a thread that has its
           own running event loop (Jupyter notebooks, an
           :func:`asyncio.to_thread` target that itself drives an
           asyncio loop, or a
           :class:`~concurrent.futures.ThreadPoolExecutor` worker
           that wraps :func:`asyncio.run`), the inner
           :func:`asyncio.run` will raise
           ``RuntimeError("asyncio.run() cannot be called from a
           running event loop")``. Prefer :meth:`fire_async` for all
           fan-out paths in async-heavy environments.
        """
        entries = self._sorted_entries(topic)
        async_entries = [
            e for e in entries
            if inspect.iscoroutinefunction(e.callback)
        ]
        if async_entries:
            raise TypeError(
                f"Cannot fire() topic {topic!r}: "
                f"{len(async_entries)} async callback(s) registered. "
                f"Use fire_async() instead.",
            )
        self._publish_to_buses(topic, payload)
        failures: list[tuple[CallbackEntry[CallbackT], BaseException]] = []
        for entry in entries:
            try:
                entry.callback(payload)
            except Exception as exc:
                self._handle_failure(entry, exc, failures)
        if (
            self._error_policy is ErrorPolicy.LOG_AND_RAISE_AT_END
            and failures
        ):
            raise BatchedCallbackError(failures)

    async def fire_async(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        """Fire ``topic`` asynchronously.

        Awaits async callbacks; runs sync callbacks inline. Ordering
        is preserved.
        """
        entries = self._sorted_entries(topic)
        await self._publish_to_buses_async(topic, payload)
        failures: list[tuple[CallbackEntry[CallbackT], BaseException]] = []
        for entry in entries:
            try:
                result = entry.callback(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._handle_failure(entry, exc, failures)
        if (
            self._error_policy is ErrorPolicy.LOG_AND_RAISE_AT_END
            and failures
        ):
            raise BatchedCallbackError(failures)

    # ----- EventBus fan-out ----- #

    def also_publish_to(
        self,
        bus: EventBus,
        *,
        topic_prefix: str = "",
        isolate_errors: bool = True,
    ) -> None:
        """Compose with an EventBus.

        Every :meth:`fire` / :meth:`fire_async` additionally publishes
        the payload to ``bus`` under ``topic_prefix + topic``. Multiple
        fan-out targets are supported; call this method once per target.
        Each fired payload is wrapped in an
        :class:`~dataknobs_common.events.Event` with
        :attr:`~dataknobs_common.events.EventType.CUSTOM` for transport.

        Fan-out publishes happen BEFORE local callbacks run. A failing
        local callback under :attr:`ErrorPolicy.LOG_AND_CONTINUE` (or
        :attr:`ErrorPolicy.LOG_AND_RAISE_AT_END`) cannot suppress bus
        delivery; under :attr:`ErrorPolicy.RAISE`, bus subscribers
        still see the event before the raise unwinds.

        Args:
            bus: the EventBus to forward fired payloads to.
            topic_prefix: prepended to the fire topic to form the bus
                topic.
            isolate_errors: when ``True`` (default), a failed ``publish``
                to *this* target is logged and swallowed so the fire —
                and the operation it observes — proceeds unaffected.
                Fan-out is observability and must never break or mask
                the observed operation, so this is the safe default for
                the realistic case (a transient broker / network hiccup
                on a real bus). Pass ``False`` only when a publish
                failure SHOULD abort the fire (e.g. a durable audit
                trail). ``asyncio.CancelledError`` is always re-raised
                regardless of this flag.
        """
        self._fanout_buses.append(
            _FanoutTarget(bus, topic_prefix, isolate_errors)
        )

    # ----- introspection ----- #

    def topics(self) -> Iterable[str]:
        """Return the set of topics with at least one registered callback.

        Read-only view; modifying the returned iterable does not affect
        the registry.
        """
        return tuple(
            topic for topic, entries in self._entries.items() if entries
        )

    def callback_count(self, topic: str) -> int:
        """Return the number of callbacks registered on ``topic``."""
        return len(self._entries.get(topic, ()))

    def supports_event_bus_emission(self) -> bool:
        """Return ``True`` if this registry has at least one EventBus
        fan-out target configured via :meth:`also_publish_to`.

        The :class:`~dataknobs_common.capabilities.Capability` check
        the consumer-facing surface advertises is the technical
        condition "registry forwards to one or more buses." This
        accessor exposes that condition without requiring consumers to
        reach into private state.
        """
        return bool(self._fanout_buses)

    # ----- internals ----- #

    def _sorted_entries(
        self,
        topic: str,
    ) -> list[CallbackEntry[CallbackT]]:
        entries = self._entries.get(topic, [])
        if len(entries) <= 1:
            return list(entries)
        return sorted(
            entries,
            key=cmp_to_key(self._ordering.compare),
        )

    def _build_event(
        self,
        full_topic: str,
        payload: dict[str, Any],
    ) -> Event:
        return Event(
            type=EventType.CUSTOM,
            topic=full_topic,
            payload=payload,
        )

    def _publish_to_buses(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        if not self._fanout_buses:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — drive the publishes to completion under
            # a fresh loop. Reuses the same gather batch that
            # ``fire_async`` runs, so semantics are identical.
            asyncio.run(self._publish_to_buses_async(topic, payload))
            return
        raise TypeError(
            f"Cannot fire() topic {topic!r} with EventBus fan-out "
            f"configured from a running event loop: bus publishes "
            f"would be scheduled as fire-and-forget tasks subject "
            f"to garbage collection and silent exception loss. "
            f"Use fire_async() instead.",
        )

    async def _publish_to_buses_async(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        if not self._fanout_buses:
            return
        # ``return_exceptions=True`` so one target's failure neither
        # cancels the sibling publishes nor escapes by default — each
        # result is then dispositioned per the target's
        # ``isolate_errors`` flag. The default (isolate) keeps fan-out
        # non-load-bearing for the observed operation; an opt-out target
        # re-raises after every target has been attempted.
        results = await asyncio.gather(
            *(
                t.bus.publish(
                    t.topic_prefix + topic,
                    self._build_event(t.topic_prefix + topic, payload),
                )
                for t in self._fanout_buses
            ),
            return_exceptions=True,
        )
        reraise: BaseException | None = None
        for target, result in zip(self._fanout_buses, results):
            if not isinstance(result, BaseException):
                continue
            if isinstance(result, asyncio.CancelledError):
                # Cancellation is control-flow, not a publish failure —
                # never swallow it, whatever the isolate_errors flag.
                raise result
            if target.isolate_errors:
                full_topic = target.topic_prefix + topic
                logger.warning(
                    "EventBus fan-out to %s failed on topic %r; "
                    "observed operation continues: %s",
                    type(target.bus).__name__,
                    full_topic,
                    result,
                )
            elif reraise is None:
                reraise = result
        if reraise is not None:
            raise reraise

    def _handle_failure(
        self,
        entry: CallbackEntry[CallbackT],
        exc: BaseException,
        failures: list[tuple[CallbackEntry[CallbackT], BaseException]],
    ) -> None:
        if self._error_policy is ErrorPolicy.RAISE:
            raise exc
        if self._error_policy is ErrorPolicy.LOG_AND_CONTINUE:
            logger.exception(
                "Callback %r on topic %r raised; continuing dispatch.",
                entry.callback,
                entry.topic,
            )
            return
        # LOG_AND_RAISE_AT_END — log now, raise after all fire.
        logger.exception(
            "Callback %r on topic %r raised; will raise batched at "
            "end of dispatch.",
            entry.callback,
            entry.topic,
        )
        failures.append((entry, exc))


# --------------------------------------------------------------------- #
# Testing-friendly implementations
# --------------------------------------------------------------------- #


class CapturingCallbackRegistry(CallbackRegistry[CallbackT]):
    """Real :class:`CallbackRegistry` that also captures every dispatched
    event for test assertions.

    Use in tests to verify *what was fired*::

        registry = CapturingCallbackRegistry(
            ordering=PriorityOrdering(),
        )
        registry.register("turn_start", my_callback)
        registry.fire("turn_start", {"stage": "greet"})
        assert registry.captured == [
            ("turn_start", {"stage": "greet"}),
        ]

    Captured payloads are stored by reference. If the production code
    mutates the payload after :meth:`CallbackRegistry.fire`, the
    captured entry reflects the post-mutation state.

    Capture order: the payload is appended to :attr:`captured`
    **before** :class:`CallbackRegistry` dispatch runs. A payload
    therefore appears in :attr:`captured` even when dispatch raises
    (an async callback registered under :meth:`fire`, a callback
    that raises under :attr:`ErrorPolicy.RAISE`, a
    :class:`BatchedCallbackError` under
    :attr:`ErrorPolicy.LOG_AND_RAISE_AT_END`, etc.). Tests asserting
    "the production code attempted to fire" should read
    :attr:`captured`; tests asserting "the callbacks ran without
    error" should additionally check that ``fire`` / ``fire_async``
    did not raise.
    """

    def __init__(
        self,
        *,
        ordering: CallbackOrdering | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(ordering=ordering, error_policy=error_policy)
        self.captured: list[tuple[str, dict[str, Any]]] = []

    def fire(self, topic: str, payload: dict[str, Any]) -> None:
        self.captured.append((topic, payload))
        super().fire(topic, payload)

    async def fire_async(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        self.captured.append((topic, payload))
        await super().fire_async(topic, payload)


class RecordingCallbackRegistry:
    """Standalone test double that records ``(topic, payload)`` tuples
    without dispatching to callbacks.

    Use when the test wants to verify the production code calls
    :meth:`fire`, without running the registered callbacks at all (e.g.
    when callback side effects are expensive or irrelevant). Does NOT
    extend :class:`CallbackRegistry`; instead it implements the full
    duck-typed surface that downstream consumers (notably
    :class:`~dataknobs_bots.reasoning.lifecycle.LifecycleHooks`) call:
    ``register`` / ``unregister`` / ``clear`` / ``set_ordering`` /
    ``callback_count`` / ``fire`` / ``fire_async``. The dispatch path
    is a no-op (callbacks are recorded but never invoked) and
    :meth:`set_ordering` is a no-op (ordering is irrelevant when
    nothing dispatches), but the methods exist so a test injecting
    this double does not blow up with ``AttributeError`` when the
    production code calls them.

    ``captured`` accumulates every ``(topic, payload)`` that flows
    through :meth:`fire` / :meth:`fire_async`. :meth:`clear` drains
    *registrations only* — the captured list is the test-observable
    artifact and is preserved across clears, matching the analogous
    behavior of :meth:`CallbackRegistry.clear` (which drains the
    entry lists in place and leaves fan-out targets alone).
    """

    def __init__(self) -> None:
        self.captured: list[tuple[str, dict[str, Any]]] = []
        self._registered: list[tuple[str, Callable[..., Any]]] = []

    def register(
        self,
        topic: str,
        callback: Callable[..., Any],
        *,
        priority: int = 0,
        stage: str = "main",
    ) -> None:
        self._registered.append((topic, callback))

    def unregister(
        self,
        topic: str,
        callback: Callable[..., Any],
    ) -> bool:
        for i, (t, c) in enumerate(self._registered):
            if t == topic and c is callback:
                del self._registered[i]
                return True
        return False

    def clear(self, topic: str | None = None) -> None:
        """Drop recorded registrations.

        Mirrors :meth:`CallbackRegistry.clear` — when ``topic`` is
        ``None`` every registration is dropped; otherwise only
        registrations on ``topic`` are dropped. The :attr:`captured`
        list (test-observable fire history) is preserved across
        clears.
        """
        if topic is None:
            self._registered.clear()
            return
        self._registered = [
            (t, c) for t, c in self._registered if t != topic
        ]

    def set_ordering(self, ordering: CallbackOrdering) -> None:
        """No-op: ordering is irrelevant when dispatch is suppressed.

        Provided so production code that customizes the registry
        ordering (``hooks.registry.set_ordering(...)``) does not blow
        up with ``AttributeError`` when a test injects this double.
        """

    def callback_count(self, topic: str) -> int:
        """Return the number of registrations recorded on ``topic``."""
        return sum(1 for t, _ in self._registered if t == topic)

    def fire(self, topic: str, payload: dict[str, Any]) -> None:
        self.captured.append((topic, payload))

    async def fire_async(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        self.captured.append((topic, payload))

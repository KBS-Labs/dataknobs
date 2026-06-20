"""Named-key state bridging Protocol and reference implementations.

A :class:`StateBridge` is the discipline layer over a state-carrying host
(typically a ``ConversationManager``): it codifies HOW one component
publishes state for another component to consume, named by key.

The Protocol is intentionally minimal — two methods, both optional in
practice (a read-only bridge raises ``NotImplementedError`` on write).
Implementations choose semantics:

- ``read_inbox`` semantics:
    - consume-on-read (pop) — the default; the inbox is owned by the
      writer for one turn and consumed by the reader.
    - peek-without-consume — read repeatedly without consumption
      (consumer responsibility to clear).
- ``write_outbox`` semantics:
    - unsupported (read-only bridge) — raises ``NotImplementedError``.
    - assign — overwrite ``host.metadata[key]``.
    - merge — combine with the existing ``host.metadata[key]`` via a
      ``merge_fn``.
    - project — write a projection (subset / transformed) of state.
- side effects:
    - none (pure read/write).
    - callback-firing (observability-aware via a ``CallbackRegistry``).

The bridge does NOT define the payload shape. Per-key payload
conventions live in consumer-controlled documentation (e.g. a wizard's
stage-output payload shape is documented by the wizard's sub-strategy
declaring it; the bridge carries the payload as-is).

Composition with the callback substrate (``CallbackRegistry``):
    :class:`SubscribingBridge` composes a real bridge implementation
    with callback fan-out. Read AND write events fire on the configured
    ``CallbackRegistry`` topic; observability consumers register
    callbacks to monitor every bridge read / write without modifying the
    production bridge code. Composing the registry's
    ``also_publish_to(bus, ...)`` fan-out extends that observability
    across replicas via a shared ``EventBus``.

Composition with scope projection:
    :class:`SubsetBridge` accepts EITHER a bare
    ``Callable[[Any], OutboxT]`` OR any object exposing a ``project``
    method (a scope projector such as a ``WhitelistProjector``). A
    consumer's projector drops straight into a bridge with no glue. The
    two Protocols stay distinct — a scope projector transforms a source
    for a safe-eval scope; a state bridge's project transforms a host
    snapshot for an outbox write — only the ``project`` callable is
    shared.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

__all__ = [
    "BiDirectionalBridge",
    "InboxOnlyBridge",
    "PeekBridge",
    "StateBridge",
    "SubscribingBridge",
    "SubsetBridge",
]

InboxT = TypeVar("InboxT")
OutboxT = TypeVar("OutboxT")


@runtime_checkable
class StateBridge(Protocol, Generic[InboxT, OutboxT]):
    """Named-key state bridging between components on a shared host.

    Two methods:

    - ``read_inbox(host, key)`` — read state published under ``key``.
      Returns the payload or ``None``. Semantics are
      implementation-defined: most production bridges consume the key on
      read (pop semantics) so the writer-side owns the key for exactly
      one turn.

    - ``write_outbox(host, key, value)`` — publish state under ``key``.
      Implementation-defined: read-only bridges raise
      ``NotImplementedError``; bi-directional bridges assign or merge;
      subset bridges project state through a callable.

    ``host`` is typically a ``ConversationManager`` instance carrying
    ``host.metadata`` as the state-carrying mapping; the Protocol does
    not require that exact shape — any object exposing a mapping-like
    ``metadata`` attribute conforms.
    """

    def read_inbox(self, host: Any, key: str) -> InboxT | None: ...

    def write_outbox(self, host: Any, key: str, value: OutboxT) -> None: ...


def _host_metadata(host: Any) -> dict[str, Any]:
    """Return the host's ``metadata`` mapping or raise a clear error.

    Shared by every reference implementation so the "host must expose a
    ``metadata`` mapping" contract is enforced in exactly one place.
    """
    try:
        return host.metadata
    except AttributeError as exc:
        raise AttributeError(
            "StateBridge host must expose a `metadata` mapping attribute "
            "(typically a ConversationManager).",
        ) from exc


# --------------------------------------------------------------------- #
# Reference implementations
# --------------------------------------------------------------------- #


class InboxOnlyBridge(Generic[InboxT]):
    """Consume-on-read inbox; write side unsupported.

    The wizard ``make_metadata_inbox_hook`` semantic. The writer (a
    sub-strategy via ``write_to_inbox``) places a payload under a named
    key in ``host.metadata``; the reader (the wizard's auto-registered
    ``on_turn_start`` hook) pops it.

    Read returns the payload and removes the key from ``host.metadata``.
    Missing key returns ``None``.

    Write raises ``NotImplementedError`` so a consumer trying to use this
    bridge bi-directionally surfaces immediately rather than silently
    no-op'ing.
    """

    def read_inbox(self, host: Any, key: str) -> InboxT | None:
        return _host_metadata(host).pop(key, None)

    def write_outbox(self, host: Any, key: str, value: Any) -> None:
        raise NotImplementedError(
            "InboxOnlyBridge is read-only. To enable a writer-side, use "
            "BiDirectionalBridge or SubsetBridge.",
        )


class PeekBridge(Generic[InboxT]):
    """Read-without-consume inbox; write side unsupported.

    Identical to :class:`InboxOnlyBridge` except read = ``dict.get``
    (peek) rather than ``dict.pop`` — the key survives the read so
    multiple hook stages within one turn can observe the same inbox
    payload. The consumer is responsible for clearing the key when done.

    Write raises ``NotImplementedError`` (same as
    :class:`InboxOnlyBridge`).
    """

    def read_inbox(self, host: Any, key: str) -> InboxT | None:
        return _host_metadata(host).get(key)

    def write_outbox(self, host: Any, key: str, value: Any) -> None:
        raise NotImplementedError(
            "PeekBridge is read-only. To enable a writer-side, use "
            "BiDirectionalBridge or SubsetBridge.",
        )


class BiDirectionalBridge(Generic[InboxT, OutboxT]):
    """Symmetric bridge: read = pop; write = assign or merge.

    Construct with an optional ``merge_fn`` taking ``(existing, new)``
    and mutating ``existing`` in place. The default ``None`` overwrites
    ``host.metadata[key]`` on every write.

    Forward-compat for a bi-directional state-bridging consumer surface
    (outbox-key registration). The Protocol stays fixed; future
    configuration knobs construct this bridge instead of the inbox-only
    one.
    """

    def __init__(
        self,
        merge_fn: Callable[[dict[str, Any], Mapping[str, Any]], None]
        | None = None,
    ) -> None:
        self._merge_fn = merge_fn

    def read_inbox(self, host: Any, key: str) -> InboxT | None:
        return _host_metadata(host).pop(key, None)

    def write_outbox(self, host: Any, key: str, value: OutboxT) -> None:
        metadata = _host_metadata(host)
        if self._merge_fn is None:
            metadata[key] = value
            return
        existing = metadata.get(key)
        if existing is None:
            metadata[key] = value
            return
        if not isinstance(existing, dict) or not isinstance(value, Mapping):
            # merge_fn requires a mutable-dict existing + mapping value.
            metadata[key] = value
            return
        self._merge_fn(existing, value)


class SubsetBridge(Generic[InboxT, OutboxT]):
    """Partial-state projection bridge.

    The ``project`` callable transforms the full state-bearing source
    (typically a wizard ``state`` or a manager-metadata snapshot) into
    the projected outbox value the consumer wants exposed. Read = pop.
    Write = ``host.metadata[key] = project(value)`` where ``value`` is
    the source supplied by the caller.

    Use case: a wizard publishes only a chosen subset of state to a
    downstream sub-strategy, without exposing internals like raw
    intermediate parses or LLM-call metadata.

    The ``project`` argument accepts EITHER a bare
    ``Callable[[Any], OutboxT]`` OR a scope projector (anything exposing
    a ``project`` method, e.g. a ``WhitelistProjector`` /
    ``CallableProjector``). A consumer's projector drops straight into a
    ``SubsetBridge`` with no glue. The two Protocols stay distinct (a
    scope projector is not a ``StateBridge``); only the ``project``
    callable is shared.
    """

    def __init__(self, project: Callable[[Any], OutboxT] | Any) -> None:
        # Normalize: a scope projector exposes `.project`; a bare callable
        # is used directly. Duck-typed — no import of the projector layer.
        proj_method = getattr(project, "project", None)
        self._project: Callable[[Any], OutboxT] = (
            proj_method if callable(proj_method) else project
        )

    def read_inbox(self, host: Any, key: str) -> InboxT | None:
        return _host_metadata(host).pop(key, None)

    def write_outbox(self, host: Any, key: str, value: Any) -> None:
        _host_metadata(host)[key] = self._project(value)


class SubscribingBridge(Generic[InboxT, OutboxT]):
    """Bridge that fires callbacks on read and write.

    Composes with the callback substrate: every read and write fires its
    respective topic on the configured ``CallbackRegistry`` with a
    standard event payload (``{"host", "key", "value"}`` for writes;
    ``{"host", "key", "popped"}`` for reads).

    Use case: observability — a metrics consumer registers callbacks on
    the read / write topics to count bridge activity per key without
    modifying the production bridge code.
    """

    def __init__(
        self,
        inner: StateBridge[InboxT, OutboxT],
        *,
        registry: Any,  # CallbackRegistry; typed Any to avoid an import cycle
        read_topic: str = "state_bridge:read",
        write_topic: str = "state_bridge:write",
    ) -> None:
        self._inner = inner
        self._registry = registry
        self._read_topic = read_topic
        self._write_topic = write_topic

    def read_inbox(self, host: Any, key: str) -> InboxT | None:
        popped = self._inner.read_inbox(host, key)
        self._registry.fire(
            self._read_topic,
            {"host": host, "key": key, "popped": popped},
        )
        return popped

    def write_outbox(self, host: Any, key: str, value: OutboxT) -> None:
        self._inner.write_outbox(host, key, value)
        self._registry.fire(
            self._write_topic,
            {"host": host, "key": key, "value": value},
        )

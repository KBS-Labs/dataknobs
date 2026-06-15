"""Generic resource-lookup-by-key Protocols and reference implementations.

A resource resolver classifies a key into a resource (or ``None`` for
opt-out). Implementations may wrap mappings, callables, caches, or
compose other resolvers. Consumer code writes against the Protocol;
the choice of implementation is a configuration concern, not a code
concern.

This module ships both the generic Protocols and a small library of
reference implementations covering the common option space:

Generic implementations:
    MappingResolver       — wraps a Mapping[KeyT, ValueT]
    CallableResolver      — wraps a Callable[[KeyT], ValueT | None]
    DefaultingResolver    — substitutes a default for None returns
    CachedResolver        — adds LRU caching to any inner resolver
    CompositeResolver     — chains resolvers; first non-None wins
    NullResolver          — always returns None (testing / feature-off)

Async implementations:
    AsyncCallableResolver — wraps an async callable
    AsyncCachedResolver   — async cache with asyncio.Lock-guarded insertion

Vector-store partition implementations (consumers can use these
directly or as templates for custom partition logic):
    NullPartitionResolver           — single-partition default
    MetadataKeyPartitionResolver    — extracts from record metadata
    TemporalPartitionResolver       — time bucket from a metadata field
    CallablePartitionResolver       — escape hatch for arbitrary logic
    JoiningPartitionResolver        — joins sub-resolvers with a separator
                                      (distinct from CompositeResolver,
                                      which ALTERNATES — first-non-None
                                      wins — rather than concatenates)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

KeyT_contra = TypeVar("KeyT_contra", contravariant=True)
ValueT_co = TypeVar("ValueT_co", covariant=True)
_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ResourceResolver(Protocol, Generic[KeyT_contra, ValueT_co]):
    """Synchronous resource lookup-by-key.

    Implementations resolve a key to a value, or to ``None`` when no value
    is available (callers decide whether ``None`` is an error or a normal
    opt-out).

    Implementations MUST be safe under repeated calls with the same key
    (idempotent at the protocol level — caching strategies may vary).
    Implementations MUST NOT mutate the input key.
    """

    def resolve(self, key: KeyT_contra) -> ValueT_co | None: ...


@runtime_checkable
class AsyncResourceResolver(Protocol, Generic[KeyT_contra, ValueT_co]):
    """Asynchronous resource lookup-by-key.

    Used when resolution requires I/O (factory functions building
    network-bound resources, dynamic configuration loading). Same
    idempotence / non-mutation contract as :class:`ResourceResolver`.
    """

    async def resolve(self, key: KeyT_contra) -> ValueT_co | None: ...


# ---------------------------------------------------------------------------
# Generic reference implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingResolver(Generic[_KeyT, _ValueT]):
    """Wraps a :class:`~collections.abc.Mapping`. Returns ``mapping[key]``
    if present, ``None`` otherwise.

    Production-ready for static, in-memory registrations. Thread-safe iff
    the wrapped mapping is thread-safe.
    """

    mapping: Mapping[_KeyT, _ValueT]

    def resolve(self, key: _KeyT) -> _ValueT | None:
        try:
            return self.mapping[key]
        except KeyError:
            return None


@dataclass(frozen=True)
class CallableResolver(Generic[_KeyT, _ValueT]):
    """Wraps a ``Callable[[KeyT], ValueT | None]``. Production-ready for
    dynamic lookups (e.g. factories that construct resources on demand).
    """

    fn: Callable[[_KeyT], _ValueT | None]

    def resolve(self, key: _KeyT) -> _ValueT | None:
        return self.fn(key)


@dataclass(frozen=True)
class DefaultingResolver(Generic[_KeyT, _ValueT]):
    """Wraps an inner resolver; substitutes ``default`` for ``None``
    returns.

    Useful when downstream code expects always-non-None but the inner
    resolver may return ``None``. The :meth:`resolve` return type is
    deliberately tighter than the :class:`ResourceResolver` Protocol
    (``_ValueT`` rather than ``_ValueT | None``) — the whole point of
    this wrapper is to eliminate the ``None`` case for consumers.
    Passing a ``None`` default defeats the purpose and falls outside
    the contract.
    """

    inner: ResourceResolver[_KeyT, _ValueT]
    default: _ValueT

    def resolve(self, key: _KeyT) -> _ValueT:
        result = self.inner.resolve(key)
        return result if result is not None else self.default


class CachedResolver(Generic[_KeyT, _ValueT]):
    """Adds LRU caching to any inner resolver.

    The cache is keyed on the input key (which must be hashable).
    ``None`` returns are NOT cached (so a transient ``None`` doesn't
    persist when the underlying resource becomes available later).
    """

    def __init__(
        self,
        inner: ResourceResolver[_KeyT, _ValueT],
        max_size: int = 128,
    ) -> None:
        self._inner = inner

        @lru_cache(maxsize=max_size)
        def _cached(key: _KeyT) -> _ValueT:
            # raise KeyError on None so lru_cache doesn't memoize the miss
            result = inner.resolve(key)
            if result is None:
                raise KeyError(key)
            return result

        self._cached = _cached

    def resolve(self, key: _KeyT) -> _ValueT | None:
        try:
            return self._cached(key)
        except KeyError:
            return None


@dataclass(frozen=True)
class CompositeResolver(Generic[_KeyT, _ValueT]):
    """Chains resolvers; first non-None result wins.

    Useful for layered lookup (e.g. consumer override → tenant default →
    global default). Empty chain always returns ``None``.
    """

    resolvers: Sequence[ResourceResolver[_KeyT, _ValueT]]

    def resolve(self, key: _KeyT) -> _ValueT | None:
        for resolver in self.resolvers:
            result = resolver.resolve(key)
            if result is not None:
                return result
        return None


@dataclass(frozen=True)
class NullResolver(Generic[_KeyT, _ValueT]):
    """Always returns ``None``. Useful as a testing stub OR a production
    sentinel for "feature disabled".
    """

    def resolve(self, key: _KeyT) -> _ValueT | None:
        return None


# ---------------------------------------------------------------------------
# Async reference implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsyncCallableResolver(Generic[_KeyT, _ValueT]):
    """Wraps an async :class:`~collections.abc.Callable`. Used when
    resource construction is itself async (e.g. backend factories,
    network-bound lookups).
    """

    fn: Callable[[_KeyT], Awaitable[_ValueT | None]]

    async def resolve(self, key: _KeyT) -> _ValueT | None:
        return await self.fn(key)


class AsyncCachedResolver(Generic[_KeyT, _ValueT]):
    """Async cache. Uses :class:`asyncio.Lock` for task-safe insertion."""

    def __init__(
        self,
        inner: AsyncResourceResolver[_KeyT, _ValueT],
        max_size: int = 128,
    ) -> None:
        self._inner = inner
        self._max_size = max_size
        self._cache: dict[Any, _ValueT] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, key: _KeyT) -> _ValueT | None:
        if key in self._cache:
            return self._cache[key]
        async with self._lock:
            if key in self._cache:
                return self._cache[key]
            result = await self._inner.resolve(key)
            if result is None:
                return None
            if len(self._cache) >= self._max_size:
                # Simple FIFO eviction; full LRU semantics not required
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result
            return result


# ---------------------------------------------------------------------------
# Vector-store partition reference implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullPartitionResolver:
    """Always returns ``default`` partition. Backwards-compatible default
    for single-partition vector stores.
    """

    default: str = "default"

    def resolve(self, record: Any) -> str | None:
        return self.default


@dataclass(frozen=True)
class MetadataKeyPartitionResolver:
    """Extracts the partition name from a record's metadata field.

    Covers the common shapes: per-tenant (``metadata_key="tenant_id"``),
    per-content-type (``"content_type"``), per-language (``"language"``),
    per-classification (``"classification"``), per-region (``"region"``),
    per-cohort (``"cohort"``), per-version (``"schema_version"``).

    The record is expected to expose a ``metadata`` mapping; resolves to
    ``default`` when the key is missing.

    Only scalar metadata values (``str``, ``int``, ``float``, ``bool``)
    are accepted as partition identifiers. Non-scalar values (lists,
    dicts, sets, custom objects) resolve to ``default`` rather than
    being silently ``str()``-coerced into garbage partition names like
    ``"[1, 2]"`` or ``"{'k': 'v'}"`` — partition names typically become
    table suffixes, collection names, or filesystem paths where such
    coercions are corrupting.
    """

    metadata_key: str
    default: str = "default"

    def resolve(self, record: Any) -> str | None:
        metadata = getattr(record, "metadata", None)
        if metadata is None:
            return self.default
        value = metadata.get(self.metadata_key)
        if value is None:
            return self.default
        if not isinstance(value, (str, int, float, bool)):
            return self.default
        return str(value)


_SUPPORTED_TEMPORAL_BUCKETS = frozenset({"year", "quarter", "month"})


@dataclass(frozen=True)
class TemporalPartitionResolver:
    """Buckets records into partitions by a timestamp metadata field.

    ``bucket`` is one of ``"year"``, ``"quarter"``, ``"month"``;
    validated at construction (``__post_init__``) so a typo fails fast
    rather than silently routing every record to ``default`` until the
    first valid datetime is seen. The metadata value at
    ``timestamp_key`` must be a :class:`datetime` or ISO-8601 string;
    returns ``default`` when the field is missing or unparseable.

    Examples:
        TemporalPartitionResolver("ingested_at", bucket="quarter")
          → "2026_q2" for a May 2026 timestamp
    """

    timestamp_key: str
    bucket: str = "quarter"
    default: str = "default"

    def __post_init__(self) -> None:
        if self.bucket not in _SUPPORTED_TEMPORAL_BUCKETS:
            raise ValueError(
                f"Unsupported bucket: {self.bucket!r} "
                f"(expected one of {sorted(_SUPPORTED_TEMPORAL_BUCKETS)})"
            )

    def resolve(self, record: Any) -> str | None:
        metadata = getattr(record, "metadata", None)
        if metadata is None:
            return self.default
        raw = metadata.get(self.timestamp_key)
        if raw is None:
            return self.default
        dt = _coerce_datetime(raw)
        if dt is None:
            return self.default
        return _format_bucket(dt, self.bucket)


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _format_bucket(dt: datetime, bucket: str) -> str:
    if bucket == "year":
        return f"{dt.year}"
    if bucket == "quarter":
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}_q{quarter}"
    if bucket == "month":
        return f"{dt.year}_m{dt.month:02d}"
    raise ValueError(f"Unsupported bucket: {bucket!r}")


@dataclass(frozen=True)
class CallablePartitionResolver:
    """Wraps a ``Callable[[record], str | None]`` for arbitrary partition
    logic.

    Escape hatch for partition shapes not covered by the in-tree reference
    implementations (composite tenant-x-time-x-content-type, computed
    partition keys, etc.).
    """

    fn: Callable[[Any], str | None]

    def resolve(self, record: Any) -> str | None:
        return self.fn(record)


@dataclass(frozen=True)
class JoiningPartitionResolver:
    """Joins sub-resolver partition names with a separator.

    Useful for composite partitioning: tenant x time x content-type.
    The name spells out the JOIN semantic to keep the distinction from
    :class:`CompositeResolver` unambiguous —
    :class:`CompositeResolver` ALTERNATES over its inner resolvers
    (first-non-``None`` wins); this class CONCATENATES every inner
    resolver's output.

    Example::

        JoiningPartitionResolver(
            resolvers=[
                MetadataKeyPartitionResolver("tenant_id"),
                TemporalPartitionResolver("ingested_at", bucket="quarter"),
            ],
            sep="::",
        ).resolve(record)
          → "acme::2026_q2"

    A ``None`` from any sub-resolver short-circuits the result to ``None``.

    Warning:
        Choose ``sep`` to be a character no sub-resolver produces in its
        output. :class:`TemporalPartitionResolver` returns strings
        containing ``_`` (e.g. ``"2026_q2"`` / ``"2026_m05"``), so
        ``sep="_"`` over a chain that includes a temporal resolver
        produces ambiguously-parseable joined keys
        (``"acme_2026_q2"`` could split as ``["acme", "2026", "q2"]``
        or ``["acme", "2026_q2"]``). Use ``"::"``, ``"/"``, ``"|"``,
        or another character no resolver produces when partition keys
        will be reverse-parsed downstream.
    """

    resolvers: Sequence[Any] = field(default_factory=tuple)
    sep: str = "_"

    def resolve(self, record: Any) -> str | None:
        parts: list[str] = []
        for resolver in self.resolvers:
            value = resolver.resolve(record)
            if value is None:
                return None
            parts.append(value)
        return self.sep.join(parts)


__all__ = [
    "AsyncCachedResolver",
    "AsyncCallableResolver",
    "AsyncResourceResolver",
    "CachedResolver",
    "CallablePartitionResolver",
    "CallableResolver",
    "CompositeResolver",
    "DefaultingResolver",
    "JoiningPartitionResolver",
    "MappingResolver",
    "MetadataKeyPartitionResolver",
    "NullPartitionResolver",
    "NullResolver",
    "ResourceResolver",
    "TemporalPartitionResolver",
]

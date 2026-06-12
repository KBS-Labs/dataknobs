"""Behavior tests for the generic resolver reference implementations."""

from __future__ import annotations

import pytest

from dataknobs_common.resolver import (
    AsyncCachedResolver,
    AsyncCallableResolver,
    CachedResolver,
    CallableResolver,
    CompositeResolver,
    DefaultingResolver,
    MappingResolver,
    NullResolver,
)


# ---- MappingResolver ----


def test_mapping_resolver_returns_value_for_present_key() -> None:
    r = MappingResolver({"a": 1, "b": 2})
    assert r.resolve("a") == 1


def test_mapping_resolver_returns_none_for_absent_key() -> None:
    r = MappingResolver({"a": 1})
    assert r.resolve("missing") is None


def test_mapping_resolver_is_frozen() -> None:
    r = MappingResolver({"a": 1})
    with pytest.raises(Exception):
        r.mapping = {"b": 2}  # type: ignore[misc]


# ---- CallableResolver ----


def test_callable_resolver_dispatches() -> None:
    r = CallableResolver(lambda key: len(key) if key else None)
    assert r.resolve("abc") == 3
    assert r.resolve("") is None


# ---- DefaultingResolver ----


def test_defaulting_resolver_substitutes_for_none() -> None:
    inner: NullResolver[str, int] = NullResolver()
    r = DefaultingResolver(inner, default=42)
    assert r.resolve("anything") == 42


def test_defaulting_resolver_passes_through_non_none() -> None:
    inner = MappingResolver({"a": 1})
    r = DefaultingResolver(inner, default=42)
    assert r.resolve("a") == 1
    assert r.resolve("missing") == 42


# ---- CachedResolver ----


def test_cached_resolver_caches_hits() -> None:
    calls: list[str] = []

    def fn(key: str) -> int:
        calls.append(key)
        return len(key)

    r = CachedResolver(CallableResolver(fn), max_size=8)
    assert r.resolve("abc") == 3
    assert r.resolve("abc") == 3
    assert calls == ["abc"]


def test_cached_resolver_does_not_cache_misses() -> None:
    """None returns are not cached — they're treated as transient."""
    calls: list[str] = []

    def fn(key: str) -> int | None:
        calls.append(key)
        return None if key == "miss" else len(key)

    r = CachedResolver(CallableResolver(fn))
    assert r.resolve("miss") is None
    assert r.resolve("miss") is None
    assert calls == ["miss", "miss"]


def test_cached_resolver_lru_eviction() -> None:
    calls: list[str] = []

    def fn(key: str) -> int:
        calls.append(key)
        return len(key)

    r = CachedResolver(CallableResolver(fn), max_size=2)
    r.resolve("a")
    r.resolve("b")
    r.resolve("c")  # evicts "a"
    r.resolve("a")  # recomputes
    assert calls == ["a", "b", "c", "a"]


# ---- CompositeResolver ----


def test_composite_resolver_first_non_none_wins() -> None:
    r = CompositeResolver([
        NullResolver[str, int](),
        MappingResolver({"a": 99}),
        MappingResolver({"a": 1}),  # masked by the earlier resolver
    ])
    assert r.resolve("a") == 99


def test_composite_resolver_all_none_returns_none() -> None:
    r = CompositeResolver([
        NullResolver[str, int](),
        NullResolver[str, int](),
    ])
    assert r.resolve("anything") is None


def test_composite_resolver_empty_chain_returns_none() -> None:
    r: CompositeResolver[str, int] = CompositeResolver([])
    assert r.resolve("anything") is None


# ---- NullResolver ----


def test_null_resolver_always_returns_none() -> None:
    r: NullResolver[str, int] = NullResolver()
    assert r.resolve("a") is None
    assert r.resolve("") is None


# ---- Async impls ----


@pytest.mark.asyncio
async def test_async_callable_resolver_dispatches() -> None:
    async def fn(key: str) -> int:
        return len(key)

    r = AsyncCallableResolver(fn)
    assert await r.resolve("abc") == 3


@pytest.mark.asyncio
async def test_async_cached_resolver_caches_hits() -> None:
    calls: list[str] = []

    async def fn(key: str) -> int:
        calls.append(key)
        return len(key)

    r = AsyncCachedResolver(AsyncCallableResolver(fn), max_size=8)
    assert await r.resolve("abc") == 3
    assert await r.resolve("abc") == 3
    assert calls == ["abc"]


@pytest.mark.asyncio
async def test_async_cached_resolver_eviction() -> None:
    calls: list[str] = []

    async def fn(key: str) -> int:
        calls.append(key)
        return len(key)

    r = AsyncCachedResolver(AsyncCallableResolver(fn), max_size=2)
    await r.resolve("a")
    await r.resolve("b")
    await r.resolve("c")  # evicts "a"
    await r.resolve("a")  # recomputes
    assert calls == ["a", "b", "c", "a"]

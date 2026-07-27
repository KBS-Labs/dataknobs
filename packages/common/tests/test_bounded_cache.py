"""Behavioral tests for ``BoundedLRUCache`` (the T8 primitive suite).

Covers the contract the first adopter (DynaBot's conversation cache) relies
on: access-ordered eviction, the ``max_size=None`` unbounded opt-out, the
single-fire ``on_evict`` hook, refcounted ``pin`` / ``unpin`` protection
(including the all-pinned transient-overflow escape hatch), and the
manual-removal-does-not-fire-``on_evict`` boundary.
"""

from __future__ import annotations

import pytest

from dataknobs_common import BoundedLRUCache


class TestConstruction:
    def test_default_is_unbounded(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        assert cache.max_size is None

    def test_max_size_must_be_positive(self):
        with pytest.raises(ValueError):
            BoundedLRUCache(max_size=0)
        with pytest.raises(ValueError):
            BoundedLRUCache(max_size=-1)


class TestAccessOrderedEviction:
    def test_evicts_least_recently_used(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3  # exceeds bound -> evict LRU ("a")

        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache
        assert len(cache) == 2

    def test_read_touches_recency(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache["a"] = 1
        cache["b"] = 2

        # Touch "a" so "b" becomes the LRU entry.
        assert cache["a"] == 1
        cache["c"] = 3  # evict LRU ("b"), not "a"

        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    def test_get_touches_recency(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache["a"] = 1
        cache["b"] = 2

        assert cache.get("a") == 1  # touch via get()
        cache["c"] = 3

        assert "a" in cache
        assert "b" not in cache

    def test_membership_does_not_touch_recency(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache["a"] = 1
        cache["b"] = 2

        assert "a" in cache  # membership must NOT resurrect recency
        cache["c"] = 3  # "a" is still LRU -> evicted

        assert "a" not in cache
        assert "b" in cache

    def test_update_in_place_does_not_evict(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 99  # update existing -> len unchanged, no eviction

        assert len(cache) == 2
        assert cache["a"] == 99
        assert "b" in cache


class TestUnbounded:
    def test_max_size_none_never_evicts(self):
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            on_evict=lambda k, v: evicted.append((k, v))
        )
        for i in range(1000):
            cache[f"k{i}"] = i

        assert len(cache) == 1000
        assert evicted == []


class TestOnEvictHook:
    def test_fires_once_with_evicted_pair(self):
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=1, on_evict=lambda k, v: evicted.append((k, v))
        )
        cache["a"] = 1
        cache["b"] = 2  # evicts ("a", 1)

        assert evicted == [("a", 1)]

    def test_manual_pop_does_not_fire_on_evict(self):
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=2, on_evict=lambda k, v: evicted.append((k, v))
        )
        cache["a"] = 1
        assert cache.pop("a") == 1
        assert evicted == []

    def test_manual_delete_does_not_fire_on_evict(self):
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=2, on_evict=lambda k, v: evicted.append((k, v))
        )
        cache["a"] = 1
        del cache["a"]
        assert evicted == []

    def test_clear_does_not_fire_on_evict(self):
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=2, on_evict=lambda k, v: evicted.append((k, v))
        )
        cache["a"] = 1
        cache["b"] = 2
        cache.pin("a")
        cache.clear()
        assert evicted == []
        assert len(cache) == 0
        # ``clear`` reclaims pins too — a pin must not survive the entry it
        # protected and guard a later value reinserted on the same key.
        assert not cache.is_pinned("a")


class TestPinning:
    def test_pin_protects_entry_from_lru_eviction(self):
        evicted: list[str] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=2, on_evict=lambda k, v: evicted.append(k)
        )
        cache["a"] = 1
        cache["b"] = 2
        cache.pin("a")  # "a" is LRU but pinned
        cache["c"] = 3  # over bound: evict LRU *unpinned* ("b"), spare "a"

        assert "a" in cache  # survived only because pinned
        assert "b" not in cache
        assert "c" in cache
        assert evicted == ["b"]

    def test_write_never_evicts_the_entry_it_just_inserted(self):
        # max_size=1 with the sole existing entry pinned: inserting "b"
        # must NOT evict "b" (the just-inserted, only-unpinned entry).
        evicted: list[str] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=1, on_evict=lambda k, v: evicted.append(k)
        )
        cache["a"] = 1
        cache.pin("a")
        cache["b"] = 2  # "a" pinned, "b" is MRU -> transient overflow

        assert "b" in cache  # the just-inserted entry survives its own write
        assert "a" in cache
        assert len(cache) == 2
        assert evicted == []

    def test_all_pinned_allows_transient_overflow(self):
        evicted: list[str] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=1, on_evict=lambda k, v: evicted.append(k)
        )
        cache["a"] = 1
        cache.pin("a")
        cache["b"] = 2
        cache.pin("b")
        cache["c"] = 3
        cache.pin("c")

        # Every entry is pinned -> no eviction; bound is exceeded transiently.
        assert len(cache) == 3
        assert evicted == []
        assert all(k in cache for k in ("a", "b", "c"))

    def test_unpin_restores_eviction_eligibility(self):
        evicted: list[str] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=2, on_evict=lambda k, v: evicted.append(k)
        )
        cache["a"] = 1
        cache["b"] = 2
        cache.pin("a")
        cache["c"] = 3  # "a" pinned -> evict "b"; "a" survives
        assert evicted == ["b"]

        cache.unpin("a")
        cache["d"] = 4  # "a" now unpinned & LRU -> evicted
        assert "a" not in cache
        assert "d" in cache
        assert evicted == ["b", "a"]

    def test_pin_is_refcounted(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache["a"] = 1
        cache["b"] = 2
        cache.pin("a")
        cache.pin("a")  # two pins on "a"

        cache.unpin("a")  # one released; "a" still pinned
        assert cache.is_pinned("a")

        cache["c"] = 3  # "a" still pinned -> evict "b", spare "a"
        assert "a" in cache
        assert "b" not in cache

        cache.unpin("a")  # last pin released
        assert not cache.is_pinned("a")

    def test_unpin_is_idempotent_when_unpinned(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache["a"] = 1
        # Unpinning a never-pinned key is a no-op, not an error.
        cache.unpin("a")
        cache.unpin("a")
        assert not cache.is_pinned("a")

    def test_pop_reclaims_pin_so_it_cannot_outlive_the_entry(self):
        # A stale pin surviving a pop would permanently protect a *different*
        # value later reinserted on the same key and defeat the size bound.
        evicted: list[str] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=1, on_evict=lambda k, v: evicted.append(k)
        )
        cache["a"] = 1
        cache.pin("a")
        assert cache.pop("a") == 1  # teardown reclaims the pin too
        assert not cache.is_pinned("a")

        # Reinserting "a" gets a fresh, unpinned entry — a later write can
        # evict it, so the bound is honored (no permanent overflow).
        cache["a"] = 2
        cache["b"] = 3  # "a" is LRU and unpinned -> evicted
        assert "a" not in cache
        assert "b" in cache
        assert len(cache) == 1
        assert evicted == ["a"]

    def test_delete_reclaims_pin_so_it_cannot_outlive_the_entry(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=1)
        cache["a"] = 1
        cache.pin("a")
        del cache["a"]  # manual removal reclaims the pin
        assert not cache.is_pinned("a")

        cache["a"] = 2
        cache["b"] = 3  # "a" unpinned again -> evictable
        assert "a" not in cache
        assert len(cache) == 1


class TestMappingSurface:
    def test_getitem_missing_raises_keyerror(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        with pytest.raises(KeyError):
            _ = cache["nope"]

    def test_get_returns_default_when_absent(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        assert cache.get("nope") is None
        assert cache.get("nope", -1) == -1

    def test_pop_missing_without_default_raises(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        with pytest.raises(KeyError):
            cache.pop("nope")

    def test_pop_missing_with_default_returns_default(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        assert cache.pop("nope", None) is None
        assert cache.pop("nope", -1) == -1

    def test_len_and_iter_reflect_contents(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache["a"] = 1
        cache["b"] = 2
        assert len(cache) == 2
        assert set(cache) == {"a", "b"}
        assert cache.keys() == ["a", "b"]

    def test_iter_snapshot_tolerates_mutation(self):
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache["a"] = 1
        cache["b"] = 2
        # Iterating and mutating must not raise (snapshot semantics).
        for key in cache:
            if key == "a":
                cache.pop("b", None)
        assert "b" not in cache

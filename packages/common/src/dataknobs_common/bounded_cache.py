"""A small access-ordered LRU cache with an optional size bound.

A long-running process that caches per-key state (a chatbot caching one
``ConversationManager`` per conversation, a service caching one session
per client) needs that cache to stay *bounded* — otherwise it grows once
per distinct key for the lifetime of the process. The recurring shape is:
an access-ordered LRU map that evicts the least-recently-used entry when a
maximum size is exceeded, runs a caller hook on each eviction (to co-drop
any satellite state keyed by the same key), and can *pin* an entry so an
in-flight operation is never evicted out from under itself.

``BoundedLRUCache`` is that primitive:

* **Access-ordered.** Every read (``cache[key]`` / ``cache.get(key)``) and
  every write marks the entry most-recently-used, so eviction always
  targets genuinely cold entries.
* **Optionally bounded.** ``max_size=None`` (the default) never evicts —
  a pure opt-out that preserves unbounded behavior for a single-user
  embedded deployment that should never expire its one live entry. A
  positive ``max_size`` evicts the LRU entry at each write that would
  exceed the bound.
* **Eviction hook.** ``on_evict(key, value)`` fires exactly once per
  *automatic* eviction, with the evicted pair. Manual removal
  (``pop`` / ``del``) does **not** fire it — the caller is already doing
  the removal and can co-drop satellite state itself. The hook must **not**
  mutate the cache (insert/evict re-entrantly); it is a co-drop notifier,
  not a place to grow the cache being drained.
* **Refcounted pinning.** ``pin(key)`` / ``unpin(key)`` protect an entry
  from eviction. Pins are counted so concurrent operations on the same
  key each hold their own pin and one finishing does not unpin the other.
  Eviction targets the least-recently-used *unpinned* entry and never the
  most-recently-used entry — so a write never evicts the entry it just
  inserted. If every eligible entry is pinned (or is the just-inserted
  MRU entry), the cache is transiently allowed to exceed ``max_size``
  rather than evict an in-flight entry — the bound is a target and
  in-flight correctness wins. Removing an entry (``pop`` / ``del`` /
  ``clear``) reclaims its pin bookkeeping too, so a pin can never outlive
  the entry it protected and silently guard a later value reusing the key.

**Concurrency.** Like the per-conversation dicts it replaces, the cache is
lock-free and assumes a single-threaded event-loop caller: all mutation
(and thus eviction) happens synchronously between awaits. It is not safe
to share across OS threads without external synchronization.

DynaBot is the first adopter — its ``_conversation_managers`` cache
becomes a ``BoundedLRUCache[str, ConversationManager]`` whose ``on_evict``
co-drops the conversation's undo checkpoints, with the in-flight turn
pinning its conversation id.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar, overload

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")

# Sentinel distinguishing "no default given" from an explicit ``default=None``
# in ``pop`` (so ``pop(missing_key)`` raises like ``dict.pop`` does).
_MISSING: Any = object()


class BoundedLRUCache(Generic[K, V]):
    """Access-ordered LRU cache bounded by an optional maximum size.

    Args:
        max_size: Maximum number of entries to retain. ``None`` (default)
            disables eviction entirely (unbounded). Must be ``>= 1`` when
            set.
        on_evict: Optional ``(key, value) -> None`` hook fired once per
            automatic eviction with the evicted pair. Not fired by manual
            removal (``pop`` / ``del`` / ``clear``).

    Example:
        ```python
        dropped: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=2, on_evict=lambda k, v: dropped.append((k, v))
        )
        cache["a"] = 1
        cache["b"] = 2
        _ = cache["a"]        # touch "a" -> "b" is now LRU
        cache["c"] = 3        # evicts "b"
        assert dropped == [("b", 2)]
        assert "b" not in cache
        ```
    """

    def __init__(
        self,
        *,
        max_size: int | None = None,
        on_evict: Callable[[K, V], None] | None = None,
    ) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError(f"max_size must be >= 1 or None, got {max_size!r}")
        self._max_size = max_size
        self._on_evict = on_evict
        self._data: OrderedDict[K, V] = OrderedDict()
        # key -> pin refcount (>0 means protected from eviction)
        self._pins: dict[K, int] = {}

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    @property
    def max_size(self) -> int | None:
        """The configured size bound (``None`` = unbounded)."""
        return self._max_size

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        # Membership is a pure predicate — it does NOT touch recency.
        return key in self._data

    def __iter__(self) -> Iterator[K]:
        # Snapshot so a caller may mutate the cache while iterating.
        return iter(list(self._data))

    def keys(self) -> list[K]:
        """Keys in least- to most-recently-used order (a snapshot)."""
        return list(self._data)

    def is_pinned(self, key: K) -> bool:
        """Whether ``key`` currently holds one or more pins."""
        return self._pins.get(key, 0) > 0

    # ------------------------------------------------------------------ #
    # Reads (touch most-recently-used)                                   #
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: K) -> V:
        value = self._data[key]  # raises KeyError if absent
        self._data.move_to_end(key)
        return value

    def get(self, key: K, default: T | None = None) -> V | T | None:
        """Return the value for ``key`` (touching it MRU) or ``default``."""
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return default

    # ------------------------------------------------------------------ #
    # Writes                                                             #
    # ------------------------------------------------------------------ #
    def __setitem__(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        self._evict_if_needed()

    @overload
    def pop(self, key: K) -> V: ...
    @overload
    def pop(self, key: K, default: V) -> V: ...
    @overload
    def pop(self, key: K, default: T) -> V | T: ...
    def pop(self, key: K, default: Any = _MISSING) -> Any:
        """Remove ``key`` and return its value; ``on_evict`` is NOT fired.

        Mirrors ``dict.pop``: raises ``KeyError`` when ``key`` is absent
        and no ``default`` was supplied. Manual removal is the caller's own
        teardown, so the eviction hook deliberately does not run; the key's
        pin bookkeeping is reclaimed so a stale pin can't outlive the entry.
        """
        if key in self._data:
            self._pins.pop(key, None)
            return self._data.pop(key)
        if default is _MISSING:
            raise KeyError(key)
        return default

    def __delitem__(self, key: K) -> None:
        # Manual removal — does not fire ``on_evict``; reclaims any pin so
        # it can't outlive the entry and protect a later value on this key.
        del self._data[key]
        self._pins.pop(key, None)

    def clear(self) -> None:
        """Drop all entries and pins without firing ``on_evict``."""
        self._data.clear()
        self._pins.clear()

    # ------------------------------------------------------------------ #
    # Pinning                                                            #
    # ------------------------------------------------------------------ #
    def pin(self, key: K) -> None:
        """Protect ``key`` from eviction (refcounted; nestable)."""
        self._pins[key] = self._pins.get(key, 0) + 1

    def unpin(self, key: K) -> None:
        """Release one pin on ``key``. Idempotent when ``key`` is unpinned."""
        count = self._pins.get(key, 0)
        if count <= 1:
            self._pins.pop(key, None)
        else:
            self._pins[key] = count - 1

    # ------------------------------------------------------------------ #
    # Eviction                                                           #
    # ------------------------------------------------------------------ #
    def _evict_if_needed(self) -> None:
        if self._max_size is None:
            return
        while len(self._data) > self._max_size:
            if not self._evict_one_unpinned():
                # Every over-limit entry is pinned: allow transient overflow
                # rather than evict an in-flight (pinned) entry.
                break

    def _evict_one_unpinned(self) -> bool:
        """Evict the LRU unpinned entry, firing ``on_evict``. False if none.

        The most-recently-used entry (the tail) is never a candidate: a
        write makes its key MRU, so this guarantees a write never evicts the
        entry it just inserted. When every other entry is pinned there is no
        eligible victim and the caller allows a transient overflow.
        """
        if not self._data:
            return False
        mru_key = next(reversed(self._data))
        for key in list(self._data):  # least- to most-recently-used
            if key == mru_key:
                continue  # never evict the most-recently-used entry
            if self._pins.get(key, 0) == 0:
                value = self._data.pop(key)
                if self._on_evict is not None:
                    self._on_evict(key, value)
                return True
        return False

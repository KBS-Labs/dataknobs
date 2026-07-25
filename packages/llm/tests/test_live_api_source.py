"""Unit tests for the generic ``LiveApiSource`` model-metadata source.

``LiveApiSource`` lifts the live Models-API ceiling cache (formerly pinned in
``anthropic.py`` as ``_MODEL_LIMITS_CACHE`` / ``_CeilingEntry`` /
``_refresh_model_limits`` module globals) into a reusable
:class:`~dataknobs_llm.llm.model_profile.ModelMetadataSource` any provider
serving live model metadata can compose. These tests exercise the class
directly — provider-agnostic — with a scripted ``list_models`` and a trivial
ceiling extractor (no ``anthropic`` package, no live API).

They pin the three refresh properties the lift carries verbatim (TTL gating,
per-loop-locked dedup, source-aware non-degradation), the per-loop weak-keying
+ dead-loop eviction (which moved here from ``test_anthropic_model_constraints``
with the cache), and the per-facet family-alias ``resolve``.
"""

from __future__ import annotations

import asyncio
import gc
import time
import weakref
from typing import Any

from dataknobs_llm.llm.model_profile import (
    LiveApiSource,
    ModelMetadataSource,
    ModelProfile,
)


# ---------------------------------------------------------------------------
# Scripted list_models + a trivial ceiling extractor
# ---------------------------------------------------------------------------


class _Model:
    """Minimal live-API model object: ``id`` + two optional ceiling columns."""

    def __init__(
        self, model_id: str, out: int | None = None, inp: int | None = None
    ) -> None:
        self.id = model_id
        self.out = out
        self.inp = inp


def _extractor(model_obj: _Model) -> ModelProfile:
    """Project a ``_Model`` into a two-ceiling partial profile."""
    return ModelProfile(
        max_output_tokens=model_obj.out, context_window=model_obj.inp
    )


def _lister(
    models: list[_Model],
    *,
    calls: list[int] | None = None,
    raises: bool = False,
    delay: float = 0.0,
) -> Any:
    """Build an async ``list_models`` returning *models* (tracks/fails/hangs)."""

    async def _list() -> list[_Model]:
        if calls is not None:
            calls.append(1)
        if raises:
            raise RuntimeError("simulated Models API failure")
        if delay:
            await asyncio.sleep(delay)
        return list(models)

    return _list


def _source(models: list[_Model], **kwargs: Any) -> tuple[LiveApiSource, list[int]]:
    """A ``LiveApiSource`` over *models* + a call-count list for ``list_models``."""
    calls: list[int] = []
    kwargs.setdefault("ttl", 3600.0)
    src = LiveApiSource(_lister(models, calls=calls), _extractor, **kwargs)
    return src, calls


# ---------------------------------------------------------------------------
# Structural conformance + resolve (read path)
# ---------------------------------------------------------------------------


class TestResolve:
    """``resolve`` is a per-facet family-alias cache read (sync, I/O-free)."""

    def test_is_model_metadata_source(self) -> None:
        src, _ = _source([])
        assert isinstance(src, ModelMetadataSource)

    def test_empty_cache_resolves_all_none(self) -> None:
        src, _ = _source([])
        profile = src.resolve("claude-sonnet-5")
        assert profile.max_output_tokens is None
        assert profile.context_window is None

    async def test_refresh_populates_then_resolves(self) -> None:
        src, calls = _source([_Model("claude-sonnet-5", out=128000, inp=1_000_000)])
        await src.force_refresh()
        profile = src.resolve("claude-sonnet-5")
        assert profile.max_output_tokens == 128000
        assert profile.context_window == 1_000_000
        assert len(calls) == 1

    async def test_bare_alias_resolves_dated_cache_key(self) -> None:
        """A bare-alias request resolves against a dated cache key."""
        src, _ = _source([_Model("claude-sonnet-5-20260930", out=200000)])
        await src.force_refresh()
        assert src.resolve("claude-sonnet-5").max_output_tokens == 200000

    async def test_dated_request_resolves_family_cache_key(self) -> None:
        """A dated request resolves against a bare family cache key."""
        src, _ = _source([_Model("claude-sonnet-5", out=200000)])
        await src.force_refresh()
        assert (
            src.resolve("claude-sonnet-5-20261231").max_output_tokens == 200000
        )

    async def test_per_facet_input_only_entry(self) -> None:
        """A model reporting only its input window contributes input, not output.

        The per-facet resolution: the output facet stays ``None`` (no fabricated
        clamp) while the input facet resolves from the same entry.
        """
        src, _ = _source([_Model("claude-sonnet-5", out=None, inp=200000)])
        await src.force_refresh()
        profile = src.resolve("claude-sonnet-5")
        assert profile.max_output_tokens is None
        assert profile.context_window == 200000

    async def test_no_facet_model_is_not_cached(self) -> None:
        """A model reporting neither ceiling never enters the cache."""
        src, _ = _source([_Model("claude-empty", out=None, inp=None)])
        await src.force_refresh()
        assert src.resolve("claude-empty").max_output_tokens is None
        assert src._cache == {}

    async def test_unrelated_cache_key_does_not_match(self) -> None:
        src, _ = _source([_Model("claude-zzz-unrelated-9", out=111)])
        await src.force_refresh()
        assert src.resolve("claude-opus-5").max_output_tokens is None

    def test_seed_helper_populates_cache(self) -> None:
        src, _ = _source([])
        src.seed("claude-sonnet-5", ModelProfile(max_output_tokens=64000))
        assert src.resolve("claude-sonnet-5").max_output_tokens == 64000


# ---------------------------------------------------------------------------
# TTL gating
# ---------------------------------------------------------------------------


class TestTTL:
    """The refresh is TTL-gated per loop — a fresh cache is a no-op."""

    async def test_cold_cache_is_stale(self) -> None:
        src, _ = _source([])
        assert src.is_stale() is True

    async def test_fresh_after_refresh(self) -> None:
        src, _ = _source([_Model("m", out=1)])
        await src.refresh_if_stale()
        assert src.is_stale() is False

    async def test_long_ttl_refreshes_once(self) -> None:
        src, calls = _source([_Model("m", out=1)], ttl=3600.0)
        await src.refresh_if_stale()
        await src.refresh_if_stale()
        assert len(calls) == 1

    async def test_ttl_zero_refreshes_each_call(self) -> None:
        src, calls = _source([_Model("m", out=1)], ttl=0.0)
        await src.refresh_if_stale()
        await src.refresh_if_stale()
        assert len(calls) == 2

    async def test_force_refresh_bypasses_ttl(self) -> None:
        src, calls = _source([_Model("m", out=1)], ttl=3600.0)
        await src.refresh_if_stale()
        await src.force_refresh()
        assert len(calls) == 2

    async def test_disabled_never_polls(self) -> None:
        src, calls = _source([_Model("m", out=1)], enabled=False)
        await src.refresh_if_stale()
        await src.force_refresh()
        assert len(calls) == 0
        assert src.resolve("m").max_output_tokens is None


# ---------------------------------------------------------------------------
# Per-loop lock: concurrent refresh dedup
# ---------------------------------------------------------------------------


class TestRefreshDedup:
    """Concurrent callers on a cold cache coalesce into one poll."""

    async def test_concurrent_refresh_if_stale_dedups(self) -> None:
        src, calls = _source([_Model("m", out=1)])
        await asyncio.gather(*[src.refresh_if_stale() for _ in range(8)])
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Source-aware non-degradation
# ---------------------------------------------------------------------------


class TestNonDegradation:
    """A failed refresh leaves a known-good live value intact (never dropped)."""

    async def test_failed_refresh_leaves_cache_intact(self) -> None:
        src = LiveApiSource(
            _lister([_Model("claude-sonnet-5", out=200000)]),
            _extractor,
            ttl=0.0,
        )
        await src.force_refresh()
        assert src.resolve("claude-sonnet-5").max_output_tokens == 200000
        # Swap the lister for one that fails, then refresh — value must persist.
        src._list_models = _lister([], raises=True)  # type: ignore[assignment]
        await src.force_refresh()
        assert src.resolve("claude-sonnet-5").max_output_tokens == 200000

    async def test_failed_cold_refresh_is_permissive(self) -> None:
        src = LiveApiSource(_lister([], raises=True), _extractor)
        await src.force_refresh()  # swallows the error
        assert src.resolve("claude-sonnet-5").max_output_tokens is None

    async def test_outage_bounded_to_one_attempt_per_ttl(self) -> None:
        """The timer re-arms before the poll, so an outage cannot busy-retry."""
        calls: list[int] = []
        src = LiveApiSource(
            _lister([], calls=calls, raises=True), _extractor, ttl=3600.0
        )
        await src.refresh_if_stale()
        await src.refresh_if_stale()
        assert len(calls) == 1  # second call sees a fresh (armed) timer


# ---------------------------------------------------------------------------
# Refresh timeout bound
# ---------------------------------------------------------------------------


class TestRefreshTimeout:
    """A *hung* list_models is bounded by ``refresh_timeout``."""

    async def test_hung_poll_bounded(self) -> None:
        src = LiveApiSource(
            _lister([_Model("m", out=1)], delay=2.0),
            _extractor,
            refresh_timeout=0.05,
        )
        start = time.monotonic()
        await src.force_refresh()  # best-effort — never raises
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
        # The poll was abandoned; nothing cached.
        assert src.resolve("m").max_output_tokens is None


# ---------------------------------------------------------------------------
# Per-instance cache isolation (module-global -> per-instance behavior change)
# ---------------------------------------------------------------------------


class TestInstanceIsolation:
    """Each ``LiveApiSource`` owns its cache — no cross-instance leakage.

    The lift moved the ceiling cache from an ``anthropic.py`` module global to a
    per-instance attribute (``self._cache``), so two providers on distinct
    accounts no longer share ceiling entries keyed only by model id — the
    correctness improvement the CHANGELOG claims. This pins that claim as a
    regression guard: a future refactor reintroducing a shared (class- or
    module-level) cache would make ``other`` observe ``populated``'s entry and
    fail here.
    """

    async def test_refresh_does_not_leak_across_instances(self) -> None:
        populated, _ = _source([_Model("claude-sonnet-5", out=128000)])
        other, other_calls = _source([])  # a second "account" that lists nothing

        await populated.force_refresh()

        # The populated instance resolves its live ceiling...
        assert populated.resolve("claude-sonnet-5").max_output_tokens == 128000
        # ...but the second instance's cache is untouched (no shared state) and
        # it never polled.
        assert other.resolve("claude-sonnet-5").max_output_tokens is None
        assert other._cache == {}
        assert other_calls == []

    def test_seed_is_isolated_per_instance(self) -> None:
        a, _ = _source([])
        b, _ = _source([])
        a.seed("claude-opus-5", ModelProfile(max_output_tokens=64000))
        assert a.resolve("claude-opus-5").max_output_tokens == 64000
        assert b.resolve("claude-opus-5").max_output_tokens is None


# ---------------------------------------------------------------------------
# Per-loop state weak-keying (moved from test_anthropic_model_constraints)
# ---------------------------------------------------------------------------


class TestPerLoopStateKeying:
    """Per-loop refresh state is keyed by the loop *object*, not ``id(loop)``.

    ``id(loop)`` in a plain dict never evicts a dead loop's entry (a leak) and,
    worse, lets a *new* loop that reuses a freed id inherit the dead loop's
    stale last-fetch timestamp and wrongly skip a needed refresh. A
    ``WeakKeyDictionary`` keyed on the loop object evicts the entry when the loop
    is collected and makes every new loop a distinct key. The mechanism now lives
    on :class:`LiveApiSource` (it moved here with the cache from the provider).
    """

    def test_state_is_weak_keyed(self) -> None:
        src, _ = _source([])
        assert isinstance(src._last_fetch, weakref.WeakKeyDictionary)
        assert isinstance(src._locks, weakref.WeakKeyDictionary)

    def test_dead_loop_entry_is_evicted(self) -> None:
        src, _ = _source([_Model("claude-sonnet-5", out=200000)])

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(src.force_refresh())
        finally:
            loop.close()

        # The refresh recorded per-loop state keyed by the loop object.
        assert len(src._last_fetch) == 1

        del loop
        gc.collect()

        # WeakKeyDictionary drops the dead loop's entry — no leak, and no stale
        # timestamp a future (id-reused) loop could inherit. A plain
        # ``dict[int, float]`` (the pre-lift keying) would still hold it.
        assert len(src._last_fetch) == 0

    def test_clear_drops_cache_and_state(self) -> None:
        src, _ = _source([_Model("m", out=1)])
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(src.force_refresh())
        finally:
            loop.close()
        assert src._cache and len(src._last_fetch) == 1
        src.clear()
        assert src._cache == {}
        assert len(src._last_fetch) == 0
        assert len(src._locks) == 0

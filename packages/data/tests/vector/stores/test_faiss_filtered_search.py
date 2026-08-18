"""``FaissVectorStore.search`` under a metadata filter.

FAISS retrieves a top-``k`` window from its index and then drops the
rows that do not match the filter, so a filtered search used to return
only the matching rows that happened to fall inside that window —
frequently none of them. The store reported the full matching count
from ``count(filter=...)`` the whole time, so nothing about a populated
store looked wrong.

The condition needs no multi-domain configuration to reach: any
caller-supplied ``filter=`` whose matches sit outside the unfiltered
top-``k`` hits it. A store configured with a ``domain_id`` merely fires
it on every call, because the scope is AND-merged into every filter.

These tests fix the behavior at "a filtered search returns ``k`` rows
whenever ``k`` rows match, and returns them in the same order and with
the same scores an unfiltered search would give them" — which is what
``MemoryVectorStore`` and ``PgVectorStore`` already do, and what the
cross-backend semantics doc already promises.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dataknobs_common.testing import is_faiss_available

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

from dataknobs_data.vector.stores.memory import MemoryVectorStore

requires_faiss = pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed")

pytestmark = [pytest.mark.asyncio, requires_faiss]

FAISS_LOGGER = "dataknobs_data.vector.stores.faiss"

DIMENSIONS = 8
NEAR = 3
FAR = 200


def _fan(offsets: list[float]) -> np.ndarray:
    """Rows ``[1, t, 0, ...]`` for each ``t``, fanning off the probe axis.

    Similarity to :func:`_probe` falls monotonically as ``t`` grows, for
    every metric these stores support, so "nearest" is a property of the
    corpus rather than of the index type under test.
    """
    rows = np.zeros((len(offsets), DIMENSIONS), dtype=np.float32)
    rows[:, 0] = 1.0
    rows[:, 1] = np.asarray(offsets, dtype=np.float32)
    return rows


def _probe() -> np.ndarray:
    """The query every test searches with."""
    return _fan([0.0])[0]


def _scoped_corpus() -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """``FAR`` rows in domain ``a`` behind ``NEAR`` rows in domain ``b``.

    Domain ``b`` owns the whole unfiltered top-3 while holding three
    rows to domain ``a``'s two hundred, which is the shape a small
    co-tenant produces in a shared index.
    """
    offsets = [0.001 * (i + 1) for i in range(NEAR)]
    offsets += [0.01 * (i + 1) for i in range(FAR)]
    ids = [f"b{i}" for i in range(NEAR)] + [f"a{i}" for i in range(FAR)]
    metadata: list[dict[str, Any]] = [{"domain_id": "b"} for _ in range(NEAR)]
    metadata += [{"domain_id": "a"} for _ in range(FAR)]
    return _fan(offsets), ids, metadata


async def _seeded(config: dict[str, Any]) -> Any:
    """A FAISS store holding :func:`_scoped_corpus`."""
    store = FaissVectorStore({"dimensions": DIMENSIONS, "metric": "cosine", **config})
    await store.initialize()
    vectors, ids, metadata = _scoped_corpus()
    await store.add_vectors(vectors, ids=ids, metadata=metadata)
    return store


@pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivfflat"])
async def test_filtered_search_returns_k_when_k_rows_match(index_type: str) -> None:
    """``k`` matching rows in, ``k`` rows out — on every index type.

    The parametrization is the point of the test, not extra coverage of
    one behavior. Compensating for a post-filter by over-fetching, or by
    handing FAISS an id selector, both give the right answer on ``flat``
    and the wrong one on the approximate index types: a filtered graph
    traversal cannot reach nodes the graph does not route to, and an IVF
    selector only applies inside the probed lists. Scoring the matching
    subset directly is exact for all three.
    """
    store = await _seeded({"index_type": index_type})
    try:
        results = await store.search(_probe(), k=3, filter={"domain_id": "a"})

        assert len(results) == 3
        assert [r[0] for r in results] == ["a0", "a1", "a2"]
    finally:
        await store.close()


async def test_domain_scoped_search_matches_memory_backend() -> None:
    """A ``domain_id``-scoped FAISS store answers as the memory store does.

    The published cross-backend contract says a config-scoped store
    behaves identically under ``search`` whichever backend is bound.
    This pins that executably for the case that used to diverge: the
    scope is AND-merged into every search, so every search on such a
    store is a filtered search.
    """
    vectors, ids, metadata = _scoped_corpus()
    faiss_store = FaissVectorStore({"dimensions": DIMENSIONS, "metric": "cosine", "domain_id": "a"})
    memory_store = MemoryVectorStore(
        {"dimensions": DIMENSIONS, "metric": "cosine", "domain_id": "a"}
    )
    try:
        for store in (faiss_store, memory_store):
            await store.initialize()
            await store.add_vectors(vectors, ids=list(ids), metadata=[dict(m) for m in metadata])

        from_faiss = await faiss_store.search(_probe(), k=5)
        from_memory = await memory_store.search(_probe(), k=5)

        assert [r[0] for r in from_faiss] == [r[0] for r in from_memory]
        assert len(from_faiss) == 5
    finally:
        await faiss_store.close()
        await memory_store.close()


async def test_plain_metadata_filter_not_only_domain_scope() -> None:
    """An ordinary ``filter=`` on an unscoped store has the same hole.

    Nothing here configures a domain. One row carries the filtered
    value and sits behind fifty that do not, which is enough: ``count``
    finds it and ``search`` used not to.
    """
    offsets = [0.01 * (i + 1) for i in range(50)] + [5.0]
    ids = [f"other{i}" for i in range(50)] + ["wanted"]
    metadata: list[dict[str, Any]] = [{"source": "other.md"} for _ in range(50)]
    metadata += [{"source": "handbook.md"}]

    store = FaissVectorStore({"dimensions": DIMENSIONS, "metric": "cosine"})
    await store.initialize()
    try:
        await store.add_vectors(_fan(offsets), ids=ids, metadata=metadata)

        assert await store.count(filter={"source": "handbook.md"}) == 1

        results = await store.search(_probe(), k=3, filter={"source": "handbook.md"})
        assert [r[0] for r in results] == ["wanted"]
    finally:
        await store.close()


async def test_filtered_search_never_exceeds_k() -> None:
    """Whatever the store fetches internally, the caller gets at most ``k``."""
    store = await _seeded({})
    try:
        results = await store.search(_probe(), k=5, filter={"domain_id": "a"})
        assert len(results) == 5

        # More rows match than the index holds room to over-fetch for.
        everything = await store.search(_probe(), k=FAR + NEAR, filter={"domain_id": "a"})
        assert len(everything) == FAR
    finally:
        await store.close()


async def test_filtered_search_ordering_and_scores_match_unfiltered() -> None:
    """A filtered search reports the ranking an unfiltered one would.

    Both the order and the score values are asserted: the filtered path
    computes similarity itself rather than reading it off the index, so
    a second, divergent score conversion would be invisible to an
    order-only assertion.
    """
    store = await _seeded({"index_type": "flat"})
    try:
        unfiltered = await store.search(_probe(), k=FAR + NEAR)
        expected = [(r[0], r[1]) for r in unfiltered if r[0].startswith("a")][:5]

        filtered = await store.search(_probe(), k=5, filter={"domain_id": "a"})

        assert [r[0] for r in filtered] == [ext_id for ext_id, _ in expected]
        assert [r[1] for r in filtered] == pytest.approx([score for _, score in expected], rel=1e-5)
        assert [r[1] for r in filtered] == sorted((r[1] for r in filtered), reverse=True)
    finally:
        await store.close()


async def test_empty_vector_sidecar_falls_back_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A pickle written before the vector side-car existed still answers.

    ``_load_from_disk`` reads the side-car with ``.get("vectors", {})``,
    so such a store comes up with a populated index and no stored rows.
    Scoring the matching subset would return nothing at all there, so
    the filtered path falls back to searching the index and says so —
    the rows are recoverable by re-ingesting, and silence would leave a
    consumer reading a shortfall as an empty result set.
    """
    persist = tmp_path / "faiss_legacy.index"
    config = {
        "dimensions": DIMENSIONS,
        "metric": "cosine",
        "persist_path": str(persist),
    }
    store = await _seeded(config)
    await store.close()  # triggers save()

    meta_path = str(persist) + ".meta"
    with open(meta_path, "rb") as fh:
        legacy = pickle.load(fh)
    legacy.pop("vectors", None)
    with open(meta_path, "wb") as fh:
        pickle.dump(legacy, fh)

    reopened = FaissVectorStore(config)
    await reopened.initialize()
    try:
        assert reopened.vectors == {}
        assert reopened.index.ntotal == FAR + NEAR

        with caplog.at_level(logging.WARNING, logger=FAISS_LOGGER):
            results = await reopened.search(_probe(), k=3, filter={"domain_id": "a"})

        assert [r[0] for r in results] == ["a0", "a1", "a2"]
        warnings = [r for r in caplog.records if r.name == FAISS_LOGGER]
        assert len(warnings) == 1
        assert "re-ingest" in warnings[0].getMessage().lower()
    finally:
        await reopened.close()


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
async def test_partial_sidecar_still_ranks_the_rows_it_cannot_score(metric: str) -> None:
    """A row missing from the side-car is ranked from the index, not dropped.

    Answering only from the rows the side-car holds would silently omit
    ``a0`` and ``a1`` here — the two best matches — and return ``a2``
    onward as though they were the top of the corpus. Answering only from
    the index would give up the exact scoring of the 198 rows that need
    no help. Both sources are used, merged on the raw metric value each
    produces, which is the same scale precisely because
    ``_raw_index_scores`` reproduces what the index returns.

    Parametrized over the two sort *directions*, not for extra coverage
    of one behavior: the merge negates before sorting under inner
    product (high wins) and does not under L2 (low wins). A test on one
    metric leaves the other branch of that decision unexecuted, and
    ``_fan`` orders the corpus the same way under both, so a broken
    direction shows up as a reversed result rather than as noise.
    """
    store = await _seeded({"index_type": "flat", "metric": metric})
    try:
        exact = await store.search(_probe(), k=3, filter={"domain_id": "a"})

        # Drop the two best matches from the side-car, leaving them in
        # the index — the desync shape a partial side-car produces.
        for ext_id in ("a0", "a1"):
            del store.vectors[store.id_map[ext_id]]

        partial = await store.search(_probe(), k=3, filter={"domain_id": "a"})

        assert [r[0] for r in partial] == ["a0", "a1", "a2"]
        assert [r[0] for r in partial] == [r[0] for r in exact]
        assert [r[1] for r in partial] == pytest.approx([r[1] for r in exact], rel=1e-5)
    finally:
        await store.close()


async def test_sidecar_shortfall_is_reported_once_not_per_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The shortfall belongs to the store, so it is said once.

    The condition is a property of the loaded file: every filtered search
    this instance serves meets it. Warning per call puts one line per
    user turn into the log of a RAG read path, all of them naming the
    same one-off remedy.
    """
    store = await _seeded({"index_type": "flat"})
    try:
        store.vectors.clear()

        with caplog.at_level(logging.WARNING, logger=FAISS_LOGGER):
            for _ in range(5):
                await store.search(_probe(), k=3, filter={"domain_id": "a"})

        warnings = [r for r in caplog.records if r.name == FAISS_LOGGER]
        assert len(warnings) == 1
        assert "re-ingest" in warnings[0].getMessage().lower()
    finally:
        await store.close()


@pytest.mark.parametrize("k", [0, -1])
async def test_non_positive_k_returns_empty_on_both_paths(k: int) -> None:
    """``k <= 0`` is answered the same way whether or not a filter is set.

    The filtered path returned ``[]`` for a negative ``k`` while the
    unfiltered one handed it to ``index.search``.
    """
    store = await _seeded({"index_type": "flat"})
    try:
        assert await store.search(_probe(), k=k) == []
        assert await store.search(_probe(), k=k, filter={"domain_id": "a"}) == []
    finally:
        await store.close()


async def test_unfiltered_search_path_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    """An unfiltered search still answers from the index alone.

    Emptying the side-car leaves the index intact, so a search that
    still reads its candidates from the index is unaffected — and one
    that had quietly started scoring the side-car for every call would
    return nothing. No fallback warning is emitted either: the side-car
    is not on the unfiltered path to be missing from.
    """
    store = await _seeded({"index_type": "flat"})
    try:
        store.vectors.clear()

        with caplog.at_level(logging.WARNING, logger=FAISS_LOGGER):
            results = await store.search(_probe(), k=3)

        assert [r[0] for r in results] == ["b0", "b1", "b2"]
        assert [r.name for r in caplog.records] == []
    finally:
        await store.close()

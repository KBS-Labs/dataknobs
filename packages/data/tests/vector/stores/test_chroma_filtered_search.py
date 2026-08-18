"""``ChromaVectorStore.search`` under a residual post-filter.

Chroma truncates to ``n_results`` before this store's Python-side filter
runs, so every candidate that filter drops is a row the caller asked for
and did not get. Over-fetching ``k * POST_FILTER_OVERFETCH`` compensates
for the common case and no more: a filter matching fewer than one
candidate in ``POST_FILTER_OVERFETCH`` still under-returned, and could
return nothing at all while ``count(filter=...)`` reported many matches.

That is the same "count says N, search says zero" shape the FAISS
filtered path was fixed for, at a wider window rather than a different
mechanism — which is why the fix is the shared over-fetch policy
escalating to a real ceiling rather than a bigger constant.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pytest
from dataknobs_common.testing import is_chromadb_available

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

from dataknobs_data.vector.stores.common import POST_FILTER_OVERFETCH
from dataknobs_data.vector.types import DistanceMetric

requires_chromadb = pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed")

pytestmark = [pytest.mark.asyncio, requires_chromadb]

DIMENSIONS = 4
K = 3
# Far more decoys than the single over-fetched window covers, so the
# matches cannot be reached without escalating.
DECOYS = K * POST_FILTER_OVERFETCH * 8
TARGETS = 5


def _probe() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _corpus() -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Matches so sparse that the first over-fetch reaches none of them."""
    offsets = [0.001 * (i + 1) for i in range(DECOYS)]
    offsets += [0.5 + 0.1 * i for i in range(TARGETS)]
    vectors = np.array([[1.0, t, 0.0, 0.0] for t in offsets], dtype=np.float32)
    ids = [f"other{i}" for i in range(DECOYS)] + [f"target{i}" for i in range(TARGETS)]
    metadata: list[dict[str, Any]] = [{"group": "other"} for _ in range(DECOYS)]
    metadata += [{"group": "target"} for _ in range(TARGETS)]
    return vectors, ids, metadata


async def _seeded(**config: Any) -> Any:
    store = ChromaVectorStore(
        {
            "dimensions": DIMENSIONS,
            "collection_name": f"test_filtered_{uuid.uuid4().hex[:8]}",
            **config,
        }
    )
    await store.initialize()
    vectors, ids, metadata = _corpus()
    await store.add_vectors(vectors, ids=ids, metadata=metadata)
    return store


async def _drop(store: Any) -> None:
    await store.clear()
    await store.close()


async def test_diluted_post_filter_still_returns_k() -> None:
    """``k`` matching rows in, ``k`` rows out, however sparse the matches.

    ``"group"`` is not declared scalar, so this filter is a residual
    Python post-filter. Every match sits beyond the first over-fetched
    window, which is exactly the case a fixed multiplier cannot cover.
    """
    store = await _seeded()
    try:
        assert await store.count(filter={"group": "target"}) == TARGETS

        results = await store.search(_probe(), k=K, filter={"group": "target"})

        assert len(results) == K, (
            f"store holds {TARGETS} matching rows; search(k={K}) returned "
            f"{len(results)}: {[r[0] for r in results]}"
        )
        assert [r[0] for r in results] == ["target0", "target1", "target2"]
    finally:
        await _drop(store)


async def test_search_never_exceeds_k_and_can_return_every_match() -> None:
    """Escalation stops at the corpus, and the caller still gets at most ``k``."""
    store = await _seeded()
    try:
        assert len(await store.search(_probe(), k=1, filter={"group": "target"})) == 1

        everything = await store.search(_probe(), k=DECOYS + TARGETS, filter={"group": "target"})
        assert [r[0] for r in everything] == [f"target{i}" for i in range(TARGETS)]
    finally:
        await _drop(store)


async def test_pushed_down_filter_is_unaffected() -> None:
    """A declared-scalar key needs no compensation and must not regress.

    With the key in ``scalar_metadata_keys`` the predicate becomes a
    native ``where``, there is no residual filter, and the escalation
    path is skipped entirely — including the ``count()`` it would
    otherwise take.
    """
    store = await _seeded(scalar_metadata_keys=["group"])
    try:
        results = await store.search(_probe(), k=K, filter={"group": "target"})
        assert [r[0] for r in results] == ["target0", "target1", "target2"]
    finally:
        await _drop(store)


async def test_unsatisfiable_and_empty_filters_are_unchanged() -> None:
    """The escalation must not disturb the two degenerate filter shapes."""
    store = await _seeded()
    try:
        assert await store.search(_probe(), k=K, filter={"group": []}) == []
        # An empty filter drops nothing, so it is not a post-filter.
        assert len(await store.search(_probe(), k=K, filter={})) == K
    finally:
        await _drop(store)


async def test_include_timestamps_is_accepted_and_answers_none() -> None:
    """The argument is on the ABC, so a swap must not break on it.

    Chroma tracks no timestamps, so both keys come back ``None`` — the
    answer the contract already defines for a row with none tracked.
    Before, both methods raised ``TypeError`` on the keyword while every
    other backend accepted it, which broke exactly the runtime backend
    swap the semantics doc promises.
    """
    store = await _seeded()
    try:
        results = await store.search(_probe(), k=1, include_timestamps=True)
        _, _, meta = results[0]
        assert meta is not None
        assert meta["_created_at"] is None
        assert meta["_updated_at"] is None
        # Consumer metadata survives alongside the injected keys.
        assert meta["group"] == "other"

        vectors = await store.get_vectors(["other0"], include_timestamps=True)
        _, fetched = vectors[0]
        assert fetched is not None
        assert fetched["_created_at"] is None
        assert fetched["_updated_at"] is None

        # Off by default, and never injected without metadata.
        plain = await store.get_vectors(["other0"])
        assert "_created_at" not in (plain[0][1] or {})
        no_meta = await store.get_vectors(
            ["other0"], include_metadata=False, include_timestamps=True
        )
        assert no_meta[0][1] is None
    finally:
        await _drop(store)


async def test_a_missing_id_yields_a_none_pair_not_a_gap() -> None:
    """``get_vectors`` stays positionally aligned with ``ids``.

    Which is why the ABC's return type is now optional in the vector
    slot: every backend already answered this way.
    """
    store = await _seeded()
    try:
        results = await store.get_vectors(["other0", "nope", "other1"])
        assert len(results) == 3
        assert results[1] == (None, None)
        assert results[0][0] is not None
        assert results[2][0] is not None
    finally:
        await _drop(store)


@pytest.mark.parametrize(
    "metric,distance,expected",
    [
        ("cosine", 0.25, 0.75),
        ("euclidean", 3.0, 0.25),
        ("l2", 3.0, 0.25),
        ("dot_product", 4.0, 4.0),
        ("inner_product", 4.0, 4.0),
    ],
)
async def test_distance_conversion_follows_the_configured_metric(
    metric: str, distance: float, expected: float
) -> None:
    """One conversion, used by both search methods.

    The collection is created with ``hnsw:space`` from the configured
    metric, so its distances are cosine, L2 or inner-product accordingly.
    ``search`` switched on the metric; ``search_documents`` applied the
    cosine formula unconditionally and so reported wrong scores — a
    negative one, for any L2 distance above 1 — on every non-cosine
    store. They now share this.

    Asserted on the conversion itself rather than through
    ``search_documents``: that method embeds its query with Chroma's own
    embedding function, which this suite will not download a model for.
    ``search``'s use of the same helper is covered by the cases above.
    """
    store = ChromaVectorStore(
        {
            "dimensions": DIMENSIONS,
            "metric": metric,
            "collection_name": f"test_metric_{uuid.uuid4().hex[:8]}",
        }
    )
    assert store._score_from_distance(distance) == pytest.approx(expected)
    # The cosine formula would have gone negative on the L2 cases, which
    # is what made the divergence a defect rather than a discrepancy.
    if store.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
        assert 1.0 - distance < 0 < store._score_from_distance(distance)

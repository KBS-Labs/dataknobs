"""ChromaVectorStore ``scalar_metadata_keys`` opt-in tests.

Item 118 review #13: pre-fix, ``ChromaVectorStore.count(filter=...)``
materialized all matching metadata in process whenever the filter
included a scalar value, because Chroma's ``$eq`` does not match
list-valued metadata and the partitioner conservatively post-filtered
all scalars in Python. Post-fix, consumers can declare always-scalar
metadata keys via ``scalar_metadata_keys`` so the partitioner pushes
``$eq`` down to Chroma natively, eliminating metadata materialization
for the common multi-tenant scoping pattern.

These tests exercise ``count`` because it is the operation most
sensitive to materialization (the result is a single int and any
materialization is purely overhead). The same partition logic also
benefits ``search`` and ``clear``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataknobs_common.testing import is_chromadb_available

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

requires_chromadb = pytest.mark.skipif(
    not is_chromadb_available(), reason="chromadb not installed"
)


def _vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(4, dtype=np.float32)


@requires_chromadb
@pytest.mark.asyncio
async def test_undeclared_scalar_key_post_filters():
    """Without ``scalar_metadata_keys``, scalar filters post-filter as before.

    Backward-compat guard: the new opt-in does not change behavior
    for keys consumers haven't declared.
    """
    store = ChromaVectorStore(
        {"dimensions": 4, "collection_name": f"test_undecl_{id(object())}"}
    )
    await store.initialize()

    await store.add_vectors(
        vectors=[_vec(0), _vec(1), _vec(2)],
        ids=["a", "b", "c"],
        metadata=[
            {"domain_id": "x", "tags": ["red"]},
            {"domain_id": "x", "tags": ["blue"]},
            {"domain_id": "y", "tags": ["red"]},
        ],
    )

    # ``domain_id`` not declared scalar → post-filtered.
    where, post = store._partition_filter_for_chroma({"domain_id": "x"})
    assert where is None, (
        f"undeclared scalar key must NOT push down to Chroma where; got {where!r}"
    )
    assert post == {"domain_id": "x"}

    # Behavior is correct regardless.
    assert await store.count(filter={"domain_id": "x"}) == 2


@requires_chromadb
@pytest.mark.asyncio
async def test_declared_scalar_key_pushes_down_eq():
    """Declared scalar keys produce a Chroma-native ``$eq`` predicate."""
    store = ChromaVectorStore(
        {
            "dimensions": 4,
            "collection_name": f"test_decl_{id(object())}",
            "scalar_metadata_keys": ["domain_id"],
        }
    )
    await store.initialize()

    await store.add_vectors(
        vectors=[_vec(0), _vec(1), _vec(2)],
        ids=["a", "b", "c"],
        metadata=[
            {"domain_id": "x"},
            {"domain_id": "x"},
            {"domain_id": "y"},
        ],
    )

    # Push-down: native ``$eq``, no post-filter remainder.
    where, post = store._partition_filter_for_chroma({"domain_id": "x"})
    assert where == {"domain_id": {"$eq": "x"}}, (
        f"declared scalar key must push ``$eq`` to Chroma; got {where!r}"
    )
    assert post == {}

    assert await store.count(filter={"domain_id": "x"}) == 2
    assert await store.count(filter={"domain_id": "y"}) == 1


@requires_chromadb
@pytest.mark.asyncio
async def test_count_with_pure_pushdown_skips_metadata_materialization():
    """``count`` with no post-filter remainder fetches IDs only.

    This is the primary win of the Tier A optimization combined with
    Tier B opt-in: when the entire filter pushes down (either
    list-valued, or declared-scalar with scalar value), ``count``
    fetches IDs only — no metadata materialization, regardless of
    collection size.
    """
    store = ChromaVectorStore(
        {
            "dimensions": 4,
            "collection_name": f"test_count_pushdown_{id(object())}",
            "scalar_metadata_keys": ["domain_id"],
        }
    )
    await store.initialize()

    await store.add_vectors(
        vectors=[_vec(i) for i in range(3)],
        ids=["a", "b", "c"],
        metadata=[
            {"domain_id": "x"},
            {"domain_id": "x"},
            {"domain_id": "y"},
        ],
    )

    # Probe the underlying collection to verify the include arg.
    real_get = store.collection.get
    captured: dict = {}

    def spy_get(**kwargs):
        captured["include"] = kwargs.get("include")
        return real_get(**kwargs)

    store.collection.get = spy_get  # type: ignore[method-assign]
    try:
        n = await store.count(filter={"domain_id": "x"})
    finally:
        store.collection.get = real_get  # type: ignore[method-assign]

    assert n == 2
    assert captured.get("include") == [], (
        f"Pure-pushdown count must use include=[] (IDs only); "
        f"got include={captured.get('include')!r}"
    )


@requires_chromadb
@pytest.mark.asyncio
async def test_count_with_post_filter_still_materializes():
    """Mixed/post-filter cases still materialize metadata (correct behavior)."""
    store = ChromaVectorStore(
        {
            "dimensions": 4,
            "collection_name": f"test_count_postfilter_{id(object())}",
            # scalar_metadata_keys NOT set → ``domain_id`` post-filtered.
        }
    )
    await store.initialize()

    await store.add_vectors(
        vectors=[_vec(i) for i in range(3)],
        ids=["a", "b", "c"],
        metadata=[
            {"domain_id": "x"},
            {"domain_id": "x"},
            {"domain_id": "y"},
        ],
    )

    real_get = store.collection.get
    captured: dict = {}

    def spy_get(**kwargs):
        captured["include"] = kwargs.get("include")
        return real_get(**kwargs)

    store.collection.get = spy_get  # type: ignore[method-assign]
    try:
        n = await store.count(filter={"domain_id": "x"})
    finally:
        store.collection.get = real_get  # type: ignore[method-assign]

    assert n == 2
    assert "metadatas" in (captured.get("include") or []), (
        f"Post-filter count must materialize metadata; "
        f"got include={captured.get('include')!r}"
    )

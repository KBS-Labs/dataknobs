"""FaissVectorStore upsert / orphan-leak tests.

Pre-existing bug surfaced by Item 118 review: ``add_vectors`` on
``FaissVectorStore`` overwrote ``id_map[ext_id]`` for an already-
present external ID without removing the prior internal ID's entries
from the FAISS index or ``metadata_store``. Result: silent residuals
that:

* Cannot be reached via ``get_vectors`` (the only external→internal
  bridge is ``id_map``, which now points to the new internal ID).
* Cannot be reached via filtered ``clear`` (which walks ``id_map``).
* CAN still be ranked and returned by ``search`` (FAISS searches the
  index directly, sees the orphan internal IDs, and returns them with
  whatever stale metadata happens to remain — and matches no external
  ID via ``reverse_id_map`` so they get logged-but-skipped).

The fix: when ``add_vectors`` finds an existing ``ext_id``, look up
the stale internal ID and remove it from FAISS + ``metadata_store``
BEFORE assigning the new mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataknobs_common.testing import is_faiss_available

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

requires_faiss = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss not installed"
)


@requires_faiss
@pytest.mark.asyncio
async def test_upsert_does_not_leak_orphan_metadata():
    """Re-adding the same ext_id replaces — no orphan in metadata_store."""
    store = FaissVectorStore({"dimensions": 4})
    await store.initialize()

    vec_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    await store.add_vectors(
        vectors=[vec_a],
        ids=["doc-1"],
        metadata=[{"version": "v1", "tenant": "a"}],
    )
    # Upsert: same external ID, new vector + metadata.
    await store.add_vectors(
        vectors=[vec_b],
        ids=["doc-1"],
        metadata=[{"version": "v2", "tenant": "a"}],
    )

    # External-id view: only the new metadata visible.
    results = await store.get_vectors(["doc-1"])
    assert len(results) == 1
    _, md = results[0]
    assert md == {"version": "v2", "tenant": "a"}, (
        "get_vectors must reflect the upserted metadata, not the original."
    )

    # metadata_store should hold exactly one entry — no orphan.
    assert len(store.metadata_store) == 1, (
        f"Orphan metadata leaked on upsert: metadata_store = "
        f"{store.metadata_store!r}"
    )


@requires_faiss
@pytest.mark.asyncio
async def test_upsert_does_not_leak_orphan_in_filtered_clear():
    """Filtered clear after upsert finds no stale residual."""
    store = FaissVectorStore({"dimensions": 4})
    await store.initialize()

    # Original record under tenant=a.
    await store.add_vectors(
        vectors=[np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)],
        ids=["doc-1"],
        metadata=[{"tenant": "a"}],
    )
    # Upsert: same ext_id, but tenant flips to b.
    await store.add_vectors(
        vectors=[np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)],
        ids=["doc-1"],
        metadata=[{"tenant": "b"}],
    )

    # Clear tenant=a — pre-fix, the original metadata's
    # internal ID is orphaned (still in metadata_store and in
    # FAISS) but unreachable via id_map. Filtered clear walks
    # id_map, so the orphan survives.
    await store.clear(filter={"tenant": "a"})

    # Whatever's left should be ONLY tenant=b. If the orphan
    # leaked, count would still include the stale tenant-a entry
    # buried inside metadata_store.
    tenant_a = await store.count(filter={"tenant": "a"})
    assert tenant_a == 0, (
        f"Filtered clear didn't reach the orphan tenant-a record "
        f"(count={tenant_a}). The internal ID must be cleaned up at "
        f"upsert time."
    )
    tenant_b = await store.count(filter={"tenant": "b"})
    assert tenant_b == 1, "Upserted tenant-b record must survive clear(tenant=a)"

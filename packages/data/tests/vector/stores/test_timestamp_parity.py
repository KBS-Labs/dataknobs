"""Cross-backend parity tests for timestamp exposure.

The shared timestamp abstraction exists so consumers can runtime-swap between
vector store backends without behavioral surprises. These tests
parameterize the same body over every shipping backend and assert
identical timestamp semantics:

- ``_created_at`` / ``_updated_at`` are present when
  ``include_timestamps=True`` and absent by default.
- Upsert preserves ``_created_at`` and advances ``_updated_at``.

Every in-tree backend runs: memory, faiss and chroma need no
services, and pgvector joins them when one is reachable. That is the
point of the suite — a backend that implements the feature without
appearing here is a divergence nothing would catch.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_data.testing import vector as _vector

# The backend roster, the availability marks, the per-backend construction
# and the Chroma teardown all live in ``conftest.py``. This suite needs an
# empty initialized store on every backend, which is exactly what
# ``initialized_vector_store`` is — so it declares no fixture of its own.


@pytest.mark.asyncio
async def test_timestamps_present_and_ordered(initialized_vector_store: Any) -> None:
    """include_timestamps=True exposes _created_at / _updated_at on every backend."""
    vec = _vector(4)
    await initialized_vector_store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])

    results = await initialized_vector_store.get_vectors(["t1"], include_timestamps=True)
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" in meta
    assert "_updated_at" in meta
    assert meta["_created_at"] is not None
    assert meta["_updated_at"] is not None
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_timestamps_absent_by_default(initialized_vector_store: Any) -> None:
    """Default get_vectors() omits timestamp keys on every backend."""
    vec = _vector(4)
    await initialized_vector_store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])

    results = await initialized_vector_store.get_vectors(["t1"])
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" not in meta
    assert "_updated_at" not in meta
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_upsert_refreshes_updated_consistently(
    initialized_vector_store: Any,
) -> None:
    """Second add_vectors with same id: created preserved, updated advances."""
    vec1 = _vector(4)
    vec2 = _vector(4, seed=1)

    await initialized_vector_store.add_vectors([vec1], ids=["t1"])
    first_results = await initialized_vector_store.get_vectors(["t1"], include_timestamps=True)
    first = first_results[0][1]
    assert first is not None

    # Sleep longer than the backend clock resolution so updated_at
    # must strictly advance under ISO-string lexicographic comparison.
    await asyncio.sleep(0.05)

    await initialized_vector_store.add_vectors([vec2], ids=["t1"])
    second_results = await initialized_vector_store.get_vectors(["t1"], include_timestamps=True)
    second = second_results[0][1]
    assert second is not None

    assert second["_created_at"] == first["_created_at"], (
        "created_at must not change on upsert (backend-dependent parity violation)"
    )
    assert second["_updated_at"] > first["_updated_at"], "updated_at must advance on upsert"


@pytest.mark.asyncio
async def test_update_vectors_preserves_created(
    initialized_vector_store: Any,
) -> None:
    """``update_vectors`` is an upsert, so it keeps ``created_at`` too.

    The documented rule is that ``created_at`` survives every write to
    an id the store already tracks — ``add_vectors`` on the same id is
    the case the suite above pins. ``update_vectors`` is that same
    operation reached through a different verb, but it was implemented
    as ``delete_vectors`` followed by ``add_vectors``, and the delete
    takes the tracking entry with the row. The re-add then had nothing
    to preserve and stamped a fresh creation date.

    The delete bought nothing: ``add_vectors`` already replaces a row's
    metadata outright on all four backends, which is the only thing the
    delete was there to guarantee. So the row was destroyed and rebuilt
    to achieve what a plain upsert already did, and the tracking loss
    was the whole of the difference.

    The consequence is the one the null-timestamp rationale warns
    about: a re-ingest sweep that calls ``update_vectors`` rewrites
    every row's creation date to the moment of the sweep, and nothing
    afterwards can tell a fabricated date from a real one.
    """
    vec1 = _vector(4)
    vec2 = _vector(4, seed=1)

    await initialized_vector_store.add_vectors([vec1], ids=["u1"], metadata=[{"v": 1}])
    first = (await initialized_vector_store.get_vectors(["u1"], include_timestamps=True))[0][1]
    assert first is not None

    await asyncio.sleep(0.05)

    await initialized_vector_store.update_vectors([vec2], ids=["u1"], metadata=[{"v": 2}])
    second = (await initialized_vector_store.get_vectors(["u1"], include_timestamps=True))[0][1]
    assert second is not None

    assert second["_created_at"] == first["_created_at"], (
        "update_vectors reset created_at — the delete discarded the tracking entry"
    )
    assert second["_updated_at"] > first["_updated_at"], "updated_at must advance"
    # And the replacement is still a replacement, not a merge.
    assert second["v"] == 2

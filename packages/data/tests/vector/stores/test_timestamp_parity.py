"""Cross-backend parity tests for timestamp exposure.

The shared timestamp abstraction exists so consumers can runtime-swap between
vector store backends without behavioral surprises. These tests
parameterize the same body over every shipping backend and assert
identical timestamp semantics:

- ``_created_at`` / ``_updated_at`` are present when
  ``include_timestamps=True`` and absent by default.
- Upsert preserves ``_created_at`` and advances ``_updated_at``.

Only **memory** and **pgvector** run today. FAISS and Chroma are
deferred — when they land,
each is added as a single ``pytest.param`` line; the test bodies do
not change.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from dataknobs_common.testing import requires_postgres

from dataknobs_data.vector.stores.memory import MemoryVectorStore

try:
    import asyncpg  # noqa: F401

    from dataknobs_data.vector.stores.pgvector import PgVectorStore

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


# Gate pgvector parameterization on the same env flag as
# test_pgvector_store.py. ``requires_postgres`` only checks that the
# TCP port is open; ``TEST_POSTGRES=true`` is the canonical "postgres
# is fully provisioned (db + extension)" signal and is what
# ``bin/test.sh`` sets when it brings services up.
_pgvector_marks = [
    requires_postgres,
    pytest.mark.skipif(
        os.environ.get("TEST_POSTGRES", "").lower() != "true"
        or not ASYNCPG_AVAILABLE,
        reason="pgvector parity tests require TEST_POSTGRES=true and asyncpg",
    ),
]


@pytest.fixture
def pgvector_config(make_pgvector_test_table: Any) -> Iterator[dict[str, Any]]:
    """Per-test pgvector config from the shared ``dataknobs-common``
    fixture (pre-drop + teardown drop + pgvector-extension ensure live
    there now). ``metric`` is preserved at ``cosine`` to keep behavior
    byte-identical to the prior hand-rolled config.
    """
    gen = make_pgvector_test_table("test_parity_", dimensions=4)
    cfg = next(gen)
    cfg["metric"] = "cosine"
    try:
        yield cfg
    finally:
        gen.close()


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def any_vector_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Yield a freshly-initialized VectorStore for each backend param.

    FAISS and Chroma will be added to ``params`` when their
    timestamp support lands.
    """
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
        await store.initialize()
        try:
            yield store
        finally:
            await store.close()
    elif backend == "pgvector":
        store = PgVectorStore(pgvector_config)
        await store.initialize()
        try:
            yield store
        finally:
            # The shared make_pgvector_test_table fixture owns the
            # table drop (it tears down after this store is closed,
            # since any_vector_store depends on pgvector_config).
            await store.close()
    else:
        pytest.fail(f"Unknown backend param: {backend}")


@pytest.mark.asyncio
async def test_timestamps_present_and_ordered(any_vector_store: Any) -> None:
    """include_timestamps=True exposes _created_at / _updated_at on every backend."""
    vec = np.random.rand(4).astype(np.float32)
    await any_vector_store.add_vectors(
        [vec], ids=["t1"], metadata=[{"k": "v"}]
    )

    results = await any_vector_store.get_vectors(
        ["t1"], include_timestamps=True
    )
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" in meta
    assert "_updated_at" in meta
    assert meta["_created_at"] is not None
    assert meta["_updated_at"] is not None
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_timestamps_absent_by_default(any_vector_store: Any) -> None:
    """Default get_vectors() omits timestamp keys on every backend."""
    vec = np.random.rand(4).astype(np.float32)
    await any_vector_store.add_vectors(
        [vec], ids=["t1"], metadata=[{"k": "v"}]
    )

    results = await any_vector_store.get_vectors(["t1"])
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" not in meta
    assert "_updated_at" not in meta
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_upsert_refreshes_updated_consistently(
    any_vector_store: Any,
) -> None:
    """Second add_vectors with same id: created preserved, updated advances."""
    vec1 = np.random.rand(4).astype(np.float32)
    vec2 = np.random.rand(4).astype(np.float32)

    await any_vector_store.add_vectors([vec1], ids=["t1"])
    first_results = await any_vector_store.get_vectors(
        ["t1"], include_timestamps=True
    )
    first = first_results[0][1]
    assert first is not None

    # Sleep longer than the backend clock resolution so updated_at
    # must strictly advance under ISO-string lexicographic comparison.
    await asyncio.sleep(0.05)

    await any_vector_store.add_vectors([vec2], ids=["t1"])
    second_results = await any_vector_store.get_vectors(
        ["t1"], include_timestamps=True
    )
    second = second_results[0][1]
    assert second is not None

    assert second["_created_at"] == first["_created_at"], (
        "created_at must not change on upsert "
        "(backend-dependent parity violation)"
    )
    assert second["_updated_at"] > first["_updated_at"], (
        "updated_at must advance on upsert"
    )

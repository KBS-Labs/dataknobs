"""One definition of "run this over every vector-store backend".

Six fixtures across this directory each carried their own copy of the
same three things: the four-way ``pytest.param`` list with its
availability marks, the build-by-backend-name dispatch, and the Chroma
collection teardown. Copies of a backend matrix are how a family grows a
fifth backend that only three suites exercise — and a divergence nothing
catches is the defect the cross-backend suites exist to catch. Adding a
backend is a one-place change here, which is the property those suites
are supposed to have.

What stays in each fixture is what actually differs: the seed corpus and
the config overrides. Both are arguments, so a suite that needs a scoped
store or a corpus larger than ``k`` says so in one line.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio
from dataknobs_common.testing import (
    is_chromadb_available,
    is_faiss_available,
    is_package_available,
    requires_real_postgres,
)

from dataknobs_data.vector.stores.memory import MemoryVectorStore

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

if is_package_available("asyncpg"):
    from dataknobs_data.vector.stores.pgvector import PgVectorStore


logger = logging.getLogger(__name__)

# Every in-tree backend. A backend absent from this list is a backend no
# cross-backend suite runs, which is the divergence those suites exist to
# catch — so it is deliberately the only place the roster is written.
ALL_BACKENDS: tuple[str, ...] = ("memory", "faiss", "chroma", "pgvector")

# Backends that need no reachable service. ``pgvector`` is the only one
# that does; ``requires_real_postgres`` is exactly the three terms each
# copy of this list used to assemble by hand — a reachable server,
# TEST_POSTGRES=true, and asyncpg installed.
_MARKS: dict[str, list[Any]] = {
    "memory": [],
    "faiss": [pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed")],
    "chroma": [pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed")],
    "pgvector": [requires_real_postgres],
}


def backend_params(*, without: Sequence[str] = ()) -> list[Any]:
    """The ``params=`` list for a fixture running over every backend.

    Args:
        without: Backends to leave out — for a suite whose subject does
            not exist on one of them. Naming the exclusion beats
            hand-writing a shorter list, because the roster stays in one
            place and the omission stays visible at the fixture.

    Returns:
        A fresh list of ``pytest.param`` entries, id'd by backend name and
        carrying each backend's availability marks.
    """
    unknown = set(without) - set(ALL_BACKENDS)
    if unknown:
        raise ValueError(f"unknown backend(s) in `without`: {sorted(unknown)}")
    return [
        pytest.param(name, id=name, marks=_MARKS[name])
        for name in ALL_BACKENDS
        if name not in without
    ]


@pytest.fixture
def pgvector_config(make_pgvector_test_table: Any) -> Iterator[dict[str, Any]]:
    """Per-test pgvector config from the shared ``dataknobs-common`` fixture.

    That fixture owns the table lifecycle (pre-drop, teardown drop, and
    the pgvector-extension ensure), so nothing here tears the table down.
    ``metric`` is pinned to ``cosine`` because every suite in this
    directory built it that way.
    """
    gen = make_pgvector_test_table("test_vec_", dimensions=4)
    cfg = next(gen)
    cfg["metric"] = "cosine"
    try:
        yield cfg
    finally:
        gen.close()


def build_vector_store(
    backend: str,
    request: pytest.FixtureRequest,
    *,
    collection_prefix: str,
    dimensions: int = 4,
    **overrides: Any,
) -> Any:
    """Construct (but do not initialize) the store named by ``backend``.

    Args:
        backend: One of :data:`ALL_BACKENDS`.
        request: The fixture request, used to reach ``pgvector_config``
            lazily — so a run without Postgres never builds that config.
        collection_prefix: Chroma only. A per-test collection name is
            derived from it, so two tests never share a collection.
        dimensions: Vector width. Every suite here uses 4.
        **overrides: Merged into the config dict last, so a suite can add
            ``domain_id`` or ``persist_path`` without a second builder.

    Returns:
        An unopened store; the caller owns ``initialize`` / ``close``, or
        uses :func:`running_vector_store` which owns both.
    """
    if backend == "memory":
        return MemoryVectorStore({"dimensions": dimensions, **overrides})
    if backend == "faiss":
        return FaissVectorStore({"dimensions": dimensions, "metric": "cosine", **overrides})
    if backend == "chroma":
        return ChromaVectorStore(
            {
                "dimensions": dimensions,
                "collection_name": f"{collection_prefix}{uuid.uuid4().hex[:8]}",
                **overrides,
            }
        )
    if backend == "pgvector":
        return PgVectorStore({**request.getfixturevalue("pgvector_config"), **overrides})
    pytest.fail(f"Unknown backend param: {backend}")


async def teardown_vector_store(backend: str, store: Any) -> None:
    """Drop the per-test Chroma collection a fixture created.

    pgvector tables belong to ``make_pgvector_test_table``, and the
    in-process backends hold nothing outside the instance, so Chroma is
    the only backend with state to reclaim. The failure is logged rather
    than swallowed, so an orphaned collection is visible in pytest output
    instead of accumulating silently.
    """
    if backend == "chroma":
        try:
            store.client.delete_collection(name=store.collection_name)
        except Exception as exc:  # pragma: no cover - teardown best effort
            logger.warning(
                "Chroma teardown failed for collection %r: %s",
                store.collection_name,
                exc,
            )


@asynccontextmanager
async def running_vector_store(
    backend: str,
    request: pytest.FixtureRequest,
    *,
    collection_prefix: str,
    dimensions: int = 4,
    **overrides: Any,
) -> AsyncIterator[Any]:
    """An initialized store, torn down and closed on the way out.

    Wraps :func:`build_vector_store` with the lifecycle every fixture in
    this directory repeated. Seed inside the ``async with`` body — the
    corpus is the part that legitimately differs per suite.
    """
    store = build_vector_store(
        backend,
        request,
        collection_prefix=collection_prefix,
        dimensions=dimensions,
        **overrides,
    )
    await store.initialize()
    try:
        yield store
    finally:
        await teardown_vector_store(backend, store)
        await store.close()


@pytest_asyncio.fixture(params=backend_params())
async def initialized_vector_store(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    """An empty, initialized store for each backend.

    The unseeded default. A suite needing a corpus writes its own fixture
    over :func:`running_vector_store` rather than seeding this one, so the
    seed stays visible next to the assertions that depend on it.
    """
    async with running_vector_store(
        request.param, request, collection_prefix="test_store_"
    ) as store:
        yield store

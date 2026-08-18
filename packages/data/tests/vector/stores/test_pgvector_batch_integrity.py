"""``PgVectorStore.add_vectors`` batch integrity: one failure, no residue.

Two properties of a batch write that only a real server can show, and
that a scoped store made newly relevant.

**The whole batch commits or none of it does.** ``add_vectors`` inserts
row-by-row over a pooled connection. asyncpg puts a bare ``execute`` in
its own implicit transaction, so each row committed as it went and a
failure part-way through left every earlier row behind — the caller
retrying after the error retried on top of a half-applied write. The
``VectorDomainScopeError`` this backend can now raise says "nothing in
the batch is written", which was true of the scope check (it runs
first) and false of every other failure the same call can hit.

**A malformed id names itself.** ``delete_vectors`` validates UUID-typed
ids client-side before its ``ANY($1::uuid[])`` bind, precisely so the
error names the offending id instead of dumping the array. The scoped
ownership probe introduced a second bulk bind on the write path without
that guard, so a scoped store answered a single bad id by listing the
whole batch — and only a scoped store did, which is the kind of
asymmetry that survives review.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

from dataknobs_common.testing import (
    is_package_available,
    postgres_dsn,
    postgres_env_params,
    requires_real_postgres,
    safe_sql_ident,
)

if is_package_available("asyncpg"):
    import asyncpg

    from dataknobs_data.vector.stores.pgvector import PgVectorStore

_pgvector_marks = [requires_real_postgres]

DIMS = 4


def _dsn() -> str:
    return postgres_dsn(postgres_env_params())


def _vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(DIMS, dtype=np.float32)


@pytest.fixture(scope="session")
def _ensure_pgvector_extension() -> None:
    async def _setup() -> None:
        conn = await asyncpg.connect(_dsn())
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()

    try:
        asyncio.run(_setup())
    except (OSError, asyncpg.PostgresError):
        pass


@pytest_asyncio.fixture
async def pg_table(_ensure_pgvector_extension: None) -> AsyncIterator[str]:
    table = f"test_pgv_batch_{uuid.uuid4().hex[:8]}"

    async def _drop() -> None:
        conn = await asyncpg.connect(_dsn())
        try:
            await conn.execute(f"DROP TABLE IF EXISTS public.{safe_sql_ident(table)} CASCADE")
        finally:
            await conn.close()

    await _drop()
    try:
        yield table
    finally:
        await _drop()


def _config(table: str, **kw: Any) -> dict[str, Any]:
    return {
        "connection_string": _dsn(),
        "dimensions": DIMS,
        "metric": "cosine",
        "schema": "public",
        "table_name": table,
        "auto_create_table": True,
        "id_type": "text",
        **kw,
    }


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_a_failed_row_rolls_the_whole_batch_back(_m: None, pg_table: str) -> None:
    """A mid-batch failure leaves the table exactly as it found it.

    The second row carries a ``chunk_index`` the ``INT`` column will not
    take, so asyncpg rejects it at bind time — after the first row has
    already been sent. Without an enclosing transaction the first row is
    committed and the caller is left with one third of a batch it was
    told had failed.
    """
    store = PgVectorStore(_config(pg_table))
    await store.initialize()
    try:
        with pytest.raises(Exception, match=r"chunk_index|invalid input|integer"):
            await store.add_vectors(
                [_vec(0), _vec(1), _vec(2)],
                ids=["r1", "r2", "r3"],
                metadata=[{"k": "a"}, {"chunk_index": "not-an-int"}, {"k": "c"}],
            )

        assert await store.count() == 0, "a rejected batch committed its leading rows"
        assert [r[0] for r in await store.get_vectors(["r1", "r2", "r3"])] == [None] * 3
    finally:
        await store.close()


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_a_successful_batch_still_commits(_m: None, pg_table: str) -> None:
    """The rollback guard does not hold a good batch hostage.

    Worth its own case: wrapping the loop in a transaction is only
    correct if the transaction actually commits on the way out, and a
    context manager that swallowed the commit would leave every write in
    this backend silently dropped while every in-call read still saw it.
    """
    store = PgVectorStore(_config(pg_table))
    await store.initialize()
    try:
        await store.add_vectors([_vec(0), _vec(1)], ids=["r1", "r2"], metadata=[{"k": "a"}, {}])
        await store.close()

        # A *different* store object, so the rows are read back through a
        # fresh connection rather than the one that wrote them.
        reader = PgVectorStore(_config(pg_table))
        await reader.initialize()
        try:
            assert await reader.count() == 2
        finally:
            await reader.close()
    finally:
        await store.close()


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_a_malformed_uuid_names_itself_on_a_scoped_store(_m: None, pg_table: str) -> None:
    """The guided error names the bad id, not the batch it arrived in.

    A scoped store runs a bulk ownership probe before the inserts, bound
    as ``ANY($1::uuid[])``. Postgres answers a malformed element with
    "invalid input for array element at index N", and the guided-error
    wrapper is handed the whole array to interpolate — so the message
    grew with the batch and never said which id was wrong. An unscoped
    store, which skips the probe and fails at the single-row insert
    instead, always got the useful message.
    """
    ids = ["not-a-uuid", *(str(uuid.uuid4()) for _ in range(4))]
    vectors = [_vec(i) for i in range(5)]

    store = PgVectorStore(_config(pg_table, id_type="uuid", domain_id="t1"))
    await store.initialize()
    try:
        with pytest.raises(ValueError) as excinfo:
            await store.add_vectors(vectors, ids=ids, metadata=[{} for _ in ids])
    finally:
        await store.close()

    message = str(excinfo.value)
    assert "not-a-uuid" in message
    for good in ids[1:]:
        assert good not in message, f"the guided error listed a well-formed id: {message}"

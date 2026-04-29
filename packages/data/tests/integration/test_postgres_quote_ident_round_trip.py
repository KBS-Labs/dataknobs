"""Integration tests: identifier quoting in AsyncPostgresDatabase edge cases.

Covers regressions introduced in / missed by PR #290:
  P2 — copy_records_to_table double-quoting (stream_write with mixed-case table)
  A2 — build_vector_index_sql / get_vector_count_sql unquoted helpers
  A3 — _ensure_vector_column unquoted column name in ALTER TABLE
  A4 — vector_search / hybrid_search unquoted column name in DML

Requires a running PostgreSQL instance with the pgvector extension.
"""

from collections import OrderedDict
from collections.abc import AsyncIterator

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase

pytestmark = requires_postgres

np = pytest.importorskip("numpy")


async def _aiterate(records: list[Record]) -> AsyncIterator[Record]:
    for r in records:
        yield r


# ---------------------------------------------------------------------------
# P2 — stream_write with mixed-case table name
# ---------------------------------------------------------------------------


class TestStreamWriteMixedCaseTable:
    @pytest.mark.asyncio
    async def test_stream_write_mixed_case_table(self, postgres_test_db):
        """copy_records_to_table must not double-quote an already-quoted name.

        Before the P2 fix, passing self._q_qualified to copy_records_to_table
        caused asyncpg to re-quote it, producing triple-quoted SQL that
        PostgreSQL rejected.
        """
        postgres_test_db["table"] = "MyMixedCase"
        db = AsyncPostgresDatabase(postgres_test_db)
        await db.connect()
        try:
            records = [Record({"idx": i}) for i in range(5)]
            from dataknobs_data import StreamConfig

            count_before = await db.count()
            result = await db.stream_write(_aiterate(records), StreamConfig(batch_size=3))
            assert result.successful == 5
            assert await db.count() == count_before + 5
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Reserved-word table name — create / read / delete round-trip
# ---------------------------------------------------------------------------


class TestReservedWordTableName:
    @pytest.mark.asyncio
    async def test_crud_with_reserved_word_table(self, postgres_test_db):
        """A table named after a SQL reserved word must work end-to-end.

        "select" is unambiguously reserved in every SQL dialect.  Without
        quote_ident the generated DDL / DML would be a syntax error.
        """
        postgres_test_db["table"] = "select"
        db = AsyncPostgresDatabase(postgres_test_db)
        await db.connect()
        try:
            record_id = await db.create(Record({"value": 42}))
            assert record_id

            fetched = await db.read(record_id)
            assert fetched is not None

            deleted = await db.delete(record_id)
            assert deleted is True
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# A2 — create_vector_index with mixed-case table name
# ---------------------------------------------------------------------------


class TestCreateVectorIndexMixedCaseTable:
    @pytest.mark.asyncio
    async def test_create_vector_index_mixed_case_table(self, postgres_test_db):
        """build_vector_index_sql must receive pre-quoted names.

        Before A2, the helper received raw self.table_name / self.schema_name
        and produced unquoted ON schema.table SQL that failed for mixed-case.
        """
        postgres_test_db["table"] = "MyVectorTable"
        postgres_test_db["vector_enabled"] = True
        db = AsyncPostgresDatabase(postgres_test_db)
        await db.connect()
        try:
            record = Record({"text": "hello"})
            await db.create(record)

            created = await db.create_vector_index(
                vector_field="embedding",
                dimensions=4,
                metric="cosine",
                index_type="ivfflat",
            )
            assert created is True
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# A3 + A4 — vector_search with mixed-case field name
# ---------------------------------------------------------------------------


class TestVectorSearchMixedCaseField:
    @pytest.mark.asyncio
    async def test_vector_search_mixed_case_field(self, postgres_test_db):
        """_ensure_vector_column and vector_search must quote column names.

        Before A3/A4, a field_name like "MyField" produced
        ADD COLUMN vector_MyField and SELECT vector_MyField without quoting,
        which fails in PostgreSQL when the name contains uppercase characters.
        """
        postgres_test_db["table"] = "VecSearchTest"
        postgres_test_db["vector_enabled"] = True
        db = AsyncPostgresDatabase(postgres_test_db)
        await db.connect()
        try:
            from dataknobs_data import VectorField

            vec = [0.1, 0.2, 0.3, 0.4]
            record = Record(
                data=OrderedDict({
                    "MyEmbedding": VectorField(
                        name="MyEmbedding",
                        value=vec,
                        dimensions=4,
                    )
                }),
                metadata={"label": "test"},
            )
            await db.create(record)

            results = await db.vector_search(
                query_vector=vec,
                field_name="MyEmbedding",
                k=1,
            )
            assert len(results) == 1
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# create / update / upsert all write vector data via _collect_vector_inserts
# ---------------------------------------------------------------------------


class TestVectorUpdateMixedCaseField:
    @pytest.mark.asyncio
    async def test_update_writes_vector_column(self, postgres_test_db):
        """update() must write vector data to the quoted pgvector column.

        Before the _collect_vector_inserts fix, update() never wrote vector
        data — the column stayed NULL and vector_search returned 0 rows.
        """
        postgres_test_db["table"] = "VecUpdateTest"
        postgres_test_db["vector_enabled"] = True
        db = AsyncPostgresDatabase(postgres_test_db)
        await db.connect()
        try:
            from dataknobs_data import VectorField

            vec = [0.1, 0.2, 0.3, 0.4]
            record_id = await db.create(Record({"placeholder": True}))

            updated_record = Record(
                data=OrderedDict({
                    "MyEmbedding": VectorField(
                        name="MyEmbedding",
                        value=vec,
                        dimensions=4,
                    )
                }),
                metadata={"label": "updated"},
            )
            ok = await db.update(record_id, updated_record)
            assert ok is True

            results = await db.vector_search(
                query_vector=vec,
                field_name="MyEmbedding",
                k=1,
            )
            assert len(results) == 1
        finally:
            await db.close()


class TestVectorUpsertMixedCaseField:
    @pytest.mark.asyncio
    async def test_upsert_insert_path_writes_vector_column(self, postgres_test_db):
        """upsert() insert path must write vector data to the quoted pgvector column."""
        postgres_test_db["table"] = "VecUpsertTest"
        postgres_test_db["vector_enabled"] = True
        db = AsyncPostgresDatabase(postgres_test_db)
        await db.connect()
        try:
            from dataknobs_data import VectorField

            vec = [0.1, 0.2, 0.3, 0.4]
            record = Record(
                data=OrderedDict({
                    "MyEmbedding": VectorField(
                        name="MyEmbedding",
                        value=vec,
                        dimensions=4,
                    )
                }),
                metadata={"label": "upserted"},
            )
            await db.upsert(record)

            results = await db.vector_search(
                query_vector=vec,
                field_name="MyEmbedding",
                k=1,
            )
            assert len(results) == 1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_upsert_update_path_writes_vector_column(self, postgres_test_db):
        """upsert() update (ON CONFLICT) path must also write vector data."""
        postgres_test_db["table"] = "VecUpsertUpdateTest"
        postgres_test_db["vector_enabled"] = True
        db = AsyncPostgresDatabase(postgres_test_db)
        await db.connect()
        try:
            from dataknobs_data import VectorField

            vec1 = [0.1, 0.2, 0.3, 0.4]
            vec2 = [0.5, 0.6, 0.7, 0.8]

            # Insert without a vector first
            record_id = await db.create(Record({"placeholder": True}))

            # Upsert the same id with a vector — exercises the ON CONFLICT path
            record = Record(
                data=OrderedDict({
                    "MyEmbedding": VectorField(
                        name="MyEmbedding",
                        value=vec2,
                        dimensions=4,
                    )
                }),
                metadata={"label": "conflict-update"},
                id=record_id,
            )
            await db.upsert(record)

            # The most-similar result should be the vec2 record
            results = await db.vector_search(
                query_vector=vec2,
                field_name="MyEmbedding",
                k=1,
            )
            assert len(results) == 1
        finally:
            await db.close()

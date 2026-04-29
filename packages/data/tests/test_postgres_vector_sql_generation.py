"""Unit tests for postgres_vector.py SQL-generation functions.

Pure-Python — no database connection required.

Covers the quoting contract for index names: build_vector_index_sql must
quote the index name (via quote_ident) so that it is consistent with
drop_vector_index, which already calls quote_ident(index_name).

Without quoting, PostgreSQL folds the index name to lowercase at CREATE time.
The quoted DROP then looks for the case-preserved name and silently finds
nothing, leaving orphaned indexes that cannot be dropped programmatically.
"""

from dataknobs_data.backends.postgres_vector import (
    build_vector_index_sql,
    get_vector_index_name,
)


class TestBuildVectorIndexSqlQuotesIndexName:
    """Index name in CREATE INDEX must be quoted to match drop_vector_index."""

    def test_ivfflat_index_name_is_quoted(self):
        """IVFFlat branch must quote the index name."""
        sql = build_vector_index_sql(
            q_table_name='"MyTable"',
            q_schema_name='"public"',
            column_name='"vector_MyEmbedding"::vector(4)',
            dimensions=4,
            metric="cosine",
            index_type="ivfflat",
            field_name="MyEmbedding",
        )
        index_name = get_vector_index_name("MyTable", "MyEmbedding", "cosine")
        assert f'"{index_name}"' in sql, (
            f"Expected quoted '\"{ index_name}\"' in CREATE INDEX SQL; "
            f"got unquoted '{index_name}'. PostgreSQL folds it to lowercase "
            "in the catalog, so the quoted DROP INDEX in drop_vector_index "
            "silently finds nothing."
        )

    def test_hnsw_index_name_is_quoted(self):
        """HNSW branch must quote the index name."""
        sql = build_vector_index_sql(
            q_table_name='"MyTable"',
            q_schema_name='"public"',
            column_name='"vector_MyEmbedding"::vector(4)',
            dimensions=4,
            metric="cosine",
            index_type="hnsw",
            field_name="MyEmbedding",
        )
        index_name = get_vector_index_name("MyTable", "MyEmbedding", "cosine")
        assert f'"{index_name}"' in sql

    def test_default_index_name_is_quoted(self):
        """Default (btree) branch must quote the index name."""
        sql = build_vector_index_sql(
            q_table_name='"MyTable"',
            q_schema_name='"public"',
            column_name='"vector_MyEmbedding"',
            dimensions=None,
            metric="cosine",
            index_type="btree",
            field_name="MyEmbedding",
        )
        index_name = get_vector_index_name("MyTable", "MyEmbedding", "cosine")
        assert f'"{index_name}"' in sql

    def test_lowercase_table_name_index_is_quoted(self):
        """Quoting is required even for all-lowercase names (consistency with drop)."""
        sql = build_vector_index_sql(
            q_table_name='"records"',
            q_schema_name='"public"',
            column_name='"vector_embedding"::vector(4)',
            dimensions=4,
            metric="cosine",
            index_type="ivfflat",
            field_name="embedding",
        )
        index_name = get_vector_index_name("records", "embedding", "cosine")
        assert f'"{index_name}"' in sql

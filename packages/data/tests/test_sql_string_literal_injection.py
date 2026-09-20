"""Reproduce-first tests for SQL injection in string-literal field positions (A5).

get_vector_extraction_sql and _build_text_field_concat both embed caller-supplied
field names inside SQL single-quoted string literals (JSONB key positions).  A field
name like "field'name" breaks the SQL syntax; a name like "'; DROP TABLE records;--"
is a SQL injection vector.  quote_ident() is wrong here — the fix is allowlist
validation with validate_field_name(), consistent with _build_json_field_expr.
"""

import pytest

from dataknobs_data import Filter, Operator, Query
from dataknobs_data.backends.sql_base import SQLRecordSerializer


class TestGetVectorExtractionSqlValidation:
    """get_vector_extraction_sql must reject field names that are unsafe in SQL string literals."""

    def test_single_quote_raises(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            SQLRecordSerializer.get_vector_extraction_sql("field'name")

    def test_injection_attempt_raises(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            SQLRecordSerializer.get_vector_extraction_sql("x'; DROP TABLE records;--")

    def test_hyphen_raises(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            SQLRecordSerializer.get_vector_extraction_sql("my-field")

    def test_dot_raises(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            SQLRecordSerializer.get_vector_extraction_sql("my.field")

    def test_sqlite_dialect_also_validated(self):
        """SQLite path embeds field_name in '$.{field_name}' — same injection risk."""
        with pytest.raises(ValueError):
            SQLRecordSerializer.get_vector_extraction_sql("bad'field", dialect="sqlite")

    def test_generic_dialect_also_validated(self):
        with pytest.raises(ValueError):
            SQLRecordSerializer.get_vector_extraction_sql("bad'field", dialect="other")

    def test_valid_identifier_postgres(self):
        sql = SQLRecordSerializer.get_vector_extraction_sql("embedding")
        assert "embedding" in sql
        assert "::vector" in sql

    def test_valid_identifier_sqlite(self):
        sql = SQLRecordSerializer.get_vector_extraction_sql("my_vector", dialect="sqlite")
        assert "my_vector" in sql
        assert "json_extract" in sql

    def test_leading_underscore_valid(self):
        sql = SQLRecordSerializer.get_vector_extraction_sql("_embedding")
        assert "_embedding" in sql


class TestBuildTextFieldConcatValidation:
    """_build_text_field_concat must reject field names unsafe in SQL string literals.

    Each entry in text_fields goes into COALESCE(data->>'...' ...) — same injection
    class as get_vector_extraction_sql.
    """

    def _make_db(self):
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        return AsyncPostgresDatabase({})

    def test_single_quote_in_field_raises(self):
        db = self._make_db()
        with pytest.raises(ValueError, match="Invalid field name"):
            db._build_text_field_concat(["field'name"])

    def test_injection_in_mixed_list_raises(self):
        db = self._make_db()
        with pytest.raises(ValueError, match="Invalid field name"):
            db._build_text_field_concat(["content", "'; DROP TABLE records;--"])

    def test_hyphen_raises(self):
        db = self._make_db()
        with pytest.raises(ValueError, match="Invalid field name"):
            db._build_text_field_concat(["my-field"])

    def test_valid_single_field(self):
        db = self._make_db()
        sql = db._build_text_field_concat(["body"])
        assert "COALESCE(data->>'body', '')" in sql

    def test_valid_multiple_fields(self):
        db = self._make_db()
        sql = db._build_text_field_concat(["content", "title", "text"])
        assert "COALESCE(data->>'content', '')" in sql
        assert "COALESCE(data->>'title', '')" in sql
        assert "COALESCE(data->>'text', '')" in sql

    def test_empty_list_returns_default(self):
        db = self._make_db()
        sql = db._build_text_field_concat([])
        assert "content" in sql


class TestStreamReadFilterValidation:
    """stream_read must reject filter.field values unsafe in JSONB key positions.

    Both SyncPostgresDatabase.stream_read and AsyncPostgresDatabase.stream_read put
    filter.field into a data->>'<key>' JSONB key position — the same injection class
    as _build_text_field_concat.  They open-coded that interpolation when these tests
    were written and now reach it through SQLQueryBuilder.build_where_clause, which
    validates at the point of interpolation; the twins pre-flight the same check so a
    malformed field is still refused before a connection is acquired.  That ordering
    is what makes the refusal reachable on first iteration with no live database.
    """

    def _bad_query(self, field: str = "x'; DROP TABLE records;--") -> Query:
        return Query(filters=[Filter(field=field, operator=Operator.EQ, value="x")])

    def test_sync_injection_raises(self):
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        db = SyncPostgresDatabase({})
        with pytest.raises(ValueError, match="Invalid field name"):
            next(iter(db.stream_read(self._bad_query())))

    def test_sync_hyphen_raises(self):
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        db = SyncPostgresDatabase({})
        with pytest.raises(ValueError, match="Invalid field name"):
            next(iter(db.stream_read(self._bad_query("my-field"))))

    async def test_async_injection_raises(self):
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        db = AsyncPostgresDatabase({})
        with pytest.raises(ValueError, match="Invalid field name"):
            async for _ in db.stream_read(self._bad_query()):
                pass  # pragma: no cover

    async def test_async_hyphen_raises(self):
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        db = AsyncPostgresDatabase({})
        with pytest.raises(ValueError, match="Invalid field name"):
            async for _ in db.stream_read(self._bad_query("my-field")):
                pass  # pragma: no cover


class TestStreamReadAppliesTheBuilderGrammar:
    """stream_read's pre-check must apply the same grammar its builder applies.

    The pre-check above exists because the filter field reaches a JSONB key in
    SQL *string-literal* position, where ``quote_ident()`` does not apply.  When
    it was written, ``stream_read`` interpolated that position itself.  It no
    longer does — the field now reaches SQL through ``build_where_clause``, and
    the builder validates each dot-separated **segment** at the point of
    interpolation.

    So the grammars have to agree, and the whole-identifier check did not: it
    read ``metadata.work_order_id`` as one name containing an illegal ``.`` and
    refused it, while ``search`` over the same ``Query`` answered rows through
    ``metadata->>'work_order_id'``.  One backend, one ``Query``, two answers.
    """

    DOTTED_FIELD = "metadata.work_order_id"

    def _dotted_query(self) -> Query:
        return Query(filters=[Filter(field=self.DOTTED_FIELD, operator=Operator.EQ, value="W-1")])

    def test_builder_accepts_the_dotted_field(self):
        """Positive control: the grammar the pre-check is required to match."""
        from dataknobs_data.backends.sql_base import SQLQueryBuilder

        builder = SQLQueryBuilder(table_name="records", dialect="postgres")
        where_clause, params = builder.build_where_clause(self._dotted_query())
        assert where_clause == " AND metadata->>'work_order_id' = $1"
        assert params == ["W-1"]

    def test_sync_stream_read_accepts_the_dotted_field(self):
        """Reaching the connection check is the assertion: the grammar let it by."""
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        db = SyncPostgresDatabase({})
        with pytest.raises(RuntimeError, match="not connected"):
            next(iter(db.stream_read(self._dotted_query())))

    async def test_async_stream_read_accepts_the_dotted_field(self):
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        db = AsyncPostgresDatabase({})
        with pytest.raises(RuntimeError, match="not connected"):
            async for _ in db.stream_read(self._dotted_query()):
                pass  # pragma: no cover

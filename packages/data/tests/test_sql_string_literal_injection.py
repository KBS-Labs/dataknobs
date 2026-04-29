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

    Both SyncPostgresDatabase.stream_read and AsyncPostgresDatabase.stream_read embed
    filter.field into data->>'{filter.field}' — the same injection class as
    _build_text_field_concat.  Validation fires before _check_connection() so it is
    reachable on first iteration without a live database connection.
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

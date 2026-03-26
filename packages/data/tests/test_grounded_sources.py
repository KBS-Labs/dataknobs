"""Tests for the grounded source abstraction in dataknobs-data."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.query import Filter, Operator
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema
from dataknobs_data.sources.base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
    SourceSchema,
)
from dataknobs_data.sources.database import DatabaseSource


# ------------------------------------------------------------------
# Data type tests
# ------------------------------------------------------------------


class TestSourceResult:
    def test_to_dict(self) -> None:
        result = SourceResult(
            content="hello world",
            source_id="rec_1",
            source_name="test_db",
            source_type="database",
            relevance=0.95,
            metadata={"field": "value"},
        )
        d = result.to_dict()
        assert d["text"] == "hello world"
        assert d["source"] == "test_db"
        assert d["similarity"] == 0.95
        assert d["metadata"]["field"] == "value"


class TestRetrievalIntent:
    def test_defaults(self) -> None:
        intent = RetrievalIntent()
        assert intent.text_queries == []
        assert intent.filters == {}
        assert intent.scope == "focused"

    def test_with_filters(self) -> None:
        intent = RetrievalIntent(
            text_queries=["algorithms"],
            filters={"courses": {"department": "CS"}},
            scope="exact",
        )
        assert intent.filters["courses"]["department"] == "CS"


class TestSourceSchema:
    def test_basic(self) -> None:
        schema = SourceSchema(
            source_name="test",
            fields={"name": {"type": "string"}},
            description="A test source",
        )
        assert schema.source_name == "test"
        assert "name" in schema.fields


# ------------------------------------------------------------------
# DatabaseSource schema generation tests
# ------------------------------------------------------------------


def _make_course_schema() -> DatabaseSchema:
    """Create a sample schema for testing."""
    schema = DatabaseSchema.create(
        title=FieldType.STRING,
        department=FieldType.STRING,
        level=FieldType.INTEGER,
        active=FieldType.BOOLEAN,
        description=FieldType.TEXT,
    )
    schema.fields["department"].metadata["enum"] = ["CS", "Math", "Physics"]
    schema.fields["department"].metadata["description"] = "Academic department"
    return schema


class TestDatabaseSourceSchemaGeneration:
    """Test auto-generation of SourceSchema from DatabaseSchema."""

    def test_generates_string_fields(self) -> None:
        source = DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=_make_course_schema(),
            name="courses",
        )
        schema = source.get_schema()
        assert schema is not None
        assert schema.source_name == "courses"
        assert schema.fields["title"]["type"] == "string"

    def test_generates_integer_fields(self) -> None:
        source = DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=_make_course_schema(),
            name="courses",
        )
        schema = source.get_schema()
        assert schema is not None
        assert schema.fields["level"]["type"] == "integer"

    def test_generates_boolean_fields(self) -> None:
        source = DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=_make_course_schema(),
            name="courses",
        )
        schema = source.get_schema()
        assert schema is not None
        assert schema.fields["active"]["type"] == "boolean"

    def test_enum_fields_have_extraction_hints(self) -> None:
        source = DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=_make_course_schema(),
            name="courses",
        )
        schema = source.get_schema()
        assert schema is not None
        dept = schema.fields["department"]
        assert dept["enum"] == ["CS", "Math", "Physics"]
        assert dept["x-extraction"]["normalize"] is True

    def test_datetime_fields_generate_range_properties(self) -> None:
        db_schema = DatabaseSchema.create(
            created=FieldType.DATETIME,
        )
        source = DatabaseSource(
            db=AsyncMemoryDatabase(), schema=db_schema, name="events",
        )
        schema = source.get_schema()
        assert schema is not None
        assert "created_after" in schema.fields
        assert "created_before" in schema.fields
        assert schema.fields["created_after"]["format"] == "date-time"

    def test_vector_fields_skipped(self) -> None:
        db_schema = DatabaseSchema.create(
            name=FieldType.STRING,
            embedding=FieldType.VECTOR,
        )
        source = DatabaseSource(
            db=AsyncMemoryDatabase(), schema=db_schema, name="docs",
        )
        schema = source.get_schema()
        assert schema is not None
        assert "name" in schema.fields
        assert "embedding" not in schema.fields

    def test_description_passed_through(self) -> None:
        source = DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=_make_course_schema(),
            name="courses",
            description="Course catalog",
        )
        schema = source.get_schema()
        assert schema is not None
        assert schema.description == "Course catalog"


# ------------------------------------------------------------------
# DatabaseSource query building tests
# ------------------------------------------------------------------


class TestDatabaseSourceQueryBuilding:
    """Test deterministic translation of intent to Query."""

    def _make_source(self) -> DatabaseSource:
        return DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=_make_course_schema(),
            name="courses",
            content_field="description",
            text_search_fields=["title", "description"],
        )

    def test_eq_filter(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=[],
            filters={"courses": {"department": "CS"}},
        )
        query = source._build_query(intent)
        assert len(query.filters) == 1
        assert query.filters[0].field == "department"
        assert query.filters[0].operator == Operator.EQ
        assert query.filters[0].value == "CS"

    def test_in_filter(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=[],
            filters={"courses": {"department": ["CS", "Math"]}},
        )
        query = source._build_query(intent)
        assert len(query.filters) == 1
        assert query.filters[0].operator == Operator.IN
        assert query.filters[0].value == ["CS", "Math"]

    def test_range_filter(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=[],
            filters={"courses": {"level": {"min": 200, "max": 400}}},
        )
        query = source._build_query(intent)
        assert len(query.filters) == 2
        ops = {f.operator for f in query.filters}
        assert Operator.GTE in ops
        assert Operator.LTE in ops

    def test_datetime_range_filter(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=[],
            filters={"courses": {"created_after": "2024-01-01"}},
        )
        query = source._build_query(intent)
        assert len(query.filters) == 1
        assert query.filters[0].field == "created"
        assert query.filters[0].operator == Operator.GTE

    def test_text_search(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=["algorithms"],
            filters={},
        )
        query = source._build_query(intent)
        # 1 text query × 2 text_search_fields = 2 LIKE filters
        assert len(query.filters) == 2
        for f in query.filters:
            assert f.operator == Operator.LIKE
            assert "%algorithms%" in f.value

    def test_combined_filters_and_text(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=["intro"],
            filters={"courses": {"department": "CS", "level": 100}},
        )
        query = source._build_query(intent)
        # 2 structured filters + 2 LIKE filters (1 query × 2 fields)
        assert len(query.filters) == 4

    def test_other_source_filters_ignored(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=[],
            filters={"other_source": {"field": "value"}},
        )
        query = source._build_query(intent)
        assert len(query.filters) == 0

    def test_empty_intent(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent()
        query = source._build_query(intent)
        assert len(query.filters) == 0


# ------------------------------------------------------------------
# DatabaseSource execution tests
# ------------------------------------------------------------------


class TestDatabaseSourceExecution:
    """Test query execution against AsyncMemoryDatabase."""

    @pytest.mark.asyncio
    async def test_basic_query(self) -> None:
        db = AsyncMemoryDatabase()
        schema = _make_course_schema()
        db.set_schema(schema)

        # Insert records
        await db.create(Record.from_dict({
            "title": "Intro to Algorithms",
            "department": "CS",
            "level": 100,
            "active": True,
            "description": "Learn basic algorithms and data structures.",
        }))
        await db.create(Record.from_dict({
            "title": "Linear Algebra",
            "department": "Math",
            "level": 200,
            "active": True,
            "description": "Matrix operations and vector spaces.",
        }))

        source = DatabaseSource(
            db=db,
            schema=schema,
            name="courses",
            content_field="description",
        )

        intent = RetrievalIntent(
            text_queries=[],
            filters={"courses": {"department": "CS"}},
        )
        results = await source.query(intent)
        assert len(results) == 1
        assert results[0].source_name == "courses"
        assert results[0].source_type == "database"
        assert "algorithms" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        db = AsyncMemoryDatabase()
        source = DatabaseSource(
            db=db, schema=DatabaseSchema.create(name=FieldType.STRING), name="empty",
        )
        results = await source.query(RetrievalIntent(text_queries=["nothing"]))
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self) -> None:
        db = AsyncMemoryDatabase()
        schema = DatabaseSchema.create(name=FieldType.STRING, content=FieldType.TEXT)
        db.set_schema(schema)

        for i in range(10):
            await db.create(Record.from_dict({
                "name": f"record_{i}",
                "content": f"Content for record {i}",
            }))

        source = DatabaseSource(db=db, schema=schema, name="test", content_field="content")
        results = await source.query(RetrievalIntent(), top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_source_properties(self) -> None:
        source = DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=DatabaseSchema.create(name=FieldType.STRING),
            name="my_db",
        )
        assert source.name == "my_db"
        assert source.source_type == "database"

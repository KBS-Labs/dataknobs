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

    def test_text_search_not_in_structural_query(self) -> None:
        """Text search uses OR semantics via retrieve(), not _build_query()."""
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=["algorithms"],
            filters={},
        )
        query = source._build_query(intent)
        # _build_query returns only structural filters; text search
        # is handled separately in _text_or_search with OR semantics
        assert len(query.filters) == 0

    def test_combined_structural_only(self) -> None:
        source = self._make_source()
        intent = RetrievalIntent(
            text_queries=["intro"],
            filters={"courses": {"department": "CS", "level": 100}},
        )
        query = source._build_query(intent)
        # Only structural filters in the query (text search handled via OR)
        assert len(query.filters) == 2

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

    @pytest.mark.asyncio
    async def test_text_search_or_semantics(self) -> None:
        """Multiple text queries use OR — matching any query counts."""
        db = AsyncMemoryDatabase()
        schema = _make_course_schema()
        db.set_schema(schema)

        await db.create(Record.from_dict({
            "title": "Algorithms",
            "department": "CS",
            "level": 100,
            "active": True,
            "description": "Learn about algorithms.",
        }))
        await db.create(Record.from_dict({
            "title": "Calculus",
            "department": "Math",
            "level": 200,
            "active": True,
            "description": "Derivatives and integrals.",
        }))

        source = DatabaseSource(
            db=db, schema=schema, name="courses",
            content_field="description",
            text_search_fields=["title", "description"],
        )
        # Two queries — should find both records (OR, not AND)
        results = await source.query(RetrievalIntent(
            text_queries=["algorithms", "calculus"],
        ))
        assert len(results) == 2
        titles = {r.metadata.get("title") for r in results}
        assert "Algorithms" in titles
        assert "Calculus" in titles


class TestDatabaseSourceRelevanceScoring:
    """Test term-coverage relevance scoring."""

    @pytest.mark.asyncio
    async def test_full_match_scores_high(self) -> None:
        """Record matching all query terms scores near 1.0."""
        db = AsyncMemoryDatabase()
        schema = DatabaseSchema.create(
            title=FieldType.STRING,
            summary=FieldType.TEXT,
        )
        db.set_schema(schema)
        await db.create(Record.from_dict({
            "title": "OAuth Grant Types",
            "summary": "OAuth 2.0 defines several grant types for authorization.",
        }))

        source = DatabaseSource(
            db=db, schema=schema, name="docs",
            content_field="summary",
            text_search_fields=["title", "summary"],
        )
        results = await source.query(RetrievalIntent(
            text_queries=["OAuth"],
        ))
        assert len(results) == 1
        # "OAuth" appears in both title and summary → high score
        assert results[0].relevance > 0.5

    def test_partial_match_scores_lower_than_full(self) -> None:
        """Unit test: partial term coverage scores lower than full."""
        source = DatabaseSource(
            db=AsyncMemoryDatabase(),
            schema=DatabaseSchema.create(title=FieldType.STRING, body=FieldType.TEXT),
            name="docs",
            content_field="body",
            text_search_fields=["title", "body"],
        )
        data = {"title": "OAuth Grants", "body": "OAuth grant types overview."}

        # All terms match
        full_score = source._score_record(data, ["OAuth"])
        # Only 1 of 2 terms match
        partial_score = source._score_record(data, ["OAuth", "refresh tokens"])

        assert partial_score < full_score
        assert partial_score > 0.0

    @pytest.mark.asyncio
    async def test_no_text_queries_scores_1(self) -> None:
        """Records matched by filters only get relevance=1.0."""
        db = AsyncMemoryDatabase()
        schema = DatabaseSchema.create(dept=FieldType.STRING, summary=FieldType.TEXT)
        db.set_schema(schema)
        await db.create(Record.from_dict({"dept": "CS", "summary": "Algorithms"}))

        source = DatabaseSource(
            db=db, schema=schema, name="courses",
            content_field="summary",
        )
        results = await source.query(RetrievalIntent(
            filters={"courses": {"dept": "CS"}},
        ))
        assert len(results) == 1
        assert results[0].relevance == 1.0

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance(self) -> None:
        """Results are returned sorted by relevance descending."""
        db = AsyncMemoryDatabase()
        schema = DatabaseSchema.create(
            title=FieldType.STRING,
            body=FieldType.TEXT,
        )
        db.set_schema(schema)
        await db.create(Record.from_dict({
            "title": "OAuth Security",
            "body": "OAuth security considerations and threat model.",
        }))
        await db.create(Record.from_dict({
            "title": "Unrelated Topic",
            "body": "This mentions OAuth only in passing.",
        }))

        source = DatabaseSource(
            db=db, schema=schema, name="docs",
            content_field="body",
            text_search_fields=["title", "body"],
        )
        results = await source.query(RetrievalIntent(
            text_queries=["OAuth", "security"],
        ))
        if len(results) >= 2:
            assert results[0].relevance >= results[1].relevance

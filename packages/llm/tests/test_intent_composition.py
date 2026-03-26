"""Tests for intent schema composition and parsing."""

from __future__ import annotations

from typing import Any

from dataknobs_data.sources.base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
    SourceSchema,
)
from dataknobs_llm.sources.intent import compose_intent_schema, parse_intent


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


class StubSource(GroundedSource):
    """Minimal source for testing schema composition."""

    def __init__(
        self,
        name: str,
        schema: SourceSchema | None = None,
    ) -> None:
        self._name = name
        self._schema = schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return "stub"

    def get_schema(self) -> SourceSchema | None:
        return self._schema

    async def query(
        self,
        intent: RetrievalIntent,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SourceResult]:
        return []


# ------------------------------------------------------------------
# Schema composition tests
# ------------------------------------------------------------------


class TestComposeIntentSchema:
    """Test compose_intent_schema()."""

    def test_base_schema_always_present(self) -> None:
        schema = compose_intent_schema([])
        props = schema["properties"]
        assert "text_queries" in props
        assert "scope" in props
        assert props["text_queries"]["type"] == "array"
        assert "broad" in props["scope"]["enum"]
        assert schema["required"] == ["text_queries"]

    def test_no_filter_source_adds_nothing(self) -> None:
        """A source with no schema doesn't add properties."""
        source = StubSource("docs", schema=None)
        schema = compose_intent_schema([source])
        props = schema["properties"]
        assert "docs" not in props
        assert "text_queries" in props

    def test_single_source_with_filters(self) -> None:
        source = StubSource("courses", schema=SourceSchema(
            source_name="courses",
            fields={
                "department": {
                    "type": "string",
                    "enum": ["CS", "Math"],
                    "x-extraction": {"normalize": True},
                },
                "level": {"type": "integer"},
            },
            description="Course catalog",
        ))
        schema = compose_intent_schema([source])
        props = schema["properties"]

        assert "courses" in props
        assert props["courses"]["type"] == "object"
        assert "department" in props["courses"]["properties"]
        assert props["courses"]["properties"]["department"]["enum"] == ["CS", "Math"]
        assert props["courses"]["description"] == "Course catalog"

    def test_multiple_sources_namespaced(self) -> None:
        """Each source's fields are nested under its name."""
        source_a = StubSource("docs", schema=SourceSchema(
            source_name="docs",
            fields={"category": {"type": "string"}},
        ))
        source_b = StubSource("courses", schema=SourceSchema(
            source_name="courses",
            fields={"department": {"type": "string"}},
        ))
        schema = compose_intent_schema([source_a, source_b])
        props = schema["properties"]

        assert "docs" in props
        assert "courses" in props
        assert "category" in props["docs"]["properties"]
        assert "department" in props["courses"]["properties"]

    def test_domain_context_appended(self) -> None:
        schema = compose_intent_schema([], domain_context="OAuth 2.0")
        desc = schema["properties"]["text_queries"]["description"]
        assert "OAuth 2.0" in desc

    def test_source_required_fields_propagated(self) -> None:
        source = StubSource("db", schema=SourceSchema(
            source_name="db",
            fields={"status": {"type": "string"}},
            required_fields=["status"],
        ))
        schema = compose_intent_schema([source])
        assert schema["properties"]["db"]["required"] == ["status"]

    def test_empty_fields_source_skipped(self) -> None:
        """Source with empty fields dict doesn't add a property."""
        source = StubSource("empty", schema=SourceSchema(
            source_name="empty",
            fields={},
        ))
        schema = compose_intent_schema([source])
        assert "empty" not in schema["properties"]

    def test_mixed_filter_and_no_filter_sources(self) -> None:
        """Only sources with schemas add properties."""
        no_filter = StubSource("kb", schema=None)
        with_filter = StubSource("db", schema=SourceSchema(
            source_name="db",
            fields={"name": {"type": "string"}},
        ))
        schema = compose_intent_schema([no_filter, with_filter])
        assert "kb" not in schema["properties"]
        assert "db" in schema["properties"]


# ------------------------------------------------------------------
# Intent parsing tests
# ------------------------------------------------------------------


class TestParseIntent:
    """Test parse_intent()."""

    def test_basic_parsing(self) -> None:
        data = {
            "text_queries": ["OAuth grant types", "authorization code"],
            "scope": "focused",
        }
        intent = parse_intent(data)
        assert intent.text_queries == ["OAuth grant types", "authorization code"]
        assert intent.scope == "focused"
        assert intent.filters == {}
        assert intent.raw_data == data

    def test_source_filters_extracted(self) -> None:
        data = {
            "text_queries": ["algorithms"],
            "scope": "exact",
            "courses": {"department": "CS", "level": 100},
        }
        intent = parse_intent(data)
        assert "courses" in intent.filters
        assert intent.filters["courses"]["department"] == "CS"
        assert intent.filters["courses"]["level"] == 100

    def test_multiple_source_filters(self) -> None:
        data = {
            "text_queries": ["security"],
            "docs": {"category": "auth"},
            "events": {"date_after": "2024-01-01"},
        }
        intent = parse_intent(data)
        assert len(intent.filters) == 2
        assert intent.filters["docs"]["category"] == "auth"
        assert intent.filters["events"]["date_after"] == "2024-01-01"

    def test_defaults_when_missing(self) -> None:
        intent = parse_intent({})
        assert intent.text_queries == []
        assert intent.scope == "focused"
        assert intent.filters == {}

    def test_string_text_queries_wrapped_in_list(self) -> None:
        """Single string text_queries is wrapped in a list."""
        intent = parse_intent({"text_queries": "single query"})
        assert intent.text_queries == ["single query"]

    def test_non_dict_values_not_treated_as_filters(self) -> None:
        """Only dict values become source filters."""
        data = {
            "text_queries": ["test"],
            "courses": {"department": "CS"},
            "extra_string": "not a filter",
            "extra_list": [1, 2, 3],
        }
        intent = parse_intent(data)
        assert "courses" in intent.filters
        assert "extra_string" not in intent.filters
        assert "extra_list" not in intent.filters

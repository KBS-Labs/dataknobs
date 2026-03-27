"""Database-backed grounded source.

Wraps an :class:`AsyncDatabase` with a :class:`DatabaseSchema` to provide
structured retrieval via the :class:`GroundedSource` interface.

The source auto-generates a JSON schema fragment from the database schema
so that an intent extractor knows which fields can be filtered on and
what values are valid.  At query time, extracted intent filters are
translated deterministically to :class:`Query` / :class:`Filter` objects —
no LLM generates SQL or query DSL.
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_data.database import AsyncDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.schema import DatabaseSchema

from .base import GroundedSource, RetrievalIntent, SourceResult, SourceSchema

logger = logging.getLogger(__name__)

# FieldType → JSON schema type mapping
_FIELD_TYPE_MAP: dict[FieldType, str] = {
    FieldType.STRING: "string",
    FieldType.TEXT: "string",
    FieldType.INTEGER: "integer",
    FieldType.FLOAT: "number",
    FieldType.BOOLEAN: "boolean",
    FieldType.DATETIME: "string",
    FieldType.JSON: "object",
}

# Field types we skip in schema generation (not filterable)
_SKIP_TYPES: set[FieldType] = {
    FieldType.VECTOR,
    FieldType.SPARSE_VECTOR,
    FieldType.BINARY,
}


class DatabaseSource(GroundedSource):
    """Grounded source backed by an :class:`AsyncDatabase`.

    Auto-generates a :class:`SourceSchema` from the database's
    :class:`DatabaseSchema`, translating field types and enum metadata
    into JSON schema properties with ``x-extraction`` hints.

    At query time, :meth:`query` deterministically maps
    :class:`RetrievalIntent` filters to :class:`Query` with
    :class:`Filter` objects, then executes via ``db.search()``.

    Args:
        db: The async database to query.
        schema: Database schema describing field types.
        name: Unique source name (used in intent filter namespacing).
        content_field: Field whose value becomes
            :attr:`SourceResult.content`.  Defaults to ``"content"``.
        text_search_fields: Fields to apply ``LIKE`` text search on
            when ``text_queries`` are present in the intent.
        description: Human-readable description for the extraction
            prompt.

    Example::

        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        from dataknobs_data.schema import DatabaseSchema
        from dataknobs_data.fields import FieldType

        schema = DatabaseSchema.create(
            title=FieldType.STRING,
            department=FieldType.STRING,
            level=FieldType.INTEGER,
            description=FieldType.TEXT,
        )
        schema.fields["department"].metadata["enum"] = ["CS", "Math", "Physics"]

        db = AsyncMemoryDatabase()
        db.set_schema(schema)

        source = DatabaseSource(
            db=db,
            schema=schema,
            name="courses",
            content_field="description",
            text_search_fields=["title", "description"],
        )
    """

    def __init__(
        self,
        db: AsyncDatabase,
        schema: DatabaseSchema,
        *,
        name: str = "database",
        content_field: str = "content",
        text_search_fields: list[str] | None = None,
        description: str = "",
    ) -> None:
        self._db = db
        self._schema = schema
        self._name = name
        self._content_field = content_field
        self._text_search_fields = text_search_fields or []
        self._description = description

    # ------------------------------------------------------------------
    # GroundedSource interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return "database"

    def get_schema(self) -> SourceSchema:
        """Auto-generate JSON schema properties from database fields.

        Maps :class:`FieldType` to JSON schema types.  Adds
        ``x-extraction`` hints for fields with enum metadata.
        Datetime fields produce ``{field}_after`` / ``{field}_before``
        range filter properties.
        """
        properties: dict[str, Any] = {}

        for field_name, field_schema in self._schema.fields.items():
            if field_schema.type in _SKIP_TYPES:
                continue

            json_type = _FIELD_TYPE_MAP.get(field_schema.type)
            if json_type is None:
                continue

            prop: dict[str, Any] = {
                "type": json_type,
                "description": field_schema.metadata.get(
                    "description", f"Filter on {field_name}",
                ),
            }

            # Enum values from field metadata
            enum_values = field_schema.metadata.get("enum")
            if enum_values:
                prop["enum"] = list(enum_values)
                prop["x-extraction"] = {"normalize": True}

            # Datetime fields → range filters
            if field_schema.type == FieldType.DATETIME:
                prop["format"] = "date-time"
                properties[f"{field_name}_after"] = {
                    "type": "string",
                    "format": "date-time",
                    "description": f"Only include records where {field_name} is after this date",
                }
                properties[f"{field_name}_before"] = {
                    "type": "string",
                    "format": "date-time",
                    "description": f"Only include records where {field_name} is before this date",
                }
                continue  # Don't add the raw datetime field as a direct filter

            properties[field_name] = prop

        return SourceSchema(
            source_name=self._name,
            fields=properties,
            description=self._description,
        )

    async def query(
        self,
        intent: RetrievalIntent,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SourceResult]:
        """Build a :class:`Query` from intent and execute it.

        Relevance scoring is based on term coverage: what fraction
        of the ``text_queries`` appear in the record's searchable
        fields, with 2x weight for the content field.  Records matched
        by structured filters alone receive ``relevance=1.0``.

        Translation rules (deterministic):

        - String field + single value → ``Filter(field, EQ, value)``
        - String field + list → ``Filter(field, IN, values)``
        - Integer/float + single value → ``Filter(field, EQ, value)``
        - Integer/float + range ``{min, max}`` → ``GTE`` + ``LTE`` filters
        - Boolean → ``Filter(field, EQ, value)``
        - ``{field}_after`` / ``{field}_before`` → ``GTE`` / ``LTE`` filters
        - ``text_queries`` + ``text_search_fields`` →
          ``Filter(field, LIKE, f"%{q}%")`` per query per field,
          combined with OR semantics (any match counts)
        """
        records = await self._text_or_search(intent, top_k)

        results: list[SourceResult] = []
        for record in records:
            data = record.to_dict() if hasattr(record, "to_dict") else {}
            content = str(data.get(self._content_field, ""))
            record_id = str(record.id) if hasattr(record, "id") and record.id else ""
            relevance = self._score_record(data, intent.text_queries)
            results.append(SourceResult(
                content=content,
                source_id=record_id,
                source_name=self._name,
                source_type="database",
                relevance=relevance,
                metadata=data,
            ))

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results

    async def _text_or_search(
        self,
        intent: RetrievalIntent,
        top_k: int,
    ) -> list[Any]:
        """Run text search with OR semantics across queries and fields.

        Structural filters are AND-combined in every sub-query.  Text
        search runs one sub-query per (text_query, text_field) pair so
        that a record matching *any* query in *any* field is included.
        Results are deduplicated by record id.

        When there are no text queries (or no text_search_fields), a
        single structural-only query is executed.
        """
        structural_filters = self._build_structural_filters(intent)

        if not intent.text_queries or not self._text_search_fields:
            query = Query(filters=structural_filters).limit(top_k)
            try:
                return await self._db.search(query)
            except Exception:
                logger.warning(
                    "Database query failed for source '%s'",
                    self._name, exc_info=True,
                )
                return []

        # OR across (text_query x text_field) combinations
        seen_ids: set[str] = set()
        records: list[Any] = []
        for tq in intent.text_queries:
            for text_field in self._text_search_fields:
                text_filter = Filter(
                    field=text_field,
                    operator=Operator.LIKE,
                    value=f"%{tq}%",
                )
                q = Query(
                    filters=structural_filters + [text_filter],
                ).limit(top_k)
                try:
                    batch = await self._db.search(q)
                except Exception:
                    logger.warning(
                        "Database text search failed for source '%s' "
                        "(query=%r, field=%s)",
                        self._name, tq, text_field, exc_info=True,
                    )
                    continue
                for r in batch:
                    rid = str(r.id) if hasattr(r, "id") and r.id else ""
                    if rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        records.append(r)
                    elif not rid:
                        records.append(r)
        return records

    async def close(self) -> None:
        """Close the underlying database connection."""
        if hasattr(self._db, "close"):
            await self._db.close()

    # ------------------------------------------------------------------
    # Query building (deterministic)
    # ------------------------------------------------------------------

    def _build_structural_filters(
        self, intent: RetrievalIntent,
    ) -> list[Filter]:
        """Build structural (non-text-search) filters from intent.

        These are AND-combined.  Text search filters use OR semantics
        and are handled separately in :meth:`retrieve`.
        """
        filters: list[Filter] = []
        source_filters = intent.filters.get(self._name, {})
        for field_name, value in source_filters.items():
            new_filters = self._translate_filter(field_name, value)
            filters.extend(new_filters)
        return filters

    def _build_query(self, intent: RetrievalIntent) -> Query:
        """Translate intent filters to a :class:`Query`.

        Reads only the filter slice for this source
        (``intent.filters.get(self.name, {})``).

        Structural filters are AND-combined.  Text search uses OR
        semantics across queries and fields — see :meth:`retrieve`.
        """
        return Query(filters=self._build_structural_filters(intent))

    def _translate_filter(
        self,
        field_name: str,
        value: Any,
    ) -> list[Filter]:
        """Translate a single intent filter to one or more :class:`Filter` objects."""
        # Range filter for datetime fields: {field}_after / {field}_before
        if field_name.endswith("_after"):
            base_field = field_name[: -len("_after")]
            return [Filter(field=base_field, operator=Operator.GTE, value=value)]
        if field_name.endswith("_before"):
            base_field = field_name[: -len("_before")]
            return [Filter(field=base_field, operator=Operator.LTE, value=value)]

        # Range dict: {"min": x, "max": y}
        if isinstance(value, dict) and ("min" in value or "max" in value):
            result: list[Filter] = []
            if "min" in value:
                result.append(Filter(field=field_name, operator=Operator.GTE, value=value["min"]))
            if "max" in value:
                result.append(Filter(field=field_name, operator=Operator.LTE, value=value["max"]))
            return result

        # List → IN
        if isinstance(value, list):
            return [Filter(field=field_name, operator=Operator.IN, value=value)]

        # Scalar → EQ
        return [Filter(field=field_name, operator=Operator.EQ, value=value)]

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _score_record(
        self,
        data: dict[str, Any],
        text_queries: list[str],
    ) -> float:
        """Compute a term-coverage relevance score for a database record.

        Scoring is based on how many query terms appear in the record's
        searchable fields.  The content field receives a 2x weight
        relative to secondary text search fields.

        Returns a score in the range (0.0, 1.0].  Records with no text
        queries to score against receive 1.0 (all-match, since they
        were selected by structured filters alone).
        """
        if not text_queries:
            return 1.0

        total_weight = 0.0
        matched_weight = 0.0

        for query in text_queries:
            query_lower = query.lower()

            # Content field gets double weight
            content_val = str(data.get(self._content_field, "")).lower()
            total_weight += 2.0
            if query_lower in content_val:
                matched_weight += 2.0

            # Secondary text search fields get weight 1.0 each
            for field_name in self._text_search_fields:
                if field_name == self._content_field:
                    continue  # Already scored above
                field_val = str(data.get(field_name, "")).lower()
                total_weight += 1.0
                if query_lower in field_val:
                    matched_weight += 1.0

        if total_weight == 0.0:
            return 1.0

        # Scale to (0.0, 1.0] — a small floor ensures filter-matched
        # records never get 0.0 (they passed the DB query for a reason).
        raw = matched_weight / total_weight
        return max(raw, 0.05)

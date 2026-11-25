# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Elasticsearch backend implementation for the data package."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import SyncDatabase, version_conflict_error
from ..exceptions import DuplicateRecordError
from ..query import Query
from ..query_logic import ComplexQuery
from ..streaming import StreamConfig, StreamingMixin, StreamResult
from ..vector import SyncVectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.mixins import resolve_metric
from ..vector.types import DistanceMetric, VectorSearchResult
from .config import SyncElasticsearchDatabaseConfig
from .elasticsearch_errors import bulk_outcomes, is_existing_index, is_missing_document
from .elasticsearch_mixins import (
    ElasticsearchBaseConfig,
    ElasticsearchErrorHandler,
    ElasticsearchIndexManager,
    ElasticsearchQueryBuilder,
    ElasticsearchRecordSerializer,
    ElasticsearchVectorSupport,
    es_version_token,
    parse_es_version_token,
    vector_tracking_metadata,
)
from .elasticsearch_query import (
    PIT_KEEP_ALIVE,
    FieldTypes,
    build_query,
    build_sort,
    client_search_kwargs,
    plan_search,
    query_fields,
    run_search_plan,
)
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import ClassVar

    import numpy as np
    from elasticsearch import Elasticsearch

    from ..records import Record

logger = logging.getLogger(__name__)


class SyncElasticsearchDatabase(
    StructuredConfigConsumer[SyncElasticsearchDatabaseConfig],
    SyncDatabase,
    StreamingMixin,
    VectorConfigMixin,
    BulkEmbedMixin,
    SyncVectorOperationsMixin,
    ElasticsearchBaseConfig,
    ElasticsearchIndexManager,
    ElasticsearchVectorSupport,
    ElasticsearchErrorHandler,
    ElasticsearchRecordSerializer,
    ElasticsearchQueryBuilder,
):
    """Synchronous Elasticsearch database backend.

    Constructed through :class:`SyncElasticsearchDatabaseConfig` — every
    documented config key is a typed field on that dataclass, so
    ``self.config`` is the typed config (not a dict) and the
    ``from_config`` / factory paths share one construction route.

    **It carried no vector mixin at all** until ``vector_search`` moved onto
    one. Its async twin has had :class:`AsyncVectorOperationsMixin` and
    :class:`AsyncBulkEmbedMixin` throughout; this class defined
    ``vector_search`` and ``create_vector_index`` from nowhere and had none
    of the other eight members of that surface. ``bulk_embed_and_store``,
    ``update_vector``, ``delete_from_index``, ``drop_vector_index``,
    ``get_vector_index_stats`` and ``hybrid_search`` raised
    ``AttributeError`` here and answered on the twin, and the two private
    helpers ``hybrid_search`` calls --- ``_text_search_for_hybrid`` and
    ``_supports_native_hybrid`` --- were missing with it.

    Mixing the two in is what lets the one ``vector_search`` reach this
    class; that the other eight arrive with it is the twin-parity gap
    closing, not a side effect to be undone.

    **The gap closes on presence, and not yet on behaviour.** The async twin
    overrides ``hybrid_search``, ``_supports_native_hybrid`` and
    ``_text_search_for_hybrid`` to use Elasticsearch's own RRF in a single
    request; this class takes the mixin's implementations, so it answers
    ``False`` to native support and fuses two rankings client-side --- a
    ``LIKE`` text arm through the shared filter translation, then the same
    ``fuse_hybrid_results`` the other eight backends use. That is a correct
    hybrid search rather than a stub, and it is slower and worse at text than
    what the twin does. Porting the native path is a capability gap, recorded
    here rather than left for a reader to infer from three absent methods.
    """

    CONFIG_CLS: ClassVar[type[SyncElasticsearchDatabaseConfig]] = SyncElasticsearchDatabaseConfig

    def _setup(self) -> None:
        """Derive backend attributes from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op for Elasticsearch — connection setup
        is deferred to :meth:`connect`).
        """
        cfg = self.config
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)

        # Observed vector fields (field_name -> dimensions).
        self.vector_fields: dict[str, int] = {}

        self.host = cfg.host
        self.port = cfg.port
        self.index_name = cfg.index
        self.refresh = cfg.refresh
        self.max_result_window = cfg.max_result_window
        self.search_page_size = cfg.search_page_size

        self._client: Elasticsearch | None = None
        self._connected = False
        self._known_types = FieldTypes()

    def connect(self) -> None:
        """Connect to the Elasticsearch database."""
        if self._connected:
            return  # Already connected

        cfg = self.config

        # If vector is enabled but no vector fields defined yet, set up default
        if self._vector_enabled and not self.vector_fields:
            self.vector_fields[cfg.default_vector_field] = cfg.vector_dimensions

        mappings, settings = self.index_definition(cfg.mappings, cfg.settings, self.vector_fields)

        from elasticsearch import Elasticsearch

        self._client = Elasticsearch([f"http://{self.host}:{self.port}"])
        try:
            if not self._index_exists():
                self._create_index(mappings, settings)
        except Exception:
            self._client.close()
            self._client = None
            raise
        self._connected = True

    def close(self) -> None:
        """Close the database connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._connected = False
        self._known_types = FieldTypes()

    def _initialize(self) -> None:
        """Initialize method - connection setup moved to connect()."""
        # Configuration parsing stays here if needed
        pass

    def _index_exists(self) -> bool:
        """Whether this database's index is already there."""
        return bool(self._require_client().indices.exists(index=self.index_name))

    def _create_index(self, mappings: dict[str, Any], settings: dict[str, Any]) -> bool:
        """Create the index; ``False`` when another process created it first."""
        from elasticsearch import BadRequestError

        try:
            self._require_client().indices.create(
                index=self.index_name, mappings=mappings, settings=settings
            )
        except BadRequestError as e:
            if is_existing_index(e):
                return False
            raise
        return True

    def _check_connection(self) -> None:
        """Check if database is connected."""
        self._require_client()

    def _require_client(self) -> Elasticsearch:
        """The connected client, or ``RuntimeError`` when there is none."""
        if self._client is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._client

    def _record_to_doc(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to an Elasticsearch document."""
        # Create a copy of the record to avoid modifying the original
        record_copy = record.copy(deep=True)

        # Update vector tracking if needed
        if self._has_vector_fields(record_copy):
            self._update_vector_tracking(record_copy)

            # Add vector field metadata to copied record metadata
            if "vector_fields" not in record_copy.metadata:
                record_copy.metadata["vector_fields"] = {}

            for field_name in self.vector_fields:
                if field_name in record_copy.fields:
                    field = record_copy.fields[field_name]
                    if hasattr(field, "source_field"):
                        record_copy.metadata["vector_fields"][field_name] = (
                            vector_tracking_metadata(field, self.vector_fields[field_name])
                        )

        doc = self._record_to_document(record_copy)
        if id:
            doc["id"] = id
        elif not doc.get("id"):
            doc["id"] = self._generate_id()

        return doc

    def _doc_to_record(self, doc: dict[str, Any]) -> Record:
        """Convert an Elasticsearch document to a Record."""
        # Handle both direct documents and search results
        if "_source" in doc:
            source_doc = doc
        else:
            source_doc = {"_source": doc}

        record = self._document_to_record(source_doc)

        # Add score if present
        if "_score" in doc:
            record.metadata["_score"] = doc.get("_score")

        return record

    def create(self, record: Record) -> str:
        """Create a new record."""
        # Use record's ID if it has one, otherwise mint via the shared hook
        id = record.id if record.id else self._generate_id()
        doc = self._record_to_doc(record, id)

        # op_type="create" makes this an atomic insert: a colliding id is a 409
        # conflict rather than an overwrite.
        from elasticsearch import ConflictError

        try:
            response = self._require_client().index(
                index=self.index_name,
                id=id,
                document=doc,
                refresh=self.refresh,
                op_type="create",
            )
        except ConflictError as e:
            raise DuplicateRecordError(id) from e

        return str(response["_id"])

    def _get_document(self, id: str, *, source: bool = True) -> dict[str, Any] | None:
        """GET one document, or ``None`` when it does not exist.

        Every other failure raises, a missing index included.
        """
        client = self._require_client()
        if not id:
            # Not an id Elasticsearch can hold, so nothing is stored under it.
            return None

        from elasticsearch import NotFoundError

        try:
            response = client.get(index=self.index_name, id=id, source=source)
        except NotFoundError as e:
            if is_missing_document(e):
                return None
            raise
        return dict(response)

    def read(self, id: str) -> Record | None:
        """Read a record by ID; ``None`` when it does not exist."""
        document = self._get_document(id)
        if document is None:
            return None
        return self._doc_to_record(document.get("_source", {}))

    def get_version(self, id: str) -> str | None:
        """Return the document's ``_seq_no``/``_primary_term`` version token.

        Elasticsearch's native optimistic-concurrency pair
        (``_seq_no``, ``_primary_term``) advances on every write, so it is a
        native version — ABA-safe, unlike the base content-hash default this
        overrides. The two values are combined into one opaque token.
        """
        document = self._get_document(id, source=False)
        if document is None:
            return None
        seq_no = document.get("_seq_no")
        primary_term = document.get("_primary_term")
        if seq_no is None or primary_term is None:
            return None
        return es_version_token(seq_no, primary_term)

    def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record; ``False`` when it does not exist.

        When ``expected_version`` is provided the update carries ES's
        ``if_seq_no``/``if_primary_term`` guards so the compare-and-set is
        enforced server-side; a stale token raises ``ConcurrencyError``. When
        ``None`` the update is unconditional. Every other failure raises.
        """
        client = self._require_client()
        if not id:
            return False
        # The target id is stamped so a partial update keeps the top-level id
        # field consistent with ``_id``.
        doc = self._record_to_doc(record, id)

        from elasticsearch import ConflictError, NotFoundError

        seq_no: int | None = None
        primary_term: int | None = None
        if expected_version is not None:
            seq_no, primary_term = parse_es_version_token(expected_version)
        try:
            client.update(
                index=self.index_name,
                id=id,
                doc=doc,
                refresh=self.refresh,
                if_seq_no=seq_no,
                if_primary_term=primary_term,
            )
        except NotFoundError as e:
            if is_missing_document(e):
                return False
            raise
        except ConflictError as e:
            if expected_version is None:
                raise
            # Either the token is stale, or the document has gone since.
            current = self.get_version(id)
            if current is None:
                return False
            raise version_conflict_error(id, expected_version, current) from e
        return True

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID; ``False`` when it does not exist.

        When ``expected_version`` is provided the delete carries ES's
        ``if_seq_no``/``if_primary_term`` guards so the compare-and-set is
        enforced server-side; a stale token raises ``ConcurrencyError``. When
        ``None`` the delete is unconditional. Every other failure raises.
        """
        client = self._require_client()
        if not id:
            return False

        from elasticsearch import ConflictError, NotFoundError

        seq_no: int | None = None
        primary_term: int | None = None
        if expected_version is not None:
            seq_no, primary_term = parse_es_version_token(expected_version)
        try:
            client.delete(
                index=self.index_name,
                id=id,
                refresh=self.refresh,
                if_seq_no=seq_no,
                if_primary_term=primary_term,
            )
        except NotFoundError as e:
            if is_missing_document(e):
                return False
            raise
        except ConflictError as e:
            if expected_version is None:
                raise
            # Either the token is stale, or the document has gone since.
            current = self.get_version(id)
            if current is None:
                return False
            raise version_conflict_error(id, expected_version, current) from e
        return True

    def exists(self, id: str) -> bool:
        """Check if a record exists.

        A GET without the source rather than a HEAD, whose 404 has no body to
        tell a missing record from a missing index.
        """
        return self._get_document(id, source=False) is not None

    def upsert(
        self,
        id_or_record: str | Record,
        record: Record | None = None,
        *,
        expected_version: str | None = None,
    ) -> str:
        """Update or insert a record.

        Can be called as:
        - upsert(id, record) - explicit ID and record
        - upsert(record) - extract ID from record using Record's built-in logic

        When ``expected_version`` is provided the upsert is conditional: the
        record must already exist with a matching version token, otherwise it
        raises ``ConcurrencyError``. A conditional upsert never inserts.
        """
        # Resolve the storage id via the shared helper (honors an explicit id;
        # mints via the overridable _generate_id() hook when the record has none).
        id, record = self._resolve_upsert_id(id_or_record, record)

        if expected_version is not None:
            # A conditional upsert never inserts. Delegate to update()'s
            # server-side seq_no/primary_term compare-and-set: a True return is
            # the update; a stale token raises straight out; a False return
            # means the doc is absent, which for a conditional upsert is itself
            # a conflict. Acting on the return (not a separate exists() probe)
            # closes the exists()->update() TOCTOU.
            if self.update(id, record, expected_version=expected_version):
                return id
            raise version_conflict_error(id, expected_version, None)

        doc = self._record_to_doc(record, id)
        self._require_client().index(
            index=self.index_name, id=id, document=doc, refresh=self.refresh
        )
        return id

    def _bulk(self, actions: list[dict[str, Any]], operation: str) -> list[bool]:
        """Run bulk actions and report each one; see :func:`bulk_outcomes`.

        Sent in requests of up to 500 actions. A request that fails as a whole
        raises, and the ones before it have been applied.
        """
        from elasticsearch import helpers

        client = self._require_client()
        items = [
            item
            for _, item in helpers.streaming_bulk(
                client, actions, refresh=self.refresh, raise_on_error=False
            )
        ]
        return bulk_outcomes(items, operation)

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records using the bulk API, failing closed on a colliding id.

        Uses the bulk ``create`` op, honoring a caller-supplied ``record.id``
        and minting one only when absent. The bulk API is per-item
        (non-atomic), so the other records in the batch have been indexed when
        either error is raised.

        Args:
            records: List of records to create

        Returns:
            The created record ids, in input order

        Raises:
            DuplicateRecordError: a record collides with an existing id, or two
                records in the batch share an id.
            DatabaseError: any other record could not be indexed.
        """
        ids = [record.id or self._generate_id() for record in records]
        seen: set[str] = set()
        for record_id in ids:
            if record_id in seen:
                raise DuplicateRecordError(record_id)
            seen.add(record_id)
        self._bulk_index("create", ids, records)
        return ids

    def upsert_batch(self, records: list[Record]) -> list[str]:
        """Insert-or-overwrite multiple records using the bulk API.

        Uses the bulk ``index`` op, honoring a caller-supplied ``record.id``
        and minting one only when absent; a colliding id is overwritten. The
        bulk API is per-item (non-atomic), so the other records in the batch
        have been indexed when the error is raised.

        Returns:
            The written record ids, in input order

        Raises:
            DatabaseError: a record could not be indexed.
        """
        ids = [record.id or self._generate_id() for record in records]
        self._bulk_index("index", ids, records)
        return ids

    def _bulk_index(self, op_type: str, ids: list[str], records: list[Record]) -> None:
        """Index records under ids with the given bulk op."""
        actions = [
            {
                "_op_type": op_type,
                "_index": self.index_name,
                "_id": record_id,
                "_source": self._record_to_doc(record, record_id),
            }
            for record_id, record in zip(ids, records, strict=True)
        ]
        self._bulk(actions, "create_batch" if op_type == "create" else "upsert_batch")

    def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records in batch."""
        records = []
        for id in ids:
            record = self.read(id)
            records.append(record)
        return records

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records using the bulk API.

        Args:
            ids: List of record IDs to delete

        Returns:
            ``True`` per record deleted, ``False`` per record that does not exist

        Raises:
            DatabaseError: a delete failed for any other reason. The bulk API
                is per-item, so the others have been applied.
        """
        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": record_id} for record_id in ids
        ]
        return self._bulk(actions, "delete_batch")

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records using the bulk API.

        Args:
            updates: List of (id, record) tuples to update

        Returns:
            ``True`` per record updated, ``False`` per record that does not exist

        Raises:
            DatabaseError: an update failed for any other reason. The bulk API
                is per-item, so the others have been applied.
        """
        actions = [
            {
                "_op_type": "update",
                "_index": self.index_name,
                "_id": record_id,
                "doc": self._record_to_doc(record, record_id),
            }
            for record_id, record in updates
        ]
        return self._bulk(actions, "update_batch")

    def _field_types(self, query: Query | ComplexQuery | None) -> FieldTypes:
        """How the index maps the fields ``query`` names, asking for those not known yet."""
        unknown = self._known_types.unknown(query_fields(query))
        if unknown:
            response = self._require_client().field_caps(
                index=self.index_name, fields=FieldTypes.request_fields(unknown)
            )
            self._known_types.learn(unknown, response["fields"])
        return self._known_types

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        # Translated by functions the twins share, so they cannot drift apart.
        types = self._field_types(query)
        es_query = build_query(query, types)
        sort = build_sort(query.sort_specs, types)

        # One request inside the result window, ``search_after`` pages past
        # it or for an unbounded read; the plan is shared with the async twin.
        plan = plan_search(
            es_query,
            sort,
            query.offset_value,
            query.limit_value,
            window=self.max_result_window,
            page_size=self.search_page_size,
        )
        hits = run_search_plan(plan, self._search_hits, self._open_pit, self._close_pit)

        records = [self._doc_to_record(hit.get("_source", {})) for hit in hits]

        # Apply field projection if specified
        if query.fields:
            for record in records:
                # Keep only specified fields
                field_names = list(record.fields.keys())
                for field_name in field_names:
                    if field_name not in query.fields:
                        del record.fields[field_name]

        return records

    def _search_hits(self, body: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        """Send one planned search request; return its hits and point-in-time id."""
        response = self._require_client().search(**client_search_kwargs(body, self.index_name))
        return list(response["hits"]["hits"]), response.get("pit_id")

    def _open_pit(self) -> str:
        """Open a point in time on this index for one paged read."""
        opened = self._require_client().open_point_in_time(
            index=self.index_name, keep_alive=PIT_KEEP_ALIVE
        )
        return str(opened["id"])

    def _close_pit(self, pit_id: str) -> None:
        """Close a paged read's point in time."""
        self._require_client().close_point_in_time(id=pit_id)

    def count(self, query: Query | None = None) -> int:
        """Count records matching a query using efficient Elasticsearch count.

        Args:
            query: Optional search query (counts all if None)

        Returns:
            Number of matching records
        """
        if not query or not query.filters:
            return self._count_all()

        # Same shared translator as search()/ComplexQuery, so count() cannot
        # drift from them (e.g. drop an operator and fall back to match_all).
        es_query = build_query(query, self._field_types(query))
        client = self._require_client()
        response = client.count(index=self.index_name, query=es_query)
        return int(response["count"])

    def _count_all(self) -> int:
        """Count all records in the database."""
        response = self._require_client().count(index=self.index_name)
        return int(response["count"])

    def clear(self) -> int:
        """Clear all records from the database."""
        client = self._require_client()
        count = self._count_all()

        response = client.delete_by_query(
            index=self.index_name, query={"match_all": {}}, refresh=self.refresh
        )
        return int(response.get("deleted", count))

    def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from Elasticsearch."""
        config = config or StreamConfig()

        # Use search to get all matching records
        if query:
            records = self.search(query)
        else:
            records = self.search(Query())

        # Yield records in batches for consistency
        for i in range(0, len(records), config.batch_size):
            batch = records[i : i + config.batch_size]
            for record in batch:
                yield record

    def stream_write(
        self, records: Iterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into Elasticsearch.

        INSERT routes through per-record ``create()`` (``insert_batch_func=None``)
        rather than the bulk ``create_batch`` fast-path: the Elasticsearch bulk
        API is non-atomic, so a partial-success batch followed by the per-record
        fallback would re-write the already-indexed rows and count them as
        spurious duplicate failures. Per-doc ``create`` is the natural granularity
        and fails closed cleanly. UPSERT keeps the bulk ``upsert_batch`` fast-path
        (overwrite is idempotent, so partial success is benign).
        """
        from ..streaming import (
            StreamConfig,
            resolve_conflict_write,
            run_stream_write,
        )

        config = config or StreamConfig()
        # insert_batch_func=None routes INSERT through per-record create(); the
        # resolver only consults it for the INSERT policy (UPSERT uses
        # upsert_batch, SKIP is per-record), so None is safe for every policy.
        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=None,
            single_create_func=self.create,
            upsert_func=self.upsert,
            upsert_batch_func=self.upsert_batch,
        )
        return run_stream_write(
            records,
            batch_write_func=batch_write_func,
            single_write_func=single_write_func,
            skip_on_duplicate=skip_on_duplicate,
            config=config,
        )

    def _pre_filter(self, filter: Query | None) -> dict[str, Any] | None:
        """The clause a vector or hybrid search filters by first, if any.

        Asking for the mapping can fail as the search itself can, and is
        reported the same way.
        """
        try:
            types = self._field_types(filter)
        except Exception as e:
            self._handle_elasticsearch_error(e, "vector pre-filter")
        return self._build_filter_query(filter, types)

    def _vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str,
        k: int,
        metric: DistanceMetric,
        filter: Query | None,
    ) -> list[VectorSearchResult]:
        """Raw k-NN through Elasticsearch's ``knn`` clause.

        The threshold and the source assembly are the mixin's; this is the
        part that knows how to ask Elasticsearch.

        ``metric`` reaches the result metadata and nothing else ---
        :func:`build_knn_query` does not take one, so the similarity
        Elasticsearch computes is whatever the field's mapping declares.

        **The ``metric`` argument does not choose the ranking here.**
        Elasticsearch's kNN ranks by the ``similarity`` in the field's
        *mapping*, fixed when the index was created, and
        :func:`~dataknobs_data.vector.elasticsearch_utils.build_knn_query`
        carries no metric at all. So what arrives resolved is recorded in each
        hit's metadata and changes nothing about the order. On a field whose
        mapping was built under a different metric, that record is the metric
        the caller *asked for*, not the one that ran.

        Honouring an arbitrary per-query metric would mean either a
        ``script_score`` query, which gives up the approximate-nearest-
        neighbour index, or reading the mapping back to convert --- a round
        trip, a cache, and an answer for a field that has no mapping yet.
        That is its own design pass; the falsehood worth removing now is the
        silent one, which is this paragraph's absence.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Maximum number of results to return
            metric: Distance metric, already resolved by the mixin and
                recorded on each hit --- see above, it does not rank
            filter: Optional query filter to apply before vector search

        Returns:
            List of search results ordered by similarity
        """
        self._check_connection()

        # Import vector utilities
        from ..vector.elasticsearch_utils import (
            build_knn_query,
        )

        # Build filter query if provided
        filter_query = self._pre_filter(filter)

        # Build KNN query
        query = build_knn_query(
            query_vector=query_vector,
            # `build_knn_query` spells it `field_name`; this method's own
            # parameter is `vector_field`, as the whole backend family's is.
            field_name=vector_field,
            k=k,
            filter_query=filter_query,
        )

        # Execute search
        try:
            response = self._require_client().search(
                index=self.index_name,
                **query,
                size=k,
                # Always, now that ``include_source`` no longer reaches
                # here. It used to be forwarded straight into this
                # parameter, so ``include_source=False`` told Elasticsearch
                # to omit ``_source`` --- and ``VectorSearchResult.record``
                # is required, so the loop below had nothing to build a hit
                # from and returned an empty list. The record is not that
                # knob's subject; the assembly is.
                #
                # ``source``, which is what ``Elasticsearch.search`` declares.
                # The async twin spelled it ``_source``; both reach the
                # transport, because elasticsearch-py rewrites body-field
                # aliases before dispatch, so nothing at runtime could tell
                # them apart. The type checker can: ``_source`` is a
                # ``call-arg`` error against the published signature, and the
                # twin's copy escaped only because its client is not typed.
                source=True,
            )
        except Exception as e:
            self._handle_elasticsearch_error(e, "vector search")

        # Process results
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            record = self._doc_to_record(hit)
            if not record.has_storage_id():
                record.storage_id = hit["_id"]

            results.append(
                VectorSearchResult(
                    record=record,
                    score=hit.get("_score", 0.0),
                    vector_field=vector_field,
                    metadata={
                        "index": self.index_name,
                        "metric": metric.value,
                        "doc_id": hit["_id"],
                    },
                )
            )

        return results

    def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric | str | None = None,
        index_type: str = "auto",
    ) -> bool:
        """Create or update index mapping for vector field.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions
            metric: Distance metric for the index. ``None`` means the metric
                this database was configured with, falling back to cosine.
                Unlike on :meth:`_vector_search`, this one decides: the
                mapping it writes is what every later search over the field
                ranks by.
            index_type: Type of index (ignored for ES, always uses HNSW)

        Returns:
            True if index was created/updated successfully
        """
        self._check_connection()

        if not dimensions:
            if vector_field not in self.vector_fields:
                raise ValueError(f"Unknown dimensions for field '{vector_field}'")
            dimensions = self.vector_fields[vector_field]

        # Import vector utilities
        from ..vector.elasticsearch_utils import (
            get_similarity_for_metric,
            get_vector_mapping,
        )

        # Get similarity function for metric
        # Through the one resolver, so that a caller spelling the metric
        # as a string reaches the same mapping `vector_search` does.
        similarity = get_similarity_for_metric(resolve_metric(self, metric))

        # Build mapping for the vector field
        mapping = get_vector_mapping(dimensions, similarity)

        # Update index mapping
        try:
            self._require_client().indices.put_mapping(
                index=self.index_name, properties={f"data.{vector_field}": mapping}
            )

            # Track the vector field
            self.vector_fields[vector_field] = dimensions
            self._vector_enabled = True

            logger.info(
                f"Created vector mapping for field '{vector_field}' with {dimensions} dimensions"
            )
            return True

        except Exception as e:
            self._handle_elasticsearch_error(e, "create vector index")

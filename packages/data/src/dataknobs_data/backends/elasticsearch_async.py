# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Native async Elasticsearch backend implementation with connection pooling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import AsyncDatabase, version_conflict_error
from ..exceptions import DuplicateRecordError
from ..pooling import ConnectionPoolManager
from ..pooling.elasticsearch import (
    ElasticsearchPoolConfig,
    close_elasticsearch_client,
    create_async_elasticsearch_client,
    validate_elasticsearch_client,
)
from ..query import Query
from ..query_logic import ComplexQuery
from ..streaming import (
    StreamConfig,
    StreamResult,
    async_run_stream_write,
    resolve_conflict_write,
)
from ..vector.bulk_embed_mixin import AsyncBulkEmbedMixin
from ..vector.mixins import AsyncVectorOperationsMixin
from ..vector.mixins import resolve_metric
from ..vector.types import DistanceMetric, VectorSearchResult
from .config import AsyncElasticsearchDatabaseConfig
from .elasticsearch_errors import (
    bulk_outcomes,
    is_existing_index,
    is_missing_document,
    require_complete,
)
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
    arun_search_plan,
    FieldTypes,
    build_query,
    build_sort,
    client_search_kwargs,
    plan_search,
    query_fields,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from typing import ClassVar

    import numpy as np
    from elasticsearch import AsyncElasticsearch

    from ..records import Record

logger = logging.getLogger(__name__)

# Global pool manager for Elasticsearch clients
_client_manager: ConnectionPoolManager[Any] = ConnectionPoolManager()


class AsyncElasticsearchDatabase(
    StructuredConfigConsumer[AsyncElasticsearchDatabaseConfig],
    AsyncDatabase,
    # Before `AsyncVectorOperationsMixin`, whose `bulk_embed_and_store` is
    # abstract: the MRO must reach the implementation first. This is the same
    # ordering the other five async backends use.
    AsyncBulkEmbedMixin,
    AsyncVectorOperationsMixin,
    ElasticsearchBaseConfig,
    ElasticsearchIndexManager,
    ElasticsearchVectorSupport,
    ElasticsearchErrorHandler,
    ElasticsearchRecordSerializer,
    ElasticsearchQueryBuilder,
):
    """Native async Elasticsearch database backend with connection pooling.

    Constructed through :class:`AsyncElasticsearchDatabaseConfig` — every
    documented config key is a typed field on that dataclass, so
    ``self.config`` is the typed config (not a dict). The
    ``ElasticsearchPoolConfig`` is derived from it in :meth:`_setup`.
    """

    CONFIG_CLS: ClassVar[type[AsyncElasticsearchDatabaseConfig]] = AsyncElasticsearchDatabaseConfig

    def _setup(self) -> None:
        """Derive the pool config and connection state from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op — connection setup is deferred to
        :meth:`connect`). The pool config is built through
        ``ElasticsearchPoolConfig.from_dict`` so the ``hosts`` ⇄
        ``host``/``port`` derivation stays in one place; only the keys the
        caller actually set are forwarded so the pool config applies its
        own defaults.
        """
        cfg = self.config

        # Vector support (async ES uses AsyncVectorOperationsMixin, not the
        # VectorConfigMixin; it discovers vector fields lazily at write time).
        self.vector_fields: dict[str, int] = {}
        self.vector_enabled = False

        pool_input: dict[str, Any] = {"index": cfg.index}
        if cfg.hosts is not None:
            pool_input["hosts"] = cfg.hosts
        if cfg.host is not None:
            pool_input["host"] = cfg.host
        if cfg.port is not None:
            pool_input["port"] = cfg.port
        if cfg.api_key is not None:
            pool_input["api_key"] = cfg.api_key
        if cfg.basic_auth is not None:
            pool_input["basic_auth"] = cfg.basic_auth
        pool_input["verify_certs"] = cfg.verify_certs
        if cfg.ca_certs is not None:
            pool_input["ca_certs"] = cfg.ca_certs
        if cfg.client_cert is not None:
            pool_input["client_cert"] = cfg.client_cert
        if cfg.client_key is not None:
            pool_input["client_key"] = cfg.client_key
        pool_input["ssl_show_warn"] = cfg.ssl_show_warn

        self._pool_config = ElasticsearchPoolConfig.from_dict(pool_input)
        self.index_name = self._pool_config.index
        self.refresh = cfg.refresh
        self.max_result_window = cfg.max_result_window
        self.search_page_size = cfg.search_page_size
        self._client: AsyncElasticsearch | None = None
        self._connected = False
        self._known_types = FieldTypes()

    async def connect(self) -> None:
        """Connect to the Elasticsearch database."""
        if self._connected:
            return

        # Get or create client for current event loop
        from ..pooling import BasePoolConfig

        self._client = await _client_manager.get_pool(
            self._pool_config,
            cast("Callable[[BasePoolConfig], Awaitable[Any]]", create_async_elasticsearch_client),
            validate_elasticsearch_client,
            close_elasticsearch_client,
        )

        # get_pool incremented this holder's claim on the shared client.
        # If index setup fails we never set _connected, so close() (which
        # guards on _connected) would never release — leaking the holder
        # slot for the life of the process. Balance the increment here.
        try:
            await self._ensure_index()
        except Exception:
            await _client_manager.release_pool(self._pool_config)
            self._client = None
            raise
        self._connected = True

    async def close(self) -> None:
        """Release this holder's claim on the shared Elasticsearch client.

        The client is shared across instances on a loop (keyed on
        hosts + index) and owned by the pool manager. ``close()`` releases
        this holder; the registered close-func runs when the last holder
        releases. (Prior behavior dropped the local reference only, so the
        client was never closed until ``close_all()``/``atexit`` — a
        resource leak under instance churn.)
        """
        if self._connected:
            try:
                await _client_manager.release_pool(self._pool_config)
            except Exception as e:
                logger.warning("Error releasing Elasticsearch client: %s", e)
            self._client = None
            self._connected = False
            self._known_types = FieldTypes()

    def _initialize(self) -> None:
        """Initialize is handled in connect."""
        pass

    async def _index_exists(self) -> bool:
        """Whether this database's index is already there."""
        return bool(await self._require_client().indices.exists(index=self.index_name))

    async def _create_index(self, mappings: dict[str, Any], settings: dict[str, Any]) -> bool:
        """Create the index; ``False`` when another process created it first."""
        from elasticsearch import BadRequestError

        try:
            await self._require_client().indices.create(
                index=self.index_name, mappings=mappings, settings=settings
            )
        except BadRequestError as e:
            if is_existing_index(e):
                return False
            raise
        return True

    async def _ensure_index(self) -> None:
        """Ensure the index exists with proper mappings."""
        if await self._index_exists():
            return

        mappings, settings = self.index_definition(
            self.config.mappings, self.config.settings, self.vector_fields
        )
        if await self._create_index(mappings, settings) and self.vector_fields:
            self.vector_enabled = True
            logger.info(f"Created index '{self.index_name}' with vector support")

    def _check_connection(self) -> None:
        """Check if database is connected."""
        self._require_client()

    def _require_client(self) -> AsyncElasticsearch:
        """The connected client, or the same error :meth:`_check_connection` raises."""
        if self._client is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._client

    def _record_to_doc(self, record: Record, id: str | None = None) -> dict[str, Any]:
        """Convert a Record to an Elasticsearch document.

        ``id`` stamps the resolved storage key into the document's top-level
        ``id`` keyword field so it mirrors ``_id`` and stays filterable via
        ``Filter("id", ...)`` — the same invariant the sync backend guarantees
        in its own ``_record_to_doc``. Every write path resolves the id it uses
        for ``_id`` and passes it here, so a record created without an explicit
        id (minted uuid) is still findable by that id. When ``id`` is omitted,
        an absent doc id is minted here as a safety net.
        """
        # Update vector tracking if needed
        if self._has_vector_fields(record):
            self._update_vector_tracking(record)

            # Add vector field metadata to record metadata
            if "vector_fields" not in record.metadata:
                record.metadata["vector_fields"] = {}

            for field_name in self.vector_fields:
                if field_name in record.fields:
                    field = record.fields[field_name]
                    if hasattr(field, "source_field"):
                        record.metadata["vector_fields"][field_name] = vector_tracking_metadata(
                            field, self.vector_fields[field_name]
                        )

        doc = self._record_to_document(record)
        if id:
            doc["id"] = id
        elif not doc.get("id"):
            doc["id"] = self._generate_id()
        return doc

    def _doc_to_record(self, doc: dict[str, Any]) -> Record:
        """Convert an Elasticsearch document to a Record."""
        doc_id = doc.get("_id")
        record = self._document_to_record(doc, doc_id)

        # Add score if present
        if "_score" in doc:
            record.metadata["_score"] = doc["_score"]

        return record

    async def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()

        # Mint the id client-side when the record has none, so create() always
        # supplies an explicit id and returns a known value — uniform with the
        # sync backend and every other backend. op_type="create" makes this an
        # atomic insert: a colliding id yields a 409 conflict instead of
        # silently overwriting the existing document.
        record_id = record.id if record.id else self._generate_id()
        # Stamp the resolved id into the doc so a minted-id record stays
        # findable by ``Filter("id", ...)``.
        doc = self._record_to_doc(record, record_id)

        from elasticsearch import ConflictError

        try:
            response = await self._require_client().index(
                index=self.index_name,
                id=record_id,
                document=doc,
                refresh=self.refresh,
                op_type="create",
            )
        except ConflictError as e:
            raise DuplicateRecordError(record_id) from e

        return str(response["_id"])

    async def _bulk(self, actions: list[dict[str, Any]], operation: str) -> list[bool]:
        """Run bulk actions and report each one; see :func:`bulk_outcomes`.

        Sent in requests of up to 500 actions. A request that fails as a whole
        raises, and the ones before it have been applied.
        """
        from elasticsearch import helpers

        client = self._require_client()
        items = [
            item
            async for _, item in helpers.async_streaming_bulk(
                client, actions, refresh=self.refresh, raise_on_error=False
            )
        ]
        return bulk_outcomes(items, operation)

    async def create_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Create multiple records using the bulk API, failing closed on a colliding id.

        Uses the bulk ``create`` op, honoring a caller-supplied ``record.id``
        and minting one only when absent. The bulk API is per-item
        (non-atomic), so the other records in the batch have been indexed when
        either error is raised.

        ``_tx`` is accepted for interface parity with the transactional backends
        and ignored: Elasticsearch has no transaction to join.

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
        await self._bulk_index("create", ids, records)
        return ids

    async def upsert_batch(self, records: list[Record], *, _tx: Any = None) -> list[str]:
        """Insert-or-overwrite multiple records using the bulk API.

        Uses the bulk ``index`` op, honoring a caller-supplied ``record.id``
        and minting one only when absent; a colliding id is overwritten. The
        bulk API is per-item (non-atomic), so the other records in the batch
        have been indexed when the error is raised.

        ``_tx`` is accepted for interface parity with the transactional backends
        and ignored: Elasticsearch has no transaction to join.

        Returns:
            The written record ids, in input order

        Raises:
            DatabaseError: a record could not be indexed.
        """
        ids = [record.id or self._generate_id() for record in records]
        await self._bulk_index("index", ids, records)
        return ids

    async def _bulk_index(self, op_type: str, ids: list[str], records: list[Record]) -> None:
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
        await self._bulk(actions, "create_batch" if op_type == "create" else "upsert_batch")

    async def _get_document(self, id: str, *, source: bool = True) -> dict[str, Any] | None:
        """GET one document, or ``None`` when it does not exist.

        Every other failure raises, a missing index included.
        """
        client = self._require_client()
        if not id:
            # Not an id Elasticsearch can hold, so nothing is stored under it.
            return None

        from elasticsearch import NotFoundError

        try:
            response = await client.get(index=self.index_name, id=id, source=source)
        except NotFoundError as e:
            if is_missing_document(e):
                return None
            raise
        return dict(response)

    async def read(self, id: str) -> Record | None:
        """Read a record by ID; ``None`` when it does not exist."""
        document = await self._get_document(id)
        if document is None:
            return None
        return self._doc_to_record(document)

    async def get_version(self, id: str) -> str | None:
        """Return the document's ``_seq_no``/``_primary_term`` version token.

        Elasticsearch's native optimistic-concurrency pair
        (``_seq_no``, ``_primary_term``) advances on every write, so it is a
        native version — ABA-safe, unlike the base content-hash default this
        overrides. The two values are combined into one opaque token.
        """
        document = await self._get_document(id, source=False)
        if document is None:
            return None
        seq_no = document.get("_seq_no")
        primary_term = document.get("_primary_term")
        if seq_no is None or primary_term is None:
            return None
        return es_version_token(seq_no, primary_term)

    async def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
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
            await client.update(
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
            current = await self.get_version(id)
            if current is None:
                return False
            raise version_conflict_error(id, expected_version, current) from e
        return True

    async def delete(self, id: str, *, expected_version: str | None = None) -> bool:
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
            await client.delete(
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
            current = await self.get_version(id)
            if current is None:
                return False
            raise version_conflict_error(id, expected_version, current) from e
        return True

    async def exists(self, id: str) -> bool:
        """Check if a record exists.

        A GET without the source rather than a HEAD, whose 404 has no body to
        tell a missing record from a missing index.
        """
        return await self._get_document(id, source=False) is not None

    async def upsert(
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
        self._check_connection()

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
            if await self.update(id, record, expected_version=expected_version):
                return id
            raise version_conflict_error(id, expected_version, None)

        doc = self._record_to_doc(record, id)
        await self._require_client().index(
            index=self.index_name, id=id, document=doc, refresh=self.refresh
        )
        return id

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
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
        return await self._bulk(actions, "update_batch")

    async def delete_batch(self, ids: list[str], *, _tx: Any = None) -> list[bool]:
        """Delete multiple records using the bulk API.

        ``_tx`` is accepted for interface parity and ignored.

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
        return await self._bulk(actions, "delete_batch")

    async def _field_types(self, query: Query | ComplexQuery | None) -> FieldTypes:
        """How the index maps the fields ``query`` names, asking for those not known yet."""
        unknown = self._known_types.unknown(query_fields(query))
        if unknown:
            response = await self._require_client().field_caps(
                index=self.index_name, fields=FieldTypes.request_fields(unknown)
            )
            # Learned for good, so never from part of an alias.
            require_complete(response, "field lookup")
            self._known_types.learn(unknown, response["fields"])
        return self._known_types

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # Translated by functions the twins share, so they cannot drift apart.
        types = await self._field_types(query)
        es_query = build_query(query, types)
        sort = build_sort(query.sort_specs, types)

        # One request inside the result window, ``search_after`` pages past
        # it or for an unbounded read; the plan is shared with the sync twin.
        plan = plan_search(
            es_query,
            sort,
            query.offset_value,
            query.limit_value,
            window=self.max_result_window,
            page_size=self.search_page_size,
        )
        hits = await arun_search_plan(plan, self._search_hits, self._open_pit, self._close_pit)

        # Convert to records
        records = []
        for hit in hits:
            record = self._doc_to_record(hit)

            # Apply field projection if specified
            if query.fields:
                record = record.project(query.fields)

            records.append(record)

        return records

    async def _search_hits(self, body: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        """Send one planned search request; return its hits and point-in-time id."""
        client = self._require_client()
        response = await client.search(**client_search_kwargs(body, self.index_name))
        require_complete(response, "search")
        return list(response["hits"]["hits"]), response.get("pit_id")

    async def _open_pit(self) -> str:
        """Open a point in time on this index for one paged read."""
        opened = await self._require_client().open_point_in_time(
            index=self.index_name, keep_alive=PIT_KEEP_ALIVE
        )
        return str(opened["id"])

    async def _close_pit(self, pit_id: str) -> None:
        """Close a paged read's point in time."""
        await self._require_client().close_point_in_time(id=pit_id)

    async def count(self, query: Query | None = None) -> int:
        """Count records matching a query using efficient Elasticsearch count.

        Args:
            query: Optional search query (counts all if None)

        Returns:
            Number of matching records
        """
        if not query or not query.filters:
            return await self._count_all()

        # Same shared translator as search(), so count() honors the full
        # operator set and cannot drift from the sync backend.
        es_query = build_query(query, await self._field_types(query))

        client = self._require_client()
        response = await client.count(index=self.index_name, query=es_query)
        return int(require_complete(response, "count")["count"])

    async def _count_all(self) -> int:
        """Count all records in the database."""
        response = await self._require_client().count(index=self.index_name)
        return int(require_complete(response, "count")["count"])

    async def clear(self) -> int:
        """Clear all records from the database."""
        client = self._require_client()
        count = await self._count_all()

        response = await client.delete_by_query(
            index=self.index_name, query={"match_all": {}}, refresh=self.refresh
        )
        return int(response.get("deleted", count))

    async def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from Elasticsearch using scroll API."""
        self._check_connection()
        config = config or StreamConfig()

        # Same shared translator as search()/count(), so streamed reads honor
        # the full operator set instead of only equality.
        query = query or Query()
        es_query = build_query(query, await self._field_types(query))

        # Initial search with scroll
        response = await self._require_client().search(
            index=self.index_name, query=es_query, scroll="2m", size=config.batch_size
        )

        scroll_id = response["_scroll_id"]

        try:
            hits = require_complete(response, "stream read")["hits"]["hits"]
            while hits:
                for hit in hits:
                    record = self._doc_to_record(hit)
                    if query.fields:
                        record = record.project(query.fields)
                    yield record

                # Get next batch
                response = await self._require_client().scroll(scroll_id=scroll_id, scroll="2m")
                hits = require_complete(response, "stream read")["hits"]["hits"]
        finally:
            # Clear scroll
            await self._require_client().clear_scroll(scroll_id=scroll_id)

    async def stream_write(
        self, records: AsyncIterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into Elasticsearch using bulk API.

        INSERT routes through per-record ``create()`` (``insert_batch_func=None``)
        rather than a bulk fast-path: the Elasticsearch bulk API is non-atomic,
        so a partial-success batch followed by the per-record fallback would
        re-write the already-indexed rows and count them as spurious duplicate
        failures. Per-doc ``create`` is the natural granularity and fails closed
        cleanly. UPSERT keeps the bulk ``upsert_batch`` fast-path (overwrite is
        idempotent, so partial success is benign).
        """
        self._check_connection()
        config = config or StreamConfig()

        # insert_batch_func=None routes INSERT through per-record create(); the
        # resolver only consults it for the INSERT policy.
        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=None,
            single_create_func=self.create,
            upsert_func=self.upsert,
            upsert_batch_func=self.upsert_batch,
        )
        return await async_run_stream_write(
            records,
            batch_write_func=batch_write_func,
            single_write_func=single_write_func,
            skip_on_duplicate=skip_on_duplicate,
            config=config,
        )

    async def _pre_filter(self, filter: Query | None) -> dict[str, Any] | None:
        """The clause a vector or hybrid search filters by first, if any.

        Asking for the mapping can fail as the search itself can, and is
        reported the same way.
        """
        try:
            types = await self._field_types(filter)
        except Exception as e:
            self._handle_elasticsearch_error(e, "vector pre-filter")
        return self._build_filter_query(filter, types)

    async def _vector_search(
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
        filter_query = await self._pre_filter(filter)

        # Build KNN query
        query = build_knn_query(
            query_vector=query_vector,
            field_name=vector_field,
            k=k,
            filter_query=filter_query,
        )

        # Execute search
        try:
            response = await self._require_client().search(
                index=self.index_name,
                **query,  # Unpack the query dict directly
                size=k,
                # Always, now that ``include_source`` no longer reaches
                # here. It used to be forwarded straight into this
                # parameter, so ``include_source=False`` told Elasticsearch
                # to omit ``_source`` --- and ``VectorSearchResult.record``
                # is required, so the loop below had nothing to build a hit
                # from and returned an empty list. The record is not that
                # knob's subject; the assembly is.
                # ``source``, as the sync twin spells it and as
                # ``AsyncElasticsearch.search`` declares it. ``_source`` works
                # --- elasticsearch-py rewrites the alias --- but it is not a
                # parameter of the method, so it type-checks only where the
                # client is untyped, which is here and not on the twin.
                source=True,
            )
            require_complete(response, "vector search")
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

    async def create_vector_index(
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
            metric: Distance metric for the index
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
            await self._require_client().indices.put_mapping(
                index=self.index_name, properties={f"data.{vector_field}": mapping}
            )

            # Track the vector field
            self.vector_fields[vector_field] = dimensions
            self.vector_enabled = True

            logger.info(
                f"Created vector mapping for field '{vector_field}' with {dimensions} dimensions"
            )
            return True

        except Exception as e:
            self._handle_elasticsearch_error(e, "create vector index")

    async def _supports_native_hybrid(self) -> bool:
        """Check if this Elasticsearch backend supports native hybrid search.

        Elasticsearch 8.x supports native RRF hybrid search.

        Returns:
            True since Elasticsearch supports native hybrid search
        """
        return True

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray | list[float],
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        k: int = 10,
        config: Any = None,  # HybridSearchConfig
        filter: Query | None = None,
        metric: DistanceMetric | str | None = None,
    ) -> list[Any]:  # list[HybridSearchResult]
        """Perform native Elasticsearch hybrid search using RRF.

        Uses Elasticsearch's native RRF (Reciprocal Rank Fusion) for combining
        BM25 text search with KNN vector search. This is more efficient than
        client-side fusion as it's executed in a single request.

        Args:
            query_text: Text query for BM25 matching
            query_vector: Vector for KNN similarity search
            text_fields: Fields to search for text matching
            vector_field: Name of the vector field to search
            k: Number of results to return
            config: Hybrid search configuration (weights, fusion strategy)
            filter: Optional additional filters to apply
            metric: Distance metric for vector search

        Returns:
            List of HybridSearchResult ordered by RRF score (descending)
        """
        from ..vector.hybrid import (
            FusionStrategy,
            HybridSearchConfig,
            HybridSearchResult,
        )

        self._check_connection()

        config = config or HybridSearchConfig()

        # If not using native strategy, fall back to parent implementation
        if config.fusion_strategy != FusionStrategy.NATIVE:
            # Import parent class to call its implementation
            from ..vector.mixins import AsyncVectorOperationsMixin

            return await AsyncVectorOperationsMixin.hybrid_search(
                self,
                query_text=query_text,
                query_vector=query_vector,
                text_fields=text_fields,
                vector_field=vector_field,
                k=k,
                config=config,
                filter=filter,
                metric=metric,
            )

        # Use config.text_fields if provided, otherwise use parameter
        search_text_fields = config.text_fields or text_fields or ["content", "title", "text"]

        # Build filter query if provided
        filter_query = await self._pre_filter(filter)

        # Build text search query with multi_match
        text_query: dict[str, Any] = {
            "multi_match": {
                "query": query_text,
                "fields": [f"data.{f}" for f in search_text_fields],
                "type": "best_fields",
                "operator": "or",
            }
        }

        # Build RRF query combining both searches
        # Note: RRF requires Elasticsearch 8.8+ with appropriate license
        # For older versions, we need to use sub_searches
        try:
            # Try native RRF (ES 8.8+)
            body: dict[str, Any] = {
                "retriever": {
                    "rrf": {
                        "retrievers": [
                            {"standard": {"query": text_query}},
                            {
                                "knn": {
                                    "field": f"data.{vector_field}",
                                    "query_vector": query_vector.tolist()
                                    if hasattr(query_vector, "tolist")
                                    else list(query_vector),
                                    "k": k,
                                    "num_candidates": k * 3,
                                }
                            },
                        ],
                        "rank_constant": config.rrf_k,
                        "rank_window_size": k * 3,
                    }
                },
                "size": k,
            }

            if filter_query:
                body["post_filter"] = filter_query

            response = await self._require_client().search(
                index=self.index_name,
                body=body,
            )
            require_complete(response, "hybrid search")
        except Exception as e:
            # Fall back to client-side fusion if native RRF not available
            logger.warning(f"Native RRF not available ({e}), falling back to client-side fusion")
            from ..vector.mixins import AsyncVectorOperationsMixin

            return await AsyncVectorOperationsMixin.hybrid_search(
                self,
                query_text=query_text,
                query_vector=query_vector,
                text_fields=text_fields,
                vector_field=vector_field,
                k=k,
                config=HybridSearchConfig(
                    text_weight=config.text_weight,
                    vector_weight=config.vector_weight,
                    fusion_strategy=FusionStrategy.RRF,
                    rrf_k=config.rrf_k,
                    text_fields=config.text_fields,
                ),
                filter=filter,
                metric=metric,
            )

        # Process results
        results: list[HybridSearchResult] = []
        hits = response.get("hits", {}).get("hits", [])

        for i, hit in enumerate(hits):
            record = self._doc_to_record(hit)
            if record:
                if not record.has_storage_id():
                    record.storage_id = hit["_id"]

                # RRF doesn't provide individual scores, just the fused score
                combined_score = hit.get("_score", 1.0 / (config.rrf_k + i + 1))

                results.append(
                    HybridSearchResult(
                        record=record,
                        combined_score=combined_score,
                        text_score=None,  # Not available with native RRF
                        vector_score=None,  # Not available with native RRF
                        text_rank=None,
                        vector_rank=None,
                        metadata={
                            "fusion_strategy": "native_rrf",
                            "index": self.index_name,
                            "doc_id": hit["_id"],
                        },
                    )
                )

        return results

    async def _text_search_for_hybrid(
        self,
        query_text: str,
        text_fields: list[str] | None,
        k: int,
        filter: Query | None = None,
    ) -> list[tuple[Record, float]]:
        """Perform BM25 text search for hybrid search fusion.

        Uses Elasticsearch's native BM25 scoring for text relevance.

        Args:
            query_text: Text to search for
            text_fields: Fields to search in
            k: Maximum results to return
            filter: Additional filters

        Returns:
            List of (record, score) tuples ordered by BM25 relevance
        """
        self._check_connection()

        search_fields = text_fields or ["content", "title", "text"]

        # Build multi_match query
        query: dict[str, Any] = {
            "multi_match": {
                "query": query_text,
                "fields": [f"data.{f}" for f in search_fields],
                "type": "best_fields",
                "operator": "or",
            }
        }

        # Build filter if provided
        filter_query = await self._pre_filter(filter)

        body: dict[str, Any] = {
            "query": query
            if not filter_query
            else {
                "bool": {
                    "must": query,
                    "filter": filter_query,
                }
            },
            "size": k,
        }

        try:
            response = await self._require_client().search(
                index=self.index_name,
                body=body,
            )
            require_complete(response, "text search for hybrid")
        except Exception as e:
            self._handle_elasticsearch_error(e, "text search for hybrid")

        # Process results
        results: list[tuple[Record, float]] = []
        hits = response.get("hits", {}).get("hits", [])
        max_score = response.get("hits", {}).get("max_score", 1.0) or 1.0

        for hit in hits:
            record = self._doc_to_record(hit)
            if record:
                if not record.has_storage_id():
                    record.storage_id = hit["_id"]

                # Normalize BM25 score to 0-1 range
                score = hit.get("_score", 0.0) / max_score if max_score > 0 else 0.0
                results.append((record, score))

        return results

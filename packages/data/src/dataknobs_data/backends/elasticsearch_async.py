"""Native async Elasticsearch backend implementation with connection pooling."""

from __future__ import annotations

import logging
import uuid
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
from ..query import Query, SortOrder, is_storage_key_field
from ..query_logic import ComplexQuery
from ..streaming import (
    StreamConfig,
    StreamResult,
    async_run_stream_write,
    resolve_conflict_write,
)
from ..vector.mixins import VectorOperationsMixin
from ..vector.types import DistanceMetric, VectorSearchResult
from .config import AsyncElasticsearchDatabaseConfig
from .elasticsearch_mixins import (
    ElasticsearchBaseConfig,
    ElasticsearchErrorHandler,
    ElasticsearchIndexManager,
    ElasticsearchQueryBuilder,
    ElasticsearchRecordSerializer,
    ElasticsearchVectorSupport,
    es_version_token,
    parse_es_version_token,
)
from .elasticsearch_query import build_bool_query, build_complex_es_query

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from typing import ClassVar

    import numpy as np

    from ..records import Record

logger = logging.getLogger(__name__)

# Global pool manager for Elasticsearch clients
_client_manager = ConnectionPoolManager()


class AsyncElasticsearchDatabase(
    StructuredConfigConsumer[AsyncElasticsearchDatabaseConfig],
    AsyncDatabase,
    VectorOperationsMixin,
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

    CONFIG_CLS: ClassVar[type[AsyncElasticsearchDatabaseConfig]] = (
        AsyncElasticsearchDatabaseConfig
    )

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

        # Vector support (async ES uses VectorOperationsMixin, not the
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
        self._client = None
        self._connected = False

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
            close_elasticsearch_client
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

    def _initialize(self) -> None:
        """Initialize is handled in connect."""
        pass

    async def _ensure_index(self) -> None:
        """Ensure the index exists with proper mappings."""
        if not self._client:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Check if index exists
        if not await self._client.indices.exists(index=self.index_name):  # type: ignore[unreachable]
            # Get mappings with vector field support
            mappings = self.get_index_mappings(self.vector_fields)

            # Get settings optimized for KNN if we have vector fields
            settings = self.get_knn_index_settings() if self.vector_fields else {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }

            await self._client.indices.create(
                index=self.index_name,
                mappings=mappings,
                settings=settings
            )

            if self.vector_fields:
                self.vector_enabled = True
                logger.info(f"Created index '{self.index_name}' with vector support")

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self._client:
            raise RuntimeError("Database not connected. Call connect() first.")

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
                        record.metadata["vector_fields"][field_name] = {
                            "type": "vector",
                            "dimensions": self.vector_fields[field_name],
                            "source_field": field.source_field,
                            "model": getattr(field, "model_name", None),
                            "model_version": getattr(field, "model_version", None),
                        }

        doc = self._record_to_document(record)
        if id:
            doc["id"] = id
        elif not doc.get("id"):
            doc["id"] = str(uuid.uuid4())
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
        record_id = record.id if record.id else str(uuid.uuid4())
        # Stamp the resolved id into the doc so a minted-id record stays
        # findable by ``Filter("id", ...)``.
        doc = self._record_to_doc(record, record_id)

        from elasticsearch import ConflictError

        try:
            response = await self._client.index(
                index=self.index_name,
                id=record_id,
                document=doc,
                refresh=self.refresh,
                op_type="create",
            )
        except ConflictError as e:
            raise DuplicateRecordError(record_id) from e

        return response["_id"]

    @staticmethod
    def _extract_bulk_index_ids(
        response: Any, ids: list[str] | None = None
    ) -> list[str]:
        """Return the ids of successfully-indexed items from a bulk response.

        Reconciles ``response['items']`` per operation so a partial bulk
        failure is not reported as success: an item carrying an ``error`` (or
        a ``status`` >= 400) is dropped. The async sibling of the sync
        backend's ``_execute_bulk_index`` reconciliation — the two use
        different ES client APIs (raw ``client.bulk`` here vs the ``helpers``
        module there) so the response shapes differ, but the per-item
        drop-the-failures contract is identical.

        Both ``create_batch`` and ``upsert_batch`` pass the client-minted,
        input-order ``ids`` list (explicit-``_id`` writes), so only the failed
        positions are dropped. The ``ids=None`` branch — reading each successful
        item's ``_id`` out of the response — is retained for a caller relying on
        server-assigned ids.
        """
        result: list[str] = []
        for pos, item in enumerate(response.get("items", [])):
            op = item.get("index") or item.get("create") or {}
            # An ``error`` key is ES's canonical per-item failure signal; the
            # status check is a secondary guard (a successful item always
            # carries status 200/201, so a missing status defaults to success).
            if "error" in op or op.get("status", 200) >= 400:
                # Failed operation — do not report its id as written.
                continue
            if ids is not None:
                if pos < len(ids):
                    result.append(ids[pos])
            else:
                _id = op.get("_id")
                if _id is not None:
                    result.append(_id)
        return result

    @staticmethod
    def _raise_on_bulk_conflict(response: Any) -> None:
        """Raise ``DuplicateRecordError`` on the first 409 in a bulk response.

        The bulk ``create`` op fails closed on a colliding id (409), matching a
        single-record ``create()``. The Elasticsearch bulk API is per-item
        (non-atomic), so non-colliding records may already be indexed when the
        conflict is raised — exactly like a ``create()`` loop.
        """
        for item in response.get("items", []):
            op = item.get("create") or item.get("index") or {}
            if op.get("status") == 409:
                raise DuplicateRecordError(op.get("_id"))

    async def create_batch(
        self, records: list[Record], *, _tx: Any = None
    ) -> list[str]:
        """Create multiple records in batch, failing closed on a colliding id.

        Uses the bulk ``create`` op keyed on ``record.id`` (honoring a
        caller-supplied id, minting a uuid only when absent). Like ``create()``,
        a colliding id raises ``DuplicateRecordError`` (bulk 409); other per-item
        errors are reconciled via :meth:`_extract_bulk_index_ids`, so a record
        that fails to index is not reported as created.

        ``_tx`` is accepted for interface parity with the transactional backends
        and ignored — Elasticsearch exposes no native transaction to join.

        Raises:
            DuplicateRecordError: a record collides with an existing id, or two
                records in the batch share an id.
        """
        self._check_connection()

        if not records:
            return []

        ids: list[str] = []
        operations: list[dict] = []
        seen: set[str] = set()
        for record in records:
            record_id = record.id or str(uuid.uuid4())
            if record_id in seen:
                raise DuplicateRecordError(record_id)
            seen.add(record_id)
            ids.append(record_id)
            doc = self._record_to_doc(record, record_id)
            operations.append(
                {"create": {"_index": self.index_name, "_id": record_id}}
            )
            operations.append(doc)

        response = await self._client.bulk(
            operations=operations,
            refresh=self.refresh
        )
        self._raise_on_bulk_conflict(response)
        return self._extract_bulk_index_ids(response, ids)

    async def upsert_batch(
        self, records: list[Record], *, _tx: Any = None
    ) -> list[str]:
        """Insert-or-overwrite multiple records in batch using the bulk API.

        Uses the bulk ``index`` op keyed on ``record.id`` (upsert-by-id),
        honoring a caller-supplied ``record.id`` (minting a uuid only when
        absent); a colliding id is overwritten (never raised). Returns the ids
        that were written, in input order. Like ``create_batch``, this supplies
        ``_id`` explicitly so the write is addressable and idempotent; the two
        differ only in op type (``index`` overwrites, ``create`` fails closed).
        Per-item errors are reconciled via
        :meth:`_extract_bulk_index_ids`, so an id whose write failed is dropped
        rather than reported as written. ``_tx`` is accepted for interface parity
        and ignored (see :meth:`create_batch`).
        """
        self._check_connection()

        if not records:
            return []

        ids: list[str] = []
        operations: list[dict] = []
        for record in records:
            record_id = record.id or str(uuid.uuid4())
            ids.append(record_id)
            doc = self._record_to_doc(record, record_id)
            operations.append(
                {"index": {"_index": self.index_name, "_id": record_id}}
            )
            operations.append(doc)

        response = await self._client.bulk(
            operations=operations, refresh=self.refresh
        )
        return self._extract_bulk_index_ids(response, ids)

    async def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()

        try:
            response = await self._client.get(
                index=self.index_name,
                id=id
            )
            return self._doc_to_record(response)
        except Exception as e:
            # Log the error for debugging
            logger.debug(f"Error reading document {id}: {e}")
            return None

    async def get_version(self, id: str) -> str | None:
        """Return the document's ``_seq_no``/``_primary_term`` version token.

        Elasticsearch's native optimistic-concurrency pair
        (``_seq_no``, ``_primary_term``) advances on every write, so it is a
        native version — ABA-safe, unlike the base content-hash default this
        overrides. The two values are combined into one opaque token.
        """
        self._check_connection()
        from elasticsearch import NotFoundError

        try:
            response = await self._client.get(index=self.index_name, id=id)
        except NotFoundError:
            return None
        seq_no = response.get("_seq_no")
        primary_term = response.get("_primary_term")
        if seq_no is None or primary_term is None:
            return None
        return es_version_token(seq_no, primary_term)

    async def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        """Update an existing record.

        When ``expected_version`` is provided the update carries ES's
        ``if_seq_no``/``if_primary_term`` guards so the compare-and-set is
        enforced server-side; a stale token raises ``ConcurrencyError``. When
        ``None`` the update is unconditional, byte-identical to prior behavior.
        """
        self._check_connection()
        # Target id stamped so a partial update heals/keeps the top-level id
        # field consistent with ``_id``.
        doc = self._record_to_doc(record, id)

        if expected_version is not None:
            from elasticsearch import ConflictError, NotFoundError

            seq_no, primary_term = parse_es_version_token(expected_version)
            try:
                await self._client.update(
                    index=self.index_name,
                    id=id,
                    doc=doc,
                    refresh=self.refresh,
                    if_seq_no=seq_no,
                    if_primary_term=primary_term,
                )
                return True
            except NotFoundError:
                # The document was absent all along. A conditional update never
                # inserts, so return False (uniform with the sync ES backend
                # and every other backend) rather than surfacing ES's raw 404.
                # A document deleted *after* a real seq_no conflict surfaces as
                # ConflictError below, not here.
                return False
            except ConflictError as e:
                # The token is stale (concurrent modification). If the doc has
                # since been deleted, treat it as gone (-> False); otherwise
                # raise the standard conflict.
                current = await self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current) from e

        try:
            await self._client.update(
                index=self.index_name,
                id=id,
                doc=doc,
                refresh=self.refresh
            )
            return True
        except Exception:
            return False

    async def delete(
        self, id: str, *, expected_version: str | None = None
    ) -> bool:
        """Delete a record by ID.

        When ``expected_version`` is provided the delete carries ES's
        ``if_seq_no``/``if_primary_term`` guards so the compare-and-set is
        enforced server-side; a stale token raises ``ConcurrencyError`` and a
        missing document returns ``False``. When ``None`` the delete is
        unconditional, byte-identical to prior behavior.
        """
        self._check_connection()

        if expected_version is not None:
            from elasticsearch import ConflictError, NotFoundError

            seq_no, primary_term = parse_es_version_token(expected_version)
            try:
                await self._client.delete(
                    index=self.index_name,
                    id=id,
                    refresh=self.refresh,
                    if_seq_no=seq_no,
                    if_primary_term=primary_term,
                )
                return True
            except NotFoundError:
                # A conditional delete never conflicts on an absent id.
                return False
            except ConflictError as e:
                current = await self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current) from e

        try:
            await self._client.delete(
                index=self.index_name,
                id=id,
                refresh=self.refresh
            )
            return True
        except Exception:
            return False

    async def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()

        return await self._client.exists(
            index=self.index_name,
            id=id
        )

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

        # Determine ID and record based on arguments
        if isinstance(id_or_record, str):
            id = id_or_record
            if record is None:
                raise ValueError("Record required when ID is provided")
        else:
            record = id_or_record
            id = record.id
            if id is None:
                import uuid  # type: ignore[unreachable]
                id = str(uuid.uuid4())
                record.storage_id = id

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

        await self._client.index(
            index=self.index_name,
            id=id,
            document=doc,
            refresh=self.refresh
        )

        return id

    async def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using the bulk API.
        
        Uses AsyncElasticsearch's bulk API for efficient batch updates.
        
        Args:
            updates: List of (id, record) tuples to update
            
        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        self._check_connection()

        # Build bulk operations for AsyncElasticsearch
        operations: list[dict[str, Any]] = []
        for record_id, record in updates:
            # Add update operation
            operations.append({
                "update": {
                    "_index": self.index_name,
                    "_id": record_id
                }
            })
            # Add document data
            doc = self._record_to_doc(record, record_id)
            operations.append({
                "doc": doc,
                "doc_as_upsert": False  # Don't create if doesn't exist
            })

        try:
            # Execute bulk update using AsyncElasticsearch
            response = await self._client.bulk(
                operations=operations,
                refresh=self.refresh
            )

            # Process the response to determine which updates succeeded
            results = []
            if response.get("items"):
                for item in response["items"]:
                    if "update" in item:
                        update_result = item["update"]
                        # Check if update was successful (status 200) or not found (404)
                        results.append(update_result.get("status") == 200)
                    else:
                        results.append(False)
            else:
                # If no items in response, mark all as failed
                results = [False] * len(updates)

            return results

        except Exception as e:
            # If bulk operation fails, mark all as failed
            import logging
            logging.error(f"Bulk update failed: {e}")
            return [False] * len(updates)

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # Translate through the shared filter->DSL functions so the async path
        # stays at parity with the sync backend and the vector pre-filter.
        if isinstance(query, ComplexQuery):
            es_query = (
                build_complex_es_query(query.condition)
                if query.condition
                else {"match_all": {}}
            )
        else:
            es_query = build_bool_query(query.filters)

        # Build sort
        sort = []
        if query.sort_specs:
            for sort_spec in query.sort_specs:
                direction = "desc" if sort_spec.order == SortOrder.DESC else "asc"
                # The 'id' field is the storage key: sort on the top-level ``id``
                # keyword (mirroring ``_id``), not a data field named ``id``,
                # matching the sync backend. ``_id`` itself is unsortable
                # (fielddata disabled by default), so the keyword mirror is used.
                sort_path = (
                    "id"
                    if is_storage_key_field(sort_spec.field)
                    else f"data.{sort_spec.field}"
                )
                sort.append({sort_path: {"order": direction}})

        # Build request body
        body = {"query": es_query}
        if sort:
            body["sort"] = sort

        # Add size and from for pagination.  ``is not None`` so the
        # caller-facing ``limit=0`` becomes ES ``size=0`` (count-only,
        # zero hits) rather than being silently coerced to the default.
        size = query.limit_value if query.limit_value is not None else 10000
        from_param = (
            query.offset_value if query.offset_value is not None else 0
        )

        # Execute search
        response = await self._client.search(
            index=self.index_name,
            query=es_query,
            sort=sort if sort else None,
            size=size,
            from_=from_param
        )

        # Convert to records
        records = []
        for hit in response["hits"]["hits"]:
            record = self._doc_to_record(hit)

            # Apply field projection if specified
            if query.fields:
                record = record.project(query.fields)

            records.append(record)

        return records

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
        es_query = build_bool_query(query.filters)

        self._check_connection()
        response = await self._client.count(index=self.index_name, query=es_query)
        return response["count"]

    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()

        response = await self._client.count(index=self.index_name)
        return response["count"]

    async def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()

        # Get count before deletion
        count = await self._count_all()

        # Delete by query - delete all documents
        response = await self._client.delete_by_query(
            index=self.index_name,
            query={"match_all": {}},
            refresh=self.refresh
        )

        return response.get("deleted", count)

    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from Elasticsearch using scroll API."""
        self._check_connection()
        config = config or StreamConfig()

        # Same shared translator as search()/count(), so streamed reads honor
        # the full operator set instead of only equality.
        es_query = (
            build_bool_query(query.filters)
            if query and query.filters
            else {"match_all": {}}
        )

        # Initial search with scroll
        response = await self._client.search(
            index=self.index_name,
            query=es_query,
            scroll="2m",
            size=config.batch_size
        )

        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        try:
            while hits:
                for hit in hits:
                    record = self._doc_to_record(hit)
                    if query and query.fields:
                        record = record.project(query.fields)
                    yield record

                # Get next batch
                response = await self._client.scroll(
                    scroll_id=scroll_id,
                    scroll="2m"
                )
                hits = response["hits"]["hits"]
        finally:
            # Clear scroll
            await self._client.clear_scroll(scroll_id=scroll_id)

    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
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

    async def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        vector_field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using Elasticsearch KNN.
        
        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Number of results to return
            metric: Distance metric to use
            filter: Optional query filter to apply before vector search
            include_source: Whether to include source document in results
            score_threshold: Optional minimum similarity score
            
        Returns:
            List of search results ordered by similarity
        """
        self._check_connection()

        # Import vector utilities
        from ..vector.elasticsearch_utils import (
            build_knn_query,
        )

        # Build filter query if provided
        filter_query = self._build_filter_query(filter) if filter else None

        # Build KNN query
        query = build_knn_query(
            query_vector=query_vector,
            field_name=vector_field,
            k=k,
            filter_query=filter_query,
        )

        # Execute search
        try:
            response = await self._client.search(
                index=self.index_name,
                **query,  # Unpack the query dict directly
                size=k,
                _source=include_source,
            )
        except Exception as e:
            self._handle_elasticsearch_error(e, "vector search")
            return []

        # Process results
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            score = hit.get("_score", 0.0)

            # Apply score threshold if specified
            if score_threshold is not None and score < score_threshold:
                continue

            # Convert document to record if source included
            record = None
            if include_source:
                record = self._doc_to_record(hit)

            # Set the storage ID on the record if we have one
            if record and not record.has_storage_id():
                record.storage_id = hit["_id"]

            # Skip if no record (shouldn't happen if include_source is True)
            if record is None:
                continue

            results.append(VectorSearchResult(
                record=record,
                score=score,
                vector_field=vector_field,
                metadata={
                    "index": self.index_name,
                    "metric": metric.value,
                    "doc_id": hit["_id"],
                },
            ))

        return results

    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Any | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.
        
        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            
        Returns:
            List of record IDs that were processed
        """
        # This is a stub implementation
        # Full implementation would require an actual embedding function
        logger.warning("bulk_embed_and_store is not fully implemented for Elasticsearch")
        return []

    async def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        index_type: str = "auto",
        **kwargs: Any,
    ) -> bool:
        """Create or update index mapping for vector field.
        
        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions
            metric: Distance metric for the index
            index_type: Type of index (ignored for ES, always uses HNSW)
            **kwargs: Additional index parameters
            
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
        similarity = get_similarity_for_metric(metric)

        # Build mapping for the vector field
        mapping = get_vector_mapping(dimensions, similarity)

        # Update index mapping
        try:
            await self._client.indices.put_mapping(
                index=self.index_name,
                properties={
                    f"data.{vector_field}": mapping
                }
            )

            # Track the vector field
            self.vector_fields[vector_field] = dimensions
            self.vector_enabled = True

            logger.info(f"Created vector mapping for field '{vector_field}' with {dimensions} dimensions")
            return True

        except Exception as e:
            self._handle_elasticsearch_error(e, "create vector index")
            return False

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
        metric: DistanceMetric = DistanceMetric.COSINE,
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
            from ..vector.mixins import VectorOperationsMixin
            return await VectorOperationsMixin.hybrid_search(
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
        filter_query = self._build_filter_query(filter) if filter else None

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
                            {
                                "standard": {
                                    "query": text_query
                                }
                            },
                            {
                                "knn": {
                                    "field": f"data.{vector_field}",
                                    "query_vector": query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector),
                                    "k": k,
                                    "num_candidates": k * 3,
                                }
                            }
                        ],
                        "rank_constant": config.rrf_k,
                        "rank_window_size": k * 3,
                    }
                },
                "size": k,
            }

            if filter_query:
                body["post_filter"] = filter_query

            response = await self._client.search(
                index=self.index_name,
                body=body,
            )
        except Exception as e:
            # Fall back to client-side fusion if native RRF not available
            logger.warning(f"Native RRF not available ({e}), falling back to client-side fusion")
            from ..vector.mixins import VectorOperationsMixin
            return await VectorOperationsMixin.hybrid_search(
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

                results.append(HybridSearchResult(
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
                ))

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
        filter_query = self._build_filter_query(filter) if filter else None

        body: dict[str, Any] = {
            "query": query if not filter_query else {
                "bool": {
                    "must": query,
                    "filter": filter_query,
                }
            },
            "size": k,
        }

        try:
            response = await self._client.search(
                index=self.index_name,
                body=body,
            )
        except Exception as e:
            self._handle_elasticsearch_error(e, "text search for hybrid")
            return []

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

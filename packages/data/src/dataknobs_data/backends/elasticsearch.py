"""Elasticsearch backend implementation for the data package."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from dataknobs_common.structured_config import StructuredConfigConsumer

from dataknobs_utils.elasticsearch_utils import (
    ElasticsearchConflictError,
    SimplifiedElasticsearchIndex,
)

from ..database import SyncDatabase, version_conflict_error
from ..exceptions import DatabaseError, DuplicateRecordError
from ..query import Query, SortOrder
from ..query_logic import ComplexQuery
from ..streaming import StreamConfig, StreamingMixin, StreamResult
from ..vector.types import DistanceMetric, VectorSearchResult
from .config import SyncElasticsearchDatabaseConfig
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
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import ClassVar

    import numpy as np

    from ..records import Record

logger = logging.getLogger(__name__)


class SyncElasticsearchDatabase(
    StructuredConfigConsumer[SyncElasticsearchDatabaseConfig],
    SyncDatabase,
    StreamingMixin,
    VectorConfigMixin,
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
    """

    CONFIG_CLS: ClassVar[type[SyncElasticsearchDatabaseConfig]] = (
        SyncElasticsearchDatabaseConfig
    )

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

        self.es_index = None  # Will be initialized in connect()
        self._connected = False

    def connect(self) -> None:
        """Connect to the Elasticsearch database."""
        if self._connected:
            return  # Already connected

        cfg = self.config

        # If vector is enabled but no vector fields defined yet, set up default
        if self._vector_enabled and not self.vector_fields:
            self.vector_fields[cfg.default_vector_field] = cfg.vector_dimensions

        # Get mappings with vector field support
        base_mappings = self.get_index_mappings(self.vector_fields)

        # Allow custom mappings to override
        mappings = cfg.mappings if cfg.mappings else base_mappings

        # Get settings optimized for KNN if we have vector fields
        settings = cfg.settings
        if not settings:
            settings = self.get_knn_index_settings() if (self.vector_fields or self._vector_enabled) else {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }

        # Initialize the Elasticsearch index
        self.es_index = SimplifiedElasticsearchIndex(
            index_name=self.index_name,
            host=self.host,
            port=self.port,
            settings=settings,
            mappings=mappings,
        )

        # Ensure index exists
        if not self.es_index.exists():
            self.es_index.create()

        # Create an Elasticsearch client for bulk operations
        from elasticsearch import Elasticsearch
        self.es_client = Elasticsearch([f"http://{self.host}:{self.port}"])

        self._connected = True

    def close(self) -> None:
        """Close the database connection."""
        if self.es_index:
            # ElasticsearchIndex manages its own connections
            self._connected = False  # type: ignore[unreachable]

    def _initialize(self) -> None:
        """Initialize method - connection setup moved to connect()."""
        # Configuration parsing stays here if needed
        pass

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.es_index:
            raise RuntimeError("Database not connected. Call connect() first.")

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
                        record_copy.metadata["vector_fields"][field_name] = {
                            "type": "vector",
                            "dimensions": self.vector_fields[field_name],
                            "source_field": field.source_field,
                            "model": getattr(field, "model_name", None),
                            "model_version": getattr(field, "model_version", None),
                        }

        doc = self._record_to_document(record_copy)
        if id:
            doc["id"] = id
        elif not doc.get("id"):
            doc["id"] = str(uuid.uuid4())

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
        # Use record's ID if it has one, otherwise generate a new one
        id = record.id if record.id else str(uuid.uuid4())
        doc = self._record_to_doc(record, id)

        # Index the document as an atomic insert. op_type="create" makes a
        # colliding id fail closed with a 409 conflict rather than overwrite.
        # Mirrors the async backend's try/except ConflictError shape so the two
        # conflict-detection paths cannot silently drift.
        try:
            response = self.es_index.index(
                body=doc,
                doc_id=id,
                refresh=self.refresh,
                op_type="create",
            )
        except ElasticsearchConflictError as e:
            raise DuplicateRecordError(id) from e

        if not response.get("_id"):
            raise DatabaseError(f"Failed to create record: {response}")

        return response["_id"]

    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        response = self.es_index.get(doc_id=id)

        if not response:
            return None

        doc = response.get("_source", {})
        return self._doc_to_record(doc)

    def get_version(self, id: str) -> str | None:
        """Return the document's ``_seq_no``/``_primary_term`` version token.

        Elasticsearch's native optimistic-concurrency pair
        (``_seq_no``, ``_primary_term``) advances on every write, so it is a
        native version — ABA-safe, unlike the base content-hash default this
        overrides. The two values are combined into one opaque token.
        """
        response = self.es_index.get(doc_id=id)
        if not response:
            return None
        seq_no = response.get("_seq_no")
        primary_term = response.get("_primary_term")
        if seq_no is None or primary_term is None:
            return None
        return es_version_token(seq_no, primary_term)

    def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record.

        When ``expected_version`` is provided the update carries ES's
        ``if_seq_no``/``if_primary_term`` guards so the compare-and-set is
        enforced server-side; a stale token raises ``ConcurrencyError``. When
        ``None`` the update is unconditional, byte-identical to prior behavior.
        """
        doc = self._record_to_doc(record, id)

        if expected_version is not None:
            seq_no, primary_term = parse_es_version_token(expected_version)
            try:
                return self.es_index.update(
                    doc_id=id,
                    body={"doc": doc},
                    refresh=self.refresh,
                    if_seq_no=seq_no,
                    if_primary_term=primary_term,
                )
            except ElasticsearchConflictError as e:
                # Either the doc is gone (update never inserts -> False) or the
                # token is stale (concurrent modification -> raise).
                current = self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current) from e

        # Update the document
        success = self.es_index.update(
            doc_id=id,
            body={"doc": doc},
            refresh=self.refresh,
        )

        return success

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record by ID.

        When ``expected_version`` is provided the delete carries ES's
        ``if_seq_no``/``if_primary_term`` guards so the compare-and-set is
        enforced server-side; a stale token raises ``ConcurrencyError`` and a
        missing document returns ``False``. When ``None`` the delete is
        unconditional, byte-identical to prior behavior.
        """
        if expected_version is not None:
            seq_no, primary_term = parse_es_version_token(expected_version)
            try:
                success = self.es_index.delete(
                    doc_id=id, if_seq_no=seq_no, if_primary_term=primary_term
                )
            except ElasticsearchConflictError as e:
                # Stale token (doc exists, version mismatch). If the doc is
                # already gone, treat it as not-found -> False.
                current = self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current) from e
        else:
            success = self.es_index.delete(doc_id=id)

        # Refresh if needed
        if success and self.refresh:
            self.es_index.refresh()

        return success

    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        return self.es_index.exists(doc_id=id)

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
            if self.update(id, record, expected_version=expected_version):
                return id
            raise version_conflict_error(id, expected_version, None)

        doc = self._record_to_doc(record, id)
        response = self.es_index.index(body=doc, doc_id=id, refresh=self.refresh)

        if response.get("_id"):
            return id
        else:
            raise DatabaseError(f"Failed to upsert record {id}: {response}")

    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records efficiently using the bulk API.

        Uses the bulk ``create`` op (fail-closed-by-id), honoring a
        caller-supplied ``record.id`` (minting a uuid only when absent). Like
        ``create()``, a colliding id fails closed: the bulk response is scanned
        for a 409 conflict and raises ``DuplicateRecordError``. The Elasticsearch
        bulk API is per-item (non-atomic), so — exactly like a ``create()`` loop —
        non-colliding records in the same batch may already be indexed when the
        conflict is raised.

        Args:
            records: List of records to create

        Returns:
            List of created record IDs

        Raises:
            DuplicateRecordError: a record collides with an existing id, or two
                records in the batch share an id.
        """
        if not records:
            return []

        # Build bulk operations
        bulk_operations = []
        ids = []
        seen: set[str] = set()

        for record in records:
            record_id = record.id or str(uuid.uuid4())
            if record_id in seen:
                raise DuplicateRecordError(record_id)
            seen.add(record_id)
            ids.append(record_id)

            # op_type "create" fails closed on a colliding id (409) instead of
            # overwriting, mirroring the single-record create().
            doc = self._record_to_doc(record, record_id)
            action = {
                "_op_type": "create",
                "_index": self.es_index.index_name,
                "_id": record_id,
                "_source": doc
            }
            bulk_operations.append(action)

        return self._execute_bulk_index(bulk_operations, ids, raise_on_conflict=True)

    def upsert_batch(self, records: list[Record]) -> list[str]:
        """Insert-or-overwrite multiple records efficiently using the bulk API.

        Uses the bulk ``index`` op (upsert-by-id), honoring a caller-supplied
        ``record.id`` (minting a uuid only when absent); a colliding id is
        overwritten (never raised). Returns ids in input order.
        """
        if not records:
            return []

        bulk_operations = []
        ids = []
        for record in records:
            record_id = record.id or str(uuid.uuid4())
            ids.append(record_id)
            doc = self._record_to_doc(record, record_id)
            bulk_operations.append(
                {
                    "_op_type": "index",  # index = upsert-by-id
                    "_index": self.es_index.index_name,
                    "_id": record_id,
                    "_source": doc,
                }
            )

        return self._execute_bulk_index(bulk_operations, ids)

    def _execute_bulk_index(
        self,
        bulk_operations: list[dict],
        ids: list[str],
        *,
        raise_on_conflict: bool = False,
    ) -> list[str]:
        """Run a bulk index/create request, returning the ids that succeeded.

        Shared by ``create_batch`` and ``upsert_batch``: the request execution
        and per-item error reconciliation are identical; only how each caller
        derives the id and op type differs.

        ``raise_on_conflict`` (set by ``create_batch``, whose ``create`` op fails
        closed) turns a per-item 409 conflict into ``DuplicateRecordError``,
        matching a ``create()`` loop; ``upsert_batch`` leaves it ``False`` (its
        ``index`` op cannot conflict).
        """
        from elasticsearch import helpers

        try:
            # Note: helpers.BulkIndexError may be raised if raise_on_error=True
            _success_count, errors = helpers.bulk(
                self.es_client,
                bulk_operations,
                refresh=self.refresh,
                raise_on_error=False,
                stats_only=False
            )
            # Process results to return actual IDs
            if errors:
                # Some operations failed - need to check which ones
                error_dict = {}
                for err in errors:
                    # Error dict can have 'index', 'create', 'update', or 'delete' keys
                    for op_type in ['index', 'create']:
                        if op_type in err:
                            op = err[op_type]
                            if raise_on_conflict and op.get('status') == 409:
                                raise DuplicateRecordError(op.get('_id'))
                            error_dict[op.get('_id')] = err
                            break

                result_ids = []
                for record_id in ids:
                    if record_id not in error_dict:
                        result_ids.append(record_id)
                    # Skip failed records
                return result_ids
            else:
                # All succeeded
                return ids

        except DuplicateRecordError:
            # A create() collision — propagate the fail-closed signal.
            raise
        except Exception as e:
            # Check if this is a BulkIndexError from the helpers module
            if hasattr(e, 'errors'):
                # Extract which operations failed. An error entry carries an
                # 'index' or 'create' key depending on the op type, so probe
                # both — mirroring the returned-errors reconciliation above, and
                # honoring raise_on_conflict so a 409 surfaced via this path also
                # fails closed rather than silently passing as success.
                failed_ids = set()
                for err in e.errors:
                    for op_type in ['index', 'create']:
                        if op_type in err:
                            op = err[op_type]
                            if raise_on_conflict and op.get('status') == 409:
                                raise DuplicateRecordError(op.get('_id')) from e
                            failed_ids.add(op.get('_id'))
                            break
                result_ids = []
                for record_id in ids:
                    if record_id not in failed_ids:
                        result_ids.append(record_id)
                    # Skip failed records
                return result_ids
            else:
                # Complete failure - return empty list
                return []

    def read_batch(self, ids: list[str]) -> list[Record | None]:
        """Read multiple records in batch."""
        records = []
        for id in ids:
            record = self.read(id)
            records.append(record)
        return records

    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records efficiently using the bulk API.
        
        Uses Elasticsearch's bulk API for efficient batch deletion.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            List of success flags for each deletion
        """
        if not ids:
            return []

        # Build bulk operations
        bulk_operations = []
        for record_id in ids:
            # Create action dict for bulk delete
            action = {
                "_op_type": "delete",
                "_index": self.es_index.index_name,
                "_id": record_id
            }
            bulk_operations.append(action)

        # Execute bulk delete
        from elasticsearch import helpers

        try:
            # Use the bulk helper for deletion
            _success_count, errors = helpers.bulk(
                self.es_client,
                bulk_operations,
                refresh=self.refresh,
                raise_on_error=False,
                stats_only=False
            )

            # Process results to determine which deletes succeeded
            results = []
            if errors:
                error_dict = {}
                for err in errors:
                    if 'delete' in err:
                        error_dict[err['delete'].get('_id')] = err

                for record_id in ids:
                    if record_id in error_dict:
                        # Check if error was "not found" (404) - that's still a successful delete
                        error = error_dict[record_id]
                        status = error.get('delete', {}).get('status')
                        results.append(status == 200 or status == 404)
                    else:
                        results.append(True)
            else:
                # All operations completed (either deleted or not found)
                results = [True] * len(ids)

            return results

        except Exception as e:
            # Check if this is a BulkIndexError from the helpers module
            if hasattr(e, 'errors'):
                # Extract which operations failed
                results = []
                failed_ids = {err.get('delete', {}).get('_id') for err in e.errors}

                for record_id in ids:
                    results.append(record_id not in failed_ids)

                return results
            else:
                # If bulk operation completely fails, mark all as failed
                return [False] * len(ids)

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records efficiently using the bulk API.
        
        Uses Elasticsearch's bulk API for efficient batch updates.
        
        Args:
            updates: List of (id, record) tuples to update
            
        Returns:
            List of success flags for each update
        """
        if not updates:
            return []

        # Build bulk operations
        bulk_operations = []
        for record_id, record in updates:
            # Create action dict for bulk update
            doc = self._record_to_doc(record, record_id)
            action = {
                "_op_type": "update",
                "_index": self.es_index.index_name,
                "_id": record_id,
                "doc": doc,
                "doc_as_upsert": False  # Don't create if doesn't exist
            }
            bulk_operations.append(action)

        # Execute bulk update
        from elasticsearch import helpers

        try:
            # Use the bulk helper for the update
            _success_count, errors = helpers.bulk(
                self.es_client,
                bulk_operations,
                refresh=self.refresh,
                raise_on_error=False,
                stats_only=False
            )

            # Process results to determine which updates succeeded
            results = []
            error_dict = {}
            if errors:
                for err in errors:
                    if 'update' in err:
                        error_dict[err['update']['_id']] = err

            for record_id, _ in updates:
                # Check if this ID had an error
                if record_id in error_dict:
                    error = error_dict[record_id]
                    # If error is 404 (not found), mark as failed
                    status = error.get('update', {}).get('status')
                    results.append(status == 200)  # Only 200 is success for update
                else:
                    results.append(True)

            return results

        except Exception as e:
            # Check if this is a BulkIndexError from the helpers module
            if hasattr(e, 'errors'):
                # Extract which operations failed
                results = []
                failed_ids = {err['update']['_id'] for err in e.errors}

                for record_id, _ in updates:
                    results.append(record_id not in failed_ids)

                return results
            else:
                # If bulk operation completely fails, mark all as failed
                return [False] * len(updates)

    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        # Translate through the shared filter->DSL functions so the sync,
        # async and vector-pre-filter sites cannot drift apart.
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
                # Special handling for 'id' field - sort by the id field in source data
                # We can't sort by _id directly as it requires fielddata which is disabled by default
                # The id field is already of type keyword, so no .keyword suffix needed
                if sort_spec.field == 'id':
                    field_path = "id"
                else:
                    field_path = f"data.{sort_spec.field}"
                    # Don't add .keyword if user already specified it or for common numeric fields
                    # This is a heuristic - ideally we'd check the mapping
                    numeric_fields = ['age', 'salary', 'balance', 'count', 'score', 'amount', 'price', 'index', 'number', 'total', 'quantity']
                    if (not sort_spec.field.endswith('.keyword') and
                        not sort_spec.field.endswith('.raw') and
                        sort_spec.field.lower() not in numeric_fields):
                        # Likely a text field, add .keyword for sorting
                        field_path = f"data.{sort_spec.field}.keyword"
                order = "desc" if sort_spec.order == SortOrder.DESC else "asc"
                sort.append({field_path: {"order": order}})

        # Build search body
        search_body = {"query": es_query}
        if sort:
            search_body["sort"] = sort
        # ``is not None`` so ``limit=0`` is sent as ``size=0`` (ES
        # interprets that as a count-only request returning zero hits)
        # rather than being silently dropped.
        if query.limit_value is not None:
            search_body["size"] = query.limit_value
        if query.offset_value is not None:
            search_body["from"] = query.offset_value

        # Execute search
        response = self.es_index.search(body=search_body)

        # Check if the response is valid (has the expected structure)
        # An empty result set is still a valid response
        if not hasattr(response, 'json') or response.json is None:
            raise DatabaseError(f"Invalid search response: {response}")

        # Check for actual errors in the response
        if 'error' in response.json:
            raise DatabaseError(f"Failed to search records: {response.json['error']}")

        # Parse results
        records = []
        hits = response.json.get("hits", {}).get("hits", [])
        for hit in hits:
            doc = hit.get("_source", {})
            records.append(self._doc_to_record(doc))

        # Apply field projection if specified
        if query.fields:
            for record in records:
                # Keep only specified fields
                field_names = list(record.fields.keys())
                for field_name in field_names:
                    if field_name not in query.fields:
                        del record.fields[field_name]

        return records

    def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        return self.es_index.count()

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
        es_query = build_bool_query(query.filters)
        return self.es_index.count(body={"query": es_query})

    def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()
        # Get count before deletion
        count = self._count_all()

        # Delete by query - delete all documents
        response = self.es_index.delete_by_query(
            body={"query": {"match_all": {}}}
        )

        # Refresh if needed
        if self.refresh:
            self.es_index.refresh()

        return response.get("deleted", count)

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
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
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
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

    def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        field_name: str = "embedding",
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using Elasticsearch KNN.
        
        Note: This is a synchronous wrapper around the async implementation.
        For production use, consider using the async version for better performance.
        
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
            field_name=field_name,
            k=k,
            filter_query=filter_query,
        )

        # Execute search using the es_client
        try:
            response = self.es_client.search(
                index=self.index_name,
                **query,
                size=k,
                source=include_source,
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
                if not record.has_storage_id():
                    record.storage_id = hit["_id"]

            # Skip if no record (shouldn't happen if include_source is True)
            if record is None:
                continue

            results.append(VectorSearchResult(
                record=record,
                score=score,
                vector_field=field_name,
                metadata={
                    "index": self.index_name,
                    "metric": metric.value,
                    "doc_id": hit["_id"],
                },
            ))

        return results

    def create_vector_index(
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

        # Update index mapping using the es_client
        try:
            self.es_client.indices.put_mapping(
                index=self.index_name,
                properties={
                    f"data.{vector_field}": mapping
                }
            )

            # Track the vector field
            self.vector_fields[vector_field] = dimensions
            self._vector_enabled = True

            logger.info(f"Created vector mapping for field '{vector_field}' with {dimensions} dimensions")
            return True

        except Exception as e:
            self._handle_elasticsearch_error(e, "create vector index")
            return False


# Import the native async implementation

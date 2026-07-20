"""S3 backend implementation with proper connection management."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dataknobs_common.aws import AwsSessionConfig
from dataknobs_common.structured_config import StructuredConfigConsumer

from dataknobs_data.database import SyncDatabase, version_conflict_error
from dataknobs_data.exceptions import DuplicateRecordError
from dataknobs_data.pooling.s3 import create_boto3_s3_client, is_s3_conditional_conflict
from dataknobs_data.query import Query, is_storage_key_field
from dataknobs_data.records import Record
from dataknobs_data.streaming import (
    StreamConfig,
    StreamResult,
    resolve_conflict_write,
    run_stream_write,
)

from ..vector import VectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .config import SyncS3DatabaseConfig
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import ClassVar


logger = logging.getLogger(__name__)


class SyncS3Database(  # type: ignore[misc]
    StructuredConfigConsumer[SyncS3DatabaseConfig],
    SyncDatabase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    """S3-based database backend with proper connection management.

    Stores records as JSON objects in S3 with metadata as tags.
    Constructed through :class:`SyncS3DatabaseConfig` — every documented
    config key is a typed field on that dataclass, so ``self.config`` is
    the typed config (not a dict) and the ``from_config`` / factory paths
    share one construction route.
    """

    CONFIG_CLS: ClassVar[type[SyncS3DatabaseConfig]] = SyncS3DatabaseConfig

    def _setup(self) -> None:
        """Derive the session config and connection state from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op — connection setup is deferred to
        :meth:`connect`). ``bucket`` validation and ``prefix`` /
        credential-alias normalization already happened in the config
        dataclass.
        """
        cfg = self.config

        # Connection state
        self.s3_client = None
        self._connected = False

        # Cache for performance
        self._index_cache: dict[str, Any] = {}
        self._cache_dirty = True

        self.bucket = cfg.bucket
        self.prefix = cfg.prefix
        self.multipart_threshold = cfg.multipart_threshold
        self.multipart_chunksize = cfg.multipart_chunksize

        # Single normalized session config built from the typed (canonical)
        # fields. Alias acceptance (``region``, ``max_workers`` …) happened
        # in ``SyncS3DatabaseConfig._normalize_dict``.
        self._session_config = AwsSessionConfig(
            region_name=cfg.region_name,
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
            aws_session_token=cfg.aws_session_token,
            max_pool_connections=cfg.max_pool_connections,
            max_attempts=cfg.max_attempts,
            retry_mode=cfg.retry_mode,
            extra_client_kwargs=cfg.extra_client_kwargs,
        )

        # Backward-compat attributes used elsewhere in this file and by
        # downstream consumers reading ``db.region`` / ``db.endpoint_url``
        # / ``db.max_workers``. ``self.region`` may be ``None`` until
        # ``connect()`` runs; resolved-region reads should prefer
        # ``self.s3_client.meta.region_name`` post-connect.
        self.region = self._session_config.region_name
        self.endpoint_url = self._session_config.endpoint_url
        self.max_workers = self._session_config.max_pool_connections
        self.max_retries = self._session_config.max_attempts
        self.aws_access_key_id = self._session_config.aws_access_key_id
        self.aws_secret_access_key = self._session_config.aws_secret_access_key
        self.aws_session_token = self._session_config.aws_session_token

        # Initialize vector support
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)
        self._init_vector_state()

    def connect(self) -> None:
        """Connect to S3 service."""
        if self._connected:
            return  # Already connected

        from botocore.exceptions import ClientError

        self.s3_client = create_boto3_s3_client(self._session_config)
        self.ClientError = ClientError

        # Verify bucket exists or create it
        self._ensure_bucket_exists()

        self._connected = True
        logger.info(
            "Connected to S3 with bucket=%s, prefix=%s, region=%s",
            self.bucket,
            self.prefix,
            self.s3_client.meta.region_name,
        )

    def close(self) -> None:
        """Close the S3 connection."""
        if self.s3_client:
            # S3 client doesn't need explicit closing, but clear cache
            self._index_cache = {}  # type: ignore[unreachable]
            self._connected = False
            logger.info(f"Closed S3 connection to bucket={self.bucket}")

    def _initialize(self) -> None:
        """Initialize method - connection setup moved to connect()."""
        pass

    def _check_connection(self) -> None:
        """Check if S3 client is connected."""
        if not self._connected or not self.s3_client:
            raise RuntimeError("S3 not connected. Call connect() first.")

    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create if necessary."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.debug(f"Bucket {self.bucket} exists")
        except self.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.info(f"Creating bucket {self.bucket}")
                # ``self.region`` may be ``None`` if config relies on
                # boto's default chain; the constructed client always
                # has a concrete region in ``meta.region_name``.
                effective_region = (
                    self.region or self.s3_client.meta.region_name
                )
                if effective_region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={
                            'LocationConstraint': effective_region
                        },
                    )
            else:
                raise

    def _get_object_key(self, record_id: str) -> str:
        """Generate S3 object key for a record ID."""
        return f"{self.prefix}{record_id}.json"

    def _record_to_s3_object(self, record: Record) -> dict[str, Any]:
        """Convert a Record to S3 object data."""
        # Use Record's built-in serialization which handles VectorFields
        # Use non-flattened format to preserve field metadata
        record_dict = record.to_dict(include_metadata=True, flatten=False)

        return record_dict

    def _s3_object_to_record(self, obj_data: dict[str, Any]) -> Record:
        """Convert S3 object data to a Record."""
        # Use Record's built-in deserialization
        return Record.from_dict(obj_data)

    def create(self, record: Record) -> str:
        """Create a new record in S3.

        Atomic create-if-absent is enforced with a conditional PUT
        (``If-None-Match: *``): a colliding id raises ``DuplicateRecordError``.
        The guarantee holds against any S3 implementation that honors
        conditional writes (real AWS S3, recent LocalStack). Older stores
        that ignore the header degrade to last-writer-wins.
        """
        self._check_connection()

        # Use centralized method to prepare record
        record_copy, storage_id = self._prepare_record_for_storage(record)
        key = self._get_object_key(storage_id)

        # Set metadata
        record_copy.metadata = record_copy.metadata or {}
        record_copy.metadata["id"] = storage_id
        now = datetime.now(UTC)
        record_copy.metadata["created_at"] = now.isoformat()
        record_copy.metadata["updated_at"] = now.isoformat()

        # Convert record to JSON
        obj_data = self._record_to_s3_object(record_copy)
        body = json.dumps(obj_data)

        # Store in S3 as an atomic insert. IfNoneMatch="*" makes the PUT fail
        # closed if the key already exists (412 PreconditionFailed) or if a
        # concurrent conditional write races it (409 ConditionalRequestConflict),
        # so a colliding id cannot silently overwrite an existing record.
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType='application/json',
                IfNoneMatch='*',
            )
        except self.ClientError as e:
            if is_s3_conditional_conflict(e):
                raise DuplicateRecordError(storage_id) from e
            raise

        # Invalidate cache
        self._cache_dirty = True

        logger.debug(f"Created record {storage_id} at {key}")
        return storage_id

    def read(self, id: str) -> Record | None:
        """Read a record from S3."""
        self._check_connection()

        key = self._get_object_key(id)

        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            body = response['Body'].read()
            obj_data = json.loads(body)
            record = self._s3_object_to_record(obj_data)
            # Use centralized method to prepare record
            return self._prepare_record_from_storage(record, id)
        except self.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def get_version(self, id: str) -> str | None:
        """Return the object's S3 ``ETag`` as the version token.

        The ETag changes whenever the object's bytes change, and S3 enforces
        it server-side via ``If-Match`` on the conditional PUT, so it is a
        native version token — overrides the base content-hash default.
        """
        self._check_connection()
        key = self._get_object_key(id)
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=key)
        except self.ClientError as e:
            if e.response['Error']['Code'] in ('404', 'NoSuchKey'):
                return None
            raise
        return response.get('ETag')

    def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record in S3.

        When ``expected_version`` is provided the PUT carries an
        ``If-Match: <ETag>`` guard so the compare-and-set is enforced by S3; a
        stale token raises ``ConcurrencyError``. When ``None`` the update is
        unconditional, byte-identical to prior behavior. The compare-and-set
        holds against any S3 implementation that honors conditional writes
        (real AWS S3, recent LocalStack).
        """
        self._check_connection()

        key = self._get_object_key(id)

        # Check if exists and get existing metadata
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            existing_data = json.loads(response['Body'].read())
            existing_metadata = existing_data.get("metadata", {})
        except self.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise

        # Preserve and update metadata
        record.metadata = record.metadata or {}
        record.metadata["id"] = id
        record.metadata["created_at"] = existing_metadata.get("created_at", datetime.now(UTC).isoformat())
        record.metadata["updated_at"] = datetime.now(UTC).isoformat()

        # Update the object
        obj_data = self._record_to_s3_object(record)
        body = json.dumps(obj_data)

        put_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": body,
            "ContentType": "application/json",
        }
        if expected_version is not None:
            put_kwargs["IfMatch"] = expected_version

        try:
            self.s3_client.put_object(**put_kwargs)
        except self.ClientError as e:
            if expected_version is not None and is_s3_conditional_conflict(e):
                # The guarded PUT lost: either the object is gone (update never
                # inserts -> False) or the ETag is stale (-> raise).
                current = self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current) from e
            raise

        # Invalidate cache
        self._cache_dirty = True

        logger.debug(f"Updated record {id} at {key}")
        return True

    def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record from S3.

        When ``expected_version`` is provided the ``DeleteObject`` carries an
        ``If-Match: <ETag>`` guard so the compare-and-set is enforced by S3; a
        stale token raises ``ConcurrencyError`` and a missing object returns
        ``False``. When ``None`` the delete is unconditional, byte-identical to
        prior behavior. The compare-and-set holds against any S3 implementation
        that honors conditional deletes (real AWS S3, recent LocalStack).
        """
        self._check_connection()

        key = self._get_object_key(id)

        if expected_version is not None:
            try:
                self.s3_client.delete_object(
                    Bucket=self.bucket, Key=key, IfMatch=expected_version
                )
            except self.ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey"):
                    # A conditional delete never conflicts on an absent id.
                    return False
                if is_s3_conditional_conflict(e):
                    # The guarded delete lost: the ETag is stale (-> raise) or
                    # the object vanished mid-op (-> False).
                    current = self.get_version(id)
                    if current is None:
                        return False
                    raise version_conflict_error(id, expected_version, current) from e
                raise
            self._cache_dirty = True
            logger.debug(f"Conditionally deleted record {id} at {key}")
            return True

        # Check if exists
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
        except self.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

        # Delete the object
        self.s3_client.delete_object(Bucket=self.bucket, Key=key)

        # Invalidate cache
        self._cache_dirty = True

        logger.debug(f"Deleted record {id} at {key}")
        return True

    def list_all(self) -> list[str]:
        """List all record IDs in the database.
        
        Returns:
            List of all record IDs
        """
        self._check_connection()
        record_ids = []

        # Use paginator for large buckets
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.prefix
        )

        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract record ID from key
                    if key.startswith(self.prefix) and key.endswith('.json'):
                        record_id = key[len(self.prefix):-5]  # Remove prefix and .json
                        record_ids.append(record_id)

        logger.debug(f"Listed {len(record_ids)} records from S3")
        return record_ids

    def exists(self, id: str) -> bool:
        """Check if a record exists in S3."""
        self._check_connection()

        key = self._get_object_key(id)

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    def search(self, query: Query) -> list[Record]:
        """Search for records matching the query.
        
        Note: S3 doesn't support complex queries, so we need to list and filter.
        """
        self._check_connection()

        # List all objects with the prefix. Collect (id, record) tuples and
        # let the shared ``_process_search_results`` helper apply sorting /
        # offset / limit / projection — the single canonical implementation
        # every backend shares (it correctly orders falsy sort values such
        # as a numeric ``0``).
        results: list[tuple[str, Record]] = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        for page in pages:
            if 'Contents' not in page:
                continue

            # Fetch objects in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json'):
                        future = executor.submit(self._fetch_and_filter, obj['Key'], query)
                        futures.append(future)

                for future in as_completed(futures):
                    record = future.result()
                    if record:
                        results.append((record.id or "", record))

        # Records are freshly deserialized from S3 (no shared aliasing), so
        # ``ensure_record_id`` inside the helper is the only copy needed.
        return self._process_search_results(results, query, deep_copy=False)

    def _fetch_and_filter(self, key: str, query: Query) -> Record | None:
        """Fetch an object and apply query filters."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            body = response['Body'].read()
            obj_data = json.loads(body)
            record = self._s3_object_to_record(obj_data)

            # Apply filters
            for filter in query.filters:
                # The reserved storage-key field routes to the storage key.
                if is_storage_key_field(filter.field):
                    field_value = record.id
                else:
                    field_value = record.get_value(filter.field)
                if not filter.matches(field_value):
                    return None

            return record
        except Exception as e:
            logger.warning(f"Error fetching {key}: {e}")
            return None

    def _count_all(self) -> int:
        """Count all records in S3."""
        self._check_connection()

        count = 0
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        for page in pages:
            if 'Contents' in page:
                count += sum(1 for obj in page['Contents'] if obj['Key'].endswith('.json'))

        return count

    def clear(self) -> int:
        """Clear all records from S3."""
        self._check_connection()

        # List and delete all objects
        count = 0
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        for page in pages:
            if 'Contents' not in page:
                continue

            # Delete in batches
            objects = [{'Key': obj['Key']} for obj in page['Contents'] if obj['Key'].endswith('.json')]
            if objects:
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': objects}
                )
                count += len(objects)

        # Clear cache
        self._index_cache = {}
        self._cache_dirty = True

        logger.info(f"Cleared {count} records from S3")
        return count

    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from S3."""
        self._check_connection()
        config = config or StreamConfig()

        # List objects and stream them
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        batch = []
        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                if not obj['Key'].endswith('.json'):
                    continue

                record = self._fetch_and_filter(obj['Key'], query or Query())
                if record:
                    batch.append(record)

                    if len(batch) >= config.batch_size:
                        for r in batch:
                            yield r
                        batch = []

        # Yield remaining records
        for r in batch:
            yield r

    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into S3.

        INSERT routes through per-record ``create()`` (``insert_batch_func=None``)
        rather than the batch fast-path: the per-key S3 writes are
        non-transactional, so a partial-success batch followed by the per-record
        fallback would re-write the already-written keys and count them as
        spurious duplicate failures. A per-key conditional ``create`` PUT is the
        natural granularity and fails closed cleanly. UPSERT keeps the bulk
        ``upsert_batch`` fast-path (overwrite is idempotent, so partial success
        is benign).
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
        return run_stream_write(
            records,
            batch_write_func=batch_write_func,
            single_write_func=single_write_func,
            skip_on_duplicate=skip_on_duplicate,
            config=config,
        )

    def vector_search(
        self,
        query_vector,
        vector_field: str = "embedding",
        k: int = 10,
        filter=None,
        metric=None,
        **kwargs
    ):
        """Perform vector similarity search using Python calculations.
        
        WARNING: This implementation downloads all records from S3 to perform
        the search locally. This is inefficient for large datasets. Consider
        using a vector-enabled backend like PostgreSQL or Elasticsearch for
        production use with large datasets.
        """
        return self.python_vector_search_sync(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )


# Import the native async implementation

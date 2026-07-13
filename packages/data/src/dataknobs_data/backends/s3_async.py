"""Native async S3 backend implementation with aioboto3 and connection pooling."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from dataknobs_common.aws import create_aioboto3_session
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import AsyncDatabase, version_conflict_error
from ..exceptions import DuplicateRecordError
from ..pooling import ConnectionPoolManager
from ..pooling.s3 import S3PoolConfig, is_s3_conditional_conflict, validate_s3_session
from ..query import Operator, Query
from ..records import Record
from ..streaming import StreamConfig, StreamResult, async_process_batch_with_fallback
from ..vector import VectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .config import AsyncS3DatabaseConfig
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import ClassVar


logger = logging.getLogger(__name__)

# Global pool manager for S3 sessions
_session_manager = ConnectionPoolManager()


class AsyncS3Database(  # type: ignore[misc]
    StructuredConfigConsumer[AsyncS3DatabaseConfig],
    AsyncDatabase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    """Native async S3 database backend with aioboto3 and session pooling.

    Constructed through :class:`AsyncS3DatabaseConfig` — every documented
    config key is a typed field on that dataclass, so ``self.config`` is
    the typed config (not a dict). The ``S3PoolConfig`` is derived from it
    in :meth:`_setup`.
    """

    CONFIG_CLS: ClassVar[type[AsyncS3DatabaseConfig]] = AsyncS3DatabaseConfig

    def _setup(self) -> None:
        """Derive the pool config and connection state from the typed config.

        Runs after the cooperative base chain has set ``self.schema`` and
        run ``_initialize`` (a no-op — connection setup is deferred to
        :meth:`connect`). ``bucket`` validation and credential-alias
        normalization already happened in the config dataclass.
        """
        cfg = self.config

        self._pool_config = S3PoolConfig(
            bucket=cast("str", cfg.bucket),
            prefix=cfg.prefix,
            region_name=cfg.region_name,
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
            aws_session_token=cfg.aws_session_token,
            endpoint_url=cfg.endpoint_url,
        )
        # Public, symmetric with ``SyncS3Database.region``: the resolved
        # region (``None`` when config relies on the boto default chain).
        # Lets callers/tests inspect region resolution without reaching
        # into ``_pool_config``.
        self.region = self._pool_config.region_name
        self._session = None
        self._connected = False

        # Initialize vector support
        self._apply_vector_config(cfg.vector_enabled, cfg.vector_metric)
        self._init_vector_state()  # From SQLiteVectorSupport

    async def connect(self) -> None:
        """Connect to S3 service."""
        if self._connected:
            return

        # Get or create session for current event loop
        from ..pooling import BasePoolConfig
        self._session = await _session_manager.get_pool(
            self._pool_config,
            cast("Callable[[BasePoolConfig], Awaitable[Any]]", create_aioboto3_session),
            lambda session: validate_s3_session(
                session, self._pool_config.bucket, self._pool_config
            ),
        )

        # Invariant: nothing fallible runs between the get_pool increment and
        # _connected = True, so the holder count is always balanced (close()
        # releases on the success path). If a step that can raise is added
        # here, it MUST release the holder on failure (release_pool) — see the
        # elasticsearch/postgres connect() paths — or the slot leaks.
        self._connected = True

    async def close(self) -> None:
        """Release this holder's claim on the shared aioboto3 session.

        Sessions are shared by DSN across instances on a loop and owned by
        the pool manager. ``close()`` releases this holder; the session is
        evicted when the last holder releases. (aioboto3 sessions are
        stateless, so there is no registered close-func; the release simply
        keeps the manager's accounting honest and the contract uniform
        across backends.)
        """
        if self._connected:
            try:
                await _session_manager.release_pool(self._pool_config)
            except Exception as e:
                logger.warning("Error releasing S3 session: %s", e)
            self._session = None
            self._connected = False

    def _initialize(self) -> None:
        """Initialize is handled in connect."""
        pass

    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self._session:
            raise RuntimeError("Database not connected. Call connect() first.")

    def _get_key(self, id: str) -> str:
        """Get the S3 key for a given record ID."""
        if self._pool_config.prefix:
            return f"{self._pool_config.prefix}/{id}.json"
        return f"{id}.json"

    def _record_to_s3_object(self, record: Record) -> dict[str, Any]:
        """Convert a Record to an S3 object."""
        # Use Record's built-in serialization which handles VectorFields
        record_dict = record.to_dict(include_metadata=True, flatten=False)

        # Add timestamps
        now = datetime.now(UTC).isoformat()
        if "metadata" not in record_dict:
            record_dict["metadata"] = {}
        record_dict["metadata"]["created_at"] = record_dict["metadata"].get("created_at", now)
        record_dict["metadata"]["updated_at"] = now

        return record_dict

    def _s3_object_to_record(self, obj: dict[str, Any]) -> Record:
        """Convert an S3 object to a Record."""
        # Use Record's built-in deserialization
        return Record.from_dict(obj)

    async def create(self, record: Record) -> str:
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
        key = self._get_key(storage_id)
        obj = self._record_to_s3_object(record_copy)

        # Add ID to metadata
        obj["metadata"]["id"] = storage_id

        from botocore.exceptions import ClientError

        # Atomic insert. IfNoneMatch="*" makes the PUT fail closed if the key
        # already exists (412 PreconditionFailed) or if a concurrent conditional
        # write races it (409 ConditionalRequestConflict), so a colliding id
        # cannot silently overwrite an existing record.
        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            try:
                await s3.put_object(
                    Bucket=self._pool_config.bucket,
                    Key=key,
                    Body=json.dumps(obj),
                    ContentType="application/json",
                    IfNoneMatch="*",
                )
            except ClientError as e:
                if is_s3_conditional_conflict(e):
                    raise DuplicateRecordError(storage_id) from e
                raise

        return storage_id

    async def read(self, id: str) -> Record | None:
        """Read a record from S3."""
        self._check_connection()

        key = self._get_key(id)

        try:
            async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
                response = await s3.get_object(
                    Bucket=self._pool_config.bucket,
                    Key=key
                )

                # Read the object body
                body = await response['Body'].read()
                obj = json.loads(body)

                record = self._s3_object_to_record(obj)
                # Use centralized method to prepare record
                record = self._prepare_record_from_storage(record, id)
                # Ensure ID is in metadata
                record.metadata["id"] = id

                return record
        except Exception:
            return None

    async def get_version(self, id: str) -> str | None:
        """Return the object's S3 ``ETag`` as the version token.

        The ETag changes whenever the object's bytes change, and S3 enforces
        it server-side via ``If-Match`` on the conditional PUT, so it is a
        native version token — overrides the base content-hash default.
        """
        self._check_connection()
        key = self._get_key(id)
        from botocore.exceptions import ClientError

        try:
            async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
                response = await s3.head_object(
                    Bucket=self._pool_config.bucket,
                    Key=key,
                )
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise
        return response.get("ETag")

    async def update(
        self, id: str, record: Record, *, expected_version: str | None = None
    ) -> bool:
        """Update an existing record in S3.

        When ``expected_version`` is provided the PUT carries an
        ``If-Match: <ETag>`` guard so the compare-and-set is enforced by S3; a
        stale token raises ``ConcurrencyError``. When ``None`` the update is
        unconditional, byte-identical to prior behavior.
        """
        self._check_connection()

        # Check if record exists
        if not await self.exists(id):
            return False

        key = self._get_key(id)
        obj = self._record_to_s3_object(record)

        # Preserve ID in metadata
        obj["metadata"]["id"] = id

        put_kwargs: dict[str, Any] = {
            "Bucket": self._pool_config.bucket,
            "Key": key,
            "Body": json.dumps(obj),
            "ContentType": "application/json",
        }
        if expected_version is not None:
            put_kwargs["IfMatch"] = expected_version

        from botocore.exceptions import ClientError

        try:
            async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
                await s3.put_object(**put_kwargs)
        except ClientError as e:
            if expected_version is not None and is_s3_conditional_conflict(e):
                # The guarded PUT lost: either the object is gone (update never
                # inserts -> False) or the ETag is stale (-> raise).
                current = await self.get_version(id)
                if current is None:
                    return False
                raise version_conflict_error(id, expected_version, current) from e
            raise

        return True

    async def delete(self, id: str) -> bool:
        """Delete a record from S3."""
        self._check_connection()

        key = self._get_key(id)

        try:
            async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
                await s3.delete_object(
                    Bucket=self._pool_config.bucket,
                    Key=key
                )
            return True
        except Exception:
            return False

    async def exists(self, id: str) -> bool:
        """Check if a record exists in S3."""
        self._check_connection()

        key = self._get_key(id)

        try:
            async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
                await s3.head_object(
                    Bucket=self._pool_config.bucket,
                    Key=key
                )
            return True
        except Exception:
            return False

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
        record must already exist with a matching ETag, otherwise it raises
        ``ConcurrencyError``. A conditional upsert never inserts.
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
            # Conditional upsert never inserts: require an existing object and
            # let update() enforce the ETag If-Match compare-and-set.
            if not await self.exists(id):
                raise version_conflict_error(id, expected_version, None)
            await self.update(id, record, expected_version=expected_version)
            return id

        key = self._get_key(id)
        obj = self._record_to_s3_object(record)

        # Add ID to metadata
        obj["metadata"]["id"] = id

        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            await s3.put_object(
                Bucket=self._pool_config.bucket,
                Key=key,
                Body=json.dumps(obj),
                ContentType="application/json"
            )

        return id

    async def search(self, query: Query) -> list[Record]:
        """Search for records matching the query."""
        self._check_connection()

        # S3 doesn't support complex queries, so we need to list and filter.
        # Collect (id, record) tuples; the shared ``_process_search_results``
        # helper applies sorting / offset / limit / projection consistently
        # with every other backend (and correctly handles falsy sort values
        # such as a numeric ``0``, which a hand-rolled ``or ""`` key coerces
        # to a string and crashes on under mixed types).
        results: list[tuple[str, Record]] = []

        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            # List all objects
            paginator = s3.get_paginator('list_objects_v2')

            params = {
                'Bucket': self._pool_config.bucket,
            }
            if self._pool_config.prefix:
                params['Prefix'] = self._pool_config.prefix

            async for page in paginator.paginate(**params):
                if 'Contents' not in page:
                    continue

                # Process each object
                for obj_summary in page['Contents']:
                    key = obj_summary['Key']

                    # Skip non-JSON files
                    if not key.endswith('.json'):
                        continue

                    # Get the object
                    response = await s3.get_object(
                        Bucket=self._pool_config.bucket,
                        Key=key
                    )

                    body = await response['Body'].read()
                    obj = json.loads(body)
                    record = self._s3_object_to_record(obj)

                    # Extract ID from key
                    id = key.replace(self._pool_config.prefix + '/', '').replace('.json', '')
                    record.metadata["id"] = id

                    # Apply filters
                    if self._matches_filters(record, query.filters):
                        results.append((id, record))

        # Records are freshly deserialized from S3 (no shared aliasing), so
        # ``ensure_record_id`` inside the helper is the only copy needed.
        return self._process_search_results(results, query, deep_copy=False)

    def _matches_filters(self, record: Record, filters: list) -> bool:
        """Check if a record matches all filters."""
        for filter in filters:
            field = record.get_field(filter.field)
            if not field:
                return False

            value = field.value

            if filter.operator == Operator.EQ:
                if value != filter.value:
                    return False
            elif filter.operator == Operator.NEQ:
                if value == filter.value:
                    return False
            elif filter.operator == Operator.GT:
                if value <= filter.value:
                    return False
            elif filter.operator == Operator.LT:
                if value >= filter.value:
                    return False
            elif filter.operator == Operator.GTE:
                if value < filter.value:
                    return False
            elif filter.operator == Operator.LTE:
                if value > filter.value:
                    return False
            elif filter.operator == Operator.LIKE:
                if str(filter.value) not in str(value):
                    return False
            elif filter.operator == Operator.IN:
                if value not in filter.value:
                    return False
            elif filter.operator == Operator.NOT_IN:
                if value in filter.value:
                    return False

        return True

    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()

        count = 0
        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            paginator = s3.get_paginator('list_objects_v2')

            params = {
                'Bucket': self._pool_config.bucket,
            }
            if self._pool_config.prefix:
                params['Prefix'] = self._pool_config.prefix

            async for page in paginator.paginate(**params):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.json'):
                            count += 1

        return count

    async def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()

        count = 0
        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            # List and delete all objects
            paginator = s3.get_paginator('list_objects_v2')

            params = {
                'Bucket': self._pool_config.bucket,
            }
            if self._pool_config.prefix:
                params['Prefix'] = self._pool_config.prefix

            async for page in paginator.paginate(**params):
                if 'Contents' not in page:
                    continue

                # Build delete request
                objects_to_delete = []
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json'):
                        objects_to_delete.append({'Key': obj['Key']})
                        count += 1

                # Delete in batch
                if objects_to_delete:
                    await s3.delete_objects(
                        Bucket=self._pool_config.bucket,
                        Delete={'Objects': objects_to_delete}
                    )

        return count

    async def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from S3."""
        self._check_connection()
        config = config or StreamConfig()

        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            paginator = s3.get_paginator('list_objects_v2')

            params = {
                'Bucket': self._pool_config.bucket,
                'MaxKeys': config.batch_size
            }
            if self._pool_config.prefix:
                params['Prefix'] = self._pool_config.prefix

            async for page in paginator.paginate(**params):
                if 'Contents' not in page:
                    continue

                for obj_summary in page['Contents']:
                    key = obj_summary['Key']

                    if not key.endswith('.json'):
                        continue

                    # Get the object
                    response = await s3.get_object(
                        Bucket=self._pool_config.bucket,
                        Key=key
                    )

                    body = await response['Body'].read()
                    obj = json.loads(body)
                    record = self._s3_object_to_record(obj)

                    # Extract ID from key
                    id = key.replace(self._pool_config.prefix + '/', '').replace('.json', '')
                    record.metadata["id"] = id

                    # Apply filters if query provided
                    if query and query.filters:
                        if not self._matches_filters(record, query.filters):
                            continue

                    # Apply field projection
                    if query and query.fields:
                        record = record.project(query.fields)

                    yield record

    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into S3."""
        self._check_connection()
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False

        batch = []
        async for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch with graceful fallback
                async def batch_func(b):
                    await self._write_batch(b)
                    return [r.id for r in b]

                continue_processing = await async_process_batch_with_fallback(
                    batch,
                    batch_func,
                    self.create,
                    result,
                    config
                )

                if not continue_processing:
                    quitting = True
                    break

                batch = []

        # Write remaining batch
        if batch and not quitting:
            async def batch_func(b):
                await self._write_batch(b)
                return [r.id for r in b]

            await async_process_batch_with_fallback(
                batch,
                batch_func,
                self.create,
                result,
                config
            )

        result.duration = time.time() - start_time
        return result

    async def _write_batch(self, records: list[Record]) -> None:
        """Write a batch of records to S3."""
        if not records:
            return

        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            # Write each record (S3 doesn't have native batch write)
            # We could potentially use multipart upload for very large batches
            tasks = []
            for record in records:
                id = str(uuid.uuid4())
                key = self._get_key(id)
                obj = self._record_to_s3_object(record)
                obj["metadata"]["id"] = id

                task = s3.put_object(
                    Bucket=self._pool_config.bucket,
                    Key=key,
                    Body=json.dumps(obj),
                    ContentType="application/json"
                )
                tasks.append(task)

            # Execute all uploads concurrently
            await asyncio.gather(*tasks)

    async def list_all(self) -> list[str]:
        """List all record IDs in the database."""
        self._check_connection()

        ids = []
        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            paginator = s3.get_paginator('list_objects_v2')

            params = {
                'Bucket': self._pool_config.bucket,
            }
            if self._pool_config.prefix:
                params['Prefix'] = self._pool_config.prefix

            async for page in paginator.paginate(**params):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.json'):
                        # Extract ID from key
                        id = key.replace(self._pool_config.prefix + '/', '').replace('.json', '')
                        ids.append(id)

        return ids

    async def vector_search(
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
        
        Future optimization: Override this method to use AWS OpenSearch or
        similar vector-enabled service when available.
        """
        return await self.python_vector_search_async(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
            **kwargs
        )

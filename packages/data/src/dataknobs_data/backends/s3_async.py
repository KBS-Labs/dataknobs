# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Native async S3 backend implementation with aioboto3 and connection pooling."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_common.aws import create_aioboto3_session
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..database import AsyncDatabase, version_conflict_error
from ..exceptions import DuplicateRecordError
from ..pooling import ConnectionPoolManager
from ..pooling.s3 import (
    S3PoolConfig,
    is_s3_conditional_conflict,
    is_s3_missing_object,
    validate_s3_session,
)
from ..query import Query, is_storage_key_field
from ..query_logic import ComplexQuery
from ..records import Record
from ..streaming import (
    StreamConfig,
    StreamResult,
    async_run_stream_write,
    resolve_conflict_write,
)
from ..vector import AsyncVectorOperationsMixin
from ..vector.bulk_embed_mixin import AsyncBulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .config import AsyncS3DatabaseConfig
from .s3_mixins import S3ObjectMixin
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import ClassVar

    import numpy as np

    from ..vector.types import DistanceMetric, VectorSearchResult


logger = logging.getLogger(__name__)

# Global pool manager for S3 sessions. `Any` is the pool type because that is
# what `create_aioboto3_session` returns and what `aioboto3.*` resolves to under
# this package's `ignore_missing_imports` override -- naming `aioboto3.Session`
# instead would add an import and check exactly as much.
_session_manager: ConnectionPoolManager[Any] = ConnectionPoolManager()


class AsyncS3Database(
    StructuredConfigConsumer[AsyncS3DatabaseConfig],
    S3ObjectMixin,
    AsyncDatabase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    AsyncBulkEmbedMixin,
    AsyncVectorOperationsMixin,
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
        self.prefix = cfg.prefix
        self._max_concurrency = cfg.max_pool_connections
        # Public, symmetric with ``SyncS3Database.region``: the resolved
        # region (``None`` when config relies on the boto default chain).
        # Lets callers/tests inspect region resolution without reaching
        # into ``_pool_config``.
        self.region = self._pool_config.region_name
        # Same `Any` as the pool manager above: without the annotation this
        # infers as `None`, and every `self._session.client(...)` in the file
        # is then an attribute error on `None`.
        self._session: Any = None
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
        return f"{self.prefix}{id}.json"

    def _client(self) -> Any:
        """A client context manager over the shared session, for one operation."""
        from aiobotocore.config import AioConfig

        return self._session.client(
            "s3",
            endpoint_url=self._pool_config.endpoint_url,
            config=AioConfig(max_pool_connections=self._max_concurrency),
        )

    def _id_from_key(self, key: str) -> str:
        """The record id :meth:`_get_key` made ``key`` from."""
        return key[len(self.prefix) : -len(".json")]

    async def _iter_keys(self, s3: Any, page_size: int | None = None) -> AsyncIterator[str]:
        """Every record's object key, in key order."""
        params: dict[str, Any] = {"Bucket": self._pool_config.bucket, "Prefix": self.prefix}
        if page_size is not None:
            params["MaxKeys"] = page_size
        async for page in s3.get_paginator("list_objects_v2").paginate(**params):
            for obj in page.get("Contents", ()):
                if obj["Key"].endswith(".json"):
                    yield obj["Key"]

    async def _fetch_matching(
        self, s3: Any, query: Query, page_size: int | None = None
    ) -> AsyncIterator[Record]:
        """The records matching ``query``'s filters, in key order.

        Reads up to ``max_pool_connections`` objects ahead of the caller, so
        one that stops early has fetched at most that many it did not use.
        """
        pending: deque[asyncio.Task[Record | None]] = deque()
        try:
            async with aclosing_iter(self._iter_keys(s3, page_size)) as keys:
                async for key in keys:
                    pending.append(asyncio.create_task(self._fetch_and_filter(s3, key, query)))
                    if len(pending) >= self._max_concurrency:
                        record = await pending[0]
                        pending.popleft()
                        if record is not None:
                            yield record
            while pending:
                record = await pending[0]
                pending.popleft()
                if record is not None:
                    yield record
        finally:
            # The client closes when the caller's block does, so no fetch may
            # outlive this one.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _fetch_and_filter(self, s3: Any, key: str, query: Query) -> Record | None:
        """The record at ``key``, or ``None`` when ``query`` does not match it.

        ``None`` also when the object was deleted after it was listed. Any
        other failure is raised.
        """
        record = await self._fetch(s3, key)
        if record is None:
            return None
        record.metadata["id"] = self._id_from_key(key)
        return record if self._matches_filters(record, query.filters) else None

    async def _fetch(self, s3: Any, key: str) -> Record | None:
        """The record at ``key``, or ``None`` when there is no such object."""
        body = await self._fetch_body(s3, key)
        return None if body is None else self._s3_object_to_record(json.loads(body))

    async def _fetch_body(self, s3: Any, key: str) -> bytes | None:
        """The bytes at ``key``, or ``None`` when there is no such object."""
        from botocore.exceptions import ClientError

        try:
            response = await s3.get_object(Bucket=self._pool_config.bucket, Key=key)
        except ClientError as e:
            if is_s3_missing_object(e):
                return None
            raise
        body: bytes = await response["Body"].read()
        return body

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
        obj = self._object_for_write(record_copy, storage_id)

        from botocore.exceptions import ClientError

        # Atomic insert. IfNoneMatch="*" makes the PUT fail closed if the key
        # already exists (412 PreconditionFailed) or if a concurrent conditional
        # write races it (409 ConditionalRequestConflict), so a colliding id
        # cannot silently overwrite an existing record.
        async with self._client() as s3:
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

        async with self._client() as s3:
            stored = await self._fetch(s3, self._get_key(id))

        prepared = self._prepare_record_from_storage(stored, id)
        if prepared is None:
            return None
        # Ensure ID is in metadata
        prepared.metadata["id"] = id
        return prepared

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
            async with self._client() as s3:
                response = await s3.head_object(
                    Bucket=self._pool_config.bucket,
                    Key=key,
                )
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise
        # aioboto3 is untyped, so the response is `Any`; the ETag is a string
        # or absent. Same coercion the memory backend applies to its version.
        etag = response.get("ETag")
        return None if etag is None else str(etag)

    async def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        """Update an existing record in S3.

        When ``expected_version`` is provided the PUT carries an
        ``If-Match: <ETag>`` guard so the compare-and-set is enforced by S3; a
        stale token raises ``ConcurrencyError``. When ``None`` the update is
        unconditional.

        The stored object's ``created_at`` is kept, and its body is read for
        that: one that is not JSON raises ``json.JSONDecodeError``.
        """
        self._check_connection()

        key = self._get_key(id)
        put_kwargs: dict[str, Any] = {
            "Bucket": self._pool_config.bucket,
            "Key": key,
            "ContentType": "application/json",
        }
        if expected_version is not None:
            put_kwargs["IfMatch"] = expected_version

        from botocore.exceptions import ClientError

        try:
            async with self._client() as s3:
                body = await self._fetch_body(s3, key)
                # An update never inserts. A conditional one does not rely on
                # this read: the IfMatch PUT decides, and answers for a key
                # deleted since.
                if body is None:
                    return False
                obj = self._object_for_write(record, id, self._stored_metadata(body))
                await s3.put_object(Body=json.dumps(obj), **put_kwargs)
        except ClientError as e:
            if expected_version is not None:
                if is_s3_missing_object(e):
                    # A conditional update never inserts: an absent id is a
                    # miss, not a conflict, so an IfMatch PUT against a missing
                    # key (S3 -> 404 NoSuchKey) returns False.
                    return False
                if is_s3_conditional_conflict(e):
                    # The guarded PUT lost: either the object vanished mid-op
                    # (update never inserts -> False) or the ETag is stale
                    # (-> raise).
                    current = await self.get_version(id)
                    if current is None:
                        return False
                    raise version_conflict_error(id, expected_version, current) from e
            raise

        return True

    async def delete(self, id: str, *, expected_version: str | None = None) -> bool:
        """Delete a record from S3.

        When ``expected_version`` is provided the ``DeleteObject`` carries an
        ``If-Match: <ETag>`` guard so the compare-and-set is enforced by S3; a
        stale token raises ``ConcurrencyError`` and a missing object returns
        ``False``. When ``None`` the delete is unconditional, byte-identical to
        prior behavior.
        """
        self._check_connection()

        key = self._get_key(id)

        if expected_version is not None:
            from botocore.exceptions import ClientError

            try:
                async with self._client() as s3:
                    await s3.delete_object(
                        Bucket=self._pool_config.bucket,
                        Key=key,
                        IfMatch=expected_version,
                    )
                return True
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey"):
                    # A conditional delete never conflicts on an absent id.
                    return False
                if is_s3_conditional_conflict(e):
                    # The guarded delete lost: the ETag is stale (-> raise) or
                    # the object vanished mid-op (-> False).
                    current = await self.get_version(id)
                    if current is None:
                        return False
                    raise version_conflict_error(id, expected_version, current) from e
                raise

        async with self._client() as s3:
            await s3.delete_object(Bucket=self._pool_config.bucket, Key=key)
        return True

    async def exists(self, id: str) -> bool:
        """Check if a record exists in S3."""
        self._check_connection()

        key = self._get_key(id)

        from botocore.exceptions import ClientError

        try:
            async with self._client() as s3:
                await s3.head_object(Bucket=self._pool_config.bucket, Key=key)
            return True
        except ClientError as e:
            if is_s3_missing_object(e):
                return False
            raise

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

        An object already stored keeps its ``created_at``, and its body is read
        for that: one that is not JSON raises ``json.JSONDecodeError``.
        """
        self._check_connection()

        # Resolve the storage id via the shared helper (honors an explicit id;
        # mints via the overridable _generate_id() hook when the record has none).
        id, record = self._resolve_upsert_id(id_or_record, record)

        if expected_version is not None:
            # A conditional upsert never inserts. Delegate to update()'s ETag
            # If-Match compare-and-set: a True return is the update; a stale
            # ETag raises straight out; a False return means the object is
            # absent, which for a conditional upsert is itself a conflict.
            # Acting on the return (not a separate exists() probe) closes the
            # exists()->update() TOCTOU.
            if await self.update(id, record, expected_version=expected_version):
                return id
            raise version_conflict_error(id, expected_version, None)

        key = self._get_key(id)
        async with self._client() as s3:
            # Read for the created_at it keeps. The PUT is plain, so a write
            # that lands in between is overwritten: last writer wins.
            body = await self._fetch_body(s3, key)
            stored = None if body is None else self._stored_metadata(body)
            obj = self._object_for_write(record, id, stored)
            await s3.put_object(
                Bucket=self._pool_config.bucket,
                Key=key,
                Body=json.dumps(obj),
                ContentType="application/json",
            )

        return id

    async def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching the query.

        A ``ComplexQuery`` goes to the base class's in-memory fallback. The
        narrowed signature this replaces did not keep one out -- the abstract
        base accepts the union -- it only meant the body read ``query.filters``
        on a type that has none.
        """
        if isinstance(query, ComplexQuery):
            return await self._search_with_complex_query(query)

        self._check_connection()

        # S3 doesn't support complex queries, so we need to list and filter.
        # Collect (id, record) tuples; the shared ``_process_search_results``
        # helper applies sorting / offset / limit / projection consistently
        # with every other backend (and correctly handles falsy sort values
        # such as a numeric ``0``, which a hand-rolled ``or ""`` key coerces
        # to a string and crashes on under mixed types).
        results: list[tuple[str, Record]] = []

        async with self._client() as s3:
            async with aclosing_iter(self._fetch_matching(s3, query)) as records:
                async for record in records:
                    results.append((record.metadata["id"], record))

        # Records are freshly deserialized from S3 (no shared aliasing), so
        # ``ensure_record_id`` inside the helper is the only copy needed.
        return self._process_search_results(results, query, deep_copy=False)

    def _matches_filters(self, record: Record, filters: list) -> bool:
        """Check if a record matches all filters.

        Delegates every operator to the canonical ``Filter.matches`` matcher
        (mirroring the sync S3 backend), so the ``id`` field resolves to the
        record's storage key and every operator — including ``BETWEEN`` /
        ``EXISTS`` / ``REGEX`` / ``STARTS_WITH`` — is honored uniformly.
        """
        for filter in filters:
            # The reserved storage-key field routes to the storage key.
            if is_storage_key_field(filter.field):
                field_value = record.id
            else:
                field_value = record.get_value(filter.field)
            if not filter.matches(field_value):
                return False

        return True

    async def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()

        async with self._client() as s3:
            return sum([1 async for _ in self._iter_keys(s3)])

    async def clear(self) -> int:
        """Clear all records from the database."""
        self._check_connection()

        count = 0
        async with self._client() as s3:
            # List and delete all objects
            paginator = s3.get_paginator("list_objects_v2")

            params = {"Bucket": self._pool_config.bucket, "Prefix": self.prefix}

            async for page in paginator.paginate(**params):
                if "Contents" not in page:
                    continue

                # Build delete request
                objects_to_delete = []
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".json"):
                        objects_to_delete.append({"Key": obj["Key"]})
                        count += 1

                # Delete in batch
                if objects_to_delete:
                    await s3.delete_objects(
                        Bucket=self._pool_config.bucket, Delete={"Objects": objects_to_delete}
                    )

        return count

    async def stream_read(
        self, query: Query | None = None, config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Stream records from S3."""
        self._check_connection()
        config = config or StreamConfig()

        async with (
            self._client() as s3,
            aclosing_iter(
                self._fetch_matching(s3, query or Query(), page_size=config.batch_size)
            ) as records,
        ):
            async for record in records:
                # Apply field projection
                if query and query.fields:
                    yield record.project(query.fields)
                else:
                    yield record

    async def stream_write(
        self, records: AsyncIterator[Record], config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into S3.

        INSERT routes through per-record ``create()`` (``insert_batch_func=None``)
        rather than a bulk fast-path: the per-key S3 writes are non-transactional,
        so a partial-success batch followed by the per-record fallback would
        re-write the already-written keys and count them as spurious duplicate
        failures. A per-key conditional ``create`` PUT is the natural granularity
        and fails closed cleanly. UPSERT keeps the bulk ``upsert_batch`` fast-path
        (overwrite is idempotent, so partial success is benign).
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

    async def list_all(self) -> list[str]:
        """List all record IDs in the database."""
        self._check_connection()

        async with self._client() as s3:
            return [self._id_from_key(key) async for key in self._iter_keys(s3)]

    async def _vector_search(
        self,
        query_vector: np.ndarray | list[float],
        *,
        vector_field: str,
        k: int,
        metric: DistanceMetric,
        filter: Query | None,
    ) -> list[VectorSearchResult]:
        """Raw k-NN over every record, in Python.

        WARNING: this downloads all records from S3 to search locally, which
        is inefficient for large datasets. Consider a vector-enabled backend
        such as PostgreSQL or Elasticsearch for production use at scale.

        Future optimization: override this hook to use AWS OpenSearch or a
        similar vector-enabled service when available.
        """
        return await self.python_vector_search_async(
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            filter=filter,
            metric=metric,
        )

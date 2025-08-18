"""Native async S3 backend implementation with aioboto3 and connection pooling."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from dataknobs_config import ConfigurableBase

from ..database import AsyncDatabase
from ..pooling import ConnectionPoolManager
from ..pooling.s3 import S3PoolConfig, create_aioboto3_session, validate_s3_session
from ..query import Operator, Query, SortOrder
from ..records import Record
from ..streaming import StreamConfig, StreamResult, async_process_batch_with_fallback

logger = logging.getLogger(__name__)

# Global pool manager for S3 sessions
_session_manager = ConnectionPoolManager()


class AsyncS3Database(AsyncDatabase, ConfigurableBase):
    """Native async S3 database backend with aioboto3 and session pooling."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize async S3 database."""
        super().__init__(config)

        if not config or "bucket" not in config:
            raise ValueError("S3 backend requires 'bucket' in configuration")

        self._pool_config = S3PoolConfig.from_dict(config)
        self._session = None
        self._connected = False

    @classmethod
    def from_config(cls, config: dict) -> "AsyncS3Database":
        """Create from config dictionary."""
        return cls(config)

    async def connect(self) -> None:
        """Connect to S3 service."""
        if self._connected:
            return

        # Get or create session for current event loop
        self._session = await _session_manager.get_pool(
            self._pool_config,
            create_aioboto3_session,
            lambda session: validate_s3_session(session, self._pool_config)
        )

        self._connected = True

    async def close(self) -> None:
        """Close the S3 connection."""
        if self._connected:
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
        data = {}
        for field_name, field_obj in record.fields.items():
            data[field_name] = field_obj.value

        return {
            "data": data,
            "metadata": record.metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

    def _s3_object_to_record(self, obj: dict[str, Any]) -> Record:
        """Convert an S3 object to a Record."""
        data = obj.get("data", {})
        metadata = obj.get("metadata", {})

        # Add timestamps to metadata
        if "created_at" in obj:
            metadata["created_at"] = obj["created_at"]
        if "updated_at" in obj:
            metadata["updated_at"] = obj["updated_at"]

        return Record(data=data, metadata=metadata)

    async def create(self, record: Record) -> str:
        """Create a new record in S3."""
        self._check_connection()

        # Use record's ID if it has one, otherwise generate a new one
        id = record.id if record.id else str(uuid.uuid4())
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
                # Ensure ID is in metadata
                record.metadata["id"] = id

                return record
        except Exception:
            return None

    async def update(self, id: str, record: Record) -> bool:
        """Update an existing record in S3."""
        self._check_connection()

        # Check if record exists
        if not await self.exists(id):
            return False

        key = self._get_key(id)
        obj = self._record_to_s3_object(record)

        # Preserve ID in metadata
        obj["metadata"]["id"] = id

        async with self._session.client("s3", endpoint_url=self._pool_config.endpoint_url) as s3:
            await s3.put_object(
                Bucket=self._pool_config.bucket,
                Key=key,
                Body=json.dumps(obj),
                ContentType="application/json"
            )

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

    async def upsert(self, id: str, record: Record) -> str:
        """Update or insert a record with a specific ID."""
        self._check_connection()

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

        # S3 doesn't support complex queries, so we need to list and filter
        records = []

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
                        records.append(record)

        # Apply sorting
        if query.sort_specs:
            for sort_spec in reversed(query.sort_specs):
                reverse = sort_spec.order == SortOrder.DESC
                records.sort(
                    key=lambda r: r.get_field(sort_spec.field).value if r.get_field(sort_spec.field) else None,
                    reverse=reverse
                )

        # Apply offset and limit
        if query.offset_value:
            records = records[query.offset_value:]
        if query.limit_value:
            records = records[:query.limit_value]

        # Apply field projection
        if query.fields:
            records = [r.project(query.fields) for r in records]

        return records

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

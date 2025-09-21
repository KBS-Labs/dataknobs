"""S3 backend implementation with proper connection management."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from dataknobs_config import ConfigurableBase

from dataknobs_data.database import SyncDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig, StreamResult, process_batch_with_fallback

from ..vector import VectorOperationsMixin
from ..vector.bulk_embed_mixin import BulkEmbedMixin
from ..vector.python_vector_search import PythonVectorSearchMixin
from .sqlite_mixins import SQLiteVectorSupport
from .vector_config_mixin import VectorConfigMixin

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


class SyncS3Database(  # type: ignore[misc]
    SyncDatabase,
    ConfigurableBase,
    VectorConfigMixin,
    SQLiteVectorSupport,
    PythonVectorSearchMixin,
    BulkEmbedMixin,
    VectorOperationsMixin
):
    """S3-based database backend with proper connection management.
    
    Stores records as JSON objects in S3 with metadata as tags.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize S3 database configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Connection state
        self.s3_client = None
        self._connected = False

        # Cache for performance
        self._index_cache = {}
        self._cache_dirty = True

        # Store configuration for later connection
        self.bucket = self.config.get("bucket")
        if not self.bucket:
            raise ValueError("S3 bucket name is required in configuration")

        # Optional configuration with defaults
        self.prefix = self.config.get("prefix", "records/").rstrip("/") + "/"
        self.region = self.config.get("region", "us-east-1")
        self.endpoint_url = self.config.get("endpoint_url")
        self.max_workers = self.config.get("max_workers", 10)
        self.multipart_threshold = self.config.get("multipart_threshold", 8 * 1024 * 1024)
        self.multipart_chunksize = self.config.get("multipart_chunksize", 8 * 1024 * 1024)
        self.max_retries = self.config.get("max_retries", 3)

        # AWS credentials (will use environment/IAM role if not provided)
        self.aws_access_key_id = self.config.get("access_key_id")
        self.aws_secret_access_key = self.config.get("secret_access_key")
        self.aws_session_token = self.config.get("session_token")

        # Initialize vector support
        self._parse_vector_config(config or {})
        self._init_vector_state()

    @classmethod
    def from_config(cls, config: dict) -> SyncS3Database:
        """Create instance from configuration dictionary."""
        return cls(config)

    def connect(self) -> None:
        """Connect to S3 service."""
        if self._connected:
            return  # Already connected

        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import ClientError

        # Configure boto3 client
        boto_config = BotoConfig(
            region_name=self.region,
            max_pool_connections=self.max_workers,
            retries={'max_attempts': self.max_retries}
        )

        client_kwargs = {
            "config": boto_config,
            "use_ssl": not bool(self.endpoint_url)  # Disable SSL for local testing
        }

        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        if self.aws_access_key_id and self.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = self.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key

        if self.aws_session_token:
            client_kwargs["aws_session_token"] = self.aws_session_token

        # Create S3 client
        self.s3_client = boto3.client("s3", **client_kwargs)
        self.ClientError = ClientError

        # Verify bucket exists or create it
        self._ensure_bucket_exists()

        self._connected = True
        logger.info(f"Connected to S3 with bucket={self.bucket}, prefix={self.prefix}")

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
                # Bucket doesn't exist, create it
                logger.info(f"Creating bucket {self.bucket}")
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
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
        """Create a new record in S3."""
        self._check_connection()

        # Use centralized method to prepare record
        record_copy, storage_id = self._prepare_record_for_storage(record)
        key = self._get_object_key(storage_id)

        # Set metadata
        record_copy.metadata = record_copy.metadata or {}
        record_copy.metadata["id"] = storage_id
        now = datetime.utcnow()
        record_copy.metadata["created_at"] = now.isoformat()
        record_copy.metadata["updated_at"] = now.isoformat()

        # Convert record to JSON
        obj_data = self._record_to_s3_object(record_copy)
        body = json.dumps(obj_data)

        # Store in S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType='application/json'
        )

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

    def update(self, id: str, record: Record) -> bool:
        """Update an existing record in S3."""
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
        record.metadata["created_at"] = existing_metadata.get("created_at", datetime.utcnow().isoformat())
        record.metadata["updated_at"] = datetime.utcnow().isoformat()

        # Update the object
        obj_data = self._record_to_s3_object(record)
        body = json.dumps(obj_data)

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType='application/json'
        )

        # Invalidate cache
        self._cache_dirty = True

        logger.debug(f"Updated record {id} at {key}")
        return True

    def delete(self, id: str) -> bool:
        """Delete a record from S3."""
        self._check_connection()

        key = self._get_object_key(id)

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

        # List all objects with the prefix
        records = []
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
                        records.append(record)

        # Apply sorting if specified
        if query.sort_specs:
            for sort_spec in reversed(query.sort_specs):
                reverse = sort_spec.order.value == "desc"
                # Special handling for 'id' field
                if sort_spec.field == 'id':
                    records.sort(
                        key=lambda r: r.id or "",
                        reverse=reverse
                    )
                else:
                    records.sort(
                        key=lambda r: r.get_value(sort_spec.field, ""),
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

    def _fetch_and_filter(self, key: str, query: Query) -> Record | None:
        """Fetch an object and apply query filters."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            body = response['Body'].read()
            obj_data = json.loads(body)
            record = self._s3_object_to_record(obj_data)

            # Apply filters
            for filter in query.filters:
                # Special handling for 'id' field
                if filter.field == 'id':
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
        """Stream records into S3."""
        self._check_connection()
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False

        batch = []
        for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch with graceful fallback
                # Use lambda wrapper for _write_batch
                continue_processing = process_batch_with_fallback(
                    batch,
                    lambda b: self._write_batch(b),
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
            process_batch_with_fallback(
                batch,
                lambda b: self._write_batch(b),
                self.create,
                result,
                config
            )

        result.duration = time.time() - start_time
        return result

    def _write_batch(self, records: list[Record]) -> list[str]:
        """Write a batch of records to S3.
        
        Returns:
            List of created record IDs
        """
        ids = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for record in records:
                record_id = str(uuid4())
                ids.append(record_id)
                future = executor.submit(self._write_single, record_id, record)
                futures.append(future)

            # Wait for all writes to complete
            for future in as_completed(futures):
                future.result()  # This will raise if there was an error

        return ids

    def _write_single(self, record_id: str, record: Record) -> None:
        """Write a single record to S3."""
        # Set metadata
        record.metadata = record.metadata or {}
        record.metadata["id"] = record_id
        now = datetime.utcnow()
        record.metadata["created_at"] = now.isoformat()
        record.metadata["updated_at"] = now.isoformat()

        key = self._get_object_key(record_id)
        obj_data = self._record_to_s3_object(record)
        body = json.dumps(obj_data)

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType='application/json'
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

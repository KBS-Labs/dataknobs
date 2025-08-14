"""S3 backend implementation for dataknobs-data package."""

import json
import logging
from typing import Any, Dict, List, Optional, Iterator
from uuid import uuid4
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dataknobs_config import ConfigurableBase
from dataknobs_data.records import Record
from dataknobs_data.query import Query
from dataknobs_data.database import SyncDatabase as Database

logger = logging.getLogger(__name__)


class S3Database(Database, ConfigurableBase):
    """S3-based database backend.
    
    Stores records as JSON objects in S3 with metadata as tags.
    
    Configuration Options:
        bucket (str): S3 bucket name (required)
        prefix (str): Object key prefix for organization (default: "records/")
        region (str): AWS region (default: "us-east-1")
        endpoint_url (str): Custom endpoint URL (for LocalStack/MinIO)
        access_key_id (str): AWS access key ID (from env if not provided)
        secret_access_key (str): AWS secret access key (from env if not provided)
        session_token (str): AWS session token (optional)
        max_workers (int): Max threads for parallel operations (default: 10)
        multipart_threshold (int): Size threshold for multipart upload in bytes (default: 8MB)
        multipart_chunksize (int): Chunk size for multipart upload (default: 8MB)
        max_retries (int): Maximum retry attempts (default: 3)
        
    Example Configuration:
        databases:
          - name: s3_storage
            class: dataknobs_data.backends.s3.S3Database
            bucket: my-data-bucket
            prefix: records/prod/
            region: us-west-2
            endpoint_url: ${LOCALSTACK_ENDPOINT}  # For testing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize S3 database backend.
        
        Args:
            config: Configuration dictionary
        """
        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import ClientError
        
        self.config = config or {}
        
        # Required configuration
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
        aws_access_key_id = self.config.get("access_key_id")
        aws_secret_access_key = self.config.get("secret_access_key")
        aws_session_token = self.config.get("session_token")
        
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
        
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
            
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token
        
        # Create S3 client
        self.s3_client = boto3.client("s3", **client_kwargs)
        self.ClientError = ClientError
        
        # Verify bucket exists or create it
        self._ensure_bucket_exists()
        
        # Cache for performance
        self._index_cache = {}
        self._cache_dirty = True
        
        logger.info(f"Initialized S3Database with bucket={self.bucket}, prefix={self.prefix}")
    
    @classmethod
    def from_config(cls, config: dict) -> "S3Database":
        """Create instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured S3Database instance
        """
        return cls(config)
    
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
    
    def _generate_id(self) -> str:
        """Generate a unique record ID."""
        return str(uuid4())
    
    def _get_object_key(self, record_id: str) -> str:
        """Get S3 object key for a record ID."""
        return f"{self.prefix}{record_id}.json"
    
    def _serialize_record(self, record: Record) -> bytes:
        """Serialize a record to JSON bytes."""
        # Convert fields to serializable format
        fields_data = {}
        for name, field in record.fields.items():
            fields_data[name] = {
                "value": field.value,
                "type": field.type.value if field.type else None,
                "metadata": field.metadata
            }
        
        data = {
            "fields": fields_data,
            "metadata": record.metadata
        }
        return json.dumps(data, indent=2, default=str).encode('utf-8')
    
    def _deserialize_record(self, data: bytes) -> Record:
        """Deserialize JSON bytes to a Record."""
        record_data = json.loads(data.decode('utf-8'))
        
        # Convert fields back to simple values for Record constructor
        fields = {}
        for name, field_data in record_data.get("fields", {}).items():
            if isinstance(field_data, dict) and "value" in field_data:
                fields[name] = field_data["value"]
            else:
                fields[name] = field_data
        
        return Record(
            data=fields,
            metadata=record_data.get("metadata", {})
        )
    
    def _record_to_tags(self, record: Record) -> Dict[str, str]:
        """Convert record metadata to S3 tags (max 10 tags, 128 chars each)."""
        tags = {}
        
        # Add metadata as tags (S3 limit: 10 tags total)
        for i, (key, value) in enumerate(record.metadata.items()):
            if i >= 10:
                break
            # S3 tag keys/values have character limits
            tag_key = str(key)[:128]
            tag_value = str(value)[:128] if value is not None else ""
            tags[tag_key] = tag_value
        
        return tags
    
    def create(self, record: Record) -> str:
        """Create a new record in S3.
        
        Args:
            record: Record to create
            
        Returns:
            The ID of the created record
        """
        # Generate ID and set in metadata
        record_id = record.metadata.get("id", self._generate_id())
        record.metadata["id"] = record_id
        
        # Set timestamps in metadata
        now = datetime.utcnow()
        record.metadata["created_at"] = now.isoformat()
        record.metadata["updated_at"] = now.isoformat()
        
        # Serialize record
        data = self._serialize_record(record)
        
        # Prepare S3 tags
        tags = self._record_to_tags(record)
        tagging = "&".join([f"{k}={v}" for k, v in tags.items()])
        
        # Upload to S3
        key = self._get_object_key(record_id)
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType="application/json",
                Tagging=tagging,
                Metadata={
                    "record_id": record_id,
                    "dataknobs_type": "record"
                }
            )
            
            # Invalidate cache
            self._cache_dirty = True
            
            logger.debug(f"Created record {record_id} in S3")
            return record_id
            
        except self.ClientError as e:
            logger.error(f"Failed to create record in S3: {e}")
            raise
    
    def read(self, record_id: str) -> Optional[Record]:
        """Read a record from S3.
        
        Args:
            record_id: ID of the record to read
            
        Returns:
            The record if found, None otherwise
        """
        key = self._get_object_key(record_id)
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=key
            )
            
            data = response['Body'].read()
            record = self._deserialize_record(data)
            
            logger.debug(f"Read record {record_id} from S3")
            return record
            
        except self.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.debug(f"Record {record_id} not found in S3")
                return None
            else:
                logger.error(f"Failed to read record from S3: {e}")
                raise
    
    def update(self, record_id: str, record: Record) -> bool:
        """Update an existing record in S3.
        
        Args:
            record_id: ID of the record to update
            record: Updated record data
            
        Returns:
            True if successful, False if record not found
        """
        # Check if record exists
        if not self.exists(record_id):
            return False
        
        # Preserve original creation time
        existing = self.read(record_id)
        if existing and "created_at" in existing.metadata:
            record.metadata["created_at"] = existing.metadata["created_at"]
        
        # Update timestamp and ID in metadata
        record.metadata["updated_at"] = datetime.utcnow().isoformat()
        record.metadata["id"] = record_id
        
        # Serialize and upload
        data = self._serialize_record(record)
        tags = self._record_to_tags(record)
        tagging = "&".join([f"{k}={v}" for k, v in tags.items()])
        
        key = self._get_object_key(record_id)
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType="application/json",
                Tagging=tagging,
                Metadata={
                    "record_id": record_id,
                    "dataknobs_type": "record"
                }
            )
            
            # Invalidate cache
            self._cache_dirty = True
            
            logger.debug(f"Updated record {record_id} in S3")
            return True
            
        except self.ClientError as e:
            logger.error(f"Failed to update record in S3: {e}")
            raise
    
    def delete(self, record_id: str) -> bool:
        """Delete a record from S3.
        
        Args:
            record_id: ID of the record to delete
            
        Returns:
            True if successful, False if record not found
        """
        key = self._get_object_key(record_id)
        
        try:
            # Check if exists first
            if not self.exists(record_id):
                return False
            
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=key
            )
            
            # Invalidate cache
            self._cache_dirty = True
            
            logger.debug(f"Deleted record {record_id} from S3")
            return True
            
        except self.ClientError as e:
            logger.error(f"Failed to delete record from S3: {e}")
            raise
    
    def exists(self, record_id: str) -> bool:
        """Check if a record exists in S3.
        
        Args:
            record_id: ID of the record to check
            
        Returns:
            True if the record exists, False otherwise
        """
        key = self._get_object_key(record_id)
        
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Failed to check record existence in S3: {e}")
                raise
    
    def list_all(self) -> List[str]:
        """List all record IDs in the database.
        
        Returns:
            List of all record IDs
        """
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
    
    def search(self, query: Query) -> List[Record]:
        """Search for records matching the query.
        
        Note: S3 doesn't support native querying, so this loads and filters all records.
        For large datasets, consider using a proper database or search service.
        
        Args:
            query: Query object with filters and options
            
        Returns:
            List of matching records
        """
        # Build index if cache is dirty
        if self._cache_dirty:
            self._rebuild_index()
        
        # Get all record IDs
        all_ids = list(self._index_cache.keys())
        
        # Load records in parallel
        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(self.read, rid): rid for rid in all_ids}
            
            for future in as_completed(future_to_id):
                record = future.result()
                if record:
                    records.append(record)
        
        # Apply query filters
        filtered = self._apply_query(records, query)
        
        logger.debug(f"Search returned {len(filtered)} records from S3")
        return filtered
    
    def _rebuild_index(self):
        """Rebuild the index cache from S3 listings."""
        self._index_cache = {}
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.prefix
        )
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.startswith(self.prefix) and key.endswith('.json'):
                        record_id = key[len(self.prefix):-5]
                        self._index_cache[record_id] = {
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag']
                        }
        
        self._cache_dirty = False
        logger.debug(f"Rebuilt index with {len(self._index_cache)} records")
    
    def _apply_query(self, records: List[Record], query: Query) -> List[Record]:
        """Apply query filters, sorting, and pagination to records."""
        # Apply filters
        filtered = []
        for record in records:
            if self._matches_filters(record, query.filters):
                filtered.append(record)
        
        # Apply sorting
        if query.sort_specs:
            for sort_spec in reversed(query.sort_specs):
                reverse = (sort_spec.order.value == "desc")
                filtered.sort(
                    key=lambda r: r.get_value(sort_spec.field) if r.get_value(sort_spec.field) is not None else "",
                    reverse=reverse
                )
        
        # Apply pagination
        start = query.offset_value or 0
        end = start + query.limit_value if query.limit_value else None
        filtered = filtered[start:end]
        
        # Apply projection (field selection)
        if query.fields:
            for record in filtered:
                # Keep only projected fields
                projected_fields = {
                    k: v for k, v in record.fields.items()
                    if k in query.fields
                }
                record.fields = projected_fields
        
        return filtered
    
    def _matches_filters(self, record: Record, filters: List) -> bool:
        """Check if a record matches all filters."""
        from dataknobs_data.query import Operator
        
        for filter_obj in filters:
            field_value = record.get_value(filter_obj.field)
            op = filter_obj.operator
            value = filter_obj.value
            
            if op == Operator.EQ:
                if field_value != value:
                    return False
            elif op == Operator.NEQ:
                if field_value == value:
                    return False
            elif op == Operator.GT:
                if field_value is None or field_value <= value:
                    return False
            elif op == Operator.GTE:
                if field_value is None or field_value < value:
                    return False
            elif op == Operator.LT:
                if field_value is None or field_value >= value:
                    return False
            elif op == Operator.LTE:
                if field_value is None or field_value > value:
                    return False
            elif op == Operator.IN:
                if field_value not in value:
                    return False
            elif op == Operator.NOT_IN:
                if field_value in value:
                    return False
            elif op == Operator.LIKE:
                if field_value is None or not self._matches_pattern(str(field_value), value):
                    return False
        
        return True
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a SQL LIKE pattern."""
        import re
        # Convert SQL LIKE pattern to regex
        regex_pattern = pattern.replace("%", ".*").replace("_", ".")
        return bool(re.match(f"^{regex_pattern}$", text, re.IGNORECASE))
    
    def batch_create(self, records: List[Record]) -> List[str]:
        """Create multiple records in parallel.
        
        Args:
            records: List of records to create
            
        Returns:
            List of created record IDs
        """
        record_ids = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_record = {executor.submit(self.create, record): record for record in records}
            
            for future in as_completed(future_to_record):
                try:
                    record_id = future.result()
                    record_ids.append(record_id)
                except Exception as e:
                    logger.error(f"Failed to create record in batch: {e}")
                    # Continue with other records
        
        logger.info(f"Batch created {len(record_ids)} records in S3")
        return record_ids
    
    def batch_read(self, record_ids: List[str]) -> List[Optional[Record]]:
        """Read multiple records in parallel.
        
        Args:
            record_ids: List of record IDs to read
            
        Returns:
            List of records (None for not found)
        """
        records = [None] * len(record_ids)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.read, rid): i 
                for i, rid in enumerate(record_ids)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    records[index] = future.result()
                except Exception as e:
                    logger.error(f"Failed to read record in batch: {e}")
                    records[index] = None
        
        return records
    
    def batch_delete(self, record_ids: List[str]) -> List[bool]:
        """Delete multiple records in parallel.
        
        Args:
            record_ids: List of record IDs to delete
            
        Returns:
            List of success flags
        """
        results = [False] * len(record_ids)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.delete, rid): i 
                for i, rid in enumerate(record_ids)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Failed to delete record in batch: {e}")
                    results[index] = False
        
        logger.info(f"Batch deleted {sum(results)} records from S3")
        return results
    
    def clear(self):
        """Clear all records from the database.
        
        Warning: This deletes all objects with the configured prefix!
        """
        # List all objects with prefix
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.prefix
        )
        
        # Collect all keys to delete
        keys_to_delete = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    keys_to_delete.append({'Key': obj['Key']})
        
        # Delete in batches (S3 allows max 1000 per request)
        if keys_to_delete:
            for i in range(0, len(keys_to_delete), 1000):
                batch = keys_to_delete[i:i+1000]
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': batch}
                )
            
            logger.info(f"Cleared {len(keys_to_delete)} records from S3")
        
        # Clear cache
        self._index_cache = {}
        self._cache_dirty = False
    
    def count(self) -> int:
        """Count the total number of records.
        
        Returns:
            Total number of records
        """
        if self._cache_dirty:
            self._rebuild_index()
        
        return len(self._index_cache)
    
    def _count_all(self) -> int:
        """Count all records in the database.
        
        Returns:
            Total number of records
        """
        if self._cache_dirty:
            self._rebuild_index()
        
        return len(self._index_cache)
    
    def close(self):
        """Close the database connection."""
        # S3 client doesn't need explicit closing, but clear cache
        self._index_cache = {}
        logger.debug("Closed S3Database connection")
"""Synchronization tools for keeping vectors up to date with text changes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import numpy as np

from ..fields import FieldType, VectorField
from ..records import Record
from .types import VectorMetadata

if TYPE_CHECKING:
    from ..database import Database

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Configuration for vector synchronization."""
    
    auto_embed_on_create: bool = True
    auto_update_on_text_change: bool = True
    batch_size: int = 100
    track_model_version: bool = True
    embedding_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.embedding_timeout <= 0:
            raise ValueError(f"Embedding timeout must be positive, got {self.embedding_timeout}")
        if self.max_retries < 0:
            raise ValueError(f"Max retries cannot be negative, got {self.max_retries}")


@dataclass
class SyncStatus:
    """Status of a synchronization operation."""
    
    total_records: int = 0
    processed_records: int = 0
    updated_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the sync operation."""
        if self.processed_records == 0:
            return 0.0
        return (self.processed_records - self.failed_records) / self.processed_records
    
    @property
    def duration(self) -> float | None:
        """Calculate the duration of the sync operation in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "updated_records": self.updated_records,
            "failed_records": self.failed_records,
            "skipped_records": self.skipped_records,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "errors": self.errors,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class VectorTextSynchronizer:
    """Synchronizes vector embeddings with their source text fields."""
    
    def __init__(
        self,
        database: Database,
        embedding_fn: Callable[[str], np.ndarray] | Callable[[str], Coroutine[Any, Any, np.ndarray]],
        config: SyncConfig | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        """Initialize the synchronizer.
        
        Args:
            database: The database to synchronize
            embedding_fn: Function to generate embeddings from text
            config: Synchronization configuration
            model_name: Name of the embedding model
            model_version: Version of the embedding model
        """
        self.database = database
        self.embedding_fn = embedding_fn
        self.config = config or SyncConfig()
        self.config.validate()
        self.model_name = model_name
        self.model_version = model_version
        
        # Track vector fields and their source fields
        self._vector_fields: dict[str, dict[str, Any]] = {}
        self._source_fields: dict[str, list[str]] = defaultdict(list)
        self._initialize_field_mappings()
    
    def _initialize_field_mappings(self) -> None:
        """Initialize mappings between vector fields and source fields."""
        # Use schema if available
        for field_name, field_schema in self.database.schema.fields.items():
            if field_schema.is_vector_field():
                self._vector_fields[field_name] = {
                    "dimensions": field_schema.get_dimensions() or 384,
                    "source_field": field_schema.get_source_field(),
                }
                source = field_schema.get_source_field()
                if source:
                    self._source_fields[source].append(field_name)
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute a hash of the content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _has_current_vector(self, record: Record, vector_field: str) -> bool:
        """Check if a record has a current vector for the given field.
        
        Args:
            record: The record to check
            vector_field: Name of the vector field
            
        Returns:
            True if the vector is current, False otherwise
        """
        vector_data = record.get_value(vector_field)
        if not vector_data or not isinstance(vector_data, (list, np.ndarray)):
            return False
        
        # Check if model version tracking is enabled
        if self.config.track_model_version and self.model_version:
            metadata_field = f"{vector_field}_metadata"
            metadata = record.get_value(metadata_field)
            if not metadata or not isinstance(metadata, dict):
                # No metadata when version tracking is enabled means not current
                return False
            stored_version = metadata.get("model_version")
            if stored_version != self.model_version:
                return False
        
        # Check content hash if source field exists
        field_info = self._vector_fields.get(vector_field)
        if field_info and field_info.get("source_field"):
            source_content = record.get_value(field_info["source_field"], "")
            if source_content:
                hash_field = f"{vector_field}_content_hash"
                stored_hash = record.get_value(hash_field)
                current_hash = self._compute_content_hash(str(source_content))
                if stored_hash != current_hash:
                    return False
        
        return True
    
    def _needs_update(self, record: Record, vector_field: str) -> bool:
        """Check if a vector field needs to be updated.
        
        Args:
            record: The record to check
            vector_field: Name of the vector field
            
        Returns:
            True if the vector needs updating, False otherwise
        """
        return not self._has_current_vector(record, vector_field)
    
    async def _embed_text(self, text: str) -> np.ndarray | None:
        """Generate embedding for text with error handling.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not text:
            return None
        
        for attempt in range(self.config.max_retries):
            try:
                if asyncio.iscoroutinefunction(self.embedding_fn):
                    result = await asyncio.wait_for(
                        self.embedding_fn(text),
                        timeout=self.config.embedding_timeout
                    )
                else:
                    result = await asyncio.to_thread(self.embedding_fn, text)
                
                if isinstance(result, np.ndarray):
                    return result
                elif isinstance(result, list):
                    return np.array(result)
                else:
                    logger.error(f"Embedding function returned unexpected type: {type(result)}")
                    return None
                    
            except asyncio.TimeoutError:
                logger.warning(f"Embedding timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
            except Exception as e:
                logger.error(f"Embedding error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
        
        return None
    
    async def sync_record(
        self,
        record: Record,
        force: bool = False
    ) -> tuple[bool, list[str]]:
        """Synchronize vectors for a single record.
        
        Args:
            record: The record to synchronize
            force: Force update even if vectors appear current
            
        Returns:
            Tuple of (success, list of updated fields)
        """
        updated_fields = []
        success = True
        
        for vector_field, field_info in self._vector_fields.items():
            if not force and not self._needs_update(record, vector_field):
                continue
            
            # Get source text
            source_text = None
            if field_info.get("source_field"):
                source_text = record.get_value(field_info["source_field"])
            
            if not source_text:
                continue
            
            # Generate embedding
            embedding = await self._embed_text(str(source_text))
            if embedding is None:
                success = False
                continue
            
            # Update record
            record.set_value(vector_field, embedding.tolist())
            
            # Update metadata
            if self.config.track_model_version:
                metadata = VectorMetadata(
                    dimensions=len(embedding),
                    source_field=field_info.get("source_field"),
                    model_name=self.model_name,
                    model_version=self.model_version,
                    updated_at=datetime.utcnow().isoformat(),
                )
                record.set_value(f"{vector_field}_metadata", metadata.to_dict())
            
            # Store content hash
            if field_info.get("source_field"):
                hash_field = f"{vector_field}_content_hash"
                record.set_value(hash_field, self._compute_content_hash(str(source_text)))
            
            updated_fields.append(vector_field)
        
        return success, updated_fields
    
    async def bulk_sync(
        self,
        records: list[Record] | None = None,
        force: bool = False,
        progress_callback: Callable[[SyncStatus], None] | None = None,
    ) -> SyncStatus:
        """Synchronize vectors for multiple records in batches.
        
        Args:
            records: Records to sync (None for all records in database)
            force: Force update even if vectors appear current
            progress_callback: Callback for progress updates
            
        Returns:
            Synchronization status
        """
        status = SyncStatus(start_time=datetime.utcnow())
        
        try:
            # Get records if not provided
            if records is None:
                records = await self.database.all()
            
            status.total_records = len(records)
            
            # Process in batches
            for i in range(0, len(records), self.config.batch_size):
                batch = records[i:i + self.config.batch_size]
                
                for record in batch:
                    try:
                        success, updated_fields = await self.sync_record(record, force)
                        status.processed_records += 1
                        
                        if updated_fields:
                            # Update record in database
                            await self.database.update(record.id, record)
                            status.updated_records += 1
                        elif success:
                            status.skipped_records += 1
                        else:
                            status.failed_records += 1
                            
                    except Exception as e:
                        status.failed_records += 1
                        status.errors.append({
                            "record_id": record.id,
                            "error": str(e),
                        })
                        logger.error(f"Failed to sync record {record.id}: {e}")
                
                # Call progress callback
                if progress_callback:
                    progress_callback(status)
            
        finally:
            status.end_time = datetime.utcnow()
        
        logger.info(
            f"Sync completed: {status.updated_records} updated, "
            f"{status.skipped_records} skipped, {status.failed_records} failed"
        )
        
        return status
    
    async def sync_on_update(
        self,
        record_id: str,
        old_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> bool:
        """Handle record updates and sync vectors if needed.
        
        Args:
            record_id: ID of the updated record
            old_data: Previous data
            new_data: New data
            
        Returns:
            True if sync was performed, False otherwise
        """
        if not self.config.auto_update_on_text_change:
            return False
        
        # Check if any source fields changed
        fields_to_update = set()
        for source_field, vector_fields in self._source_fields.items():
            old_value = old_data.get(source_field)
            new_value = new_data.get(source_field)
            
            if old_value != new_value:
                fields_to_update.update(vector_fields)
        
        if not fields_to_update:
            return False
        
        # Create record and sync
        record = Record(id=record_id, data=new_data)
        success, updated_fields = await self.sync_record(record, force=True)
        
        if updated_fields:
            # Update only the vector fields
            update_data = {
                field: record.get_value(field)
                for field in updated_fields
                if record.get_value(field) is not None
            }
            
            # Include metadata fields
            for field in updated_fields:
                metadata_field = f"{field}_metadata"
                metadata_value = record.get_value(metadata_field)
                if metadata_value is not None:
                    update_data[metadata_field] = metadata_value
                
                hash_field = f"{field}_content_hash"
                hash_value = record.get_value(hash_field)
                if hash_value is not None:
                    update_data[hash_field] = hash_value
            
            # Get the existing record and update it
            existing_record = await self.database.read(record_id)
            if existing_record:
                for key, value in update_data.items():
                    existing_record.set_value(key, value)
                await self.database.update(record_id, existing_record)
            return True
        
        return False
    
    async def sync_on_create(self, record: Record) -> bool:
        """Handle record creation and sync vectors if needed.
        
        Args:
            record: The newly created record
            
        Returns:
            True if sync was performed, False otherwise
        """
        if not self.config.auto_embed_on_create:
            return False
        
        success, updated_fields = await self.sync_record(record)
        
        if updated_fields:
            # Update the record with vector data
            await self.database.update(record.id, record)
            return True
        
        return False

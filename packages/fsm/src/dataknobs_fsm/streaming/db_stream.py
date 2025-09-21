"""Database streaming implementation for FSM."""

import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Union

from dataknobs_data.database import AsyncDatabase, SyncDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record

from dataknobs_fsm.streaming.core import (
    IStreamSink,
    IStreamSource,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class DatabaseStreamSource(IStreamSource):
    """Database-based stream source with cursor iteration.
    
    This source supports streaming records from a database using
    efficient cursor-based iteration with configurable batch fetching.
    """
    
    def __init__(
        self,
        database: Union[SyncDatabase, AsyncDatabase],
        query: Query | None = None,
        batch_size: int = 1000,
        cursor_field: str | None = None,
        start_cursor: Any | None = None
    ):
        """Initialize database stream source.
        
        Args:
            database: Database instance to stream from.
            query: Query to filter records (None for all).
            batch_size: Number of records per batch.
            cursor_field: Field to use for cursor pagination.
            start_cursor: Starting cursor value.
        """
        self.database = database
        self.query = query or Query()
        self.batch_size = batch_size
        self.cursor_field = cursor_field or 'id'
        self.current_cursor = start_cursor
        
        self._chunk_count = 0
        self._record_count = 0
        self._exhausted = False
        
        # Get total count if possible
        try:
            self._total_records = database.count(self.query)
        except Exception:
            self._total_records = None
    
    def read_chunk(self) -> StreamChunk | None:
        """Read next chunk of records from database.
        
        Returns:
            StreamChunk with records or None if exhausted.
        """
        if self._exhausted:
            return None
        
        try:
            # Build query with cursor
            batch_query = self._build_batch_query()
            
            # Fetch batch of records
            records = self.database.search(batch_query)
            
            if not records:
                self._exhausted = True
                return None
            
            # Update cursor for next batch
            if records and self.cursor_field:
                last_record = records[-1]  # type: ignore
                if isinstance(last_record, Record):
                    # Use Record's API to get field value
                    if self.cursor_field == 'id':
                        self.current_cursor = last_record.id
                    elif last_record.has_field(self.cursor_field):
                        self.current_cursor = last_record.get_value(self.cursor_field)
                elif isinstance(last_record, dict) and self.cursor_field in last_record:
                    self.current_cursor = last_record[self.cursor_field]
                elif hasattr(last_record, self.cursor_field):
                    self.current_cursor = getattr(last_record, self.cursor_field)
            
            # Calculate progress
            progress = 0.0
            if self._total_records and self._total_records > 0:  # type: ignore
                progress = min(1.0, (self._record_count + len(records)) / self._total_records)  # type: ignore
            
            # Check if this is the last chunk
            is_last = len(records) < self.batch_size  # type: ignore
            if is_last:
                self._exhausted = True
            
            # Convert records to serializable format
            chunk_data = []
            for record in records:
                if isinstance(record, Record):
                    # Use Record's built-in serialization
                    chunk_data.append(record.to_dict(include_metadata=True))
                elif hasattr(record, 'to_dict'):
                    chunk_data.append(record.to_dict())
                elif hasattr(record, '__dict__'):
                    chunk_data.append(record.__dict__)
                else:
                    chunk_data.append(record)
            
            # Create chunk
            chunk = StreamChunk(
                data=chunk_data,
                sequence_number=self._chunk_count,
                metadata={
                    'database_type': type(self.database).__name__,
                    'query': str(self.query),
                    'batch_size': len(chunk_data),
                    'progress': progress,
                    'cursor_field': self.cursor_field,
                    'cursor_value': self.current_cursor
                },
                is_last=is_last
            )
            
            self._chunk_count += 1
            self._record_count += len(records)  # type: ignore
            
            return chunk
            
        except Exception as e:
            # Return error chunk
            self._exhausted = True
            return StreamChunk(
                data=[],
                sequence_number=self._chunk_count,
                metadata={'error': str(e)},
                is_last=True
            )
    
    def _build_batch_query(self) -> Query:
        """Build query for next batch with cursor.
        
        Returns:
            Query for next batch.
        """
        batch_query = Query()
        
        # Copy original query conditions if provided
        if self.query and hasattr(self.query, 'filters') and self.query.filters:
            batch_query.filters = self.query.filters.copy()
        
        # Add cursor condition if we have a cursor value
        if self.current_cursor is not None and self.cursor_field:
            # Use Query API to add filter for pagination
            from dataknobs_data.query import Operator
            batch_query = batch_query.filter(self.cursor_field, Operator.GT, self.current_cursor)
        
        # Set limit - this is critical for batching
        batch_query = batch_query.limit(self.batch_size)
        
        # Add ordering by cursor field for consistent pagination
        if self.cursor_field:
            batch_query = batch_query.sort_by(self.cursor_field, "asc")
        
        return batch_query
    
    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over all chunks."""
        while True:
            chunk = self.read_chunk()
            if chunk is None:
                break
            yield chunk
    
    def close(self) -> None:
        """Close the stream source."""
        # Database connections are managed separately
        pass


class DatabaseStreamSink(IStreamSink):
    """Database-based stream sink with batch operations.
    
    This sink supports writing data chunks to a database using
    efficient batch inserts with transaction support.
    """
    
    def __init__(
        self,
        database: Union[SyncDatabase, AsyncDatabase],
        table_name: str | None = None,
        batch_size: int = 1000,
        upsert: bool = False,
        transaction_batch: int = 10000,
        on_conflict_update: List[str] | None = None
    ):
        """Initialize database stream sink.
        
        Args:
            database: Database instance to write to.
            table_name: Target table name (optional).
            batch_size: Records per batch insert.
            upsert: Use upsert instead of insert.
            transaction_batch: Records per transaction.
            on_conflict_update: Fields to update on conflict.
        """
        self.database = database
        self.table_name = table_name
        self.batch_size = batch_size
        self.upsert = upsert
        self.transaction_batch = transaction_batch
        self.on_conflict_update = on_conflict_update or []
        
        self._buffer: List[Dict[str, Any]] = []
        self._chunk_count = 0
        self._record_count = 0
        self._transaction_count = 0
        self._current_transaction_size = 0
    
    def write_chunk(self, chunk: StreamChunk) -> bool:
        """Write chunk to database.
        
        Args:
            chunk: Chunk containing records to write.
            
        Returns:
            True if successful.
        """
        if not chunk.data:
            return True
        
        try:
            # Add to buffer
            if isinstance(chunk.data, list):
                self._buffer.extend(chunk.data)
            else:
                self._buffer.append(chunk.data)
            
            # Process buffer in batches
            while len(self._buffer) >= self.batch_size:
                batch = self._buffer[:self.batch_size]
                self._buffer = self._buffer[self.batch_size:]
                
                success = self._write_batch(batch)
                if not success:
                    return False
                
                self._current_transaction_size += len(batch)
                
                # Commit transaction if batch is large enough
                if self._current_transaction_size >= self.transaction_batch:
                    self._commit_transaction()
            
            # If this is the last chunk, flush buffer
            if chunk.is_last:
                self.flush()
            
            self._chunk_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error writing chunk to database: {e}")
            return False
    
    def _write_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Write a batch of records to database.
        
        Args:
            batch: Records to write.
            
        Returns:
            True if successful.
        """
        try:
            for record_data in batch:
                # Convert dict to Record if needed
                if isinstance(record_data, dict):
                    # Extract id and create proper Record
                    record_id = record_data.pop('id', None) or record_data.pop('_id', None)
                    if record_id:
                        record = Record(id=record_id, data=record_data)
                    else:
                        record = Record(data=record_data)
                else:
                    record = record_data  # type: ignore[unreachable]
                
                # Perform database operation
                if self.upsert:
                    # Use update if record exists, create otherwise
                    record_id = record.id if hasattr(record, 'id') else None
                    if record_id and self.database.read(record_id):
                        self.database.update(record_id, record)
                    else:
                        self.database.create(record)
                else:
                    # Simple insert
                    self.database.create(record)
                
                self._record_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error in batch write: {e}")
            return False
    
    def _commit_transaction(self) -> None:
        """Commit current transaction if supported."""
        try:
            # Check if database supports transactions
            if hasattr(self.database, 'commit'):
                self.database.commit()
            
            self._transaction_count += 1
            self._current_transaction_size = 0
            
        except Exception:
            # Not all backends support transactions
            pass
    
    def flush(self) -> None:
        """Flush any buffered records."""
        if self._buffer:
            # Write remaining records in buffer
            success = self._write_batch(self._buffer)
            if success:
                self._buffer = []
        
        # Commit any pending transaction
        if self._current_transaction_size > 0:
            self._commit_transaction()
    
    def close(self) -> None:
        """Close the sink and ensure all data is written."""
        self.flush()
        # Database connection is managed separately


class DatabaseBulkLoader:
    """Utility for efficient bulk loading into databases.
    
    This class provides optimized bulk loading strategies
    for different database backends.
    """
    
    def __init__(
        self,
        database: Union[SyncDatabase, AsyncDatabase],
        table_name: str | None = None
    ):
        """Initialize bulk loader.
        
        Args:
            database: Target database.
            table_name: Target table name.
        """
        self.database = database
        self.table_name = table_name
        self._stats = {
            'records_loaded': 0,
            'batches_processed': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def load_from_source(
        self,
        source: IStreamSource,
        batch_size: int = 1000,
        progress_callback: Union[Callable, None] = None
    ) -> Dict[str, Any]:
        """Load data from stream source into database.
        
        Args:
            source: Stream source to read from.
            batch_size: Batch size for inserts.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Loading statistics.
        """
        self._stats['start_time'] = time.time()
        
        sink = DatabaseStreamSink(
            self.database,
            table_name=self.table_name,
            batch_size=batch_size
        )
        
        try:
            for chunk in source:
                success = sink.write_chunk(chunk)
                
                if not success:
                    self._stats['errors'] += 1  # type: ignore
                
                self._stats['batches_processed'] += 1  # type: ignore
                
                if chunk.data:
                    self._stats['records_loaded'] += len(chunk.data)  # type: ignore
                
                # Call progress callback if provided
                if progress_callback:
                    progress = chunk.metadata.get('progress', 0.0)
                    progress_callback(progress, self._stats)
                
                if chunk.is_last:
                    break
            
            sink.flush()
            
        finally:
            sink.close()
            source.close()
            self._stats['end_time'] = time.time()
        
        return self._stats
    
    def export_to_sink(
        self,
        sink: IStreamSink,
        query: Query | None = None,
        batch_size: int = 1000,
        progress_callback: Union[Callable, None] = None
    ) -> Dict[str, Any]:
        """Export data from database to stream sink.
        
        Args:
            sink: Stream sink to write to.
            query: Query to filter records.
            batch_size: Batch size for reading.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Export statistics.
        """
        self._stats['start_time'] = time.time()
        
        source = DatabaseStreamSource(
            self.database,
            query=query,
            batch_size=batch_size
        )
        
        try:
            for chunk in source:
                success = sink.write_chunk(chunk)
                
                if not success:
                    self._stats['errors'] += 1  # type: ignore
                
                self._stats['batches_processed'] += 1  # type: ignore
                
                if chunk.data:
                    self._stats['records_loaded'] += len(chunk.data)  # type: ignore
                
                # Call progress callback if provided
                if progress_callback:
                    progress = chunk.metadata.get('progress', 0.0)
                    progress_callback(progress, self._stats)
                
                if chunk.is_last:
                    break
            
            sink.flush()
            
        finally:
            sink.close()
            source.close()
            self._stats['end_time'] = time.time()
        
        return self._stats

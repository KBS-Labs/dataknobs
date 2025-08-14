"""Backend-to-backend data migration utilities."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from dataknobs_data.database import Database, SyncDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record

logger = logging.getLogger(__name__)


@dataclass
class MigrationProgress:
    """Track migration progress."""
    total_records: int = 0
    processed_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate migration duration in seconds."""
        if not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def records_per_second(self) -> float:
        """Calculate processing rate."""
        duration = self.duration
        if duration and duration > 0:
            return self.processed_records / duration
        return 0.0


@dataclass
class MigrationResult:
    """Migration operation result."""
    success: bool
    progress: MigrationProgress
    message: str = ""
    

class DataMigrator:
    """Migrate data between database backends."""
    
    def __init__(
        self,
        source: Union[Database, SyncDatabase],
        target: Union[Database, SyncDatabase],
        max_workers: int = 4
    ):
        """Initialize data migrator.
        
        Args:
            source: Source database to migrate from
            target: Target database to migrate to
            max_workers: Maximum concurrent workers for batch operations
        """
        self.source = source
        self.target = target
        self.max_workers = max_workers
        self._is_async = isinstance(source, Database)
        
    async def migrate_async(
        self,
        query: Optional[Query] = None,
        batch_size: int = 1000,
        transform: Optional[Callable[[Record], Optional[Record]]] = None,
        progress_callback: Optional[Callable[[MigrationProgress], None]] = None,
        error_handler: Optional[Callable[[Exception, Record], bool]] = None,
        preserve_ids: bool = False
    ) -> MigrationResult:
        """Migrate data asynchronously.
        
        Args:
            query: Optional query to filter source records
            batch_size: Number of records to process in each batch
            transform: Optional transformation function for records
            progress_callback: Optional callback for progress updates
            error_handler: Optional error handler (return True to continue, False to stop)
            preserve_ids: Whether to preserve record IDs from source
            
        Returns:
            MigrationResult with success status and statistics
        """
        if not isinstance(self.source, Database):
            raise TypeError("Source must be an async Database for async migration")
        if not isinstance(self.target, Database):
            raise TypeError("Target must be an async Database for async migration")
            
        progress = MigrationProgress()
        progress.start_time = datetime.now()
        
        try:
            # Count total records
            if query:
                all_records = await self.source.search(query)
                progress.total_records = len(all_records)
            else:
                all_records = await self.source.search(Query())
                progress.total_records = len(all_records)
            
            logger.info(f"Starting migration of {progress.total_records} records")
            
            # Process in batches
            for i in range(0, progress.total_records, batch_size):
                batch = all_records[i:i + batch_size]
                batch_tasks = []
                
                for record in batch:
                    try:
                        # Apply transformation if provided
                        if transform:
                            transformed = transform(record)
                            if transformed is None:
                                progress.processed_records += 1
                                continue  # Skip this record
                            record = transformed
                        
                        # Create or upsert in target
                        if preserve_ids and hasattr(record, '_id'):
                            task = self.target.upsert(record._id, record)
                        else:
                            task = self.target.create(record)
                        
                        batch_tasks.append(task)
                        
                    except Exception as e:
                        progress.failed_records += 1
                        progress.errors.append({
                            'record': str(record),
                            'error': str(e)
                        })
                        
                        if error_handler and not error_handler(e, record):
                            raise
                
                # Execute batch operations
                if batch_tasks:
                    results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for result in results:
                        progress.processed_records += 1
                        if isinstance(result, Exception):
                            progress.failed_records += 1
                            progress.errors.append({'error': str(result)})
                        else:
                            progress.successful_records += 1
                
                # Report progress
                if progress_callback:
                    progress_callback(progress)
                
                logger.debug(f"Processed {progress.processed_records}/{progress.total_records} records")
            
            progress.end_time = datetime.now()
            
            success = progress.failed_records == 0
            message = f"Migration completed: {progress.successful_records} successful, {progress.failed_records} failed"
            logger.info(message)
            
            return MigrationResult(success=success, progress=progress, message=message)
            
        except Exception as e:
            progress.end_time = datetime.now()
            message = f"Migration failed: {str(e)}"
            logger.error(message)
            return MigrationResult(success=False, progress=progress, message=message)
    
    def migrate_sync(
        self,
        query: Optional[Query] = None,
        batch_size: int = 1000,
        transform: Optional[Callable[[Record], Optional[Record]]] = None,
        progress_callback: Optional[Callable[[MigrationProgress], None]] = None,
        error_handler: Optional[Callable[[Exception, Record], bool]] = None,
        preserve_ids: bool = False
    ) -> MigrationResult:
        """Migrate data synchronously.
        
        Args:
            query: Optional query to filter source records
            batch_size: Number of records to process in each batch
            transform: Optional transformation function for records
            progress_callback: Optional callback for progress updates
            error_handler: Optional error handler (return True to continue, False to stop)
            preserve_ids: Whether to preserve record IDs from source
            
        Returns:
            MigrationResult with success status and statistics
        """
        if not isinstance(self.source, SyncDatabase):
            raise TypeError("Source must be a SyncDatabase for sync migration")
        if not isinstance(self.target, SyncDatabase):
            raise TypeError("Target must be a SyncDatabase for sync migration")
            
        progress = MigrationProgress()
        progress.start_time = datetime.now()
        
        try:
            # Count total records
            if query:
                all_records = self.source.search(query)
                progress.total_records = len(all_records)
            else:
                all_records = self.source.search(Query())
                progress.total_records = len(all_records)
            
            logger.info(f"Starting migration of {progress.total_records} records")
            
            # Process in batches with thread pool
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i in range(0, progress.total_records, batch_size):
                    batch = all_records[i:i + batch_size]
                    futures = []
                    
                    for record in batch:
                        try:
                            # Apply transformation if provided
                            if transform:
                                transformed = transform(record)
                                if transformed is None:
                                    progress.processed_records += 1
                                    continue  # Skip this record
                                record = transformed
                            
                            # Submit to thread pool
                            if preserve_ids and hasattr(record, '_id'):
                                future = executor.submit(self.target.upsert, record._id, record)
                            else:
                                future = executor.submit(self.target.create, record)
                            
                            futures.append(future)
                            
                        except Exception as e:
                            progress.failed_records += 1
                            progress.errors.append({
                                'record': str(record),
                                'error': str(e)
                            })
                            
                            if error_handler and not error_handler(e, record):
                                raise
                    
                    # Wait for batch completion
                    for future in futures:
                        try:
                            future.result()
                            progress.processed_records += 1
                            progress.successful_records += 1
                        except Exception as e:
                            progress.processed_records += 1
                            progress.failed_records += 1
                            progress.errors.append({'error': str(e)})
                    
                    # Report progress
                    if progress_callback:
                        progress_callback(progress)
                    
                    logger.debug(f"Processed {progress.processed_records}/{progress.total_records} records")
            
            progress.end_time = datetime.now()
            
            success = progress.failed_records == 0
            message = f"Migration completed: {progress.successful_records} successful, {progress.failed_records} failed"
            logger.info(message)
            
            return MigrationResult(success=success, progress=progress, message=message)
            
        except Exception as e:
            progress.end_time = datetime.now()
            message = f"Migration failed: {str(e)}"
            logger.error(message)
            return MigrationResult(success=False, progress=progress, message=message)
    
    def migrate(self, **kwargs) -> MigrationResult:
        """Migrate data (auto-detects sync/async).
        
        Automatically uses the appropriate migration method based on
        the database types.
        
        Args:
            **kwargs: Arguments passed to migrate_async or migrate_sync
            
        Returns:
            MigrationResult with success status and statistics
        """
        if self._is_async:
            # Run async migration in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context
                task = self.migrate_async(**kwargs)
                return asyncio.create_task(task)
            else:
                # Run in new event loop
                return loop.run_until_complete(self.migrate_async(**kwargs))
        else:
            return self.migrate_sync(**kwargs)
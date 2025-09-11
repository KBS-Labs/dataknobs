"""Database storage backend for execution history using dataknobs_data.

This module provides a unified storage backend that works with ANY dataknobs_data
database backend (SQLite, PostgreSQL, MongoDB, Elasticsearch, S3, etc.) through
the common AsyncDatabase interface.
"""

import time
import uuid
from typing import Any, Dict, List, TYPE_CHECKING

from dataknobs_data.records import Record
from dataknobs_data.query import Query
from dataknobs_data.schema import DatabaseSchema, FieldSchema

if TYPE_CHECKING:
    from dataknobs_data.database import AsyncDatabase

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory, ExecutionStep, ExecutionStatus
from dataknobs_fsm.storage.base import BaseHistoryStorage, StorageBackend, StorageConfig, StorageFactory


class UnifiedDatabaseStorage(BaseHistoryStorage):
    """Unified database storage that works with any dataknobs_data backend.
    
    This single implementation works with:
    - Memory (AsyncMemoryDatabase)
    - SQLite (AsyncSQLiteDatabase)
    - PostgreSQL (AsyncPostgresDatabase)
    - MongoDB (AsyncMongoDatabase)
    - Elasticsearch (AsyncElasticsearchDatabase)
    - S3 (AsyncS3Database)
    - File (AsyncFileDatabase)
    
    All through the same AsyncDatabase interface from dataknobs_data.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize database storage.
        
        Args:
            config: Storage configuration with backend type in connection_params.
        """
        super().__init__(config)
        self._db: AsyncDatabase | None = None
        self._steps_db: AsyncDatabase | None = None  # Separate DB for steps if needed
    
    async def _setup_backend(self) -> None:
        """Set up the database backend using dataknobs_data factory."""
        # Extract backend type from config
        backend_type = self.config.connection_params.get('type', 'memory')
        
        # Prepare dataknobs_data configuration
        db_config = {
            **self.config.connection_params,
            'schema': self._create_history_schema()
        }
        
        # Remove 'type' as it's not needed by dataknobs_data
        db_config.pop('type', None)
        
        # Use AsyncDatabaseFactory to create database instance
        from dataknobs_data.factory import AsyncDatabaseFactory
        factory = AsyncDatabaseFactory()
        
        # The factory expects 'backend' not 'type'
        db_config['backend'] = backend_type
        
        self._db = factory.create(**db_config)
        
        # Connect to the database if it has a connect method
        if hasattr(self._db, 'connect'):
            await self._db.connect()
        
        # For steps, use the same database instance
        # Different backends handle collections/tables differently
        self._steps_db = self._db
    
    def _create_history_schema(self) -> DatabaseSchema:
        """Create schema for history records."""
        from dataknobs_data.fields import FieldType
        
        schema = DatabaseSchema()
        
        # Core fields
        schema.add_field(FieldSchema(
            name='id', 
            type=FieldType.TEXT,
            metadata={'primary_key': True}
        ))
        schema.add_field(FieldSchema(
            name='execution_id',
            type=FieldType.TEXT,
            metadata={'indexed': True, 'unique': True}
        ))
        schema.add_field(FieldSchema(
            name='fsm_name',
            type=FieldType.TEXT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='data_mode',
            type=FieldType.TEXT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='status',
            type=FieldType.TEXT,
            metadata={'indexed': True}
        ))
        
        # Timing fields
        schema.add_field(FieldSchema(
            name='start_time',
            type=FieldType.FLOAT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='end_time',
            type=FieldType.FLOAT,
            required=False
        ))
        
        # Metrics
        schema.add_field(FieldSchema(
            name='total_steps',
            type=FieldType.INTEGER,
            default=0
        ))
        schema.add_field(FieldSchema(
            name='failed_steps',
            type=FieldType.INTEGER,
            default=0
        ))
        schema.add_field(FieldSchema(
            name='skipped_steps',
            type=FieldType.INTEGER,
            default=0
        ))
        
        # JSON data fields
        schema.add_field(FieldSchema(
            name='history_data',
            type=FieldType.JSON
        ))
        schema.add_field(FieldSchema(
            name='metadata',
            type=FieldType.JSON
        ))
        
        # Timestamps
        schema.add_field(FieldSchema(
            name='created_at',
            type=FieldType.FLOAT
        ))
        schema.add_field(FieldSchema(
            name='updated_at',
            type=FieldType.FLOAT
        ))
        
        return schema
    
    def _create_steps_schema(self) -> DatabaseSchema:
        """Create schema for step records."""
        from dataknobs_data.fields import FieldType
        
        schema = DatabaseSchema()
        
        schema.add_field(FieldSchema(
            name='id',
            type=FieldType.TEXT,
            metadata={'primary_key': True}
        ))
        schema.add_field(FieldSchema(
            name='execution_id',
            type=FieldType.TEXT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='step_id',
            type=FieldType.TEXT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='parent_id',
            type=FieldType.TEXT,
            required=False
        ))
        schema.add_field(FieldSchema(
            name='state_name',
            type=FieldType.TEXT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='network_name',
            type=FieldType.TEXT
        ))
        schema.add_field(FieldSchema(
            name='status',
            type=FieldType.TEXT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='timestamp',
            type=FieldType.FLOAT,
            metadata={'indexed': True}
        ))
        schema.add_field(FieldSchema(
            name='step_data',
            type=FieldType.JSON
        ))
        
        return schema
    
    async def save_history(
        self,
        history: ExecutionHistory,
        metadata: Dict[str, Any] | None = None
    ) -> str:
        """Save execution history to database."""
        if not self._db:
            await self.initialize()
        
        history_id = history.execution_id
        
        # Serialize history based on data mode
        history_data = self._serialize_history(history)
        
        # Create record using dataknobs_data Record
        record = Record({
            'id': str(uuid.uuid4()),
            'execution_id': history_id,
            'fsm_name': history.fsm_name,
            'data_mode': history.data_mode.value,
            'status': 'completed' if history.end_time else 'in_progress',
            'start_time': history.start_time,
            'end_time': history.end_time,
            'total_steps': history.total_steps,
            'failed_steps': history.failed_steps,
            'skipped_steps': history.skipped_steps,
            'history_data': history_data,
            'metadata': metadata or {},
            'created_at': time.time(),
            'updated_at': time.time()
        })
        
        # Save using dataknobs_data interface - just pass the record
        await self._db.upsert(record)
        
        return history_id
    
    async def load_history(self, history_id: str) -> ExecutionHistory | None:
        """Load execution history from database."""
        if not self._db:
            await self.initialize()
        
        # Query using dataknobs_data Query builder
        query = Query().filter('execution_id', '=', history_id)
        
        # Find record
        results = await self._db.search(query)
        record = results[0] if results else None
        
        if not record:
            return None
        
        # Deserialize history
        history = self._deserialize_history(
            record['history_data'],
            record['fsm_name'],
            history_id
        )
        
        return history
    
    async def save_step(
        self,
        execution_id: str,
        step: ExecutionStep,
        parent_id: str | None = None
    ) -> str:
        """Save a single execution step."""
        if not self._steps_db:
            await self.initialize()
        
        # Create step record
        record = Record({
            'id': str(uuid.uuid4()),
            'execution_id': execution_id,
            'step_id': step.step_id,
            'parent_id': parent_id,
            'state_name': step.state_name,
            'network_name': step.network_name,
            'status': step.status.value,
            'timestamp': step.timestamp,
            'step_data': step.to_dict()
        })
        
        await self._steps_db.upsert(record)
        return step.step_id
    
    async def load_steps(
        self,
        execution_id: str,
        filters: Dict[str, Any] | None = None
    ) -> List[ExecutionStep]:
        """Load execution steps from database."""
        if not self._steps_db:
            await self.initialize()
        
        # Build query
        query = Query().filter('execution_id', '=', execution_id)
        
        if filters:
            for key, value in filters.items():
                query = query.filter(key, '=', value)
        
        # Load and reconstruct steps
        steps = []
        results = await self._steps_db.search(query)
        for record in results:
            step_data = record['step_data']
            
            step = ExecutionStep(
                step_id=step_data['step_id'],
                state_name=step_data['state_name'],
                network_name=step_data['network_name'],
                timestamp=step_data['timestamp'],
                data_mode=DataHandlingMode(step_data['data_mode']),
                status=ExecutionStatus(step_data['status'])
            )
            
            # Restore other properties
            for attr in ['start_time', 'end_time', 'arc_taken', 'metrics', 
                        'resource_usage', 'stream_progress', 'chunks_processed', 
                        'records_processed']:
                if attr in step_data:
                    setattr(step, attr, step_data[attr])
            
            if step_data.get('error'):
                step.error = Exception(step_data['error'])
            
            steps.append(step)
        
        return steps
    
    async def query_histories(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query execution histories."""
        if not self._db:
            await self.initialize()
        
        # Build query using dataknobs_data Query
        query = Query()
        
        # Map filter keys to database fields
        for key, value in filters.items():
            if key in ['fsm_name', 'data_mode', 'status']:
                query = query.filter(key, '=', value)
            elif key == 'start_time_after':
                query = query.filter('start_time', '>=', value)
            elif key == 'start_time_before':
                query = query.filter('start_time', '<=', value)
            elif key == 'failed':
                if value:
                    query = query.filter('failed_steps', '>', 0)
                else:
                    query = query.filter('failed_steps', '=', 0)
        
        # Apply pagination
        query = query.sort_by('start_time', 'desc').limit(limit).offset(offset)
        
        # Execute and return results
        results = []
        search_results = await self._db.search(query)
        for record in search_results:
            results.append({
                'id': record['execution_id'],
                'fsm_name': record['fsm_name'],
                'data_mode': record['data_mode'],
                'status': record['status'],
                'start_time': record['start_time'],
                'end_time': record.get_value('end_time'),
                'total_steps': record['total_steps'],
                'failed_steps': record['failed_steps'],
                'metadata': record.get_value('metadata', {})
            })
        
        return results
    
    async def delete_history(self, history_id: str) -> bool:
        """Delete execution history."""
        if not self._db:
            await self.initialize()
        
        # Find and delete history records
        query = Query().filter('execution_id', '=', history_id)
        records = await self._db.search(query)
        
        deleted_count = 0
        for record in records:
            # Get the storage ID from the record
            record_id = record.storage_id or record.get_value('id')
            if record_id and await self._db.delete(record_id):
                deleted_count += 1
        
        # Delete associated steps
        if self._steps_db:
            step_query = Query().filter('execution_id', '=', history_id)
            step_records = await self._steps_db.search(step_query)
            for step_record in step_records:
                step_id = step_record.storage_id or step_record.get_value('id')
                if step_id:
                    await self._steps_db.delete(step_id)
        
        return deleted_count > 0
    
    async def get_statistics(
        self,
        execution_id: str | None = None
    ) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._db:
            await self.initialize()
        
        if execution_id:
            # Specific execution stats
            query = Query().filter('execution_id', '=', execution_id)
            
            search_results = await self._db.search(query)
            for record in search_results:
                return {
                    'execution_id': execution_id,
                    'fsm_name': record['fsm_name'],
                    'data_mode': record['data_mode'],
                    'status': record['status'],
                    'total_steps': record['total_steps'],
                    'failed_steps': record['failed_steps'],
                    'start_time': record['start_time'],
                    'end_time': record.get_value('end_time')
                }
            return {}
        else:
            # Overall stats
            stats = {
                'total_histories': 0,
                'mode_distribution': {},
                'status_distribution': {},
                'backend_type': self.config.connection_params.get('type', 'unknown')
            }
            
            all_records = await self._db.search(Query())
            for record in all_records:
                stats['total_histories'] += 1
                
                mode = record['data_mode']
                stats['mode_distribution'][mode] = stats['mode_distribution'].get(mode, 0) + 1
                
                status = record['status']
                stats['status_distribution'][status] = stats['status_distribution'].get(status, 0) + 1
            
            return stats
    
    async def cleanup(
        self,
        before_timestamp: float | None = None,
        keep_failed: bool = True
    ) -> int:
        """Clean up old histories."""
        if not self._db:
            await self.initialize()
        
        if before_timestamp is None:
            before_timestamp = time.time() - (7 * 86400)  # 7 days
        
        # Build query
        query = Query().filter('start_time', '<', before_timestamp)
        
        if keep_failed:
            query = query.filter('failed_steps', '=', 0)
        
        # Get histories to delete
        to_delete = []
        search_results = await self._db.search(query)
        for record in search_results:
            to_delete.append(record['execution_id'])
        
        # Delete each
        deleted = 0
        for history_id in to_delete:
            if await self.delete_history(history_id):
                deleted += 1
        
        # Close database connection if supported
        if hasattr(self._db, 'close'):
            await self._db.close()
        
        return deleted


# Register all backends with the same implementation
for backend in [StorageBackend.MEMORY, StorageBackend.SQLITE, 
                StorageBackend.POSTGRES, StorageBackend.MONGODB,
                StorageBackend.ELASTICSEARCH, StorageBackend.S3]:
    StorageFactory.register(backend, UnifiedDatabaseStorage)

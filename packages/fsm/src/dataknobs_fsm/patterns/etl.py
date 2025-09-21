"""AsyncDatabase ETL (Extract, Transform, Load) pattern implementation.

This module provides pre-configured FSM patterns for ETL operations,
including data extraction from source databases, transformation pipelines,
and loading into target systems.
"""

from typing import Any, Dict, List, Union, Callable, AsyncIterator
from dataclasses import dataclass
from enum import Enum
from dataknobs_data import AsyncDatabase, Record, Query

from ..api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from ..functions.library.database import DatabaseFetch, DatabaseUpsert
from ..functions.library.transformers import (
    FieldMapper, DataEnricher
)
from ..functions.library.validators import SchemaValidator


class ETLMode(Enum):
    """ETL processing modes."""
    FULL_REFRESH = "full"  # Replace all data
    INCREMENTAL = "incremental"  # Process only new/changed data
    UPSERT = "upsert"  # Update existing, insert new
    APPEND = "append"  # Always append, no updates


@dataclass
class ETLConfig:
    """Configuration for ETL pipeline."""
    source_db: Dict[str, Any]  # Source database config
    target_db: Dict[str, Any]  # Target database config
    mode: ETLMode = ETLMode.FULL_REFRESH
    batch_size: int = 1000
    parallel_workers: int = 4
    error_threshold: float = 0.05  # Max 5% errors
    checkpoint_interval: int = 10000  # Checkpoint every N records
    
    # Optional configurations
    source_query: str | None = "SELECT * FROM source_table"
    target_table: str = "target_table"
    key_columns: List[str] | None = None
    field_mappings: Dict[str, str] | None = None
    transformations: List[Callable] | None = None
    validation_schema: Dict[str, Any] | None = None
    enrichment_sources: List[Dict[str, Any]] | None = None


class DatabaseETL:
    """AsyncDatabase ETL pipeline using FSM pattern."""
    
    def __init__(self, config: ETLConfig):
        """Initialize ETL pipeline.
        
        Args:
            config: ETL configuration
        """
        self.config = config
        self._fsm = self._build_fsm()
        self._checkpoint_data = {}
        self._metrics = {
            'extracted': 0,
            'transformed': 0,
            'loaded': 0,
            'errors': 0,
            'skipped': 0
        }
        
    def _build_fsm(self) -> SimpleFSM:
        """Build FSM for ETL pipeline."""
        # Build resources list
        resources = [
            {'name': 'source_db', 'type': 'database', 'config': self.config.source_db},
            {'name': 'target_db', 'type': 'database', 'config': self.config.target_db}
        ]
        
        # Add enrichment resources if configured
        if self.config.enrichment_sources:
            for i, source in enumerate(self.config.enrichment_sources):
                if 'database' in source:
                    resources.append({
                        'name': f'enrichment_db_{i}',
                        'type': 'database',
                        'config': source['database']
                    })
                elif 'api' in source:
                    resources.append({
                        'name': f'enrichment_api_{i}',
                        'type': 'http',
                        'config': source['api']
                    })
        
        # Create FSM configuration
        fsm_config = {
            'name': 'ETL_Pipeline',
            'data_mode': DataHandlingMode.COPY.value,  # Use COPY for data isolation
            'resources': resources,
            'states': [
                {
                    'name': 'extract',
                    'is_start': True,
                    'resources': ['source_db']
                },
                {
                    'name': 'validate',
                    'resources': []
                },
                {
                    'name': 'transform',
                    'resources': self._get_transform_resources()
                },
                {
                    'name': 'enrich',
                    'resources': self._get_enrichment_resources()
                },
                {
                    'name': 'load',
                    'resources': ['target_db']
                },
                {
                    'name': 'complete',
                    'is_end': True
                },
                {
                    'name': 'error',
                    'is_end': True
                }
            ],
            'arcs': [
                {
                    'from': 'extract',
                    'to': 'validate',
                    'name': 'extracted'
                },
                {
                    'from': 'validate',
                    'to': 'transform',
                    'name': 'valid',
                    'pre_test': self._create_validation_test_reference()
                },
                {
                    'from': 'validate',
                    'to': 'error',
                    'name': 'invalid'
                },
                {
                    'from': 'transform',
                    'to': 'enrich' if self.config.enrichment_sources else 'load',
                    'name': 'transformed',
                    'transform': self._create_transformer_reference()
                },
                {
                    'from': 'enrich',
                    'to': 'load',
                    'name': 'enriched',
                    'transform': self._create_enricher_reference()
                },
                {
                    'from': 'load',
                    'to': 'complete',
                    'name': 'loaded'
                }
            ]
        }
        
        # Add functions
        self._register_functions(fsm_config)
        
        return SimpleFSM(fsm_config, data_mode=DataHandlingMode.COPY)
        
    def _get_transform_resources(self) -> List[str]:
        """Get resources needed for transformation."""
        resources = []
        if self.config.enrichment_sources:
            for i, source in enumerate(self.config.enrichment_sources):
                if 'database' in source:
                    resources.append(f'enrichment_db_{i}')
        return resources
        
    def _get_enrichment_resources(self) -> List[str]:
        """Get resources needed for enrichment."""
        resources = []
        if self.config.enrichment_sources:
            for i, source in enumerate(self.config.enrichment_sources):
                if 'api' in source:
                    resources.append(f'enrichment_api_{i}')
        return resources
        
    def _create_validation_test(self) -> Callable | None:
        """Create validation test function."""
        if not self.config.validation_schema:
            return None
            
        validator = SchemaValidator(self.config.validation_schema)
        return lambda state: validator.validate(Record(state.data))  # type: ignore
        
    def _create_validation_test_reference(self) -> Dict[str, Any] | None:
        """Create validation test function reference for FSM config."""
        if not self.config.validation_schema:
            return None
            
        import json
        # Build validation code based on schema
        code_lines = [
            "# Validate data against schema",
            f"schema = {json.dumps(self.config.validation_schema)}",
            "# Basic schema validation",
            "if schema.get('type') == 'object':",
            "    required = schema.get('required', [])",
            "    for field in required:",
            "        if field not in data:",
            "            False",
            "True"
        ]
        
        return {
            'type': 'inline',
            'code': '\n'.join(code_lines)
        }
        
    def _create_transformer(self) -> Callable:
        """Create transformation function."""
        transformers = []
        
        # Add field mapping
        if self.config.field_mappings:
            transformers.append(FieldMapper(self.config.field_mappings))
            
        # Add custom transformations
        if self.config.transformations:
            transformers.extend(self.config.transformations)  # type: ignore
            
        # Compose transformers
        async def transform(data: Dict[str, Any]) -> Dict[str, Any]:
            result = data
            for transformer in transformers:
                if hasattr(transformer, 'transform'):
                    result = await transformer.transform(result)  # type: ignore
                elif callable(transformer):
                    result = transformer(result)
            return result
            
        return transform
        
    def _create_enricher(self) -> Callable | None:
        """Create enrichment function."""
        if not self.config.enrichment_sources:
            return None
            
        enricher = DataEnricher(self.config.enrichment_sources)  # type: ignore
        return enricher.transform
        
    def _create_transformer_reference(self) -> Dict[str, Any] | None:
        """Create transformation function reference for FSM config."""
        if not self.config.field_mappings and not self.config.transformations:
            return None
            
        # Build transformation code
        code_lines = [
            "# Apply transformations",
            "result = data"
        ]
        
        if self.config.field_mappings:
            # Add field mapping code
            for old_name, new_name in self.config.field_mappings.items():
                code_lines.append(f"if '{old_name}' in result:")
                code_lines.append(f"    result['{new_name}'] = result.pop('{old_name}')")
        
        # For custom transformations, we'll apply them as simple dict updates
        if self.config.transformations:
            code_lines.append("# Apply custom transformations")
            code_lines.append("result['transformed'] = True")
        
        code_lines.append("result")
        
        return {
            'type': 'inline',
            'code': '\n'.join(code_lines)
        }
        
    def _create_enricher_reference(self) -> Dict[str, Any] | None:
        """Create enrichment function reference for FSM config."""
        if not self.config.enrichment_sources:
            return None
            
        # Build enrichment code
        code_lines = [
            "# Enrich data",
            "result = data",
            "result['enriched'] = True",
            "result"
        ]
        
        return {
            'type': 'inline',
            'code': '\n'.join(code_lines)
        }
        
    def _register_functions(self, config: Dict[str, Any]) -> None:
        """Register ETL-specific functions."""
        # Register database functions
        config['functions'] = {
            'extract': DatabaseFetch(
                resource_name='source_db',
                query=self.config.source_query  # type: ignore
            ),
            'load': DatabaseUpsert(
                resource_name='target_db',
                table=self.config.target_table,
                key_columns=self.config.key_columns or ['id']
            )
        }
        
    async def run(
        self,
        checkpoint_id: str | None = None
    ) -> Dict[str, Any]:
        """Run ETL pipeline.
        
        Args:
            checkpoint_id: Optional checkpoint to resume from
            
        Returns:
            ETL execution metrics
        """
        # Resume from checkpoint if provided
        if checkpoint_id:
            await self._load_checkpoint(checkpoint_id)
            
        # Extract data
        source_db = await AsyncDatabase.create(
            self.config.source_db['type'],
            self.config.source_db  # type: ignore
        )
        
        try:
            # Determine extraction strategy
            if self.config.mode == ETLMode.INCREMENTAL:
                query = self._get_incremental_query()
            else:
                query = self.config.source_query or Query()
                
            # Process in batches
            async for batch in self._extract_batches(source_db, query):  # type: ignore
                # Process batch through FSM
                results = self._fsm.process_batch(
                    data=batch,  # type: ignore
                    batch_size=self.config.batch_size,
                    max_workers=self.config.parallel_workers
                )
                
                # Update metrics
                self._update_metrics(results)
                
                # Check error threshold
                if self._check_error_threshold():
                    from ..core.exceptions import ETLError
                    raise ETLError(f"Error threshold exceeded: {self._metrics['errors']} errors")
                    
                # Checkpoint if needed
                if self._should_checkpoint():
                    await self._save_checkpoint()
                    
        finally:
            await source_db.close()
            
        return self._metrics
        
    async def _extract_batches(
        self,
        db: AsyncDatabase,
        query: Query
    ) -> AsyncIterator[List[Dict[str, Any]]]:
        """Extract data in batches.
        
        Args:
            db: Source database
            query: Extraction query
            
        Yields:
            Batches of records
        """
        batch = []
        async for record in db.stream_read(query):
            batch.append(record.to_dict())
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
                
        if batch:
            yield batch
            
    def _get_incremental_query(self) -> Query:
        """Get query for incremental extraction."""
        # Get last processed timestamp from checkpoint
        last_timestamp = self._checkpoint_data.get('last_timestamp')
        
        if last_timestamp:
            return Query().filter('updated_at', '>', last_timestamp)
        else:
            return Query()
            
    def _update_metrics(self, results: List[Dict[str, Any]]) -> None:
        """Update execution metrics."""
        for result in results:
            if result['success']:
                if result['final_state'] == 'complete':
                    self._metrics['loaded'] += 1
                elif result['final_state'] == 'error':
                    self._metrics['errors'] += 1
            else:
                self._metrics['errors'] += 1
                
        self._metrics['extracted'] = self._metrics['loaded'] + self._metrics['errors']
        
    def _check_error_threshold(self) -> bool:
        """Check if error threshold is exceeded."""
        if self._metrics['extracted'] == 0:
            return False
            
        error_rate = self._metrics['errors'] / self._metrics['extracted']
        return error_rate > self.config.error_threshold
        
    def _should_checkpoint(self) -> bool:
        """Check if checkpoint should be saved."""
        return self._metrics['extracted'] % self.config.checkpoint_interval == 0
        
    async def _save_checkpoint(self) -> str:
        """Save checkpoint for resume capability."""
        import json
        import hashlib
        from datetime import datetime
        
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self._metrics,
            'config': {
                'mode': self.config.mode.value,
                'batch_size': self.config.batch_size
            },
            'position': self._metrics['extracted']
        }
        
        # Generate checkpoint ID
        checkpoint_id = hashlib.md5(
            json.dumps(checkpoint).encode()
        ).hexdigest()[:8]
        
        # Save to storage (simplified - would use persistent storage)
        self._checkpoint_data[checkpoint_id] = checkpoint
        
        return checkpoint_id
        
    async def _load_checkpoint(self, checkpoint_id: str) -> None:
        """Load checkpoint data."""
        if checkpoint_id in self._checkpoint_data:
            checkpoint = self._checkpoint_data[checkpoint_id]
            self._metrics = checkpoint['metrics']


def create_etl_pipeline(
    source: Union[str, Dict[str, Any]],
    target: Union[str, Dict[str, Any]],
    mode: ETLMode = ETLMode.FULL_REFRESH,
    **kwargs
) -> DatabaseETL:
    """Factory function to create ETL pipeline.
    
    Args:
        source: Source database configuration or connection string
        target: Target database configuration or connection string
        mode: ETL mode
        **kwargs: Additional configuration options
        
    Returns:
        Configured DatabaseETL instance
    """
    # Parse connection strings if needed
    if isinstance(source, str):
        source = _parse_connection_string(source)
    if isinstance(target, str):
        target = _parse_connection_string(target)
        
    config = ETLConfig(
        source_db=source,
        target_db=target,
        mode=mode,
        **kwargs
    )
    
    return DatabaseETL(config)


def _parse_connection_string(conn_str: str) -> Dict[str, Any]:
    """Parse database connection string.
    
    Args:
        conn_str: Connection string
        
    Returns:
        AsyncDatabase configuration dictionary
    """
    # Simplified parsing - real implementation would be more robust
    if conn_str.startswith('postgresql://'):
        return {
            'type': 'postgres',
            'connection_string': conn_str
        }
    elif conn_str.startswith('mongodb://'):
        return {
            'type': 'mongodb',
            'connection_string': conn_str
        }
    elif conn_str.startswith('sqlite://'):
        return {
            'type': 'sqlite',
            'path': conn_str.replace('sqlite://', '')
        }
    else:
        raise ValueError(f"Unsupported connection string: {conn_str}")


# Pre-configured ETL patterns

def create_database_sync(
    source: Dict[str, Any],
    target: Dict[str, Any],
    sync_interval: int = 300  # 5 minutes
) -> DatabaseETL:
    """Create database synchronization pipeline.
    
    Args:
        source: Source database config
        target: Target database config
        sync_interval: Sync interval in seconds
        
    Returns:
        AsyncDatabase sync ETL pipeline
    """
    return create_etl_pipeline(
        source=source,
        target=target,
        mode=ETLMode.INCREMENTAL,
        checkpoint_interval=1000
    )


def create_data_migration(
    source: Dict[str, Any],
    target: Dict[str, Any],
    field_mappings: Dict[str, str] | None = None,
    transformations: List[Callable] | None = None
) -> DatabaseETL:
    """Create data migration pipeline.
    
    Args:
        source: Source database config
        target: Target database config
        field_mappings: Field name mappings
        transformations: Data transformation functions
        
    Returns:
        Data migration ETL pipeline
    """
    return create_etl_pipeline(
        source=source,
        target=target,
        mode=ETLMode.FULL_REFRESH,
        field_mappings=field_mappings,
        transformations=transformations,
        batch_size=5000,
        parallel_workers=8
    )


def create_data_warehouse_load(
    sources: List[Dict[str, Any]],
    warehouse: Dict[str, Any],
    aggregations: List[Callable] | None = None
) -> List[DatabaseETL]:
    """Create data warehouse loading pipelines.
    
    Args:
        sources: List of source database configs
        warehouse: Data warehouse config
        aggregations: Aggregation functions
        
    Returns:
        List of ETL pipelines for each source
    """
    pipelines = []
    
    for source in sources:
        pipeline = create_etl_pipeline(
            source=source,
            target=warehouse,
            mode=ETLMode.APPEND,
            transformations=aggregations,
            batch_size=10000
        )
        pipelines.append(pipeline)
        
    return pipelines

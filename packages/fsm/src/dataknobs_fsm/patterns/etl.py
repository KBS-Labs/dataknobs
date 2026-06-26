"""AsyncDatabase ETL (Extract, Transform, Load) pattern implementation.

This module provides pre-configured FSM patterns for ETL operations,
including data extraction from source databases, transformation pipelines,
and loading into target systems.
"""

import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable, ClassVar, Dict, List, Union

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)

from dataknobs_data import AsyncDatabase, Query
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.exceptions import ETLError, InvalidConfigurationError

from ..api.async_simple import AsyncSimpleFSM
from ..functions.base import ITransformFunction, TransformError
from ..functions.library.database import DatabaseUpsert

logger = logging.getLogger(__name__)


class ETLMode(Enum):
    """ETL processing modes."""
    FULL_REFRESH = "full"  # Replace all data
    INCREMENTAL = "incremental"  # Process only new/changed data
    UPSERT = "upsert"  # Update existing, insert new
    APPEND = "append"  # Always append, no updates


@dataclass(frozen=True)
class ETLConfig(StructuredConfig):
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

    def __post_init__(self) -> None:
        """Validate that field mappings don't rename a key column away.

        ``field_mappings`` run in the ``transform`` stage, before ``load``
        derives each row's storage id from ``key_columns`` (post-transform
        names). If a mapping renames a key column's source to a different name,
        that key column no longer exists under its original name when load
        derives the id — every row would collapse onto a single ``"None"`` id,
        silently overwriting the whole target with one record. Catch that
        destructive config at construction; ``key_columns`` must reference the
        post-transform field names.

        This check is deliberately conservative: it rejects renaming a key
        column's *source* name even when another mapping would recreate the key
        under its original name (e.g. ``{"a": "id", "id": "b"}`` with
        ``key_columns=["id"]``). Mapping order makes that combination fragile,
        so it is rejected rather than reasoned about.
        """
        key_columns = self.key_columns or []
        mappings = self.field_mappings or {}
        for col in key_columns:
            new_name = mappings.get(col)
            if new_name is not None and new_name != col:
                raise InvalidConfigurationError(
                    f"field_mappings renames key column '{col}' to '{new_name}', "
                    f"which would break load's id derivation (key_columns must "
                    f"reference post-transform names). Set key_columns to the "
                    f"renamed field, or map a non-key field instead."
                )


class _ETLTransform(ITransformFunction):
    """Per-record transform step: apply field mappings then user callables.

    Wired into the ETL FSM's ``transform`` state as a registered function (the
    proven ``custom_functions=`` idiom). ``field_mappings`` rename keys; each
    entry of ``transformations`` is then applied in order. A transformation may
    be sync or async and is **map-style** — it receives the current record dict
    and must return the transformed dict. A non-dict return (including ``None``)
    raises a :class:`TransformError` rather than writing a corrupt row: the
    engine records the failed state and reports the record as a failure
    (``success=False``), so it is counted as an ``error`` (not ``loaded``) and
    counts against ``error_threshold``.
    """

    def __init__(
        self,
        field_mappings: Dict[str, str] | None,
        transformations: List[Callable] | None,
    ) -> None:
        self._field_mappings = field_mappings or {}
        self._transformations = transformations or []

    async def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        result = dict(data)
        for old_name, new_name in self._field_mappings.items():
            if old_name in result:
                result[new_name] = result.pop(old_name)
        for index, fn in enumerate(self._transformations):
            out = fn(result)
            if inspect.isawaitable(out):
                out = await out
            if not isinstance(out, dict):
                raise TransformError(
                    f"ETL transformation #{index} must return a dict, got "
                    f"{type(out).__name__}"
                )
            result = out
        return result

    def get_transform_description(self) -> str:
        return (
            f"Apply {len(self._field_mappings)} field mapping(s) and "
            f"{len(self._transformations)} transformation(s)"
        )


class DatabaseETL(StructuredConfigConsumer[ETLConfig]):
    """AsyncDatabase ETL pipeline using FSM pattern.

    Constructed from :class:`ETLConfig`: ``DatabaseETL(cfg)`` or
    ``DatabaseETL.from_config({...})`` (dict-dispatch). The config has
    required ``source_db`` / ``target_db`` fields, so an all-default
    ``DatabaseETL()`` is not valid.
    """

    CONFIG_CLS: ClassVar[type[ETLConfig]] = ETLConfig

    def _setup(self) -> None:
        self._fsm = self._build_fsm()
        self._checkpoint_data = {}
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        """Reset per-run metrics to zero.

        Called at the start of every ``run()`` so metrics do not accumulate
        across successive runs of the same pipeline instance.
        """
        self._metrics = {
            'extracted': 0,
            'transformed': 0,
            'loaded': 0,
            'errors': 0,
            'skipped': 0
        }

    def _build_fsm(self) -> AsyncSimpleFSM:
        """Build FSM for ETL pipeline."""
        # Build resources list. Only ``target_db`` is wired as an FSM resource:
        # it uses the async database resource so the DatabaseUpsert load
        # transform's ``await resource.upsert(...)`` is real and non-blocking
        # (the adapter opens its transport lazily on first use, not at
        # construction). source_db is deliberately NOT registered — the
        # ``extract`` start state is a passthrough and extraction is owned by
        # run()._extract_batches, so a registered source resource would only
        # eagerly open a (sync) backend at construction for no benefit.
        resources = [
            {'name': 'target_db', 'type': 'async_database', 'config': self.config.target_db}
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
        
        # Route through `enrich` only when enrichment is configured. `validate`
        # and `enrich` are currently honest passthroughs (their config
        # contracts — `validation_schema` / `enrichment_sources` — are
        # underspecified for real per-record wiring; see _build_custom_functions).
        post_transform = 'enrich' if self.config.enrichment_sources else 'load'

        # Create FSM configuration
        fsm_config = {
            'name': 'ETL_Pipeline',
            'data_mode': DataHandlingMode.COPY.value,  # Use COPY for data isolation
            'resources': resources,
            'states': [
                {
                    'name': 'extract',
                    'is_start': True,
                    'resources': []
                },
                {
                    'name': 'validate',
                    'resources': []
                },
                {
                    'name': 'transform',
                    'resources': [],
                    'functions': {
                        'transform': {'type': 'registered', 'name': 'transform'}
                    }
                },
                {
                    'name': 'enrich',
                    'resources': []
                },
                {
                    'name': 'load',
                    'resources': ['target_db'],
                    'functions': {
                        'transform': {'type': 'registered', 'name': 'load'}
                    }
                },
                {
                    'name': 'complete',
                    'is_end': True
                }
            ],
            'arcs': [
                {'from': 'extract', 'to': 'validate', 'name': 'extracted'},
                {'from': 'validate', 'to': 'transform', 'name': 'valid'},
                {'from': 'transform', 'to': post_transform, 'name': 'transformed'},
                {'from': 'enrich', 'to': 'load', 'name': 'enriched'},
                {'from': 'load', 'to': 'complete', 'name': 'loaded'},
            ]
        }

        # Wire the per-record functions through the proven custom_functions
        # idiom. The top-level config['functions'] dict is silently dropped by
        # FSMConfig (it has no 'functions' field); functions must flow through
        # AsyncSimpleFSM(config, custom_functions=...) and be referenced from a
        # state's 'functions' block (see examples/database_etl.py).
        return AsyncSimpleFSM(
            fsm_config,
            data_mode=DataHandlingMode.COPY,
            custom_functions=self._build_custom_functions(),
        )
        
    def _build_custom_functions(self) -> Dict[str, Callable]:
        """Build the registered functions the ETL FSM references by name.

        Two per-record steps are wired as FSM transforms:

        - ``transform`` (:class:`_ETLTransform`) applies ``field_mappings`` and
          the user ``transformations`` callables.
        - ``load`` (:class:`DatabaseUpsert`) upserts the record into the
          ``target_db`` async-database resource.

        Extraction is owned by ``run()._extract_batches`` (a per-record
        ``DatabaseFetch`` 'fetch all' would be nonsensical), so
        ``DatabaseFetch`` is repaired and exercised at the library layer
        rather than wired here. The ``validate`` / ``enrich`` states are honest
        passthroughs for now: ``validation_schema`` and ``enrichment_sources``
        have underspecified config contracts (schema format / per-record
        DB-API lookup) that need their own design before they can be wired
        without silently doing the wrong thing. The returned instances are
        passed via ``AsyncSimpleFSM(config, custom_functions=...)`` and
        referenced from each state's ``functions`` block.
        """
        return {
            'transform': _ETLTransform(
                self.config.field_mappings,
                self.config.transformations,
            ),
            'load': DatabaseUpsert(
                resource_name='target_db',
                table=self.config.target_table,
                key_columns=self.config.key_columns or ['id'],
            ),
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
        # Rebuild the FSM and reset metrics so each run() is independent. run()'s
        # finally closes the FSM, which clears its resource providers — reusing a
        # closed FSM would leave the load step with no target_db to upsert into.
        self._fsm = self._build_fsm()
        self._reset_metrics()

        # Everything that could fail after the FSM is built runs inside the try
        # so the finally always closes the freshly-rebuilt FSM — including a
        # checkpoint-load failure or source-open failure.
        source_db: AsyncDatabase | None = None
        try:
            # Resume from checkpoint if provided (after reset so checkpointed
            # metrics are not zeroed out).
            if checkpoint_id:
                await self._load_checkpoint(checkpoint_id)

            source_db = await AsyncDatabase.from_backend(
                self.config.source_db['type'],
                self.config.source_db  # type: ignore
            )

            # Determine extraction strategy
            if self.config.mode == ETLMode.INCREMENTAL:
                query = self._get_incremental_query()
            else:
                query = self.config.source_query or Query()
                
            # Process in batches
            async for batch in self._extract_batches(source_db, query):  # type: ignore
                # Process batch through FSM
                results = await self._fsm.process_batch(
                    data=batch,  # type: ignore
                    batch_size=self.config.batch_size,
                    max_workers=self.config.parallel_workers
                )
                
                # Update metrics
                self._update_metrics(results)
                
                # Check error threshold
                if self._check_error_threshold():
                    raise ETLError(f"Error threshold exceeded: {self._metrics['errors']} errors")
                    
                # Checkpoint if needed
                if self._should_checkpoint():
                    await self._save_checkpoint()
                    
        finally:
            # Close the source and the FSM independently so a failing source
            # close still flushes and closes the FSM's async target_db adapter
            # (durable persistence of upserted rows must not depend on the
            # source closing cleanly).
            if source_db is not None:
                try:
                    await source_db.close()
                except Exception:
                    logger.exception("ETL: error closing source database")
            await self._fsm.close()

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
        """Update execution metrics.

        A record only counts as ``transformed`` + ``loaded`` when it reached
        ``complete`` *and* the per-record execution reported success. A record
        whose ``transform`` or ``load`` step raised is reported as a failure by
        the engine (``success=False`` — see
        ``BaseExecutionEngine.finalize_single_result``), so it counts as an
        ``error`` and is NOT counted as loaded, even though it reached a final
        state. This keeps ``error_threshold`` honest: a target-write outage
        surfaces as errors rather than being silently reported as loaded.
        """
        for result in results:
            if result['success'] and result['final_state'] == 'complete':
                self._metrics['transformed'] += 1
                self._metrics['loaded'] += 1
            else:
                self._metrics['errors'] += 1

        # ``extracted`` is the count of records actually processed this run
        # (every record ends as either loaded or errored); it is recomputed
        # from outcomes rather than counted at the source.
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
        import hashlib
        import json
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

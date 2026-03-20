"""In-memory storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's memory backend.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from dataknobs_fsm.storage.base import StorageBackend, StorageConfig, StorageFactory
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage

if TYPE_CHECKING:
    from dataknobs_data.database import AsyncDatabase


class InMemoryStorage(UnifiedDatabaseStorage):
    """In-memory storage implementation using dataknobs_data's memory backend.

    Creates two separate ``AsyncMemoryDatabase`` instances (one for history
    records, one for step records) to avoid namespace collisions in the
    flat memory backend.  SQL-backed storages can safely share a single
    database because tables provide natural isolation.
    """

    def __init__(
        self,
        config: StorageConfig,
        *,
        database: AsyncDatabase | None = None,
        steps_database: AsyncDatabase | None = None,
    ):
        """Initialize in-memory storage.

        Args:
            config: Storage configuration.
            database: Optional pre-built AsyncDatabase instance.
            steps_database: Optional separate AsyncDatabase for step records.
        """
        # Copy config to avoid mutating the caller's object
        config = copy.copy(config)
        config.connection_params = dict(config.connection_params)
        config.mode_specific_config = dict(config.mode_specific_config)

        # Ensure we use the memory backend
        if 'type' not in config.connection_params:
            config.connection_params['type'] = 'memory'

        # Set memory-specific defaults
        if 'max_size' not in config.connection_params:
            config.connection_params['max_size'] = 1000

        # Enable indexing for fast queries
        if 'enable_indexing' not in config.connection_params:
            config.connection_params['enable_indexing'] = True

        # When no databases are injected, create separate instances so that
        # history and step records don't share a single namespace.  Without
        # this, load_steps() queries by execution_id and finds both history
        # records (which lack 'step_data') and step records, crashing with
        # KeyError.  (Bug B3)
        if database is None and steps_database is None:
            from dataknobs_data.backends.memory import AsyncMemoryDatabase
            database = AsyncMemoryDatabase()
            steps_database = AsyncMemoryDatabase()
            owns_databases = True
        else:
            owns_databases = None  # defer to parent's inference

        super().__init__(
            config,
            database=database,
            steps_database=steps_database,
            owns_databases=owns_databases,
        )


# Register memory backend
StorageFactory.register(StorageBackend.MEMORY, InMemoryStorage)

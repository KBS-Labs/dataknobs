"""In-memory storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's memory backend with sensible defaults.
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

    Sets memory-specific config defaults (backend type, max size, indexing).
    Record-type isolation when sharing a single database is handled by the
    base class via ``_history_query()`` / ``_steps_query()`` EXISTS filters.
    """

    def __init__(
        self,
        config: StorageConfig,
        *,
        database: AsyncDatabase | None = None,
        steps_database: AsyncDatabase | None = None,
        owns_databases: bool | None = None,
    ):
        """Initialize in-memory storage.

        Args:
            config: Storage configuration.
            database: Optional pre-built AsyncDatabase instance.
            steps_database: Optional separate AsyncDatabase for step records.
            owns_databases: Explicit ownership override (see base class).
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

        super().__init__(
            config,
            database=database,
            steps_database=steps_database,
            owns_databases=owns_databases,
        )


# Register memory backend
StorageFactory.register(StorageBackend.MEMORY, InMemoryStorage)

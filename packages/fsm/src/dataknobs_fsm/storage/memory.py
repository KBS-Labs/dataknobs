"""In-memory storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's memory backend with sensible defaults.
"""

from __future__ import annotations

from dataclasses import replace
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
        # Build a local working copy of the connection params and apply
        # memory-backend defaults, leaving the (immutable) caller config
        # untouched.
        #
        # Backend selection is now driven by ``StorageConfig.backend``
        # (the canonical enum), so no ``'type'`` injection is needed.
        # This class is registered for
        # ``StorageBackend.MEMORY`` and the parent's ``_setup_backend``
        # reads from the enum, so the memory backend is selected
        # automatically.
        params = dict(config.connection_params)

        # Set memory-specific defaults
        params.setdefault('max_size', 1000)

        # Enable indexing for fast queries
        params.setdefault('enable_indexing', True)

        config = replace(config, connection_params=params)

        super().__init__(
            config,
            database=database,
            steps_database=steps_database,
            owns_databases=owns_databases,
        )


# Register memory backend
StorageFactory.register(StorageBackend.MEMORY, InMemoryStorage)

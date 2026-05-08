"""File storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's file backend.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from dataknobs_fsm.storage.base import StorageBackend, StorageConfig, StorageFactory
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage

if TYPE_CHECKING:
    from dataknobs_data.database import AsyncDatabase


class FileStorage(UnifiedDatabaseStorage):
    """File storage implementation using dataknobs_data's file backend.

    This storage backend uses dataknobs_data's AsyncFileDatabase which
    stores records as JSON or YAML files with support for:
    - Directory-based organization
    - File rotation policies
    - Compression
    - Indexing via metadata files
    """

    def __init__(
        self,
        config: StorageConfig,
        *,
        database: AsyncDatabase | None = None,
        steps_database: AsyncDatabase | None = None,
    ):
        """Initialize file storage.

        Args:
            config: Storage configuration.
            database: Optional pre-built AsyncDatabase instance.
            steps_database: Optional separate AsyncDatabase for step records.
        """
        # Copy config to avoid mutating the caller's object
        config = copy.copy(config)
        config.connection_params = dict(config.connection_params)

        # Backend selection is now driven by ``StorageConfig.backend``
        # (the canonical enum), so no ``'type'`` injection is needed
        # — see Item 116.  This class is registered for
        # ``StorageBackend.FILE`` and the parent's ``_setup_backend``
        # reads from the enum, so the file backend is selected
        # automatically.

        # Set default file path if not provided
        if 'path' not in config.connection_params:
            config.connection_params['path'] = './fsm_history'

        # Set file format (json or yaml)
        if 'format' not in config.connection_params:
            config.connection_params['format'] = 'json'

        # Forward FSM ``StorageConfig.compression`` to the data
        # backend's ``compression`` config key (the established
        # ``AsyncFileDatabase`` API at backends/file.py — read at
        # ``self.config.get("compression", None)``).  An earlier
        # implementation injected ``'compress'`` here, which the data
        # backend silently ignored, so file storage was effectively
        # uncompressed regardless of ``StorageConfig.compression``.
        if 'compression' not in config.connection_params:
            config.connection_params['compression'] = (
                'gzip' if config.compression else None
            )

        super().__init__(config, database=database, steps_database=steps_database)


# Register file backend
StorageFactory.register(StorageBackend.FILE, FileStorage)

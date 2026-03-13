"""File storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's file backend.
"""

from __future__ import annotations

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
        # Ensure we use the file backend
        if 'type' not in config.connection_params:
            config.connection_params['type'] = 'file'

        # Set default file path if not provided
        if 'path' not in config.connection_params:
            config.connection_params['path'] = './fsm_history'

        # Set file format (json or yaml)
        if 'format' not in config.connection_params:
            config.connection_params['format'] = 'json'

        # Enable compression by default for file storage
        if 'compress' not in config.connection_params:
            config.connection_params['compress'] = config.compression

        super().__init__(config, database=database, steps_database=steps_database)


# Register file backend
StorageFactory.register(StorageBackend.FILE, FileStorage)

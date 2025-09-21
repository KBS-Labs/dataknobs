"""File storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's file backend.
"""

from dataknobs_fsm.storage.base import StorageBackend, StorageConfig, StorageFactory
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage


class FileStorage(UnifiedDatabaseStorage):
    """File storage implementation using dataknobs_data's file backend.
    
    This storage backend uses dataknobs_data's AsyncFileDatabase which
    stores records as JSON or YAML files with support for:
    - Directory-based organization
    - File rotation policies
    - Compression
    - Indexing via metadata files
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize file storage.
        
        Args:
            config: Storage configuration.
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
        
        super().__init__(config)


# Register file backend
StorageFactory.register(StorageBackend.FILE, FileStorage)

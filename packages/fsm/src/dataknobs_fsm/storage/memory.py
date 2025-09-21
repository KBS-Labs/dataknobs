"""In-memory storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's memory backend.
"""

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.storage.base import StorageBackend, StorageConfig, StorageFactory
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage


class InMemoryStorage(UnifiedDatabaseStorage):
    """In-memory storage implementation using dataknobs_data's memory backend.
    
    This storage backend uses dataknobs_data's AsyncMemoryDatabase which
    provides in-memory storage with support for:
    - LRU eviction based on max_size
    - Mode-specific compression for REFERENCE mode
    - Fast queries with in-memory indexing
    - Automatic cleanup of old entries
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize in-memory storage.
        
        Args:
            config: Storage configuration.
        """
        # Ensure we use the memory backend
        if 'type' not in config.connection_params:
            config.connection_params['type'] = 'memory'
        
        # Set memory-specific defaults
        if 'max_size' not in config.connection_params:
            config.connection_params['max_size'] = 1000
        
        # Enable indexing for fast queries
        if 'enable_indexing' not in config.connection_params:
            config.connection_params['enable_indexing'] = True
        
        # Configure mode-specific optimizations
        self._configure_mode_optimizations(config)
        
        super().__init__(config)
    
    def _configure_mode_optimizations(self, config: StorageConfig) -> None:
        """Configure mode-specific optimizations for memory storage.
        
        Args:
            config: Storage configuration to modify.
        """
        # For REFERENCE mode, enable compression by default
        if not config.mode_specific_config:
            config.mode_specific_config = {}
        
        if DataHandlingMode.REFERENCE not in config.mode_specific_config:
            config.mode_specific_config[DataHandlingMode.REFERENCE] = {
                'compress': True,
                'eviction_policy': 'lru',
                'cache_size': 100
            }
        
        # For DIRECT mode, use minimal storage
        if DataHandlingMode.DIRECT not in config.mode_specific_config:
            config.mode_specific_config[DataHandlingMode.DIRECT] = {
                'store_paths': False,
                'store_snapshots': False,
                'max_history': 10  # Keep only last 10 for DIRECT mode
            }
        
        # For COPY mode, full storage but with size limits
        if DataHandlingMode.COPY not in config.mode_specific_config:
            config.mode_specific_config[DataHandlingMode.COPY] = {
                'store_paths': True,
                'store_snapshots': True,
                'max_size_mb': 100  # Limit total size for COPY mode
            }


# Register memory backend
StorageFactory.register(StorageBackend.MEMORY, InMemoryStorage)

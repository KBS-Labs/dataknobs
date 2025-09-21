"""Base interfaces and classes for history storage."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Type

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory, ExecutionStep


class StorageBackend(Enum):
    """Available storage backends."""
    MEMORY = "memory"
    FILE = "file"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGODB = "mongodb"
    S3 = "s3"
    ELASTICSEARCH = "elasticsearch"


class StorageConfig:
    """Configuration for history storage."""
    
    def __init__(
        self,
        backend: StorageBackend = StorageBackend.MEMORY,
        connection_params: Dict[str, Any] | None = None,
        retention_policy: Dict[str, Any] | None = None,
        compression: bool = False,
        batch_size: int = 100,
        mode_specific_config: Dict[DataHandlingMode, Dict[str, Any]] | None = None
    ):
        """Initialize storage configuration.
        
        Args:
            backend: Storage backend to use.
            connection_params: Backend-specific connection parameters.
            retention_policy: Policy for data retention.
            compression: Whether to compress stored data.
            batch_size: Batch size for bulk operations.
            mode_specific_config: Configuration per data mode.
        """
        self.backend = backend
        self.connection_params = connection_params or {}
        self.retention_policy = retention_policy or {}
        self.compression = compression
        self.batch_size = batch_size
        self.mode_specific_config = mode_specific_config or {}
    
    def get_mode_config(self, mode: DataHandlingMode) -> Dict[str, Any]:
        """Get configuration for a specific data mode.
        
        Args:
            mode: Data mode.
            
        Returns:
            Configuration for that mode.
        """
        return self.mode_specific_config.get(mode, {})


class IHistoryStorage(ABC):
    """Interface for history storage backends."""
    
    @abstractmethod
    async def save_history(
        self,
        history: ExecutionHistory,
        metadata: Dict[str, Any] | None = None
    ) -> str:
        """Save execution history.
        
        Args:
            history: Execution history to save.
            metadata: Optional metadata.
            
        Returns:
            Storage ID for the saved history.
        """
        pass
    
    @abstractmethod
    async def load_history(
        self,
        history_id: str
    ) -> ExecutionHistory | None:
        """Load execution history by ID.
        
        Args:
            history_id: ID of the history to load.
            
        Returns:
            ExecutionHistory if found, None otherwise.
        """
        pass
    
    @abstractmethod
    async def save_step(
        self,
        execution_id: str,
        step: ExecutionStep,
        parent_id: str | None = None
    ) -> str:
        """Save a single execution step.
        
        Args:
            execution_id: Execution ID this step belongs to.
            step: Execution step to save.
            parent_id: Parent step ID if branching.
            
        Returns:
            Storage ID for the saved step.
        """
        pass
    
    @abstractmethod
    async def load_steps(
        self,
        execution_id: str,
        filters: Dict[str, Any] | None = None
    ) -> List[ExecutionStep]:
        """Load execution steps.
        
        Args:
            execution_id: Execution ID to load steps for.
            filters: Optional filters (e.g., state_name, status).
            
        Returns:
            List of execution steps.
        """
        pass
    
    @abstractmethod
    async def query_histories(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query execution histories.
        
        Args:
            filters: Query filters.
            limit: Maximum results to return.
            offset: Result offset for pagination.
            
        Returns:
            List of history summaries.
        """
        pass
    
    @abstractmethod
    async def delete_history(
        self,
        history_id: str
    ) -> bool:
        """Delete execution history.
        
        Args:
            history_id: ID of history to delete.
            
        Returns:
            True if deleted successfully.
        """
        pass
    
    @abstractmethod
    async def get_statistics(
        self,
        execution_id: str | None = None
    ) -> Dict[str, Any]:
        """Get storage statistics.
        
        Args:
            execution_id: Optional execution ID for specific stats.
            
        Returns:
            Storage statistics.
        """
        pass
    
    @abstractmethod
    async def cleanup(
        self,
        before_timestamp: float | None = None,
        keep_failed: bool = True
    ) -> int:
        """Clean up old histories.
        
        Args:
            before_timestamp: Delete histories before this timestamp.
            keep_failed: Whether to keep failed executions.
            
        Returns:
            Number of histories deleted.
        """
        pass


class BaseHistoryStorage(IHistoryStorage):
    """Base class for history storage implementations."""
    
    def __init__(self, config: StorageConfig):
        """Initialize storage with configuration.
        
        Args:
            config: Storage configuration.
        """
        self.config = config
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        if not self._initialized:
            await self._setup_backend()
            self._initialized = True
    
    @abstractmethod
    async def _setup_backend(self) -> None:
        """Set up the backend storage."""
        pass
    
    def _serialize_history(self, history: ExecutionHistory) -> Dict[str, Any]:
        """Serialize execution history for storage.
        
        Args:
            history: Execution history to serialize.
            
        Returns:
            Serialized history.
        """
        data = history.to_dict()
        
        # Apply compression if configured
        if self.config.compression:
            import zlib
            import json
            import base64
            
            json_str = json.dumps(data)
            compressed = zlib.compress(json_str.encode('utf-8'))
            data = {
                'compressed': True,
                'data': base64.b64encode(compressed).decode('utf-8')
            }
        
        return data
    
    def _deserialize_history(
        self,
        data: Dict[str, Any],
        fsm_name: str,
        execution_id: str
    ) -> ExecutionHistory:
        """Deserialize execution history from storage.
        
        Args:
            data: Serialized history data.
            fsm_name: FSM name.
            execution_id: Execution ID.
            
        Returns:
            ExecutionHistory instance.
        """
        # Decompress if needed
        if data.get('compressed'):
            import zlib
            import json
            import base64
            
            compressed = base64.b64decode(data['data'])
            json_str = zlib.decompress(compressed).decode('utf-8')
            data = json.loads(json_str)
        
        # Use ExecutionHistory.from_dict which properly reconstructs the tree
        return ExecutionHistory.from_dict(data)
    
    def _apply_retention_policy(self, histories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply retention policy to histories.
        
        Args:
            histories: List of history records.
            
        Returns:
            Filtered list based on retention policy.
        """
        if not self.config.retention_policy:
            return histories
        
        import time
        
        max_age = self.config.retention_policy.get('max_age_days')
        max_count = self.config.retention_policy.get('max_count')
        
        if max_age:
            cutoff = time.time() - (max_age * 86400)
            histories = [h for h in histories if h.get('timestamp', 0) > cutoff]
        
        if max_count and len(histories) > max_count:
            # Keep most recent
            histories = sorted(histories, key=lambda x: x.get('timestamp', 0), reverse=True)
            histories = histories[:max_count]
        
        return histories


class StorageFactory:
    """Factory for creating history storage instances."""
    
    _registry: Dict[StorageBackend, Type[IHistoryStorage]] = {}
    
    @classmethod
    def register(
        cls,
        backend: StorageBackend,
        storage_class: Type[IHistoryStorage]
    ) -> None:
        """Register a storage backend.
        
        Args:
            backend: Backend type.
            storage_class: Storage class to register.
        """
        cls._registry[backend] = storage_class
    
    @classmethod
    def create(
        cls,
        config: StorageConfig
    ) -> IHistoryStorage:
        """Create a storage instance.
        
        Args:
            config: Storage configuration.
            
        Returns:
            Storage instance.
            
        Raises:
            ValueError: If backend not registered.
        """
        storage_class = cls._registry.get(config.backend)
        if not storage_class:
            raise ValueError(f"Unknown storage backend: {config.backend}")
        
        return storage_class(config)  # type: ignore
    
    @classmethod
    def get_available_backends(cls) -> List[StorageBackend]:
        """Get list of available backends.
        
        Returns:
            List of registered backends.
        """
        return list(cls._registry.keys())

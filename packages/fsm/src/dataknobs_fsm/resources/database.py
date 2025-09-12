"""Database resource adapter for dataknobs_data backends."""

from contextlib import contextmanager
from typing import Any, Dict

from dataknobs_data.factory import DatabaseFactory
from dataknobs_data.database import SyncDatabase, AsyncDatabase
from dataknobs_data.records import Record
from dataknobs_data.query import Query

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceStatus,
)


class DatabaseResourceAdapter(BaseResourceProvider):
    """Adapter to use dataknobs_data databases as FSM resources.
    
    This adapter wraps dataknobs_data database backends to provide
    resource management capabilities for FSM states.
    """
    
    def __init__(
        self,
        name: str,
        backend: str = "memory",
        **backend_config
    ):
        """Initialize database resource adapter.
        
        Args:
            name: Resource name.
            backend: Database backend type (memory, file, postgres, sqlite, etc).
            **backend_config: Backend-specific configuration passed to DatabaseFactory.
        """
        config = {"backend": backend, **backend_config}
        super().__init__(name, config)
        
        self.backend = backend
        self.factory = DatabaseFactory()
        self._database: SyncDatabase | None = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the database backend."""
        try:
            # Create database using factory
            self._database = self.factory.create(**self.config)
            self.status = ResourceStatus.IDLE
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to initialize database backend '{self.backend}': {e}",
                resource_name=self.name,
                operation="initialize"
            ) from e
    
    def acquire(self, **kwargs) -> SyncDatabase:
        """Acquire database connection/instance.
        
        The returned database object can be used for all database operations.
        For backends that support connection pooling (postgres, etc), this
        manages the underlying connections transparently.
        
        Args:
            **kwargs: Additional parameters (unused, for interface compatibility).
            
        Returns:
            Database instance for operations.
            
        Raises:
            ResourceError: If acquisition fails.
        """
        if self._database is None:
            raise ResourceError(
                "Database not initialized",
                resource_name=self.name,
                operation="acquire"
            )
        
        try:
            # For most backends, we return the same instance
            # Connection pooling is handled internally by the backend
            self.status = ResourceStatus.ACTIVE
            self._resources.append(self._database)
            return self._database
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to acquire database resource: {e}",
                resource_name=self.name,
                operation="acquire"
            ) from e
    
    def release(self, resource: Any) -> None:
        """Release database resource.
        
        For pooled backends, this returns connections to the pool.
        For non-pooled backends, this is a no-op.
        
        Args:
            resource: The database resource to release.
        """
        if resource in self._resources:
            self._resources.remove(resource)
        
        if not self._resources:
            self.status = ResourceStatus.IDLE
        
        # Most backends handle connection cleanup internally
        # We don't need to do anything special here
    
    def validate(self, resource: Any) -> bool:
        """Validate database resource is still usable.
        
        Args:
            resource: The database resource to validate.
            
        Returns:
            True if the resource is valid and usable.
        """
        if resource is None or not isinstance(resource, (SyncDatabase, AsyncDatabase)):
            return False
        
        try:
            # Try a simple operation to validate the connection
            # This will vary by backend but should be lightweight
            if hasattr(resource, 'count'):
                # Try to count records (should return quickly even if 0)
                _ = resource.count()
            return True
        except Exception:
            return False
    
    def health_check(self) -> ResourceHealth:
        """Check database health.
        
        Returns:
            Health status of the database backend.
        """
        if self._database is None:
            self.metrics.record_health_check(False)
            return ResourceHealth.UNKNOWN
        
        try:
            # Perform a simple health check operation
            valid = self.validate(self._database)
            
            if valid:
                self.metrics.record_health_check(True)
                return ResourceHealth.HEALTHY
            else:
                self.metrics.record_health_check(False)
                return ResourceHealth.UNHEALTHY
        except Exception:
            self.metrics.record_health_check(False)
            return ResourceHealth.UNHEALTHY
    
    @contextmanager
    def transaction_context(self, database: SyncDatabase | None = None):
        """Context manager for database transactions.
        
        Note: Transaction support depends on the backend.
        Some backends (memory, file) may not support true transactions.
        
        Args:
            database: Optional database instance to use.
            
        Yields:
            Database instance for operations within transaction.
        """
        if database is None:
            database = self.acquire()
            should_release = True
        else:
            should_release = False
        
        try:
            # For backends that support transactions, we could add
            # transaction begin/commit/rollback logic here
            # For now, we just ensure proper resource cleanup
            yield database
        finally:
            if should_release:
                self.release(database)
    
    def close(self) -> None:
        """Close the database resource and clean up."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Release all tracked resources first
        super().close()
        
        # Close the database backend if it has a close method
        if self._database and hasattr(self._database, 'close'):
            try:
                # Attempt to flush any pending operations
                if hasattr(self._database, 'flush'):
                    self._database.flush()
                
                # Close the connection
                self._database.close()
                logger.debug(f"Successfully closed database connection for {self.name}")
            except AttributeError as e:
                logger.warning(f"Database {self.name} missing expected close method: {e}")
            except Exception as e:
                logger.error(f"Error closing database {self.name}: {e}")
                # Store error for debugging but don't re-raise
                if not hasattr(self, '_cleanup_errors'):
                    self._cleanup_errors = []
                self._cleanup_errors.append(f"Database close error: {e}")
        
        self._database = None
    
    # Convenience methods that delegate to the database
    
    def create(self, record: Record, database: SyncDatabase | None = None) -> str:
        """Create a record in the database.
        
        Args:
            record: Record to create.
            database: Optional database instance.
            
        Returns:
            ID of the created record.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="create")
        
        return database.create(record)
    
    def read(self, record_id: str, database: SyncDatabase | None = None) -> Record | None:
        """Read a record from the database.
        
        Args:
            record_id: ID of the record to read.
            database: Optional database instance.
            
        Returns:
            The record if found, None otherwise.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="read")
        
        return database.read(record_id)
    
    def update(self, record_id: str, record: Record, database: SyncDatabase | None = None) -> bool:
        """Update a record in the database.
        
        Args:
            record_id: ID of the record to update.
            record: Record with updates.
            database: Optional database instance.
            
        Returns:
            True if update was successful.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="update")
        
        return database.update(record_id, record)
    
    def delete(self, record_id: str, database: SyncDatabase | None = None) -> bool:
        """Delete a record from the database.
        
        Args:
            record_id: ID of the record to delete.
            database: Optional database instance.
            
        Returns:
            True if deletion was successful.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="delete")
        
        return database.delete(record_id)
    
    def search(self, query: Query, database: SyncDatabase | None = None) -> list[Record]:
        """Search for records in the database.
        
        Args:
            query: Search query.
            database: Optional database instance.
            
        Returns:
            List of matching records.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="search")
        
        return database.search(query)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the database backend.
        
        Returns:
            Backend information including capabilities.
        """
        if self.factory:
            return self.factory.get_backend_info(self.backend)
        return {"backend": self.backend, "status": self.status.value}

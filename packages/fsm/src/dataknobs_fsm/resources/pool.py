"""Resource pooling implementation."""

import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import (
    IResourceProvider,
    ResourceMetrics,
)


@dataclass
class PoolConfig:
    """Configuration for resource pools."""
    
    min_size: int = 1
    max_size: int = 10
    acquire_timeout: float = 30.0
    idle_timeout: float = 300.0
    validation_interval: float = 60.0
    max_lifetime: float = 3600.0


@dataclass
class PooledResource:
    """A resource in a pool with metadata."""
    
    resource: Any
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    
    def is_expired(self, max_lifetime: float) -> bool:
        """Check if resource has exceeded max lifetime.
        
        Args:
            max_lifetime: Maximum lifetime in seconds.
            
        Returns:
            True if expired.
        """
        age = (datetime.now() - self.created_at).total_seconds()
        return age > max_lifetime
    
    def is_idle_too_long(self, idle_timeout: float) -> bool:
        """Check if resource has been idle too long.
        
        Args:
            idle_timeout: Maximum idle time in seconds.
            
        Returns:
            True if idle too long.
        """
        idle_time = (datetime.now() - self.last_used).total_seconds()
        return idle_time > idle_timeout


class ResourcePool:
    """Thread-safe resource pool implementation."""
    
    def __init__(
        self,
        provider: IResourceProvider,
        config: PoolConfig | None = None
    ):
        """Initialize the pool.
        
        Args:
            provider: Resource provider.
            config: Pool configuration.
        """
        self.provider = provider
        self.config = config or PoolConfig()
        self.metrics = ResourceMetrics()
        
        self._pool = queue.Queue(maxsize=self.config.max_size)
        self._active_resources = set()
        self._lock = threading.RLock()
        self._closed = False
        self._resource_map = {}  # Maps resource to PooledResource
        
        # Initialize minimum resources
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the pool with minimum resources."""
        for _ in range(self.config.min_size):
            try:
                resource = self.provider.acquire()
                pooled = PooledResource(
                    resource=resource,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                self._resource_map[id(resource)] = pooled
                self._pool.put(pooled)
            except Exception:
                pass  # Continue with fewer resources
    
    def acquire(self, timeout: float | None = None) -> Any:
        """Acquire a resource from the pool.
        
        Args:
            timeout: Acquisition timeout in seconds.
            
        Returns:
            The acquired resource.
            
        Raises:
            ResourceError: If acquisition fails.
        """
        if self._closed:
            raise ResourceError("Pool is closed", resource_name=self.provider.name, operation="acquire")
        
        timeout = timeout or self.config.acquire_timeout
        start_time = datetime.now()
        
        # Try to get from pool
        try:
            pooled = self._pool.get(timeout=timeout)
            
            # Validate the resource
            if not self._validate_pooled_resource(pooled):
                # Resource is invalid, create a new one
                self._release_pooled_resource(pooled)
                return self._create_new_resource()
            
            # Update metadata
            pooled.last_used = datetime.now()
            pooled.use_count += 1
            
            with self._lock:
                self._active_resources.add(id(pooled.resource))
            
            # Track acquisition time
            acquisition_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_acquisition(acquisition_time)
            return pooled.resource
            
        except queue.Empty:
            # Pool is empty, try to create new resource if under max
            with self._lock:
                if len(self._active_resources) < self.config.max_size:
                    resource = self._create_new_resource()
                    # Track acquisition time for newly created resource
                    acquisition_time = (datetime.now() - start_time).total_seconds()
                    self.metrics.record_acquisition(acquisition_time)
                    return resource
            
            # Record timeout event
            self.metrics.record_timeout()
            
            raise ResourceError(
                f"Failed to acquire resource within {timeout} seconds",
                resource_name=self.provider.name,
                operation="acquire"
            ) from None
    
    def release(self, resource: Any) -> None:
        """Return a resource to the pool.
        
        Args:
            resource: The resource to return.
        """
        if self._closed:
            # Pool is closed, just release the resource
            self.provider.release(resource)
            return
        
        resource_id = id(resource)
        
        with self._lock:
            if resource_id not in self._active_resources:
                return  # Resource not from this pool
            
            self._active_resources.discard(resource_id)
            pooled = self._resource_map.get(resource_id)
            
            if pooled is None:
                # Create new pooled resource wrapper
                pooled = PooledResource(
                    resource=resource,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                self._resource_map[resource_id] = pooled
            
            # Check if resource should be retired
            if (pooled.is_expired(self.config.max_lifetime) or
                not self.provider.validate(resource)):
                self._release_pooled_resource(pooled)
            else:
                # Return to pool
                pooled.last_used = datetime.now()
                try:
                    self._pool.put_nowait(pooled)
                except queue.Full:
                    # Pool is full, release the resource
                    self._release_pooled_resource(pooled)
        
        hold_time = 0.1  # Default hold time
        self.metrics.record_release(hold_time)
    
    def _validate_pooled_resource(self, pooled: PooledResource) -> bool:
        """Validate a pooled resource.
        
        Args:
            pooled: The pooled resource to validate.
            
        Returns:
            True if valid.
        """
        if pooled.is_expired(self.config.max_lifetime):
            return False
        
        if pooled.is_idle_too_long(self.config.idle_timeout):
            return False
        
        return self.provider.validate(pooled.resource)
    
    def _create_new_resource(self) -> Any:
        """Create a new resource.
        
        Returns:
            The new resource.
            
        Raises:
            ResourceError: If creation fails.
        """
        try:
            resource = self.provider.acquire()
            pooled = PooledResource(
                resource=resource,
                created_at=datetime.now(),
                last_used=datetime.now(),
                use_count=1
            )
            
            with self._lock:
                self._resource_map[id(resource)] = pooled
                self._active_resources.add(id(resource))
            
            # Note: Acquisition metrics are now tracked in acquire() method with timing
            return resource
            
        except Exception as e:
            self.metrics.record_failure()
            raise ResourceError(
                f"Failed to create resource: {e}",
                resource_name=self.provider.name,
                operation="create"
            ) from e
    
    def _release_pooled_resource(self, pooled: PooledResource) -> None:
        """Release a pooled resource.
        
        Args:
            pooled: The pooled resource to release.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        resource_id = id(pooled.resource)
        try:
            # Attempt to properly release the resource
            self.provider.release(pooled.resource)
            logger.debug(f"Successfully released pooled resource {resource_id} from pool {self.provider.name}")
        except AttributeError as e:
            logger.warning(f"Resource provider {self.provider.name} missing release method: {e}")
        except Exception as e:
            logger.error(f"Error releasing pooled resource {resource_id} from pool {self.provider.name}: {e}")
            # Track the error for debugging
            self.metrics.record_failure()
            # Still remove from map even if release failed to prevent memory leak
        
        with self._lock:
            self._resource_map.pop(resource_id, None)
    
    def size(self) -> int:
        """Get the current pool size.
        
        Returns:
            Total number of resources (active + idle).
        """
        with self._lock:
            return self._pool.qsize() + len(self._active_resources)
    
    def available(self) -> int:
        """Get the number of available resources.
        
        Returns:
            Number of idle resources in pool.
        """
        return self._pool.qsize()
    
    def close(self) -> None:
        """Close the pool and release all resources."""
        self._closed = True
        
        # Release active resources
        with self._lock:
            for resource_id in list(self._active_resources):
                pooled = self._resource_map.get(resource_id)
                if pooled:
                    self._release_pooled_resource(pooled)
            self._active_resources.clear()
        
        # Release pooled resources
        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                self._release_pooled_resource(pooled)
            except queue.Empty:
                break
        
        self._resource_map.clear()
    
    def evict_idle(self) -> int:
        """Evict idle resources that have exceeded timeout.
        
        Returns:
            Number of resources evicted.
        """
        evicted = 0
        temp_resources = []
        
        # Check all pooled resources
        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                if pooled.is_idle_too_long(self.config.idle_timeout):
                    self._release_pooled_resource(pooled)
                    evicted += 1
                else:
                    temp_resources.append(pooled)
            except queue.Empty:
                break
        
        # Put back the non-evicted resources
        for pooled in temp_resources:
            try:
                self._pool.put_nowait(pooled)
            except queue.Full:
                self._release_pooled_resource(pooled)
        
        return evicted
    
    def get_metrics(self) -> ResourceMetrics:
        """Get pool metrics.
        
        Returns:
            Current metrics.
        """
        return self.metrics

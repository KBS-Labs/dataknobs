"""Central resource manager for FSM."""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Set

from dataknobs_fsm.functions.base import ResourceError, ResourceConfig
from dataknobs_fsm.resources.base import (
    IResourceProvider,
    IResourcePool,
    ResourceStatus,
    ResourceHealth,
    ResourceMetrics,
)
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig


class ResourceManager:
    """Manages resources across the FSM system."""
    
    def __init__(self):
        """Initialize the resource manager."""
        self._providers: Dict[str, IResourceProvider] = {}
        self._pools: Dict[str, IResourcePool] = {}
        self._resources: Dict[str, Any] = {}
        self._resource_owners: Dict[str, Set[str]] = {}  # resource_name -> owner_ids
        self._lock = threading.RLock()
        self._closed = False
    
    def register_provider(
        self,
        name: str,
        provider: IResourceProvider,
        pool_config: Optional[PoolConfig] = None
    ) -> None:
        """Register a resource provider.
        
        Args:
            name: Resource name.
            provider: Resource provider.
            pool_config: Optional pool configuration.
        """
        with self._lock:
            if name in self._providers:
                raise ValueError(f"Provider '{name}' already registered")
            
            self._providers[name] = provider
            
            if pool_config:
                # Create a pool for this provider
                pool = ResourcePool(provider, pool_config)
                self._pools[name] = pool
    
    def unregister_provider(self, name: str) -> None:
        """Unregister a resource provider.
        
        Args:
            name: Resource name.
        """
        with self._lock:
            # Close pool if exists
            if name in self._pools:
                self._pools[name].close()
                del self._pools[name]
            
            # Remove provider
            if name in self._providers:
                provider = self._providers[name]
                if hasattr(provider, 'close'):
                    provider.close()
                del self._providers[name]
    
    def acquire(
        self,
        name: str,
        owner_id: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Acquire a resource.
        
        Args:
            name: Resource name.
            owner_id: ID of the owner (e.g., state instance ID).
            timeout: Acquisition timeout.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            The acquired resource.
            
        Raises:
            ResourceError: If acquisition fails.
        """
        if self._closed:
            raise ResourceError("Resource manager is closed", resource_name=name, operation="acquire")
        
        # First check if provider exists and if owner already has resource
        with self._lock:
            if name not in self._providers:
                raise ResourceError(
                    f"Unknown resource '{name}'",
                    resource_name=name,
                    operation="acquire"
                )
            
            # Check if owner already has this resource
            owner_key = f"{owner_id}:{name}"
            if owner_key in self._resources:
                return self._resources[owner_key]
            
            # Check if we have a pool for this resource
            has_pool = name in self._pools
        
        # Acquire resource outside of lock to prevent deadlock
        if has_pool:
            resource = self._pools[name].acquire(timeout)
        else:
            resource = self._providers[name].acquire(**kwargs)
        
        # Re-acquire lock to track ownership
        with self._lock:
            # Double-check that owner doesn't have resource (race condition check)
            owner_key = f"{owner_id}:{name}"
            if owner_key in self._resources:
                # Another thread already acquired for this owner, release the extra
                if has_pool:
                    self._pools[name].release(resource)
                else:
                    self._providers[name].release(resource)
                return self._resources[owner_key]
            
            # Track ownership
            self._resources[owner_key] = resource
            if name not in self._resource_owners:
                self._resource_owners[name] = set()
            self._resource_owners[name].add(owner_id)
            
            return resource
    
    def release(self, name: str, owner_id: str) -> None:
        """Release a resource.
        
        Args:
            name: Resource name.
            owner_id: ID of the owner.
        """
        with self._lock:
            owner_key = f"{owner_id}:{name}"
            
            if owner_key not in self._resources:
                return  # Resource not acquired or already released
            
            resource = self._resources[owner_key]
            
            # Release to pool or provider
            if name in self._pools:
                self._pools[name].release(resource)
            elif name in self._providers:
                self._providers[name].release(resource)
            
            # Clean up tracking
            del self._resources[owner_key]
            if name in self._resource_owners:
                self._resource_owners[name].discard(owner_id)
    
    def release_all(self, owner_id: str) -> None:
        """Release all resources owned by an owner.
        
        Args:
            owner_id: ID of the owner.
        """
        with self._lock:
            # Find all resources owned by this owner
            owner_resources = []
            for key in list(self._resources.keys()):
                if key.startswith(f"{owner_id}:"):
                    resource_name = key.split(":", 1)[1]
                    owner_resources.append(resource_name)
            
            # Release them all
            for resource_name in owner_resources:
                self.release(resource_name, owner_id)
    
    def get_resource(self, name: str, owner_id: str) -> Optional[Any]:
        """Get an acquired resource.
        
        Args:
            name: Resource name.
            owner_id: ID of the owner.
            
        Returns:
            The resource if acquired, None otherwise.
        """
        owner_key = f"{owner_id}:{name}"
        return self._resources.get(owner_key)
    
    def has_resource(self, name: str, owner_id: str) -> bool:
        """Check if an owner has acquired a resource.
        
        Args:
            name: Resource name.
            owner_id: ID of the owner.
            
        Returns:
            True if the owner has the resource.
        """
        owner_key = f"{owner_id}:{name}"
        return owner_key in self._resources
    
    def validate_resource(self, name: str) -> bool:
        """Validate a resource provider.
        
        Args:
            name: Resource name.
            
        Returns:
            True if the resource is valid.
        """
        with self._lock:
            if name not in self._providers:
                return False
            
            # Create a test resource to validate
            try:
                resource = self._providers[name].acquire()
                valid = self._providers[name].validate(resource)
                self._providers[name].release(resource)
                return valid
            except Exception:
                return False
    
    def health_check(self, name: Optional[str] = None) -> Dict[str, ResourceHealth]:
        """Check health of resources.
        
        Args:
            name: Optional specific resource name.
            
        Returns:
            Health status by resource name.
        """
        with self._lock:
            if name:
                if name in self._providers:
                    return {name: self._providers[name].health_check()}
                else:
                    return {name: ResourceHealth.UNKNOWN}
            
            # Check all resources
            health_status = {}
            for resource_name, provider in self._providers.items():
                try:
                    health_status[resource_name] = provider.health_check()
                except Exception:
                    health_status[resource_name] = ResourceHealth.UNKNOWN
            
            return health_status
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, ResourceMetrics]:
        """Get resource metrics.
        
        Args:
            name: Optional specific resource name.
            
        Returns:
            Metrics by resource name.
        """
        with self._lock:
            if name:
                metrics = {}
                if name in self._providers:
                    metrics[name] = self._providers[name].get_metrics()
                if name in self._pools:
                    metrics[f"{name}_pool"] = self._pools[name].get_metrics()
                return metrics
            
            # Get all metrics
            all_metrics = {}
            for resource_name, provider in self._providers.items():
                all_metrics[resource_name] = provider.get_metrics()
            for resource_name, pool in self._pools.items():
                all_metrics[f"{resource_name}_pool"] = pool.get_metrics()
            
            return all_metrics
    
    @contextmanager
    def resource_context(
        self,
        name: str,
        owner_id: str,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """Context manager for resource acquisition.
        
        Args:
            name: Resource name.
            owner_id: ID of the owner.
            timeout: Acquisition timeout.
            **kwargs: Additional parameters.
            
        Yields:
            The acquired resource.
        """
        resource = self.acquire(name, owner_id, timeout, **kwargs)
        try:
            yield resource
        finally:
            self.release(name, owner_id)
    
    def configure_from_requirements(
        self,
        requirements: list[ResourceConfig],
        owner_id: str
    ) -> Dict[str, Any]:
        """Configure resources from requirements.
        
        Args:
            requirements: List of resource configurations.
            owner_id: ID of the owner.
            
        Returns:
            Dictionary of acquired resources.
            
        Raises:
            ResourceError: If any resource cannot be acquired.
        """
        acquired = {}
        
        try:
            for config in requirements:
                resource = self.acquire(
                    config.name,
                    owner_id,
                    timeout=config.timeout
                )
                acquired[config.name] = resource
            
            return acquired
            
        except Exception as e:
            # Release any acquired resources on failure
            for name in acquired:
                try:
                    self.release(name, owner_id)
                except Exception:
                    pass
            raise ResourceError(f"Failed to acquire resources: {e}", resource_name="multiple", operation="configure")
    
    def close(self) -> None:
        """Close the resource manager and release all resources."""
        self._closed = True
        
        with self._lock:
            # Release all acquired resources
            for owner_id in set(key.split(":")[0] for key in self._resources.keys()):
                self.release_all(owner_id)
            
            # Close all pools
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()
            
            # Close all providers
            for provider in self._providers.values():
                if hasattr(provider, 'close'):
                    provider.close()
            self._providers.clear()
            
            self._resources.clear()
            self._resource_owners.clear()
    
    async def cleanup(self) -> None:
        """Async cleanup of all resource providers.
        
        This is an alias for close() but provides async interface for compatibility.
        """
        # For now, just call the sync close method
        # In the future, this could be made truly async
        self.close()
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
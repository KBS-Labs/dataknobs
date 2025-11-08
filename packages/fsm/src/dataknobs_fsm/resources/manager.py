"""Central resource manager for FSM."""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Set

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
        pool_config: PoolConfig | None = None
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

    def get_provider(self, name: str) -> IResourceProvider | None:
        """Get a resource provider by name.

        Args:
            name: Resource name.

        Returns:
            The resource provider or None if not found.
        """
        with self._lock:
            return self._providers.get(name)

    def get_all_providers(self) -> Dict[str, IResourceProvider]:
        """Get all registered resource providers.

        Returns:
            Dictionary of resource name to provider.
        """
        with self._lock:
            return dict(self._providers)
    
    def acquire(
        self,
        name: str,
        owner_id: str,
        timeout: float | None = None,
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
    
    def get_resource(self, name: str, owner_id: str) -> Any | None:
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
    
    def health_check(self, name: str | None = None) -> Dict[str, ResourceHealth]:
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
    
    def get_metrics(self, name: str | None = None) -> Dict[str, ResourceMetrics]:
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
        timeout: float | None = None,
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
            raise ResourceError(f"Failed to acquire resources: {e}", resource_name="multiple", operation="configure") from e
    
    def close(self) -> None:
        """Close the resource manager and release all resources."""
        self._closed = True
        
        with self._lock:
            # Release all acquired resources
            for owner_id in {key.split(":")[0] for key in self._resources.keys()}:
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
        
        This method performs async cleanup of resources that support it,
        while falling back to sync cleanup for those that don't.
        """
        import asyncio
        import logging
        logger = logging.getLogger(__name__)
        
        cleanup_tasks = []
        sync_providers = []
        
        # Separate async and sync providers
        for name, provider in self._providers.items():
            if hasattr(provider, 'aclose') or hasattr(provider, 'cleanup'):
                # Provider has async cleanup method
                if hasattr(provider, 'aclose'):
                    cleanup_tasks.append(self._async_close_provider(name, provider))
                elif hasattr(provider, 'cleanup'):
                    cleanup_tasks.append(self._async_cleanup_provider(name, provider))
            else:
                # Provider only has sync cleanup
                sync_providers.append((name, provider))
        
        # Run async cleanups concurrently
        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error during async cleanup (task {i}): {result}")
        
        # Run sync cleanups in executor to avoid blocking
        if sync_providers:
            loop = asyncio.get_event_loop()
            for name, provider in sync_providers:
                try:
                    await loop.run_in_executor(None, self._close_provider, name, provider)
                except Exception as e:
                    logger.error(f"Error closing sync provider {name}: {e}")
        
        # Clear tracking data
        with self._lock:
            self._resources.clear()
            self._resource_owners.clear()
            self._pools.clear()
            self._providers.clear()
    
    async def _async_close_provider(self, name: str, provider: IResourceProvider) -> None:
        """Close a provider that has an async close method.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            await provider.aclose()
            logger.debug(f"Successfully closed async provider {name}")
        except Exception as e:
            logger.error(f"Error closing async provider {name}: {e}")
            raise
    
    async def _async_cleanup_provider(self, name: str, provider: IResourceProvider) -> None:
        """Clean up a provider that has an async cleanup method.
        
        Args:
            name: Provider name  
            provider: Provider instance
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            await provider.cleanup()
            logger.debug(f"Successfully cleaned up async provider {name}")
        except Exception as e:
            logger.error(f"Error cleaning up async provider {name}: {e}")
            raise
    
    def _close_provider(self, name: str, provider: IResourceProvider) -> None:
        """Close a sync provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if hasattr(provider, 'close'):
                provider.close()
                logger.debug(f"Successfully closed sync provider {name}")
        except Exception as e:
            logger.error(f"Error closing sync provider {name}: {e}")
    
    def create_provider_from_dict(self, name: str, config: Dict[str, Any]) -> IResourceProvider:
        """Create a resource provider from a dictionary configuration.
        
        Args:
            name: Resource name
            config: Dictionary configuration for the resource
            
        Returns:
            Resource provider instance
        """
        # Create a simple in-memory resource provider
        class SimpleResourceProvider(IResourceProvider):
            """Simple in-memory resource provider for testing and basic use cases.

            Provides a lightweight resource provider that stores data in memory
            from configuration. Useful for testing FSMs without external dependencies
            or for simple static data resources.
            """

            def __init__(self, name: str, config: Dict[str, Any]):
                self.name = name
                self.config = config
                self.data = config.get('data', {})
                self._status = ResourceStatus.IDLE
            
            def acquire(self, **kwargs) -> Any:
                self._status = ResourceStatus.BUSY
                return self.data
            
            def release(self, resource: Any) -> None:
                self._status = ResourceStatus.IDLE
            
            def validate(self, resource: Any) -> bool:
                return resource is not None
            
            def health_check(self) -> ResourceHealth:
                return ResourceHealth.HEALTHY
            
            def get_metrics(self) -> ResourceMetrics:
                return ResourceMetrics(
                    total_acquisitions=0,
                    active_connections=1 if self._status == ResourceStatus.BUSY else 0,
                    failed_acquisitions=0
                )
            
            async def get_resource(self):
                return self.data
            
            async def close(self):
                pass
        
        return SimpleResourceProvider(name, config)
    
    def create_simple_provider(self, name: str, data: Any) -> IResourceProvider:
        """Create a simple resource provider with static data.
        
        Args:
            name: Resource name
            data: The resource data to provide
            
        Returns:
            Resource provider instance
        """
        return self.create_provider_from_dict(name, {'data': data})
    
    def register_from_dict(self, name: str, config: Dict[str, Any]) -> None:
        """Register a resource provider from a dictionary configuration.
        
        Args:
            name: Resource name
            config: Dictionary configuration for the resource
        """
        provider = self.create_provider_from_dict(name, config)
        self.register_provider(name, provider)
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
    
    def get_resource_status(self, name: str) -> Dict[str, Any]:
        """Get status information for a resource.
        
        Args:
            name: Resource name.
            
        Returns:
            Status dictionary with provider and pool information.
        """
        with self._lock:
            status = {
                "provider_exists": name in self._providers,
                "has_pool": name in self._pools,
                "active_count": 0,
                "owners": list(self._resource_owners.get(name, set()))
            }
            
            if name in self._providers:
                try:
                    metrics = self._providers[name].get_metrics()
                    status["active_count"] = metrics.active_connections
                    status["total_acquires"] = metrics.total_acquisitions
                    status["total_releases"] = metrics.total_acquisitions - metrics.active_connections
                except Exception:
                    pass
            
            return status
    
    def get_all_resources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered resources.
        
        Returns:
            Dictionary mapping resource names to their status.
        """
        with self._lock:
            all_resources = {}
            for name in self._providers:
                all_resources[name] = self.get_resource_status(name)
            return all_resources
    
    def get_resource_owners(self, name: str) -> Set[str]:
        """Get all owners of a specific resource.
        
        Args:
            name: Resource name.
            
        Returns:
            Set of owner IDs.
        """
        with self._lock:
            return self._resource_owners.get(name, set()).copy()

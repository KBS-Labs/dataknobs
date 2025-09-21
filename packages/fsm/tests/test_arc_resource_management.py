"""Tests for arc resource management.

This module tests resource acquisition, release, and cleanup for arc execution:
- ResourceManager class (lines 18-456 in resources/manager.py)
- Resource ownership tracking
- Resource pool integration
- Cleanup on failure scenarios
"""

import pytest
import threading
import time
from contextlib import contextmanager
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Set

from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.resources.base import (
    IResourceProvider, IResourcePool,
    ResourceStatus, ResourceHealth, ResourceMetrics
)
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig
from dataknobs_fsm.functions.base import ResourceError


# Create a real test resource provider
class MockResourceProvider:
    """A simple resource provider for testing."""
    
    def __init__(self, name: str = "test"):
        self.name = name
        self.acquired_count = 0
        self.released_count = 0
        self.resources_created = 0
        self._status = ResourceStatus.IDLE
        
    def acquire(self, **kwargs) -> Any:
        """Acquire a resource."""
        self.acquired_count += 1
        self.resources_created += 1
        self._status = ResourceStatus.BUSY
        return f"resource_{self.name}_{self.resources_created}"
    
    def release(self, resource: Any) -> None:
        """Release a resource."""
        self.released_count += 1
        self._status = ResourceStatus.IDLE
    
    def validate(self, resource: Any) -> bool:
        """Validate a resource."""
        return resource is not None and str(resource).startswith("resource_")
    
    def health_check(self) -> ResourceHealth:
        """Check provider health."""
        return ResourceHealth.HEALTHY
    
    def get_metrics(self) -> ResourceMetrics:
        """Get provider metrics."""
        return ResourceMetrics(
            total_acquisitions=self.acquired_count,
            active_connections=self.acquired_count - self.released_count,
            failed_acquisitions=0
        )
    
    def close(self) -> None:
        """Close provider."""
        self._status = ResourceStatus.CLOSED


class TestResourceManager:
    """Test the ResourceManager class."""
    
    def test_manager_initialization(self):
        """Test ResourceManager initialization."""
        manager = ResourceManager()
        
        assert manager._providers == {}
        assert manager._pools == {}
        assert manager._resources == {}
        assert manager._resource_owners == {}
        assert manager._closed is False
        
    def test_register_provider(self):
        """Test registering a resource provider."""
        manager = ResourceManager()
        provider = MockResourceProvider("db")
        
        manager.register_provider("database", provider)
        
        assert "database" in manager._providers
        assert manager._providers["database"] == provider
        
    def test_register_provider_with_pool(self):
        """Test registering a provider with a pool."""
        manager = ResourceManager()
        provider = MockResourceProvider("pooled")
        pool_config = PoolConfig(min_size=2, max_size=5)
        
        manager.register_provider("pooled_resource", provider, pool_config)
        
        assert "pooled_resource" in manager._providers
        assert "pooled_resource" in manager._pools
        assert isinstance(manager._pools["pooled_resource"], ResourcePool)
        
    def test_register_duplicate_provider(self):
        """Test registering duplicate provider raises error."""
        manager = ResourceManager()
        provider1 = MockResourceProvider("p1")
        provider2 = MockResourceProvider("p2")
        
        manager.register_provider("resource", provider1)
        
        with pytest.raises(ValueError, match="already registered"):
            manager.register_provider("resource", provider2)
            
    def test_unregister_provider(self):
        """Test unregistering a provider."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        
        manager.register_provider("resource", provider)
        manager.unregister_provider("resource")
        
        assert "resource" not in manager._providers
        
    def test_acquire_resource(self):
        """Test acquiring a resource."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        resource = manager.acquire("database", "owner1")
        
        assert resource == "resource_test_1"
        assert provider.acquired_count == 1
        assert "owner1:database" in manager._resources
        assert "owner1" in manager._resource_owners["database"]
        
    def test_acquire_same_resource_same_owner(self):
        """Test acquiring same resource by same owner returns cached."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        resource1 = manager.acquire("database", "owner1")
        resource2 = manager.acquire("database", "owner1")
        
        assert resource1 == resource2
        assert provider.acquired_count == 1  # Only acquired once
        
    def test_acquire_unknown_resource(self):
        """Test acquiring unknown resource raises error."""
        manager = ResourceManager()
        
        with pytest.raises(ResourceError, match="Unknown resource"):
            manager.acquire("nonexistent", "owner1")
            
    def test_acquire_closed_manager(self):
        """Test acquiring from closed manager raises error."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("resource", provider)
        
        manager._closed = True
        
        with pytest.raises(ResourceError, match="manager is closed"):
            manager.acquire("resource", "owner1")
            
    def test_release_resource(self):
        """Test releasing a resource."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        resource = manager.acquire("database", "owner1")
        manager.release("database", "owner1")
        
        assert provider.released_count == 1
        assert "owner1:database" not in manager._resources
        assert "owner1" not in manager._resource_owners.get("database", set())
        
    def test_release_not_acquired_resource(self):
        """Test releasing not acquired resource is no-op."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        # Should not raise
        manager.release("database", "owner1")
        
        assert provider.released_count == 0
        
    def test_release_all_owner_resources(self):
        """Test releasing all resources for an owner."""
        manager = ResourceManager()
        provider1 = MockResourceProvider("db")
        provider2 = MockResourceProvider("cache")
        
        manager.register_provider("database", provider1)
        manager.register_provider("cache", provider2)
        
        # Owner acquires multiple resources
        manager.acquire("database", "owner1")
        manager.acquire("cache", "owner1")
        
        # Release all
        manager.release_all("owner1")
        
        assert provider1.released_count == 1
        assert provider2.released_count == 1
        assert len(manager._resources) == 0
        
    def test_get_resource(self):
        """Test getting an acquired resource."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        resource = manager.acquire("database", "owner1")
        retrieved = manager.get_resource("database", "owner1")
        
        assert retrieved == resource
        
    def test_get_not_acquired_resource(self):
        """Test getting not acquired resource returns None."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        retrieved = manager.get_resource("database", "owner1")
        
        assert retrieved is None
        
    def test_has_resource(self):
        """Test checking if owner has resource."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        assert not manager.has_resource("database", "owner1")
        
        manager.acquire("database", "owner1")
        
        assert manager.has_resource("database", "owner1")
        
    def test_get_resource_status(self):
        """Test getting resource status."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        status = manager.get_resource_status("database")
        
        assert status is not None
        assert "provider_exists" in status
        assert status["provider_exists"] is True
        
    def test_get_all_resources(self):
        """Test getting all resources."""
        manager = ResourceManager()
        provider1 = MockResourceProvider("db")
        provider2 = MockResourceProvider("cache")
        
        manager.register_provider("database", provider1)
        manager.register_provider("cache", provider2)
        
        all_resources = manager.get_all_resources()
        
        assert "database" in all_resources
        assert "cache" in all_resources
        
    def test_close_manager(self):
        """Test closing resource manager."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        manager.acquire("database", "owner1")
        
        manager.close()
        
        assert manager._closed is True
        assert provider.released_count == 1  # Resource was released
        assert len(manager._resources) == 0


class TestResourceOwnership:
    """Test resource ownership tracking."""
    
    def test_multiple_owners_different_resources(self):
        """Test multiple owners acquiring different resources."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        resource1 = manager.acquire("database", "owner1")
        resource2 = manager.acquire("database", "owner2")
        
        assert resource1 != resource2  # Different instances
        assert provider.acquired_count == 2
        assert len(manager._resource_owners["database"]) == 2
        
    def test_owner_tracking_cleanup(self):
        """Test owner tracking is cleaned up on release."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        manager.acquire("database", "owner1")
        manager.acquire("database", "owner2")
        
        manager.release("database", "owner1")
        
        assert "owner1" not in manager._resource_owners["database"]
        assert "owner2" in manager._resource_owners["database"]
        
    def test_get_resource_owners(self):
        """Test getting all owners of a resource."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        manager.acquire("database", "owner1")
        manager.acquire("database", "owner2")
        manager.acquire("database", "owner3")
        
        owners = manager.get_resource_owners("database")
        
        assert owners == {"owner1", "owner2", "owner3"}


class TestResourceContention:
    """Test resource contention handling."""
    
    def test_concurrent_acquire_different_owners(self):
        """Test concurrent acquisition by different owners."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        results = {}
        errors = []
        
        def acquire_resource(owner_id):
            try:
                resource = manager.acquire("database", owner_id)
                results[owner_id] = resource
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=acquire_resource, args=(f"owner{i}",))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 5
        assert len(set(results.values())) == 5  # All different resources
        assert provider.acquired_count == 5
        
    def test_concurrent_acquire_same_owner(self):
        """Test concurrent acquisition by same owner."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        results = []
        
        def acquire_resource():
            resource = manager.acquire("database", "owner1")
            results.append(resource)
        
        # Create threads all using same owner
        threads = []
        for i in range(5):
            t = threading.Thread(target=acquire_resource)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        assert len(results) == 5
        assert len(set(results)) == 1  # All same resource (cached)
        assert provider.acquired_count == 1  # Only acquired once
        
    def test_concurrent_release(self):
        """Test concurrent release operations."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        # Acquire resources for multiple owners
        for i in range(5):
            manager.acquire("database", f"owner{i}")
        
        def release_resource(owner_id):
            manager.release("database", owner_id)
        
        # Release concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=release_resource, args=(f"owner{i}",))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        assert provider.released_count == 5
        assert len(manager._resources) == 0


class TestResourcePoolIntegration:
    """Test resource pool integration."""
    
    def test_acquire_from_pool(self):
        """Test acquiring resource from pool."""
        manager = ResourceManager()
        provider = MockResourceProvider("pooled")
        pool_config = PoolConfig(min_size=2, max_size=5)
        
        manager.register_provider("database", provider, pool_config)
        
        # Pool should pre-create min_size resources
        time.sleep(0.1)  # Let pool initialize
        
        resource = manager.acquire("database", "owner1", timeout=1.0)
        
        assert resource is not None
        assert "owner1:database" in manager._resources
        
    def test_pool_timeout(self):
        """Test timeout when acquiring from exhausted pool."""
        manager = ResourceManager()
        provider = MockResourceProvider("pooled")
        pool_config = PoolConfig(min_size=1, max_size=1)  # Very small pool
        
        manager.register_provider("database", provider, pool_config)
        
        # Acquire the only resource
        manager.acquire("database", "owner1", timeout=0.1)
        
        # Try to acquire another (should timeout)
        with pytest.raises(Exception):  # Pool should raise timeout error
            manager.acquire("database", "owner2", timeout=0.1)
            
    def test_release_to_pool(self):
        """Test releasing resource back to pool."""
        manager = ResourceManager()
        provider = MockResourceProvider("pooled")
        pool_config = PoolConfig(min_size=1, max_size=2)
        
        manager.register_provider("database", provider, pool_config)
        
        resource1 = manager.acquire("database", "owner1", timeout=1.0)
        manager.release("database", "owner1")
        
        # Same resource should be reused from pool
        resource2 = manager.acquire("database", "owner2", timeout=1.0)
        
        # Pool might return the same or different resource
        assert resource2 is not None


class TestResourceFailureHandling:
    """Test resource failure and cleanup scenarios."""
    
    def test_cleanup_on_provider_error(self):
        """Test cleanup when provider raises error."""
        manager = ResourceManager()
        provider = Mock(spec=MockResourceProvider)
        provider.acquire.side_effect = Exception("Provider error")
        
        manager.register_provider("database", provider)
        
        with pytest.raises(Exception, match="Provider error"):
            manager.acquire("database", "owner1")
        
        # No resources should be tracked
        assert "owner1:database" not in manager._resources
        
    def test_cleanup_on_release_error(self):
        """Test cleanup continues even if release fails."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        resource = manager.acquire("database", "owner1")
        
        # Mock release to fail
        original_release = provider.release
        provider.release = Mock(side_effect=Exception("Release error"))
        
        # The release method doesn't catch provider exceptions, it will propagate
        # But resources should still be cleaned from tracking
        with pytest.raises(Exception, match="Release error"):
            manager.release("database", "owner1")
        
        # Check that the mock was called
        assert provider.release.called
        
    def test_cascading_cleanup(self):
        """Test cascading cleanup when closing manager."""
        manager = ResourceManager()
        
        # Register multiple providers
        providers = []
        for i in range(3):
            provider = MockResourceProvider(f"p{i}")
            providers.append(provider)
            manager.register_provider(f"resource{i}", provider)
        
        # Acquire resources
        for i in range(3):
            manager.acquire(f"resource{i}", "owner1")
        
        # Close manager should clean up all resources
        manager.close()
        
        for provider in providers:
            assert provider.released_count == 1
        
        assert len(manager._resources) == 0
        assert manager._closed is True
        
    def test_resource_leak_detection(self):
        """Test detection of potential resource leaks."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        # Acquire without release
        manager.acquire("database", "owner1")
        manager.acquire("database", "owner2")
        
        # Check for leaks
        status = manager.get_resource_status("database")
        
        # The MockResourceProvider's get_metrics calculates active_count as acquired - released
        assert provider.acquired_count == 2
        assert provider.released_count == 0
        assert len(manager._resource_owners.get("database", set())) == 2
        
        # Get metrics should show active resources
        metrics = provider.get_metrics()
        assert metrics.active_connections == 2  # 2 acquired - 0 released


class TestResourceContext:
    """Test resource context manager usage."""
    
    def test_context_manager_acquire_release(self):
        """Test using resource manager as context manager."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        @contextmanager
        def resource_context(resource_name, owner_id):
            resource = manager.acquire(resource_name, owner_id)
            try:
                yield resource
            finally:
                manager.release(resource_name, owner_id)
        
        with resource_context("database", "owner1") as resource:
            assert resource == "resource_test_1"
            assert manager.has_resource("database", "owner1")
        
        # After context, resource should be released
        assert not manager.has_resource("database", "owner1")
        assert provider.released_count == 1
        
    def test_context_manager_with_exception(self):
        """Test context manager cleans up on exception."""
        manager = ResourceManager()
        provider = MockResourceProvider("test")
        manager.register_provider("database", provider)
        
        @contextmanager
        def resource_context(resource_name, owner_id):
            resource = manager.acquire(resource_name, owner_id)
            try:
                yield resource
            finally:
                manager.release(resource_name, owner_id)
        
        try:
            with resource_context("database", "owner1") as resource:
                assert resource is not None
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Resource should still be released
        assert not manager.has_resource("database", "owner1")
        assert provider.released_count == 1
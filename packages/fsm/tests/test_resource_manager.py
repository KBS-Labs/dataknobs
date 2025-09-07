"""Tests for resource management system."""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from dataknobs_fsm.functions.base import ResourceError, ResourceConfig
from dataknobs_fsm.resources.base import (
    ResourceStatus,
    ResourceHealth,
    ResourceMetrics,
    BaseResourceProvider,
)
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig, PooledResource
from dataknobs_fsm.resources.database import DatabaseResourceAdapter
from dataknobs_fsm.resources.filesystem import FileSystemResource, FileHandle
from dataknobs_fsm.resources.http import HTTPServiceResource, HTTPSession
from dataknobs_fsm.resources.llm import LLMResource, LLMProvider, LLMSession


class MockResourceProvider(BaseResourceProvider):
    """Mock resource provider for testing."""
    
    def __init__(self, name: str, fail_acquire: bool = False):
        super().__init__(name)
        self.fail_acquire = fail_acquire
        self.resources_created = 0
    
    def acquire(self, **kwargs) -> Any:
        if self.fail_acquire:
            raise ResourceError("Failed to acquire", resource_name=self.name, operation="acquire")
        self.resources_created += 1
        resource = f"resource_{self.resources_created}"
        self._resources.append(resource)
        return resource
    
    def release(self, resource: Any) -> None:
        if resource in self._resources:
            self._resources.remove(resource)


class TestResourceMetrics:
    """Tests for ResourceMetrics."""
    
    def test_record_acquisition(self):
        """Test recording resource acquisition."""
        metrics = ResourceMetrics()
        
        metrics.record_acquisition()
        
        assert metrics.total_acquisitions == 1
        assert metrics.active_connections == 1
        assert metrics.last_acquisition_time is not None
    
    def test_record_release(self):
        """Test recording resource release."""
        metrics = ResourceMetrics()
        
        metrics.record_acquisition()
        metrics.record_release(1.5)
        
        assert metrics.active_connections == 0
        assert metrics.last_release_time is not None
        assert metrics.average_hold_time == 1.5
    
    def test_record_failure(self):
        """Test recording acquisition failure."""
        metrics = ResourceMetrics()
        
        metrics.record_failure()
        
        assert metrics.failed_acquisitions == 1
    
    def test_record_health_check(self):
        """Test recording health check results."""
        metrics = ResourceMetrics()
        
        metrics.record_health_check(True)
        metrics.record_health_check(False)
        
        assert metrics.health_check_failures == 1
        assert metrics.last_health_check is not None


class TestBaseResourceProvider:
    """Tests for BaseResourceProvider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = MockResourceProvider("test_provider")
        
        assert provider.name == "test_provider"
        assert provider.status == ResourceStatus.IDLE
        assert provider.metrics is not None
    
    def test_resource_context_success(self):
        """Test resource context manager with success."""
        provider = MockResourceProvider("test_provider")
        
        with provider.resource_context() as resource:
            assert resource == "resource_1"
            assert provider.metrics.total_acquisitions == 1
            assert provider.metrics.active_connections == 1
        
        assert provider.metrics.active_connections == 0
    
    def test_resource_context_failure(self):
        """Test resource context manager with failure."""
        provider = MockResourceProvider("test_provider", fail_acquire=True)
        
        with pytest.raises(ResourceError):
            with provider.resource_context():
                pass
        
        assert provider.metrics.failed_acquisitions == 1
    
    def test_health_check(self):
        """Test health check."""
        provider = MockResourceProvider("test_provider")
        
        health = provider.health_check()
        assert health == ResourceHealth.HEALTHY
        
        provider.status = ResourceStatus.ERROR
        health = provider.health_check()
        assert health == ResourceHealth.UNHEALTHY
    
    def test_close(self):
        """Test closing provider."""
        provider = MockResourceProvider("test_provider")
        
        resource1 = provider.acquire()
        resource2 = provider.acquire()
        
        provider.close()
        
        assert len(provider._resources) == 0
        assert provider.status == ResourceStatus.CLOSED


class TestResourcePool:
    """Tests for ResourcePool."""
    
    def test_pool_initialization(self):
        """Test pool initialization with minimum resources."""
        provider = MockResourceProvider("test_provider")
        config = PoolConfig(min_size=3, max_size=10)
        
        pool = ResourcePool(provider, config)
        
        assert pool.size() >= config.min_size
        assert provider.resources_created >= config.min_size
    
    def test_acquire_from_pool(self):
        """Test acquiring resource from pool."""
        provider = MockResourceProvider("test_provider")
        pool = ResourcePool(provider, PoolConfig(min_size=2))
        
        resource = pool.acquire()
        
        assert resource is not None
        assert pool.available() == 1  # One left in pool
    
    def test_release_to_pool(self):
        """Test releasing resource back to pool."""
        provider = MockResourceProvider("test_provider")
        pool = ResourcePool(provider, PoolConfig(min_size=1))
        
        resource = pool.acquire()
        initial_available = pool.available()
        
        pool.release(resource)
        
        assert pool.available() > initial_available
    
    def test_pool_max_size(self):
        """Test pool respects max size."""
        provider = MockResourceProvider("test_provider")
        config = PoolConfig(min_size=1, max_size=2, acquire_timeout=0.5)
        pool = ResourcePool(provider, config)
        
        # Acquire all resources
        resource1 = pool.acquire()
        resource2 = pool.acquire()
        
        # Try to acquire beyond max size
        with pytest.raises(ResourceError, match="Failed to acquire resource"):
            pool.acquire(timeout=0.5)
    
    def test_evict_idle_resources(self):
        """Test evicting idle resources."""
        provider = MockResourceProvider("test_provider")
        config = PoolConfig(min_size=3, idle_timeout=0.1)
        pool = ResourcePool(provider, config)
        
        # Wait for resources to become idle
        time.sleep(0.2)
        
        evicted = pool.evict_idle()
        
        assert evicted > 0
        assert pool.size() < 3
    
    def test_pool_close(self):
        """Test closing pool."""
        provider = MockResourceProvider("test_provider")
        pool = ResourcePool(provider, PoolConfig(min_size=2))
        
        resource = pool.acquire()
        
        pool.close()
        
        assert pool._closed is True
        assert pool.size() == 0


class TestResourceManager:
    """Tests for ResourceManager."""
    
    def test_register_provider(self):
        """Test registering a provider."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        
        manager.register_provider("test", provider)
        
        assert "test" in manager._providers
    
    def test_register_provider_with_pool(self):
        """Test registering provider with pool."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        pool_config = PoolConfig(min_size=2)
        
        manager.register_provider("test", provider, pool_config)
        
        assert "test" in manager._providers
        assert "test" in manager._pools
    
    def test_acquire_resource(self):
        """Test acquiring a resource."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        manager.register_provider("test", provider)
        
        resource = manager.acquire("test", "owner1")
        
        assert resource is not None
        assert manager.has_resource("test", "owner1")
    
    def test_release_resource(self):
        """Test releasing a resource."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        manager.register_provider("test", provider)
        
        resource = manager.acquire("test", "owner1")
        manager.release("test", "owner1")
        
        assert not manager.has_resource("test", "owner1")
    
    def test_release_all_for_owner(self):
        """Test releasing all resources for an owner."""
        manager = ResourceManager()
        provider1 = MockResourceProvider("provider1")
        provider2 = MockResourceProvider("provider2")
        
        manager.register_provider("test1", provider1)
        manager.register_provider("test2", provider2)
        
        manager.acquire("test1", "owner1")
        manager.acquire("test2", "owner1")
        
        manager.release_all("owner1")
        
        assert not manager.has_resource("test1", "owner1")
        assert not manager.has_resource("test2", "owner1")
    
    def test_resource_context(self):
        """Test resource context manager."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        manager.register_provider("test", provider)
        
        with manager.resource_context("test", "owner1") as resource:
            assert resource is not None
            assert manager.has_resource("test", "owner1")
        
        assert not manager.has_resource("test", "owner1")
    
    def test_configure_from_requirements(self):
        """Test configuring resources from requirements."""
        manager = ResourceManager()
        provider1 = MockResourceProvider("provider1")
        provider2 = MockResourceProvider("provider2")
        
        manager.register_provider("db", provider1)
        manager.register_provider("cache", provider2)
        
        requirements = [
            ResourceConfig("db", "database", {}),
            ResourceConfig("cache", "memory", {})
        ]
        
        resources = manager.configure_from_requirements(requirements, "owner1")
        
        assert "db" in resources
        assert "cache" in resources
        assert manager.has_resource("db", "owner1")
        assert manager.has_resource("cache", "owner1")
    
    def test_health_check(self):
        """Test health checking resources."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        manager.register_provider("test", provider)
        
        health = manager.health_check()
        
        assert "test" in health
        assert health["test"] == ResourceHealth.HEALTHY
    
    def test_manager_close(self):
        """Test closing manager."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        manager.register_provider("test", provider)
        
        manager.acquire("test", "owner1")
        
        manager.close()
        
        assert manager._closed is True
        assert len(manager._resources) == 0


class TestDatabaseResourceAdapter:
    """Tests for DatabaseResourceAdapter."""
    
    def test_initialization(self):
        """Test database adapter initialization."""
        adapter = DatabaseResourceAdapter("test_db", backend="memory")
        
        assert adapter.name == "test_db"
        assert adapter.backend == "memory"
        assert adapter._database is not None
    
    def test_acquire_database(self):
        """Test acquiring database instance."""
        adapter = DatabaseResourceAdapter("test_db", backend="memory")
        
        db = adapter.acquire()
        
        assert db is not None
        assert adapter.status == ResourceStatus.ACTIVE
    
    def test_database_operations(self):
        """Test database CRUD operations."""
        adapter = DatabaseResourceAdapter("test_db", backend="memory")
        
        from dataknobs_data.records import Record
        
        # Create
        record = Record(data={"name": "test", "value": 123})
        record_id = adapter.create(record)
        assert record_id is not None
        
        # Read
        retrieved = adapter.read(record_id)
        assert retrieved is not None
        assert retrieved.data["name"] == "test"
        
        # Update
        retrieved.data["value"] = 456
        success = adapter.update(record_id, retrieved)
        assert success is True
        
        # Delete
        deleted = adapter.delete(record_id)
        assert deleted is True
    
    def test_health_check(self):
        """Test database health check."""
        adapter = DatabaseResourceAdapter("test_db", backend="memory")
        
        health = adapter.health_check()
        assert health == ResourceHealth.HEALTHY


class TestFileSystemResource:
    """Tests for FileSystemResource."""
    
    def test_initialization(self):
        """Test file system resource initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs_resource = FileSystemResource("test_fs", base_path=tmpdir)
            
            assert fs_resource.name == "test_fs"
            assert fs_resource.base_path == Path(tmpdir).resolve()
    
    def test_file_operations(self):
        """Test file read/write operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs_resource = FileSystemResource("test_fs", base_path=tmpdir)
            
            # Write text
            fs_resource.write_text("test.txt", "Hello, World!")
            
            # Read text
            content = fs_resource.read_text("test.txt")
            assert content == "Hello, World!"
            
            # Check existence
            assert fs_resource.exists("test.txt")
            
            # Delete
            deleted = fs_resource.delete("test.txt")
            assert deleted is True
            assert not fs_resource.exists("test.txt")
    
    def test_file_context_manager(self):
        """Test file context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs_resource = FileSystemResource("test_fs", base_path=tmpdir)
            
            with fs_resource.open("test.txt", "w") as f:
                f.write("Test content")
            
            with fs_resource.open("test.txt", "r") as f:
                content = f.read()
                assert content == "Test content"
    
    def test_temp_file(self):
        """Test temporary file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs_resource = FileSystemResource("test_fs", temp_dir=tmpdir)
            
            with fs_resource.temp_file(suffix=".txt") as f:
                f.write("Temporary content")
                temp_path = f.name
            
            # Temp file should be tracked
            assert len(fs_resource._temp_files) > 0
            
            # Cleanup
            fs_resource.cleanup_temp_files()
            assert len(fs_resource._temp_files) == 0


class TestHTTPServiceResource:
    """Tests for HTTPServiceResource."""
    
    def test_initialization(self):
        """Test HTTP service resource initialization."""
        http_resource = HTTPServiceResource(
            "test_api",
            base_url="https://api.example.com",
            headers={"User-Agent": "TestClient"}
        )
        
        assert http_resource.name == "test_api"
        assert http_resource.base_url == "https://api.example.com"
        assert "User-Agent" in http_resource.default_headers
    
    def test_session_acquisition(self):
        """Test acquiring HTTP session."""
        http_resource = HTTPServiceResource("test_api", base_url="https://api.example.com")
        
        session = http_resource.acquire()
        
        assert isinstance(session, HTTPSession)
        assert session.base_url == "https://api.example.com"
    
    def test_auth_setup(self):
        """Test authentication setup."""
        # Bearer token
        http_resource = HTTPServiceResource(
            "test_api",
            base_url="https://api.example.com",
            auth={"type": "bearer", "token": "test_token"}
        )
        assert "Authorization" in http_resource.default_headers
        assert http_resource.default_headers["Authorization"] == "Bearer test_token"
        
        # API key
        http_resource2 = HTTPServiceResource(
            "test_api2",
            base_url="https://api.example.com",
            auth={"type": "api_key", "key_value": "test_key"}
        )
        assert "X-API-Key" in http_resource2.default_headers
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        session = HTTPSession(
            base_url="https://api.example.com",
            failure_threshold=3,
            circuit_half_open_after=0.1
        )
        
        # Record failures
        for _ in range(3):
            session.record_failure()
        
        # Circuit should be open
        assert session.is_circuit_open() is True
        
        # Wait for half-open period
        time.sleep(0.2)
        
        # Circuit should allow retry
        assert session.is_circuit_open() is False
        
        # Success should reset
        session.record_success()
        assert session.failure_count == 0


class TestLLMResource:
    """Tests for LLMResource."""
    
    def test_initialization_ollama(self):
        """Test Ollama LLM resource initialization."""
        llm_resource = LLMResource(
            "test_llm",
            provider="ollama",
            model="llama2"
        )
        
        assert llm_resource.name == "test_llm"
        assert llm_resource.provider == LLMProvider.OLLAMA
        assert llm_resource.endpoint == "http://localhost:11434"
    
    def test_initialization_huggingface(self):
        """Test HuggingFace LLM resource initialization."""
        llm_resource = LLMResource(
            "test_llm",
            provider="huggingface",
            model="gpt2"
        )
        
        assert llm_resource.provider == LLMProvider.HUGGINGFACE
        assert llm_resource.model == "gpt2"
    
    def test_session_acquisition(self):
        """Test acquiring LLM session."""
        llm_resource = LLMResource("test_llm", provider="ollama")
        
        session = llm_resource.acquire()
        
        assert isinstance(session, LLMSession)
        assert session.provider == LLMProvider.OLLAMA
    
    def test_rate_limiting(self):
        """Test rate limiting for commercial providers."""
        session = LLMSession(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            requests_per_minute=2,
            tokens_per_minute=100
        )
        
        # Should allow initial requests
        assert session.check_rate_limits(50) is True
        session.record_usage(25, 25)
        
        # Should allow second request
        assert session.check_rate_limits(40) is True
        session.record_usage(20, 20)
        
        # Should block third request (exceeds request limit)
        assert session.check_rate_limits(30) is False
    
    def test_no_rate_limiting_local(self):
        """Test no rate limiting for local providers."""
        session = LLMSession(
            provider=LLMProvider.OLLAMA,
            model_name="llama2"
        )
        
        # Should always allow requests
        for _ in range(100):
            assert session.check_rate_limits(1000) is True
            session.record_usage(500, 500)
    
    def test_usage_stats(self):
        """Test usage statistics tracking."""
        llm_resource = LLMResource("test_llm", provider="ollama")
        session = llm_resource.acquire()
        
        # Record some usage
        session.record_usage(100, 50)
        session.record_usage(200, 100)
        
        stats = llm_resource.get_usage_stats(session)
        
        assert stats["total_requests"] == 2
        assert stats["total_prompt_tokens"] == 300
        assert stats["total_completion_tokens"] == 150


@pytest.mark.integration
class TestResourceIntegration:
    """Integration tests for resource management."""
    
    def test_multiple_resources_with_manager(self):
        """Test managing multiple resource types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResourceManager()
            
            # Register different resource types
            db_adapter = DatabaseResourceAdapter("db", backend="memory")
            fs_resource = FileSystemResource("fs", base_path=tmpdir)
            
            manager.register_provider("database", db_adapter)
            manager.register_provider("filesystem", fs_resource)
            
            # Acquire resources for a state
            db = manager.acquire("database", "state1")
            fs = manager.acquire("filesystem", "state1", temp=True)
            
            assert db is not None
            assert fs is not None
            
            # Release all for state
            manager.release_all("state1")
            
            assert not manager.has_resource("database", "state1")
            assert not manager.has_resource("filesystem", "state1")
    
    def test_concurrent_resource_access(self):
        """Test concurrent resource access."""
        manager = ResourceManager()
        provider = MockResourceProvider("test_provider")
        pool_config = PoolConfig(min_size=2, max_size=5)
        
        manager.register_provider("test", provider, pool_config)
        
        results = []
        
        def worker(owner_id):
            resource = manager.acquire("test", owner_id)
            results.append(resource)
            time.sleep(0.01)  # Reduced sleep time
            manager.release("test", owner_id)
        
        threads = [threading.Thread(target=worker, args=(f"owner{i}",)) 
                  for i in range(5)]
        
        for t in threads:
            t.start()
        
        # Add timeout to prevent hanging
        for t in threads:
            t.join(timeout=5.0)  # 5 second timeout
            if t.is_alive():
                # Thread is still running, which indicates a deadlock
                raise TimeoutError(f"Thread {t.name} did not complete within timeout")
        
        assert len(results) == 5
        assert len(set(results)) <= 5  # May reuse resources from pool
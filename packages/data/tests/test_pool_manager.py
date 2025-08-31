"""Tests for general-purpose connection pool management utilities."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import dataclass

from dataknobs_data.pooling import (
    ConnectionPoolManager,
    BasePoolConfig,
)
from dataknobs_data.pooling.postgres import (
    PostgresPoolConfig,
    create_asyncpg_pool,
    validate_asyncpg_pool
)


class TestPostgresPoolConfig:
    """Test PostgresPoolConfig class."""
    
    def test_from_dict_with_defaults(self):
        """Test creating PostgresPoolConfig with defaults."""
        config = PostgresPoolConfig.from_dict({})
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "postgres"
        assert config.user == "postgres"
        assert config.password == ""
        assert config.min_size == 2
        assert config.max_size == 5
    
    def test_from_dict_with_custom_values(self):
        """Test creating PostgresPoolConfig with custom values."""
        config = PostgresPoolConfig.from_dict({
            "host": "db.example.com",
            "port": 5433,
            "database": "mydb",
            "user": "myuser",
            "password": "secret",
            "min_pool_size": 5,
            "max_pool_size": 20,
            "command_timeout": 30.0
        })
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "mydb"
        assert config.user == "myuser"
        assert config.password == "secret"
        assert config.min_size == 5
        assert config.max_size == 20
        assert config.command_timeout == 30.0
    
    def test_to_connection_string(self):
        """Test connection string generation."""
        config = PostgresPoolConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass"
        )
        expected = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert config.to_connection_string() == expected
    
    def test_to_hash_key(self):
        """Test hash key generation."""
        config = PostgresPoolConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser"
        )
        assert config.to_hash_key() == ("localhost", 5432, "testdb", "testuser")


@dataclass
class MockPoolConfig(BasePoolConfig):
    """Mock implementation of BasePoolConfig for testing."""
    host: str = "localhost"
    port: int = 5432
    
    def to_connection_string(self) -> str:
        return f"test://{self.host}:{self.port}"
    
    def to_hash_key(self) -> tuple:
        return (self.host, self.port)


class MockPool:
    """Mock pool implementation."""
    
    async def acquire(self):
        """Mock acquire method."""
        pass
    
    async def close(self):
        """Mock close method."""
        pass


class TestConnectionPoolManager:
    """Test ConnectionPoolManager class."""
    
    @pytest.mark.asyncio
    async def test_get_pool_creates_new(self):
        """Test that get_pool creates a new pool when none exists."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)
        
        async def create_pool(cfg):
            return MockPool()
        
        pool = await manager.get_pool(config, create_pool)
        
        assert isinstance(pool, MockPool)
        assert manager.get_pool_count() == 1
    
    @pytest.mark.asyncio
    async def test_get_pool_reuses_existing(self):
        """Test that get_pool reuses existing pool for same loop."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)
        
        create_count = 0
        
        async def create_pool(cfg):
            nonlocal create_count
            create_count += 1
            return MockPool()
        
        # Get pool twice
        pool1 = await manager.get_pool(config, create_pool)
        pool2 = await manager.get_pool(config, create_pool)
        
        assert pool1 is pool2
        assert create_count == 1  # Should only create once
        assert manager.get_pool_count() == 1
    
    @pytest.mark.asyncio
    async def test_get_pool_with_validation(self):
        """Test that get_pool validates existing pools."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)
        
        mock_pool = MockPool()
        validation_count = 0
        
        async def create_pool(cfg):
            return mock_pool
        
        async def validate_pool(pool):
            nonlocal validation_count
            validation_count += 1
            # Validation succeeds
        
        # Get pool twice with validation
        pool1 = await manager.get_pool(config, create_pool, validate_pool)
        pool2 = await manager.get_pool(config, create_pool, validate_pool)
        
        assert pool1 is pool2
        assert validation_count == 1  # Only validates on second get
    
    @pytest.mark.asyncio
    async def test_get_pool_recreates_invalid(self):
        """Test that get_pool recreates pool when validation fails."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)
        
        pool1 = MockPool()
        pool1.close = AsyncMock()
        pool2 = MockPool()
        
        create_count = 0
        
        async def create_pool(cfg):
            nonlocal create_count
            create_count += 1
            return pool1 if create_count == 1 else pool2
        
        validation_count = 0
        
        async def validate_pool(pool):
            nonlocal validation_count
            validation_count += 1
            if validation_count == 1:
                raise Exception("Pool invalid")
        
        # Get pool twice
        first_pool = await manager.get_pool(config, create_pool)
        second_pool = await manager.get_pool(config, create_pool, validate_pool)
        
        assert first_pool is pool1
        assert second_pool is pool2
        assert create_count == 2
        pool1.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_pool(self):
        """Test removing a pool."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)
        
        mock_pool = MockPool()
        mock_pool.close = AsyncMock()
        
        async def create_pool(cfg):
            return mock_pool
        
        # Create pool
        await manager.get_pool(config, create_pool)
        assert manager.get_pool_count() == 1
        
        # Remove it
        removed = await manager.remove_pool(config)
        assert removed is True
        assert manager.get_pool_count() == 0
        mock_pool.close.assert_called_once()
        
        # Try to remove again
        removed = await manager.remove_pool(config)
        assert removed is False
    
    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all pools."""
        manager = ConnectionPoolManager[MockPool]()
        config1 = MockPoolConfig(host="localhost", port=5432)
        config2 = MockPoolConfig(host="localhost", port=5433)
        
        pool1 = MockPool()
        pool1.close = AsyncMock()
        pool2 = MockPool()
        pool2.close = AsyncMock()
        
        async def create_pool1(cfg):
            return pool1
        
        async def create_pool2(cfg):
            return pool2
        
        # Create two pools
        await manager.get_pool(config1, create_pool1)
        await manager.get_pool(config2, create_pool2)
        assert manager.get_pool_count() == 2
        
        # Close all
        await manager.close_all()
        
        assert manager.get_pool_count() == 0
        pool1.close.assert_called_once()
        pool2.close.assert_called_once()
    
    def test_get_pool_info(self):
        """Test getting pool information."""
        manager = ConnectionPoolManager[MockPool]()
        
        # Mock some pools
        manager._pools = {
            (12345, 67890): MockPool(),
            (54321, 67890): MockPool()
        }
        
        info = manager.get_pool_info()
        
        assert len(info) == 2
        assert "config_12345_loop_67890" in info
        assert "config_54321_loop_67890" in info
        assert info["config_12345_loop_67890"]["loop_id"] == 67890
        assert info["config_12345_loop_67890"]["config_hash"] == 12345


class TestAsyncpgHelpers:
    """Test asyncpg helper functions."""
    
    @pytest.mark.asyncio
    async def test_create_asyncpg_pool(self):
        """Test creating an asyncpg pool."""
        config = PostgresPoolConfig(
            host="localhost",
            port=5432,
            database="test",
            user="testuser",
            password="testpass",
            min_size=5,
            max_size=15
        )
        
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_pool = MagicMock()
            mock_create.return_value = mock_pool
            
            pool = await create_asyncpg_pool(config)
            
            assert pool == mock_pool
            mock_create.assert_called_once_with(
                "postgresql://testuser:testpass@localhost:5432/test",
                min_size=5,
                max_size=15,
                command_timeout=None,
                ssl=None
            )
    
    @pytest.mark.asyncio
    async def test_validate_asyncpg_pool_success(self):
        """Test validating a working asyncpg pool."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        
        # Setup async context manager
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Should not raise
        await validate_asyncpg_pool(mock_pool)
        
        mock_conn.fetchval.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_validate_asyncpg_pool_failure(self):
        """Test validating a broken asyncpg pool."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Connection failed"))
        
        # Setup async context manager
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Should raise
        with pytest.raises(Exception, match="Connection failed"):
            await validate_asyncpg_pool(mock_pool)

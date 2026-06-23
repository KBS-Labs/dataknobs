"""Tests for general-purpose connection pool management utilities."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dataknobs_data.pooling import (
    BasePoolConfig,
    ConnectionPoolManager,
)
from dataknobs_data.pooling.base import _PoolEntry
from dataknobs_data.pooling.postgres import (
    PostgresPoolConfig,
    create_asyncpg_pool,
    validate_asyncpg_pool,
)


class TestPostgresPoolConfig:
    """Test PostgresPoolConfig class."""

    @pytest.fixture(autouse=True)
    def _clear_postgres_env(self, monkeypatch):
        """Isolate tests from ambient env vars and workspace dotenv files.

        ``PostgresPoolConfig.from_dict`` routes through
        ``normalize_postgres_connection_config`` which, by design, fills
        empty config from env (and from ``.env`` files when
        ``python-dotenv`` is installed) as a fallback. These tests
        assert dataclass-level defaults and explicit-value wiring —
        clearing every fallback keeps the assertions meaningful
        regardless of the developer's shell and workspace state.
        """
        for key in (
            "DATABASE_URL",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            "dataknobs_common.postgres_config._load_dotenv_fallbacks",
            lambda start_path=None: {},
        )

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

    def test_from_dict_accepts_individual_keys(self):
        """Individual host/port/... keys build a valid PostgresPoolConfig."""
        config = PostgresPoolConfig.from_dict({
            "host": "h",
            "port": 5433,
            "database": "db",
            "user": "u",
            "password": "p",
        })
        assert config.host == "h"
        assert config.port == 5433
        assert config.database == "db"
        assert config.user == "u"
        assert config.password == "p"

    def test_from_dict_env_var_fallback(self, monkeypatch):
        """POSTGRES_* env vars fill in missing keys via the normalizer."""
        for key in (
            "DATABASE_URL",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("POSTGRES_HOST", "env-h")
        monkeypatch.setenv("POSTGRES_PORT", "5678")
        monkeypatch.setenv("POSTGRES_DB", "env-db")
        monkeypatch.setenv("POSTGRES_USER", "env-u")
        monkeypatch.setenv("POSTGRES_PASSWORD", "env-p")

        config = PostgresPoolConfig.from_dict({})
        assert config.host == "env-h"
        assert config.port == 5678
        assert config.database == "env-db"
        assert config.user == "env-u"
        assert config.password == "env-p"

    def test_from_dict_asyncpg_prefix_stripped(self):
        """postgresql+asyncpg:// dialect prefix is normalized."""
        config = PostgresPoolConfig.from_dict({
            "connection_string": "postgresql+asyncpg://u:p@h:5433/db",
        })
        assert config.host == "h"
        assert config.port == 5433
        assert config.database == "db"
        assert "asyncpg" not in config.to_connection_string()

    def test_from_dict_preserves_url_password_when_only_url_given(self):
        """A raw ``connection_string``-only dict keeps the URL's password.

        Regression: a previous ``already_normalized`` fast-path skipped
        ``normalize_postgres_connection_config`` whenever the dict
        carried ``connection_string`` plus host/port/database/user — but
        intentionally omitted ``password`` from that check so empty
        passwords stayed valid. The shortcut then pulled
        ``source.get("password", "")`` from the raw dict, silently
        dropping the password parsed from the URL. Authentication
        failed downstream with no hint about why.
        """
        config = PostgresPoolConfig.from_dict({
            "connection_string": "postgresql://u:secret@h:5432/db",
            "host": "h",
            "port": 5432,
            "database": "db",
            "user": "u",
        })
        assert config.password == "secret"
        assert config.host == "h"
        assert config.user == "u"


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
            (12345, 67890): _PoolEntry(MockPool()),
            (54321, 67890): _PoolEntry(MockPool())
        }

        info = manager.get_pool_info()

        assert len(info) == 2
        assert "config_12345_loop_67890" in info
        assert "config_54321_loop_67890" in info
        assert info["config_12345_loop_67890"]["loop_id"] == 67890
        assert info["config_12345_loop_67890"]["config_hash"] == 12345


class TestPoolRefcount:
    """Test the shared-pool refcount + release-on-last-holder contract.

    Pools handed out by :meth:`ConnectionPoolManager.get_pool` are shared
    by config across holders on one event loop. ``release_pool`` is the
    close path: it decrements the holder count and tears the pool down
    only when the last holder releases. These tests pin that contract at
    the manager (unit) layer.
    """

    @staticmethod
    def _pool_key(manager, config):
        """Compute the manager's internal pool key for the running loop."""
        return (hash(config.to_hash_key()), id(asyncio.get_running_loop()))

    @pytest.mark.asyncio
    async def test_release_pool_refcount_across_holders(self):
        """Three holders share one pool; only the last release closes it."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)

        pool = MockPool()
        pool.close = AsyncMock()
        create_count = 0

        async def create_pool(cfg):
            nonlocal create_count
            create_count += 1
            return pool

        # Three holders acquire the same shared pool.
        p1 = await manager.get_pool(config, create_pool)
        p2 = await manager.get_pool(config, create_pool)
        p3 = await manager.get_pool(config, create_pool)
        assert p1 is p2 is p3 is pool
        assert create_count == 1
        assert manager.get_pool_count() == 1

        # First two releases: pool survives, still counted.
        await manager.release_pool(config)
        await manager.release_pool(config)
        pool.close.assert_not_called()
        assert manager.get_pool_count() == 1

        # Last release: pool closed exactly once and evicted.
        await manager.release_pool(config)
        pool.close.assert_called_once()
        assert manager.get_pool_count() == 0

    @pytest.mark.asyncio
    async def test_release_pool_single_holder_evicts(self):
        """A single holder's release closes + evicts the pool.

        Regression guard for the pre-fix postgres entry leak: a
        single-holder ``close()`` previously hard-closed the pool but
        never dropped the manager entry, so ``get_pool_count()`` never
        fell back to zero.
        """
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)

        pool = MockPool()
        pool.close = AsyncMock()

        async def create_pool(cfg):
            return pool

        await manager.get_pool(config, create_pool)
        assert manager.get_pool_count() == 1

        await manager.release_pool(config)
        pool.close.assert_called_once()
        assert manager.get_pool_count() == 0

    @pytest.mark.asyncio
    async def test_release_pool_unknown_config_is_noop(self):
        """Releasing a never-acquired config is a safe no-op.

        Backs the double-``close()`` safety of every consumer: the second
        call (after ``self._pool = None``) reaches a config the manager no
        longer tracks, and must not raise or underflow.
        """
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=9999)

        # Must not raise; nothing tracked, count stays zero.
        await manager.release_pool(config)
        assert manager.get_pool_count() == 0

    @pytest.mark.asyncio
    async def test_validate_rebuild_preserves_refcount(self):
        """A validate-triggered rebuild carries existing holders forward."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)

        pool1 = MockPool()
        pool1.close = AsyncMock()
        pool2 = MockPool()
        pool2.close = AsyncMock()
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

        # Two holders on the original pool (no validator on the warm path).
        await manager.get_pool(config, create_pool)
        await manager.get_pool(config, create_pool)
        pool_key = self._pool_key(manager, config)
        assert manager._pools[pool_key].refcount == 2

        # Third acquire validates, fails, rebuilds — the two existing
        # holders plus the new one must be carried into the rebuilt entry.
        third = await manager.get_pool(config, create_pool, validate_pool)
        assert third is pool2
        assert create_count == 2
        pool1.close.assert_called_once()
        assert manager.get_pool_count() == 1
        assert manager._pools[pool_key].refcount == 3

        # Three releases drive it to zero with no premature/underflow close.
        await manager.release_pool(config)
        await manager.release_pool(config)
        pool2.close.assert_not_called()
        await manager.release_pool(config)
        pool2.close.assert_called_once()
        assert manager.get_pool_count() == 0

    @pytest.mark.asyncio
    async def test_get_pool_info_tolerates_refcount_entries(self):
        """get_pool_info reports the expected shape with _PoolEntry storage."""
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)

        async def create_pool(cfg):
            return MockPool()

        await manager.get_pool(config, create_pool)
        info = manager.get_pool_info()
        assert len(info) == 1
        (key,) = info
        assert info[key]["loop_id"] == id(asyncio.get_running_loop())
        assert info[key]["config_hash"] == hash(config.to_hash_key())
        assert "pool" in info[key]

    @pytest.mark.asyncio
    async def test_concurrent_cold_create_yields_one_pool(self):
        """Concurrent cold-key acquires create exactly one shared pool.

        Reproduce-first guard for 173-D: without the per-loop create
        lock, two coroutines that both enter the cold-key
        check->create->assign window each call ``create_pool_func`` and
        the second assignment clobbers the first, corrupting the
        refcount. The gate parks the create function so both tasks are
        scheduled into the create region before either returns; with the
        lock only one creates, the loser joins the winner's entry.
        """
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)

        create_count = 0
        gate = asyncio.Event()
        pool = MockPool()
        pool.close = AsyncMock()

        async def create_pool(cfg):
            nonlocal create_count
            create_count += 1
            await gate.wait()
            return pool

        task_a = asyncio.create_task(manager.get_pool(config, create_pool))
        task_b = asyncio.create_task(manager.get_pool(config, create_pool))

        # Drain the ready queue so both tasks progress to their await
        # point (the gate inside create, or the create lock) before we
        # release the gate. Yields, not timed sleeps — deterministic.
        for _ in range(10):
            await asyncio.sleep(0)

        gate.set()
        results = await asyncio.gather(task_a, task_b)

        assert results[0] is pool and results[1] is pool
        assert create_count == 1
        assert manager.get_pool_count() == 1
        assert manager._pools[self._pool_key(manager, config)].refcount == 2

        # Two releases drive the shared entry to zero with one close.
        await manager.release_pool(config)
        pool.close.assert_not_called()
        await manager.release_pool(config)
        pool.close.assert_called_once()
        assert manager.get_pool_count() == 0

    @pytest.mark.asyncio
    async def test_validate_during_release_does_not_resurrect_closed_pool(self):
        """A warm validate overlapping the last release must not return a closed pool.

        Reproduce-first for the validate-success resurrection race: holder
        B enters the warm path with a validator and parks inside it; while
        B is parked, holder A's release drops the count to zero and closes
        + evicts the pool. B's validate then succeeds — but the entry it
        validated is gone, so B must rebuild a fresh pool, NOT re-insert and
        return the now-closed one.
        """
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)

        pool1 = MockPool()
        pool1.close = AsyncMock()
        pool2 = MockPool()
        pool2.close = AsyncMock()
        create_count = 0

        async def create_pool(cfg):
            nonlocal create_count
            create_count += 1
            return pool1 if create_count == 1 else pool2

        release_gate = asyncio.Event()
        validate_started = asyncio.Event()

        async def validate_pool(pool):
            # Signal entry into validate, then park so the release can
            # interleave (close + evict) the pool we are validating.
            validate_started.set()
            await release_gate.wait()
            # Validation "succeeds" against the (now closed) pool.

        # Holder A acquires the shared pool (no validator -> no await point).
        a_pool = await manager.get_pool(config, create_pool)
        assert a_pool is pool1
        assert create_count == 1

        # Holder B begins a warm *validated* acquire and parks in validate.
        b_task = asyncio.create_task(
            manager.get_pool(config, create_pool, validate_pool)
        )
        await validate_started.wait()

        # While B is parked, A releases — last holder -> close + evict.
        await manager.release_pool(config)
        pool1.close.assert_called_once()
        assert manager.get_pool_count() == 0

        # Let B's validate finish. B must rebuild rather than resurrect pool1.
        release_gate.set()
        b_pool = await b_task

        assert b_pool is pool2, "B resurrected the closed pool instead of rebuilding"
        assert b_pool is not pool1
        assert create_count == 2
        assert manager.get_pool_count() == 1
        key = self._pool_key(manager, config)
        assert manager._pools[key].refcount == 1

    @pytest.mark.asyncio
    async def test_release_during_warm_get_does_not_hand_out_closing_pool(self):
        """A warm get overlapping the last release's close must not get the closing pool.

        Reproduce-first for the release/close atomicity race: the last
        holder's release begins closing the pool (the close awaits). The
        entry must be evicted from the live map *before* the close is
        awaited, so a concurrent warm get_pool misses it and rebuilds a
        fresh pool instead of grabbing the one being torn down.
        """
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)

        pool1 = MockPool()
        pool2 = MockPool()
        pool2.close = AsyncMock()
        create_count = 0

        async def create_pool(cfg):
            nonlocal create_count
            create_count += 1
            return pool1 if create_count == 1 else pool2

        close_gate = asyncio.Event()
        close_started = asyncio.Event()

        async def slow_close(pool):
            # Park inside the last-holder close so a warm get can interleave.
            close_started.set()
            await close_gate.wait()

        # Single holder acquires; the entry registers slow_close.
        first = await manager.get_pool(
            config, create_pool, close_pool_func=slow_close
        )
        assert first is pool1

        # Release -> last holder -> begins closing pool1 and parks.
        release_task = asyncio.create_task(manager.release_pool(config))
        await close_started.wait()

        # A concurrent warm get arrives while pool1 is mid-close. It must
        # NOT receive the closing pool1; pop-first eviction makes it rebuild.
        get_task = asyncio.create_task(manager.get_pool(config, create_pool))
        for _ in range(10):  # let the warm get reach its decision point
            await asyncio.sleep(0)

        close_gate.set()
        await release_task
        new_pool = await get_task

        assert new_pool is pool2, "warm get received the closing pool"
        assert new_pool is not pool1
        assert create_count == 2
        key = self._pool_key(manager, config)
        assert manager._pools[key].refcount == 1

    @pytest.mark.asyncio
    async def test_release_pool_underflow_is_logged(self, caplog):
        """Releasing more times than acquired logs a holder-count underflow.

        Pop-at-zero means the public API cannot normally drive a live entry
        negative, but a future accounting bug could. Inject an
        already-at-zero entry (simulating that bug) and assert the next
        release surfaces the underflow loudly rather than silently closing
        one hand-out early.
        """
        manager = ConnectionPoolManager[MockPool]()
        config = MockPoolConfig(host="localhost", port=5432)
        pool = MockPool()
        pool.close = AsyncMock()

        key = self._pool_key(manager, config)
        manager._pools[key] = _PoolEntry(pool, None, refcount=0)

        with caplog.at_level("WARNING", logger="dataknobs_data.pooling.base"):
            await manager.release_pool(config)

        assert any(
            "negative" in r.message.lower() for r in caplog.records
        ), "underflow was not logged"
        pool.close.assert_called_once()
        assert manager.get_pool_count() == 0


class TestCleanupOnExit:
    """Test _cleanup_on_exit behavior."""

    def test_cleanup_no_pools(self):
        """No-op when pool dict is empty."""
        manager = ConnectionPoolManager[MockPool]()
        assert manager.get_pool_count() == 0
        # Should return immediately without error
        manager._cleanup_on_exit()

    def test_cleanup_with_running_loop_clears(self):
        """When a running loop exists, clears pool references and warns."""

        async def _run():
            manager = ConnectionPoolManager[MockPool]()
            config = MockPoolConfig(host="localhost", port=5432)

            async def create_pool(cfg):
                return MockPool()

            await manager.get_pool(config, create_pool)
            assert manager.get_pool_count() == 1

            # Called from inside a running loop — should clear, not hang
            manager._cleanup_on_exit()
            assert manager.get_pool_count() == 0

        asyncio.run(_run())

    def test_cleanup_without_loop(self):
        """When no running loop exists, creates temp loop and closes pools."""
        manager = ConnectionPoolManager[MockPool]()

        pool = MockPool()
        pool.close = AsyncMock()

        # Manually inject a pool entry so we don't need an async context
        config = MockPoolConfig(host="localhost", port=5432)
        loop_id = id(asyncio.new_event_loop())
        config_hash = hash(config.to_hash_key())
        manager._pools[(config_hash, loop_id)] = _PoolEntry(pool)

        assert manager.get_pool_count() == 1

        # No running loop — should create temp loop and call close_all
        manager._cleanup_on_exit()
        assert manager.get_pool_count() == 0
        pool.close.assert_called_once()


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

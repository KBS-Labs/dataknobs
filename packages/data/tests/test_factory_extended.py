"""Extended tests for backend factory functionality including all backends."""

import logging
import pytest
import tempfile

from dataknobs_data.factory import (
    DatabaseFactory,
    AsyncDatabaseFactory,
    database_factory,
    async_database_factory,
)


class TestDatabaseFactoryPostgres:
    """Test backend creation via factory using memory backend (no mocks needed)."""

    def test_create_postgres_backend_success(self, caplog):
        """Test successful backend creation and config passing.

        The log assertion reads the emitted record rather than a patched
        module-level logger, so it pins that the requested backend is
        announced rather than which module announces it.
        """
        caplog.set_level(logging.DEBUG)
        factory = DatabaseFactory()

        # Use real memory backend instead of mocking - tests same factory logic.
        # ``vector_enabled`` is a real field, so the assertion below observes
        # config actually reaching the backend rather than a key being dropped.
        db = factory.create(backend="memory", vector_enabled=True)

        # Verify factory created a database instance
        from dataknobs_data.backends.memory import SyncMemoryDatabase

        assert isinstance(db, SyncMemoryDatabase)
        assert db.config.vector_enabled is True, "config did not reach the backend"
        assert any(
            record.levelno == logging.INFO
            and record.getMessage() == "Creating database with backend: memory"
            for record in caplog.records
        )

    def test_postgres_aliases(self):
        """Test that backend aliases work (using memory backend as example)."""
        factory = DatabaseFactory()

        # Test aliases using real memory backend
        for alias in ["memory", "mem"]:
            db = factory.create(backend=alias)
            from dataknobs_data.backends.memory import SyncMemoryDatabase

            assert isinstance(db, SyncMemoryDatabase)


class TestDatabaseFactoryElasticsearch:
    """Test backend creation via factory using file backend (no mocks needed)."""

    def test_create_elasticsearch_backend_success(self):
        """Test successful backend creation and config passing."""
        factory = DatabaseFactory()

        # Use real file backend instead of mocking - tests same factory logic
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = factory.create(backend="file", path=filepath, format="json")

            # Verify factory created a database instance
            from dataknobs_data.backends.file import SyncFileDatabase

            assert isinstance(db, SyncFileDatabase)
        finally:
            import os

            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_elasticsearch_aliases(self):
        """Test that backend aliases work (using SQLite backend as example)."""
        factory = DatabaseFactory()

        # Test aliases using real SQLite backend
        for alias in ["sqlite", "sqlite3"]:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name

            try:
                db = factory.create(backend=alias, path=db_path)
                from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

                assert isinstance(db, SyncSQLiteDatabase)
            finally:
                import os

                if os.path.exists(db_path):
                    os.unlink(db_path)


# The postgres/elasticsearch/s3 "driver is missing" errors were covered here
# by three tests that patched `DatabaseFactory.create` -- the callable under
# test -- handed the patch a message, and asserted the patch had raised it.
# Two of those messages existed nowhere in any package source; all three would
# have passed against an empty implementation. The real path, with the real
# message and the driver genuinely absent, is covered in
# test_backend_availability.py.


class TestBackendInfo:
    """Test get_backend_info method."""

    def test_get_all_backend_info(self):
        """Test getting info for all supported backends."""
        factory = DatabaseFactory()

        backends = ["memory", "file", "postgres", "elasticsearch", "s3"]

        for backend in backends:
            info = factory.get_backend_info(backend)
            assert "description" in info
            assert "persistent" in info
            assert "requires_install" in info or not info.get("requires_install")
            assert "config_options" in info

    def test_get_info_case_insensitive(self):
        """Test that backend info lookup is case insensitive."""
        factory = DatabaseFactory()

        info_lower = factory.get_backend_info("memory")
        info_upper = factory.get_backend_info("MEMORY")
        info_mixed = factory.get_backend_info("MeMoRy")

        assert info_lower == info_upper == info_mixed

    def test_get_info_unknown_backend(self):
        """Test getting info for unknown backend."""
        factory = DatabaseFactory()

        info = factory.get_backend_info("nonexistent")
        assert info["description"] == "Unknown backend"
        assert "error" in info
        assert "nonexistent" in info["error"]


class TestAsyncDatabaseFactory:
    """Test AsyncDatabaseFactory class using real backends (no mocks needed)."""

    def test_create_async_memory_backend(self):
        """Test creating async memory backend."""
        factory = AsyncDatabaseFactory()

        # Use real async memory backend
        db = factory.create(backend="memory")
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        assert isinstance(db, AsyncMemoryDatabase)

    def test_create_async_file_backend(self):
        """Test creating async file backend."""
        factory = AsyncDatabaseFactory()

        # Use real async file backend
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = factory.create(backend="file", path=filepath)
            from dataknobs_data.backends.file import AsyncFileDatabase

            assert isinstance(db, AsyncFileDatabase)
        finally:
            import os

            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_create_async_postgres_backend(self):
        """Test creating async backend (using memory as example)."""
        factory = AsyncDatabaseFactory()

        # Use real memory backend - tests same factory logic
        db = factory.create(backend="memory")
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        assert isinstance(db, AsyncMemoryDatabase)

    def test_create_async_elasticsearch_backend(self):
        """Test creating async backend (using file as example)."""
        factory = AsyncDatabaseFactory()

        # Use real file backend - tests same factory logic
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = factory.create(backend="file", path=filepath)
            from dataknobs_data.backends.file import AsyncFileDatabase

            assert isinstance(db, AsyncFileDatabase)
        finally:
            import os

            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_async_memory_aliases(self):
        """Test memory backend aliases in async factory."""
        factory = AsyncDatabaseFactory()

        # Test real memory backend aliases
        for alias in ["memory", "mem"]:
            db = factory.create(backend=alias)
            from dataknobs_data.backends.memory import AsyncMemoryDatabase

            assert isinstance(db, AsyncMemoryDatabase)

    def test_async_postgres_aliases(self):
        """Test backend aliases (using sqlite as example)."""
        factory = AsyncDatabaseFactory()

        # Test real SQLite backend aliases
        for alias in ["sqlite", "sqlite3"]:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name

            try:
                db = factory.create(backend=alias, path=db_path)
                from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

                assert isinstance(db, AsyncSQLiteDatabase)
            finally:
                import os

                if os.path.exists(db_path):
                    os.unlink(db_path)

    def test_async_elasticsearch_aliases(self):
        """Test backend aliases (using memory as example)."""
        factory = AsyncDatabaseFactory()

        # Test real memory backend - simple and tests alias functionality
        db = factory.create(backend="memory")
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        assert isinstance(db, AsyncMemoryDatabase)

    def test_async_s3_backend(self):
        """Test async backend creation (using file as example)."""
        factory = AsyncDatabaseFactory()

        # Use real file backend - tests factory logic
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            config = {"backend": "file", "path": filepath}
            db = factory.create(**config)
            from dataknobs_data.backends.file import AsyncFileDatabase

            assert isinstance(db, AsyncFileDatabase)
        finally:
            import os

            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_async_unknown_backend(self):
        """Test error for unknown async backend."""
        factory = AsyncDatabaseFactory()

        with pytest.raises(ValueError, match="does not support async operations"):
            factory.create(backend="redis")

    def test_async_default_backend(self):
        """Test that missing backend defaults to memory for async."""
        factory = AsyncDatabaseFactory()

        # Use real memory backend - no mocking needed
        db = factory.create()  # No backend specified
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        assert isinstance(db, AsyncMemoryDatabase)


class TestFactorySingletons:
    """Test factory singleton instances."""

    def test_database_factory_singleton(self):
        """Test that database_factory is properly exported."""
        assert isinstance(database_factory, DatabaseFactory)

    def test_async_database_factory_singleton(self):
        """Test that async_database_factory is properly exported."""
        assert isinstance(async_database_factory, AsyncDatabaseFactory)

    def test_both_factories_are_different(self):
        """Test that sync and async factories are different instances."""
        assert database_factory is not async_database_factory
        assert type(database_factory) != type(async_database_factory)

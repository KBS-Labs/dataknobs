"""Test that all backends have consistent streaming support."""

from __future__ import annotations

import inspect
from collections.abc import Generator

import pytest

from dataknobs_common.registry import PluginRegistry
from dataknobs_data.backends import async_backends, sync_backends


def _backend_items(
    registry: "PluginRegistry[type]",
) -> "Generator[tuple[str, type], None, None]":
    """Yield (name, class) pairs from a PluginRegistry."""
    for key in registry.list_keys():
        yield key, registry.get_factory(key)


class TestBackendConsistency:
    """Test that all backends follow consistent patterns."""

    def test_all_backends_have_both_versions(self):
        """Verify each backend has both sync and async versions."""
        # S3, memory, file, postgres, elasticsearch should all have both versions
        expected_backends = {"memory", "file", "s3", "postgres", "elasticsearch"}

        for backend in expected_backends:
            assert async_backends.is_registered(backend) or backend in ["postgresql", "es"], \
                f"Backend {backend} missing from async registry"
            assert sync_backends.is_registered(backend) or backend in ["postgresql", "es"], \
                f"Backend {backend} missing from sync registry"

    def test_async_backends_have_streaming_methods(self):
        """Verify all async backends have streaming methods."""
        required_methods = ["stream_read", "stream_write"]

        for backend_name, backend_class in _backend_items(async_backends):
            # Check if methods exist
            for method in required_methods:
                assert hasattr(backend_class, method), \
                    f"Async backend {backend_name} ({backend_class.__name__}) missing {method} method"

            # Check method signatures - need to check on unbound methods
            stream_read = backend_class.stream_read
            stream_write = backend_class.stream_write

            # stream_read should be async (either coroutine or async generator)
            assert inspect.iscoroutinefunction(stream_read) or inspect.isasyncgenfunction(stream_read), \
                f"{backend_class.__name__}.stream_read should be async (coroutine or async generator)"

            # stream_write should be async and return StreamResult
            assert inspect.iscoroutinefunction(stream_write), \
                f"{backend_class.__name__}.stream_write should be async"

    def test_sync_backends_have_streaming_methods(self):
        """Verify all sync backends have streaming methods."""
        required_methods = ["stream_read", "stream_write"]

        for backend_name, backend_class in _backend_items(sync_backends):
            # Check if methods exist
            for method in required_methods:
                assert hasattr(backend_class, method), \
                    f"Sync backend {backend_name} ({backend_class.__name__}) missing {method} method"

            # Check method signatures - need to check on unbound methods
            stream_read = backend_class.stream_read
            stream_write = backend_class.stream_write

            # stream_read should NOT be async
            assert not inspect.iscoroutinefunction(stream_read), \
                f"{backend_class.__name__}.stream_read should not be async"

            # stream_write should NOT be async
            assert not inspect.iscoroutinefunction(stream_write), \
                f"{backend_class.__name__}.stream_write should not be async"

    def test_backend_naming_convention(self):
        """Verify backend classes follow naming conventions."""
        # Async backends should not have 'Sync' prefix
        for backend_name, backend_class in _backend_items(async_backends):
            assert not backend_class.__name__.startswith("Sync"), \
                f"Async backend {backend_class.__name__} should not start with 'Sync'"

        # Sync backends should have 'Sync' prefix (except S3Database which was historically sync)
        for backend_name, backend_class in _backend_items(sync_backends):
            assert backend_class.__name__.startswith("Sync"), \
                f"Sync backend {backend_class.__name__} should start with 'Sync'"

    def test_backend_inheritance(self):
        """Verify backends inherit from correct base classes."""
        from dataknobs_data.database import AsyncDatabase, SyncDatabase

        # Check async backends inherit from AsyncDatabase
        for backend_name, backend_class in _backend_items(async_backends):
            assert issubclass(backend_class, AsyncDatabase), \
                f"Async backend {backend_class.__name__} should inherit from AsyncDatabase"

        # Check sync backends inherit from SyncDatabase
        for backend_name, backend_class in _backend_items(sync_backends):
            assert issubclass(backend_class, SyncDatabase), \
                f"Sync backend {backend_class.__name__} should inherit from SyncDatabase"

    def test_streaming_method_signatures(self):
        """Verify streaming methods have consistent signatures."""
        # Check async backends
        for backend_name, backend_class in _backend_items(async_backends):
            stream_read = getattr(backend_class, "stream_read")
            stream_write = getattr(backend_class, "stream_write")

            # Check parameters
            read_sig = inspect.signature(stream_read)
            write_sig = inspect.signature(stream_write)

            # stream_read should have query and config parameters
            assert "query" in read_sig.parameters, \
                f"{backend_class.__name__}.stream_read missing 'query' parameter"
            assert "config" in read_sig.parameters, \
                f"{backend_class.__name__}.stream_read missing 'config' parameter"

            # stream_write should have records and config parameters
            assert "records" in write_sig.parameters, \
                f"{backend_class.__name__}.stream_write missing 'records' parameter"
            assert "config" in write_sig.parameters, \
                f"{backend_class.__name__}.stream_write missing 'config' parameter"

        # Check sync backends
        for backend_name, backend_class in _backend_items(sync_backends):
            stream_read = getattr(backend_class, "stream_read")
            stream_write = getattr(backend_class, "stream_write")

            # Check parameters
            read_sig = inspect.signature(stream_read)
            write_sig = inspect.signature(stream_write)

            # stream_read should have query and config parameters
            assert "query" in read_sig.parameters, \
                f"{backend_class.__name__}.stream_read missing 'query' parameter"
            assert "config" in read_sig.parameters, \
                f"{backend_class.__name__}.stream_read missing 'config' parameter"

            # stream_write should have records and config parameters
            assert "records" in write_sig.parameters, \
                f"{backend_class.__name__}.stream_write missing 'records' parameter"
            assert "config" in write_sig.parameters, \
                f"{backend_class.__name__}.stream_write missing 'config' parameter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

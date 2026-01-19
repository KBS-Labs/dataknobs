"""Registry module for bot registration storage and management.

This module provides:
- Registration: Dataclass for bot registration with metadata
- RegistryBackend: Protocol for pluggable storage backends
- InMemoryBackend: Simple dict-based storage for testing/development
- DataKnobsRegistryAdapter: Adapter wrapping any dataknobs_data backend
- HTTPRegistryBackend: HTTP/REST backend for external config services
- CachingRegistryManager: Base class for caching with event-driven invalidation
- ConfigCachingManager: Caching manager for resolved configurations
- RegistryPoller: Polling-based change detection
- HotReloadManager: Coordinator for hot-reloading configurations
- create_registry_backend: Factory function for creating backends
- Portability validation utilities

Example:
    ```python
    from dataknobs_bots.registry import Registration, InMemoryBackend

    backend = InMemoryBackend()
    await backend.initialize()

    reg = await backend.register("my-bot", {"llm": {...}})
    print(f"Bot registered at {reg.created_at}")
    ```

Example with create_registry_backend factory:
    ```python
    from dataknobs_bots.registry import create_registry_backend

    # Memory backend for testing
    backend = create_registry_backend("memory", {})

    # PostgreSQL backend for production
    backend = create_registry_backend("postgres", {
        "host": "localhost",
        "database": "myapp",
    })

    # HTTP backend for external config service
    backend = create_registry_backend("http", {
        "base_url": "https://config-service/api/v1",
        "auth_token": "secret",
    })

    await backend.initialize()
    config = await backend.get_config("my-bot")
    ```

Example with CachingRegistryManager:
    ```python
    from dataknobs_bots.registry import CachingRegistryManager, InMemoryBackend

    class MyManager(CachingRegistryManager[MyInstance]):
        async def _create_instance(self, id: str, config: dict) -> MyInstance:
            return await MyInstance.from_config(config)

        async def _destroy_instance(self, instance: MyInstance) -> None:
            await instance.close()

    manager = MyManager(backend=InMemoryBackend())
    await manager.initialize()
    instance = await manager.get_or_create("my-id")
    ```

Example with HotReloadManager:
    ```python
    from dataknobs_bots.registry import (
        ConfigCachingManager,
        HotReloadManager,
        InMemoryBackend,
        ReloadMode,
    )

    backend = InMemoryBackend()
    manager = ConfigCachingManager(backend=backend)
    hot_reload = HotReloadManager(
        caching_manager=manager,
        backend=backend,
        mode=ReloadMode.POLLING,
    )

    await backend.initialize()
    await manager.initialize()
    await hot_reload.initialize()

    # Changes will be auto-detected and cached instances refreshed
    ```
"""

from __future__ import annotations

from typing import Any

from .adapter import DataKnobsRegistryAdapter
from .backend import RegistryBackend
from .caching import CachingRegistryManager
from .config_caching import ConfigCachingManager, ResolvedConfig
from .hot_reload import HotReloadManager, ReloadMode
from .http_backend import HTTPRegistryBackend
from .memory import InMemoryBackend
from .models import Registration
from .polling import RegistryPoller
from .portability import (
    PortabilityError,
    has_resource_references,
    is_portable,
    validate_portability,
)


def create_registry_backend(
    backend_type: str,
    config: dict[str, Any],
) -> RegistryBackend:
    """Create a registry backend from type and configuration.

    This factory function creates the appropriate backend implementation
    based on the backend_type parameter.

    Args:
        backend_type: Type of backend to create:
            - "memory": InMemoryBackend for testing
            - "postgres", "s3", "sqlite", "file": DataKnobsRegistryAdapter
            - "http": HTTPRegistryBackend for external services
        config: Backend-specific configuration dict

    Returns:
        RegistryBackend implementation

    Raises:
        ValueError: If backend_type is unknown

    Example:
        ```python
        # Memory backend
        backend = create_registry_backend("memory", {})

        # PostgreSQL backend
        backend = create_registry_backend("postgres", {
            "host": "localhost",
            "database": "myapp",
            "table": "bot_configs",
        })

        # HTTP backend
        backend = create_registry_backend("http", {
            "base_url": "https://config-service/api/v1",
            "auth_token": "secret-token",
        })
        ```
    """
    backend_type = backend_type.lower()

    if backend_type == "memory":
        return InMemoryBackend()

    if backend_type == "http":
        return HTTPRegistryBackend.from_config(config)

    # Use DataKnobsRegistryAdapter for data backends
    if backend_type in ("postgres", "postgresql", "s3", "sqlite", "file"):
        return DataKnobsRegistryAdapter(
            backend_type=backend_type,
            backend_config=config,
        )

    raise ValueError(
        f"Unknown backend type: {backend_type}. "
        f"Supported types: memory, postgres, s3, sqlite, file, http"
    )


__all__ = [
    "Registration",
    "RegistryBackend",
    "InMemoryBackend",
    "DataKnobsRegistryAdapter",
    "HTTPRegistryBackend",
    "CachingRegistryManager",
    "ConfigCachingManager",
    "ResolvedConfig",
    "RegistryPoller",
    "HotReloadManager",
    "ReloadMode",
    "create_registry_backend",
    "PortabilityError",
    "validate_portability",
    "has_resource_references",
    "is_portable",
]

"""Common utilities and base classes for dataknobs packages.

This package provides shared functionality used across all dataknobs packages:

- **Exceptions**: Unified exception hierarchy with context support
- **Registry**: Generic registry pattern for managing named items
- **Serialization**: Protocols and utilities for to_dict/from_dict patterns
- **Testing**: Test utilities, markers, and configuration factories

Example:
    ```python
    from dataknobs_common import DataknobsError, Registry, serialize

    # Use common exceptions
    raise DataknobsError("Something went wrong", context={"details": "here"})

    # Create a registry
    registry = Registry[MyType]("my_registry")
    registry.register("key", my_item)

    # Serialize objects
    data = serialize(my_object)
    ```
"""

# Import all public APIs from submodules
from dataknobs_common.exceptions import (
    ConcurrencyError,
    ConfigurationError,
    DataknobsError,
    NotFoundError,
    OperationError,
    ResourceError,
    SerializationError,
    TimeoutError,
    ValidationError,
)
from dataknobs_common.registry import (
    AsyncRegistry,
    CachedRegistry,
    PluginRegistry,
    Registry,
)
from dataknobs_common.serialization import (
    Serializable,
    deserialize,
    deserialize_list,
    is_deserializable,
    is_serializable,
    serialize,
    serialize_list,
)
from dataknobs_common.testing import (
    create_test_json_files,
    create_test_markdown_files,
    get_test_bot_config,
    get_test_rag_config,
    is_chromadb_available,
    is_faiss_available,
    is_ollama_available,
    is_ollama_model_available,
    is_package_available,
    is_redis_available,
    requires_chromadb,
    requires_faiss,
    requires_ollama,
    requires_ollama_model,
    requires_package,
    requires_redis,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "DataknobsError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    "NotFoundError",
    "OperationError",
    "ConcurrencyError",
    "SerializationError",
    "TimeoutError",
    # Registry
    "Registry",
    "CachedRegistry",
    "AsyncRegistry",
    "PluginRegistry",
    # Serialization
    "Serializable",
    "serialize",
    "deserialize",
    "serialize_list",
    "deserialize_list",
    "is_serializable",
    "is_deserializable",
    # Testing - Availability Checks
    "is_ollama_available",
    "is_ollama_model_available",
    "is_faiss_available",
    "is_chromadb_available",
    "is_redis_available",
    "is_package_available",
    # Testing - Pytest Markers
    "requires_ollama",
    "requires_faiss",
    "requires_chromadb",
    "requires_redis",
    "requires_package",
    "requires_ollama_model",
    # Testing - Configuration Factories
    "get_test_bot_config",
    "get_test_rag_config",
    # Testing - File Helpers
    "create_test_markdown_files",
    "create_test_json_files",
]

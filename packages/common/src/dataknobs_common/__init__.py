"""Common utilities and base classes for dataknobs packages.

This package provides shared functionality used across all dataknobs packages:

- **Exceptions**: Unified exception hierarchy with context support
- **Registry**: Generic registry pattern for managing named items
- **Serialization**: Protocols and utilities for to_dict/from_dict patterns

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
    # Serialization
    "Serializable",
    "serialize",
    "deserialize",
    "serialize_list",
    "deserialize_list",
    "is_serializable",
    "is_deserializable",
]

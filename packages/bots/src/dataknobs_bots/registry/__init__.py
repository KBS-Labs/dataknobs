"""Registry module for bot registration storage and management.

This module provides:
- Registration: Dataclass for bot registration with metadata
- RegistryBackend: Protocol for pluggable storage backends
- InMemoryBackend: Simple dict-based storage for testing/development
- Portability validation utilities

Example:
    ```python
    from dataknobs_bots.registry import Registration, InMemoryBackend

    backend = InMemoryBackend()
    await backend.initialize()

    reg = await backend.register("my-bot", {"llm": {...}})
    print(f"Bot registered at {reg.created_at}")
    ```
"""

from __future__ import annotations

from .backend import RegistryBackend
from .memory import InMemoryBackend
from .models import Registration
from .portability import (
    PortabilityError,
    has_resource_references,
    is_portable,
    validate_portability,
)

__all__ = [
    "Registration",
    "RegistryBackend",
    "InMemoryBackend",
    "PortabilityError",
    "validate_portability",
    "has_resource_references",
    "is_portable",
]

"""Knowledge resource storage backends.

This module provides storage backends for knowledge resource files.
Use create_knowledge_backend() to create a backend from configuration.

Available Backends:
    - InMemoryKnowledgeBackend: For testing (no external dependencies)
    - FileKnowledgeBackend: For local development (file system)
    - S3KnowledgeBackend: For production (Amazon S3 or compatible)

Example:
    ```python
    from dataknobs_bots.knowledge.storage import (
        create_knowledge_backend,
        KnowledgeResourceBackend,
        KnowledgeFile,
        KnowledgeBaseInfo,
    )

    # Create backend from config
    backend = create_knowledge_backend("file", {"path": "./data/knowledge"})
    await backend.initialize()

    # Create a knowledge base
    await backend.create_kb("my-domain")

    # Upload a file
    file_info = await backend.put_file(
        "my-domain",
        "content/intro.md",
        b"# Hello World"
    )

    await backend.close()
    ```
"""

from __future__ import annotations

from .backend import KnowledgeResourceBackend
from .file import FileKnowledgeBackend
from .memory import InMemoryKnowledgeBackend
from .models import IngestionStatus, KnowledgeBaseInfo, KnowledgeFile
from .s3 import S3KnowledgeBackend


def create_knowledge_backend(
    backend_type: str,
    config: dict | None = None,
) -> KnowledgeResourceBackend:
    """Create knowledge resource backend from configuration.

    Factory function to create the appropriate backend based on type.

    Args:
        backend_type: Type of backend ("memory", "file", "s3")
        config: Backend-specific configuration:
            - memory: No config needed
            - file: {"path": "/path/to/storage"}
            - s3: {"bucket": "name", "prefix": "knowledge/", "region": "us-east-1"}

    Returns:
        Configured KnowledgeResourceBackend implementation

    Raises:
        ValueError: If backend_type is not recognized

    Example:
        ```python
        # For testing
        backend = create_knowledge_backend("memory")

        # For development
        backend = create_knowledge_backend("file", {"path": "./data/kb"})

        # For production
        backend = create_knowledge_backend("s3", {
            "bucket": "my-bucket",
            "prefix": "knowledge/",
            "region": "us-east-1"
        })

        await backend.initialize()
        # ... use backend ...
        await backend.close()
        ```
    """
    backend_type_lower = backend_type.lower()
    config = config or {}

    if backend_type_lower == "memory":
        return InMemoryKnowledgeBackend.from_config(config)
    elif backend_type_lower == "file":
        return FileKnowledgeBackend.from_config(config)
    elif backend_type_lower == "s3":
        return S3KnowledgeBackend.from_config(config)
    else:
        raise ValueError(
            f"Unknown knowledge backend type: {backend_type}. "
            f"Available types: memory, file, s3"
        )


__all__ = [
    # Protocol
    "KnowledgeResourceBackend",
    # Models
    "KnowledgeFile",
    "KnowledgeBaseInfo",
    "IngestionStatus",
    # Factory
    "create_knowledge_backend",
    # Backends
    "InMemoryKnowledgeBackend",
    "FileKnowledgeBackend",
    "S3KnowledgeBackend",
]

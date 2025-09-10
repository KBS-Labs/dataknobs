"""I/O abstraction layer for FSM patterns.

This module provides a unified abstraction layer for handling various I/O operations
including read/write, streaming, sync/async operations across different data sources.
"""

from .base import (
    IOMode,
    IOFormat,
    IOProvider,
    IOConfig,
    AsyncIOProvider,
    SyncIOProvider,
)
from .adapters import (
    FileIOAdapter,
    DatabaseIOAdapter,
    HTTPIOAdapter,
    StreamIOAdapter,
)
from .utils import (
    create_io_provider,
    batch_iterator,
    async_batch_iterator,
    transform_pipeline,
    async_transform_pipeline,
)

__all__ = [
    # Base classes
    'IOMode',
    'IOFormat',
    'IOProvider',
    'IOConfig',
    'AsyncIOProvider',
    'SyncIOProvider',
    # Adapters
    'FileIOAdapter',
    'DatabaseIOAdapter',
    'HTTPIOAdapter',
    'StreamIOAdapter',
    # Utils
    'create_io_provider',
    'batch_iterator',
    'async_batch_iterator',
    'transform_pipeline',
    'async_transform_pipeline',
]

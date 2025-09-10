"""Base I/O abstraction layer components.

This module provides the base abstractions for unified I/O operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Union, AsyncIterator, Iterator,
    Callable, TypeVar, Protocol
)

T = TypeVar('T')


class IOMode(Enum):
    """I/O operation modes."""
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    STREAM = "stream"
    BATCH = "batch"


class IOFormat(Enum):
    """Supported I/O formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PARQUET = "parquet"
    AVRO = "avro"
    TEXT = "text"
    BINARY = "binary"
    DATABASE = "database"
    API = "api"


@dataclass
class IOConfig:
    """Configuration for I/O operations."""
    mode: IOMode
    format: IOFormat
    source: Union[str, Dict[str, Any]]  # Path, URL, or connection config
    target: Union[str, Dict[str, Any]] | None = None
    
    # Operation settings
    batch_size: int = 1000
    buffer_size: int = 8192
    timeout: float | None = None
    retry_count: int = 3
    retry_delay: float = 1.0
    
    # Format-specific settings
    encoding: str = "utf-8"
    compression: str | None = None
    delimiter: str = ","
    headers: Dict[str, str] | None = None
    
    # Advanced settings
    parallel_workers: int = 1
    checkpoint_enabled: bool = False
    checkpoint_interval: int = 10000
    error_handler: Callable[[Exception, Any], Any] | None = None
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)


class IOProvider(ABC):
    """Base I/O provider interface."""
    
    def __init__(self, config: IOConfig):
        """Initialize provider with configuration."""
        self.config = config
        self._is_open = False
        
    @abstractmethod
    def open(self) -> None:
        """Open the I/O connection."""
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Close the I/O connection."""
        pass
        
    @abstractmethod
    def validate(self) -> bool:
        """Validate the I/O configuration and connection."""
        pass
        
    @property
    def is_open(self) -> bool:
        """Check if connection is open."""
        return self._is_open
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncIOProvider(IOProvider):
    """Async I/O provider interface."""
    
    @abstractmethod
    async def read(self, **kwargs) -> Any:
        """Read data asynchronously."""
        pass
        
    @abstractmethod
    async def write(self, data: Any, **kwargs) -> None:
        """Write data asynchronously."""
        pass
        
    @abstractmethod
    async def stream_read(self, **kwargs) -> AsyncIterator[Any]:
        """Stream read data asynchronously."""
        pass
        
    @abstractmethod
    async def stream_write(self, data_stream: AsyncIterator[Any], **kwargs) -> None:
        """Stream write data asynchronously."""
        pass
        
    @abstractmethod
    async def batch_read(self, batch_size: int | None = None, **kwargs) -> AsyncIterator[List[Any]]:
        """Read data in batches asynchronously."""
        pass
        
    @abstractmethod
    async def batch_write(self, batches: AsyncIterator[List[Any]], **kwargs) -> None:
        """Write data in batches asynchronously."""
        pass
        
    async def open(self) -> None:
        """Open the async I/O connection."""
        self._is_open = True
        
    async def close(self) -> None:
        """Close the async I/O connection."""
        self._is_open = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.open()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class SyncIOProvider(IOProvider):
    """Synchronous I/O provider interface."""
    
    @abstractmethod
    def read(self, **kwargs) -> Any:
        """Read data synchronously."""
        pass
        
    @abstractmethod
    def write(self, data: Any, **kwargs) -> None:
        """Write data synchronously."""
        pass
        
    @abstractmethod
    def stream_read(self, **kwargs) -> Iterator[Any]:
        """Stream read data synchronously."""
        pass
        
    @abstractmethod
    def stream_write(self, data_stream: Iterator[Any], **kwargs) -> None:
        """Stream write data synchronously."""
        pass
        
    @abstractmethod
    def batch_read(self, batch_size: int | None = None, **kwargs) -> Iterator[List[Any]]:
        """Read data in batches synchronously."""
        pass
        
    @abstractmethod
    def batch_write(self, batches: Iterator[List[Any]], **kwargs) -> None:
        """Write data in batches synchronously."""
        pass
        
    def open(self) -> None:
        """Open the sync I/O connection."""
        self._is_open = True
        
    def close(self) -> None:
        """Close the sync I/O connection."""
        self._is_open = False


class TransformProtocol(Protocol):
    """Protocol for data transformations."""
    
    def transform(self, data: Any) -> Any:
        """Transform data."""
        ...
        
    async def async_transform(self, data: Any) -> Any:
        """Transform data asynchronously."""
        ...


class IOAdapter(ABC):
    """Base adapter for converting between different I/O providers."""
    
    @abstractmethod
    def adapt_config(self, config: IOConfig) -> Dict[str, Any]:
        """Adapt configuration for specific provider."""
        pass
        
    @abstractmethod
    def adapt_data(self, data: Any, direction: IOMode) -> Any:
        """Adapt data format for specific provider."""
        pass
        
    @abstractmethod
    def create_provider(self, config: IOConfig, is_async: bool = True) -> IOProvider:
        """Create appropriate provider instance."""
        pass

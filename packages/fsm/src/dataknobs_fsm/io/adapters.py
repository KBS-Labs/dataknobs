"""I/O adapters for specific data sources.

This module provides adapters for different I/O sources like files, databases, and APIs.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union, AsyncIterator, Iterator
import aiofiles
from dataknobs_data import AsyncDatabase, Record, Query

from .base import (
    IOConfig, IOMode, IOFormat, IOProvider,
    AsyncIOProvider, SyncIOProvider, IOAdapter
)


class FileIOAdapter(IOAdapter):
    """Adapter for file-based I/O operations."""
    
    def adapt_config(self, config: IOConfig) -> Dict[str, Any]:
        """Adapt configuration for file operations."""
        return {
            'path': config.source,
            'mode': self._get_file_mode(config.mode),
            'encoding': config.encoding,
            'buffering': config.buffer_size,
        }
        
    def adapt_data(self, data: Any, direction: IOMode) -> Any:
        """Adapt data format for file operations."""
        if self.format == IOFormat.JSON:
            if direction == IOMode.WRITE:
                return json.dumps(data)
            else:
                return json.loads(data) if isinstance(data, str) else data
        elif self.format == IOFormat.CSV:
            if direction == IOMode.WRITE:
                return self._dict_to_csv_row(data)
            else:
                return self._csv_row_to_dict(data)
        return data
        
    def create_provider(self, config: IOConfig, is_async: bool = True) -> IOProvider:
        """Create file I/O provider."""
        if is_async:
            return AsyncFileProvider(config)
        return SyncFileProvider(config)
        
    def _get_file_mode(self, mode: IOMode) -> str:
        """Convert IOMode to file mode string."""
        mode_map = {
            IOMode.READ: 'r',
            IOMode.WRITE: 'w',
            IOMode.APPEND: 'a',
            IOMode.STREAM: 'r',
            IOMode.BATCH: 'r',
        }
        return mode_map.get(mode, 'r')
        
    def _dict_to_csv_row(self, data: Dict[str, Any]) -> List[Any]:
        """Convert dictionary to CSV row."""
        return list(data.values())
        
    def _csv_row_to_dict(self, row: List[Any], headers: List[str] | None = None) -> Dict[str, Any]:
        """Convert CSV row to dictionary."""
        if headers:
            return dict(zip(headers, row, strict=False))
        return {f'col_{i}': val for i, val in enumerate(row)}


class AsyncFileProvider(AsyncIOProvider):
    """Async file I/O provider."""
    
    def __init__(self, config: IOConfig):
        super().__init__(config)
        self.file_handle = None
        self.adapter = FileIOAdapter()
        
    async def open(self) -> None:
        """Open file for async I/O."""
        mode = self.adapter._get_file_mode(self.config.mode)
        self.file_handle = await aiofiles.open(
            self.config.source,
            mode=mode,
            encoding=self.config.encoding
        )
        self._is_open = True
        
    async def close(self) -> None:
        """Close file handle."""
        if self.file_handle:
            await self.file_handle.close()  # type: ignore[unreachable]
        self._is_open = False
        
    async def validate(self) -> bool:
        """Validate file path and permissions."""
        path = Path(self.config.source)  # type: ignore
        if self.config.mode == IOMode.READ:
            return path.exists() and path.is_file()
        return path.parent.exists()
        
    async def read(self, **kwargs) -> Any:
        """Read entire file."""
        if not self.file_handle:
            await self.open()
        content = await self.file_handle.read()
        return self._parse_content(content)
        
    async def write(self, data: Any, **kwargs) -> None:
        """Write data to file."""
        if not self.file_handle:
            await self.open()
        content = self._format_content(data)
        await self.file_handle.write(content)
        
    async def stream_read(self, **kwargs) -> AsyncIterator[Any]:
        """Stream read file line by line."""
        if not self.file_handle:
            await self.open()
        async for line in self.file_handle:
            yield self._parse_line(line)
            
    async def stream_write(self, data_stream: AsyncIterator[Any], **kwargs) -> None:
        """Stream write data to file."""
        if not self.file_handle:
            await self.open()
        async for data in data_stream:
            content = self._format_content(data)
            await self.file_handle.write(content + '\n')
            
    async def batch_read(self, batch_size: int | None = None, **kwargs) -> AsyncIterator[List[Any]]:
        """Read file in batches."""
        batch_size = batch_size or self.config.batch_size
        batch = []
        async for item in self.stream_read(**kwargs):
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
            
    async def batch_write(self, batches: AsyncIterator[List[Any]], **kwargs) -> None:
        """Write data in batches."""
        async for batch in batches:
            for item in batch:
                await self.write(item, **kwargs)
                
    def _parse_content(self, content: str) -> Any:
        """Parse file content based on format."""
        if self.config.format == IOFormat.JSON:
            return json.loads(content)
        elif self.config.format == IOFormat.CSV:
            return list(csv.DictReader(content.splitlines()))
        return content
        
    def _parse_line(self, line: str) -> Any:
        """Parse single line based on format."""
        line = line.strip()
        if self.config.format == IOFormat.JSON:
            return json.loads(line)
        return line
        
    def _format_content(self, data: Any) -> str:
        """Format data for writing based on format."""
        if self.config.format == IOFormat.JSON:
            return json.dumps(data, indent=2)
        return str(data)


class SyncFileProvider(SyncIOProvider):
    """Synchronous file I/O provider."""
    
    def __init__(self, config: IOConfig):
        super().__init__(config)
        self.file_handle = None
        self.adapter = FileIOAdapter()
        
    def open(self) -> None:
        """Open file for sync I/O."""
        mode = self.adapter._get_file_mode(self.config.mode)
        self.file_handle = open(
            self.config.source,  # type: ignore
            mode=mode,
            encoding=self.config.encoding,
            buffering=self.config.buffer_size
        )
        self._is_open = True
        
    def close(self) -> None:
        """Close file handle."""
        if self.file_handle:
            self.file_handle.close()  # type: ignore[unreachable]
        self._is_open = False
        
    def validate(self) -> bool:
        """Validate file path and permissions."""
        path = Path(self.config.source)  # type: ignore
        if self.config.mode == IOMode.READ:
            return path.exists() and path.is_file()
        return path.parent.exists()
        
    def read(self, **kwargs) -> Any:
        """Read entire file."""
        if not self.file_handle:
            self.open()
        return self.file_handle.read()
        
    def write(self, data: Any, **kwargs) -> None:
        """Write data to file."""
        if not self.file_handle:
            self.open()
        self.file_handle.write(str(data))
        
    def stream_read(self, **kwargs) -> Iterator[Any]:
        """Stream read file line by line."""
        if not self.file_handle:
            self.open()
        for line in self.file_handle:
            yield line.strip()
            
    def stream_write(self, data_stream: Iterator[Any], **kwargs) -> None:
        """Stream write data to file."""
        if not self.file_handle:
            self.open()
        for data in data_stream:
            self.file_handle.write(str(data) + '\n')
            
    def batch_read(self, batch_size: int | None = None, **kwargs) -> Iterator[List[Any]]:
        """Read file in batches."""
        batch_size = batch_size or self.config.batch_size
        batch = []
        for item in self.stream_read(**kwargs):
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
            
    def batch_write(self, batches: Iterator[List[Any]], **kwargs) -> None:
        """Write data in batches."""
        for batch in batches:
            for item in batch:
                self.write(item, **kwargs)


class DatabaseIOAdapter(IOAdapter):
    """Adapter for database I/O operations."""
    
    def adapt_config(self, config: IOConfig) -> Dict[str, Any]:
        """Adapt configuration for database operations."""
        if isinstance(config.source, dict):
            return config.source
        # Parse connection string if needed
        return {'connection_string': config.source}
        
    def adapt_data(self, data: Any, direction: IOMode) -> Any:
        """Adapt data format for database operations."""
        if isinstance(data, Record):
            return data.to_dict() if direction == IOMode.WRITE else data
        return data
        
    def create_provider(self, config: IOConfig, is_async: bool = True) -> IOProvider:
        """Create database I/O provider."""
        if is_async:
            return AsyncDatabaseProvider(config)
        return SyncDatabaseProvider(config)


class AsyncDatabaseProvider(AsyncIOProvider):
    """Async database I/O provider."""
    
    def __init__(self, config: IOConfig):
        super().__init__(config)
        self.db = None
        self.adapter = DatabaseIOAdapter()
        
    async def open(self) -> None:
        """Open database connection."""
        db_config = self.adapter.adapt_config(self.config)
        self.db = await AsyncDatabase.create(
            db_config.get('type', 'postgresql'),
            db_config  # type: ignore
        )
        self._is_open = True
        
    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.db.close()  # type: ignore[unreachable]
        self._is_open = False
        
    async def validate(self) -> bool:
        """Validate database connection."""
        try:
            if self.db:
                # Test connection with simple query
                await self.db.execute("SELECT 1")  # type: ignore[unreachable]
                return True
        except Exception:
            return False
        return False
        
    async def read(self, query: Union[str, Query] = None, **kwargs) -> List[Dict[str, Any]]:
        """Read data from database."""
        if not self.db:
            await self.open()
        if isinstance(query, str):
            query = Query(query)  # type: ignore
        results = await self.db.read(query)
        return [r.to_dict() for r in results]
        
    async def write(self, data: Any, table: str = None, **kwargs) -> None:
        """Write data to database."""
        if not self.db:
            await self.open()
        if isinstance(data, dict):
            data = [data]
        for item in data:
            await self.db.upsert(table, item)
            
    async def stream_read(self, query: Union[str, Query] = None, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream read from database."""
        if not self.db:
            await self.open()
        if isinstance(query, str):
            query = Query(query)  # type: ignore
        async for record in self.db.stream_read(query):
            yield record.to_dict()
            
    async def stream_write(self, data_stream: AsyncIterator[Any], table: str = None, **kwargs) -> None:
        """Stream write to database."""
        if not self.db:
            await self.open()
        async for data in data_stream:
            await self.db.upsert(table, data)
            
    async def batch_read(self, query: Union[str, Query] = None, batch_size: int | None = None, **kwargs) -> AsyncIterator[List[Dict[str, Any]]]:
        """Read from database in batches."""
        batch_size = batch_size or self.config.batch_size
        batch = []
        async for item in self.stream_read(query, **kwargs):
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
            
    async def batch_write(self, batches: AsyncIterator[List[Any]], table: str = None, **kwargs) -> None:
        """Write to database in batches."""
        if not self.db:
            await self.open()
        async for batch in batches:
            # Use bulk insert if available
            await self.db.bulk_upsert(table, batch)


class SyncDatabaseProvider(SyncIOProvider):
    """Synchronous database I/O provider."""
    
    def __init__(self, config: IOConfig):
        super().__init__(config)
        self.db = None
        self.adapter = DatabaseIOAdapter()
        
    def open(self) -> None:
        """Open database connection."""
        import sqlite3
        # For sync operations, use sqlite3 as a simple fallback
        db_config = self.adapter.adapt_config(self.config)
        db_path = db_config.get('path', ':memory:')
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row  # Enable dict-like access
        self._is_open = True
        
    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()  # type: ignore[unreachable]
        self._is_open = False
        
    def validate(self) -> bool:
        """Validate database connection."""
        try:
            if self.db:
                # Test connection with simple query
                self.db.execute("SELECT 1").fetchone()  # type: ignore[unreachable]
                return True
        except Exception:
            return False
        return False
        
    def read(self, query: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Read data from database."""
        if not self.db:
            self.open()
        if not query:
            query = "SELECT * FROM data"
        cursor = self.db.execute(query)
        return [dict(row) for row in cursor.fetchall()]
        
    def write(self, data: Any, table: str = "data", **kwargs) -> None:
        """Write data to database."""
        if not self.db:
            self.open()
        if isinstance(data, dict):
            data = [data]
        for item in data:
            # Simple upsert using INSERT OR REPLACE
            columns = ', '.join(item.keys())
            placeholders = ', '.join(['?' for _ in item.keys()])
            query = f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})"
            self.db.execute(query, list(item.values()))
        self.db.commit()
            
    def stream_read(self, query: str = None, **kwargs) -> Iterator[Dict[str, Any]]:
        """Stream read from database."""
        if not self.db:
            self.open()
        if not query:
            query = "SELECT * FROM data"
        cursor = self.db.execute(query)
        for row in cursor:
            yield dict(row)
            
    def stream_write(self, data_stream: Iterator[Any], table: str = "data", **kwargs) -> None:
        """Stream write to database."""
        if not self.db:
            self.open()
        for data in data_stream:
            self.write(data, table, **kwargs)
            
    def batch_read(self, query: str = None, batch_size: int | None = None, **kwargs) -> Iterator[List[Dict[str, Any]]]:
        """Read from database in batches."""
        batch_size = batch_size or self.config.batch_size
        batch = []
        for item in self.stream_read(query, **kwargs):
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
            
    def batch_write(self, batches: Iterator[List[Any]], table: str = "data", **kwargs) -> None:
        """Write to database in batches."""
        if not self.db:
            self.open()
        for batch in batches:
            self.write(batch, table, **kwargs)


class HTTPIOAdapter(IOAdapter):
    """Adapter for HTTP/API I/O operations."""
    
    def adapt_config(self, config: IOConfig) -> Dict[str, Any]:
        """Adapt configuration for HTTP operations."""
        return {
            'url': config.source,
            'headers': config.headers or {},
            'timeout': config.timeout,
            'retry_count': config.retry_count,
        }
        
    def adapt_data(self, data: Any, direction: IOMode) -> Any:
        """Adapt data format for HTTP operations."""
        if direction == IOMode.WRITE and not isinstance(data, (str, bytes)):
            return json.dumps(data)
        elif direction == IOMode.READ and isinstance(data, bytes):
            return json.loads(data.decode('utf-8'))
        return data
        
    def create_provider(self, config: IOConfig, is_async: bool = True) -> IOProvider:
        """Create HTTP I/O provider."""
        if is_async:
            return AsyncHTTPProvider(config)
        return SyncHTTPProvider(config)


class AsyncHTTPProvider(AsyncIOProvider):
    """Async HTTP/API I/O provider."""
    
    def __init__(self, config: IOConfig):
        super().__init__(config)
        self.session = None
        self.adapter = HTTPIOAdapter()
        
    async def open(self) -> None:
        """Open HTTP session."""
        import aiohttp
        self.session = aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        self._is_open = True
        
    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()  # type: ignore[unreachable]
        self._is_open = False
        
    async def validate(self) -> bool:
        """Validate HTTP endpoint."""
        try:
            if self.session:
                async with self.session.head(self.config.source) as response:  # type: ignore[unreachable]
                    return response.status < 400
        except Exception:
            return False
        return False
        
    async def read(self, **kwargs) -> Any:
        """Read data from HTTP endpoint."""
        if not self.session:
            await self.open()
        async with self.session.get(self.config.source, **kwargs) as response:
            response.raise_for_status()
            if 'json' in response.content_type:
                return await response.json()
            return await response.text()
            
    async def write(self, data: Any, **kwargs) -> None:
        """Write data to HTTP endpoint."""
        if not self.session:
            await self.open()
        json_data = self.adapter.adapt_data(data, IOMode.WRITE)
        async with self.session.post(
            self.config.source,
            data=json_data,
            **kwargs
        ) as response:
            response.raise_for_status()
            
    async def stream_read(self, **kwargs) -> AsyncIterator[Any]:
        """Stream read from HTTP endpoint (e.g., SSE)."""
        if not self.session:
            await self.open()
        async with self.session.get(self.config.source, **kwargs) as response:
            response.raise_for_status()
            async for line in response.content:
                if line:
                    yield json.loads(line.decode('utf-8'))
                    
    async def stream_write(self, data_stream: AsyncIterator[Any], **kwargs) -> None:
        """Stream write to HTTP endpoint using chunked transfer encoding.
        
        This method supports both chunked upload for large files and
        streaming of multiple records to an API endpoint.
        
        Args:
            data_stream: Async iterator of data chunks or records
            **kwargs: Additional arguments including:
                - content_type: Content type for the upload (default: application/octet-stream)
                - chunk_size: Size of chunks for file uploads (default: 8192)
                - upload_mode: 'chunked' for file uploads, 'stream' for record streaming
        """
        upload_mode = kwargs.pop('upload_mode', 'stream')
        
        if upload_mode == 'chunked':
            await self._chunked_file_upload(data_stream, **kwargs)
        else:
            await self._stream_records(data_stream, **kwargs)
    
    async def _chunked_file_upload(self, data_stream: AsyncIterator[Any], **kwargs) -> None:
        """Upload a file using chunked transfer encoding.
        
        Args:
            data_stream: Async iterator yielding file chunks (bytes)
            **kwargs: Additional arguments
        """
        import aiohttp
        
        url = kwargs.get('url', self.config.source)
        headers = kwargs.get('headers', self.config.headers or {})
        headers['Transfer-Encoding'] = 'chunked'
        content_type = kwargs.get('content_type', 'application/octet-stream')
        headers['Content-Type'] = content_type
        
        async def chunk_generator():
            """Generate chunks for upload."""
            async for chunk in data_stream:
                if isinstance(chunk, str):
                    encoded_chunk = chunk.encode('utf-8')
                    yield encoded_chunk
                else:
                    yield chunk
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=chunk_generator(),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                return await response.json() if response.content_type == 'application/json' else await response.text()  # type: ignore
    
    async def _stream_records(self, data_stream: AsyncIterator[Any], **kwargs) -> None:
        """Stream individual records to an API endpoint.
        
        Args:
            data_stream: Async iterator yielding records
            **kwargs: Additional arguments
        """
        # Existing implementation for streaming individual records
        async for data in data_stream:
            await self.write(data, **kwargs)
    
    async def chunked_upload_from_file(self, file_path: str, chunk_size: int = 8192, **kwargs) -> Any:
        """Upload a file in chunks.
        
        Args:
            file_path: Path to the file to upload
            chunk_size: Size of each chunk in bytes
            **kwargs: Additional arguments for the upload
            
        Returns:
            Response from the server
        """
        import aiofiles
        
        async def read_chunks():
            """Read file in chunks."""
            async with aiofiles.open(file_path, 'rb') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        kwargs['upload_mode'] = 'chunked'
        kwargs['chunk_size'] = chunk_size
        return await self.stream_write(read_chunks(), **kwargs)
            
    async def batch_read(self, batch_size: int | None = None, **kwargs) -> AsyncIterator[List[Any]]:
        """Read from HTTP endpoint in batches (pagination)."""
        batch_size = batch_size or self.config.batch_size
        page = 0
        while True:
            params = kwargs.get('params', {})
            params.update({'page': page, 'limit': batch_size})
            kwargs['params'] = params
            
            data = await self.read(**kwargs)
            if not data:
                break
                
            yield data if isinstance(data, list) else [data]
            page += 1
            
    async def batch_write(self, batches: AsyncIterator[List[Any]], **kwargs) -> None:
        """Write to HTTP endpoint in batches."""
        async for batch in batches:
            # Send batch as single request
            await self.write(batch, **kwargs)


class SyncHTTPProvider(SyncIOProvider):
    """Synchronous HTTP/API I/O provider."""
    
    def __init__(self, config: IOConfig):
        super().__init__(config)
        self.session = None
        self.adapter = HTTPIOAdapter()
        
    def open(self) -> None:
        """Open HTTP session."""
        import requests
        self.session = requests.Session()
        if self.config.headers:
            self.session.headers.update(self.config.headers)
        self._is_open = True
        
    def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            self.session.close()  # type: ignore[unreachable]
        self._is_open = False
        
    def validate(self) -> bool:
        """Validate HTTP endpoint."""
        try:
            if self.session:
                response = self.session.head(  # type: ignore[unreachable]
                    self.config.source,
                    timeout=self.config.timeout or 30
                )
                return response.status_code < 400
        except Exception:
            return False
        return False
        
    def read(self, **kwargs) -> Any:
        """Read data from HTTP endpoint."""
        if not self.session:
            self.open()
        response = self.session.get(
            self.config.source,
            timeout=self.config.timeout or 30,
            **kwargs
        )
        response.raise_for_status()
        
        if 'json' in response.headers.get('content-type', '').lower():
            return response.json()
        return response.text
            
    def write(self, data: Any, **kwargs) -> None:
        """Write data to HTTP endpoint."""
        if not self.session:
            self.open()
        json_data = self.adapter.adapt_data(data, IOMode.WRITE)
        response = self.session.post(
            self.config.source,
            data=json_data,
            timeout=self.config.timeout or 30,
            **kwargs
        )
        response.raise_for_status()
            
    def stream_read(self, **kwargs) -> Iterator[Any]:
        """Stream read from HTTP endpoint."""
        if not self.session:
            self.open()
        response = self.session.get(
            self.config.source,
            stream=True,
            timeout=self.config.timeout or 30,
            **kwargs
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    yield json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    yield line.decode('utf-8')
                    
    def stream_write(self, data_stream: Iterator[Any], **kwargs) -> None:
        """Stream write to HTTP endpoint."""
        # For sync HTTP, write each item individually
        for data in data_stream:
            self.write(data, **kwargs)
            
    def batch_read(self, batch_size: int | None = None, **kwargs) -> Iterator[List[Any]]:
        """Read from HTTP endpoint in batches (pagination)."""
        batch_size = batch_size or self.config.batch_size
        page = 0
        while True:
            params = kwargs.get('params', {})
            params.update({'page': page, 'limit': batch_size})
            kwargs['params'] = params
            
            data = self.read(**kwargs)
            if not data:
                break
                
            yield data if isinstance(data, list) else [data]
            page += 1
            
    def batch_write(self, batches: Iterator[List[Any]], **kwargs) -> None:
        """Write to HTTP endpoint in batches."""
        for batch in batches:
            # Send batch as single request
            self.write(batch, **kwargs)


class StreamIOAdapter(IOAdapter):
    """Adapter for stream-based I/O operations."""
    
    def adapt_config(self, config: IOConfig) -> Dict[str, Any]:
        """Adapt configuration for stream operations."""
        return {
            'buffer_size': config.buffer_size,
            'chunk_size': config.batch_size,
        }
        
    def adapt_data(self, data: Any, direction: IOMode) -> Any:
        """Adapt data format for stream operations."""
        return data
        
    def create_provider(self, config: IOConfig, is_async: bool = True) -> IOProvider:
        """Create stream I/O provider."""
        # Determine the underlying source type and create appropriate provider
        if isinstance(config.source, str):
            if config.source.startswith(('http://', 'https://')):
                return HTTPIOAdapter().create_provider(config, is_async)
            else:
                return FileIOAdapter().create_provider(config, is_async)
        elif isinstance(config.source, dict):
            return DatabaseIOAdapter().create_provider(config, is_async)
        raise ValueError(f"Unsupported source type: {type(config.source)}")

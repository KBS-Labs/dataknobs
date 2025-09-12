"""File streaming implementation for FSM."""

import csv
import gzip
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union, TextIO, BinaryIO

from dataknobs_fsm.streaming.core import (
    IStreamSink,
    IStreamSource,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class FileFormat:
    """Supported file formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TEXT = "text"
    BINARY = "binary"
    
    @staticmethod
    def detect(file_path: Path) -> str:
        """Detect format from file extension.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Detected format string.
        """
        suffix = file_path.suffix.lower()
        if suffix in ['.json']:
            return FileFormat.JSON
        elif suffix in ['.jsonl', '.ndjson']:
            return FileFormat.JSONL
        elif suffix in ['.csv', '.tsv']:
            return FileFormat.CSV
        elif suffix in ['.txt', '.text', '.log']:
            return FileFormat.TEXT
        else:
            return FileFormat.BINARY


class CompressionFormat:
    """Supported compression formats."""
    NONE = "none"
    GZIP = "gzip"
    
    @staticmethod
    def detect(file_path: Path) -> str:
        """Detect compression from file extension.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Detected compression format.
        """
        if file_path.suffix.lower() in ['.gz', '.gzip']:
            return CompressionFormat.GZIP
        return CompressionFormat.NONE


class FileStreamSource(IStreamSource):
    """File-based stream source with format detection and decompression.
    
    This source supports reading files in chunks with automatic
    format detection, decompression, and progress tracking.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        format: str | None = None,
        compression: str | None = None,
        chunk_size: int = 1000,
        encoding: str = 'utf-8'
    ):
        """Initialize file stream source.
        
        Args:
            file_path: Path to source file.
            format: File format (auto-detected if None).
            compression: Compression format (auto-detected if None).
            chunk_size: Number of items per chunk.
            encoding: Text encoding for text formats.
        """
        self.file_path = Path(file_path)
        self.format = format or FileFormat.detect(self.file_path)
        self.compression = compression or CompressionFormat.detect(self.file_path)
        self.chunk_size = chunk_size
        self.encoding = encoding
        
        self._file_handle: Union[TextIO, BinaryIO, Any, None] = None
        self._reader: Any | None = None
        self._chunk_count = 0
        self._item_count = 0
        self._bytes_read = 0
        self._file_size = self.file_path.stat().st_size if self.file_path.exists() else 0
        
        self._open_file()
    
    def _open_file(self) -> None:
        """Open the file with appropriate decompression."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Open with decompression if needed
        if self.compression == CompressionFormat.GZIP:
            self._file_handle = gzip.open(self.file_path, 'rt', encoding=self.encoding)
        elif self.format == FileFormat.BINARY:
            self._file_handle = open(self.file_path, 'rb')
        else:
            self._file_handle = open(self.file_path, encoding=self.encoding)
        
        # Set up format-specific reader
        if self.format == FileFormat.CSV:
            self._reader = csv.DictReader(self._file_handle)  # type: ignore
        elif self.format == FileFormat.JSON:
            # Load entire JSON file (for array or object)
            content = self._file_handle.read()  # type: ignore
            data = json.loads(content)
            if isinstance(data, list):
                self._reader = iter(data)
            else:
                self._reader = iter([data])
        elif self.format == FileFormat.JSONL:
            # Will read line by line
            self._reader = self._file_handle
        elif self.format == FileFormat.TEXT or self.format == FileFormat.BINARY:
            self._reader = self._file_handle
        else:
            self._reader = self._file_handle
    
    def read_chunk(self) -> StreamChunk | None:
        """Read next chunk from file.
        
        Returns:
            StreamChunk with file data or None if exhausted.
        """
        if self._reader is None:
            return None
        
        chunk_data = []
        
        try:
            # Read up to chunk_size items
            for _ in range(self.chunk_size):
                item = self._read_next_item()
                if item is None:
                    break
                chunk_data.append(item)
                self._item_count += 1
            
            if not chunk_data:
                return None
            
            # Calculate progress
            progress = 0.0
            if self._file_size > 0 and self._file_handle:
                if hasattr(self._file_handle, 'tell'):
                    try:
                        current_pos = self._file_handle.tell()
                        progress = current_pos / self._file_size
                    except (OSError, io.UnsupportedOperation):
                        pass
            
            # Create chunk
            chunk = StreamChunk(
                data=chunk_data,
                sequence_number=self._chunk_count,
                metadata={
                    'file_path': str(self.file_path),
                    'format': self.format,
                    'progress': progress,
                    'item_count': len(chunk_data)
                },
                is_last=len(chunk_data) < self.chunk_size
            )
            
            self._chunk_count += 1
            return chunk
            
        except Exception as e:
            # Return error chunk
            return StreamChunk(
                data=[],
                sequence_number=self._chunk_count,
                metadata={'error': str(e)},
                is_last=True
            )
    
    def _read_next_item(self) -> Any | None:
        """Read next item based on format.
        
        Returns:
            Next item or None if exhausted.
        """
        try:
            if self.format == FileFormat.CSV:
                return next(self._reader, None) if self._reader else None
            elif self.format == FileFormat.JSONL:
                line = next(self._reader, None) if self._reader else None
                if line:
                    return json.loads(line.strip())
                return None
            elif self.format == FileFormat.TEXT:
                return next(self._reader, None) if self._reader else None
            elif self.format == FileFormat.BINARY:
                # Read in 4KB chunks for binary
                data = self._reader.read(4096) if self._reader else None
                if data:
                    return data
                return None
            else:
                # For JSON array, already using iterator
                return next(self._reader, None) if self._reader else None
        except StopIteration:
            return None
        except Exception:
            return None
    
    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over all chunks."""
        while True:
            chunk = self.read_chunk()
            if chunk is None:
                break
            yield chunk
    
    def close(self) -> None:
        """Close the file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self._reader = None


class FileStreamSink(IStreamSink):
    """File-based stream sink with format serialization and compression.
    
    This sink supports writing data chunks to files with automatic
    format serialization, compression, and atomic writes.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        format: str | None = None,
        compression: str | None = None,
        encoding: str = 'utf-8',
        atomic: bool = True,
        append: bool = False
    ):
        """Initialize file stream sink.
        
        Args:
            file_path: Path to target file.
            format: File format (auto-detected if None).
            compression: Compression format (auto-detected if None).
            encoding: Text encoding for text formats.
            atomic: Use atomic writes (write to temp then rename).
            append: Append to existing file instead of overwriting.
        """
        self.file_path = Path(file_path)
        self.format = format or FileFormat.detect(self.file_path)
        self.compression = compression or CompressionFormat.detect(self.file_path)
        self.encoding = encoding
        self.atomic = atomic
        self.append = append
        
        self._file_handle: Any | None = None
        self._writer: Any | None = None
        self._temp_path: Path | None = None
        self._chunk_count = 0
        self._item_count = 0
        self._bytes_written = 0
        self._buffer: List[Any] = []
        
        self._open_file()
    
    def _open_file(self) -> None:
        """Open file for writing."""
        # Create parent directories
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use temp file for atomic writes
        if self.atomic and not self.append:
            self._temp_path = self.file_path.with_suffix(
                self.file_path.suffix + '.tmp'
            )
            target_path = self._temp_path
        else:
            target_path = self.file_path
        
        # Open with compression if needed
        mode = 'ab' if self.append and self.format == FileFormat.BINARY else (
            'a' if self.append else 'w'
        )
        
        if self.compression == CompressionFormat.GZIP:
            if self.format == FileFormat.BINARY:
                self._file_handle = gzip.open(str(target_path), mode + 'b')
            else:
                self._file_handle = gzip.open(
                    str(target_path),
                    mode + 't',
                    encoding=self.encoding
                )
        elif self.format == FileFormat.BINARY:
            self._file_handle = open(str(target_path), mode + 'b')
        else:
            self._file_handle = open(str(target_path), mode, encoding=self.encoding)
        
        # Set up format-specific writer
        if self.format == FileFormat.CSV:
            # CSV writer will be initialized on first write
            self._writer = None
        elif self.format == FileFormat.JSON:
            # Buffer all data for JSON
            self._buffer = []
        else:
            self._writer = self._file_handle
    
    def write_chunk(self, chunk: StreamChunk) -> bool:
        """Write chunk to file.
        
        Args:
            chunk: Chunk to write.
            
        Returns:
            True if successful.
        """
        if self._file_handle is None:
            return False
        
        try:
            if not chunk.data:
                return True
            
            if self.format == FileFormat.CSV:
                self._write_csv_chunk(chunk.data)
            elif self.format == FileFormat.JSON:
                # Buffer for final write
                if isinstance(chunk.data, list):
                    self._buffer.extend(chunk.data)
                else:
                    self._buffer.append(chunk.data)
            elif self.format == FileFormat.JSONL:
                for item in chunk.data:
                    self._file_handle.write(json.dumps(item) + '\n')
            elif self.format == FileFormat.TEXT:
                for item in chunk.data:
                    if item is not None:
                        self._file_handle.write(str(item))
                        if not str(item).endswith('\n'):
                            self._file_handle.write('\n')
            elif self.format == FileFormat.BINARY:
                for item in chunk.data:
                    if isinstance(item, bytes):
                        self._file_handle.write(item)
                    else:
                        self._file_handle.write(str(item).encode(self.encoding))
            else:
                # Default text write
                for item in chunk.data:
                    self._file_handle.write(str(item) + '\n')
            
            self._chunk_count += 1
            self._item_count += len(chunk.data) if isinstance(chunk.data, list) else 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing chunk: {e}")
            return False
    
    def _write_csv_chunk(self, data: List[Dict[str, Any]]) -> None:
        """Write CSV data.
        
        Args:
            data: List of dictionaries to write.
        """
        if not data:
            return
        
        # Initialize CSV writer on first write
        if self._writer is None:
            fieldnames = list(data[0].keys())
            self._writer = csv.DictWriter(
                self._file_handle,  # type: ignore
                fieldnames=fieldnames
            )
            if not self.append or self._chunk_count == 0:
                self._writer.writeheader()
        
        for row in data:
            self._writer.writerow(row)
    
    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._file_handle is None:
            return
        
        try:
            # Write JSON buffer if needed
            if self.format == FileFormat.JSON and self._buffer:
                json.dump(self._buffer, self._file_handle, indent=2)
                self._buffer = []
            
            # Flush file handle
            self._file_handle.flush()
            
            if hasattr(self._file_handle, 'fileno'):
                os.fsync(self._file_handle.fileno())
        except Exception:
            pass
    
    def close(self) -> None:
        """Close file and finalize atomic write."""
        if self._file_handle is None:
            return
        
        try:
            # Flush any remaining data
            self.flush()
            
            # Close file
            self._file_handle.close()
            
            # Atomic rename if using temp file
            if self.atomic and self._temp_path and self._temp_path.exists():
                self._temp_path.replace(self.file_path)
                self._temp_path = None
                
        except Exception:
            pass
        finally:
            self._file_handle = None
            self._writer = None


class DirectoryStreamSource(IStreamSource):
    """Stream source that reads from multiple files in a directory."""
    
    def __init__(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
        format: str | None = None,
        chunk_size: int = 1000
    ):
        """Initialize directory stream source.
        
        Args:
            directory: Directory path.
            pattern: File pattern to match.
            recursive: Search recursively.
            format: File format for all files.
            chunk_size: Chunk size for reading.
        """
        self.directory = Path(directory)
        self.pattern = pattern
        self.recursive = recursive
        self.format = format
        self.chunk_size = chunk_size
        
        # Find all matching files
        if recursive:
            self.files = list(self.directory.rglob(self.pattern))
        else:
            self.files = list(self.directory.glob(self.pattern))
        
        self.files = [f for f in self.files if f.is_file()]
        self.files.sort()
        
        self._current_file_index = 0
        self._current_source: FileStreamSource | None = None
        self._total_chunks = 0
    
    def read_chunk(self) -> StreamChunk | None:
        """Read next chunk from directory files.
        
        Returns:
            Next chunk or None if exhausted.
        """
        while self._current_file_index < len(self.files):
            # Open next file if needed
            if self._current_source is None:
                file_path = self.files[self._current_file_index]
                try:
                    self._current_source = FileStreamSource(
                        file_path,
                        format=self.format,
                        chunk_size=self.chunk_size
                    )
                except Exception:
                    self._current_file_index += 1
                    continue
            
            # Read chunk from current file
            chunk = self._current_source.read_chunk()
            
            if chunk is None:
                # Current file exhausted, move to next
                self._current_source.close()
                self._current_source = None
                self._current_file_index += 1
                continue
            
            # Add file info to metadata
            chunk.metadata['source_file'] = str(
                self.files[self._current_file_index]
            )
            chunk.metadata['file_index'] = self._current_file_index
            chunk.metadata['total_files'] = len(self.files)
            
            # Update is_last flag
            chunk.is_last = (
                self._current_file_index == len(self.files) - 1 and
                chunk.is_last
            )
            
            self._total_chunks += 1
            chunk.sequence_number = self._total_chunks
            
            return chunk
        
        return None
    
    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over all chunks."""
        while True:
            chunk = self.read_chunk()
            if chunk is None:
                break
            yield chunk
    
    def close(self) -> None:
        """Close current file source."""
        if self._current_source:
            self._current_source.close()
            self._current_source = None

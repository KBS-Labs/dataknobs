"""Streaming file utilities for processing large files efficiently.

This module provides memory-efficient streaming utilities for reading and writing
large files that may not fit in memory.
"""

import asyncio
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Tuple, Union

from dataknobs_fsm.streaming.core import StreamChunk, StreamConfig, StreamMetrics
from dataknobs_fsm.utils.file_utils import detect_format, get_csv_delimiter


class StreamingFileReader:
    """Memory-efficient streaming file reader with chunking support."""

    def __init__(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 1000,
        input_format: str = 'auto',
        text_field_name: str = 'text',
        csv_delimiter: str = ',',
        csv_has_header: bool = True,
        skip_empty_lines: bool = True,
        max_memory_mb: int = 100
    ):
        """Initialize streaming file reader.

        Args:
            file_path: Path to input file
            chunk_size: Number of records per chunk
            input_format: File format ('auto', 'jsonl', 'json', 'csv', 'text')
            text_field_name: Field name for text lines
            csv_delimiter: CSV delimiter character
            csv_has_header: Whether CSV has header row
            skip_empty_lines: Skip empty lines in text files
            max_memory_mb: Maximum memory usage in MB
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.text_field_name = text_field_name
        self.csv_delimiter = csv_delimiter
        self.csv_has_header = csv_has_header
        self.skip_empty_lines = skip_empty_lines
        self.max_memory_mb = max_memory_mb

        # Auto-detect format if needed
        if input_format == 'auto':
            self.format = detect_format(self.file_path)
            if self.format == 'csv' and self.file_path.suffix.lower() == '.tsv':
                self.csv_delimiter = '\t'
        else:
            self.format = input_format

        self.metrics = StreamMetrics()
        self._chunk_count = 0

    async def read_chunks(self) -> AsyncIterator[StreamChunk]:
        """Read file in chunks, yielding StreamChunk objects.

        Yields:
            StreamChunk objects containing batches of records
        """
        self.metrics.start_time = asyncio.get_event_loop().time()

        try:
            if self.format == 'jsonl':
                async for chunk in self._read_jsonl_chunks():
                    yield chunk
            elif self.format == 'json':
                async for chunk in self._read_json_chunks():
                    yield chunk
            elif self.format == 'csv':
                async for chunk in self._read_csv_chunks():
                    yield chunk
            elif self.format == 'text':
                async for chunk in self._read_text_chunks():
                    yield chunk
            else:
                raise ValueError(f"Unsupported format: {self.format}")
        finally:
            self.metrics.end_time = asyncio.get_event_loop().time()

    async def _read_jsonl_chunks(self) -> AsyncIterator[StreamChunk]:
        """Read JSONL file in chunks."""
        chunk_data = []

        with open(self.file_path) as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        chunk_data.append(record)
                        self.metrics.items_processed += 1

                        if len(chunk_data) >= self.chunk_size:
                            yield self._create_chunk(chunk_data)
                            chunk_data = []

                            # Allow other tasks to run
                            await asyncio.sleep(0)
                    except json.JSONDecodeError:
                        self.metrics.errors_count += 1
                        continue

            # Yield remaining data
            if chunk_data:
                yield self._create_chunk(chunk_data, is_last=True)

    async def _read_json_chunks(self) -> AsyncIterator[StreamChunk]:
        """Read JSON file in chunks (for arrays)."""
        import ijson

        with open(self.file_path, 'rb') as f:
            # First try to parse as an array with streaming
            try:
                parser = ijson.items(f, 'item')
                chunk_data = []
                item_count = 0

                for item in parser:
                    chunk_data.append(item)
                    item_count += 1
                    self.metrics.items_processed += 1

                    if len(chunk_data) >= self.chunk_size:
                        yield self._create_chunk(chunk_data)
                        chunk_data = []
                        await asyncio.sleep(0)

                if chunk_data:
                    yield self._create_chunk(chunk_data, is_last=True)
                elif item_count == 0:
                    # No items found, might be a single object
                    raise ValueError("No array items found")

            except (ijson.JSONError, ValueError):
                # Not an array or empty, try as single object
                f.seek(0)
                with open(self.file_path) as text_f:
                    data = json.load(text_f)

                if isinstance(data, list):
                    # It's an array, process in chunks
                    for i in range(0, len(data), self.chunk_size):
                        chunk = data[i:i + self.chunk_size]
                        is_last = (i + self.chunk_size) >= len(data)
                        self.metrics.items_processed += len(chunk)
                        yield self._create_chunk(chunk, is_last=is_last)
                        await asyncio.sleep(0)
                else:
                    # Single object
                    self.metrics.items_processed += 1
                    yield self._create_chunk([data], is_last=True)

    async def _read_csv_chunks(self) -> AsyncIterator[StreamChunk]:
        """Read CSV file in chunks."""
        chunk_data = []
        total_rows = 0

        with open(self.file_path, newline='') as f:
            if self.csv_has_header:
                reader = csv.DictReader(f, delimiter=self.csv_delimiter)
            else:
                # For headerless CSV, create field names
                first_line = f.readline()
                f.seek(0)
                num_fields = len(first_line.split(self.csv_delimiter))
                fieldnames = [f'col_{i}' for i in range(num_fields)]
                reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=self.csv_delimiter)

            # Count total rows for determining last chunk
            reader_list = list(reader)
            total_rows = len(reader_list)

            for idx, row in enumerate(reader_list):
                chunk_data.append(dict(row))  # Convert OrderedDict to dict
                self.metrics.items_processed += 1

                if len(chunk_data) >= self.chunk_size:
                    # Check if this will be the last chunk
                    is_last = (idx + 1) >= total_rows
                    yield self._create_chunk(chunk_data, is_last=is_last)
                    chunk_data = []
                    await asyncio.sleep(0)

            if chunk_data:
                yield self._create_chunk(chunk_data, is_last=True)

    async def _read_text_chunks(self) -> AsyncIterator[StreamChunk]:
        """Read text file in chunks."""
        chunk_data = []

        with open(self.file_path) as f:
            for line in f:
                sline = line.rstrip('\n\r')
                if sline or not self.skip_empty_lines:
                    chunk_data.append({self.text_field_name: sline})
                    self.metrics.items_processed += 1

                    if len(chunk_data) >= self.chunk_size:
                        yield self._create_chunk(chunk_data)
                        chunk_data = []
                        await asyncio.sleep(0)

            if chunk_data:
                yield self._create_chunk(chunk_data, is_last=True)

    def _create_chunk(self, data: List[Dict[str, Any]], is_last: bool = False) -> StreamChunk:
        """Create a StreamChunk from data."""
        chunk = StreamChunk(
            data=data,
            sequence_number=self._chunk_count,
            metadata={
                'file': str(self.file_path),
                'format': self.format,
                'chunk_size': len(data)
            },
            is_last=is_last
        )
        self._chunk_count += 1
        self.metrics.chunks_processed += 1
        return chunk


class StreamingFileWriter:
    """Memory-efficient streaming file writer with buffering."""

    def __init__(
        self,
        file_path: Union[str, Path],
        output_format: str | None = None,
        buffer_size: int = 1000,
        flush_interval: float = 1.0
    ):
        """Initialize streaming file writer.

        Args:
            file_path: Path to output file
            output_format: Output format (auto-detected if None)
            buffer_size: Number of records to buffer before writing
            flush_interval: Time interval (seconds) to flush buffer
        """
        self.file_path = Path(file_path)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        # Auto-detect format
        self.format = output_format or detect_format(self.file_path, for_output=True)

        self._buffer: deque = deque()
        self._file_handle: Any | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._last_flush_time = asyncio.get_event_loop().time()
        self._is_first_write = True
        self.metrics = StreamMetrics()

    async def __aenter__(self):
        """Async context manager entry."""
        self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def open(self):
        """Open the output file."""
        if self.format == 'jsonl':
            self._file_handle = open(self.file_path, 'w')
        elif self.format == 'csv':
            self._file_handle = open(self.file_path, 'w', newline='')
        elif self.format == 'json':
            self._file_handle = open(self.file_path, 'w')
            self._file_handle.write('[')  # Start JSON array
        elif self.format == 'text':
            self._file_handle = open(self.file_path, 'w')
        else:
            self._file_handle = open(self.file_path, 'w')

        self.metrics.start_time = asyncio.get_event_loop().time()

    async def write_chunk(self, chunk: StreamChunk) -> None:
        """Write a chunk of data to the file.

        Args:
            chunk: StreamChunk to write
        """
        if not self._file_handle:
            self.open()

        # Add chunk data to buffer
        if isinstance(chunk.data, list):
            self._buffer.extend(chunk.data)
        else:
            self._buffer.append(chunk.data)

        # Check if we should flush
        current_time = asyncio.get_event_loop().time()
        should_flush = (
            len(self._buffer) >= self.buffer_size or
            chunk.is_last or
            (current_time - self._last_flush_time) > self.flush_interval
        )

        if should_flush:
            await self._flush_buffer()
            self._last_flush_time = current_time

        self.metrics.chunks_processed += 1

    async def _flush_buffer(self) -> None:
        """Flush the buffer to file."""
        if not self._buffer or not self._file_handle:
            return

        if self.format == 'jsonl':
            # Write each record as a JSON line
            while self._buffer:
                record = self._buffer.popleft()
                json.dump(record, self._file_handle)
                self._file_handle.write('\n')
                self.metrics.items_processed += 1

        elif self.format == 'csv':
            # Initialize CSV writer if needed
            if self._csv_writer is None and self._buffer:
                first_record = self._buffer[0]
                fieldnames = list(first_record.keys())
                delimiter = get_csv_delimiter(self.file_path)
                self._csv_writer = csv.DictWriter(
                    self._file_handle,
                    fieldnames=fieldnames,
                    delimiter=delimiter
                )
                self._csv_writer.writeheader()

            # Write records
            while self._buffer:
                record = self._buffer.popleft()
                self._csv_writer.writerow(record)
                self.metrics.items_processed += 1

        elif self.format == 'json':
            # Write as JSON array elements
            while self._buffer:
                record = self._buffer.popleft()
                if not self._is_first_write:
                    self._file_handle.write(',')
                json.dump(record, self._file_handle)
                self._is_first_write = False
                self.metrics.items_processed += 1

        elif self.format == 'text':
            # Write text lines
            while self._buffer:
                record = self._buffer.popleft()
                # Extract text from dict if needed
                if isinstance(record, dict):
                    text = record.get('text', str(record))
                else:
                    text = str(record)
                self._file_handle.write(text + '\n')
                self.metrics.items_processed += 1

        # Flush to disk
        self._file_handle.flush()

        # Allow other tasks to run
        await asyncio.sleep(0)

    async def close(self) -> None:
        """Close the file and flush remaining buffer."""
        if self._buffer:
            await self._flush_buffer()

        if self._file_handle:
            if self.format == 'json':
                self._file_handle.write(']')  # Close JSON array

            self._file_handle.close()
            self._file_handle = None

        self.metrics.end_time = asyncio.get_event_loop().time()


class StreamingFileProcessor:
    """High-level streaming file processor combining reader and writer."""

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        chunk_size: int = 1000,
        input_format: str = 'auto',
        output_format: str | None = None
    ):
        """Initialize streaming file processor.

        Args:
            input_path: Input file path
            output_path: Output file path
            transform_fn: Optional transformation function for each record
            chunk_size: Records per chunk
            input_format: Input file format
            output_format: Output file format (auto-detected if None)
        """
        self.reader = StreamingFileReader(
            input_path,
            chunk_size=chunk_size,
            input_format=input_format
        )
        self.writer = StreamingFileWriter(
            output_path,
            output_format=output_format,
            buffer_size=chunk_size
        )
        self.transform_fn = transform_fn or (lambda x: x)

    async def process(self, progress_callback: Callable[[int, int], None] | None = None) -> StreamMetrics:
        """Process the file with streaming.

        Args:
            progress_callback: Optional callback for progress updates (items_processed, total_chunks)

        Returns:
            Combined metrics from processing
        """
        async with self.writer:
            total_items = 0

            async for chunk in self.reader.read_chunks():
                # Transform each record in the chunk
                transformed_data = []
                for record in chunk.data:
                    try:
                        transformed = self.transform_fn(record)
                        if transformed is not None:
                            transformed_data.append(transformed)
                    except Exception:
                        self.reader.metrics.errors_count += 1
                        continue

                # Create new chunk with transformed data
                if transformed_data:
                    transformed_chunk = StreamChunk(
                        data=transformed_data,
                        sequence_number=chunk.sequence_number,
                        metadata=chunk.metadata,
                        is_last=chunk.is_last
                    )
                    await self.writer.write_chunk(transformed_chunk)

                total_items += len(chunk.data)

                # Report progress
                if progress_callback:
                    progress_callback(total_items, self.reader._chunk_count)

        # Combine metrics
        combined_metrics = StreamMetrics(
            chunks_processed=self.reader.metrics.chunks_processed,
            items_processed=self.reader.metrics.items_processed,
            errors_count=self.reader.metrics.errors_count,
            start_time=self.reader.metrics.start_time,
            end_time=self.writer.metrics.end_time
        )

        return combined_metrics


# Convenience functions for SimpleFSM integration

async def create_streaming_file_reader(
    file_path: Union[str, Path],
    config: StreamConfig,
    **kwargs
) -> AsyncIterator[List[Dict[str, Any]]]:
    """Create a streaming file reader compatible with SimpleFSM.

    Args:
        file_path: Input file path
        config: Stream configuration
        **kwargs: Additional reader parameters

    Yields:
        Lists of records (chunks)
    """
    reader = StreamingFileReader(
        file_path,
        chunk_size=config.chunk_size,
        **kwargs
    )

    async for chunk in reader.read_chunks():
        yield chunk.data


async def create_streaming_file_writer(
    file_path: Union[str, Path],
    config: StreamConfig,
    **kwargs
) -> Tuple[Callable, Callable]:
    """Create a streaming file writer compatible with SimpleFSM.

    Args:
        file_path: Output file path
        config: Stream configuration
        **kwargs: Additional writer parameters

    Returns:
        Tuple of (write_fn, cleanup_fn)
    """
    writer = StreamingFileWriter(
        file_path,
        buffer_size=config.buffer_size,
        **kwargs
    )

    writer.open()

    async def write_fn(results: List[Dict[str, Any]]) -> None:
        """Write results to file."""
        chunk = StreamChunk(data=results)
        await writer.write_chunk(chunk)

    async def cleanup_fn() -> None:
        """Close and cleanup."""
        await writer.close()

    return write_fn, cleanup_fn

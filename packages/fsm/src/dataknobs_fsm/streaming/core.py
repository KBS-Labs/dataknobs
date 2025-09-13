"""Core streaming interfaces and implementations for FSM data processing."""

import asyncio
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Protocol, Union, runtime_checkable
from uuid import uuid4


class StreamStatus(Enum):
    """Stream processing status."""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for stream processing.
    
    Attributes:
        chunk_size: Number of items per chunk.
        buffer_size: Maximum items to buffer in memory.
        parallelism: Number of parallel workers for processing.
        memory_limit_mb: Maximum memory usage in MB.
        backpressure_threshold: Queue size that triggers backpressure.
        timeout_seconds: Maximum time for stream processing.
        enable_metrics: Whether to collect metrics.
        retry_on_error: Whether to retry failed chunks.
        max_retries: Maximum retry attempts for failed chunks.
    """
    chunk_size: int = 1000
    buffer_size: int = 10000
    parallelism: int = 1
    memory_limit_mb: int = 512
    backpressure_threshold: int = 5000
    timeout_seconds: float | None = None
    enable_metrics: bool = True
    retry_on_error: bool = True
    max_retries: int = 3


@dataclass
class StreamChunk:
    """A chunk of data in a stream.
    
    Attributes:
        data: The chunk data.
        chunk_id: Unique chunk identifier.
        sequence_number: Position in the stream.
        metadata: Additional chunk metadata.
        timestamp: Creation timestamp.
        is_last: Whether this is the last chunk.
    """
    data: Any
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    sequence_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    is_last: bool = False


@dataclass
class StreamMetrics:
    """Metrics for stream processing.
    
    Attributes:
        chunks_processed: Number of chunks processed.
        bytes_processed: Total bytes processed.
        items_processed: Total items processed.
        errors_count: Number of errors encountered.
        retries_count: Number of retries performed.
        start_time: Processing start timestamp.
        end_time: Processing end timestamp.
        peak_memory_mb: Peak memory usage in MB.
    """
    chunks_processed: int = 0
    bytes_processed: int = 0
    items_processed: int = 0
    errors_count: int = 0
    retries_count: int = 0
    start_time: float | None = None
    end_time: float | None = None
    peak_memory_mb: float = 0.0
    
    def duration_seconds(self) -> float | None:
        """Get processing duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time
    
    def throughput_items_per_second(self) -> float:
        """Calculate throughput in items per second."""
        duration = self.duration_seconds()
        if duration and duration > 0:
            return self.items_processed / duration
        return 0.0
    
    def throughput_mb_per_second(self) -> float:
        """Calculate throughput in MB per second."""
        duration = self.duration_seconds()
        if duration and duration > 0:
            return (self.bytes_processed / (1024 * 1024)) / duration
        return 0.0


@runtime_checkable
class IStreamSource(Protocol):
    """Interface for stream data sources."""
    
    def read_chunk(self) -> StreamChunk | None:
        """Read the next chunk from the source.
        
        Returns:
            StreamChunk if available, None if exhausted.
        """
        ...
    
    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over chunks."""
        ...
    
    def close(self) -> None:
        """Close the stream source."""
        ...


@runtime_checkable
class IStreamSink(Protocol):
    """Interface for stream data sinks."""
    
    def write_chunk(self, chunk: StreamChunk) -> bool:
        """Write a chunk to the sink.
        
        Args:
            chunk: The chunk to write.
            
        Returns:
            True if successful, False otherwise.
        """
        ...
    
    def flush(self) -> None:
        """Flush any buffered data."""
        ...
    
    def close(self) -> None:
        """Close the stream sink."""
        ...


class StreamContext:
    """Context for managing stream processing.
    
    This class coordinates stream sources, sinks, and processing
    with support for backpressure, parallelism, and metrics.
    """
    
    def __init__(self, config: StreamConfig | None = None):
        """Initialize stream context.
        
        Args:
            config: Stream configuration.
        """
        self.config = config or StreamConfig()
        self.status = StreamStatus.IDLE
        self.metrics = StreamMetrics()
        
        # Internal queues for processing
        self._input_queue: queue.Queue[StreamChunk | None] = queue.Queue(
            maxsize=self.config.buffer_size
        )
        self._output_queue: queue.Queue[StreamChunk | None] = queue.Queue(
            maxsize=self.config.buffer_size
        )
        
        # Threading support
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        
        # Registered processors
        self._processors: list[Callable[[StreamChunk], StreamChunk | None]] = []
        
        # Backpressure management
        self._backpressure_active = False
        self._last_backpressure_check = time.time()
    
    def add_processor(
        self,
        processor: Callable[[StreamChunk], StreamChunk | None]
    ) -> None:
        """Add a chunk processor function.
        
        Args:
            processor: Function to process chunks.
        """
        self._processors.append(processor)
    
    def _check_backpressure(self) -> None:
        """Check and handle backpressure."""
        current_time = time.time()
        if current_time - self._last_backpressure_check < 0.1:
            return  # Throttle checks
        
        self._last_backpressure_check = current_time
        
        # Check queue sizes
        input_size = self._input_queue.qsize()
        output_size = self._output_queue.qsize()
        
        if (input_size > self.config.backpressure_threshold or
            output_size > self.config.backpressure_threshold):
            if not self._backpressure_active:
                self._backpressure_active = True
                self.status = StreamStatus.PAUSED
                # Could implement more sophisticated backpressure handling
                time.sleep(0.01)  # Brief pause
        else:
            if self._backpressure_active:
                self._backpressure_active = False
                self.status = StreamStatus.ACTIVE
    
    def get_next_chunk(self) -> StreamChunk | None:
        """Get the next chunk from the stream.
        
        Returns:
            Next chunk or None if no more chunks.
        """
        try:
            # Try to get from input queue with a short timeout
            chunk = self._input_queue.get(timeout=0.001)
            return chunk
        except queue.Empty:
            return None
    
    def add_chunk(self, chunk: StreamChunk) -> bool:
        """Add a chunk to the input queue for processing.
        
        Args:
            chunk: The chunk to add.
            
        Returns:
            True if added successfully, False if queue is full.
        """
        try:
            self._input_queue.put(chunk, timeout=0.001)
            return True
        except queue.Full:
            return False
    
    def add_data(self, data: Any, chunk_id: str | None = None, is_last: bool = False) -> bool:
        """Add data as a chunk to the stream.
        
        Args:
            data: The data to add (will be wrapped in a StreamChunk).
            chunk_id: Optional chunk ID.
            is_last: Whether this is the last chunk.
            
        Returns:
            True if added successfully, False if queue is full.
        """
        import uuid
        chunk = StreamChunk(
            data=data if isinstance(data, list) else [data],
            chunk_id=chunk_id or str(uuid.uuid4()),
            is_last=is_last
        )
        return self.add_chunk(chunk)
    
    def _process_chunk(self, chunk: StreamChunk) -> StreamChunk | None:
        """Process a chunk through all processors.
        
        Args:
            chunk: The chunk to process.
            
        Returns:
            Processed chunk or None if filtered.
        """
        result = chunk
        for processor in self._processors:
            if result is None:
                break  # type: ignore[unreachable]
            try:
                result = processor(result)
            except Exception:
                self.metrics.errors_count += 1
                if self.config.retry_on_error:
                    # Simple retry logic
                    for retry in range(self.config.max_retries):
                        try:
                            self.metrics.retries_count += 1
                            result = processor(chunk)
                            break
                        except Exception:
                            if retry == self.config.max_retries - 1:
                                return None
                else:
                    return None
        
        return result
    
    def _worker_thread(self) -> None:
        """Worker thread for processing chunks."""
        while not self._stop_event.is_set():
            try:
                # Get chunk with timeout
                chunk = self._input_queue.get(timeout=0.1)
                if chunk is None:
                    # Poison pill - propagate to output
                    self._output_queue.put(None)
                    break
                
                # Process chunk
                processed = self._process_chunk(chunk)
                
                if processed is not None:
                    # Put in output queue
                    self._output_queue.put(processed)
                    
                    # Update metrics
                    with self._lock:
                        self.metrics.chunks_processed += 1
                        if hasattr(processed.data, '__len__'):
                            self.metrics.items_processed += len(processed.data)
                
                # Check backpressure
                self._check_backpressure()
                
            except queue.Empty:
                continue
            except Exception:
                with self._lock:
                    self.metrics.errors_count += 1
                    self.status = StreamStatus.ERROR
    
    def stream(
        self,
        source: IStreamSource, 
        sink: IStreamSink,
        transform: Callable[[Any], Any] | None = None
    ) -> StreamMetrics:
        """Stream data from source to sink with optional transformation.
        
        Args:
            source: Data source.
            sink: Data sink.
            transform: Optional transformation function.
            
        Returns:
            Stream processing metrics.
        """
        if transform:
            self.add_processor(lambda c: StreamChunk(
                data=transform(c.data),
                chunk_id=c.chunk_id,
                sequence_number=c.sequence_number,
                metadata=c.metadata,
                timestamp=c.timestamp,
                is_last=c.is_last
            ))
        
        # Start metrics
        self.metrics.start_time = time.time()
        self.status = StreamStatus.ACTIVE
        
        # Start worker threads
        for i in range(self.config.parallelism):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"stream-worker-{i}"
            )
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
        
        # Reader thread
        def read_thread():
            try:
                for chunk in source:
                    if self._stop_event.is_set():
                        break
                    self._input_queue.put(chunk)
                    if chunk.is_last:
                        break
            finally:
                # Send poison pills to workers
                for _ in range(self.config.parallelism):
                    self._input_queue.put(None)
                source.close()
        
        reader = threading.Thread(target=read_thread, name="stream-reader")
        reader.daemon = True
        reader.start()
        
        # Writer thread
        def write_thread():
            poison_pills = 0
            try:
                while poison_pills < self.config.parallelism:
                    chunk = self._output_queue.get(timeout=0.1)
                    if chunk is None:
                        poison_pills += 1
                        continue
                    
                    success = sink.write_chunk(chunk)
                    if not success:
                        with self._lock:
                            self.metrics.errors_count += 1
                    
                    if chunk.is_last:
                        break
            except Exception:
                with self._lock:
                    self.metrics.errors_count += 1
                    self.status = StreamStatus.ERROR
            finally:
                sink.flush()
                sink.close()
        
        writer = threading.Thread(target=write_thread, name="stream-writer")
        writer.daemon = True
        writer.start()
        
        # Wait for completion with optional timeout
        try:
            reader.join(timeout=self.config.timeout_seconds)
            for worker in self._workers:
                worker.join(timeout=1)
            writer.join(timeout=self.config.timeout_seconds)
        except Exception:
            self._stop_event.set()
            self.status = StreamStatus.ERROR
        finally:
            # Update final metrics
            self.metrics.end_time = time.time()
            if self.status != StreamStatus.ERROR:
                self.status = StreamStatus.COMPLETED
        
        return self.metrics
    
    @contextmanager
    def streaming_context(self):
        """Context manager for streaming operations.
        
        Yields:
            This StreamContext instance.
        """
        try:
            yield self
        finally:
            self.close()
    
    def close(self) -> None:
        """Close the stream context and clean up resources."""
        self._stop_event.set()
        
        # Wait briefly for threads to finish
        for worker in self._workers:
            worker.join(timeout=0.5)
        
        self.status = StreamStatus.COMPLETED
        self.metrics.end_time = self.metrics.end_time or time.time()


class AsyncStreamContext:
    """Async version of StreamContext for async/await support."""
    
    def __init__(self, config: StreamConfig | None = None):
        """Initialize async stream context.
        
        Args:
            config: Stream configuration.
        """
        self.config = config or StreamConfig()
        self.status = StreamStatus.IDLE
        self.metrics = StreamMetrics()
        
        # Async queues
        self._input_queue: asyncio.Queue[StreamChunk | None] = asyncio.Queue(
            maxsize=self.config.buffer_size
        )
        self._output_queue: asyncio.Queue[StreamChunk | None] = asyncio.Queue(
            maxsize=self.config.buffer_size
        )
        
        self._processors: list[Callable[[StreamChunk], StreamChunk | None]] = []
        self._stop_event = asyncio.Event()
    
    async def stream_async(
        self,
        source: AsyncIterator[StreamChunk],
        sink: Callable[[StreamChunk], bool],
        transform: Callable[[Any], Any] | None = None
    ) -> StreamMetrics:
        """Async streaming from source to sink.
        
        Args:
            source: Async data source iterator.
            sink: Sink function.
            transform: Optional transformation.
            
        Returns:
            Stream processing metrics.
        """
        if transform:
            self._processors.append(lambda c: StreamChunk(
                data=transform(c.data),
                chunk_id=c.chunk_id,
                sequence_number=c.sequence_number,
                metadata=c.metadata,
                timestamp=c.timestamp,
                is_last=c.is_last
            ))
        
        self.metrics.start_time = time.time()
        self.status = StreamStatus.ACTIVE
        
        # Create async tasks for reading, processing, and writing
        async def read_task():
            try:
                async for chunk in source:
                    if self._stop_event.is_set():
                        break
                    await self._input_queue.put(chunk)
                    if chunk.is_last:
                        break
            finally:
                # Send poison pills
                for _ in range(self.config.parallelism):
                    await self._input_queue.put(None)
        
        async def process_task():
            while not self._stop_event.is_set():
                chunk = await self._input_queue.get()
                if chunk is None:
                    await self._output_queue.put(None)
                    break
                
                # Process through all processors
                result = chunk
                for processor in self._processors:
                    if result:
                        result = processor(result)
                
                if result:
                    await self._output_queue.put(result)
                    self.metrics.chunks_processed += 1
        
        async def write_task():
            poison_pills = 0
            while poison_pills < self.config.parallelism:
                chunk = await self._output_queue.get()
                if chunk is None:
                    poison_pills += 1
                    continue
                
                if not sink(chunk):
                    self.metrics.errors_count += 1
                
                if chunk.is_last:
                    break
        
        # Run all tasks concurrently
        tasks = [
            asyncio.create_task(read_task()),
            *[asyncio.create_task(process_task()) for _ in range(self.config.parallelism)],
            asyncio.create_task(write_task())
        ]
        
        try:
            await asyncio.gather(*tasks)
            self.status = StreamStatus.COMPLETED
        except Exception:
            self._stop_event.set()
            self.status = StreamStatus.ERROR
            for task in tasks:
                task.cancel()
        finally:
            self.metrics.end_time = time.time()
        
        return self.metrics


class BasicStreamProcessor:
    """Basic stream processor implementation."""
    
    def __init__(
        self,
        source: IStreamSource,
        sink: IStreamSink,
        transform_func: Union[Callable, None] = None,
        buffer_size: int = 1000
    ):
        """Initialize stream processor.
        
        Args:
            source: Stream source.
            sink: Stream sink.
            transform_func: Optional transformation function.
            buffer_size: Buffer size for processing.
        """
        self.source = source
        self.sink = sink
        self.transform_func = transform_func
        self.buffer_size = buffer_size
        self.processed_chunks = 0
        self.processed_records = 0
        self.errors = []
        
    def process(self) -> Dict[str, Any]:
        """Process the entire stream.
        
        Returns:
            Processing statistics.
        """
        start_time = time.time()
        
        try:
            # Process all chunks
            for chunk in self.source:
                try:
                    # Apply transformation if provided
                    chunk_to_write = chunk
                    if self.transform_func:
                        transformed_chunk = self.transform_func(chunk)
                        if transformed_chunk:
                            chunk_to_write = transformed_chunk
                    
                    # Write to sink
                    success = self.sink.write_chunk(chunk_to_write)
                    if success:
                        self.processed_chunks += 1
                        self.processed_records += len(chunk.data) if hasattr(chunk.data, '__len__') else 1
                    else:
                        self.errors.append(f"Failed to write chunk {self.processed_chunks}")
                        
                except Exception as e:
                    self.errors.append(f"Error processing chunk {self.processed_chunks}: {e!s}")
                    continue
            
            # Flush sink
            self.sink.flush()
            
        except Exception as e:
            self.errors.append(f"Stream processing error: {e!s}")
        finally:
            # Clean up
            self.source.close()
            self.sink.close()
            
        end_time = time.time()
        
        return {
            'processed_chunks': self.processed_chunks,
            'processed_records': self.processed_records,
            'duration': end_time - start_time,
            'errors': self.errors,
            'success': len(self.errors) == 0
        }
    
    async def process_async(self) -> Dict[str, Any]:
        """Process the stream asynchronously.
        
        Returns:
            Processing statistics.
        """
        # For now, just wrap sync processing
        # In a real implementation, this would use async iterators
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.process)


class MemoryStreamSource:
    """Simple in-memory stream source for testing."""
    
    def __init__(self, data: List[Any], chunk_size: int = 100):
        """Initialize with data.
        
        Args:
            data: List of data items.
            chunk_size: Size of each chunk.
        """
        self.data = data
        self.chunk_size = chunk_size
        self.current_index = 0
        
    def read_chunk(self) -> StreamChunk | None:
        """Read next chunk."""
        if self.current_index >= len(self.data):
            return None
            
        end_index = min(self.current_index + self.chunk_size, len(self.data))
        chunk_data = self.data[self.current_index:end_index]
        
        chunk = StreamChunk(
            data=chunk_data,
            chunk_id=f"chunk_{self.current_index // self.chunk_size}",
            timestamp=time.time(),
            is_last=end_index >= len(self.data)
        )
        
        self.current_index = end_index
        return chunk
    
    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over chunks."""
        while True:
            chunk = self.read_chunk()
            if chunk is None:
                break
            yield chunk
    
    def close(self) -> None:
        """Close source."""
        pass


class MemoryStreamSink:
    """Simple in-memory stream sink for testing."""
    
    def __init__(self):
        """Initialize sink."""
        self.chunks = []
        self.records = []
        
    def write_chunk(self, chunk: StreamChunk) -> bool:
        """Write chunk to memory."""
        try:
            self.chunks.append(chunk)
            if hasattr(chunk.data, '__iter__'):
                self.records.extend(chunk.data)
            else:
                self.records.append(chunk.data)
            return True
        except Exception:
            return False
    
    def flush(self) -> None:
        """Flush (no-op for memory)."""
        pass
    
    def close(self) -> None:
        """Close sink."""
        pass

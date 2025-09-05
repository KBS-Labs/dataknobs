"""Core streaming interfaces and implementations for FSM data processing."""

import asyncio
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Protocol, runtime_checkable
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
    timeout_seconds: Optional[float] = None
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
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    peak_memory_mb: float = 0.0
    
    def duration_seconds(self) -> Optional[float]:
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
    
    def read_chunk(self) -> Optional[StreamChunk]:
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
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize stream context.
        
        Args:
            config: Stream configuration.
        """
        self.config = config or StreamConfig()
        self.status = StreamStatus.IDLE
        self.metrics = StreamMetrics()
        
        # Internal queues for processing
        self._input_queue: queue.Queue[Optional[StreamChunk]] = queue.Queue(
            maxsize=self.config.buffer_size
        )
        self._output_queue: queue.Queue[Optional[StreamChunk]] = queue.Queue(
            maxsize=self.config.buffer_size
        )
        
        # Threading support
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        
        # Registered processors
        self._processors: list[Callable[[StreamChunk], Optional[StreamChunk]]] = []
        
        # Backpressure management
        self._backpressure_active = False
        self._last_backpressure_check = time.time()
    
    def add_processor(
        self,
        processor: Callable[[StreamChunk], Optional[StreamChunk]]
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
    
    def _process_chunk(self, chunk: StreamChunk) -> Optional[StreamChunk]:
        """Process a chunk through all processors.
        
        Args:
            chunk: The chunk to process.
            
        Returns:
            Processed chunk or None if filtered.
        """
        result = chunk
        for processor in self._processors:
            if result is None:
                break
            try:
                result = processor(result)
            except Exception as e:
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
            except Exception as e:
                with self._lock:
                    self.metrics.errors_count += 1
                    self.status = StreamStatus.ERROR
    
    def stream(
        self,
        source: IStreamSource, 
        sink: IStreamSink,
        transform: Optional[Callable[[Any], Any]] = None
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
            except Exception as e:
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
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize async stream context.
        
        Args:
            config: Stream configuration.
        """
        self.config = config or StreamConfig()
        self.status = StreamStatus.IDLE
        self.metrics = StreamMetrics()
        
        # Async queues
        self._input_queue: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue(
            maxsize=self.config.buffer_size
        )
        self._output_queue: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue(
            maxsize=self.config.buffer_size
        )
        
        self._processors: list[Callable[[StreamChunk], Optional[StreamChunk]]] = []
        self._stop_event = asyncio.Event()
    
    async def stream_async(
        self,
        source: AsyncIterator[StreamChunk],
        sink: Callable[[StreamChunk], bool],
        transform: Optional[Callable[[Any], Any]] = None
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
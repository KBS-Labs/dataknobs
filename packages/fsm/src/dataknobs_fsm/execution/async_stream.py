"""Asynchronous stream executor for real-time processing."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, List, Tuple, Union

from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.execution.stream import StreamProgress
from dataknobs_fsm.streaming.core import StreamConfig


@dataclass
class AsyncStreamResult:
    """Result from async stream processing."""
    total_processed: int
    successful: int
    failed: int
    duration: float
    throughput: float
    error_details: List[Any] = field(default_factory=list)


class AsyncStreamExecutor:
    """Asynchronous executor for stream processing.
    
    This executor handles:
    - True async stream processing
    - Async iterators and generators
    - Backpressure management
    - Real-time progress reporting
    - Memory-efficient chunk processing
    """
    
    def __init__(
        self,
        fsm: FSM,
        stream_config: StreamConfig | None = None,
        enable_backpressure: bool = True,
        progress_callback: Union[Callable, None] = None
    ):
        """Initialize async stream executor.
        
        Args:
            fsm: FSM to execute.
            stream_config: Stream configuration.
            enable_backpressure: Enable backpressure handling.
            progress_callback: Callback for progress updates.
        """
        self.fsm = fsm
        self.stream_config = stream_config or StreamConfig()
        self.enable_backpressure = enable_backpressure
        self.progress_callback = progress_callback
        
        # Create execution engine
        self.engine = ExecutionEngine(fsm)
        
        # Backpressure management
        self._pending_chunks = 0
        self._backpressure_threshold = self.stream_config.backpressure_threshold
        self._semaphore = asyncio.Semaphore(self.stream_config.parallelism)
    
    async def execute_stream(
        self,
        source: Union[AsyncIterator[Any], List[Any]],
        sink: Union[Callable, None] = None,
        chunk_size: int = 100,
        max_transitions: int = 1000
    ) -> AsyncStreamResult:
        """Execute stream processing asynchronously.
        
        Args:
            source: Async iterator or list of items.
            sink: Optional async sink function.
            chunk_size: Size of processing chunks.
            max_transitions: Maximum transitions per item.
            
        Returns:
            Stream processing result.
        """
        progress = StreamProgress()
        start_time = time.time()
        
        # Create base context
        # Use SINGLE mode since we process items individually
        context_template = ExecutionContext(
            data_mode=ProcessingMode.SINGLE,
            transaction_mode=TransactionMode.NONE
        )
        
        # Process stream
        try:
            # Convert source to async iterator if needed
            if hasattr(source, '__aiter__'):
                stream = source
            elif hasattr(source, '__iter__'):
                # Convert sync iterator to async
                stream = self._sync_to_async_iter(source)
            else:
                raise ValueError("Source must be an iterator or async iterator")
            
            # Process in chunks
            chunk = []
            chunk_num = 0
            
            async for item in stream:
                # Handle both individual items and pre-chunked lists
                if isinstance(item, list):
                    # Already chunked (e.g., from streaming file reader)
                    await self._process_chunk(
                        item,
                        chunk_num,
                        context_template,
                        max_transitions,
                        progress,
                        sink
                    )
                    chunk_num += 1
                else:
                    # Individual item - accumulate into chunks
                    chunk.append(item)

                    if len(chunk) >= chunk_size:
                        # Process chunk
                        await self._process_chunk(
                            chunk,
                            chunk_num,
                            context_template,
                            max_transitions,
                            progress,
                            sink
                        )
                        chunk = []
                        chunk_num += 1
                    
                    # Apply backpressure if needed
                    if self.enable_backpressure and self._pending_chunks >= self._backpressure_threshold:
                        await asyncio.sleep(0.1)
            
            # Process remaining items
            if chunk:
                await self._process_chunk(
                    chunk,
                    chunk_num,
                    context_template,
                    max_transitions,
                    progress,
                    sink
                )
        
        finally:
            # Clean up
            if hasattr(source, 'aclose'):
                await source.aclose()
        
        # Calculate final statistics
        duration = time.time() - start_time
        return AsyncStreamResult(
            total_processed=progress.records_processed,
            successful=progress.records_processed - len(progress.errors),
            failed=len(progress.errors),
            duration=duration,
            throughput=progress.records_processed / duration if duration > 0 else 0,
            error_details=progress.errors[:10]  # First 10 errors
        )
    
    async def _process_chunk(
        self,
        items: List[Any],
        chunk_num: int,
        context_template: ExecutionContext,
        max_transitions: int,
        progress: StreamProgress,
        sink: Union[Callable, None]
    ):
        """Process a chunk of items.
        
        Args:
            items: Items to process.
            chunk_num: Chunk number.
            context_template: Template context.
            max_transitions: Maximum transitions.
            progress: Progress tracker.
            sink: Optional sink function.
        """
        self._pending_chunks += 1
        
        try:
            # Create tasks for parallel processing
            tasks = []
            for i, item in enumerate(items):
                task = asyncio.create_task(
                    self._process_item(
                        item,
                        progress.records_processed + i,
                        context_template,
                        max_transitions
                    )
                )
                tasks.append(task)
            
            # Wait for all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    progress.errors.append((progress.records_processed + i, result))
                else:
                    # Result is a tuple[bool, Any] at this point
                    success, value = result  # type: ignore
                    if success:  # success
                        successful_results.append(value)
                    else:
                        progress.errors.append((progress.records_processed + i, Exception(value)))
            
            # Send to sink if provided
            if sink and successful_results:
                if asyncio.iscoroutinefunction(sink):
                    await sink(successful_results)
                else:
                    # Run sync sink in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, sink, successful_results)
            
            # Update progress
            progress.chunks_processed += 1
            progress.records_processed += len(items)
            progress.last_chunk_time = time.time()
            
            # Fire progress callback
            if self.progress_callback:
                await self._fire_progress_callback(progress)
        
        finally:
            self._pending_chunks -= 1
    
    async def _process_item(
        self,
        item: Any,
        index: int,
        context_template: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Process a single item.
        
        Args:
            item: Item to process.
            index: Item index.
            context_template: Template context.
            max_transitions: Maximum transitions.
            
        Returns:
            Tuple of (success, result).
        """
        async with self._semaphore:  # Control parallelism
            # Create context
            context = context_template.clone()
            context.data = item
            
            # Reset to initial state
            initial_state = self._find_initial_state()
            if initial_state:
                context.set_state(initial_state)
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.engine.execute,
                context,
                item,
                max_transitions
            )
    
    async def _sync_to_async_iter(self, sync_iter):
        """Convert sync iterator to async iterator.
        
        Args:
            sync_iter: Synchronous iterator.
            
        Yields:
            Items from the iterator.
        """
        for item in sync_iter:
            yield item
            await asyncio.sleep(0)  # Allow other tasks to run
    
    def _find_initial_state(self) -> str | None:
        """Find initial state in FSM.
        
        Returns:
            Initial state name or None.
        """
        # Get main network
        main_network = getattr(self.fsm, 'main_network', None)
        if isinstance(main_network, str):
            if main_network in self.fsm.networks:
                network = self.fsm.networks[main_network]
                if hasattr(network, 'initial_states') and network.initial_states:
                    return next(iter(network.initial_states))
        elif main_network and hasattr(main_network, 'initial_states'):
            if main_network.initial_states:
                return next(iter(main_network.initial_states))
        
        # Fallback: check all networks
        for network in self.fsm.networks.values():
            if hasattr(network, 'initial_states') and network.initial_states:
                return next(iter(network.initial_states))
        
        return None
    
    async def _fire_progress_callback(self, progress: StreamProgress):
        """Fire progress callback.
        
        Args:
            progress: Progress information.
        """
        if asyncio.iscoroutinefunction(self.progress_callback):
            await self.progress_callback(progress)
        else:
            # Run sync callback in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.progress_callback, progress)  # type: ignore

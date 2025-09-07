"""Stream executor for chunk-based processing."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import DataMode, TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.streaming.core import (
    AsyncStreamContext,
    IStreamSink,
    IStreamSource,
    StreamChunk,
    StreamConfig,
    StreamContext,
)


@dataclass
class StreamPipeline:
    """Pipeline configuration for stream processing."""
    source: IStreamSource
    sink: Optional[IStreamSink] = None
    transformations: List[callable] = field(default_factory=list)
    chunk_processors: List[callable] = field(default_factory=list)


@dataclass
class StreamProgress:
    """Progress tracking for stream processing."""
    chunks_processed: int = 0
    records_processed: int = 0
    bytes_processed: int = 0
    errors: List[Tuple[int, Exception]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_chunk_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time."""
        return time.time() - self.start_time
    
    @property
    def chunks_per_second(self) -> float:
        """Get chunk processing rate."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.chunks_processed / elapsed
    
    @property
    def records_per_second(self) -> float:
        """Get record processing rate."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.records_processed / elapsed


class StreamExecutor:
    """Executor for stream-based processing.
    
    This executor handles:
    - Chunk-based processing with backpressure
    - Pipeline coordination
    - Memory management
    - Progress reporting
    - Stream transformations
    """
    
    def __init__(
        self,
        fsm: FSM,
        stream_config: Optional[StreamConfig] = None,
        enable_backpressure: bool = True,
        progress_callback: Optional[callable] = None
    ):
        """Initialize stream executor.
        
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
        
        # Memory management
        self._memory_usage = 0
        self._memory_limit = self.stream_config.memory_limit_mb * 1024 * 1024
        
        # Backpressure management
        self._pending_chunks = 0
        self._backpressure_threshold = self.stream_config.backpressure_threshold
    
    def execute_stream(
        self,
        pipeline: StreamPipeline,
        context_template: Optional[ExecutionContext] = None,
        max_transitions: int = 1000
    ) -> Dict[str, Any]:
        """Execute stream processing pipeline.
        
        Args:
            pipeline: Stream pipeline configuration.
            context_template: Template context.
            max_transitions: Maximum transitions per record.
            
        Returns:
            Stream processing statistics.
        """
        # Create progress tracker
        progress = StreamProgress()
        
        # Create base context
        if context_template is None:
            context_template = ExecutionContext(
                data_mode=DataMode.STREAM,
                transaction_mode=TransactionMode.NONE
            )
        
        # Create stream context
        stream_context = StreamContext(config=self.stream_config)
        
        # Set stream context in execution context
        context_template.stream_context = stream_context
        
        # Process stream
        try:
            while True:
                # Check memory usage
                if self._should_apply_backpressure():
                    time.sleep(0.1)
                    continue
                
                # Read next chunk from source
                chunk = pipeline.source.read_chunk()
                if chunk is None:
                    break
                
                # Apply chunk processors
                for processor in pipeline.chunk_processors:
                    chunk = processor(chunk)
                    if chunk is None:
                        break
                
                if chunk is None:
                    continue
                
                # Process chunk
                chunk_results = self._process_chunk(
                    chunk,
                    context_template,
                    pipeline.transformations,
                    max_transitions,
                    progress
                )
                
                # Write results to sink if provided
                if pipeline.sink and chunk_results:
                    result_chunk = StreamChunk(
                        data=chunk_results,
                        sequence_number=chunk.sequence_number,
                        metadata=chunk.metadata,
                        is_last=chunk.is_last
                    )
                    pipeline.sink.write_chunk(result_chunk)
                
                # Update progress
                progress.chunks_processed += 1
                progress.records_processed += len(chunk.data)
                progress.last_chunk_time = time.time()
                
                # Fire progress callback
                if self.progress_callback:
                    self.progress_callback(progress)
                
                # Check if last chunk
                if chunk.is_last:
                    break
                
        finally:
            # Clean up
            pipeline.source.close()
            if pipeline.sink:
                pipeline.sink.flush()
                pipeline.sink.close()
        
        return self._generate_stats(progress)
    
    def _process_chunk(
        self,
        chunk: StreamChunk,
        context_template: ExecutionContext,
        transformations: List[callable],
        max_transitions: int,
        progress: StreamProgress
    ) -> List[Any]:
        """Process a single chunk.
        
        Args:
            chunk: Chunk to process.
            context_template: Template context.
            transformations: Transformations to apply.
            max_transitions: Maximum transitions.
            progress: Progress tracker.
            
        Returns:
            List of processed results.
        """
        results = []
        self._pending_chunks += 1
        
        try:
            for i, record in enumerate(chunk.data):
                # Apply transformations
                transformed = record
                for transform in transformations:
                    transformed = transform(transformed)
                    if transformed is None:
                        break
                
                if transformed is None:
                    continue
                
                # Create context for this record
                context = context_template.clone()
                context.data = transformed
                context.set_stream_chunk(chunk)
                
                # Reset to initial state
                initial_state = self._find_initial_state()
                if initial_state:
                    context.set_state(initial_state)
                    
                    # Execute FSM
                    try:
                        success, result = self.engine.execute(
                            context,
                            transformed,
                            max_transitions
                        )
                        
                        if success:
                            results.append(result)
                        else:
                            # FSM failed, but still pass the data through
                            results.append(transformed)
                            progress.errors.append((
                                progress.records_processed + i,
                                Exception(result)
                            ))
                    except Exception as e:
                        # On error, pass the data through
                        results.append(transformed)
                        progress.errors.append((
                            progress.records_processed + i,
                            e
                        ))
                else:
                    # No FSM configured, just pass data through
                    results.append(transformed)
            
        finally:
            self._pending_chunks -= 1
        
        return results
    
    def _should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied.
        
        Returns:
            True if backpressure needed.
        """
        if not self.enable_backpressure:
            return False
        
        # Check pending chunks
        if self._pending_chunks >= self._backpressure_threshold:
            return True
        
        # Check memory usage
        if self._memory_usage >= self._memory_limit:
            return True
        
        return False
    
    def _find_initial_state(self) -> Optional[str]:
        """Find initial state in FSM.
        
        Returns:
            Initial state name or None.
        """
        # Get main network
        if self.fsm.name in self.fsm.networks:
            network = self.fsm.networks[self.fsm.name]
            if network.initial_states:
                return next(iter(network.initial_states))
        return None
    
    def _generate_stats(self, progress: StreamProgress) -> Dict[str, Any]:
        """Generate stream processing statistics.
        
        Args:
            progress: Progress tracker.
            
        Returns:
            Processing statistics.
        """
        return {
            'chunks_processed': progress.chunks_processed,
            'records_processed': progress.records_processed,
            'bytes_processed': progress.bytes_processed,
            'errors': len(progress.errors),
            'elapsed_time': progress.elapsed_time,
            'chunks_per_second': progress.chunks_per_second,
            'records_per_second': progress.records_per_second,
            'error_details': progress.errors[:10]  # First 10 errors
        }
    
    async def execute_stream_async(
        self,
        pipeline: StreamPipeline,
        context_template: Optional[ExecutionContext] = None,
        max_transitions: int = 1000
    ) -> Dict[str, Any]:
        """Execute stream processing asynchronously.
        
        Args:
            pipeline: Stream pipeline configuration.
            context_template: Template context.
            max_transitions: Maximum transitions.
            
        Returns:
            Stream processing statistics.
        """
        # Create progress tracker
        progress = StreamProgress()
        
        # Create base context
        if context_template is None:
            context_template = ExecutionContext(
                data_mode=DataMode.STREAM,
                transaction_mode=TransactionMode.NONE
            )
        
        # Create async stream context
        async_context = AsyncStreamContext(config=self.stream_config)
        
        # Set stream context
        context_template.stream_context = async_context
        
        # Process stream asynchronously
        try:
            while True:
                # Check backpressure
                if self._should_apply_backpressure():
                    await asyncio.sleep(0.1)
                    continue
                
                # Read chunk asynchronously from source
                # For now, use sync read in async context
                chunk = pipeline.source.read_chunk()
                if chunk is None:
                    break
                
                # Process in parallel using asyncio
                chunk_results = await self._process_chunk_async(
                    chunk,
                    context_template,
                    pipeline.transformations,
                    max_transitions,
                    progress
                )
                
                # Write results
                if pipeline.sink and chunk_results:
                    result_chunk = StreamChunk(
                        data=chunk_results,
                        sequence_number=chunk.sequence_number,
                        metadata=chunk.metadata,
                        is_last=chunk.is_last
                    )
                    # Write synchronously for now
                    pipeline.sink.write_chunk(result_chunk)
                
                # Update progress
                progress.chunks_processed += 1
                progress.records_processed += len(chunk.data)
                
                # Fire callback
                if self.progress_callback:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.progress_callback,
                        progress
                    )
                
                if chunk.is_last:
                    break
                
        finally:
            # Clean up
            pipeline.source.close()
            if pipeline.sink:
                pipeline.sink.flush()
                pipeline.sink.close()
        
        return self._generate_stats(progress)
    
    async def _process_chunk_async(
        self,
        chunk: StreamChunk,
        context_template: ExecutionContext,
        transformations: List[callable],
        max_transitions: int,
        progress: StreamProgress
    ) -> List[Any]:
        """Process chunk asynchronously.
        
        Args:
            chunk: Chunk to process.
            context_template: Template context.
            transformations: Transformations.
            max_transitions: Maximum transitions.
            progress: Progress tracker.
            
        Returns:
            Processed results.
        """
        tasks = []
        
        for i, record in enumerate(chunk.data):
            # Create task for each record
            task = asyncio.create_task(
                self._process_record_async(
                    record,
                    i,
                    context_template,
                    transformations,
                    max_transitions,
                    progress
                )
            )
            tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                progress.errors.append((
                    progress.records_processed + i,
                    result
                ))
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _process_record_async(
        self,
        record: Any,
        index: int,
        context_template: ExecutionContext,
        transformations: List[callable],
        max_transitions: int,
        progress: StreamProgress
    ) -> Any:
        """Process single record asynchronously.
        
        Args:
            record: Record to process.
            index: Record index.
            context_template: Template context.
            transformations: Transformations.
            max_transitions: Maximum transitions.
            progress: Progress tracker.
            
        Returns:
            Processed result.
        """
        # Apply transformations
        transformed = record
        for transform in transformations:
            transformed = transform(transformed)
            if transformed is None:
                return None
        
        # Create context
        context = context_template.clone()
        context.data = transformed
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        success, result = await loop.run_in_executor(
            None,
            self.engine.execute,
            context,
            transformed,
            max_transitions
        )
        
        if not success:
            raise Exception(result)
        
        return result
    
    def create_multi_stage_pipeline(
        self,
        stages: List[Dict[str, Any]]
    ) -> StreamPipeline:
        """Create a multi-stage processing pipeline.
        
        Args:
            stages: List of stage configurations.
            
        Returns:
            Configured pipeline.
        """
        # Build pipeline from stages
        transformations = []
        chunk_processors = []
        
        for stage in stages:
            stage_type = stage.get('type')
            
            if stage_type == 'transform':
                transformations.append(stage['function'])
            elif stage_type == 'chunk_processor':
                chunk_processors.append(stage['function'])
        
        return StreamPipeline(
            source=stages[0].get('source'),
            sink=stages[-1].get('sink'),
            transformations=transformations,
            chunk_processors=chunk_processors
        )
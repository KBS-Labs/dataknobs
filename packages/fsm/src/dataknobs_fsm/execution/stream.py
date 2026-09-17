# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Stream executor for chunk-based processing."""

import logging
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union

from dataknobs_common import (
    BridgedOperation,
    OperationTimeoutError,
    SyncLoopBridge,
    bridged_operation,
)

from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.resources.base import AsyncClosable
from dataknobs_fsm.streaming.core import (
    IStreamSink,
    IStreamSource,
    StreamChunk,
    StreamConfig,
    StreamContext,
)

logger = logging.getLogger(__name__)

#: Loop-thread name for a stream operation's bridge. The bridge registers it,
#: so ``assert_no_leaked_bridge_threads`` watches this name too --- it is a
#: diagnostic label, not a way out of the leak guard.
BRIDGE_THREAD_NAME = "dk-fsm-stream"


@dataclass
class StreamPipeline:
    """Pipeline configuration for stream processing."""

    source: IStreamSource
    sink: IStreamSink | None = None
    transformations: List[Callable] = field(default_factory=list)
    chunk_processors: List[Callable] = field(default_factory=list)


@dataclass
class StreamProgress:
    """Progress tracking for stream processing."""

    chunks_processed: int = 0
    records_processed: int = 0
    records_emitted: int = 0
    excluded_by_state: Dict[str, int] = field(default_factory=dict)
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
        stream_config: StreamConfig | None = None,
        enable_backpressure: bool = True,
        progress_callback: Union[Callable, None] = None,
        *,
        bridge: SyncLoopBridge | None = None,
        timeout: float | None = None,
    ):
        """Initialize stream executor.

        Args:
            fsm: FSM to execute.
            stream_config: Stream configuration.
            enable_backpressure: Enable backpressure handling.
            progress_callback: Callback for progress updates.
            bridge: A loop this executor's operations run on, owned by the
                caller. Nothing here closes it. Required when the FSM's
                resources are already bound to a loop --- an
                ``AsyncDatabaseResourceAdapter`` keeps its ``AsyncDatabase``
                open across acquisitions, so it belongs to whichever loop
                first opened it, and a pooled backend is unusable from any
                other. It is also how a stream shares a loop with a
                ``BatchExecutor`` or ``SimpleFSM`` over the same FSM: pass
                ``fsm.get_sync_bridge()``. Omitted, each operation owns a
                bridge for its duration and ends it, leaving no thread behind.
            timeout: Seconds to allow one *stream operation*, however many
                chunks and records it carries, after which ``TimeoutError`` is
                raised. It is the only upper bound a blocked synchronous
                caller has, and it bounds the *work*: when this executor owns
                its loop, tearing that loop down afterwards can add up to five
                seconds more, letting a cancelled record's cleanup unwind
                rather than destroying it mid-flight. See
                :func:`~dataknobs_common.bridged_operation`. ``None`` (the
                default) waits for as long as the stream takes.
        """
        self.fsm = fsm
        self.stream_config = stream_config or StreamConfig()
        self.enable_backpressure = enable_backpressure
        self.progress_callback = progress_callback
        self._bridge = bridge
        self._timeout = timeout

        # The single async execution engine; sync stream entry points drive it
        # through an async→sync bridge scoped to the operation, so a discarded
        # executor never leaks a process-lifetime bridge thread.
        self.engine = fsm.get_async_engine()

        # Memory management
        self._memory_usage = 0
        self._memory_limit = self.stream_config.memory_limit_mb * 1024 * 1024

        # Backpressure management
        self._pending_chunks = 0
        self._backpressure_threshold = self.stream_config.backpressure_threshold

    def execute_stream(
        self,
        pipeline: StreamPipeline,
        context_template: ExecutionContext | None = None,
        max_transitions: int = 1000,
    ) -> Dict[str, Any]:
        """Execute stream processing pipeline.

        Args:
            pipeline: Stream pipeline configuration.
            context_template: Template context.
            max_transitions: Maximum transitions per record.

        Returns:
            Stream processing statistics.

        Raises:
            TimeoutError: If this executor's ``timeout`` elapses before the
                stream finishes. The source and sink are still closed on the
                way out.
        """
        # Create progress tracker
        progress = StreamProgress()

        # Create base context
        if context_template is None:
            # Use SINGLE mode since we process items individually
            context_template = ExecutionContext(
                data_mode=ProcessingMode.SINGLE, transaction_mode=TransactionMode.NONE
            )

        # Create stream context
        stream_context = StreamContext(config=self.stream_config)

        # Set stream context in execution context
        context_template.stream_context = stream_context

        # One operation for the whole stream: one loop for every chunk and
        # record, and one budget spent between them. The ``with`` closes after
        # ``_run_stream``'s ``finally``, so the loop outlives the source and
        # sink teardown that might still reach it --- which a ``bridge.close()``
        # inside that ``finally`` did not.
        with self._operation() as op:
            return self._run_stream(op, pipeline, context_template, progress, max_transitions)

    def _operation(self) -> AbstractContextManager[BridgedOperation]:
        """Open the loop and the time budget one stream operation runs within.

        Scoped to ``execute_stream`` rather than to the record: ``_process_chunk``
        takes the operation as an argument, so every record of every chunk
        reaches the engine on one loop and spends one budget between them.
        """
        return bridged_operation(
            bridge=self._bridge,
            timeout=self._timeout,
            thread_name=BRIDGE_THREAD_NAME,
            label="StreamExecutor",
        )

    def _run_stream(
        self,
        op: BridgedOperation,
        pipeline: StreamPipeline,
        context_template: ExecutionContext,
        progress: StreamProgress,
        max_transitions: int,
    ) -> Dict[str, Any]:
        """Read, process and write the pipeline until it is exhausted.

        Split from :meth:`execute_stream` so the operation's scope is a
        ``with`` there rather than a ``close()`` inside this ``finally``,
        which is what puts the loop's teardown *after* the source's and the
        sink's instead of before them.
        """
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
                    progress,
                    op,
                )

                # Write results to sink if provided
                if pipeline.sink and chunk_results:
                    result_chunk = StreamChunk(
                        data=chunk_results,
                        sequence_number=chunk.sequence_number,
                        metadata=chunk.metadata,
                        is_last=chunk.is_last,
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
            # Clean up. Probe and call must name the same method: admitting
            # either name and then calling `close` unconditionally raises
            # `AttributeError` here for a source offering only `aclose` --- from
            # a `finally:`, where it replaces whatever the body was propagating.
            # Same policy as `ResourceManager._close_provider`: run the
            # synchronous half if there is one, then report the half this
            # engine cannot run rather than claiming success.
            source = pipeline.source
            if hasattr(source, "close"):
                source.close()
            if isinstance(source, AsyncClosable):
                logger.error(
                    "Stream source %s must be closed with `await aclose()`; the "
                    "synchronous stream executor cannot await it, so it is still open.",
                    type(source).__name__,
                )

            if pipeline.sink:
                pipeline.sink.flush()
                pipeline.sink.close()

        return self._generate_stats(progress)

    def _process_chunk(
        self,
        chunk: StreamChunk,
        context_template: ExecutionContext,
        transformations: List[Callable],
        max_transitions: int,
        progress: StreamProgress,
        op: BridgedOperation,
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
                        success, result = op.run(
                            self.engine.execute(context, transformed, max_transitions)
                        )

                        if success:
                            results.append(result)
                        else:
                            # FSM failed, but still pass the data through
                            results.append(transformed)
                            progress.errors.append(
                                (progress.records_processed + i, Exception(result))
                            )
                    except OperationTimeoutError:
                        # The operation ran out of time. That is not this
                        # record failing --- it is the call the caller made
                        # ending --- so it goes past the per-record handler
                        # rather than being recorded as one more pass-through.
                        raise
                    except Exception as e:
                        # On error, pass the data through
                        results.append(transformed)
                        progress.errors.append((progress.records_processed + i, e))
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

    def _find_initial_state(self) -> str | None:
        """Ask the engine where this FSM starts.

        Delegated rather than reimplemented, because this answer decides
        whether the FSM runs at all: :meth:`_process_chunk` gates on it, so an
        executor that resolves fewer FSMs than the engine it drives silently
        declines to run the difference. It was reimplemented, and wrongly ---
        the lookup was by the *FSM's* name, which finds a network only when the
        FSM happens to be named after its main network. Every other FSM took
        the "no FSM configured" branch: each record passed through untouched
        and was counted as successfully processed, with nothing raised and
        nothing logged.

        Returns:
            Initial state name, or ``None`` if no network declares one.
        """
        return self.engine.find_initial_state_common()

    def _generate_stats(self, progress: StreamProgress) -> Dict[str, Any]:
        """Generate stream processing statistics.

        Args:
            progress: Progress tracker.

        Returns:
            Processing statistics.
        """
        return {
            "total_processed": progress.records_processed,
            "successful": progress.records_processed - len(progress.errors),
            "failed": len(progress.errors),
            "duration": progress.elapsed_time,
            "throughput": progress.records_per_second,
            # Additional details
            "chunks_processed": progress.chunks_processed,
            "bytes_processed": progress.bytes_processed,
            "error_details": progress.errors[:10],  # First 10 errors
        }

    def create_multi_stage_pipeline(self, stages: List[Dict[str, Any]]) -> StreamPipeline:
        """Create a multi-stage processing pipeline.

        Args:
            stages: List of stage configurations. The first must carry a
                ``source``.

        Returns:
            Configured pipeline.

        Raises:
            ValueError: If ``stages`` is empty or its first stage has no
                ``source``. A pipeline without one is unusable, and the
                failure is far more legible here than as an ``AttributeError``
                on ``None`` once ``execute_stream`` starts iterating it.
        """
        if not stages:
            raise ValueError("A multi-stage pipeline needs at least one stage")

        source = stages[0].get("source")
        if source is None:
            raise ValueError("The first stage of a multi-stage pipeline must have a 'source'")

        # Build pipeline from stages
        transformations = []
        chunk_processors = []

        for stage in stages:
            stage_type = stage.get("type")

            if stage_type == "transform":
                transformations.append(stage["function"])
            elif stage_type == "chunk_processor":
                chunk_processors.append(stage["function"])

        return StreamPipeline(
            source=source,
            sink=stages[-1].get("sink"),
            transformations=transformations,
            chunk_processors=chunk_processors,
        )

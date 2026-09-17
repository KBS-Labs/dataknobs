# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Batch executor for parallel record processing."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

from dataknobs_common import (
    BridgedOperation,
    OperationTimeoutError,
    SyncLoopBridge,
    bridged_operation,
)

from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext, ResourceStatus

#: Loop-thread name for a batch operation's bridge. The bridge registers it, so
#: ``assert_no_leaked_bridge_threads`` watches this name too --- it is a
#: diagnostic label, not a way out of the leak guard.
BRIDGE_THREAD_NAME = "dk-fsm-batch"


@dataclass
class BatchResult:
    """Result from batch processing."""

    index: int
    success: bool
    result: Any
    error: Exception | None = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""

    total: int
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time."""
        return time.time() - self.start_time

    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.completed / elapsed

    @property
    def estimated_time_remaining(self) -> float:
        """Get estimated time remaining."""
        rate = self.items_per_second
        if rate == 0:
            return float("inf")
        remaining = self.total - self.completed
        return remaining / rate


class BatchExecutor:
    """Executor for batch processing with parallelism.

    This executor handles:
    - Parallel record processing
    - Progress tracking and reporting
    - Error aggregation and handling

    It does **not** pool resources, despite ``enable_resource_pooling``: that
    flag gates per-item bookkeeping and a released-status transition, and
    nothing here hands out a resource or enforces a limit. The flag and its
    name predate this class's current behaviour; whether to implement pooling
    or drop the parameter is an open question, and the same one
    ``BatchConfig.parallel`` poses.
    """

    def __init__(
        self,
        fsm: FSM,
        parallelism: int = 4,
        batch_size: int = 100,
        enable_resource_pooling: bool = True,
        progress_callback: Union[Callable, None] = None,
        *,
        bridge: SyncLoopBridge | None = None,
        timeout: float | None = None,
    ):
        """Initialize batch executor.

        Args:
            fsm: FSM to execute.
            parallelism: Number of parallel workers.
            batch_size: Size of each batch.
            enable_resource_pooling: Gate the per-item resource bookkeeping ---
                the batch metadata written by :meth:`_acquire_resources` and
                the released-status transition in :meth:`_release_resources`.
                It does **not** pool: nothing here hands out a resource or
                enforces the ``limit`` in ``context.resource_limits``. The name
                overpromises and is kept for compatibility.
            progress_callback: Callback for progress updates.
            bridge: A loop this executor's operations run on, owned by the
                caller. Nothing here closes it. Required when the FSM's
                resources are already bound to a loop --- an
                ``AsyncDatabaseResourceAdapter`` keeps its ``AsyncDatabase``
                open across acquisitions, so it belongs to whichever loop
                first opened it, and a pooled backend is unusable from any
                other. It is also how an executor shares a loop with
                ``SimpleFSM`` or another executor over the same FSM: pass
                ``fsm.get_sync_bridge()``. Omitted, each operation owns a
                bridge for its duration and ends it, leaving no thread behind.
            timeout: Seconds to allow one *operation* --- one ``execute_batch``
                or one ``execute_batches``, however many items it runs ---
                after which ``TimeoutError`` is raised. It is the only upper
                bound a blocked synchronous caller has, and it bounds the
                *work*: when this executor owns its loop, tearing that loop
                down afterwards can add up to five seconds more, letting a
                cancelled item's cleanup unwind rather than destroying it
                mid-flight. See :func:`~dataknobs_common.bridged_operation`.
                ``None`` (the default) waits for as long as the batch takes.
        """
        self.fsm = fsm
        self.parallelism = parallelism
        self.batch_size = batch_size
        self.enable_resource_pooling = enable_resource_pooling
        self.progress_callback = progress_callback
        self._bridge = bridge
        self._timeout = timeout

        # The single async execution engine; sync batch entry points drive it
        # through an async→sync bridge scoped to the operation, so a discarded
        # executor never leaks a process-lifetime bridge thread.
        self.engine = fsm.get_async_engine()

    def _operation(self) -> AbstractContextManager[BridgedOperation]:
        """Open the loop and the time budget one public call runs within.

        Scoped to the entry point rather than to the item: the workers take
        the operation as an argument, so every item of a batch --- and every
        batch of an ``execute_batches`` --- reaches the engine on one loop and
        spends one budget between them. A per-item budget would be no bound at
        all on the call the caller made.
        """
        return bridged_operation(
            bridge=self._bridge,
            timeout=self._timeout,
            thread_name=BRIDGE_THREAD_NAME,
            label="BatchExecutor",
        )

    def execute_batch(
        self,
        items: List[Any],
        context_template: ExecutionContext | None = None,
        max_transitions: int = 1000,
    ) -> List[BatchResult]:
        """Execute batch of items.

        Args:
            items: Items to process.
            context_template: Template context to clone.
            max_transitions: Maximum transitions per item.

        Returns:
            List of batch results.

        Raises:
            TimeoutError: If this executor's ``timeout`` elapses before the
                batch finishes.
        """
        if not items:
            return []

        with self._operation() as op:
            return self._execute_batch(op, items, context_template, max_transitions)

    def _execute_batch(
        self,
        op: BridgedOperation,
        items: List[Any],
        context_template: ExecutionContext | None,
        max_transitions: int,
    ) -> List[BatchResult]:
        """Run one batch on an already-open operation.

        Split from :meth:`execute_batch` so :meth:`execute_batches` can drive
        several batches within one operation instead of opening a loop --- and
        restarting the budget --- for each.
        """
        # Create progress tracker
        progress = BatchProgress(total=len(items))

        # Create base context if not provided
        if context_template is None:
            context_template = ExecutionContext(
                data_mode=ProcessingMode.SINGLE, transaction_mode=TransactionMode.PER_RECORD
            )

        # Process based on parallelism setting
        if self.parallelism <= 1:
            return self._execute_sequential(items, context_template, max_transitions, progress, op)
        else:
            return self._execute_parallel(items, context_template, max_transitions, progress, op)

    def _execute_sequential(
        self,
        items: List[Any],
        context_template: ExecutionContext,
        max_transitions: int,
        progress: BatchProgress,
        op: BridgedOperation,
    ) -> List[BatchResult]:
        """Execute items sequentially.

        Args:
            items: Items to process.
            context_template: Template context.
            max_transitions: Maximum transitions.
            progress: Progress tracker.

        Returns:
            List of results.
        """
        results = []

        for i, item in enumerate(items):
            start_time = time.time()

            # Create context for this item
            context = context_template.clone()
            # Convert Record to dict if needed
            if hasattr(item, "to_dict"):
                context.data = item.to_dict()
            elif hasattr(item, "__dict__"):
                context.data = dict(item.__dict__)
            else:
                context.data = item

            # Add batch tracking metadata. ``batch_info`` is the channel:
            # an attribute set on the shared ``ExecutionContext`` is not part
            # of its type, so nothing could rely on it being there.
            context.metadata["batch_info"] = {
                "batch_id": i,
                "total_items": len(items),
                "item_index": i,
                "processing_mode": "sequential",
            }

            # Reset to initial state
            initial_state = self._find_initial_state()
            if initial_state:
                context.set_state(initial_state)

            # Execute
            try:
                success, result = op.run(
                    self.engine.execute(
                        context,
                        None,  # Data is already in context
                        max_transitions,
                    )
                )

                # Store final state and path in metadata
                metadata = context.metadata.copy() if context.metadata else {}
                metadata["final_state"] = context.current_state
                metadata["path"] = context.history if hasattr(context, "history") else []

                batch_result = BatchResult(
                    index=i,
                    success=success,
                    result=result,
                    processing_time=time.time() - start_time,
                    metadata=metadata,
                )

                if success:
                    progress.succeeded += 1
                else:
                    progress.failed += 1

            except OperationTimeoutError:
                # The operation ran out of time. That is not one item failing
                # --- it is the call the caller made ending --- so it goes past
                # the per-item handler rather than being recorded as a result.
                raise
            except Exception as e:
                batch_result = BatchResult(
                    index=i,
                    success=False,
                    result=None,
                    error=e,
                    processing_time=time.time() - start_time,
                )
                progress.failed += 1

            results.append(batch_result)
            progress.completed += 1

            # Fire progress callback
            if self.progress_callback:
                self.progress_callback(progress)

        return results

    def _execute_parallel(
        self,
        items: List[Any],
        context_template: ExecutionContext,
        max_transitions: int,
        progress: BatchProgress,
        op: BridgedOperation,
    ) -> List[BatchResult]:
        """Execute items in parallel.

        Args:
            items: Items to process.
            context_template: Template context.
            max_transitions: Maximum transitions.
            progress: Progress tracker.

        Returns:
            List of results.
        """
        results = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            # Submit all items
            futures = {}
            for i, item in enumerate(items):
                future = executor.submit(
                    self._process_single_item, i, item, context_template, max_transitions, op
                )
                futures[future] = i

            # Process completed items
            for future in as_completed(futures):
                index = futures[future]

                try:
                    batch_result = future.result()
                    results[index] = batch_result  # type: ignore

                    if batch_result.success:
                        progress.succeeded += 1
                    else:
                        progress.failed += 1

                except OperationTimeoutError:
                    # As in the sequential path: the operation's deadline ends
                    # the call rather than marking one item failed. Leaving the
                    # ``with`` waits for the pool, but the wait is bounded ---
                    # a queued worker finds the budget spent and refuses
                    # without reaching the loop, so only the items already
                    # in flight are still running.
                    raise
                except Exception as e:
                    results[index] = BatchResult(  # type: ignore
                        index=index, success=False, result=None, error=e
                    )
                    progress.failed += 1

                progress.completed += 1

                # Fire progress callback
                if self.progress_callback:
                    self.progress_callback(progress)

        return results  # type: ignore

    def _process_single_item(
        self,
        index: int,
        item: Any,
        context_template: ExecutionContext,
        max_transitions: int,
        op: BridgedOperation,
    ) -> BatchResult:
        """Process a single item.

        Args:
            index: Item index.
            item: Item to process.
            context_template: Template context.
            max_transitions: Maximum transitions.

        Returns:
            Batch result.
        """
        start_time = time.time()

        # Create context for this item
        context = context_template.clone()
        # Convert Record to dict if needed
        if hasattr(item, "to_dict"):
            context.data = item.to_dict()
        elif hasattr(item, "__dict__"):
            context.data = dict(item.__dict__)
        else:
            context.data = item

        # Add batch tracking metadata (see ``_execute_sequential``).
        context.metadata["batch_info"] = {
            "batch_id": index,
            "item_index": index,
            "processing_mode": "parallel",
            "worker_thread": threading.current_thread().name,
        }

        # Get resource from pool if available
        if self.enable_resource_pooling:
            self._acquire_resources(context)

        try:
            # Reset to initial state
            initial_state = self._find_initial_state()
            if initial_state:
                context.set_state(initial_state)

            # Execute
            success, result = op.run(
                self.engine.execute(
                    context,
                    None,  # Data is already in context
                    max_transitions,
                )
            )

            # Store final state and path in metadata
            metadata = context.metadata.copy() if context.metadata else {}
            metadata["final_state"] = context.current_state
            metadata["path"] = context.history if hasattr(context, "history") else []

            return BatchResult(
                index=index,
                success=success,
                result=result,
                processing_time=time.time() - start_time,
                metadata=metadata,
            )

        except OperationTimeoutError:
            # Surfaced through the future to ``_execute_parallel``, which lets
            # it past its own per-item handler for the same reason.
            raise

        except Exception as e:
            return BatchResult(
                index=index,
                success=False,
                result=None,
                error=e,
                processing_time=time.time() - start_time,
            )

        finally:
            # Release resources back to pool
            if self.enable_resource_pooling:
                self._release_resources(context)

    def _acquire_resources(self, context: ExecutionContext) -> None:
        """Record what this context was allowed, for the batch's metadata.

        ``enable_resource_pooling`` names something this class does not do:
        nothing here hands out a resource, and nothing enforces ``limit``. What
        the flag actually gates is this bookkeeping and the status transition
        in :meth:`_release_resources`. It is written down rather than implied
        because the name promises otherwise --- see the class docstring.

        Args:
            context: Execution context.
        """
        batch_id = context.metadata.get("batch_info", {}).get("batch_id")
        if batch_id is None:
            return

        for resource_type, limit in context.resource_limits.items():
            context.metadata[f"batch_{batch_id}_resources"] = {
                "resource_type": resource_type,
                "limit": limit,
                "acquired_at": context.metadata.get("start_time"),
            }

    def _release_resources(self, context: ExecutionContext) -> None:
        """Mark this context's allocations released.

        The status test compares ``ResourceStatus`` members. It used to compare
        one against the string ``"allocated"`` --- the member's *value* --- so
        it was never true and this whole body was unreachable: every allocation
        stayed marked as held for the life of the context. mypy reported both
        halves of that (``comparison-overlap`` on the test, then ``unreachable``
        on the body). ``AVAILABLE`` is the released state, which is what
        :meth:`ExecutionContext.release_resource` sets; there is no
        ``RELEASED`` member and the string the old code assigned was not one.

        Restoring the transition also exposed what the body around it did. It
        appended each released id to a per-type list that nothing ever read
        back or drained --- ``_acquire_resources`` took its length for a
        ``pool_size`` field and nothing else --- so a working release would
        have grown that list once per allocation per item for the life of the
        executor. A pool nothing draws from is an accumulator, and the id it
        accumulated was already the caller's to reuse. It is gone, along with
        the ``asyncio.Lock`` per type that was created beside it and never
        acquired.

        Args:
            context: Execution context.
        """
        batch_id = context.metadata.get("batch_info", {}).get("batch_id")
        batch_key = None if batch_id is None else f"batch_{batch_id}_resources"

        for allocation in context.resources.values():
            if allocation.status is not ResourceStatus.ALLOCATED:
                continue

            if batch_key is not None and batch_key in context.metadata:
                context.metadata[batch_key]["released_at"] = context.metadata.get("end_time")

            allocation.status = ResourceStatus.AVAILABLE

    def _find_initial_state(self) -> str | None:
        """Ask the engine where this FSM starts.

        Delegated rather than reimplemented, because this executor *drives*
        that engine: an executor that disagreed with it about the start state
        would gate on one answer and run on another. It was reimplemented, and
        wrongly --- the lookup was by the *FSM's* name, which finds a network
        only when the FSM happens to be named after its main network. That was
        harmless here, since this path calls the engine either way and the
        engine resolves the start state for itself, and fatal in the stream
        executor, which used the same lookup to decide whether to run at all.

        Returns:
            Initial state name, or ``None`` if no network declares one.
        """
        return self.engine.find_initial_state_common()

    def execute_batches(
        self,
        items: List[Any],
        context_template: ExecutionContext | None = None,
        max_transitions: int = 1000,
    ) -> Dict[str, Any]:
        """Execute items in batches.

        Args:
            items: All items to process.
            context_template: Template context.
            max_transitions: Maximum transitions.

        Returns:
            Aggregated results.

        Raises:
            TimeoutError: If this executor's ``timeout`` elapses before every
                batch has finished. The budget spans the whole call, not each
                batch: restarting it per batch would multiply it by the batch
                count and bound nothing the caller asked about.
        """
        # One operation for the whole call. Opening one per batch put each
        # batch on a loop of its own, so an FSM resource opened during the
        # first was unusable from the second --- a pooled backend fails there
        # with an error naming the connection rather than the loop.
        with self._operation() as op:
            return self._execute_batches(op, items, context_template, max_transitions)

    def _execute_batches(
        self,
        op: BridgedOperation,
        items: List[Any],
        context_template: ExecutionContext | None,
        max_transitions: int,
    ) -> Dict[str, Any]:
        """Run every batch on an already-open operation.

        Split from :meth:`execute_batches` so :meth:`create_benchmark` can
        drive several configurations within one operation instead of opening a
        loop --- and restarting the budget --- for each.
        """
        all_results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch = items[start_idx:end_idx]

            # Process batch
            batch_results = self._execute_batch(op, batch, context_template, max_transitions)

            all_results.extend(batch_results)

        # Aggregate results
        total = len(all_results)
        succeeded = sum(1 for r in all_results if r.success)
        failed = total - succeeded

        total_time = sum(r.processing_time for r in all_results)
        avg_time = total_time / total if total > 0 else 0

        errors_by_type: Dict[str, int] = {}
        for result in all_results:
            if result.error:
                error_type = type(result.error).__name__
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate": succeeded / total if total > 0 else 0,
            "total_processing_time": total_time,
            "average_processing_time": avg_time,
            "errors_by_type": errors_by_type,
            "results": all_results,
        }

    def create_benchmark(
        self, items: List[Any], configurations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run performance benchmark with different configurations.

        Args:
            items: Items to process.
            configurations: List of configuration dicts with:
                - 'name': Configuration name
                - 'parallelism': Parallelism level
                - 'batch_size': Batch size
                - 'strategy': Traversal strategy

        Returns:
            Benchmark results.

        Raises:
            TimeoutError: If this executor's ``timeout`` elapses before every
                configuration has run. The budget spans the whole benchmark,
                not each configuration: restarting it per configuration would
                multiply it by their number and bound nothing the caller asked
                about.
        """
        benchmark_results = {}

        # A measurement must not reconfigure what it measured. ``strategy`` is
        # the half that reaches furthest: it is set on the FSM's *one* async
        # engine, shared with every other executor, ``SimpleFSM`` and
        # ``FSM.execute`` over that FSM, so leaving it set re-strategised all
        # of them.
        saved_parallelism = self.parallelism
        saved_batch_size = self.batch_size
        saved_strategy = self.engine.strategy

        try:
            # One operation for the whole benchmark, for the same reason
            # ``execute_batches`` holds one across its batches.
            with self._operation() as op:
                for config in configurations:
                    name = config.get("name", "unnamed")

                    # Update executor settings
                    self.parallelism = config.get("parallelism", self.parallelism)
                    self.batch_size = config.get("batch_size", self.batch_size)

                    if "strategy" in config:
                        self.engine.strategy = config["strategy"]

                    # Run benchmark
                    start_time = time.time()
                    results = self._execute_batches(op, items, None, 1000)
                    elapsed_time = time.time() - start_time

                    # Calculate metrics
                    throughput = len(items) / elapsed_time if elapsed_time > 0 else 0

                    benchmark_results[name] = {
                        "configuration": config,
                        "elapsed_time": elapsed_time,
                        "throughput": throughput,
                        "success_rate": results["success_rate"],
                        "average_processing_time": results["average_processing_time"],
                    }
        finally:
            self.parallelism = saved_parallelism
            self.batch_size = saved_batch_size
            self.engine.strategy = saved_strategy

        return benchmark_results

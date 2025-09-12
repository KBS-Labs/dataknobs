"""Batch executor for parallel record processing."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine


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
            return float('inf')
        remaining = self.total - self.completed
        return remaining / rate


class BatchExecutor:
    """Executor for batch processing with parallelism.
    
    This executor handles:
    - Parallel record processing
    - Resource pooling and management
    - Progress tracking and reporting
    - Error aggregation and handling
    - Performance optimization
    """
    
    def __init__(
        self,
        fsm: FSM,
        parallelism: int = 4,
        batch_size: int = 100,
        enable_resource_pooling: bool = True,
        progress_callback: Union[Callable, None] = None
    ):
        """Initialize batch executor.
        
        Args:
            fsm: FSM to execute.
            parallelism: Number of parallel workers.
            batch_size: Size of each batch.
            enable_resource_pooling: Enable resource pooling.
            progress_callback: Callback for progress updates.
        """
        self.fsm = fsm
        self.parallelism = parallelism
        self.batch_size = batch_size
        self.enable_resource_pooling = enable_resource_pooling
        self.progress_callback = progress_callback
        
        # Create execution engine
        self.engine = ExecutionEngine(fsm)
        
        # Resource pool
        self._resource_pool: Dict[str, List[Any]] = {}
        self._resource_locks: Dict[str, asyncio.Lock] = {}
    
    def execute_batch(
        self,
        items: List[Any],
        context_template: ExecutionContext | None = None,
        max_transitions: int = 1000
    ) -> List[BatchResult]:
        """Execute batch of items.
        
        Args:
            items: Items to process.
            context_template: Template context to clone.
            max_transitions: Maximum transitions per item.
            
        Returns:
            List of batch results.
        """
        if not items:
            return []
        
        # Create progress tracker
        progress = BatchProgress(total=len(items))
        
        # Create base context if not provided
        if context_template is None:
            context_template = ExecutionContext(
                data_mode=ProcessingMode.SINGLE,
                transaction_mode=TransactionMode.PER_RECORD
            )
        
        # Process based on parallelism setting
        if self.parallelism <= 1:
            return self._execute_sequential(
                items,
                context_template,
                max_transitions,
                progress
            )
        else:
            return self._execute_parallel(
                items,
                context_template,
                max_transitions,
                progress
            )
    
    def _execute_sequential(
        self,
        items: List[Any],
        context_template: ExecutionContext,
        max_transitions: int,
        progress: BatchProgress
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
            if hasattr(item, 'to_dict'):
                context.data = item.to_dict()
            elif hasattr(item, '__dict__'):
                context.data = dict(item.__dict__)
            else:
                context.data = item
            
            # Add batch tracking metadata
            context.batch_id = i
            context.metadata['batch_info'] = {
                'batch_id': i,
                'total_items': len(items),
                'item_index': i,
                'processing_mode': 'sequential'
            }
            
            # Reset to initial state
            initial_state = self._find_initial_state()
            if initial_state:
                context.set_state(initial_state)
            
            # Execute
            try:
                success, result = self.engine.execute(
                    context,
                    None,  # Data is already in context
                    max_transitions
                )
                
                # Store final state and path in metadata
                metadata = context.metadata.copy() if context.metadata else {}
                metadata['final_state'] = context.current_state
                metadata['path'] = context.history if hasattr(context, 'history') else []
                
                batch_result = BatchResult(
                    index=i,
                    success=success,
                    result=result,
                    processing_time=time.time() - start_time,
                    metadata=metadata
                )
                
                if success:
                    progress.succeeded += 1
                else:
                    progress.failed += 1
                
            except Exception as e:
                batch_result = BatchResult(
                    index=i,
                    success=False,
                    result=None,
                    error=e,
                    processing_time=time.time() - start_time
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
        progress: BatchProgress
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
                    self._process_single_item,
                    i,
                    item,
                    context_template,
                    max_transitions
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
                    
                except Exception as e:
                    results[index] = BatchResult(  # type: ignore
                        index=index,
                        success=False,
                        result=None,
                        error=e
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
        max_transitions: int
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
        if hasattr(item, 'to_dict'):
            context.data = item.to_dict()
        elif hasattr(item, '__dict__'):
            context.data = dict(item.__dict__)
        else:
            context.data = item
        
        # Add batch tracking metadata
        context.batch_id = index
        context.metadata['batch_info'] = {
            'batch_id': index,
            'item_index': index,
            'processing_mode': 'parallel',
            'worker_thread': threading.current_thread().name
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
            success, result = self.engine.execute(
                context,
                None,  # Data is already in context
                max_transitions
            )
            
            # Store final state and path in metadata
            metadata = context.metadata.copy() if context.metadata else {}
            metadata['final_state'] = context.current_state
            metadata['path'] = context.history if hasattr(context, 'history') else []
            
            return BatchResult(
                index=index,
                success=success,
                result=result,
                processing_time=time.time() - start_time,
                metadata=metadata
            )
            
        except Exception as e:
            return BatchResult(
                index=index,
                success=False,
                result=None,
                error=e,
                processing_time=time.time() - start_time
            )
        
        finally:
            # Release resources back to pool
            if self.enable_resource_pooling:
                self._release_resources(context)
    
    def _acquire_resources(self, context: ExecutionContext) -> None:
        """Acquire resources from pool for context.
        
        Args:
            context: Execution context.
        """
        # Initialize resource pools if needed
        for resource_type, limit in context.resource_limits.items():
            if resource_type not in self._resource_pool:
                self._resource_pool[resource_type] = []
                self._resource_locks[resource_type] = asyncio.Lock()
            
            # Track batch-specific resource allocation
            if hasattr(context, 'batch_id'):
                context.metadata[f'batch_{context.batch_id}_resources'] = {
                    'resource_type': resource_type,
                    'limit': limit,
                    'acquired_at': context.metadata.get('start_time'),
                    'pool_size': len(self._resource_pool[resource_type])
                }
    
    def _release_resources(self, context: ExecutionContext) -> None:
        """Release resources back to pool.
        
        Args:
            context: Execution context.
        """
        # Release allocated resources back to pool
        for allocation in context.resources.values():
            if allocation.status == 'allocated':
                resource_type = allocation.resource_type
                if resource_type in self._resource_pool:
                    self._resource_pool[resource_type].append(
                        allocation.resource_id
                    )
                    
                    # Track batch-specific resource release
                    if hasattr(context, 'batch_id'):
                        batch_key = f'batch_{context.batch_id}_resources'
                        if batch_key in context.metadata:
                            context.metadata[batch_key]['released_at'] = context.metadata.get('end_time')
                            context.metadata[batch_key]['final_pool_size'] = len(self._resource_pool[resource_type])
                
                # Mark as released
                allocation.status = 'released'
    
    def _find_initial_state(self) -> str | None:
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
    
    def execute_batches(
        self,
        items: List[Any],
        context_template: ExecutionContext | None = None,
        max_transitions: int = 1000
    ) -> Dict[str, Any]:
        """Execute items in batches.
        
        Args:
            items: All items to process.
            context_template: Template context.
            max_transitions: Maximum transitions.
            
        Returns:
            Aggregated results.
        """
        all_results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch = items[start_idx:end_idx]
            
            # Process batch
            batch_results = self.execute_batch(
                batch,
                context_template,
                max_transitions
            )
            
            all_results.extend(batch_results)
        
        # Aggregate results
        total = len(all_results)
        succeeded = sum(1 for r in all_results if r.success)
        failed = total - succeeded
        
        total_time = sum(r.processing_time for r in all_results)
        avg_time = total_time / total if total > 0 else 0
        
        errors_by_type = {}
        for result in all_results:
            if result.error:
                error_type = type(result.error).__name__
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        return {
            'total': total,
            'succeeded': succeeded,
            'failed': failed,
            'success_rate': succeeded / total if total > 0 else 0,
            'total_processing_time': total_time,
            'average_processing_time': avg_time,
            'errors_by_type': errors_by_type,
            'results': all_results
        }
    
    def create_benchmark(
        self,
        items: List[Any],
        configurations: List[Dict[str, Any]]
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
        """
        benchmark_results = {}
        
        for config in configurations:
            name = config.get('name', 'unnamed')
            
            # Update executor settings
            self.parallelism = config.get('parallelism', self.parallelism)
            self.batch_size = config.get('batch_size', self.batch_size)
            
            if 'strategy' in config:
                self.engine.strategy = config['strategy']
            
            # Run benchmark
            start_time = time.time()
            results = self.execute_batches(items)
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            throughput = len(items) / elapsed_time if elapsed_time > 0 else 0
            
            benchmark_results[name] = {
                'configuration': config,
                'elapsed_time': elapsed_time,
                'throughput': throughput,
                'success_rate': results['success_rate'],
                'average_processing_time': results['average_processing_time']
            }
        
        return benchmark_results

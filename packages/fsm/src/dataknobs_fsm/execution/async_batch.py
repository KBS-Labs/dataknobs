"""Asynchronous batch executor for parallel processing."""

import asyncio
import time
from typing import Any, Callable, Dict, List, Union

from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.batch import BatchResult, BatchProgress
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine


class AsyncBatchExecutor:
    """Asynchronous executor for batch processing.
    
    This executor handles:
    - True async parallel execution
    - Resource pooling
    - Progress reporting
    - Error recovery
    - Transaction management
    """
    
    def __init__(
        self,
        fsm: FSM,
        parallelism: int = 10,
        batch_size: int = 100,
        enable_transactions: bool = False,
        progress_callback: Union[Callable, None] = None
    ):
        """Initialize async batch executor.
        
        Args:
            fsm: FSM to execute.
            parallelism: Maximum parallel executions.
            batch_size: Size of each batch.
            enable_transactions: Enable transaction support.
            progress_callback: Callback for progress updates.
        """
        self.fsm = fsm
        self.parallelism = parallelism
        self.batch_size = batch_size
        self.enable_transactions = enable_transactions
        self.progress_callback = progress_callback
        
        # Create execution engine
        self.engine = ExecutionEngine(fsm)
        
        # Semaphore for parallelism control
        self._semaphore = asyncio.Semaphore(parallelism)
    
    async def execute_batch(
        self,
        items: List[Any],
        context_template: ExecutionContext | None = None,
        max_transitions: int = 1000
    ) -> List[BatchResult]:
        """Execute batch of items asynchronously.
        
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
                transaction_mode=TransactionMode.PER_RECORD if self.enable_transactions else TransactionMode.NONE
            )
        
        # Process items in parallel
        tasks = []
        for i, item in enumerate(items):
            task = asyncio.create_task(
                self._process_item(i, item, context_template, max_transitions, progress)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Fire final progress callback
        if self.progress_callback:
            await self._fire_progress_callback(progress)
        
        return results
    
    async def _process_item(
        self,
        index: int,
        item: Any,
        context_template: ExecutionContext,
        max_transitions: int,
        progress: BatchProgress
    ) -> BatchResult:
        """Process a single item asynchronously.
        
        Args:
            index: Item index.
            item: Item to process.
            context_template: Template context.
            max_transitions: Maximum transitions.
            progress: Progress tracker.
            
        Returns:
            Batch result.
        """
        async with self._semaphore:  # Control parallelism
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
            
            # Reset to initial state
            initial_state = self._find_initial_state()
            if initial_state:
                context.set_state(initial_state)
            else:
                # If no initial state found, try to get from first network
                if self.fsm.networks:
                    first_network = next(iter(self.fsm.networks.values()))
                    if hasattr(first_network, 'states') and first_network.states:
                        # Find a state marked as is_start
                        for state in first_network.states.values():
                            if getattr(state, 'is_start', False):
                                context.set_state(state.name)
                                break
            
            try:
                # Execute in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                success, result = await loop.run_in_executor(
                    None,
                    self.engine.execute,
                    context,
                    None,  # Data is already in context
                    max_transitions
                )
                
                # Store final state and path in metadata
                metadata = context.metadata.copy() if context.metadata else {}
                metadata['final_state'] = context.current_state
                metadata['path'] = context.history if hasattr(context, 'history') else []
                
                batch_result = BatchResult(
                    index=index,
                    success=success,
                    result=result,
                    processing_time=time.time() - start_time,
                    metadata=metadata
                )
                
                # Update progress
                progress.completed += 1
                if success:
                    progress.succeeded += 1
                else:
                    progress.failed += 1
                
                # Fire progress callback
                if self.progress_callback and progress.completed % 10 == 0:
                    await self._fire_progress_callback(progress)
                
                return batch_result
                
            except Exception as e:
                progress.completed += 1
                progress.failed += 1
                
                return BatchResult(
                    index=index,
                    success=False,
                    result=None,
                    error=e,
                    processing_time=time.time() - start_time
                )
    
    async def execute_batches(
        self,
        items: List[Any],
        context_template: ExecutionContext | None = None,
        max_transitions: int = 1000
    ) -> Dict[str, Any]:
        """Execute items in multiple batches.
        
        Args:
            items: All items to process.
            context_template: Template context.
            max_transitions: Maximum transitions.
            
        Returns:
            Execution statistics.
        """
        all_results = []
        total_start = time.time()
        
        # Process in chunks
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self.execute_batch(
                batch,
                context_template,
                max_transitions
            )
            all_results.extend(batch_results)
        
        # Calculate statistics
        total_time = time.time() - total_start
        successful = sum(1 for r in all_results if r.success)
        failed = sum(1 for r in all_results if not r.success)
        
        return {
            'total': len(all_results),
            'successful': successful,
            'failed': failed,
            'duration': total_time,
            'throughput': len(all_results) / total_time if total_time > 0 else 0,
            'results': all_results
        }
    
    def _find_initial_state(self) -> str | None:
        """Find initial state in FSM.
        
        Returns:
            Initial state name or None.
        """
        # Get main network
        main_network_name = getattr(self.fsm, 'main_network', None)
        if main_network_name and main_network_name in self.fsm.networks:
            network = self.fsm.networks[main_network_name]
            # Check for initial_states (set) or get_initial_states() method
            if hasattr(network, 'initial_states') and network.initial_states:
                return next(iter(network.initial_states))
            elif hasattr(network, 'get_initial_states'):
                initial_states = network.get_initial_states()
                if initial_states:
                    return next(iter(initial_states))
        
        # Fallback: check all networks
        for network in self.fsm.networks.values():
            if hasattr(network, 'initial_states') and network.initial_states:
                return next(iter(network.initial_states))
        
        return None
    
    async def _fire_progress_callback(self, progress: BatchProgress):
        """Fire progress callback asynchronously.
        
        Args:
            progress: Progress information.
        """
        if asyncio.iscoroutinefunction(self.progress_callback):
            await self.progress_callback(progress)
        else:
            # Run sync callback in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.progress_callback, progress)  # type: ignore

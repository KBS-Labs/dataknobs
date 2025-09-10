"""Simple API for common FSM operations.

This module provides a simplified interface for common FSM use cases,
abstracting away the complexity of configuration, resource management,
and execution strategies.
"""

from typing import Any, Callable, Dict, List, Union, AsyncIterator
from pathlib import Path
import asyncio
from dataknobs_data import Record

from ..core.data_modes import DataHandlingMode
from ..core.context_factory import ContextFactory
from ..core.result_formatter import ResultFormatter
from ..execution.engine import ExecutionEngine
from ..execution.async_engine import AsyncExecutionEngine
from ..config.loader import ConfigLoader
from ..config.builder import FSMBuilder
from ..resources.manager import ResourceManager


class SimpleFSM:
    """Simplified FSM interface for common operations.
    
    This class provides an easy-to-use API for processing data through
    an FSM without requiring detailed knowledge of the underlying
    components.
    """
    
    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        resources: Dict[str, Any] | None = None
    ):
        """Initialize SimpleFSM from configuration.
        
        Args:
            config: Path to config file or config dictionary
            data_mode: Default data mode for processing
            resources: Optional resource configurations
        """
        self.data_mode = data_mode
        self._resources = resources or {}
        
        # Load configuration
        if isinstance(config, (str, Path)):
            loader = ConfigLoader()
            self._config = loader.load_from_file(Path(config))
        else:
            loader = ConfigLoader()
            self._config = loader.load_from_dict(config)
            
        # Build FSM
        builder = FSMBuilder()
        self._fsm = builder.build(self._config)
        
        # Initialize resource manager
        self._resource_manager = ResourceManager()
        self._setup_resources()
        
        # Create execution engines
        self._engine = ExecutionEngine(self._fsm)
        self._async_engine = AsyncExecutionEngine(self._fsm)
        
    def _setup_resources(self) -> None:
        """Set up resources from configuration."""
        # Register resources from config
        if hasattr(self._config, 'resources'):
            for resource_config in self._config.resources:
                try:
                    resource = self._create_resource_provider(resource_config)
                    self._resource_manager.register_provider(resource_config.name, resource)
                except Exception:
                    # Continue if resource creation fails - this is for simplified API
                    pass
                
        # Register additional resources passed to constructor
        for name, resource_config in self._resources.items():
            try:
                # Use ResourceManager factory method
                self._resource_manager.register_from_dict(name, resource_config)
            except Exception:
                # Continue if resource creation fails
                pass
    
    def process(
        self,
        data: Union[Dict[str, Any], Record],
        initial_state: str | None = None,
        timeout: float | None = None
    ) -> Dict[str, Any]:
        """Process a single data record through the FSM.
        
        Args:
            data: Input data to process
            initial_state: Optional starting state (defaults to FSM start state)
            timeout: Optional timeout in seconds
            
        Returns:
            Dict containing:
                - final_state: Name of the final state reached
                - data: Processed data from final state
                - path: List of states traversed
                - success: Whether processing completed successfully
                - error: Error message if processing failed
        """
        # Create context using factory
        from ..core.modes import ProcessingMode
        context = ContextFactory.create_context(
            fsm=self._fsm,
            data=data,
            initial_state=initial_state,
            data_mode=ProcessingMode.SINGLE,
            resource_manager=self._resource_manager
        )
        
        # Execute using the sync engine
        try:
            if timeout:
                # Implement timeout using concurrent.futures
                import concurrent.futures
                import threading
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._engine.execute, context)
                    try:
                        success, result = future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        # Cancel the future and raise timeout error
                        future.cancel()
                        raise TimeoutError(f"FSM execution exceeded timeout of {timeout} seconds")
            else:
                success, result = self._engine.execute(context)
                
            # Format result using ResultFormatter
            return ResultFormatter.format_single_result(
                context=context,
                success=success,
                result=result
            )
        except Exception as e:
            return ResultFormatter.format_error_result(
                context=context,
                error=e
            )
    
    async def process_async(
        self,
        data: Union[Dict[str, Any], Record],
        initial_state: str | None = None,
        timeout: float | None = None
    ) -> Dict[str, Any]:
        """Process a single data record through the FSM asynchronously.
        
        Args:
            data: Input data to process
            initial_state: Optional starting state (defaults to FSM start state)
            timeout: Optional timeout in seconds
            
        Returns:
            Dict containing processing results
        """
        # Create context using factory
        from ..core.modes import ProcessingMode
        context = ContextFactory.create_context(
            fsm=self._fsm,
            data=data,
            initial_state=initial_state,
            data_mode=ProcessingMode.SINGLE,
            resource_manager=self._resource_manager
        )
        
        try:
            if timeout:
                success, result = await asyncio.wait_for(
                    self._async_engine.execute(context),
                    timeout=timeout
                )
            else:
                success, result = await self._async_engine.execute(context)
                
            return ResultFormatter.format_async_result(
                context=context,
                success=success,
                result=result
            )
        except Exception as e:
            return ResultFormatter.format_error_result(
                context=context,
                error=e
            )
    
    def process_batch(
        self,
        data: List[Union[Dict[str, Any], Record]],
        batch_size: int = 10,
        max_workers: int = 4,
        on_progress: Union[Callable, None] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple records in parallel batches.
        
        Args:
            data: List of input records to process
            batch_size: Number of records per batch
            max_workers: Maximum parallel workers
            on_progress: Optional callback for progress updates
            
        Returns:
            List of results for each input record
        """
        from ..execution.async_batch import AsyncBatchExecutor
        
        batch_executor = AsyncBatchExecutor(
            fsm=self._fsm,
            parallelism=max_workers,
            batch_size=batch_size,
            progress_callback=on_progress
        )
        
        # Convert to Records
        records = []
        for item in data:
            if isinstance(item, dict):
                records.append(Record(item))
            else:
                records.append(item)
                
        # Execute batch using async executor
        results = asyncio.run(batch_executor.execute_batch(
            items=records
        ))
        
        # Format results
        formatted_results = []
        for result in results:
            if result.success:
                formatted_results.append({
                    'final_state': result.metadata.get('final_state', 'unknown'),
                    'data': result.result,
                    'path': result.metadata.get('path', []),
                    'success': True,
                    'error': None
                })
            else:
                formatted_results.append({
                    'final_state': result.metadata.get('final_state', None),
                    'data': result.result if result.result else {},
                    'path': result.metadata.get('path', []),
                    'success': False,
                    'error': str(result.error) if result.error else str(result.result)
                })
                
        return formatted_results
    
    async def process_stream(
        self,
        source: Union[str, AsyncIterator[Dict[str, Any]]],
        sink: str | None = None,
        chunk_size: int = 100,
        on_progress: Union[Callable, None] = None
    ) -> Dict[str, Any]:
        """Process a stream of data through the FSM.
        
        Args:
            source: Data source (file path or async iterator)
            sink: Optional output destination
            chunk_size: Size of processing chunks
            on_progress: Optional progress callback
            
        Returns:
            Dict containing stream processing statistics
        """
        from ..execution.async_stream import AsyncStreamExecutor
        from ..streaming.core import StreamConfig as CoreStreamConfig
        
        # Configure streaming
        stream_config = CoreStreamConfig(
            chunk_size=chunk_size,
            parallelism=4,
            memory_limit_mb=1024
        )
        
        # Create async stream executor
        stream_executor = AsyncStreamExecutor(
            fsm=self._fsm,
            stream_config=stream_config,
            progress_callback=on_progress
        )
        
        # Handle file source
        if isinstance(source, str):
            # Read file and convert to async iterator
            async def file_reader():
                import json
                with open(source) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            stream_source = file_reader()
        else:
            # Already an async iterator
            stream_source = source
        
        # Handle sink
        sink_func = None
        if sink:
            # Create a simple file writer
            def write_to_file(results):
                import json
                with open(sink, 'a') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            sink_func = write_to_file
        
        # Execute stream using async executor
        result = await stream_executor.execute_stream(
            source=stream_source,
            sink=sink_func,
            chunk_size=chunk_size
        )
        
        return {
            'total_processed': result.total_processed,
            'successful': result.successful,
            'failed': result.failed,
            'duration': result.duration,
            'throughput': result.throughput
        }
    
    def validate(self, data: Union[Dict[str, Any], Record]) -> Dict[str, Any]:
        """Validate data against FSM's start state schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Dict containing validation results
        """
        # Convert to Record if needed
        if isinstance(data, dict):
            record = Record(data)
        else:
            record = data
            
        # Get start state
        start_state = self._fsm.get_start_state()
        
        # Validate against schema
        if start_state.schema:
            validation_result = start_state.schema.validate(record)
            return {
                'valid': validation_result.valid,
                'errors': validation_result.errors if not validation_result.valid else []
            }
        else:
            return {
                'valid': True,
                'errors': []
            }
    
    def get_states(self) -> List[str]:
        """Get list of all state names in the FSM."""
        states = []
        # The FSM has networks, and each network has states
        for network in self._fsm.networks.values():
            for state in network.states.values():
                states.append(state.name)
        return states
    
    def get_resources(self) -> List[str]:
        """Get list of registered resource names."""
        return list(self._resource_manager._resources.keys())
    
    def close(self) -> None:
        """Clean up resources and close connections."""
        asyncio.run(self._resource_manager.cleanup())
    
    def _create_resource_provider(self, resource_config):
        """Create a resource provider from ResourceConfig."""
        # Use the same logic as FSMBuilder
        from ..config.builder import FSMBuilder
        builder = FSMBuilder()
        return builder._create_resource(resource_config)
    


def create_fsm(
    config: Union[str, Path, Dict[str, Any]],
    **kwargs
) -> SimpleFSM:
    """Factory function to create a SimpleFSM instance.
    
    Args:
        config: Configuration file path or dictionary
        **kwargs: Additional arguments passed to SimpleFSM
        
    Returns:
        Configured SimpleFSM instance
    """
    return SimpleFSM(config, **kwargs)


# Convenience functions for common operations

def process_file(
    fsm_config: Union[str, Path, Dict[str, Any]],
    input_file: str,
    output_file: str | None = None,
    format: str = 'json',
    chunk_size: int = 1000,
    timeout: float | None = None
) -> Dict[str, Any]:
    """Process a file through an FSM.
    
    Args:
        fsm_config: FSM configuration
        input_file: Path to input file
        output_file: Optional output file path
        format: File format (json, csv, etc.)
        chunk_size: Processing chunk size
        timeout: Optional timeout in seconds for processing
        
    Returns:
        Processing statistics
    """
    fsm = create_fsm(fsm_config)
    
    try:
        if timeout:
            result = asyncio.run(asyncio.wait_for(
                fsm.process_stream(
                    source=input_file,
                    sink=output_file,
                    chunk_size=chunk_size
                ),
                timeout=timeout
            ))
        else:
            result = asyncio.run(fsm.process_stream(
                source=input_file,
                sink=output_file,
                chunk_size=chunk_size
            ))
        return result
    finally:
        fsm.close()


def validate_data(
    fsm_config: Union[str, Path, Dict[str, Any]],
    data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Validate multiple data records against FSM schema.
    
    Args:
        fsm_config: FSM configuration
        data: List of data records to validate
        
    Returns:
        List of validation results
    """
    fsm = create_fsm(fsm_config)
    
    try:
        results = []
        for record in data:
            results.append(fsm.validate(record))
        return results
    finally:
        fsm.close()


def batch_process(
    fsm_config: Union[str, Path, Dict[str, Any]],
    data: List[Dict[str, Any]],
    batch_size: int = 10,
    max_workers: int = 4,
    timeout: float | None = None
) -> List[Dict[str, Any]]:
    """Process multiple records in parallel.
    
    Args:
        fsm_config: FSM configuration
        data: List of input records
        batch_size: Batch size for processing
        max_workers: Maximum parallel workers
        timeout: Optional timeout in seconds for entire batch processing
        
    Returns:
        List of processing results
    """
    fsm = create_fsm(fsm_config)
    
    try:
        if timeout:
            # Use threading timeout for batch processing
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    fsm.process_batch,
                    data=data,
                    batch_size=batch_size,
                    max_workers=max_workers
                )
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Batch processing exceeded timeout of {timeout} seconds")
        else:
            return fsm.process_batch(
                data=data,
                batch_size=batch_size,
                max_workers=max_workers
            )
    finally:
        fsm.close()

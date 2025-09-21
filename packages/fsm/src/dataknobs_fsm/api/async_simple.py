"""Async-first API for FSM operations.

This module provides an async-first interface for FSM operations,
designed to work natively in async contexts without any asyncio.run() calls.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from dataknobs_data import Record

from ..config.builder import FSMBuilder
from ..config.loader import ConfigLoader
from ..core.context_factory import ContextFactory
from ..core.data_modes import DataHandlingMode
from ..core.result_formatter import ResultFormatter
from ..execution.async_batch import AsyncBatchExecutor
from ..execution.async_engine import AsyncExecutionEngine
from ..execution.async_stream import AsyncStreamExecutor
from ..resources.manager import ResourceManager
from ..streaming.core import StreamConfig as CoreStreamConfig


class AsyncSimpleFSM:
    """Async-first FSM interface for processing data.

    This class provides a fully asynchronous API for FSM operations,
    designed to work natively in async contexts.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        resources: dict[str, Any] | None = None,
        custom_functions: dict[str, Callable] | None = None
    ):
        """Initialize AsyncSimpleFSM from configuration.

        Args:
            config: Path to config file or config dictionary
            data_mode: Default data mode for processing
            resources: Optional resource configurations
            custom_functions: Optional custom functions to register
        """
        self.data_mode = data_mode
        self._resources = resources or {}
        self._custom_functions = custom_functions or {}

        # Create loader with knowledge of custom functions
        loader = ConfigLoader()

        # Tell the loader about registered function names
        if self._custom_functions:
            for name in self._custom_functions.keys():
                loader.add_registered_function(name)

        # Load configuration
        if isinstance(config, (str, Path)):
            self._config = loader.load_from_file(Path(config))
        else:
            self._config = loader.load_from_dict(config)

        # Build FSM with custom functions
        builder = FSMBuilder()

        # Register custom functions with the builder
        for name, func in self._custom_functions.items():
            builder.register_function(name, func)

        self._fsm = builder.build(self._config)

        # Initialize resource manager
        self._resource_manager = ResourceManager()
        self._setup_resources()

        # Create async execution engine
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

    def _create_resource_provider(self, resource_config):
        """Create a resource provider from ResourceConfig."""
        # Use the same logic as FSMBuilder
        from ..config.builder import FSMBuilder
        builder = FSMBuilder()
        return builder._create_resource(resource_config)

    async def process(self, data: dict[str, Any] | Record) -> dict[str, Any]:
        """Process a single record through the FSM asynchronously.

        Args:
            data: Input data to process

        Returns:
            Dict containing the processed result
        """
        # Convert to Record if needed
        if isinstance(data, dict):
            record = Record(data)
        else:
            record = data

        # Create context
        from ..core.modes import ProcessingMode
        context = ContextFactory.create_context(
            fsm=self._fsm,
            data=record,
            data_mode=ProcessingMode.SINGLE,
            resource_manager=self._resource_manager
        )

        try:
            # Execute FSM asynchronously
            success, result = await self._async_engine.execute(context)

            # Format result
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

    async def process_batch(
        self,
        data: list[dict[str, Any] | Record],
        batch_size: int = 10,
        max_workers: int = 4,
        on_progress: Callable | None = None
    ) -> list[dict[str, Any]]:
        """Process multiple records in parallel batches asynchronously.

        Args:
            data: List of input records to process
            batch_size: Number of records per batch
            max_workers: Maximum parallel workers
            on_progress: Optional callback for progress updates

        Returns:
            List of results for each input record
        """
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

        # Execute batch
        results = await batch_executor.execute_batch(items=records)

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
        source: str | AsyncIterator[dict[str, Any]],
        sink: str | None = None,
        chunk_size: int = 100,
        on_progress: Callable | None = None,
        input_format: str = 'auto',
        text_field_name: str = 'text',
        csv_delimiter: str = ',',
        csv_has_header: bool = True,
        skip_empty_lines: bool = True,
        use_streaming: bool = False
    ) -> dict[str, Any]:
        """Process a stream of data through the FSM asynchronously.

        Args:
            source: Data source (file path or async iterator)
            sink: Optional output destination
            chunk_size: Size of processing chunks
            on_progress: Optional progress callback
            input_format: Input file format ('auto', 'jsonl', 'json', 'csv', 'text')
            text_field_name: Field name for text lines when converting to dict
            csv_delimiter: CSV delimiter character
            csv_has_header: Whether CSV file has header row
            skip_empty_lines: Skip empty lines in text files
            use_streaming: Use memory-efficient streaming for large files

        Returns:
            Dict containing stream processing statistics
        """
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

        # Choose between streaming and regular mode
        if use_streaming and isinstance(source, str):
            # Use memory-efficient streaming for large files
            from ..utils.streaming_file_utils import (
                create_streaming_file_reader,
                create_streaming_file_writer,
            )

            stream_source = create_streaming_file_reader(
                file_path=source,
                config=stream_config,
                input_format=input_format,
                text_field_name=text_field_name,
                csv_delimiter=csv_delimiter,
                csv_has_header=csv_has_header,
                skip_empty_lines=skip_empty_lines
            )

            # Handle sink for streaming mode
            sink_func = None
            cleanup_func = None
            if sink:
                sink_func, cleanup_func = await create_streaming_file_writer(
                    file_path=sink,
                    config=stream_config
                )
        else:
            # Use regular mode (loads full chunks into memory)
            from ..utils.file_utils import create_file_reader, create_file_writer

            # Handle file source
            if isinstance(source, str):
                stream_source = create_file_reader(
                    file_path=source,
                    input_format=input_format,
                    text_field_name=text_field_name,
                    csv_delimiter=csv_delimiter,
                    csv_has_header=csv_has_header,
                    skip_empty_lines=skip_empty_lines
                )
            else:
                # Already an async iterator
                stream_source = source

            # Handle sink for regular mode
            sink_func = None
            cleanup_func = None
            if sink:
                sink_func, cleanup_func = create_file_writer(sink)

        try:
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
        finally:
            # Clean up any resources (e.g., close files)
            if cleanup_func:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()

    async def validate(self, data: dict[str, Any] | Record) -> dict[str, Any]:
        """Validate data against FSM's start state schema asynchronously.

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

    def get_states(self) -> list[str]:
        """Get list of all state names in the FSM."""
        states = []
        # The FSM has networks, and each network has states
        for network in self._fsm.networks.values():
            for state in network.states.values():
                states.append(state.name)
        return states

    def get_resources(self) -> list[str]:
        """Get list of registered resource names."""
        return list(self._resource_manager._resources.keys())

    @property
    def config(self) -> Any:
        """Get the FSM configuration object."""
        return self._config

    async def close(self) -> None:
        """Clean up resources and close connections asynchronously."""
        await self._resource_manager.cleanup()

    # Alias for consistency with other async libraries
    aclose = close


# Factory function for AsyncSimpleFSM
async def create_async_fsm(
    config: str | Path | dict[str, Any],
    custom_functions: dict[str, Callable] | None = None,
    **kwargs
) -> AsyncSimpleFSM:
    """Factory function to create an AsyncSimpleFSM instance.

    Args:
        config: Configuration file path or dictionary
        custom_functions: Optional custom functions to register
        **kwargs: Additional arguments passed to AsyncSimpleFSM

    Returns:
        Configured AsyncSimpleFSM instance
    """
    return AsyncSimpleFSM(config, custom_functions=custom_functions, **kwargs)

"""Simple API for common FSM operations.

This module provides a simplified synchronous interface for common FSM use cases,
wrapping the async-first AsyncSimpleFSM implementation.
"""

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dataknobs_data import Record

from ..core.data_modes import DataHandlingMode
from .async_simple import AsyncSimpleFSM


class SimpleFSM:
    """Synchronous FSM interface wrapping AsyncSimpleFSM.

    This class provides a purely synchronous API for FSM operations,
    internally using AsyncSimpleFSM with a dedicated event loop.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        resources: dict[str, Any] | None = None,
        custom_functions: dict[str, Callable] | None = None
    ):
        """Initialize SimpleFSM from configuration.

        Args:
            config: Path to config file or config dictionary
            data_mode: Default data mode for processing
            resources: Optional resource configurations
            custom_functions: Optional custom functions to register
        """
        # Store data_mode for compatibility
        self.data_mode = data_mode

        # Create the async FSM
        self._async_fsm = AsyncSimpleFSM(
            config=config,
            data_mode=data_mode,
            resources=resources,
            custom_functions=custom_functions
        )

        # Expose internal attributes for compatibility
        self._fsm = self._async_fsm._fsm
        self._resource_manager = self._async_fsm._resource_manager
        self._async_engine = self._async_fsm._async_engine

        # Create synchronous engine for compatibility
        from ..execution.engine import ExecutionEngine
        self._engine = ExecutionEngine(self._fsm)

        # Create a dedicated event loop for sync operations
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._setup_event_loop()

    def _setup_event_loop(self) -> None:
        """Set up a dedicated event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()

        def run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

    def _run_async(self, coro: Any) -> Any:
        """Run an async operation in the dedicated event loop.

        Args:
            coro: Coroutine to run

        Returns:
            The result of the coroutine
        """
        if not self._loop or not self._loop.is_running():
            self._setup_event_loop()

        if self._loop is None:
            raise RuntimeError("Failed to setup event loop")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def process(
        self,
        data: dict[str, Any] | Record,
        initial_state: str | None = None,
        timeout: float | None = None
    ) -> dict[str, Any]:
        """Process a single record through the FSM synchronously.

        Args:
            data: Input data to process
            initial_state: Optional starting state (defaults to FSM start state)
            timeout: Optional timeout in seconds

        Returns:
            Dict containing the processed result with fields:
            - final_state: Name of the final state reached
            - data: The transformed data
            - path: List of states traversed
            - success: Whether processing succeeded
            - error: Any error message (None if successful)
        """
        # Create the coroutine with the async process method
        async def _process():
            # Import here to avoid circular dependency
            from ..core.context_factory import ContextFactory
            from ..core.modes import ProcessingMode
            from ..core.result_formatter import ResultFormatter

            # Convert to Record if needed
            if isinstance(data, dict):
                from dataknobs_data import Record
                record = Record(data)
            else:
                record = data

            # Create context
            context = ContextFactory.create_context(
                fsm=self._fsm,
                data=record,
                initial_state=initial_state,
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
            except asyncio.TimeoutError:
                # Return error result instead of raising
                return ResultFormatter.format_error_result(
                    context=context,
                    error=TimeoutError(f"FSM execution exceeded timeout of {timeout} seconds")
                )
            except Exception as e:
                return ResultFormatter.format_error_result(
                    context=context,
                    error=e
                )

        if timeout:
            # Use threading for timeout support
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_async, _process())
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    # Return an error result instead of raising
                    return {
                        'success': False,
                        'error': f"FSM execution exceeded timeout of {timeout} seconds",
                        'final_state': None,
                        'data': data if isinstance(data, dict) else data.data,
                        'path': []
                    }
        else:
            return self._run_async(_process())

    def process_batch(
        self,
        data: list[dict[str, Any] | Record],
        batch_size: int = 10,
        max_workers: int = 4,
        on_progress: Callable | None = None
    ) -> list[dict[str, Any]]:
        """Process multiple records in parallel batches synchronously.

        Args:
            data: List of input records to process
            batch_size: Number of records per batch
            max_workers: Maximum parallel workers
            on_progress: Optional callback for progress updates

        Returns:
            List of results for each input record
        """
        return self._run_async(
            self._async_fsm.process_batch(
                data=data,
                batch_size=batch_size,
                max_workers=max_workers,
                on_progress=on_progress
            )
        )

    def process_stream(
        self,
        source: str | Any,
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
        """Process a stream of data through the FSM synchronously.

        Args:
            source: Data source file path or async iterator
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
        # If source is a string (file path), use the async version directly
        if isinstance(source, str):
            return self._run_async(
                self._async_fsm.process_stream(
                    source=source,
                    sink=sink,
                    chunk_size=chunk_size,
                    on_progress=on_progress,
                    input_format=input_format,
                    text_field_name=text_field_name,
                    csv_delimiter=csv_delimiter,
                    csv_has_header=csv_has_header,
                    skip_empty_lines=skip_empty_lines,
                    use_streaming=use_streaming
                )
            )
        else:
            # Source is an async iterator, need to handle it properly
            async def _process():
                return await self._async_fsm.process_stream(
                    source=source,
                    sink=sink,
                    chunk_size=chunk_size,
                    on_progress=on_progress,
                    input_format=input_format,
                    text_field_name=text_field_name,
                    csv_delimiter=csv_delimiter,
                    csv_has_header=csv_has_header,
                    skip_empty_lines=skip_empty_lines,
                    use_streaming=use_streaming
                )
            return self._run_async(_process())

    def validate(self, data: dict[str, Any] | Record) -> dict[str, Any]:
        """Validate data against FSM's start state schema synchronously.

        Args:
            data: Data to validate

        Returns:
            Dict containing validation results
        """
        return self._run_async(self._async_fsm.validate(data))

    def get_states(self) -> list[str]:
        """Get list of all state names in the FSM."""
        return self._async_fsm.get_states()

    def get_resources(self) -> list[str]:
        """Get list of registered resource names."""
        return self._async_fsm.get_resources()

    @property
    def config(self) -> Any:
        """Get the FSM configuration object."""
        return self._async_fsm._config

    def close(self) -> None:
        """Clean up resources and close connections synchronously."""
        self._run_async(self._async_fsm.close())

        # Shut down the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=1.0)

    async def aclose(self) -> None:
        """Async version of close for use in async contexts."""
        await self._async_fsm.close()


def create_fsm(
    config: str | Path | dict[str, Any],
    custom_functions: dict[str, Callable] | None = None,
    **kwargs
) -> SimpleFSM:
    """Factory function to create a SimpleFSM instance.

    Args:
        config: Configuration file path or dictionary
        custom_functions: Optional custom functions to register
        **kwargs: Additional arguments passed to SimpleFSM

    Returns:
        Configured SimpleFSM instance
    """
    return SimpleFSM(config, custom_functions=custom_functions, **kwargs)


# Convenience functions for common operations

def process_file(
    fsm_config: str | Path | dict[str, Any],
    input_file: str,
    output_file: str | None = None,
    input_format: str = 'auto',
    chunk_size: int = 1000,
    timeout: float | None = None,
    text_field_name: str = 'text',
    csv_delimiter: str = ',',
    csv_has_header: bool = True,
    skip_empty_lines: bool = True,
    use_streaming: bool = False
) -> dict[str, Any]:
    """Process a file through an FSM with automatic format detection.

    Args:
        fsm_config: FSM configuration
        input_file: Path to input file
        output_file: Optional output file path (format auto-detected from extension)
        input_format: Input format ('auto', 'jsonl', 'json', 'csv', 'text')
        chunk_size: Processing chunk size
        timeout: Optional timeout in seconds for processing
        text_field_name: Field name for text lines when converting to dict
        csv_delimiter: CSV delimiter character
        csv_has_header: Whether CSV file has header row
        skip_empty_lines: Skip empty lines in text files
        use_streaming: Use memory-efficient streaming for large files

    Returns:
        Processing statistics

    Examples:
        # Process plain text file
        results = process_file('config.yaml', 'input.txt', 'output.jsonl')

        # Process large CSV file with streaming
        results = process_file('config.yaml', 'large_data.csv', 'results.json', use_streaming=True)

        # Process with custom text field name
        results = process_file('config.yaml', 'input.txt', text_field_name='content')
    """
    fsm = create_fsm(fsm_config)

    try:
        if timeout:
            # Use threading timeout
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    fsm.process_stream,
                    source=input_file,
                    sink=output_file,
                    chunk_size=chunk_size,
                    input_format=input_format,
                    text_field_name=text_field_name,
                    csv_delimiter=csv_delimiter,
                    csv_has_header=csv_has_header,
                    skip_empty_lines=skip_empty_lines,
                    use_streaming=use_streaming
                )
                try:
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError as e:
                    future.cancel()
                    raise TimeoutError(f"File processing exceeded timeout of {timeout} seconds") from e
        else:
            result = fsm.process_stream(
                source=input_file,
                sink=output_file,
                chunk_size=chunk_size,
                input_format=input_format,
                text_field_name=text_field_name,
                csv_delimiter=csv_delimiter,
                csv_has_header=csv_has_header,
                skip_empty_lines=skip_empty_lines,
                use_streaming=use_streaming
            )
        return result
    finally:
        fsm.close()


def validate_data(
    fsm_config: str | Path | dict[str, Any],
    data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
    fsm_config: str | Path | dict[str, Any],
    data: list[dict[str, Any] | Record],
    batch_size: int = 10,
    max_workers: int = 4,
    timeout: float | None = None
) -> list[dict[str, Any]]:
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
                except concurrent.futures.TimeoutError as e:
                    future.cancel()
                    raise TimeoutError(f"Batch processing exceeded timeout of {timeout} seconds") from e
        else:
            return fsm.process_batch(
                data=data,
                batch_size=batch_size,
                max_workers=max_workers
            )
    finally:
        fsm.close()

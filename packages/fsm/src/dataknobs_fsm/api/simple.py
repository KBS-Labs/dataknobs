"""Simple synchronous API for FSM operations.

This module provides a simplified synchronous interface for common FSM use cases,
making it easy to build data transformation pipelines, validation workflows, and
processing systems without dealing with async/await complexity.

Architecture:
    The dataknobs-fsm package provides three API tiers:

    1. **SimpleFSM** (This Module):
       - Synchronous interface (no async/await)
       - Automatic event loop management
       - Best for: Scripts, prototypes, simple pipelines
       - Trade-off: Thread overhead for event loop

    2. **AsyncSimpleFSM** (async_simple.py):
       - Async/await interface
       - Production-ready for async contexts
       - Best for: Web services, concurrent processing, async applications
       - Trade-off: Requires async programming knowledge

    3. **AdvancedFSM** (advanced.py):
       - Full manual control with debugging support
       - Step-by-step execution, breakpoints, profiling
       - Best for: Complex workflows, debugging, custom execution strategies
       - Trade-off: More complex API

    **Choosing the Right API:**

    Use SimpleFSM if:
    - You're building scripts or prototypes
    - You want the simplest possible API
    - You're working in synchronous code
    - You don't need async/await capabilities

    Use AsyncSimpleFSM if:
    - You're building production services
    - Your application is already async
    - You need high concurrency
    - You want best performance

    Use AdvancedFSM if:
    - You need debugging capabilities
    - You want step-by-step execution
    - You need profiling and tracing
    - You're building complex workflows with custom logic

Data Handling Modes:
    FSMs can process data in three modes:

    **COPY Mode** (default):
    - Deep copies data for each state
    - Safe for concurrent processing
    - Higher memory usage
    - Best for: Production systems, parallel processing

    **REFERENCE Mode**:
    - Lazy loading with optimistic locking
    - Memory-efficient
    - Moderate performance
    - Best for: Large datasets, memory-constrained environments

    **DIRECT Mode**:
    - In-place data modification
    - Fastest performance
    - Not thread-safe
    - Best for: Single-threaded pipelines, performance-critical paths

Common Workflow Patterns:
    This module enables several common patterns:

    **Data Transformation Pipeline:**
    ```python
    from dataknobs_fsm.api.simple import SimpleFSM

    # Create FSM for data cleaning
    fsm = SimpleFSM('data_pipeline.yaml')

    # Process single record
    result = fsm.process({
        'text': 'Some input text',
        'metadata': {'source': 'user_input'}
    })
    print(result['data'])  # Transformed output
    ```

    **Batch Processing:**
    ```python
    # Process multiple records in parallel
    records = [
        {'text': 'Record 1', 'id': 1},
        {'text': 'Record 2', 'id': 2},
        {'text': 'Record 3', 'id': 3}
    ]
    results = fsm.process_batch(records, batch_size=10, max_workers=4)
    ```

    **File Processing:**
    ```python
    from dataknobs_fsm.api.simple import process_file

    # Process large file with streaming
    stats = process_file(
        fsm_config='validate.yaml',
        input_file='input.jsonl',
        output_file='output.jsonl',
        chunk_size=1000,
        use_streaming=True
    )
    print(f"Processed {stats['total_processed']} records in {stats['duration']:.2f}s")
    print(f"Throughput: {stats['throughput']:.2f} records/sec")
    ```

    **Data Validation:**
    ```python
    from dataknobs_fsm.api.simple import validate_data

    # Validate records against schema
    records = [
        {'name': 'Alice', 'age': 30},
        {'name': 'Bob', 'age': 'invalid'},  # Will fail validation
    ]
    validation_results = validate_data('schema.yaml', records)
    for i, result in enumerate(validation_results):
        if not result['valid']:
            print(f"Record {i} failed: {result['errors']}")
    ```

Example:
    Complete ETL pipeline using SimpleFSM:

    ```python
    from dataknobs_fsm.api.simple import SimpleFSM
    from dataknobs_fsm.core.data_modes import DataHandlingMode

    # Configuration defines states and transitions
    config = {
        'name': 'data_etl',
        'states': [
            {
                'name': 'extract',
                'type': 'START',
                'function': 'extract_data'
            },
            {
                'name': 'transform',
                'type': 'NORMAL',
                'function': 'clean_and_normalize'
            },
            {
                'name': 'load',
                'type': 'END',
                'function': 'save_to_database'
            }
        ],
        'arcs': [
            {'from': 'extract', 'to': 'transform'},
            {'from': 'transform', 'to': 'load'}
        ]
    }

    # Create FSM with custom functions
    def extract_data(data):
        # Extract logic
        return {'records': load_from_source(data['source'])}

    def clean_and_normalize(data):
        # Transform logic
        records = [normalize(r) for r in data['records']]
        return {'records': records}

    def save_to_database(data):
        # Load logic
        db.bulk_insert(data['records'])
        return {'status': 'success', 'count': len(data['records'])}

    # Initialize FSM
    fsm = SimpleFSM(
        config=config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={
            'extract_data': extract_data,
            'clean_and_normalize': clean_and_normalize,
            'save_to_database': save_to_database
        }
    )

    # Process data
    result = fsm.process({'source': 'input.csv'})
    print(f"ETL completed: {result['data']['status']}")
    print(f"Records processed: {result['data']['count']}")
    print(f"States traversed: {' -> '.join(result['path'])}")

    # Clean up
    fsm.close()
    ```

See Also:
    - :class:`AsyncSimpleFSM`: Async version for production applications
    - :class:`AdvancedFSM`: Advanced API with debugging and profiling
    - :class:`DataHandlingMode`: Data processing mode options
    - :mod:`dataknobs_fsm.patterns.etl`: ETL workflow patterns
    - :mod:`dataknobs_fsm.patterns.file_processing`: File processing patterns
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
    """Synchronous FSM interface for simple workflows.

    This class provides a purely synchronous API for FSM operations,
    internally using AsyncSimpleFSM with a dedicated event loop managed
    automatically in a background thread.

    SimpleFSM is designed for ease of use in scripts, prototypes, and simple
    pipelines where async/await complexity is not desired. It handles all
    async execution transparently, providing a familiar synchronous interface.

    Attributes:
        data_mode (DataHandlingMode): Data processing mode (COPY/REFERENCE/DIRECT)
        _async_fsm (AsyncSimpleFSM): Internal async FSM implementation
        _fsm (FSM): Core FSM engine
        _resource_manager (ResourceManager): Resource lifecycle manager
        _loop (AbstractEventLoop): Dedicated event loop for async operations
        _loop_thread (Thread): Background thread running the event loop

    Methods:
        process: Process a single record through the FSM
        process_batch: Process multiple records in parallel batches
        process_stream: Process a stream of data from file or iterator
        validate: Validate data against FSM's start state schema
        get_states: List all state names in the FSM
        get_resources: List all registered resource names
        close: Clean up resources and close connections

    Use Cases:
        **Data Transformation:**
        Transform data through a pipeline of state functions. Each state receives
        the output of the previous state, enabling sequential transformations.

        **Data Validation:**
        Validate data against schemas defined in state configurations. States can
        enforce data quality rules and reject invalid records.

        **File Processing:**
        Process large files line-by-line or in chunks using `process_stream()`.
        Supports automatic format detection (JSON, JSONL, CSV, text).

        **Batch Processing:**
        Process multiple records in parallel using `process_batch()`. Configurable
        batch size and worker count for optimal throughput.

        **ETL Workflows:**
        Extract-Transform-Load pipelines where data flows through extraction,
        transformation, and loading states with error handling.

    Note:
        **Thread Safety:**
        SimpleFSM manages its own event loop in a background thread. While the
        synchronous API is thread-safe, concurrent calls will serialize due to
        the single event loop. For true concurrent processing, use AsyncSimpleFSM
        with multiple event loops or process_batch() with max_workers > 1.

        **Resource Management:**
        Always call close() when done to properly release resources. Use context
        managers (with statement) when available in client code, or ensure close()
        is called in a finally block.

        **Data Mode Selection:**
        - Use COPY (default) for production: safe, predictable, memory-intensive
        - Use REFERENCE for large datasets: memory-efficient, moderate overhead
        - Use DIRECT for performance: fastest, but not thread-safe

        **Error Handling:**
        The process() method returns a dict with 'success' and 'error' keys rather
        than raising exceptions. This allows for graceful error handling in batch
        processing scenarios.

    Example:
        Basic usage with configuration file:

        ```python
        from dataknobs_fsm.api.simple import SimpleFSM

        # Create FSM from YAML config
        fsm = SimpleFSM('pipeline.yaml')

        # Process single record
        result = fsm.process({
            'text': 'Input text to process',
            'metadata': {'source': 'user'}
        })

        if result['success']:
            print(f"Result: {result['data']}")
            print(f"Path: {' -> '.join(result['path'])}")
        else:
            print(f"Error: {result['error']}")

        # Clean up
        fsm.close()
        ```

        With custom functions and resources:

        ```python
        from dataknobs_fsm.api.simple import SimpleFSM
        from dataknobs_fsm.core.data_modes import DataHandlingMode

        # Define custom state functions
        def validate(data):
            if 'required_field' not in data:
                raise ValueError("Missing required field")
            return data

        def transform(data):
            from datetime import datetime
            data['processed'] = True
            data['timestamp'] = datetime.now().isoformat()
            return data

        # Create FSM with config dict
        config = {
            'name': 'validation_pipeline',
            'states': [
                {'name': 'validate', 'type': 'START', 'function': 'validate'},
                {'name': 'transform', 'type': 'END', 'function': 'transform'}
            ],
            'arcs': [
                {'from': 'validate', 'to': 'transform'}
            ]
        }

        # Initialize with custom functions and resources
        fsm = SimpleFSM(
            config=config,
            data_mode=DataHandlingMode.COPY,
            resources={
                'database': {
                    'type': 'DATABASE',
                    'backend': 'memory'
                }
            },
            custom_functions={
                'validate': validate,
                'transform': transform
            }
        )

        # Process data
        result = fsm.process({'required_field': 'value'})
        print(f"Success: {result['success']}")

        fsm.close()
        ```

        Batch processing with progress callback:

        ```python
        # Define progress callback
        def on_progress(current, total):
            pct = (current / total) * 100
            print(f"Progress: {current}/{total} ({pct:.1f}%)")

        # Process batch
        records = [{'id': i, 'text': f'Record {i}'} for i in range(100)]
        results = fsm.process_batch(
            data=records,
            batch_size=10,
            max_workers=4,
            on_progress=on_progress
        )

        # Check results
        successful = sum(1 for r in results if r['success'])
        print(f"Processed {successful}/{len(records)} successfully")
        ```

    See Also:
        - :class:`AsyncSimpleFSM`: Async version for production applications
        - :class:`AdvancedFSM`: Full control with debugging capabilities
        - :class:`DataHandlingMode`: Data processing mode options
        - :func:`process_file`: Convenience function for file processing
        - :func:`batch_process`: Convenience function for batch processing
        - :func:`validate_data`: Convenience function for data validation
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
            config: FSM configuration. Can be:
                - Path to YAML/JSON config file (str or Path)
                - Dictionary containing config (inline configuration)
                Must define states, arcs, and optionally resources.
            data_mode: Data handling mode controlling how data is passed between
                states. Options:
                - DataHandlingMode.COPY (default): Deep copy for safety
                - DataHandlingMode.REFERENCE: Lazy loading with locking
                - DataHandlingMode.DIRECT: In-place modification (fastest)
            resources: Optional resource configurations. Dict mapping resource
                names to configuration dicts. Each config must have a 'type' key
                (DATABASE, FILESYSTEM, HTTP, etc.) and type-specific parameters.
                Example: {'db': {'type': 'DATABASE', 'backend': 'postgres', ...}}
            custom_functions: Optional custom state functions. Dict mapping function
                names to callables. Functions receive data dict and return data dict.
                Function names must match 'function' fields in state definitions.
                Example: {'my_func': lambda data: {'result': data['input'] * 2}}

        Example:
            From configuration file:

            ```python
            from dataknobs_fsm.api.simple import SimpleFSM

            # Load from YAML file
            fsm = SimpleFSM('config.yaml')
            ```

            With inline configuration:

            ```python
            config = {
                'name': 'simple_pipeline',
                'states': [
                    {'name': 'start', 'type': 'START'},
                    {'name': 'process', 'type': 'NORMAL', 'function': 'transform'},
                    {'name': 'end', 'type': 'END'}
                ],
                'arcs': [
                    {'from': 'start', 'to': 'process'},
                    {'from': 'process', 'to': 'end'}
                ]
            }

            def transform(data):
                data['transformed'] = True
                return data

            fsm = SimpleFSM(
                config=config,
                custom_functions={'transform': transform}
            )
            ```

            With data mode selection:

            ```python
            from dataknobs_fsm.core.data_modes import DataHandlingMode

            # Use COPY for safe concurrent processing
            fsm_safe = SimpleFSM('config.yaml', data_mode=DataHandlingMode.COPY)

            # Use REFERENCE for memory efficiency
            fsm_efficient = SimpleFSM('config.yaml', data_mode=DataHandlingMode.REFERENCE)

            # Use DIRECT for maximum performance (single-threaded only)
            fsm_fast = SimpleFSM('config.yaml', data_mode=DataHandlingMode.DIRECT)
            ```

            With resources:

            ```python
            resources = {
                'database': {
                    'type': 'DATABASE',
                    'backend': 'postgres',
                    'host': 'localhost',
                    'database': 'mydb',
                    'user': 'admin',
                    'password': 'secret'
                },
                'http_client': {
                    'type': 'HTTP',
                    'base_url': 'https://api.example.com',
                    'timeout': 30
                }
            }

            fsm = SimpleFSM('config.yaml', resources=resources)
            ```
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

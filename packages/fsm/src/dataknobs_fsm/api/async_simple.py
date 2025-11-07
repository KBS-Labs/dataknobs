"""Async-first API for production FSM operations.

This module provides an async-first interface for FSM operations, designed to
work natively in async contexts (web services, FastAPI, async applications)
without any asyncio.run() overhead. This is the recommended API for production
use when your application is already async.

Architecture:
    AsyncSimpleFSM is the foundation for the dataknobs-fsm async API tier:

    **Design Philosophy:**
    - Async/await native - no blocking calls, no thread overhead
    - Production-ready with proper error handling and resource management
    - High concurrency support - process thousands of requests concurrently
    - Memory efficient - streaming support for large datasets
    - Framework agnostic - works with FastAPI, aiohttp, asyncio, etc.

    **Compared to SimpleFSM:**
    - SimpleFSM: Synchronous wrapper with event loop overhead
    - AsyncSimpleFSM: Native async, no overhead, better performance

    **Compared to AdvancedFSM:**
    - AsyncSimpleFSM: Simple API, automatic execution
    - AdvancedFSM: Manual control, debugging, profiling

Async Patterns:
    This module enables several async patterns for production systems:

    **Web Service Integration (FastAPI):**
    ```python
    from fastapi import FastAPI
    from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
    from dataknobs_fsm.core.data_modes import DataHandlingMode

    app = FastAPI()

    # Initialize FSM at startup
    fsm = None

    @app.on_event("startup")
    async def startup():
        global fsm
        fsm = AsyncSimpleFSM(
            'pipeline.yaml',
            data_mode=DataHandlingMode.COPY  # Safe for concurrent requests
        )

    @app.on_event("shutdown")
    async def shutdown():
        if fsm:
            await fsm.close()

    @app.post("/process")
    async def process_endpoint(data: dict):
        result = await fsm.process(data)
        return result
    ```

    **Concurrent Processing:**
    ```python
    import asyncio
    from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

    fsm = AsyncSimpleFSM('config.yaml')

    # Process multiple requests concurrently
    async def process_many(items):
        tasks = [fsm.process(item) for item in items]
        results = await asyncio.gather(*tasks)
        return results

    # Run
    items = [{'id': i, 'text': f'Item {i}'} for i in range(100)]
    results = await process_many(items)
    ```

    **Streaming Large Files:**
    ```python
    # Memory-efficient processing of large files
    fsm = AsyncSimpleFSM('pipeline.yaml')

    stats = await fsm.process_stream(
        source='large_input.jsonl',
        sink='output.jsonl',
        chunk_size=1000,
        use_streaming=True  # Memory-efficient mode
    )
    print(f"Processed {stats['total_processed']} records")
    print(f"Throughput: {stats['throughput']:.2f} records/sec")
    ```

    **Background Task Processing:**
    ```python
    import asyncio
    from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

    async def background_processor(queue: asyncio.Queue):
        fsm = AsyncSimpleFSM('processor.yaml')

        try:
            while True:
                # Get work from queue
                item = await queue.get()
                if item is None:  # Shutdown signal
                    break

                # Process asynchronously
                result = await fsm.process(item)

                # Handle result
                if not result['success']:
                    print(f"Error: {result['error']}")

                queue.task_done()
        finally:
            await fsm.close()

    # Start background processor
    work_queue = asyncio.Queue()
    task = asyncio.create_task(background_processor(work_queue))
    ```

Production Considerations:
    **Concurrency:**
    - Use DataHandlingMode.COPY for concurrent processing (default)
    - Each request gets its own data copy - safe for parallel execution
    - For high throughput, use process_batch() with appropriate max_workers

    **Resource Management:**
    - Always call close() to release database connections, file handles, etc.
    - Use startup/shutdown hooks in web frameworks (FastAPI, aiohttp)
    - Configure connection pooling in resource definitions

    **Error Handling:**
    - Process methods return {'success': bool, 'error': str} for graceful degradation
    - Use try/except for critical failures that should halt execution
    - Configure retry logic in error_recovery patterns

    **Memory Management:**
    - Use use_streaming=True for large file processing
    - Configure appropriate chunk_size based on available memory
    - Monitor memory usage with profiling tools

    **Performance Optimization:**
    - Use REFERENCE mode for read-only transformations (memory-efficient)
    - Use DIRECT mode for single-threaded pipelines (fastest)
    - Tune batch_size and max_workers for optimal CPU utilization
    - Profile with AdvancedFSM.profile_execution() to identify bottlenecks

Example:
    Complete production FastAPI service:

    ```python
    from fastapi import FastAPI, BackgroundTasks, HTTPException
    from pydantic import BaseModel
    from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
    from dataknobs_fsm.core.data_modes import DataHandlingMode
    import asyncio

    app = FastAPI(title="FSM Processing Service")

    # Global FSM instance
    fsm: AsyncSimpleFSM | None = None

    # Request/Response models
    class ProcessRequest(BaseModel):
        text: str
        metadata: dict = {}

    class ProcessResponse(BaseModel):
        success: bool
        result: dict
        path: list[str]
        error: str | None = None

    # Lifecycle management
    @app.on_event("startup")
    async def startup():
        global fsm
        # Initialize FSM with production config
        fsm = AsyncSimpleFSM(
            config='production.yaml',
            data_mode=DataHandlingMode.COPY,  # Safe for concurrent requests
            resources={
                'database': {
                    'type': 'DATABASE',
                    'backend': 'postgres',
                    'host': 'db.prod.example.com',
                    'database': 'app_data',
                    'pool_size': 20  # Connection pooling
                }
            }
        )
        print("FSM initialized")

    @app.on_event("shutdown")
    async def shutdown():
        if fsm:
            await fsm.close()
            print("FSM closed")

    # Endpoints
    @app.post("/process", response_model=ProcessResponse)
    async def process_single(request: ProcessRequest):
        \"\"\"Process a single request through the FSM.\"\"\"
        if not fsm:
            raise HTTPException(status_code=503, detail="Service not ready")

        result = await fsm.process({
            'text': request.text,
            'metadata': request.metadata
        })

        return ProcessResponse(
            success=result['success'],
            result=result['data'],
            path=result['path'],
            error=result.get('error')
        )

    @app.post("/batch", response_model=list[ProcessResponse])
    async def process_batch_endpoint(requests: list[ProcessRequest]):
        \"\"\"Process multiple requests in parallel.\"\"\"
        if not fsm:
            raise HTTPException(status_code=503, detail="Service not ready")

        # Convert to processing format
        data = [
            {'text': req.text, 'metadata': req.metadata}
            for req in requests
        ]

        # Process batch
        results = await fsm.process_batch(
            data=data,
            batch_size=10,
            max_workers=4
        )

        # Format responses
        return [
            ProcessResponse(
                success=r['success'],
                result=r['data'],
                path=r['path'],
                error=r.get('error')
            )
            for r in results
        ]

    @app.post("/file")
    async def process_file_endpoint(
        background_tasks: BackgroundTasks,
        input_path: str,
        output_path: str
    ):
        \"\"\"Process a file in the background.\"\"\"
        if not fsm:
            raise HTTPException(status_code=503, detail="Service not ready")

        async def process_file_task():
            stats = await fsm.process_stream(
                source=input_path,
                sink=output_path,
                chunk_size=1000,
                use_streaming=True
            )
            print(f"File processing complete: {stats}")

        background_tasks.add_task(process_file_task)
        return {"status": "processing", "message": "File processing started"}

    # Run: uvicorn app:app --host 0.0.0.0 --port 8000
    ```

See Also:
    - :class:`SimpleFSM`: Synchronous wrapper for scripts and prototypes
    - :class:`AdvancedFSM`: Advanced API with debugging and profiling
    - :class:`DataHandlingMode`: Data processing mode options
    - :mod:`dataknobs_fsm.patterns.error_recovery`: Production error handling patterns
    - :mod:`dataknobs_fsm.resources.manager`: Resource management and pooling
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
    """Async-first FSM interface for production workflows.

    This class provides a fully asynchronous API for FSM operations, designed
    to work natively in async contexts without blocking calls or thread overhead.
    This is the recommended FSM implementation for production systems.

    AsyncSimpleFSM handles all the complexity of async execution, resource
    management, and concurrent processing while providing a simple, clean API.
    It's optimized for high-throughput scenarios with proper connection pooling,
    error handling, and memory management.

    Attributes:
        data_mode (DataHandlingMode): Data processing mode (COPY/REFERENCE/DIRECT)
        _config: Loaded FSM configuration
        _fsm (FSM): Core FSM engine
        _resource_manager (ResourceManager): Resource lifecycle and pooling manager
        _async_engine (AsyncExecutionEngine): Async execution engine

    Methods:
        process: Process single record asynchronously
        process_batch: Process multiple records with concurrency control
        process_stream: Stream-process large datasets
        validate: Validate data against schema
        get_states: List all FSM state names
        get_resources: List all registered resources
        close: Release all resources and cleanup

    Production Use Cases:
        **Web API Backend:**
        Handle thousands of concurrent requests in FastAPI/aiohttp services.
        Each request processes independently with automatic resource pooling.

        **Data Pipeline Processing:**
        Transform large datasets with memory-efficient streaming and parallel
        batch processing. Configurable chunk sizes and worker counts.

        **Real-time Event Processing:**
        Process events from queues (RabbitMQ, Kafka) with async consumers.
        High throughput with concurrent processing of independent events.

        **Batch Job Processing:**
        Schedule and run large batch jobs with progress tracking and error
        handling. Configurable parallelism for optimal resource utilization.

    Note:
        **Concurrency Safety:**
        AsyncSimpleFSM is safe for concurrent use when using DataHandlingMode.COPY
        (default). Each process() call operates on independent data. For REFERENCE
        or DIRECT modes, ensure external synchronization.

        **Resource Pooling:**
        Resources (databases, HTTP clients) use connection pooling automatically.
        Configure pool_size in resource definitions for optimal performance.

        **Error Handling:**
        Process methods return success/error dicts rather than raising exceptions,
        allowing graceful degradation. Use try/except only for critical failures.

        **Memory Management:**
        For large datasets, use process_stream() with use_streaming=True for
        constant memory usage regardless of file size.

    Example:
        Basic async processing:

        ```python
        import asyncio
        from dataknobs_fsm.api.async_simple import AsyncSimpleFSM

        async def main():
            # Create FSM
            fsm = AsyncSimpleFSM('pipeline.yaml')

            try:
                # Process single record
                result = await fsm.process({
                    'text': 'Input data',
                    'metadata': {'source': 'api'}
                })

                if result['success']:
                    print(f"Result: {result['data']}")
                    print(f"States: {' -> '.join(result['path'])}")
                else:
                    print(f"Error: {result['error']}")
            finally:
                await fsm.close()

        asyncio.run(main())
        ```

        Concurrent processing with asyncio.gather:

        ```python
        async def process_concurrent():
            fsm = AsyncSimpleFSM('config.yaml')

            try:
                # Create tasks for concurrent execution
                tasks = [
                    fsm.process({'id': 1, 'text': 'Item 1'}),
                    fsm.process({'id': 2, 'text': 'Item 2'}),
                    fsm.process({'id': 3, 'text': 'Item 3'})
                ]

                # Execute concurrently
                results = await asyncio.gather(*tasks)

                # Check results
                for i, result in enumerate(results, 1):
                    status = "✓" if result['success'] else "✗"
                    print(f"{status} Item {i}: {result.get('data', result.get('error'))}")
            finally:
                await fsm.close()
        ```

        Production web service pattern:

        ```python
        from fastapi import FastAPI
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: initialize FSM
            app.state.fsm = AsyncSimpleFSM(
                'production.yaml',
                resources={
                    'db': {
                        'type': 'DATABASE',
                        'backend': 'postgres',
                        'pool_size': 20,  # Connection pooling
                        'host': 'db.prod.example.com'
                    }
                }
            )
            yield
            # Shutdown: cleanup
            await app.state.fsm.close()

        app = FastAPI(lifespan=lifespan)

        @app.post("/process")
        async def process(request: dict):
            result = await app.state.fsm.process(request)
            return result
        ```

        Batch processing with progress tracking:

        ```python
        async def process_with_progress():
            fsm = AsyncSimpleFSM('pipeline.yaml')

            # Progress callback
            def on_progress(current, total):
                pct = (current / total) * 100
                print(f"Progress: {current}/{total} ({pct:.1f}%)")

            try:
                records = [{'id': i} for i in range(1000)]
                results = await fsm.process_batch(
                    data=records,
                    batch_size=50,
                    max_workers=10,
                    on_progress=on_progress
                )

                successful = sum(1 for r in results if r['success'])
                print(f"Success rate: {successful}/{len(results)}")
            finally:
                await fsm.close()
        ```

        Stream processing large files:

        ```python
        async def process_large_file():
            fsm = AsyncSimpleFSM('transform.yaml')

            try:
                # Memory-efficient streaming
                stats = await fsm.process_stream(
                    source='input_100gb.jsonl',
                    sink='output.jsonl',
                    chunk_size=1000,
                    use_streaming=True  # Constant memory usage
                )

                print(f"Processed: {stats['total_processed']}")
                print(f"Success: {stats['successful']}")
                print(f"Failed: {stats['failed']}")
                print(f"Duration: {stats['duration']:.2f}s")
                print(f"Throughput: {stats['throughput']:.2f} records/sec")
            finally:
                await fsm.close()
        ```

    See Also:
        - :class:`SimpleFSM`: Synchronous wrapper for scripts
        - :class:`AdvancedFSM`: Advanced API with debugging
        - :func:`create_async_fsm`: Factory function for creating instances
        - :mod:`dataknobs_fsm.execution.async_engine`: Async execution engine
        - :mod:`dataknobs_fsm.resources.manager`: Resource management
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

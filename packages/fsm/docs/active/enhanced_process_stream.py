"""Enhanced process_stream implementation supporting multiple file formats."""

import asyncio
import csv
import json
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Union, Callable


async def process_stream_enhanced(
    self,
    source: Union[str, AsyncIterator[Dict[str, Any]]],
    sink: str | None = None,
    chunk_size: int = 100,
    on_progress: Union[Callable, None] = None,
    input_format: str = 'auto',  # 'auto', 'jsonl', 'json', 'csv', 'text'
    text_field_name: str = 'text',  # Field name for text lines
    csv_delimiter: str = ',',
    csv_has_header: bool = True
) -> Dict[str, Any]:
    """Process a stream of data through the FSM with multiple format support.

    Args:
        source: Data source (file path or async iterator)
        sink: Optional output destination
        chunk_size: Size of processing chunks
        on_progress: Optional progress callback
        input_format: Input file format ('auto' to detect, or specific format)
        text_field_name: Field name to use when converting text lines to dicts
        csv_delimiter: CSV delimiter character
        csv_has_header: Whether CSV file has header row

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
        file_path = Path(source)

        # Auto-detect format if needed
        if input_format == 'auto':
            suffix = file_path.suffix.lower()
            if suffix in ['.jsonl', '.ndjson']:
                input_format = 'jsonl'
            elif suffix == '.json':
                input_format = 'json'
            elif suffix in ['.csv', '.tsv']:
                input_format = 'csv'
                if suffix == '.tsv':
                    csv_delimiter = '\t'
            elif suffix in ['.txt', '.text', '.log']:
                input_format = 'text'
            else:
                # Default to text for unknown extensions
                input_format = 'text'

        # Create appropriate file reader based on format
        if input_format == 'jsonl':
            async def file_reader():
                with open(file_path) as f:
                    for line in f:
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue

        elif input_format == 'json':
            async def file_reader():
                with open(file_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                    else:
                        yield data

        elif input_format == 'csv':
            async def file_reader():
                with open(file_path, newline='') as f:
                    reader = csv.DictReader(f, delimiter=csv_delimiter) if csv_has_header else csv.reader(f, delimiter=csv_delimiter)

                    if csv_has_header:
                        # DictReader yields dicts
                        for row in reader:
                            yield row
                    else:
                        # Regular reader yields lists, convert to dict with numeric keys
                        for row in reader:
                            yield {f'col_{i}': val for i, val in enumerate(row)}

        elif input_format == 'text':
            async def file_reader():
                with open(file_path) as f:
                    for line in f:
                        line = line.rstrip('\n\r')  # Remove line endings but preserve other whitespace
                        if line or not skip_empty_lines:  # Include empty lines if not skipping
                            yield {text_field_name: line}

        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        stream_source = file_reader()
    else:
        # Already an async iterator
        stream_source = source

    # Handle sink with format detection
    sink_func = None
    if sink:
        sink_path = Path(sink)
        sink_format = 'jsonl'  # Default output format

        # Auto-detect output format
        suffix = sink_path.suffix.lower()
        if suffix in ['.csv', '.tsv']:
            sink_format = 'csv'
            output_delimiter = '\t' if suffix == '.tsv' else ','
        elif suffix == '.json':
            sink_format = 'json'

        if sink_format == 'jsonl':
            def write_to_file(results):
                from dataknobs_fsm.utils.json_encoder import dumps
                with open(sink, 'a') as f:
                    for result in results:
                        f.write(dumps(result) + '\n')

        elif sink_format == 'csv':
            # CSV writer needs to know fields upfront
            csv_writer = None
            csv_file = None

            def write_to_file(results):
                nonlocal csv_writer, csv_file

                if not csv_file:
                    csv_file = open(sink, 'w', newline='')

                for result in results:
                    if not csv_writer:
                        # Initialize CSV writer with fields from first result
                        fieldnames = list(result.keys())
                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=output_delimiter)
                        csv_writer.writeheader()
                    csv_writer.writerow(result)

        elif sink_format == 'json':
            all_results = []

            def write_to_file(results):
                nonlocal all_results
                all_results.extend(results)
                # Write all at once when done
                with open(sink, 'w') as f:
                    json.dump(all_results, f, indent=2)

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


# Example usage patterns:
"""
# Process plain text file
results = await fsm.process_stream(
    source='input.txt',
    input_format='text',
    text_field_name='content',
    sink='output.jsonl'
)

# Process CSV with custom delimiter
results = await fsm.process_stream(
    source='data.csv',
    input_format='csv',
    csv_delimiter=',',
    csv_has_header=True,
    sink='results.jsonl'
)

# Auto-detect format from extension
results = await fsm.process_stream(
    source='data.jsonl',  # Auto-detects JSONL
    sink='output.csv'     # Auto-detects CSV output
)

# Process TSV file
results = await fsm.process_stream(
    source='data.tsv',
    input_format='auto',  # Will detect TSV and use tab delimiter
    sink='output.json'
)
"""
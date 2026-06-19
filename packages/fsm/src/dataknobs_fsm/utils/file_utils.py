"""File processing utilities for FSM.

This module provides utilities for reading and writing various file formats
in the context of FSM stream processing.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Union

from dataknobs_common import aiter_sync_in_thread


def detect_format(file_path: Union[str, Path], for_output: bool = False) -> str:
    """Detect file format from extension.

    Args:
        file_path: Path to the file
        for_output: If True, detect output format (defaults to jsonl for unknown)

    Returns:
        Detected format string
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in ['.jsonl', '.ndjson']:
        return 'jsonl'
    elif suffix == '.json':
        return 'json'
    elif suffix in ['.csv', '.tsv']:
        return 'csv'
    elif suffix in ['.txt', '.text', '.log']:
        return 'text'
    else:
        # Default to jsonl for output, text for input
        return 'jsonl' if for_output else 'text'


def get_csv_delimiter(file_path: Union[str, Path]) -> str:
    """Get CSV delimiter based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        Delimiter character
    """
    path = Path(file_path)
    return '\t' if path.suffix.lower() == '.tsv' else ','


async def create_file_reader(
    file_path: Union[str, Path],
    input_format: str = 'auto',
    text_field_name: str = 'text',
    csv_delimiter: str = ',',
    csv_has_header: bool = True,
    skip_empty_lines: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """Create an async iterator for reading files in various formats.

    Args:
        file_path: Path to the input file
        input_format: File format ('auto', 'jsonl', 'json', 'csv', 'text')
        text_field_name: Field name for text lines
        csv_delimiter: Delimiter for CSV files
        csv_has_header: Whether CSV has header row
        skip_empty_lines: Skip empty lines in text files

    Yields:
        Dictionaries representing each record from the file

    Raises:
        ValueError: If input format is not supported
    """
    path = Path(file_path)

    # Auto-detect format if needed
    if input_format == 'auto':
        input_format = detect_format(path)
        if input_format == 'csv' and path.suffix.lower() == '.tsv':
            csv_delimiter = '\t'

    if input_format == 'jsonl':
        async for record in read_jsonl_file(path):
            yield record

    elif input_format == 'json':
        async for record in read_json_file(path):
            yield record

    elif input_format == 'csv':
        async for record in read_csv_file(path, csv_delimiter, csv_has_header):
            yield record

    elif input_format == 'text':
        async for record in read_text_file(path, text_field_name, skip_empty_lines):
            yield record

    else:
        raise ValueError(f"Unsupported input format: {input_format}")


async def read_jsonl_file(file_path: Path) -> AsyncIterator[Dict[str, Any]]:
    """Read a JSONL (JSON Lines) file.

    The blocking ``open`` + line iteration runs on a worker thread via
    :func:`~dataknobs_common.aiter_sync_in_thread`, so the file read never
    stalls the event loop while streaming stays lazy (bounded look-ahead).

    Args:
        file_path: Path to the JSONL file

    Yields:
        Dictionaries from each valid JSON line
    """
    async for record in aiter_sync_in_thread(lambda: _read_jsonl_sync(file_path)):
        yield record


def _read_jsonl_sync(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Synchronous JSONL line reader — driven on a worker thread."""
    with open(file_path) as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue


async def read_json_file(file_path: Path) -> AsyncIterator[Dict[str, Any]]:
    """Read a JSON file (single object or array).

    The whole-file read + parse is offloaded with :func:`asyncio.to_thread`
    so the blocking ``open`` / ``json.load`` never runs on the event loop.

    Args:
        file_path: Path to the JSON file

    Yields:
        Dictionary or dictionaries from the JSON file
    """
    data = await asyncio.to_thread(_load_json, file_path)
    if isinstance(data, list):
        for item in data:
            yield item
    else:
        yield data


def _load_json(file_path: Path) -> Any:
    """Synchronous whole-file JSON load — run via ``to_thread``."""
    with open(file_path) as f:
        return json.load(f)


async def read_csv_file(
    file_path: Path,
    delimiter: str = ',',
    has_header: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """Read a CSV file.

    The blocking ``open`` + row iteration runs on a worker thread via
    :func:`~dataknobs_common.aiter_sync_in_thread`, keeping the read off
    the event loop while streaming stays lazy.

    Args:
        file_path: Path to the CSV file
        delimiter: CSV delimiter character
        has_header: Whether the CSV has a header row

    Yields:
        Dictionaries representing each row
    """
    async for row in aiter_sync_in_thread(
        lambda: _read_csv_sync(file_path, delimiter, has_header)
    ):
        yield row


def _read_csv_sync(
    file_path: Path,
    delimiter: str,
    has_header: bool,
) -> Iterator[Dict[str, Any]]:
    """Synchronous CSV row reader — driven on a worker thread."""
    with open(file_path, newline='') as f:
        if has_header:
            dict_reader = csv.DictReader(f, delimiter=delimiter)
            yield from dict_reader
        else:
            list_reader = csv.reader(f, delimiter=delimiter)
            for row_list in list_reader:
                yield {f'col_{i}': val for i, val in enumerate(row_list)}


async def read_text_file(
    file_path: Path,
    field_name: str = 'text',
    skip_empty: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """Read a plain text file line by line.

    The blocking ``open`` + line iteration runs on a worker thread via
    :func:`~dataknobs_common.aiter_sync_in_thread`, keeping the read off
    the event loop while streaming stays lazy.

    Args:
        file_path: Path to the text file
        field_name: Field name to use for each line
        skip_empty: Skip empty lines

    Yields:
        Dictionaries with each line as a field
    """
    async for record in aiter_sync_in_thread(
        lambda: _read_text_sync(file_path, field_name, skip_empty)
    ):
        yield record


def _read_text_sync(
    file_path: Path,
    field_name: str,
    skip_empty: bool,
) -> Iterator[Dict[str, Any]]:
    """Synchronous text line reader — driven on a worker thread."""
    with open(file_path) as f:
        for line in f:
            sline = line.rstrip('\n\r')
            if sline or not skip_empty:
                yield {field_name: sline}


def create_file_writer(
    file_path: Union[str, Path],
    output_format: str | None = None
) -> tuple[Callable[[List[Dict[str, Any]]], None], Callable[[], None] | None]:
    """Create a file writer function for the specified format.

    Args:
        file_path: Path to the output file
        output_format: Output format (auto-detected if None)

    Returns:
        Tuple of (writer_function, cleanup_function)
        The cleanup_function is None for formats that don't need cleanup
    """
    path = Path(file_path)

    # Auto-detect format if not specified
    if output_format is None:
        output_format = detect_format(path, for_output=True)

    if output_format == 'jsonl':
        return create_jsonl_writer(path), None

    elif output_format == 'csv':
        delimiter = get_csv_delimiter(path)
        return create_csv_writer(path, delimiter)

    elif output_format == 'json':
        return create_json_writer(path)

    else:
        # Default to JSONL
        return create_jsonl_writer(path), None


def create_jsonl_writer(file_path: Path) -> Callable[[List[Dict[str, Any]]], None]:
    """Create a JSONL writer function.

    Args:
        file_path: Path to the output file

    Returns:
        Writer function that appends to JSONL file
    """
    def write_jsonl(results: List[Dict[str, Any]]) -> None:
        from dataknobs_fsm.utils.json_encoder import dumps
        with open(file_path, 'a') as f:
            for result in results:
                f.write(dumps(result) + '\n')

    return write_jsonl


def create_csv_writer(
    file_path: Path,
    delimiter: str = ','
) -> tuple[Callable[[List[Dict[str, Any]]], None], Callable[[], None]]:
    """Create a CSV writer function with state management.

    Args:
        file_path: Path to the output file
        delimiter: CSV delimiter character

    Returns:
        Tuple of (writer_function, cleanup_function)
    """
    csv_writer: csv.DictWriter | None = None
    csv_file: Any | None = None

    def write_csv(results: List[Dict[str, Any]]) -> None:
        nonlocal csv_writer, csv_file

        if not csv_file:
            csv_file = open(file_path, 'w', newline='')

        for result in results:
            if not csv_writer:
                # Initialize CSV writer with fields from first result
                fieldnames = list(result.keys())
                csv_writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames,
                    delimiter=delimiter
                )
                csv_writer.writeheader()
            csv_writer.writerow(result)

    def cleanup() -> None:
        if csv_file:
            csv_file.close()

    return write_csv, cleanup


def create_json_writer(
    file_path: Path
) -> tuple[Callable[[List[Dict[str, Any]]], None], Callable[[], None]]:
    """Create a JSON writer function that accumulates results.

    Args:
        file_path: Path to the output file

    Returns:
        Tuple of (writer_function, cleanup_function)
    """
    all_results: List[Dict[str, Any]] = []

    def write_json(results: List[Dict[str, Any]]) -> None:
        nonlocal all_results
        all_results.extend(results)

    def cleanup() -> None:
        # Write all results at once
        with open(file_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    return write_json, cleanup

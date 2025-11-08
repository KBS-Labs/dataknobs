"""Built-in streaming functions for FSM.

This module provides streaming-related functions that can be referenced
in FSM configurations for processing large data sets efficiently.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from dataknobs_fsm.functions.base import ITransformFunction, TransformError
from dataknobs_fsm.streaming.core import IStreamSource


class ChunkReader(ITransformFunction):
    """Read data in chunks from a source."""

    def __init__(
        self,
        source: Union[str, IStreamSource],
        chunk_size: int = 1000,
        format: str = "auto",  # "auto", "json", "csv", "lines"
    ):
        """Initialize the chunk reader.
        
        Args:
            source: Data source (file path or stream source).
            chunk_size: Number of records per chunk.
            format: Data format to expect.
        """
        self.source = source
        self.chunk_size = chunk_size
        self.format = format

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by reading next chunk from source.
        
        Args:
            data: Input data (may contain chunk state).
            
        Returns:
            Data with next chunk of records.
        """
        # Get or initialize chunk state
        chunk_state = data.get("_chunk_state", {})
        
        if isinstance(self.source, str):
            # File source
            file_path = Path(self.source)
            if not file_path.exists():
                raise TransformError(f"File not found: {self.source}")
            
            # Determine format
            format = self.format
            if format == "auto":
                format = self._detect_format(file_path)
            
            # Read chunk based on format
            if format == "json":
                chunk = await self._read_json_chunk(file_path, chunk_state)
            elif format == "csv":
                chunk = await self._read_csv_chunk(file_path, chunk_state)
            elif format == "lines":
                chunk = await self._read_lines_chunk(file_path, chunk_state)
            else:
                raise TransformError(f"Unsupported format: {format}")
        
        else:
            # Stream source
            chunk = await self._read_stream_chunk(self.source, chunk_state)
        
        return {
            **data,
            "chunk": chunk["records"],
            "has_more": chunk["has_more"],
            "_chunk_state": chunk["state"],
        }

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            return "json"
        elif suffix == ".csv":
            return "csv"
        else:
            return "lines"

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        source_str = str(self.source) if isinstance(self.source, str) else "stream"
        return f"Read {self.chunk_size} records from {source_str} in {self.format} format"

    async def _read_json_chunk(
        self, file_path: Path, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read chunk from JSON file."""
        offset = state.get("offset", 0)
        
        # For JSON, we need to load the entire file (or use streaming JSON parser)
        with open(file_path) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            chunk = data[offset:offset + self.chunk_size]
            has_more = offset + self.chunk_size < len(data)
            new_offset = offset + len(chunk)
        else:
            # Single object
            if offset == 0:
                chunk = [data]
                has_more = False
                new_offset = 1
            else:
                chunk = []
                has_more = False
                new_offset = offset
        
        return {
            "records": chunk,
            "has_more": has_more,
            "state": {"offset": new_offset},
        }

    async def _read_csv_chunk(
        self, file_path: Path, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read chunk from CSV file."""
        offset = state.get("offset", 0)
        records = []
        
        with open(file_path) as f:
            reader = csv.DictReader(f)
            
            # Skip to offset
            for _ in range(offset):
                try:
                    next(reader)
                except StopIteration:
                    break
            
            # Read chunk
            for _ in range(self.chunk_size):
                try:
                    records.append(next(reader))
                except StopIteration:
                    break
        
        has_more = len(records) == self.chunk_size
        new_offset = offset + len(records)
        
        return {
            "records": records,
            "has_more": has_more,
            "state": {"offset": new_offset},
        }

    async def _read_lines_chunk(
        self, file_path: Path, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read chunk of lines from file."""
        offset = state.get("offset", 0)
        records = []
        
        with open(file_path) as f:
            # Skip to offset
            for _ in range(offset):
                if not f.readline():
                    break
            
            # Read chunk
            for _ in range(self.chunk_size):
                line = f.readline()
                if not line:
                    break
                records.append({"line": line.strip()})
        
        has_more = len(records) == self.chunk_size
        new_offset = offset + len(records)
        
        return {
            "records": records,
            "has_more": has_more,
            "state": {"offset": new_offset},
        }

    async def _read_stream_chunk(
        self, source: IStreamSource, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read chunk from stream source."""
        records = []
        
        async for record in source.read(self.chunk_size):
            records.append(record)
        
        has_more = len(records) == self.chunk_size
        
        return {
            "records": records,
            "has_more": has_more,
            "state": {"stream_position": source.position if hasattr(source, "position") else None},
        }


class RecordParser(ITransformFunction):
    """Parse records from various formats."""

    def __init__(
        self,
        format: str,
        field: str = "raw",
        output_field: str = "parsed",
        options: Dict[str, Any] | None = None,
    ):
        """Initialize the record parser.
        
        Args:
            format: Format to parse ("json", "csv", "xml", "yaml").
            field: Field containing raw data to parse.
            output_field: Field to store parsed data.
            options: Format-specific parsing options.
        """
        self.format = format
        self.field = field
        self.output_field = output_field
        self.options = options or {}

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by parsing records.
        
        Args:
            data: Input data containing raw records.
            
        Returns:
            Data with parsed records.
        """
        raw_data = data.get(self.field)
        if raw_data is None:
            return data
        
        try:
            if self.format == "json":
                parsed = self._parse_json(raw_data)
            elif self.format == "csv":
                parsed = self._parse_csv(raw_data)
            elif self.format == "yaml":
                parsed = self._parse_yaml(raw_data)
            elif self.format == "xml":
                parsed = self._parse_xml(raw_data)
            else:
                raise TransformError(f"Unsupported format: {self.format}")
            
            return {
                **data,
                self.output_field: parsed,
            }
        
        except Exception as e:
            raise TransformError(f"Failed to parse {self.format}: {e}") from e

    def _parse_json(self, raw: Union[str, bytes]) -> Any:
        """Parse JSON data."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _parse_csv(self, raw: Union[str, bytes]) -> List[Dict[str, Any]]:
        """Parse CSV data."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        
        import io
        reader = csv.DictReader(io.StringIO(raw), **self.options)
        return list(reader)

    def _parse_yaml(self, raw: Union[str, bytes]) -> Any:
        """Parse YAML data."""
        import yaml
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return yaml.safe_load(raw)

    def _parse_xml(self, raw: Union[str, bytes]) -> Dict[str, Any]:
        """Parse XML data."""
        import xml.etree.ElementTree as ET
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        
        root = ET.fromstring(raw)
        return self._xml_to_dict(root)

    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}

        # Add attributes
        if element.attrib:
            result["@attributes"] = element.attrib

        # Add text content
        if element.text and element.text.strip():
            result["text"] = element.text.strip()

        # Add children
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                # Convert to list if multiple children with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Parse {self.format} data from '{self.field}' to '{self.output_field}'"


class FileAppender(ITransformFunction):
    """Append data to a file."""

    def __init__(
        self,
        file_path: str,
        format: str = "json",  # "json", "csv", "lines"
        field: str = "data",
        buffer_size: int = 100,
        create_if_missing: bool = True,
    ):
        """Initialize the file appender.
        
        Args:
            file_path: Path to file to append to.
            format: Format to write data in.
            field: Field containing data to append.
            buffer_size: Number of records to buffer before writing.
            create_if_missing: Create file if it doesn't exist.
        """
        self.file_path = Path(file_path)
        self.format = format
        self.field = field
        self.buffer_size = buffer_size
        self.create_if_missing = create_if_missing
        self._buffer: List[Any] = []

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by appending to file.
        
        Args:
            data: Input data containing records to append.
            
        Returns:
            Data with append status.
        """
        records = data.get(self.field)
        if records is None:
            return data
        
        # Add to buffer
        if isinstance(records, list):
            self._buffer.extend(records)
        else:
            self._buffer.append(records)
        
        # Write if buffer is full
        written = 0
        if len(self._buffer) >= self.buffer_size:
            written = await self._write_buffer()
        
        return {
            **data,
            "appended_count": written,
            "buffer_size": len(self._buffer),
        }

    async def _write_buffer(self) -> int:
        """Write buffer to file."""
        if not self._buffer:
            return 0
        
        # Create file if needed
        if self.create_if_missing and not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.touch()
        
        count = len(self._buffer)
        
        if self.format == "json":
            # Append to JSON array
            existing = []
            if self.file_path.exists() and self.file_path.stat().st_size > 0:
                with open(self.file_path) as f:
                    existing = json.load(f)
            
            existing.extend(self._buffer)
            
            with open(self.file_path, "w") as f:
                json.dump(existing, f, indent=2)
        
        elif self.format == "csv":
            # Append to CSV
            import csv
            
            file_exists = self.file_path.exists() and self.file_path.stat().st_size > 0
            
            with open(self.file_path, "a", newline="") as f:
                if self._buffer and isinstance(self._buffer[0], dict):
                    writer = csv.DictWriter(f, fieldnames=self._buffer[0].keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(self._buffer)
                else:
                    writer = csv.writer(f)
                    writer.writerows(self._buffer)
        
        elif self.format == "lines":
            # Append lines
            with open(self.file_path, "a") as f:
                for record in self._buffer:
                    if isinstance(record, dict):
                        f.write(json.dumps(record) + "\n")
                    else:
                        f.write(str(record) + "\n")
        
        else:
            raise TransformError(f"Unsupported format: {self.format}")
        
        self._buffer.clear()
        return count

    async def flush(self) -> int:
        """Flush any remaining buffered data."""
        return await self._write_buffer()

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Append {self.format} data from '{self.field}' to {self.file_path}"


class StreamAggregator(ITransformFunction):
    """Aggregate streaming data using various functions."""

    def __init__(
        self,
        aggregations: Dict[str, Dict[str, Any]],
        group_by: List[str] | None = None,
        window_size: int | None = None,
    ):
        """Initialize the stream aggregator.
        
        Args:
            aggregations: Dictionary of aggregation specifications.
                         Keys are output field names, values are:
                         {"function": "sum|avg|min|max|count", "field": "source_field"}
            group_by: Fields to group by before aggregating.
            window_size: Number of records in sliding window.
        """
        self.aggregations = aggregations
        self.group_by = group_by
        self.window_size = window_size
        self._window: List[Dict[str, Any]] = []
        self._groups: Dict[tuple, List[Dict[str, Any]]] = {}

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by aggregating stream.
        
        Args:
            data: Input data (single record or batch).
            
        Returns:
            Data with aggregation results.
        """
        # Add to window/groups
        records = data.get("records", [data])
        
        if self.group_by:
            # Group-based aggregation
            for record in records:
                key = tuple(record.get(field) for field in self.group_by)
                if key not in self._groups:
                    self._groups[key] = []
                self._groups[key].append(record)
                
                # Apply window size per group
                if self.window_size and len(self._groups[key]) > self.window_size:
                    self._groups[key] = self._groups[key][-self.window_size:]
            
            # Compute aggregations per group
            results = []
            for key, group_records in self._groups.items():
                result = dict(zip(self.group_by, key, strict=False))
                for output_field, agg_spec in self.aggregations.items():
                    result[output_field] = self._compute_aggregation(group_records, agg_spec)
                results.append(result)
            
            return {**data, "aggregations": results}
        
        else:
            # Global aggregation
            self._window.extend(records)
            
            # Apply window size
            if self.window_size and len(self._window) > self.window_size:
                self._window = self._window[-self.window_size:]
            
            # Compute aggregations
            result = {}
            for output_field, agg_spec in self.aggregations.items():
                result[output_field] = self._compute_aggregation(self._window, agg_spec)
            
            return {**data, "aggregation": result}

    def _compute_aggregation(
        self, records: List[Dict[str, Any]], spec: Dict[str, Any]
    ) -> Any:
        """Compute a single aggregation."""
        func = spec["function"]
        field = spec.get("field")
        
        if func == "count":
            return len(records)
        
        if not field:
            raise TransformError(f"Field required for {func} aggregation")
        
        values: List[Any] = [r.get(field) for r in records if r.get(field) is not None]
        
        if not values:
            return None
        
        if func == "sum":
            return sum(values)  # type: ignore
        elif func == "avg":
            return sum(values) / len(values)  # type: ignore
        elif func == "min":
            return min(values)  # type: ignore
        elif func == "max":
            return max(values)  # type: ignore
        else:
            raise TransformError(f"Unknown aggregation function: {func}")

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        agg_list = list(self.aggregations.keys())[:3]
        agg_str = ", ".join(agg_list)
        if len(self.aggregations) > 3:
            agg_str += "..."
        group_str = f" grouped by {', '.join(self.group_by)}" if self.group_by else ""
        return f"Aggregate {agg_str}{group_str}"


# Convenience functions for creating streaming functions
def read_chunks(source: str, size: int = 1000, **kwargs) -> ChunkReader:
    """Create a ChunkReader."""
    return ChunkReader(source, size, **kwargs)


def parse(format: str, **kwargs) -> RecordParser:
    """Create a RecordParser."""
    return RecordParser(format, **kwargs)


def append_to_file(path: str, **kwargs) -> FileAppender:
    """Create a FileAppender."""
    return FileAppender(path, **kwargs)


def aggregate(**aggregations: Dict[str, Any]) -> StreamAggregator:
    """Create a StreamAggregator."""
    return StreamAggregator(aggregations)

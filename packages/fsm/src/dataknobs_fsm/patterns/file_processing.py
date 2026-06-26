"""File processing pattern implementation.

This module provides pre-configured FSM patterns for processing files,
including CSV, JSON, XML, and other formats with streaming support.
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, ClassVar, Dict, Iterator, List

from dataknobs_common import aiter_sync_in_thread
from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)

from dataknobs_fsm.core.data_modes import DataHandlingMode

from ..api.async_simple import AsyncSimpleFSM
from ..functions.base import ITransformFunction, TransformError


class FileFormat(Enum):
    """Supported file formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PARQUET = "parquet"
    TEXT = "text"
    BINARY = "binary"


class ProcessingMode(Enum):
    """File processing modes."""
    STREAM = "stream"  # Process file as stream
    BATCH = "batch"  # Process in batches
    WHOLE = "whole"  # Load entire file


@dataclass(frozen=True)
class FileProcessingConfig(StructuredConfig):
    """Configuration for file processing."""
    input_path: str
    output_path: str | None = None
    format: FileFormat | None = None  # Auto-detect if not specified
    mode: ProcessingMode = ProcessingMode.STREAM
    chunk_size: int = 1000
    parallel_chunks: int = 4
    encoding: str = "utf-8"
    
    # Processing options
    validation_schema: Dict[str, Any] | None = None
    transformations: List[Callable] | None = None
    filters: List[Callable] | None = None
    aggregations: Dict[str, Callable] | None = None
    
    # Output options
    output_format: FileFormat | None = None
    compression: str | None = None  # gzip, bz2, etc.
    partition_by: str | None = None  # Field to partition output
    
    # Format-specific configs
    json_config: Dict[str, Any] = field(default_factory=dict)
    log_config: Dict[str, Any] = field(default_factory=dict)


_VALIDATION_TYPE_MAP: Dict[str, type] = {
    "str": str, "string": str, "int": int, "integer": int,
    "float": float, "number": float, "bool": bool, "boolean": bool,
    "list": list, "array": list, "dict": dict, "object": dict,
}


class _FileTransform(ITransformFunction):
    """Per-record transform: apply each configured transformation in order.

    Wired into the FileProcessor FSM's ``transform`` state as a registered
    function (the proven ``custom_functions=`` idiom). Each transformation is a
    **map-style** callable ``record -> record``; they run in order. The step is
    synchronous so it executes identically on the async (batch/whole) and
    streaming execution paths. A non-dict return raises :class:`TransformError`
    rather than writing a corrupt record (the engine then reports the record as
    a failure, so it is counted as an error rather than written).
    """

    def __init__(self, transformations: List[Callable]) -> None:
        self._transformations = transformations or []

    def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        result = data
        for index, fn in enumerate(self._transformations):
            out = fn(result)
            if not isinstance(out, dict):
                raise TransformError(
                    f"File transformation #{index} must return a dict, got "
                    f"{type(out).__name__}"
                )
            result = out
        return result

    def get_transform_description(self) -> str:
        return f"Apply {len(self._transformations)} transformation(s)"


class _FileAggregator(ITransformFunction):
    """Per-record aggregation: reduce a record into a summary dict.

    Each entry of ``aggregations`` maps a result key to a callable
    ``record -> value`` (e.g. ``lambda r: sum(r["values"])``). The record is
    replaced by ``{name: fn(record) for name, fn in aggregations.items()}``.
    Aggregation here is per-record (a map producing a summary), not a
    cross-record reduce.
    """

    def __init__(self, aggregations: Dict[str, Callable]) -> None:
        self._aggregations = aggregations or {}

    def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        return {name: fn(data) for name, fn in self._aggregations.items()}

    def get_transform_description(self) -> str:
        return f"Apply {len(self._aggregations)} aggregation(s)"


def _make_filter(filters: List[Callable]) -> Callable[..., bool]:
    """Build the ``filter`` arc condition: a record passes iff all filters pass.

    The returned callable takes ``(data, context=None)`` so the engine invokes
    it with the raw record dict — a one-argument callable would be treated as
    expecting a wrapped state object. Each user filter is a ``record -> bool``
    predicate.
    """

    def filter_pass(data: Dict[str, Any], context: Any = None) -> bool:
        return all(predicate(data) for predicate in filters)

    return filter_pass


def _make_validator(schema: Dict[str, Any]) -> Callable[..., bool]:
    """Build the ``validate`` arc condition from a validation schema.

    A record is valid iff it satisfies every field constraint. Supported
    per-field constraints: ``required``, ``type`` (mapped to a Python type for
    an ``isinstance`` check), ``min`` / ``max`` (inclusive numeric bounds), and
    ``pattern`` (regex). A field whose constraint is the literal ``True`` is
    treated as simply required.
    """

    def validate_check(data: Dict[str, Any], context: Any = None) -> bool:
        for field_name, constraints in schema.items():
            if constraints is True:
                if field_name not in data:
                    return False
                continue
            if not isinstance(constraints, dict):
                continue
            if constraints.get("required") and field_name not in data:
                return False
            if "type" in constraints:
                expected = _VALIDATION_TYPE_MAP.get(constraints["type"])
                if expected is not None and not isinstance(
                    data.get(field_name), expected
                ):
                    return False
            if "min" in constraints and not (
                data.get(field_name, 0) >= constraints["min"]
            ):
                return False
            if "max" in constraints and not (
                data.get(field_name, 0) <= constraints["max"]
            ):
                return False
            if "pattern" in constraints and not re.match(
                constraints["pattern"], str(data.get(field_name, ""))
            ):
                return False
        return True

    return validate_check


class FileProcessor(StructuredConfigConsumer[FileProcessingConfig]):
    """File processor using FSM pattern.

    Constructed from :class:`FileProcessingConfig`: ``FileProcessor(cfg)`` or
    ``FileProcessor.from_config({...})`` (dict-dispatch). The config has a
    required ``input_path`` field, so an all-default ``FileProcessor()`` is
    not valid.
    """

    CONFIG_CLS: ClassVar[type[FileProcessingConfig]] = FileProcessingConfig

    # Resolved formats live on the processor, not the frozen config; set in
    # ``_detect_format`` (always called from ``_setup``).
    _format: FileFormat
    _output_format: FileFormat

    def _setup(self) -> None:
        self._detect_format()
        self._fsm = self._build_fsm()
        self._metrics = {
            'lines_read': 0,
            'records_processed': 0,
            'records_written': 0,
            'errors': 0,
            'skipped': 0
        }
        
    def _detect_format(self) -> None:
        """Resolve the effective input/output formats onto the processor.

        Auto-detects the input format from the file extension when the
        config leaves ``format`` unset, and mirrors it to the output format
        when ``output_format`` is unset. The resolved values are stored on
        the processor (``self._format`` / ``self._output_format``) rather
        than written back to the immutable config — the config keeps the
        caller-supplied values (often ``None``, meaning "auto-detect").
        """
        resolved_format = self.config.format
        if not resolved_format:
            path = Path(self.config.input_path)
            ext = path.suffix.lower()

            format_map = {
                '.json': FileFormat.JSON,
                '.jsonl': FileFormat.JSON,
                '.csv': FileFormat.CSV,
                '.tsv': FileFormat.CSV,
                '.xml': FileFormat.XML,
                '.parquet': FileFormat.PARQUET,
                '.txt': FileFormat.TEXT
            }

            resolved_format = format_map.get(ext, FileFormat.BINARY)

        self._format = resolved_format
        self._output_format = self.config.output_format or resolved_format

    @property
    def resolved_format(self) -> FileFormat:
        """The effective input format after auto-detection.

        Equals ``config.format`` when the caller set it; otherwise the
        format inferred from the input path's extension. Read-only — the
        resolution lives on the processor, not the (frozen) config.
        """
        return self._format

    @property
    def resolved_output_format(self) -> FileFormat:
        """The effective output format after resolution.

        Equals ``config.output_format`` when set, else mirrors
        :attr:`resolved_format`. Read-only.
        """
        return self._output_format

    def _build_fsm(self) -> AsyncSimpleFSM:
        """Build the FSM for file processing.

        Only the *enabled* stages (parse always; validate iff a schema; filter
        iff filters; transform iff transformations; aggregate iff aggregations;
        write always) are wired into a single connected chain to
        ``write -> complete``, so no stage is ever a dead-end (a passthrough
        config flows ``read -> parse -> write -> complete``). Per-record
        functions reach the FSM through the ``custom_functions=`` channel and
        are referenced from state ``functions`` blocks / arc conditions — the
        top-level ``config['functions']`` dict is silently dropped by
        ``FSMConfig`` (it has no ``functions`` field).
        """
        # Determine data mode based on processing mode
        if self.config.mode == ProcessingMode.STREAM:
            data_mode = DataHandlingMode.REFERENCE  # Use reference for streaming
        elif self.config.mode == ProcessingMode.BATCH:
            data_mode = DataHandlingMode.COPY  # Use copy for batch isolation
        else:
            data_mode = DataHandlingMode.DIRECT  # Use direct for whole file

        active = self._active_stages()
        fsm_config = {
            'name': 'File_Processor',
            'data_mode': data_mode.value,
            'states': self._build_states(active),
            'arcs': self._build_arcs(active),
        }

        return AsyncSimpleFSM(
            fsm_config,
            data_mode=data_mode,
            custom_functions=self._build_custom_functions(),
        )

    def _active_stages(self) -> List[str]:
        """Ordered list of enabled pipeline stages from ``parse`` to ``write``.

        ``parse`` and ``write`` are always present; the middle stages appear
        only when their config section is set. Consecutive entries are wired
        directly to one another in :meth:`_build_arcs`, so a disabled stage is
        simply absent from the chain (never a dead-end).
        """
        stages = ['parse']
        if self.config.validation_schema:
            stages.append('validate')
        if self.config.filters:
            stages.append('filter')
        if self.config.transformations:
            stages.append('transform')
        if self.config.aggregations:
            stages.append('aggregate')
        stages.append('write')
        return stages

    def _build_states(self, active: List[str]) -> List[Dict[str, Any]]:
        """Build the state list for the enabled stages.

        ``transform`` / ``aggregate`` carry a registered ``ITransformFunction``
        via a ``functions`` block. ``complete`` emits output; the ``filtered``
        and ``error`` terminals (present only when filtering / validation is
        enabled) set ``emit_output=False`` so the records they hold are kept out
        of the output in every mode.
        """
        states: List[Dict[str, Any]] = [{'name': 'read', 'is_start': True}]
        for stage in active:
            if stage == 'transform':
                states.append({
                    'name': 'transform',
                    'functions': {
                        'transform': {'type': 'registered', 'name': 'transform'}
                    },
                })
            elif stage == 'aggregate':
                states.append({
                    'name': 'aggregate',
                    'functions': {
                        'transform': {'type': 'registered', 'name': 'aggregate'}
                    },
                })
            else:
                states.append({'name': stage})
        states.append({'name': 'complete', 'is_end': True})
        if 'filter' in active:
            states.append(
                {'name': 'filtered', 'is_end': True, 'emit_output': False}
            )
        if 'validate' in active:
            states.append(
                {'name': 'error', 'is_end': True, 'emit_output': False}
            )
        return states
        
    def _build_arcs(self, active: List[str]) -> List[Dict[str, Any]]:
        """Connect the enabled stages into one chain to ``write -> complete``.

        Each enabled stage transitions to the next enabled stage (``write``
        falls through to ``complete``). ``validate`` and ``filter`` are gates:
        a registered condition routes passing records onward while the
        fall-through arc diverts the rest to ``error`` / ``filtered`` (both
        non-emitting terminals). ``transform`` / ``aggregate`` apply their
        registered state transform on entry, then take a plain forward arc.
        """
        arcs: List[Dict[str, Any]] = [
            {'from': 'read', 'to': active[0], 'name': 'read_line'}
        ]
        for index, stage in enumerate(active):
            nxt = active[index + 1] if index + 1 < len(active) else 'complete'
            if stage == 'validate':
                arcs.append({
                    'from': 'validate', 'to': nxt, 'name': 'valid',
                    'condition': {'type': 'registered', 'name': 'validate_check'},
                })
                arcs.append(
                    {'from': 'validate', 'to': 'error', 'name': 'invalid'}
                )
            elif stage == 'filter':
                arcs.append({
                    'from': 'filter', 'to': nxt, 'name': 'passed',
                    'condition': {'type': 'registered', 'name': 'filter_pass'},
                })
                arcs.append(
                    {'from': 'filter', 'to': 'filtered', 'name': 'filtered_out'}
                )
            else:
                arcs.append(
                    {'from': stage, 'to': nxt, 'name': f'{stage}_done'}
                )
        return arcs
    
    def _build_custom_functions(self) -> Dict[str, Any]:
        """Build the registered functions the FSM references by name.

        Only functions for configured stages are returned, so a state never
        references a missing function:

        - ``transform`` (:class:`_FileTransform`) applies the configured
          transformations in order; referenced by the ``transform`` state.
        - ``aggregate`` (:class:`_FileAggregator`) reduces each record to a
          summary dict; referenced by the ``aggregate`` state.
        - ``filter_pass`` is the ``filter`` arc condition (all filters pass).
        - ``validate_check`` is the ``validate`` arc condition (schema holds).

        Routed through ``AsyncSimpleFSM(config, custom_functions=...)`` and
        referenced from each state's ``functions`` block / arc condition.
        """
        functions: Dict[str, Any] = {}
        if self.config.transformations:
            functions['transform'] = _FileTransform(self.config.transformations)
        if self.config.aggregations:
            functions['aggregate'] = _FileAggregator(self.config.aggregations)
        if self.config.filters:
            functions['filter_pass'] = _make_filter(self.config.filters)
        if self.config.validation_schema:
            functions['validate_check'] = _make_validator(
                self.config.validation_schema
            )
        return functions
        
    async def process(self) -> Dict[str, Any]:
        """Process the file.

        Returns:
            Processing metrics

        Raises:
            NotImplementedError: If ``compression`` is configured. No
                execution path currently writes compressed output (the former
                stream-mode ``FileStreamSink`` path was removed when the
                pattern moved onto the async engine), so the option is
                rejected loudly rather than silently emitting uncompressed
                output the caller believes is compressed.
        """
        if self.config.compression:
            raise NotImplementedError(
                "FileProcessor does not support compressed output "
                f"(compression={self.config.compression!r}). Write uncompressed "
                "output and compress it separately, or omit the 'compression' "
                "config field."
            )
        if self.config.mode == ProcessingMode.STREAM:
            return await self._process_stream()
        elif self.config.mode == ProcessingMode.BATCH:
            return await self._process_batch()
        else:
            return await self._process_whole()
            
    async def _process_stream(self) -> Dict[str, Any]:
        """Process file as stream.

        Awaits the async FSM's streaming executor directly (no sync-bridge
        loop blocking). The executor opens the input/output paths itself and
        auto-detects the format from the extension, so the input/output
        paths are passed as strings rather than wrapped stream objects.
        """
        result = await self._fsm.process_stream(
            source=self.config.input_path,
            sink=self.config.output_path,
            chunk_size=self.config.chunk_size,
            input_format='auto',
        )

        self._metrics.update(result)
        return self._metrics
        
    async def _process_batch(self) -> Dict[str, Any]:
        """Process the file in batches.

        Each record is driven through the FSM; records that reach ``complete``
        are written to the output (once, after all batches — the writer
        truncates), records that reach ``filtered`` are counted as skipped, and
        records that reach ``error`` or fail a transform are counted as errors.
        """
        batches = [batch async for batch in self._read_batches()]

        emitted: List[Dict[str, Any]] = []
        for batch in batches:
            results = await self._fsm.process_batch(
                data=batch,  # type: ignore
                batch_size=self.config.chunk_size,
                max_workers=self.config.parallel_chunks,
            )
            for result in results:
                self._account_result(result, emitted)

        if self.config.output_path and emitted:
            await self._write_output(emitted)

        return self._metrics

    def _account_result(
        self, result: Dict[str, Any], emitted: List[Dict[str, Any]]
    ) -> None:
        """Update metrics for one FSM result and collect emitted records.

        A record reaching ``complete`` is written; one reaching ``filtered`` is
        skipped; one reaching ``error`` (validation reject) or failing a
        transform is an error. ``emitted`` accumulates the results to write so
        the output is produced in a single pass (the writer truncates).
        """
        if not result['success']:
            self._metrics['errors'] += 1
            return
        final_state = result.get('final_state')
        if final_state == 'complete':
            self._metrics['records_processed'] += 1
            self._metrics['records_written'] += 1
            emitted.append(result)
        elif final_state == 'filtered':
            self._metrics['records_processed'] += 1
            self._metrics['skipped'] += 1
        elif final_state == 'error':
            self._metrics['errors'] += 1
        else:
            # Reached some other terminal cleanly; count it as processed.
            self._metrics['records_processed'] += 1
        
    async def _process_whole(self) -> Dict[str, Any]:
        """Process entire file at once."""
        # Read entire file off the event loop (blocking whole-file read).
        content = await asyncio.to_thread(self._read_whole)

        # Parse content
        if self._format == FileFormat.JSON:
            import json
            data = json.loads(content)
        elif self._format == FileFormat.CSV:
            import csv
            from io import StringIO
            reader = csv.DictReader(StringIO(content))
            data = list(reader)
        else:
            data = {'content': content}
            
        # Process data
        if isinstance(data, list):
            results = await self._fsm.process_batch(data)
        else:
            results = [await self._fsm.process(data)]
            
        # Only records that reached `complete` are part of the output;
        # filtered/invalid records are excluded (consistent with batch/stream).
        emitted = [
            r for r in results
            if r['success'] and r.get('final_state') == 'complete'
        ]
        if self.config.output_path and emitted:
            await self._write_output(emitted)

        self._metrics['records_processed'] = len(results)
        self._metrics['records_written'] = len(emitted)
        self._metrics['skipped'] = sum(
            1 for r in results
            if r['success'] and r.get('final_state') == 'filtered'
        )
        self._metrics['errors'] = sum(
            1 for r in results
            if not r['success'] or r.get('final_state') == 'error'
        )

        return self._metrics

    def _read_whole(self) -> str:
        """Synchronous whole-file read — run via ``to_thread``."""
        with open(self.config.input_path, encoding=self.config.encoding) as f:
            return f.read()

    async def _read_batches(self) -> AsyncIterator[List[Dict[str, Any]]]:
        """Read file in batches.

        The blocking ``open`` + line iteration and batch assembly run on a
        worker thread via :func:`~dataknobs_common.aiter_sync_in_thread`;
        batches cross a bounded queue so streaming stays lazy and the loop
        is never stalled.
        """
        async for batch in aiter_sync_in_thread(self._read_batches_sync):
            yield batch

    def _read_batches_sync(self) -> Iterator[List[Dict[str, Any]]]:
        """Synchronous batch reader — driven on a worker thread."""
        batch: List[Dict[str, Any]] = []
        with open(self.config.input_path, encoding=self.config.encoding) as f:
            for line in f:
                self._metrics['lines_read'] += 1

                # Parse line based on format
                if self._format == FileFormat.JSON:
                    import json
                    try:
                        record = json.loads(line)
                        batch.append(record)
                    except json.JSONDecodeError:
                        self._metrics['errors'] += 1
                        continue
                else:
                    batch.append({'line': line.strip()})

                if len(batch) >= self.config.chunk_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    async def _write_output(self, results: List[Dict[str, Any]]) -> None:
        """Write processed results to output file.

        The whole write runs on a worker thread via
        :func:`asyncio.to_thread` so the blocking ``open`` + write never
        stalls the event loop.
        """
        output_data = [r['data'] for r in results if r['success']]
        await asyncio.to_thread(self._write_output_sync, output_data)

    def _write_output_sync(self, output_data: List[Dict[str, Any]]) -> None:
        """Synchronous output write — run via ``to_thread``."""
        with open(self.config.output_path, 'w', encoding=self.config.encoding) as f:  # type: ignore
            if self._output_format == FileFormat.JSON:
                import json
                json.dump(output_data, f, indent=2)
            elif self._output_format == FileFormat.CSV:
                import csv
                if output_data:
                    writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
                    writer.writeheader()
                    writer.writerows(output_data)
            else:
                for item in output_data:
                    f.write(str(item) + '\n')


# Factory functions for common file processing patterns

def create_csv_processor(
    input_file: str,
    output_file: str | None = None,
    transformations: List[Callable] | None = None,
    filters: List[Callable] | None = None
) -> FileProcessor:
    """Create CSV file processor.
    
    Args:
        input_file: Input CSV file path
        output_file: Optional output file path
        transformations: Data transformations
        filters: Row filters
        
    Returns:
        Configured FileProcessor
    """
    config = FileProcessingConfig(
        input_path=input_file,
        output_path=output_file,
        format=FileFormat.CSV,
        mode=ProcessingMode.STREAM,
        transformations=transformations,
        filters=filters
    )
    
    return FileProcessor(config)


def create_json_stream_processor(
    input_file: str,
    output_file: str | None = None,
    validation_schema: Dict[str, Any] | None = None,
    chunk_size: int = 1000
) -> FileProcessor:
    """Create JSON lines stream processor.
    
    Args:
        input_file: Input JSONL file path
        output_file: Optional output file path
        validation_schema: JSON schema for validation
        chunk_size: Processing chunk size
        
    Returns:
        Configured FileProcessor
    """
    config = FileProcessingConfig(
        input_path=input_file,
        output_path=output_file,
        format=FileFormat.JSON,
        mode=ProcessingMode.STREAM,
        chunk_size=chunk_size,
        validation_schema=validation_schema
    )
    
    return FileProcessor(config)


def create_log_analyzer(
    log_file: str,
    output_file: str | None = None,
    patterns: List[str] | None = None,
    aggregations: Dict[str, Callable] | None = None
) -> FileProcessor:
    """Create log file analyzer.
    
    Args:
        log_file: Log file path
        output_file: Optional analysis output
        patterns: Regex patterns to extract
        aggregations: Aggregation functions
        
    Returns:
        Configured FileProcessor
    """
    # Create pattern extractors
    transformations = []
    if patterns:
        def extract_patterns(data):
            result = data.copy()
            for pattern in patterns:
                match = re.search(pattern, data.get('line', ''))
                if match:
                    result.update(match.groupdict())
            return result
        transformations.append(extract_patterns)
        
    config = FileProcessingConfig(
        input_path=log_file,
        output_path=output_file,
        format=FileFormat.TEXT,
        mode=ProcessingMode.STREAM,
        transformations=transformations,
        aggregations=aggregations
    )
    
    return FileProcessor(config)


def create_file_processor(
    input_path: str,
    output_path: str,
    pattern: str = "*",
    mode: ProcessingMode = ProcessingMode.WHOLE,
    transformations: List[Callable] | None = None
) -> FileProcessor:
    """Create generic file processor.
    
    Args:
        input_path: Input directory or file
        output_path: Output directory or file  
        pattern: File pattern to match (currently unused)
        mode: Processing mode
        transformations: Data transformation functions
        
    Returns:
        Configured FileProcessor
    """
    # Note: pattern parameter is currently not used in FileProcessingConfig
    config = FileProcessingConfig(
        input_path=input_path,
        output_path=output_path,
        format=FileFormat.TEXT,
        mode=mode,
        transformations=transformations or []
    )
    
    return FileProcessor(config)


def create_json_processor(
    input_path: str,
    output_path: str,
    pretty_print: bool = False,
    array_processing: bool = False
) -> FileProcessor:
    """Create JSON file processor.
    
    Args:
        input_path: Input directory
        output_path: Output directory
        pretty_print: Whether to pretty print JSON
        array_processing: Process as JSON arrays
        
    Returns:
        Configured FileProcessor  
    """
    config = FileProcessingConfig(
        input_path=input_path,
        output_path=output_path,
        format=FileFormat.JSON,
        mode=ProcessingMode.WHOLE,
        json_config={
            'pretty_print': pretty_print,
            'array_processing': array_processing
        }
    )
    
    return FileProcessor(config)


def create_log_processor(
    input_path: str,
    output_path: str,
    parse_timestamps: bool = False,
    extract_errors: bool = False
) -> FileProcessor:
    """Create log file processor.
    
    Args:
        input_path: Input directory
        output_path: Output directory
        pattern: Log file pattern
        parse_timestamps: Whether to parse timestamps
        extract_errors: Whether to extract error entries
        
    Returns:
        Configured FileProcessor
    """
    config = FileProcessingConfig(
        input_path=input_path,
        output_path=output_path,
        format=FileFormat.TEXT,
        mode=ProcessingMode.STREAM,
        log_config={
            'parse_timestamps': parse_timestamps,
            'extract_errors': extract_errors
        }
    )
    
    return FileProcessor(config)


def create_batch_file_processor(
    input_paths: List[str],
    output_path: str,
    patterns: List[str],
    batch_size: int = 10
) -> FileProcessor:
    """Create batch file processor.
    
    Args:
        input_paths: List of input directories
        output_path: Output directory
        patterns: File patterns to match
        batch_size: Batch processing size
        
    Returns:
        Configured FileProcessor
    """
    # Use first input path for config, handle multiple paths in processor.
    # ``patterns`` is currently unused — ``FileProcessingConfig`` has no
    # pattern field (mirrors ``create_file_processor``'s unused ``pattern``).
    config = FileProcessingConfig(
        input_path=input_paths[0] if input_paths else "",
        output_path=output_path,
        format=FileFormat.TEXT,
        mode=ProcessingMode.BATCH,
        batch_size=batch_size
    )
    
    return FileProcessor(config)

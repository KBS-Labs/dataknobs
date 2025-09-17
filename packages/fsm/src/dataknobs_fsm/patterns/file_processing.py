"""File processing pattern implementation.

This module provides pre-configured FSM patterns for processing files,
including CSV, JSON, XML, and other formats with streaming support.
"""

from typing import Any, Dict, List, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from dataknobs_data import Record

from ..api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from ..streaming.file_stream import FileStreamSource, FileStreamSink
from ..functions.library.validators import SchemaValidator


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


@dataclass
class FileProcessingConfig:
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


class FileProcessor:
    """File processor using FSM pattern."""
    
    def __init__(self, config: FileProcessingConfig):
        """Initialize file processor.
        
        Args:
            config: File processing configuration
        """
        self.config = config
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
        """Auto-detect file format if not specified."""
        if not self.config.format:
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
            
            self.config.format = format_map.get(ext, FileFormat.BINARY)
            
        # Set output format if not specified
        if not self.config.output_format:
            self.config.output_format = self.config.format
            
    def _build_fsm(self) -> SimpleFSM:
        """Build FSM for file processing."""
        # Determine data mode based on processing mode
        if self.config.mode == ProcessingMode.STREAM:
            data_mode = DataHandlingMode.REFERENCE  # Use reference for streaming
        elif self.config.mode == ProcessingMode.BATCH:
            data_mode = DataHandlingMode.COPY  # Use copy for batch isolation
        else:
            data_mode = DataHandlingMode.DIRECT  # Use direct for whole file
            
        # Create FSM configuration
        fsm_config = {
            'name': 'File_Processor',
            'data_mode': data_mode.value,
            'states': [
                {
                    'name': 'read',
                    'is_start': True
                },
                {
                    'name': 'parse',
                },
                {
                    'name': 'validate',
                },
                {
                    'name': 'filter',
                },
                {
                    'name': 'transform',
                },
                {
                    'name': 'aggregate',
                },
                {
                    'name': 'write',
                },
                {
                    'name': 'complete',
                    'is_end': True
                },
                {
                    'name': 'error',
                    'is_end': True
                }
            ],
            'arcs': self._build_arcs(),
            'functions': self._build_functions()
        }
        
        return SimpleFSM(fsm_config, data_mode=data_mode)
        
    def _build_arcs(self) -> List[Dict[str, Any]]:
        """Build FSM arcs based on configuration."""
        arcs = [
            {
                'from': 'read',
                'to': 'parse',
                'name': 'read_line'
            },
            {
                'from': 'parse',
                'to': 'validate' if self.config.validation_schema else 'filter',
                'name': 'parsed',
                'transform': {'type': 'inline', 'code': self._get_parser_code()}
            }
        ]
        
        # Add validation arc if schema provided
        if self.config.validation_schema:
            arcs.extend([
                {
                    'from': 'validate',
                    'to': 'filter' if self.config.filters else 'transform',
                    'name': 'valid',
                    'condition': {'type': 'inline', 'code': self._get_validator_code()}
                },
                {
                    'from': 'validate',
                    'to': 'error',
                    'name': 'invalid'
                }
            ])
            
        # Add filter arc if filters provided
        if self.config.filters:
            arcs.extend([
                {
                    'from': 'filter',
                    'to': 'transform' if self.config.transformations else 'aggregate',
                    'name': 'passed',
                    'condition': {'type': 'inline', 'code': self._get_filter_code()}
                },
                {
                    'from': 'filter',
                    'to': 'complete',
                    'name': 'filtered_out'
                }
            ])
            
        # Add transformation arc if transformations provided
        if self.config.transformations:
            next_state = 'aggregate' if self.config.aggregations else 'write'
            arcs.append({
                'from': 'transform',
                'to': next_state,
                'name': 'transformed',
                'transform': {'type': 'inline', 'code': self._get_transformer_code()}
            })
            
        # Add aggregation arc if aggregations provided
        if self.config.aggregations:
            arcs.append({
                'from': 'aggregate',
                'to': 'write',
                'name': 'aggregated',
                'transform': {'type': 'inline', 'code': self._get_aggregator_code()}
            })
            
        # Add write arc
        arcs.extend([
            {
                'from': 'write',
                'to': 'complete',
                'name': 'written'
            }
        ])
        
        return arcs  # type: ignore
    
    def _build_functions(self) -> Dict[str, Any]:
        """Build functions registry for FSM."""
        functions = {}
        
        # Register filter functions
        if self.config.filters:
            for i, filter_func in enumerate(self.config.filters):
                functions[f'filter_{i}'] = filter_func
        
        # Register transformation functions  
        if self.config.transformations:
            for i, transform_func in enumerate(self.config.transformations):
                functions[f'transform_{i}'] = transform_func
        
        # Register aggregation functions
        if self.config.aggregations:
            for agg_name, agg_func in self.config.aggregations.items():
                functions[f'agg_{agg_name}'] = agg_func
        
        # Register parser function
        functions['parser'] = self._create_parser()
        
        return functions
        
    def _get_parser_code(self) -> str:
        """Get parser code for file format."""
        if self.config.format == FileFormat.JSON:
            return "import json; json.loads(data) if isinstance(data, str) else data"
        elif self.config.format == FileFormat.CSV:
            return """
import csv
from io import StringIO
if isinstance(data, str):
    reader = csv.DictReader(StringIO(data))
    rows = list(reader)
    data = rows[0] if rows else {}
data
"""
        elif self.config.format == FileFormat.XML:
            return """
import xml.etree.ElementTree as ET
if isinstance(data, str):
    root = ET.fromstring(data)
    data = {child.tag: child.text for child in root}
data
"""
        else:
            return "data"
    
    def _get_validator_code(self) -> str:
        """Get validator code."""
        if not self.config.validation_schema:
            return "True"
        
        # Generate validation code based on schema
        validations = []
        for field_name, constraints in self.config.validation_schema.items():
            if isinstance(constraints, dict):
                if constraints.get('required'):
                    validations.append(f"'{field_name}' in data")
                if 'type' in constraints:
                    python_type = constraints['type']
                    validations.append(f"isinstance(data.get('{field_name}'), {python_type})")
                if 'min' in constraints:
                    validations.append(f"data.get('{field_name}', 0) >= {constraints['min']}")
                if 'max' in constraints:
                    validations.append(f"data.get('{field_name}', 0) <= {constraints['max']}")
                if 'pattern' in constraints:
                    validations.append(f"re.match(r'{constraints['pattern']}', str(data.get('{field_name}', '')))")
            elif constraints is True:  # Required field
                validations.append(f"'{field_name}' in data")
        
        return " and ".join(validations) if validations else "True"
    
    def _get_filter_code(self) -> str:
        """Get filter code."""
        if not self.config.filters:
            return "True"
        
        # Apply all filters in sequence - data must pass all filters
        filter_conditions = []
        for i, _filter_func in enumerate(self.config.filters):
            # Use registered function name
            filter_conditions.append(f"filter_{i}(data)")
        
        return " and ".join(filter_conditions) if filter_conditions else "True"
    
    def _get_transformer_code(self) -> str:
        """Get transformer code."""
        if not self.config.transformations:
            return "data"
        
        # Apply transformations in sequence
        result_code = "data"
        for i, _transform_func in enumerate(self.config.transformations):
            # Each transformation receives the result of the previous one
            result_code = f"transform_{i}({result_code})"
        
        return result_code
    
    def _get_aggregator_code(self) -> str:
        """Get aggregator code."""
        if not self.config.aggregations:
            return "data"
        
        # Apply aggregations to create summary data
        aggregation_results = []
        for agg_name in self.config.aggregations.keys():
            aggregation_results.append(f"'{agg_name}': agg_{agg_name}(data)")
        
        if aggregation_results:
            return "{" + ", ".join(aggregation_results) + "}"
        else:
            return "data"
    
    def _create_parser(self) -> Callable:
        """Create parser for file format."""
        if self.config.format == FileFormat.JSON:
            import json
            return lambda data: json.loads(data) if isinstance(data, str) else data
        elif self.config.format == FileFormat.CSV:
            import csv
            from io import StringIO
            
            def parse_csv(data):
                if isinstance(data, str):
                    reader = csv.DictReader(StringIO(data))
                    return next(iter(reader)) if reader else {}
                return data
            return parse_csv
        elif self.config.format == FileFormat.XML:
            import xml.etree.ElementTree as ET
            
            def parse_xml(data):
                if isinstance(data, str):
                    root = ET.fromstring(data)
                    return {child.tag: child.text for child in root}
                return data
            return parse_xml
        else:
            return lambda data: data
            
    def _create_validator(self) -> Callable | None:
        """Create validator function."""
        if not self.config.validation_schema:
            return None
            
        validator = SchemaValidator(self.config.validation_schema)
        return lambda state: validator.validate(Record(state.data))  # type: ignore
        
    def _create_filter(self) -> Callable | None:
        """Create filter function."""
        if not self.config.filters:
            return None
            
        def apply_filters(state):
            return all(filter_func(state.data) for filter_func in self.config.filters)
            
        return apply_filters
        
    def _create_transformer(self) -> Callable | None:
        """Create transformation function."""
        if not self.config.transformations:
            return None
            
        async def transform(data: Dict[str, Any]) -> Dict[str, Any]:
            result = data
            for transformer in self.config.transformations:
                if hasattr(transformer, 'transform'):
                    result = await transformer.transform(result)
                elif callable(transformer):
                    result = transformer(result)
            return result
            
        return transform
        
    def _create_aggregator(self) -> Callable | None:
        """Create aggregation function."""
        if not self.config.aggregations:
            return None
            
        # Store aggregation state
        agg_state = {key: [] for key in self.config.aggregations}
        
        def aggregate(data: Dict[str, Any]) -> Dict[str, Any]:
            # Accumulate values
            for key in self.config.aggregations:
                if key in data:
                    agg_state[key].append(data[key])
                    
            # Return aggregated results
            return {
                key: self.config.aggregations[key](values)  # type: ignore
                for key, values in agg_state.items()
            }
            
        return aggregate
        
    async def process(self) -> Dict[str, Any]:
        """Process the file.
        
        Returns:
            Processing metrics
        """
        if self.config.mode == ProcessingMode.STREAM:
            return await self._process_stream()
        elif self.config.mode == ProcessingMode.BATCH:
            return await self._process_batch()
        else:
            return await self._process_whole()
            
    async def _process_stream(self) -> Dict[str, Any]:
        """Process file as stream."""
        # Create stream source
        source = FileStreamSource(
            self.config.input_path,
            chunk_size=self.config.chunk_size,
            encoding=self.config.encoding
        )
        
        # Create stream sink if output specified
        sink = None
        if self.config.output_path:
            sink = FileStreamSink(
                self.config.output_path,
                encoding=self.config.encoding,
                compression=self.config.compression
            )
            
        # Process stream
        result = self._fsm.process_stream(
            source=source,
            sink=sink,
            chunk_size=self.config.chunk_size,
            on_progress=self._update_progress
        )
        
        self._metrics.update(result)
        return self._metrics
        
    async def _process_batch(self) -> Dict[str, Any]:
        """Process file in batches."""
        # Read file in batches
        batches = []
        async for batch in self._read_batches():
            batches.append(batch)
            
        # Process batches
        for batch in batches:
            results = self._fsm.process_batch(
                data=batch,  # type: ignore
                batch_size=self.config.chunk_size,
                max_workers=self.config.parallel_chunks
            )
            
            # Update metrics
            for result in results:
                if result['success']:
                    self._metrics['records_processed'] += 1
                    if result['final_state'] == 'complete':
                        self._metrics['records_written'] += 1
                else:
                    self._metrics['errors'] += 1
                    
        return self._metrics
        
    async def _process_whole(self) -> Dict[str, Any]:
        """Process entire file at once."""
        # Read entire file
        with open(self.config.input_path, encoding=self.config.encoding) as f:
            content = f.read()
            
        # Parse content
        if self.config.format == FileFormat.JSON:
            import json
            data = json.loads(content)
        elif self.config.format == FileFormat.CSV:
            import csv
            from io import StringIO
            reader = csv.DictReader(StringIO(content))
            data = list(reader)
        else:
            data = {'content': content}
            
        # Process data
        if isinstance(data, list):
            results = self._fsm.process_batch(data)
        else:
            results = [self._fsm.process(data)]
            
        # Write output if specified
        if self.config.output_path and results:
            await self._write_output(results)
            
        self._metrics['records_processed'] = len(results)
        self._metrics['records_written'] = sum(
            1 for r in results if r['success']
        )
        
        return self._metrics
        
    async def _read_batches(self) -> AsyncIterator[List[Dict[str, Any]]]:
        """Read file in batches."""
        batch = []
        with open(self.config.input_path, encoding=self.config.encoding) as f:
            for line in f:
                self._metrics['lines_read'] += 1
                
                # Parse line based on format
                if self.config.format == FileFormat.JSON:
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
        """Write processed results to output file."""
        output_data = [r['data'] for r in results if r['success']]
        
        with open(self.config.output_path, 'w', encoding=self.config.encoding) as f:  # type: ignore
            if self.config.output_format == FileFormat.JSON:
                import json
                json.dump(output_data, f, indent=2)
            elif self.config.output_format == FileFormat.CSV:
                import csv
                if output_data:
                    writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
                    writer.writeheader()
                    writer.writerows(output_data)
            else:
                for item in output_data:
                    f.write(str(item) + '\n')
                    
    def _update_progress(self, progress: Dict[str, Any]) -> None:
        """Update progress metrics."""
        self._metrics.update(progress)


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
    # Use first input path for config, handle multiple paths in processor
    config = FileProcessingConfig(  # type: ignore
        input_path=input_paths[0] if input_paths else "",
        output_path=output_path,
        pattern=patterns[0] if patterns else "*",
        format=FileFormat.TEXT,
        mode=ProcessingMode.BATCH,
        batch_size=batch_size
    )
    
    return FileProcessor(config)

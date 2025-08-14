"""Data transformation utilities for migrations."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from dataknobs_data.fields import Field
from dataknobs_data.records import Record

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Mapping between source and target fields."""
    source_field: str
    target_field: str
    transformer: Optional[Callable[[Any], Any]] = None
    default_value: Any = None
    
    def apply(self, source_value: Any) -> Any:
        """Apply transformation to field value."""
        if source_value is None and self.default_value is not None:
            return self.default_value
        
        if self.transformer:
            try:
                return self.transformer(source_value)
            except Exception as e:
                logger.warning(f"Transformation failed for {self.source_field}: {e}")
                return self.default_value
        
        return source_value


class ValueTransformer:
    """Common value transformation functions."""
    
    @staticmethod
    def to_string(value: Any) -> str:
        """Convert any value to string."""
        if value is None:
            return ""
        return str(value)
    
    @staticmethod
    def to_int(value: Any) -> Optional[int]:
        """Convert value to integer."""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                # Handle string representations
                value = value.strip()
                if value == "":
                    return None
            return int(float(value))  # Handle floats
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def to_float(value: Any) -> Optional[float]:
        """Convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def to_bool(value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value)
    
    @staticmethod
    def parse_json(value: str) -> Any:
        """Parse JSON string."""
        import json
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    @staticmethod
    def to_json(value: Any) -> str:
        """Convert value to JSON string."""
        import json
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)
    
    @staticmethod
    def normalize_string(value: str) -> str:
        """Normalize string (lowercase, strip whitespace)."""
        if not isinstance(value, str):
            value = str(value)
        return value.lower().strip()
    
    @staticmethod
    def truncate(max_length: int) -> Callable[[str], str]:
        """Create a truncation transformer."""
        def truncator(value: str) -> str:
            if not isinstance(value, str):
                value = str(value)
            return value[:max_length]
        return truncator
    
    @staticmethod
    def regex_extract(pattern: str, group: int = 0) -> Callable[[str], Optional[str]]:
        """Create a regex extraction transformer."""
        import re
        compiled_pattern = re.compile(pattern)
        
        def extractor(value: str) -> Optional[str]:
            if not isinstance(value, str):
                value = str(value)
            match = compiled_pattern.search(value)
            if match:
                return match.group(group)
            return None
        
        return extractor
    
    @staticmethod
    def map_values(mapping: Dict[Any, Any], default: Any = None) -> Callable[[Any], Any]:
        """Create a value mapping transformer."""
        def mapper(value: Any) -> Any:
            return mapping.get(value, default)
        return mapper
    
    @staticmethod
    def chain(*transformers: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Chain multiple transformers."""
        def chained(value: Any) -> Any:
            for transformer in transformers:
                value = transformer(value)
            return value
        return chained


class DataTransformer:
    """Transform records during migration."""
    
    def __init__(self):
        """Initialize data transformer."""
        self.field_mappings: List[FieldMapping] = []
        self.record_filters: List[Callable[[Record], bool]] = []
        self.record_transformers: List[Callable[[Record], Record]] = []
        self.field_filters: List[str] = []  # Fields to exclude
    
    def add_field_mapping(
        self,
        source_field: str,
        target_field: Optional[str] = None,
        transformer: Optional[Callable[[Any], Any]] = None,
        default_value: Any = None
    ) -> "DataTransformer":
        """Add a field mapping.
        
        Args:
            source_field: Source field name
            target_field: Target field name (defaults to source_field)
            transformer: Optional value transformer
            default_value: Default value if source is None
            
        Returns:
            Self for chaining
        """
        self.field_mappings.append(FieldMapping(
            source_field=source_field,
            target_field=target_field or source_field,
            transformer=transformer,
            default_value=default_value
        ))
        return self
    
    def rename_field(self, old_name: str, new_name: str) -> "DataTransformer":
        """Rename a field.
        
        Args:
            old_name: Current field name
            new_name: New field name
            
        Returns:
            Self for chaining
        """
        return self.add_field_mapping(old_name, new_name)
    
    def exclude_fields(self, *field_names: str) -> "DataTransformer":
        """Exclude fields from transformation.
        
        Args:
            *field_names: Field names to exclude
            
        Returns:
            Self for chaining
        """
        self.field_filters.extend(field_names)
        return self
    
    def add_record_filter(self, filter_func: Callable[[Record], bool]) -> "DataTransformer":
        """Add a record filter.
        
        Records that don't pass the filter will be skipped.
        
        Args:
            filter_func: Function that returns True to keep the record
            
        Returns:
            Self for chaining
        """
        self.record_filters.append(filter_func)
        return self
    
    def add_record_transformer(self, transformer: Callable[[Record], Record]) -> "DataTransformer":
        """Add a record-level transformer.
        
        Args:
            transformer: Function that transforms the entire record
            
        Returns:
            Self for chaining
        """
        self.record_transformers.append(transformer)
        return self
    
    def transform(self, record: Record) -> Optional[Record]:
        """Transform a record.
        
        Args:
            record: Source record
            
        Returns:
            Transformed record or None if filtered out
        """
        # Apply record filters
        for filter_func in self.record_filters:
            if not filter_func(record):
                return None
        
        # Create new record
        new_record = Record()
        
        # Apply field mappings
        if self.field_mappings:
            for mapping in self.field_mappings:
                if mapping.source_field in record.fields:
                    source_field = record.fields[mapping.source_field]
                    value = mapping.apply(source_field.value)
                    
                    new_record.fields[mapping.target_field] = Field(
                        name=mapping.target_field,
                        value=value,
                        type=type(value).__name__ if value is not None else 'str',
                        metadata=source_field.metadata.copy() if source_field.metadata else {}
                    )
                elif mapping.default_value is not None:
                    new_record.fields[mapping.target_field] = Field(
                        name=mapping.target_field,
                        value=mapping.default_value,
                        type=type(mapping.default_value).__name__
                    )
        else:
            # No explicit mappings, copy all fields
            for field_name, field in record.fields.items():
                if field_name not in self.field_filters:
                    new_record.fields[field_name] = field.copy()
        
        # Exclude filtered fields
        for field_name in self.field_filters:
            if field_name in new_record.fields:
                del new_record.fields[field_name]
        
        # Copy metadata
        new_record.metadata = record.metadata.copy() if record.metadata else {}
        
        # Apply record transformers
        for transformer in self.record_transformers:
            new_record = transformer(new_record)
            if new_record is None:
                return None
        
        return new_record


class TransformationPipeline:
    """Chain multiple data transformers."""
    
    def __init__(self, *transformers: Union[DataTransformer, Callable[[Record], Optional[Record]]]):
        """Initialize transformation pipeline.
        
        Args:
            *transformers: Transformers to chain
        """
        self.transformers = list(transformers)
    
    def add(self, transformer: Union[DataTransformer, Callable[[Record], Optional[Record]]]) -> "TransformationPipeline":
        """Add a transformer to the pipeline.
        
        Args:
            transformer: Transformer to add
            
        Returns:
            Self for chaining
        """
        self.transformers.append(transformer)
        return self
    
    def transform(self, record: Record) -> Optional[Record]:
        """Apply all transformations in sequence.
        
        Args:
            record: Source record
            
        Returns:
            Transformed record or None if filtered out
        """
        current = record
        
        for transformer in self.transformers:
            if isinstance(transformer, DataTransformer):
                current = transformer.transform(current)
            else:
                current = transformer(current)
            
            if current is None:
                return None
        
        return current
    
    def __call__(self, record: Record) -> Optional[Record]:
        """Make pipeline callable.
        
        Args:
            record: Source record
            
        Returns:
            Transformed record or None if filtered out
        """
        return self.transform(record)
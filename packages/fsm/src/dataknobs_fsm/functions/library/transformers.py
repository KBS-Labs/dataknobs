"""Built-in transformer functions for FSM.

This module provides commonly used transformation functions that can be
referenced in FSM configurations.
"""

import copy
import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Union

from dataknobs_fsm.functions.base import ITransformFunction, TransformError


class FieldMapper(ITransformFunction):
    """Map fields from source to target names."""

    def __init__(
        self,
        field_map: Dict[str, str],
        drop_unmapped: bool = False,
        copy_unmapped: bool = True,
    ):
        """Initialize the field mapper.
        
        Args:
            field_map: Dictionary mapping source field names to target names.
            drop_unmapped: If True, drop fields not in the mapping.
            copy_unmapped: If True, copy unmapped fields as-is.
        """
        self.field_map = field_map
        self.drop_unmapped = drop_unmapped
        self.copy_unmapped = copy_unmapped

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by mapping field names.
        
        Args:
            data: Input data.
            
        Returns:
            Transformed data with mapped field names.
        """
        result = {}
        
        # Map specified fields
        for source, target in self.field_map.items():
            if source in data:
                # Handle nested field paths
                if "." in source:
                    value = self._get_nested(data, source)
                else:
                    value = data[source]
                
                if "." in target:
                    self._set_nested(result, target, value)
                else:
                    result[target] = value
        
        # Handle unmapped fields
        if not self.drop_unmapped and self.copy_unmapped:
            for key, value in data.items():
                if key not in self.field_map and key not in result:
                    result[key] = value
        
        return result

    def _get_nested(self, data: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = path.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    def _set_nested(self, data: Dict, path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        parts = path.split(".")
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        mappings = list(self.field_map.items())
        mapping_str = ", ".join(f"{s}->{t}" for s, t in mappings[:3])
        if len(mappings) > 3:
            mapping_str += "..."
        return f"Map fields: {mapping_str}"


class ValueNormalizer(ITransformFunction):
    """Normalize values in data fields."""

    def __init__(
        self,
        normalizations: Dict[str, str],
        fields: List[str] | None = None,
    ):
        """Initialize the value normalizer.
        
        Args:
            normalizations: Dictionary of normalization types:
                - "lowercase": Convert to lowercase
                - "uppercase": Convert to uppercase
                - "trim": Remove leading/trailing whitespace
                - "snake_case": Convert to snake_case
                - "camel_case": Convert to camelCase
                - "pascal_case": Convert to PascalCase
                - "remove_special": Remove special characters
                - "normalize_spaces": Replace multiple spaces with single space
            fields: List of fields to normalize. If None, apply to all string fields.
        """
        self.normalizations = normalizations
        self.fields = fields

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by normalizing values.
        
        Args:
            data: Input data.
            
        Returns:
            Transformed data with normalized values.
        """
        result = copy.deepcopy(data)
        
        # Determine which fields to process
        fields_to_process = self.fields if self.fields else list(result.keys())
        
        for field in fields_to_process:
            if field not in result:
                continue
            
            value = result[field]
            if not isinstance(value, str):
                continue
            
            # Apply normalizations for this field
            field_normalizations = self.normalizations.get(
                field, self.normalizations.get("*", [])
            )
            
            if isinstance(field_normalizations, str):
                field_normalizations = [field_normalizations]
            
            for normalization in field_normalizations:
                value = self._apply_normalization(value, normalization)
            
            result[field] = value
        
        return result

    def _apply_normalization(self, value: str, normalization: str) -> str:
        """Apply a single normalization to a value."""
        if normalization == "lowercase":
            return value.lower()
        elif normalization == "uppercase":
            return value.upper()
        elif normalization == "trim":
            return value.strip()
        elif normalization == "snake_case":
            # Convert to snake_case
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', value)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        elif normalization == "camel_case":
            # Convert to camelCase
            parts = value.replace("-", "_").split("_")
            return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
        elif normalization == "pascal_case":
            # Convert to PascalCase
            parts = value.replace("-", "_").split("_")
            return "".join(p.capitalize() for p in parts)
        elif normalization == "remove_special":
            return re.sub(r'[^a-zA-Z0-9\s]', '', value)
        elif normalization == "normalize_spaces":
            return re.sub(r'\s+', ' ', value).strip()
        else:
            return value

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        fields = self.fields if self.fields else ["all fields"]
        norm_types = set()
        for val in self.normalizations.values():
            if isinstance(val, list):
                norm_types.update(val)
            else:
                norm_types.add(val)
        return f"Normalize {', '.join(str(f) for f in fields[:3])} using {', '.join(list(norm_types)[:3])}"


class TypeConverter(ITransformFunction):
    """Convert field types in data."""

    def __init__(
        self,
        conversions: Dict[str, Union[str, type, Callable]],
        strict: bool = False,
    ):
        """Initialize the type converter.
        
        Args:
            conversions: Dictionary mapping field names to target types.
                        Can be type names (str, int, float, bool, list, dict),
                        type objects, or callable converters.
            strict: If True, raise error on conversion failure.
        """
        self.conversions = conversions
        self.strict = strict

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by converting field types.
        
        Args:
            data: Input data.
            
        Returns:
            Transformed data with converted types.
        """
        result = copy.deepcopy(data)
        
        for field, target_type in self.conversions.items():
            if field not in result:
                continue
            
            value = result[field]
            
            try:
                result[field] = self._convert_value(value, target_type)
            except Exception as e:
                if self.strict:
                    raise TransformError(
                        f"Failed to convert field '{field}': {e}"
                    ) from e
                # Keep original value if conversion fails and not strict
        
        return result

    def _convert_value(self, value: Any, target_type: Union[str, type, Callable]) -> Any:
        """Convert a single value to target type."""
        if value is None:
            return None
        
        # Handle callable converters
        if callable(target_type) and not isinstance(target_type, type):
            return target_type(value)
        
        # Handle type names
        if isinstance(target_type, str):
            target_type = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "datetime": datetime.fromisoformat,
                "json": json.loads,
            }.get(target_type, str)
        
        # Special handling for bool conversion
        if target_type == bool and isinstance(value, str):
            return value.lower() in ["true", "yes", "1", "on"]
        
        # Special handling for datetime
        if target_type == datetime.fromisoformat and isinstance(value, str):
            return datetime.fromisoformat(value)

        # Standard type conversion
        return target_type(value)  # type: ignore

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        conversions = list(self.conversions.items())
        conv_str = ", ".join(f"{k}:{v}" for k, v in conversions[:3])
        if len(conversions) > 3:
            conv_str += "..."
        return f"Convert types: {conv_str}"


class DataEnricher(ITransformFunction):
    """Enrich data with additional fields."""

    def __init__(
        self,
        enrichments: Dict[str, Any],
        overwrite: bool = False,
    ):
        """Initialize the data enricher.
        
        Args:
            enrichments: Dictionary of fields to add/update.
                        Values can be static or callables.
            overwrite: If True, overwrite existing fields.
        """
        self.enrichments = enrichments
        self.overwrite = overwrite

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by adding enrichment fields.
        
        Args:
            data: Input data.
            
        Returns:
            Transformed data with enrichments.
        """
        result = copy.deepcopy(data)
        
        for field, value in self.enrichments.items():
            # Skip if field exists and not overwriting
            if field in result and not self.overwrite:
                continue
            
            # Evaluate value if callable
            if callable(value):
                try:
                    result[field] = value(data)
                except Exception as e:
                    raise TransformError(
                        f"Failed to compute enrichment for '{field}': {e}"
                    ) from e
            else:
                result[field] = value
        
        return result
    
    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        fields = list(self.enrichments.keys())
        return f"Enrich data with fields: {', '.join(fields[:3])}{'...' if len(fields) > 3 else ''}"


class FieldFilter(ITransformFunction):
    """Filter fields from data."""

    def __init__(
        self,
        include: List[str] | None = None,
        exclude: List[str] | None = None,
    ):
        """Initialize the field filter.
        
        Args:
            include: List of fields to include (whitelist).
            exclude: List of fields to exclude (blacklist).
        """
        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        
        self.include = include
        self.exclude = exclude

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by filtering fields.

        Args:
            data: Input data.

        Returns:
            Transformed data with filtered fields.
        """
        if self.include:
            # Include only specified fields
            return {k: v for k, v in data.items() if k in self.include}
        elif self.exclude:
            # Exclude specified fields
            return {k: v for k, v in data.items() if k not in self.exclude}
        else:
            # No filtering
            return data.copy()

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        if self.include:
            fields = ', '.join(self.include[:3])
            if len(self.include) > 3:
                fields += "..."
            return f"Include only fields: {fields}"
        elif self.exclude:
            fields = ', '.join(self.exclude[:3])
            if len(self.exclude) > 3:
                fields += "..."
            return f"Exclude fields: {fields}"
        else:
            return "No field filtering"


class ValueReplacer(ITransformFunction):
    """Replace specific values in data fields."""

    def __init__(
        self,
        replacements: Dict[str, Dict[Any, Any]],
        default_replacements: Dict[Any, Any] | None = None,
    ):
        """Initialize the value replacer.
        
        Args:
            replacements: Dictionary mapping field names to replacement mappings.
            default_replacements: Default replacements for all fields.
        """
        self.replacements = replacements
        self.default_replacements = default_replacements or {}

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by replacing values.

        Args:
            data: Input data.

        Returns:
            Transformed data with replaced values.
        """
        result = copy.deepcopy(data)

        for field, value in result.items():
            # Get replacements for this field
            field_replacements = self.replacements.get(field, self.default_replacements)

            if value in field_replacements:
                result[field] = field_replacements[value]

        return result

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        fields = list(self.replacements.keys())[:3]
        field_str = ', '.join(fields)
        if len(self.replacements) > 3:
            field_str += "..."
        return f"Replace values in fields: {field_str if fields else 'all fields'}"


class ArrayFlattener(ITransformFunction):
    """Flatten nested arrays in data."""

    def __init__(
        self,
        fields: List[str],
        depth: int = 1,
    ):
        """Initialize the array flattener.
        
        Args:
            fields: List of fields containing arrays to flatten.
            depth: Number of levels to flatten (0 = fully flatten).
        """
        self.fields = fields
        self.depth = depth

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by flattening arrays.
        
        Args:
            data: Input data.
            
        Returns:
            Transformed data with flattened arrays.
        """
        result = copy.deepcopy(data)
        
        for field in self.fields:
            if field not in result:
                continue
            
            value = result[field]
            if isinstance(value, list):
                result[field] = self._flatten(value, self.depth)
        
        return result

    def _flatten(self, arr: List, depth: int) -> List:
        """Recursively flatten an array."""
        if depth == 0:
            # Fully flatten
            result = []
            for item in arr:
                if isinstance(item, list):
                    result.extend(self._flatten(item, 0))
                else:
                    result.append(item)
            return result
        else:
            # Flatten to specified depth
            result = []
            for item in arr:
                if isinstance(item, list) and depth > 1:
                    result.extend(self._flatten(item, depth - 1))
                elif isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return result

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        fields = ', '.join(self.fields[:3])
        if len(self.fields) > 3:
            fields += "..."
        depth_str = "fully" if self.depth == 0 else f"to depth {self.depth}"
        return f"Flatten arrays in {fields} {depth_str}"


class DataSplitter(ITransformFunction):
    """Split data into multiple records based on a field."""

    def __init__(
        self,
        split_field: str,
        output_field: str = "records",
    ):
        """Initialize the data splitter.
        
        Args:
            split_field: Field containing array to split on.
            output_field: Name of output field containing split records.
        """
        self.split_field = split_field
        self.output_field = output_field

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by splitting into multiple records.
        
        Args:
            data: Input data.
            
        Returns:
            Transformed data with split records.
        """
        if self.split_field not in data:
            raise TransformError(f"Split field '{self.split_field}' not found")
        
        split_values = data[self.split_field]
        if not isinstance(split_values, list):
            raise TransformError("Split field must be a list")
        
        # Create a record for each value
        records = []
        base_data = {k: v for k, v in data.items() if k != self.split_field}

        for value in split_values:
            record = copy.deepcopy(base_data)
            record[self.split_field] = value
            records.append(record)

        return {self.output_field: records}

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Split data on field '{self.split_field}' into '{self.output_field}'"


class ChainTransformer(ITransformFunction):
    """Chain multiple transformers together."""

    def __init__(self, transformers: List[ITransformFunction]):
        """Initialize the chain transformer.
        
        Args:
            transformers: List of transformers to apply in sequence.
        """
        self.transformers = transformers

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transformers in sequence.

        Args:
            data: Input data.

        Returns:
            Transformed data after all transformers.
        """
        result = data
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        count = len(self.transformers)
        return f"Chain {count} transformer{'s' if count != 1 else ''} in sequence"


# Convenience functions for creating transformers
def map_fields(mapping: Dict[str, str], **kwargs) -> FieldMapper:
    """Create a FieldMapper."""
    return FieldMapper(mapping, **kwargs)


def normalize(**normalizations: str) -> ValueNormalizer:
    """Create a ValueNormalizer."""
    return ValueNormalizer(normalizations)


def convert_types(**conversions: Union[str, type, Callable]) -> TypeConverter:
    """Create a TypeConverter."""
    return TypeConverter(conversions)


def enrich(**enrichments: Any) -> DataEnricher:
    """Create a DataEnricher."""
    return DataEnricher(enrichments)


def filter_fields(include: List[str] | None = None, exclude: List[str] | None = None) -> FieldFilter:
    """Create a FieldFilter."""
    return FieldFilter(include, exclude)


def replace_values(**replacements: Dict[Any, Any]) -> ValueReplacer:
    """Create a ValueReplacer."""
    return ValueReplacer(replacements)


def flatten(*fields: str, depth: int = 1) -> ArrayFlattener:
    """Create an ArrayFlattener."""
    return ArrayFlattener(list(fields), depth)


def split_on(field: str, output: str = "records") -> DataSplitter:
    """Create a DataSplitter."""
    return DataSplitter(field, output)


def chain(*transformers: ITransformFunction) -> ChainTransformer:
    """Create a ChainTransformer."""
    return ChainTransformer(list(transformers))

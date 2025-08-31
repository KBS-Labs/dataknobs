"""Core converter between DataKnobs Records and Pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from dataknobs_data.fields import Field
from dataknobs_data.records import Record

from .metadata import MetadataStrategy
from .type_mapper import TypeMapper


@dataclass
class ConversionOptions:
    """Options for conversion between Records and DataFrames."""
    include_metadata: bool = False
    metadata_columns: list[str] | None = None  # Columns to treat as metadata
    flatten_nested: bool = False  # Flatten nested structures
    preserve_index: bool = True
    use_index_as_id: bool = False  # Use DataFrame index as record ID
    type_mapping: dict[str, str] | None = None  # Custom type mappings
    null_handling: str = "preserve"  # "preserve", "drop", "fill"
    datetime_format: str | None = None  # Format for datetime conversion
    timezone: str | None = None  # Timezone for datetime conversion

    # Keep these for backward compatibility
    preserve_types: bool = True
    index_column: str | None = None  # Use specific field as index
    flatten_json: bool = False
    metadata_strategy: MetadataStrategy = MetadataStrategy.ATTRS
    handle_missing: str = "preserve"  # "preserve", "drop", "fill"
    fill_value: Any = None

    def __post_init__(self):
        """Initialize default values for mutable parameters."""
        if self.metadata_columns is None:
            self.metadata_columns = []
        if self.type_mapping is None:
            self.type_mapping = {}

    def merge_metadata(self, meta1: dict[str, Any], meta2: dict[str, Any]) -> dict[str, Any]:
        """Merge two metadata dictionaries.
        
        Args:
            meta1: First metadata dict
            meta2: Second metadata dict (overwrites meta1 on conflicts)
            
        Returns:
            Merged metadata dictionary
        """
        result = meta1.copy()
        for key, value in meta2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self.merge_metadata(result[key], value)
            else:
                result[key] = value
        return result


class DataFrameConverter:
    """Converts between DataKnobs Records and Pandas DataFrames."""

    def __init__(self, type_mapper: TypeMapper | None = None):
        """Initialize converter.
        
        Args:
            type_mapper: Custom type mapper (uses default if None)
        """
        self.type_mapper = type_mapper or TypeMapper()

    def records_to_dataframe(
        self,
        records: list[Record],
        options: ConversionOptions | None = None
    ) -> pd.DataFrame:
        """Convert list of Records to DataFrame.
        
        Args:
            records: List of Records to convert
            options: Conversion options
            
        Returns:
            Pandas DataFrame
        """
        options = options or ConversionOptions()

        if not records:
            return pd.DataFrame()

        # Extract data from records
        data_rows = []
        for record in records:
            row = {}

            # Add field values
            for field_name, field in record.fields.items():
                if options.flatten_nested and isinstance(field.value, dict):
                    # Flatten nested dictionaries
                    for nested_key, nested_val in field.value.items():
                        row[f"{field_name}.{nested_key}"] = nested_val
                else:
                    row[field_name] = field.value

            # Add metadata as a column if requested
            if options.include_metadata and record.metadata:
                row["_metadata"] = record.metadata

            data_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Preserve column order from the first record for consistency
        # This maintains the order fields were added to records
        if not df.empty and records:
            # Get column order from first record's fields
            first_record = records[0]
            column_order = []

            # Add field columns in the order they appear in the record
            for field_name in first_record.fields.keys():
                if options.flatten_nested and isinstance(first_record.fields[field_name].value, dict):
                    # Add flattened columns
                    for nested_key in first_record.fields[field_name].value.keys():
                        col_name = f"{field_name}.{nested_key}"
                        if col_name in df.columns:
                            column_order.append(col_name)
                elif field_name in df.columns:
                    column_order.append(field_name)

            # Add any remaining columns (like _metadata) at the end
            for col in df.columns:
                if col not in column_order:
                    column_order.append(col)

            # Reorder DataFrame columns
            df = df[column_order]

        # Set index if specified
        if options.index_column and options.index_column in df.columns:
            df = df.set_index(options.index_column)
        elif options.preserve_index:
            # Only use record IDs as index if:
            # 1. They exist
            # 2. They're not coming from a data field (id or record_id columns)
            # 3. They don't look like auto-generated UUIDs
            record_ids = [r.id for r in records]

            # Check if the IDs are from data fields
            ids_from_fields = any(
                'id' in r.fields or 'record_id' in r.fields
                for r in records
            )

            # Only set index if IDs exist, aren't from fields, and aren't UUIDs
            if (any(record_ids) and not ids_from_fields and not all(
                id and len(id) == 36 and id.count('-') == 4
                for id in record_ids if id
            )):
                df.index = record_ids
                df.index.name = "record_id"

        return df

    def dataframe_to_records(
        self,
        df: pd.DataFrame,
        options: ConversionOptions | None = None
    ) -> list[Record]:
        """Convert DataFrame to list of Records.
        
        Args:
            df: DataFrame to convert
            options: Conversion options
            
        Returns:
            List of Records
        """
        options = options or ConversionOptions()

        records = []

        # Convert each row to a Record
        for idx, row in df.iterrows():
            # Extract metadata for this row from metadata columns
            row_metadata = {}
            if options.metadata_columns:
                for col in options.metadata_columns:
                    if col in row.index:
                        col_value = row[col]
                        # If the column value is a dict, merge it into metadata
                        if isinstance(col_value, dict):
                            row_metadata.update(col_value)
                        else:
                            # Otherwise store it with the column name (without leading underscore)
                            row_metadata[col.lstrip('_')] = col_value

            # Prepare row data (excluding metadata columns)
            row_data = {}
            for col in row.index:
                if options.metadata_columns is None or col not in options.metadata_columns:
                    row_data[col] = row[col]

            # Determine record ID
            record_id = None
            if options.use_index_as_id:
                if isinstance(idx, str):
                    record_id = idx
                elif idx is not None and not pd.isna(idx):  # type: ignore[call-overload]
                    record_id = str(idx)

            # Create record
            record = Record(data=row_data, metadata=row_metadata, id=record_id)
            records.append(record)

        return records

    def record_to_series(self, record: Record) -> pd.Series:
        """Convert a single Record to a Pandas Series.
        
        Args:
            record: Record to convert
            
        Returns:
            Pandas Series
        """
        data = {}
        for field_name, field in record.fields.items():
            if field.type is not None:
                value = self.type_mapper.convert_value_to_pandas(field.value, field.type)
            else:
                value = field.value
            data[field_name] = value

        series = pd.Series(data)
        if record.id:
            series.name = record.id

        return series

    def series_to_record(
        self,
        series: pd.Series,
        record_id: str | None = None
    ) -> Record:
        """Convert a Pandas Series to a Record.
        
        Args:
            series: Series to convert
            record_id: Optional record ID
            
        Returns:
            Record
        """
        # Get ID - series.name is Hashable, we need str | None
        id_value = record_id
        if not id_value and hasattr(series, 'name'):
            name = series.name
            id_value = str(name) if name is not None else None
        record = Record(id=id_value)

        for column, value in series.items():
            # Skip metadata columns
            if isinstance(column, str) and column.startswith("_meta_"):
                continue

            # Infer field type
            field_type = self.type_mapper.infer_field_type_from_value(value)

            # Convert value
            field_value = self.type_mapper.convert_value_from_pandas(value, field_type)

            # Create field
            field = Field(
                name=str(column),
                value=field_value,
                type=field_type
            )
            record.fields[str(column)] = field

        return record

    def _series_to_record(
        self,
        row: pd.Series,
        idx: Any,
        options: ConversionOptions
    ) -> Record:
        """Convert a DataFrame row to a Record.
        
        Args:
            row: DataFrame row as Series
            idx: Row index
            options: Conversion options
            
        Returns:
            Record
        """
        # Determine record ID
        record_id = None
        if options.preserve_index:
            if isinstance(idx, str):
                record_id = idx
            elif idx is not None and not pd.isna(idx):
                record_id = str(idx)

        record = Record(id=record_id)

        for original_column, value in row.items():
            # Skip metadata columns
            if isinstance(original_column, str) and original_column.startswith("_meta_"):
                continue

            # Handle multi-index columns
            column = original_column
            if isinstance(column, tuple):
                column = column[0]  # Use first level

            # Infer field type
            field_type = self.type_mapper.infer_field_type_from_value(value)

            # Convert value
            field_value = self.type_mapper.convert_value_from_pandas(value, field_type)

            # Create field
            field = Field(
                name=str(column),
                value=field_value,
                type=field_type
            )
            record.fields[str(column)] = field

        return record

    def _flatten_json_value(self, value: Any) -> Any:
        """Flatten JSON value for DataFrame insertion.
        
        Args:
            value: JSON value (dict or list)
            
        Returns:
            Flattened value or string representation
        """
        # Check for None explicitly first
        if value is None:
            return value

        # Check for pandas NA types
        try:
            if pd.isna(value):
                return value
        except (TypeError, ValueError):
            # pd.isna doesn't work with lists/dicts
            pass

        if isinstance(value, dict):
            # For dict, could expand to multiple columns
            # For now, convert to string
            return str(value)
        elif isinstance(value, list):
            # For list, convert to string
            return str(value)

        return value

    def validate_conversion(
        self,
        records: list[Record],
        df: pd.DataFrame,
        options: ConversionOptions | None = None
    ) -> dict[str, Any]:
        """Validate conversion accuracy.
        
        Args:
            records: Original records
            df: Converted DataFrame
            options: Conversion options used
            
        Returns:
            Validation report
        """
        options = options or ConversionOptions()

        report: dict[str, Any] = {
            "record_count_match": len(records) == len(df),
            "original_record_count": len(records),
            "dataframe_row_count": len(df),
            "field_preservation": {},
            "type_preservation": {},
            "value_accuracy": {}
        }

        # Check field preservation
        original_fields = set()
        for record in records:
            original_fields.update(record.fields.keys())

        df_columns = set(df.columns)
        if options.metadata_strategy == MetadataStrategy.COLUMNS:
            df_columns = {col for col in df_columns if not col.startswith("_meta_")}

        report["field_preservation"] = {
            "original_fields": sorted(original_fields),
            "dataframe_columns": sorted(df_columns),
            "missing_fields": sorted(original_fields - df_columns),
            "extra_columns": sorted(df_columns - original_fields)
        }

        # Check type preservation if enabled
        if options.preserve_types:
            for record in records[:10]:  # Sample first 10 records
                for field_name, field in record.fields.items():
                    if field_name in df.columns and field.type is not None:
                        df_dtype = str(df[field_name].dtype)
                        expected_dtype = str(self.type_mapper.field_type_to_pandas(field.type))
                        if df_dtype != expected_dtype:
                            type_preservation = report["type_preservation"]
                            if not isinstance(type_preservation, dict):
                                raise TypeError("type_preservation should be a dict")
                            type_preservation[field_name] = {
                                "expected": expected_dtype,
                                "actual": df_dtype
                            }

        return report

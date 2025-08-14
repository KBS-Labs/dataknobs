"""Core converter between DataKnobs Records and Pandas DataFrames."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from dataknobs_data.records import Record
from dataknobs_data.fields import Field, FieldType
from .type_mapper import TypeMapper
from .metadata import MetadataHandler, MetadataConfig, MetadataStrategy


@dataclass
class ConversionOptions:
    """Options for conversion between Records and DataFrames."""
    include_metadata: bool = False
    preserve_types: bool = True
    preserve_index: bool = True
    index_column: Optional[str] = None  # Use specific field as index
    flatten_json: bool = False
    metadata_strategy: MetadataStrategy = MetadataStrategy.ATTRS
    handle_missing: str = "preserve"  # "preserve", "drop", "fill"
    fill_value: Any = None


class DataFrameConverter:
    """Converts between DataKnobs Records and Pandas DataFrames."""
    
    def __init__(self, type_mapper: Optional[TypeMapper] = None):
        """Initialize converter.
        
        Args:
            type_mapper: Custom type mapper (uses default if None)
        """
        self.type_mapper = type_mapper or TypeMapper()
        
    def records_to_dataframe(
        self,
        records: List[Record],
        options: Optional[ConversionOptions] = None
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
        
        # Extract data
        data_dict = self._records_to_dict(records, options)
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Set index if specified
        if options.index_column and options.index_column in df.columns:
            df = df.set_index(options.index_column)
        elif options.preserve_index:
            # Use record IDs as index
            record_ids = [r.id for r in records]
            if any(record_ids):  # If at least some records have IDs
                df.index = record_ids
                df.index.name = "record_id"
        
        # Apply type conversion if requested
        if options.preserve_types:
            df = self._apply_field_types(df, records)
        
        # Handle metadata
        if options.include_metadata:
            metadata_config = MetadataConfig(strategy=options.metadata_strategy)
            metadata_handler = MetadataHandler(metadata_config)
            metadata = metadata_handler.extract_metadata_from_records(records)
            df = metadata_handler.apply_metadata_to_dataframe(df, metadata, records)
        
        # Handle missing values
        df = self._handle_missing_values(df, options)
        
        return df
    
    def dataframe_to_records(
        self,
        df: pd.DataFrame,
        options: Optional[ConversionOptions] = None
    ) -> List[Record]:
        """Convert DataFrame to list of Records.
        
        Args:
            df: DataFrame to convert
            options: Conversion options
            
        Returns:
            List of Records
        """
        options = options or ConversionOptions()
        
        records = []
        
        # Extract metadata if present
        metadata = None
        if options.include_metadata:
            metadata_config = MetadataConfig(strategy=options.metadata_strategy)
            metadata_handler = MetadataHandler(metadata_config)
            metadata = metadata_handler.extract_metadata_from_dataframe(df)
        
        # Convert each row to a Record
        for idx, row in df.iterrows():
            record = self._series_to_record(row, idx, options)
            records.append(record)
        
        # Apply metadata if extracted
        if metadata:
            metadata_handler = MetadataHandler(MetadataConfig(strategy=options.metadata_strategy))
            records = metadata_handler.create_records_with_metadata(df, records, metadata)
        
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
            value = self.type_mapper.convert_value_to_pandas(field.value, field.type)
            data[field_name] = value
        
        series = pd.Series(data)
        if record.id:
            series.name = record.id
        
        return series
    
    def series_to_record(
        self,
        series: pd.Series,
        record_id: Optional[str] = None
    ) -> Record:
        """Convert a Pandas Series to a Record.
        
        Args:
            series: Series to convert
            record_id: Optional record ID
            
        Returns:
            Record
        """
        record = Record(id=record_id or series.name if hasattr(series, 'name') else None)
        
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
    
    def _records_to_dict(
        self,
        records: List[Record],
        options: ConversionOptions
    ) -> Dict[str, List]:
        """Convert records to dictionary format for DataFrame creation.
        
        Args:
            records: List of records
            options: Conversion options
            
        Returns:
            Dictionary of columns
        """
        # Collect all field names
        all_fields = set()
        for record in records:
            all_fields.update(record.fields.keys())
        
        # Initialize data dictionary
        data_dict = {field: [] for field in all_fields}
        
        # Populate data
        for record in records:
            for field_name in all_fields:
                if field_name in record.fields:
                    field = record.fields[field_name]
                    value = self.type_mapper.convert_value_to_pandas(field.value, field.type)
                    
                    # Handle JSON flattening if requested
                    if options.flatten_json and field.type == FieldType.JSON:
                        value = self._flatten_json_value(value)
                    
                    data_dict[field_name].append(value)
                else:
                    # Field not present in this record
                    data_dict[field_name].append(pd.NA)
        
        return data_dict
    
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
        
        for column, value in row.items():
            # Skip metadata columns
            if isinstance(column, str) and column.startswith("_meta_"):
                continue
            
            # Handle multi-index columns
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
    
    def _apply_field_types(self, df: pd.DataFrame, records: List[Record]) -> pd.DataFrame:
        """Apply field types from records to DataFrame columns.
        
        Args:
            df: DataFrame to type
            records: Source records with type information
            
        Returns:
            DataFrame with proper types
        """
        # Collect field types from records
        field_types = {}
        for record in records:
            for field_name, field in record.fields.items():
                if field_name not in field_types and field.type:
                    field_types[field_name] = field.type
        
        # Apply types to DataFrame columns
        for column in df.columns:
            if column in field_types:
                field_type = field_types[column]
                df[column] = self.type_mapper.cast_series(df[column], field_type)
        
        return df
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        options: ConversionOptions
    ) -> pd.DataFrame:
        """Handle missing values based on options.
        
        Args:
            df: DataFrame to process
            options: Conversion options
            
        Returns:
            Processed DataFrame
        """
        if options.handle_missing == "drop":
            return df.dropna()
        elif options.handle_missing == "fill":
            return df.fillna(options.fill_value)
        else:  # "preserve"
            return df
    
    def _flatten_json_value(self, value: Any) -> Any:
        """Flatten JSON value for DataFrame insertion.
        
        Args:
            value: JSON value (dict or list)
            
        Returns:
            Flattened value or string representation
        """
        if pd.isna(value):
            return value
        
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
        records: List[Record],
        df: pd.DataFrame,
        options: Optional[ConversionOptions] = None
    ) -> Dict[str, Any]:
        """Validate conversion accuracy.
        
        Args:
            records: Original records
            df: Converted DataFrame
            options: Conversion options used
            
        Returns:
            Validation report
        """
        options = options or ConversionOptions()
        
        report = {
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
                    if field_name in df.columns:
                        df_dtype = str(df[field_name].dtype)
                        expected_dtype = str(self.type_mapper.field_type_to_pandas(field.type))
                        if df_dtype != expected_dtype:
                            report["type_preservation"][field_name] = {
                                "expected": expected_dtype,
                                "actual": df_dtype
                            }
        
        return report
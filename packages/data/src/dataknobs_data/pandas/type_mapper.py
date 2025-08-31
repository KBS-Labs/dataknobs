"""Type mapping between DataKnobs Field types and Pandas dtypes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from dataknobs_data.fields import FieldType

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class PandasTypeMapping:
    """Mapping configuration for type conversion."""
    field_type: FieldType
    pandas_dtype: str | type | np.dtype
    nullable: bool = True
    converter: Callable | None = None
    reverse_converter: Callable | None = None


class TypeMapper:
    """Handles type mapping between DataKnobs Field types and Pandas dtypes."""

    def __init__(self):
        """Initialize type mapper with default mappings."""
        self._init_mappings()

    def _init_mappings(self):
        """Initialize type mappings."""
        self._field_to_pandas: dict[FieldType, PandasTypeMapping] = {
            FieldType.STRING: PandasTypeMapping(
                field_type=FieldType.STRING,
                pandas_dtype="string",  # pd.StringDtype()
                nullable=True
            ),
            FieldType.INTEGER: PandasTypeMapping(
                field_type=FieldType.INTEGER,
                pandas_dtype="Int64",  # pd.Int64Dtype()
                nullable=True
            ),
            FieldType.FLOAT: PandasTypeMapping(
                field_type=FieldType.FLOAT,
                pandas_dtype="Float64",  # pd.Float64Dtype()
                nullable=True
            ),
            FieldType.BOOLEAN: PandasTypeMapping(
                field_type=FieldType.BOOLEAN,
                pandas_dtype="boolean",  # pd.BooleanDtype()
                nullable=True
            ),
            FieldType.DATETIME: PandasTypeMapping(
                field_type=FieldType.DATETIME,
                pandas_dtype="datetime64[ns]",
                nullable=True,
                converter=self._to_datetime,
                reverse_converter=self._from_datetime
            ),
            FieldType.JSON: PandasTypeMapping(
                field_type=FieldType.JSON,
                pandas_dtype="object",
                nullable=True,
                converter=self._to_json_object,
                reverse_converter=self._from_json_object
            ),
            FieldType.BINARY: PandasTypeMapping(
                field_type=FieldType.BINARY,
                pandas_dtype="object",
                nullable=True
            ),
            FieldType.TEXT: PandasTypeMapping(
                field_type=FieldType.TEXT,
                pandas_dtype="string",
                nullable=True
            ),
        }

        # Reverse mapping from pandas to field types
        self._pandas_to_field: dict[str, FieldType] = {
            "string": FieldType.STRING,
            "int64": FieldType.INTEGER,
            "float64": FieldType.FLOAT,
            "boolean": FieldType.BOOLEAN,
            "datetime64[ns]": FieldType.DATETIME,
            "object": FieldType.STRING,  # Default object to STRING, not JSON
        }

    def field_type_to_pandas(self, field_type: FieldType) -> str | type | np.dtype:
        """Convert FieldType to pandas dtype.
        
        Args:
            field_type: DataKnobs FieldType
            
        Returns:
            Corresponding pandas dtype
        """
        mapping = self._field_to_pandas.get(field_type)
        if mapping:
            return mapping.pandas_dtype
        return "object"  # Default fallback

    def pandas_to_field_type(self, dtype: str | np.dtype | type) -> FieldType:
        """Infer FieldType from pandas dtype.
        
        Args:
            dtype: Pandas dtype
            
        Returns:
            Corresponding FieldType
        """
        dtype_str = str(dtype).lower()

        # Direct mapping
        if dtype_str in self._pandas_to_field:
            return self._pandas_to_field[dtype_str]

        # Infer from dtype categories
        if "int" in dtype_str:
            return FieldType.INTEGER
        elif "float" in dtype_str:
            return FieldType.FLOAT
        elif "bool" in dtype_str:
            return FieldType.BOOLEAN
        elif "datetime" in dtype_str or "timestamp" in dtype_str:
            return FieldType.DATETIME
        elif dtype_str == "string" or dtype_str == "object":
            return FieldType.STRING
        elif "bytes" in dtype_str:
            return FieldType.BINARY

        return FieldType.STRING  # Default fallback

    def convert_value_to_pandas(self, value: Any, field_type: FieldType) -> Any:
        """Convert a field value to pandas-compatible format.
        
        Args:
            value: Value to convert
            field_type: Source field type
            
        Returns:
            Pandas-compatible value
        """
        if value is None:
            return pd.NA

        mapping = self._field_to_pandas.get(field_type)
        if mapping and mapping.converter:
            return mapping.converter(value)

        return value

    def convert_value_from_pandas(self, value: Any, field_type: FieldType) -> Any:
        """Convert a pandas value to field-compatible format.
        
        Args:
            value: Pandas value
            field_type: Target field type
            
        Returns:
            Field-compatible value
        """
        # Handle pandas NA/NaN/None
        # Use try-except to handle arrays and other special cases
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            # pd.isna can fail on arrays/lists
            pass

        mapping = self._field_to_pandas.get(field_type)
        if mapping and mapping.reverse_converter:
            return mapping.reverse_converter(value)

        # Handle numpy types
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()

        return value

    def infer_field_type_from_value(self, value: Any) -> FieldType:
        """Infer FieldType from a Python value.
        
        Args:
            value: Value to analyze
            
        Returns:
            Inferred FieldType
        """
        if value is None:
            return FieldType.STRING  # Default for null

        # Check for pandas NA separately to avoid array ambiguity
        try:
            if pd.isna(value):
                return FieldType.STRING
        except (TypeError, ValueError):
            # pd.isna can fail on some types like lists
            pass

        if isinstance(value, (bool, np.bool_)):
            return FieldType.BOOLEAN
        elif isinstance(value, (int, np.integer)):
            return FieldType.INTEGER
        elif isinstance(value, (float, np.floating)):
            return FieldType.FLOAT
        elif isinstance(value, (datetime, pd.Timestamp)):
            return FieldType.DATETIME
        elif isinstance(value, bytes):  # type: ignore[unreachable]
            return FieldType.BINARY
        elif isinstance(value, (dict, list)):
            return FieldType.JSON
        elif isinstance(value, str):
            if len(value) > 1000:
                return FieldType.TEXT
            return FieldType.STRING
        else:
            # Complex objects as JSON
            return FieldType.JSON

    def cast_series(self, series: pd.Series, field_type: FieldType) -> pd.Series:
        """Cast a pandas Series to the appropriate dtype for a FieldType.
        
        Args:
            series: Series to cast
            field_type: Target field type
            
        Returns:
            Casted Series
        """
        target_dtype = self.field_type_to_pandas(field_type)

        try:
            # Special handling for datetime
            if field_type == FieldType.DATETIME:
                return pd.to_datetime(series, errors='coerce')

            # Special handling for JSON
            if field_type == FieldType.JSON:
                return series.apply(self._ensure_json_serializable)

            # Standard casting
            return series.astype(target_dtype)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # If casting fails, return as object dtype
            return series.astype("object")

    @staticmethod
    def _to_datetime(value: Any) -> pd.Timestamp:
        """Convert value to pandas Timestamp."""
        if isinstance(value, (str, datetime)):
            return pd.Timestamp(value)
        elif isinstance(value, (int, float)):
            # Assume Unix timestamp
            return pd.Timestamp(value, unit='s')
        return value

    @staticmethod
    def _from_datetime(value: Any) -> datetime:
        """Convert pandas Timestamp to datetime."""
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        elif isinstance(value, str):
            return pd.Timestamp(value).to_pydatetime()
        return value

    @staticmethod
    def _to_json_object(value: Any) -> Any:
        """Ensure value is JSON-serializable object."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    @staticmethod
    def _from_json_object(value: Any) -> Any:
        """Convert object to JSON-compatible format."""
        if isinstance(value, (dict, list)):
            return value
        elif isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    @staticmethod
    def _ensure_json_serializable(value: Any) -> Any:
        """Ensure value is JSON-serializable."""
        if pd.isna(value):
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        # Convert other types to string representation
        return str(value)

    def infer_field_type(self, series: pd.Series) -> str:
        """Infer field type from a pandas Series.
        
        Args:
            series: Series to analyze
            
        Returns:
            Field type string
        """
        # Remove nulls for analysis
        non_null = series.dropna()

        if len(non_null) == 0:
            return "string"  # Default for empty

        # Check dtypes
        if is_bool_dtype(non_null):
            return "boolean"
        elif is_datetime64_any_dtype(non_null):
            return "datetime"
        elif is_numeric_dtype(non_null):
            # Check if all values are integers
            if non_null.apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())).all():
                return "integer"
            else:
                return "number"
        else:
            # Check values for special types
            sample = non_null.iloc[0] if len(non_null) > 0 else None
            if sample is not None:
                if isinstance(sample, (datetime, pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp, pd._libs.tslibs.nattype.NaTType)):
                    return "datetime"
                elif hasattr(sample, '__class__') and 'date' in sample.__class__.__name__.lower():  # type: ignore[unreachable]
                    return "date"
                elif hasattr(sample, '__class__') and 'time' in sample.__class__.__name__.lower():
                    return "time"
                else:
                    # Other object types
                    return "string"
            else:
                # No sample available
                return "string"

    def get_pandas_dtype(self, field_type: str) -> str:
        """Get pandas dtype for a field type string.
        
        Args:
            field_type: Field type string
            
        Returns:
            Pandas dtype string
        """
        dtype_map = {
            "string": "object",
            "integer": "int64",
            "number": "float64",
            "float": "float64",
            "boolean": "bool",
            "datetime": "datetime64[ns]",
            "date": "object",
            "time": "object",
            "json": "object",
            "binary": "object",
            "text": "object",
        }
        return dtype_map.get(field_type.lower(), "object")

    def convert_value(self, value: Any, target_type: str) -> Any:
        """Convert a value to target type.
        
        Args:
            value: Value to convert
            target_type: Target type string
            
        Returns:
            Converted value
        """
        if value is None or pd.isna(value):
            return None

        target_type = target_type.lower()

        if target_type == "integer":
            if isinstance(value, str):
                return int(float(value))
            return int(value)
        elif target_type == "number" or target_type == "float":
            return float(value)
        elif target_type == "string":
            return str(value)
        elif target_type == "boolean":
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return bool(value)
        elif target_type == "datetime":
            if isinstance(value, str):
                return pd.Timestamp(value)
            return value
        elif target_type == "date":
            if isinstance(value, str):
                return pd.Timestamp(value).date()
            elif hasattr(value, 'date'):
                return value.date()
            return value
        elif target_type == "time":
            if isinstance(value, str):
                return pd.Timestamp(value).time()
            elif hasattr(value, 'time'):
                return value.time()
            return value

        return value

    def cast_dataframe_dtypes(self, df: pd.DataFrame, dtype_map: dict[str, str]) -> pd.DataFrame:
        """Cast DataFrame columns to specified dtypes.
        
        Args:
            df: DataFrame to cast
            dtype_map: Dictionary of column: dtype
            
        Returns:
            DataFrame with casted dtypes
        """
        result_df = df.copy()

        for col, dtype in dtype_map.items():
            if col in result_df.columns:
                try:
                    if dtype == "string":
                        # Use string dtype for nullable strings
                        result_df[col] = result_df[col].astype("string")
                    else:
                        result_df[col] = result_df[col].astype(dtype)  # type: ignore[call-overload]
                except (TypeError, ValueError):
                    # If casting fails, leave as is
                    pass

        return result_df

    def normalize_timezone(self, series: pd.Series, target_tz: str) -> pd.Series:
        """Normalize timezone for datetime series.
        
        Args:
            series: Datetime series
            target_tz: Target timezone
            
        Returns:
            Series with normalized timezone
        """
        if not is_datetime64_any_dtype(series):
            # Try to convert to datetime first
            series = pd.to_datetime(series, errors='coerce')

        # If series is timezone-naive, localize it
        if series.dt.tz is None:
            return series.dt.tz_localize(target_tz)
        else:
            # If timezone-aware, convert to target timezone
            return series.dt.tz_convert(target_tz)

    def get_optimal_dtype(self, series: pd.Series) -> str:
        """Determine optimal dtype for a Series based on its values.
        
        Args:
            series: Series to analyze
            
        Returns:
            Optimal dtype string
        """
        # Remove nulls for analysis
        non_null = series.dropna()

        if len(non_null) == 0:
            return "string"  # Default for empty

        # Try to infer the best dtype
        try:
            # Check for boolean
            if non_null.apply(lambda x: isinstance(x, bool)).all():
                return "bool"

            # Check for integer
            if non_null.apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())).all():
                # Determine the smallest int type that can hold the values
                min_val = non_null.min()
                max_val = non_null.max()

                if min_val >= -128 and max_val <= 127:
                    return "int8"
                elif min_val >= -32768 and max_val <= 32767:
                    return "int16"
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return "int32"
                else:
                    return "int64"

            # Check for float
            if non_null.apply(lambda x: isinstance(x, (int, float, np.number))).all():
                # For floats, prefer float32 for small ranges
                max_val = non_null.abs().max()
                if max_val <= 3.4e38:
                    return "float32"
                else:
                    return "float64"

            # Check for datetime
            try:
                # Use infer_datetime_format to suppress the warning
                pd.to_datetime(non_null, format='mixed')
                return "datetime64[ns]"
            except (ValueError, TypeError):
                pass

            # Default to object for strings and mixed types
            return "object"
        except Exception:
            return "object"

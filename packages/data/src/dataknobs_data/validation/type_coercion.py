"""Type coercion utilities for schema validation."""

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Type, Union

from dataknobs_data.fields import FieldType

logger = logging.getLogger(__name__)


class CoercionError(Exception):
    """Raised when type coercion fails."""
    pass


class TypeCoercer:
    """Handle type coercion for field values."""
    
    def __init__(self):
        """Initialize type coercer."""
        self._coercion_map = {
            str: self._to_string,
            int: self._to_int,
            float: self._to_float,
            bool: self._to_bool,
            list: self._to_list,
            dict: self._to_dict,
            datetime: self._to_datetime,
        }
        
        self._field_type_map = {
            FieldType.STRING: self._to_string,
            FieldType.INTEGER: self._to_int,
            FieldType.FLOAT: self._to_float,
            FieldType.BOOLEAN: self._to_bool,
            FieldType.JSON: self._to_dict,  # JSON can represent lists/dicts
            FieldType.DATETIME: self._to_datetime,
        }
    
    def coerce(self, value: Any, target_type: Union[Type, FieldType, str]) -> Any:
        """Coerce a value to the target type.
        
        Args:
            value: Value to coerce
            target_type: Target type (Type, FieldType, or string)
            
        Returns:
            Coerced value
            
        Raises:
            CoercionError: If coercion fails
        """
        # Handle None values
        if value is None:
            return None
        
        # Determine coercion function
        if isinstance(target_type, type):
            coercion_func = self._coercion_map.get(target_type)
        elif isinstance(target_type, FieldType):
            coercion_func = self._field_type_map.get(target_type)
        elif isinstance(target_type, str):
            # String type name
            type_name_map = {
                'str': self._to_string,
                'string': self._to_string,
                'int': self._to_int,
                'integer': self._to_int,
                'float': self._to_float,
                'double': self._to_float,
                'bool': self._to_bool,
                'boolean': self._to_bool,
                'list': self._to_list,
                'array': self._to_list,
                'dict': self._to_dict,
                'object': self._to_dict,
                'datetime': self._to_datetime,
                'date': self._to_datetime,
            }
            coercion_func = type_name_map.get(target_type.lower())
        else:
            coercion_func = None
        
        if not coercion_func:
            # No coercion needed or unknown type
            return value
        
        try:
            return coercion_func(value)
        except Exception as e:
            raise CoercionError(f"Failed to coerce {value!r} to {target_type}: {e}")
    
    def _to_string(self, value: Any) -> str:
        """Coerce to string."""
        if isinstance(value, str):
            return value
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        elif isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return str(value)
    
    def _to_int(self, value: Any) -> int:
        """Coerce to integer."""
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value, str):
            # Handle various string formats
            value = value.strip()
            if value == "":
                raise ValueError("Empty string cannot be converted to int")
            
            # Handle boolean strings
            if value.lower() in ('true', 'yes', 'on'):
                return 1
            elif value.lower() in ('false', 'no', 'off'):
                return 0
            
            # Handle numeric strings
            try:
                # Try direct int conversion
                return int(value)
            except ValueError:
                # Try float then int (handles "1.0")
                return int(float(value))
        elif isinstance(value, bool):
            return 1 if value else 0
        else:
            raise ValueError(f"Cannot convert {type(value).__name__} to int")
    
    def _to_float(self, value: Any) -> float:
        """Coerce to float."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            value = value.strip()
            if value == "":
                raise ValueError("Empty string cannot be converted to float")
            
            # Handle special values
            if value.lower() == 'inf':
                return float('inf')
            elif value.lower() == '-inf':
                return float('-inf')
            elif value.lower() == 'nan':
                return float('nan')
            
            return float(value)
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        else:
            raise ValueError(f"Cannot convert {type(value).__name__} to float")
    
    def _to_bool(self, value: Any) -> bool:
        """Coerce to boolean."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            value = value.strip().lower()
            if value in ('true', 'yes', '1', 'on', 't', 'y'):
                return True
            elif value in ('false', 'no', '0', 'off', 'f', 'n', ''):
                return False
            else:
                raise ValueError(f"Cannot interpret '{value}' as boolean")
        elif isinstance(value, (int, float)):
            return value != 0
        elif value is None:
            return False
        else:
            return bool(value)
    
    def _to_list(self, value: Any) -> List[Any]:
        """Coerce to list."""
        if isinstance(value, list):
            return value
        elif isinstance(value, tuple):
            return list(value)
        elif isinstance(value, set):
            return list(value)
        elif isinstance(value, dict):
            # Convert dict to list of key-value pairs
            return list(value.items())
        elif isinstance(value, str):
            # Try to parse JSON array
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            
            # Split comma-separated values
            if ',' in value:
                return [v.strip() for v in value.split(',')]
            
            # Single value as list
            return [value] if value else []
        else:
            # Wrap single value in list
            return [value]
    
    def _to_dict(self, value: Any) -> Dict[str, Any]:
        """Coerce to dictionary."""
        if isinstance(value, dict):
            return value
        elif isinstance(value, str):
            # Try to parse JSON object
            value = value.strip()
            if value.startswith('{') and value.endswith('}'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            
            # Try key=value format
            if '=' in value:
                result = {}
                for item in value.split(','):
                    if '=' in item:
                        key, val = item.split('=', 1)
                        result[key.strip()] = val.strip()
                return result
            
            # Empty dict for empty string
            return {} if not value else {'value': value}
        elif isinstance(value, (list, tuple)):
            # Convert list of pairs to dict
            if value and isinstance(value[0], (list, tuple)) and len(value[0]) == 2:
                return dict(value)
            # Convert list to dict with numeric keys
            return {str(i): v for i, v in enumerate(value)}
        elif hasattr(value, '__dict__'):
            # Convert object to dict
            return value.__dict__
        else:
            # Wrap value in dict
            return {'value': value}
    
    def _to_datetime(self, value: Any) -> datetime:
        """Coerce to datetime."""
        if isinstance(value, datetime):
            return value
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("Empty string cannot be converted to datetime")
            
            # Try common datetime formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%m/%d/%Y %H:%M:%S',
                '%d/%m/%Y',
                '%d/%m/%Y %H:%M:%S',
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            
            # Try ISO format
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                pass
            
            # Try timestamp
            try:
                timestamp = float(value)
                return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                pass
            
            raise ValueError(f"Cannot parse '{value}' as datetime")
        elif isinstance(value, (int, float)):
            # Assume Unix timestamp
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, OSError) as e:
                raise ValueError(f"Invalid timestamp {value}: {e}")
        else:
            raise ValueError(f"Cannot convert {type(value).__name__} to datetime")
    
    def register_coercion(
        self,
        target_type: Union[Type, str],
        coercion_func: Callable[[Any], Any]
    ) -> None:
        """Register a custom coercion function.
        
        Args:
            target_type: Target type or type name
            coercion_func: Function to coerce values to target type
        """
        if isinstance(target_type, type):
            self._coercion_map[target_type] = coercion_func
        elif isinstance(target_type, str):
            # Store in a separate map for string type names
            if not hasattr(self, '_custom_coercions'):
                self._custom_coercions = {}
            self._custom_coercions[target_type.lower()] = coercion_func
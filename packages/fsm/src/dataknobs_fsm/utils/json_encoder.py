"""Custom JSON encoder for FSM objects."""

import json
from typing import Any


class FSMJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles FSM-specific types."""

    def default(self, obj: Any) -> Any:
        """Convert FSM objects to JSON-serializable forms.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        # Check for our custom serialization methods
        if hasattr(obj, '__json__'):
            return obj.__json__()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()

        # Handle FSMData specifically
        from dataknobs_fsm.core.data_wrapper import FSMData
        if isinstance(obj, FSMData):
            return obj.to_dict()

        # Handle ExecutionResult specifically
        from dataknobs_fsm.functions.base import ExecutionResult
        if isinstance(obj, ExecutionResult):
            return obj.to_dict()

        # Fall back to default encoder
        return super().default(obj)


def dumps(obj: Any, **kwargs) -> str:
    """Serialize obj to JSON string with FSM support.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string
    """
    kwargs.setdefault('cls', FSMJSONEncoder)
    return json.dumps(obj, **kwargs)


def loads(s: str, **kwargs) -> Any:
    """Deserialize JSON string to Python object.

    Args:
        s: JSON string
        **kwargs: Additional arguments for json.loads

    Returns:
        Python object
    """
    return json.loads(s, **kwargs)

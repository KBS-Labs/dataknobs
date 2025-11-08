"""Field type definitions and metadata for structured data records.

This module defines field types, validation, and metadata structures used by
Record objects to represent typed data fields with constraints and transformations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Callable
else:
    from typing import Callable


class FieldType(Enum):
    """Enumeration of supported field types.

    Defines the data types that can be stored in Record fields. Field types enable
    type validation, schema enforcement, and backend-specific optimizations.

    Attributes:
        STRING: Short text (< 1000 chars)
        TEXT: Long text content
        INTEGER: Whole numbers
        FLOAT: Decimal numbers
        BOOLEAN: True/False values
        DATETIME: Date and time values
        JSON: Structured JSON data (dicts, lists)
        BINARY: Binary data (bytes)
        VECTOR: Dense vector embeddings for similarity search
        SPARSE_VECTOR: Sparse vector representations

    Example:
        ```python
        from dataknobs_data import Field, FieldType

        # Create typed fields
        name_field = Field(name="name", value="Alice", type=FieldType.STRING)
        age_field = Field(name="age", value=30, type=FieldType.INTEGER)
        tags_field = Field(name="tags", value=["python", "data"], type=FieldType.JSON)

        # Auto-detection (type is inferred from value)
        auto_field = Field(name="score", value=95.5)  # Auto-detected as FLOAT
        ```
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    BINARY = "binary"
    TEXT = "text"
    VECTOR = "vector"
    SPARSE_VECTOR = "sparse_vector"


@dataclass
class Field:
    """Represents a single field in a record.

    A Field encapsulates a named value along with its type and optional metadata.
    Field types are automatically detected if not explicitly provided.

    Attributes:
        name: The field name
        value: The field value (can be any Python type)
        type: The field type (auto-detected if None)
        metadata: Optional metadata dictionary

    Example:
        ```python
        from dataknobs_data import Field, FieldType

        # Auto-detected type
        name = Field(name="name", value="Alice")
        print(name.type)  # FieldType.STRING

        # Explicit type
        score = Field(name="score", value=95.5, type=FieldType.FLOAT)

        # With metadata
        vector = Field(
            name="embedding",
            value=[0.1, 0.2, 0.3],
            type=FieldType.VECTOR,
            metadata={"dimensions": 3, "model": "text-embedding-3-small"}
        )

        # Validation
        is_valid = name.validate()  # True

        # Type conversion
        str_score = score.convert_to(FieldType.STRING)
        print(str_score.value)  # "95.5"
        ```
    """

    name: str
    value: Any
    type: FieldType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect type if not provided."""
        if self.type is None:
            self.type = self._detect_type(self.value)

    def _detect_type(self, value: Any) -> FieldType:
        """Detect the field type from the value.

        Args:
            value: The value to analyze

        Returns:
            The detected FieldType

        Example:
            ```python
            field = Field(name="data", value=[1, 2, 3])
            detected_type = field._detect_type([1, 2, 3])
            print(detected_type)  # FieldType.JSON
            ```
        """
        if value is None:
            return FieldType.STRING
        elif isinstance(value, bool):
            return FieldType.BOOLEAN
        elif isinstance(value, int):
            return FieldType.INTEGER
        elif isinstance(value, float):
            return FieldType.FLOAT
        elif isinstance(value, datetime):
            return FieldType.DATETIME
        elif isinstance(value, (dict, list)):
            return FieldType.JSON
        elif isinstance(value, bytes):
            return FieldType.BINARY
        elif isinstance(value, str):
            if len(value) > 1000:
                return FieldType.TEXT
            return FieldType.STRING
        else:
            return FieldType.JSON

    def copy(self) -> Field:
        """Create a deep copy of the field."""
        return Field(
            name=self.name,
            value=copy.deepcopy(self.value),
            type=self.type,
            metadata=copy.deepcopy(self.metadata)
        )

    def validate(self) -> bool:
        """Validate that the value matches the field type.

        Returns:
            True if the value is valid for the field type, False otherwise

        Example:
            ```python
            # Valid field
            age = Field(name="age", value=30, type=FieldType.INTEGER)
            print(age.validate())  # True

            # Invalid field (wrong type for value)
            bad_field = Field(name="count", value="not a number", type=FieldType.INTEGER)
            print(bad_field.validate())  # False
            ```
        """
        if self.value is None:
            return True

        type_validators = {
            FieldType.STRING: lambda v: isinstance(v, str),
            FieldType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            FieldType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            FieldType.BOOLEAN: lambda v: isinstance(v, bool),
            FieldType.DATETIME: lambda v: isinstance(v, datetime),
            FieldType.JSON: lambda v: isinstance(v, (dict, list)),
            FieldType.BINARY: lambda v: isinstance(v, bytes),
            FieldType.TEXT: lambda v: isinstance(v, str),
        }

        if self.type is None:
            return True
        validator = type_validators.get(self.type)
        if validator:
            return validator(self.value)
        return True

    def convert_to(self, target_type: FieldType) -> Field:
        """Convert the field to a different type.

        Args:
            target_type: The target FieldType to convert to

        Returns:
            A new Field with the converted value and type

        Raises:
            ValueError: If conversion is not possible or fails

        Example:
            ```python
            # Integer to string
            age = Field(name="age", value=30, type=FieldType.INTEGER)
            age_str = age.convert_to(FieldType.STRING)
            print(age_str.value)  # "30"

            # String to integer
            count = Field(name="count", value="42", type=FieldType.STRING)
            count_int = count.convert_to(FieldType.INTEGER)
            print(count_int.value)  # 42
            ```
        """
        if self.type == target_type:
            return self

        converters: dict[tuple[FieldType, FieldType], Callable[[Any], Any]] = {
            (FieldType.INTEGER, FieldType.STRING): str,
            (FieldType.INTEGER, FieldType.FLOAT): float,
            (FieldType.FLOAT, FieldType.STRING): str,
            (FieldType.FLOAT, FieldType.INTEGER): int,
            (FieldType.BOOLEAN, FieldType.STRING): lambda v: "true" if v else "false",
            (FieldType.BOOLEAN, FieldType.INTEGER): int,
            (FieldType.STRING, FieldType.INTEGER): int,
            (FieldType.STRING, FieldType.FLOAT): float,
            (FieldType.STRING, FieldType.BOOLEAN): lambda v: v.lower() in ("true", "1", "yes"),
            (FieldType.STRING, FieldType.TEXT): lambda v: v,
            (FieldType.TEXT, FieldType.STRING): lambda v: v,
        }

        if self.type is None:
            raise ValueError(f"Cannot convert {self.name} from None to {target_type}")
        
        converter_key = (self.type, target_type)
        if converter_key in converters:
            try:
                converter = converters[converter_key]
                new_value = converter(self.value)
                return Field(
                    name=self.name, value=new_value, type=target_type, metadata=self.metadata.copy()
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert {self.name} from {self.type} to {target_type}: {e}"
                ) from e
        else:
            raise ValueError(f"No converter available from {self.type} to {target_type}")

    def to_dict(self) -> dict[str, Any]:
        """Convert the field to a dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value if self.type else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Field:
        """Create a field from a dictionary representation."""
        field_type = None
        if data.get("type"):
            field_type = FieldType(data["type"])

        # Handle vector fields specially
        if field_type in (FieldType.VECTOR, FieldType.SPARSE_VECTOR):
            return VectorField.from_dict(data)

        return cls(
            name=data["name"],
            value=data["value"],
            type=field_type,
            metadata=data.get("metadata", {}),
        )


class VectorField(Field):
    """Represents a vector field with embeddings and metadata.

    Examples:
        # Simple usage - name optional when used in Record
        record = Record({
            "embedding": VectorField(value=[0.1, 0.2, 0.3])
        })

        # With explicit configuration
        import numpy as np
        embedding_array = np.array([0.1, 0.2, 0.3])
        field = VectorField(
            value=embedding_array,
            name="doc_embedding",
            model_name="all-MiniLM-L6-v2",
            source_field="content"
        )

        # From text using embedding function
        def my_embedding_fn(text):
            # In practice, use a real model like sentence-transformers
            return np.array([0.1, 0.2, 0.3])

        field = VectorField.from_text(
            "This is the text to embed",
            embedding_fn=my_embedding_fn
        )
    """

    def __init__(
        self,
        value: np.ndarray | list[float],
        name: str | None = None,  # Made optional
        dimensions: int | None = None,  # Auto-detected from value
        source_field: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a vector field.

        Args:
            value: Vector data as numpy array or list of floats
            name: Field name (optional, defaults to "embedding")
            dimensions: Expected dimensions (auto-detected if not provided)
            source_field: Name of the text field this vector was generated from
            model_name: Name of the embedding model used
            model_version: Version of the embedding model
            metadata: Additional metadata
        """
        # Import numpy lazily to avoid hard dependency
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "numpy is required for vector fields. Install with: pip install numpy"
            ) from e

        # Set default name if not provided
        if name is None:
            name = "embedding"

        # Convert to numpy array if needed
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            # Ensure float32 dtype for consistency
            if value.dtype != np.float32:
                value = value.astype(np.float32)
        else:
            raise TypeError(
                f"Vector value must be numpy array or list, got {type(value)}"
            )

        # Auto-detect dimensions if not provided
        actual_dims = len(value) if value.ndim == 1 else value.shape[-1]
        if dimensions is None:
            dimensions = actual_dims
        elif dimensions != actual_dims:
            raise ValueError(
                f"Vector dimension mismatch for field '{name}': "
                f"expected {dimensions}, got {actual_dims}"
            )

        # Store vector metadata
        vector_metadata = metadata or {}
        vector_metadata.update({
            "dimensions": dimensions,
            "source_field": source_field,
            "model": {
                "name": model_name,
                "version": model_version,
            } if model_name else None,
        })

        super().__init__(
            name=name,
            value=value,
            type=FieldType.VECTOR,
            metadata=vector_metadata,
        )

        self.dimensions = dimensions
        self.source_field = source_field
        self.model_name = model_name
        self.model_version = model_version

    @classmethod
    def from_text(
        cls,
        text: str,
        embedding_fn: Callable[[str], Any],
        name: str | None = None,
        dimensions: int | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        **kwargs: Any
    ) -> VectorField:
        """Create a VectorField from text using an embedding function.
        
        Args:
            text: Text to embed
            embedding_fn: Function that takes text and returns embedding vector
            name: Field name (optional, defaults to "embedding")
            dimensions: Expected dimensions (auto-detected if not provided)
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            **kwargs: Additional arguments passed to VectorField constructor
            
        Returns:
            VectorField instance with the generated embedding
            
        Example:
            field = VectorField.from_text(
                "Machine learning is fascinating",
                embedding_fn=model.encode,
                model_name="all-MiniLM-L6-v2"
            )
        """
        embedding = embedding_fn(text)
        return cls(
            value=embedding,
            name=name,
            dimensions=dimensions,
            source_field="text",  # Indicate it came from text
            model_name=model_name,
            model_version=model_version,
            **kwargs
        )

    def validate(self) -> bool:
        """Validate the vector field."""
        if self.value is None:
            return True

        try:
            import numpy as np

            if not isinstance(self.value, np.ndarray):
                return False

            if self.value.ndim not in (1, 2):
                return False

            # Check dimensions match metadata
            actual_dims = len(self.value) if self.value.ndim == 1 else self.value.shape[-1]
            expected_dims = self.metadata.get("dimensions")
            if expected_dims and actual_dims != expected_dims:
                return False

            return True
        except ImportError:
            return False

    def to_list(self) -> list[float]:
        """Convert vector to a list of floats."""
        import numpy as np

        if isinstance(self.value, np.ndarray):
            return self.value.tolist()
        return list(self.value)

    def cosine_similarity(self, other: VectorField | np.ndarray | list[float]) -> float:
        """Compute cosine similarity with another vector."""
        import numpy as np

        if isinstance(other, VectorField):
            other_vec = other.value
        elif isinstance(other, list):
            other_vec = np.array(other, dtype=np.float32)
        else:
            other_vec = other

        # Compute cosine similarity
        dot_product = np.dot(self.value, other_vec)
        norm_a = np.linalg.norm(self.value)
        norm_b = np.linalg.norm(other_vec)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def euclidean_distance(self, other: VectorField | np.ndarray | list[float]) -> float:
        """Compute Euclidean distance to another vector."""
        import numpy as np

        if isinstance(other, VectorField):
            other_vec = other.value
        elif isinstance(other, list):
            other_vec = np.array(other, dtype=np.float32)
        else:
            other_vec = other

        return float(np.linalg.norm(self.value - other_vec))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.to_list(),
            "type": self.type.value,
            "metadata": self.metadata,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorField:
        """Create from dictionary representation."""
        metadata = data.get("metadata", {})
        model_info = metadata.get("model", {})

        return cls(
            name=data["name"],
            value=data["value"],
            dimensions=data.get("dimensions") or metadata.get("dimensions"),
            source_field=metadata.get("source_field"),
            model_name=model_info.get("name") if model_info else None,
            model_version=model_info.get("version") if model_info else None,
            metadata=metadata,
        )

"""Vector field type, and the record field vocabulary re-exported.

``FieldType``, ``Field`` and the field-type registry are defined in
:mod:`dataknobs_common.fields`. They are re-exported here so that
``from dataknobs_data.fields import Field`` keeps resolving, and because
``VectorField`` -- which needs ``numpy`` and therefore cannot live in
``dataknobs-common`` -- subclasses ``Field``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataknobs_common.fields import (
    Field,
    FieldType,
    field_type_backends,
    register_field_class,
)

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Callable
else:
    from typing import Callable

__all__ = [
    "Field",
    "FieldType",
    "VectorField",
    "field_type_backends",
    "register_field_class",
]


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
            raise TypeError(f"Vector value must be numpy array or list, got {type(value)}")

        # Auto-detect dimensions if not provided
        actual_dims = len(value) if value.ndim == 1 else value.shape[-1]
        if dimensions is None:
            dimensions = actual_dims
        elif dimensions != actual_dims:
            raise ValueError(
                f"Vector dimension mismatch for field '{name}': "
                f"expected {dimensions}, got {actual_dims}"
            )

        # Store vector metadata. Copied, not adopted: the update below would
        # otherwise write three keys into the caller's dict, which is now a
        # dict callers build with `content_hash_metadata` and may reuse.
        vector_metadata = dict(metadata) if metadata else {}
        vector_metadata.update(
            {
                "dimensions": dimensions,
                "source_field": source_field,
                "model": {
                    "name": model_name,
                    "version": model_version,
                }
                if model_name
                else None,
            }
        )

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
        **kwargs: Any,
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
            **kwargs,
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
            return [float(v) for v in self.value.tolist()]
        return [float(v) for v in self.value]

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
            # Always set: `VectorField` passes `FieldType.VECTOR` to `Field`,
            # and `__post_init__` detects one for anything that does not.
            "type": self.type.value if self.type else None,
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


register_field_class(FieldType.VECTOR, VectorField)
register_field_class(FieldType.SPARSE_VECTOR, VectorField)

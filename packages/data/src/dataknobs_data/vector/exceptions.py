"""Vector-specific exceptions.

This module defines exception types for vector operations,
built on the common exception framework from dataknobs_common.
"""

from __future__ import annotations

from dataknobs_common import (
    DataknobsError,
    OperationError,
    ResourceError,
    ValidationError,
)

# Create VectorError as alias to DataknobsError for backward compatibility
VectorError = DataknobsError


class VectorDimensionError(ValidationError):
    """Raised when vector dimensions don't match expectations."""

    def __init__(self, expected: int, actual: int, field_name: str | None = None):
        """Initialize dimension error.

        Args:
            expected: Expected number of dimensions
            actual: Actual number of dimensions
            field_name: Optional field name for context
        """
        self.expected = expected
        self.actual = actual
        self.field_name = field_name

        message = f"Vector dimension mismatch: expected {expected}, got {actual}"
        if field_name:
            message = f"{message} for field '{field_name}'"

        context = {"expected": expected, "actual": actual}
        if field_name:
            context["field_name"] = field_name

        super().__init__(message, context=context)


class VectorDomainScopeError(OperationError, ValueError):
    """Raised when a scoped write would capture a row it cannot see.

    A vector store configured with a ``domain_id`` confines every
    surface to that scope, so a row belonging to another domain reads as
    absent: ``get_vectors`` returns a placeholder, ``delete_vectors``
    refuses it, ``update_metadata`` skips it. The write verbs cannot
    answer "absent" the same way. ``add_vectors`` and ``add_documents``
    upsert on id conflict, and the row they would write carries the
    configured scope — so writing an id another domain owns neither
    inserts alongside it nor edits it, but *takes* it, silently and
    without trace.

    Failing closed is the only answer that is neither a capture nor a
    silent drop: the ids are shared across domains by construction (they
    are routinely derived from content), so a collision is a real
    possibility rather than a caller error, and a store that returned
    ids it had not written would be worse than one that raised. Nothing
    in the batch is written — the check runs before the first write, so
    a partial batch cannot be left behind on the backends that have no
    transaction to roll back.

    Also subclasses ``ValueError`` so a caller with generic write-error
    handling catches it, matching ``DuplicateRecordError``.
    """

    def __init__(self, ids: list[str], domain_id: str):
        self.ids = ids
        self.domain_id = domain_id
        shown = ", ".join(repr(i) for i in ids[:5])
        if len(ids) > 5:
            shown += f", … ({len(ids)} total)"
        super().__init__(
            f"Cannot write {shown}: outside the configured domain {domain_id!r}",
            context={"ids": ids, "domain_id": domain_id},
        )


class VectorBackendError(ResourceError):
    """Raised when vector backend operations fail."""

    pass


class VectorIndexError(OperationError):
    """Raised when vector index operations fail."""

    pass


class VectorNotSupportedError(OperationError):
    """Raised when vector operations are not supported by backend."""

    def __init__(self, backend: str, operation: str | None = None):
        """Initialize not supported error.

        Args:
            backend: Name of the backend
            operation: Optional specific operation that's not supported
        """
        self.backend = backend
        self.operation = operation

        message = f"Vector operations not supported by {backend} backend"
        if operation:
            message = f"{message}: {operation}"

        context = {"backend": backend}
        if operation:
            context["operation"] = operation

        super().__init__(message, context=context)


class VectorValidationError(ValidationError):
    """Raised when vector validation fails."""

    pass

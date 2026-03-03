"""Standardized tool response format (AD-9).

All bank and catalog tools return dicts with a consistent structure:

- Success: ``{"success": True, ...additional_fields}``
- Error: ``{"success": False, "error": "message"}``
- Error with details: ``{"success": False, "error": "message", "details": [...]}``

Example::

    from dataknobs_bots.tools.response import error_response, success_response

    return success_response(record_id="abc123", data={"name": "flour"})
    return error_response("Missing required parameter: bank_name")
    return error_response("Validation failed", details=["name is required"])
"""

from __future__ import annotations

from typing import Any


def success_response(**data: Any) -> dict[str, Any]:
    """Create a standardized success response.

    Args:
        **data: Additional fields to include in the response.

    Returns:
        Dict with ``"success": True`` plus any additional fields.
    """
    return {"success": True, **data}


def error_response(
    error: str,
    details: list[str] | None = None,
    **data: Any,
) -> dict[str, Any]:
    """Create a standardized error response.

    Args:
        error: Human-readable error message.
        details: Optional list of specific error details.
        **data: Additional fields to include in the response.

    Returns:
        Dict with ``"success": False``, ``"error"``, and optionally ``"details"``
        plus any additional fields.
    """
    result: dict[str, Any] = {"success": False, "error": error}
    if details is not None:
        result["details"] = details
    result.update(data)
    return result

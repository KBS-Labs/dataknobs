"""Custom exceptions and exception handlers for FastAPI applications.

This module provides a consistent exception hierarchy and handlers
for bot-related API errors. The exceptions extend from dataknobs_common
for consistency across the codebase.

Example:
    ```python
    from fastapi import FastAPI
    from dataknobs_bots.api.exceptions import (
        register_exception_handlers,
        BotNotFoundError,
    )

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/bots/{bot_id}")
    async def get_bot(bot_id: str):
        bot = await manager.get(bot_id)
        if not bot:
            raise BotNotFoundError(bot_id)
        return {"bot_id": bot_id}
    ```
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from dataknobs_common.exceptions import (
    ConfigurationError as CommonConfigurationError,
)
from dataknobs_common.exceptions import (
    DataknobsError,
)
from dataknobs_common.exceptions import (
    NotFoundError as CommonNotFoundError,
)
from dataknobs_common.exceptions import (
    ValidationError as CommonValidationError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse


class APIError(DataknobsError):
    """Base exception for API errors.

    Extends DataknobsError to provide HTTP-specific error handling
    with status codes and structured error responses.

    Attributes:
        message: Error message
        status_code: HTTP status code
        detail: Error details (maps to DataknobsError.context)
        error_code: Machine-readable error code
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: dict[str, Any] | None = None,
        error_code: str | None = None,
    ):
        """Initialize API error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code (default: 500)
            detail: Optional dictionary with error details
            error_code: Optional machine-readable error code
        """
        # Pass detail as context to DataknobsError
        super().__init__(message, context=detail)
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__

    @property
    def detail(self) -> dict[str, Any]:
        """Alias for context to maintain API compatibility."""
        return self.context

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for JSON response.

        Returns:
            Dictionary representation of the error
        """
        return {
            "error": self.error_code,
            "message": str(self),
            "detail": self.context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class BotNotFoundError(APIError, CommonNotFoundError):
    """Exception raised when bot instance is not found."""

    def __init__(self, bot_id: str):
        APIError.__init__(
            self,
            message=f"Bot with ID '{bot_id}' not found",
            status_code=404,
            detail={"bot_id": bot_id},
        )


class BotCreationError(APIError):
    """Exception raised when bot creation fails."""

    def __init__(self, bot_id: str, reason: str):
        super().__init__(
            message=f"Failed to create bot '{bot_id}': {reason}",
            status_code=500,
            detail={"bot_id": bot_id, "reason": reason},
        )


class ConversationNotFoundError(APIError, CommonNotFoundError):
    """Exception raised when conversation is not found."""

    def __init__(self, conversation_id: str):
        APIError.__init__(
            self,
            message=f"Conversation with ID '{conversation_id}' not found",
            status_code=404,
            detail={"conversation_id": conversation_id},
        )


class ValidationError(APIError, CommonValidationError):
    """Exception raised when input validation fails."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None):
        APIError.__init__(
            self,
            message=message,
            status_code=422,
            detail=detail,
        )


class ConfigurationError(APIError, CommonConfigurationError):
    """Exception raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str | None = None):
        detail = {}
        if config_key:
            detail["config_key"] = config_key
        APIError.__init__(
            self,
            message=message,
            status_code=500,
            detail=detail,
        )


class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ):
        detail = {}
        if retry_after:
            detail["retry_after"] = retry_after
        super().__init__(
            message=message,
            status_code=429,
            detail=detail,
        )


# Exception Handlers
# Note: These use TYPE_CHECKING imports to avoid requiring FastAPI at import time


async def api_error_handler(
    request: Request,  # type: ignore[name-defined]
    exc: APIError,
) -> JSONResponse:  # type: ignore[name-defined]
    """Handle API errors with standardized response format.

    Args:
        request: FastAPI request object
        exc: API error exception

    Returns:
        JSON response with error details
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def http_exception_handler(
    request: Request,  # type: ignore[name-defined]
    exc: HTTPException,  # type: ignore[name-defined]
) -> JSONResponse:  # type: ignore[name-defined]
    """Handle FastAPI HTTP exceptions.

    Args:
        request: FastAPI request object
        exc: HTTP exception

    Returns:
        JSON response with error details
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": str(exc.detail),
            "detail": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def general_exception_handler(
    request: Request,  # type: ignore[name-defined]
    exc: Exception,
) -> JSONResponse:  # type: ignore[name-defined]
    """Handle unexpected exceptions.

    Args:
        request: FastAPI request object
        exc: Generic exception

    Returns:
        JSON response with error details

    Note:
        This handler logs the full exception but returns a generic
        message to avoid leaking internal details.
    """
    import logging

    from fastapi.responses import JSONResponse

    logger = logging.getLogger(__name__)
    logger.exception(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": {"exception_type": type(exc).__name__},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def register_exception_handlers(
    app: FastAPI,  # type: ignore[name-defined]
) -> None:
    """Register all exception handlers with a FastAPI app.

    Args:
        app: FastAPI application instance

    Example:
        ```python
        from fastapi import FastAPI
        from dataknobs_bots.api.exceptions import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)
        ```
    """
    from fastapi import HTTPException

    app.add_exception_handler(APIError, api_error_handler)  # type: ignore
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore
    app.add_exception_handler(Exception, general_exception_handler)

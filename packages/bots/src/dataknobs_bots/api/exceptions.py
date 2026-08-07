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

import logging
import math
from datetime import UTC, datetime
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
    OperationError as CommonOperationError,
)
from dataknobs_common.exceptions import (
    RateLimitError as CommonRateLimitError,
)
from dataknobs_common.exceptions import (
    ValidationError as CommonValidationError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


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
            "timestamp": datetime.now(UTC).isoformat(),
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


class BotCreationError(APIError, CommonOperationError):
    """Exception raised when bot creation fails.

    Subclasses the common ``OperationError`` for the same reason the API
    ``RateLimitError`` subclasses the common one: creating a bot is an
    operation, and failing to create it is an operation failure. There is no
    same-named counterpart to reach for, so a consumer catching
    ``OperationError`` has no way to discover that this particular failure is
    excluded from it.
    """

    def __init__(self, bot_id: str, reason: str):
        APIError.__init__(
            self,
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


class RateLimitError(APIError, CommonRateLimitError):
    """Exception raised when rate limit is exceeded.

    Subclasses the common ``RateLimitError`` so that
    ``except dataknobs_common.exceptions.RateLimitError`` catches both this
    API-layer variant and the one
    ``dataknobs_llm.conversations.middleware.RateLimitMiddleware`` raises —
    matching every other twinned pair in this module.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
    ):
        detail: dict[str, Any] = {}
        # `is not None`, not truthiness: zero is a legitimate retry hint
        # ("try again now"), and rate limiters do report it — a drained
        # window yields `reset_after=0.0`. Under a truthiness test the
        # attribute keeps the zero while the response body loses the field
        # entirely, so the two views of one value disagree.
        if retry_after is not None:
            detail["retry_after"] = retry_after
        APIError.__init__(
            self,
            message=message,
            status_code=429,
            detail=detail,
        )
        # Must follow APIError.__init__: with the widened base list, that
        # call reaches CommonRateLimitError.__init__ through the MRO, which
        # sets self.retry_after = None (its own default). Assigning before
        # the call would be silently clobbered.
        self.retry_after = retry_after


# Exception Handlers
# Note: These use TYPE_CHECKING imports to avoid requiring FastAPI at import time


def _retry_after_headers(exc: APIError) -> dict[str, str] | None:
    """Build the ``Retry-After`` header for an error that carries a hint.

    Returns ``None`` when the exception has no hint, so the header is omitted
    rather than defaulted — emitting a made-up wait would assert something
    the server never computed.

    RFC 7231 defines the value as delay-seconds: a non-negative integer. The
    rate limiters report a float, so a fractional wait rounds *up* (rounding
    down returns the client while it is still throttled) and a negative wait
    clamps to zero (``Retry-After: -5`` is unparseable, and a client that
    gives up on parsing may simply retry at once).
    """
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        return None
    return {"Retry-After": str(max(0, math.ceil(retry_after)))}


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

    Note:
        An exception carrying a ``retry_after`` hint also gets a
        ``Retry-After`` header. ``detail.retry_after`` is this project's own
        JSON shape and nothing outside it knows to look there, whereas the
        header is what HTTP clients, proxies, and SDK retry policies already
        act on — and RFC 6585 says a 429 SHOULD carry one.
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
        headers=_retry_after_headers(exc),
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
            "timestamp": datetime.now(UTC).isoformat(),
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
    from fastapi.responses import JSONResponse

    # Lazy `%s`, not an f-string: the interpolated form is evaluated before
    # the logging call and discards the exception object, which is what
    # carries the traceback `logger.exception` is here to record.
    logger.exception("Unhandled exception: %s", exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": {"exception_type": type(exc).__name__},
            "timestamp": datetime.now(UTC).isoformat(),
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

    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

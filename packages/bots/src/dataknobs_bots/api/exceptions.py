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

Handling DataKnobs' own errors:
    ``register_exception_handlers`` also registers a handler for
    ``dataknobs_common.exceptions.DataknobsError``, so an error raised
    anywhere in the stack — not only from API code — returns a status
    matching the failure rather than a generic 500. The mapping is
    ``DEFAULT_ERROR_POLICY``, resolved by walking the exception's MRO, so a
    subclass this table has never heard of inherits the policy of its nearest
    listed ancestor instead of falling through.

    Each entry also decides whether the error is disclosed to the caller. A
    ``client_safe=True`` entry returns the error's message *and* its
    ``context``; a ``False`` entry returns a generic message and an empty
    ``detail``, and the diagnostic goes to the log instead.

    **Adding a row means reading what that type's raise sites put in
    ``context``, not only what its message says.** ``client_safe`` is a single
    bit gating both, so a type whose message is harmless but whose context can
    carry a query string or a credential is not client-safe.

    A deployment overrides any row, including for its own subclasses:

    ```python
    from dataknobs_bots.api.exceptions import ErrorPolicy
    from dataknobs_common.exceptions import ConfigurationError

    register_exception_handlers(
        app, error_policy={ConfigurationError: ErrorPolicy(500, False)}
    )
    ```
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

from dataknobs_common.exceptions import (
    ConcurrencyError as CommonConcurrencyError,
)
from dataknobs_common.exceptions import (
    ConfigurationError as CommonConfigurationError,
)
from dataknobs_common.exceptions import (
    ConsentRequiredError as CommonConsentRequiredError,
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
    ResourceError as CommonResourceError,
)
from dataknobs_common.exceptions import (
    SerializationError as CommonSerializationError,
)
from dataknobs_common.exceptions import (
    TimeoutError as CommonTimeoutError,
)
from dataknobs_common.exceptions import (
    ValidationError as CommonValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

#: What a masked error tells the caller. Shared by ``general_exception_handler``
#: and by any policy entry with ``client_safe=False``, so the two cannot drift
#: into describing the same situation differently.
MASKED_MESSAGE = "An unexpected error occurred"


def _error_body(
    *,
    error_code: str,
    message: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON body shape every error response in this module returns.

    One function so the four handlers cannot disagree about the shape. They
    did before this existed: three built it inline and the fourth was about to.

    Args:
        error_code: Machine-readable error identifier.
        message: Human-readable message, already masked if it needed to be.
        detail: Structured error detail, already masked if it needed to be.

    Returns:
        The response body dictionary.
    """
    return {
        "error": error_code,
        "message": message,
        "detail": detail,
        "timestamp": datetime.now(UTC).isoformat(),
    }


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
        return _error_body(
            error_code=self.error_code,
            message=str(self),
            detail=self.context,
        )


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


# Error Policy
# How a DataknobsError is rendered at the HTTP boundary


class ErrorPolicy(NamedTuple):
    """How a ``DataknobsError`` type is rendered at the HTTP boundary.

    Attributes:
        status_code: The HTTP status the error maps to.
        client_safe: Whether the error's message and ``context`` are returned
            to the caller. When ``False`` the response carries
            ``MASKED_MESSAGE`` and an empty ``detail``, and the diagnostic is
            logged instead.
    """

    status_code: int
    client_safe: bool


#: The default type -> policy mapping, resolved by MRO walk (see
#: :func:`resolve_error_policy`). Eleven entries govern more than fifty
#: reachable classes: every ``DataknobsError`` subclass the other packages
#: define resolves to its nearest listed ancestor, so ``RecordNotFoundError``
#: returns 404 without appearing here. An exact-type table would cover the
#: eleven and silently 500 the rest -- which is the behaviour this handler
#: replaces.
#:
#: Ordered by tier rather than alphabetically, because the tier is what a
#: reviewer has to check. Nothing about the literal order is load-bearing:
#: resolution is by MRO, which is why ``RateLimitError`` wins over its own base
#: ``OperationError`` regardless of where either sits in this dict.
DEFAULT_ERROR_POLICY: dict[type[DataknobsError], ErrorPolicy] = {
    # --- client-safe: message and context are authored for the caller ---
    # context: field names and offending values -- the caller's own input.
    CommonValidationError: ErrorPolicy(422, True),
    # context: the id the caller supplied, plus the table/collection name.
    CommonNotFoundError: ErrorPolicy(404, True),
    # context: `scope`, which the caller must act on, and `user_id`. That id is
    # the caller's own, so returning it to them discloses nothing new -- unless
    # a route attempts access on behalf of a third party, where it would.
    CommonConsentRequiredError: ErrorPolicy(403, True),
    # context: expected and actual versions -- what the caller retries against.
    CommonConcurrencyError: ErrorPolicy(409, True),
    # context: category, limit, interval. Also carries the `retry_after` hint,
    # which reaches the client as a header as well (see `_retry_after_headers`).
    CommonRateLimitError: ErrorPolicy(429, True),
    # context: the timeout value -- but this type's own documented example puts
    # the SQL query there too, so a raiser following it discloses the query.
    # The single strongest case for splitting message-safety from
    # context-safety, which is one bit here (see the module docstring).
    CommonTimeoutError: ErrorPolicy(504, True),
    # context: the config key and the available keys. A config error is about
    # the deployment's own config and is authored for whoever wrote it, so it
    # is disclosed -- but a route serving unauthenticated traffic should say
    # otherwise via `error_policy=` (see the module docstring). The status stays
    # 500: a bad config is a server-side fault however readable we make it.
    CommonConfigurationError: ErrorPolicy(500, True),
    # --- masked: the message may embed infrastructure detail ---
    # e.g. a connection string, with credentials, from a failed connect.
    CommonResourceError: ErrorPolicy(503, False),
    CommonSerializationError: ErrorPolicy(500, False),
    CommonOperationError: ErrorPolicy(500, False),
    # Terminal fallback. Here as a row rather than a `.get` default so the
    # exhaustiveness guard has nothing to special-case and a consumer can
    # override it like any other entry.
    DataknobsError: ErrorPolicy(500, False),
}


def resolve_error_policy(
    exc: DataknobsError,
    policy: Mapping[type[DataknobsError], ErrorPolicy] | None = None,
) -> ErrorPolicy:
    """Resolve the policy for an error by walking its MRO.

    The first class in ``type(exc).__mro__`` present in the table wins — the
    same rule Starlette uses to pick a handler, so a subclass DataKnobs has
    never heard of inherits its nearest listed ancestor's policy rather than
    falling through to a masked 500.

    Args:
        exc: The error to resolve a policy for.
        policy: Optional table to resolve against, replacing the default.

    Returns:
        The resolved policy.
    """
    table = policy if policy is not None else DEFAULT_ERROR_POLICY
    for cls in type(exc).__mro__:
        if cls in table:
            return table[cls]
    # Unreachable while DataknobsError is in the table; kept as a fail-closed
    # guard for a consumer-supplied table that omits it.
    return ErrorPolicy(500, False)


# Exception Handlers
# Note: These use TYPE_CHECKING imports to avoid requiring FastAPI at import time


def _retry_after_headers(exc: DataknobsError) -> dict[str, str] | None:
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


def _error_response(
    *,
    status_code: int,
    content: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> JSONResponse:  # type: ignore[name-defined]
    """Build the JSON response every handler in this module returns.

    Split from :func:`_error_body` so ``api_error_handler`` can keep routing
    through :meth:`APIError.to_dict` — that method is public and a consumer
    subclass may override it, which building the body inline here would
    silently stop honouring.

    Args:
        status_code: HTTP status for the response.
        content: The response body, from :func:`_error_body` or ``to_dict()``.
        headers: Optional response headers.

    Returns:
        The JSON response.
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content=content, headers=headers)


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
    return _error_response(
        status_code=exc.status_code,
        content=exc.to_dict(),
        headers=_retry_after_headers(exc),
    )


async def dataknobs_error_handler(
    request: Request,  # type: ignore[name-defined]
    exc: DataknobsError,
    *,
    error_policy: Mapping[type[DataknobsError], ErrorPolicy] | None = None,
) -> JSONResponse:  # type: ignore[name-defined]
    """Handle a DataKnobs error according to its resolved policy.

    Without this handler every ``DataknobsError`` that is not an ``APIError``
    reaches the ``Exception`` catch-all and returns
    ``500 / "An unexpected error occurred"`` with its message and ``context``
    discarded — including errors, like a configuration diagnostic, whose whole
    value is the message DataKnobs generated one layer earlier.

    Args:
        request: FastAPI request object
        exc: The DataKnobs error
        error_policy: Optional table to resolve against, replacing the default.
            ``register_exception_handlers`` binds the merged table here.

    Returns:
        JSON response with the error's status, disclosed per its policy
    """
    policy = resolve_error_policy(exc, error_policy)

    if policy.client_safe:
        logger.warning(
            "%s handled as HTTP %d: %s",
            type(exc).__name__,
            policy.status_code,
            exc,
        )
    else:
        # The response discloses nothing, so this line is the only place the
        # diagnostic survives — context included, since that is the half the
        # caller never sees. Detailed server-side, generic client-side.
        logger.exception(
            "%s handled as HTTP %d (masked in response): %s | context=%s",
            type(exc).__name__,
            policy.status_code,
            exc,
            exc.context,
        )

    detail: dict[str, Any] = dict(exc.context) if policy.client_safe else {}
    retry_after = getattr(exc, "retry_after", None)
    if policy.client_safe and retry_after is not None:
        # The common RateLimitError keeps retry_after as an attribute only,
        # while the API twin also writes it into context — so without this the
        # same condition yields detail.retry_after from one variant and an
        # empty detail from the other. setdefault, so a raiser who deliberately
        # put a different value in context keeps theirs.
        detail.setdefault("retry_after", retry_after)

    return _error_response(
        status_code=policy.status_code,
        content=_error_body(
            error_code=type(exc).__name__,
            message=str(exc) if policy.client_safe else MASKED_MESSAGE,
            detail=detail,
        ),
        # Emitted regardless of client_safe: a retry hint is a flow-control
        # signal, not a diagnostic, and withholding it from a masked error
        # would break the caller's back-off for no disclosure gained.
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
    return _error_response(
        status_code=exc.status_code,
        content=_error_body(
            error_code="HTTPException",
            message=str(exc.detail),
            detail={},
        ),
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

        DataKnobs' own errors no longer arrive here — ``dataknobs_error_handler``
        takes them, and gives them a status matching the failure.
    """
    # Lazy `%s`, not an f-string: the interpolated form is evaluated before
    # the logging call and discards the exception object, which is what
    # carries the traceback `logger.exception` is here to record.
    logger.exception("Unhandled exception: %s", exc)

    return _error_response(
        status_code=500,
        content=_error_body(
            error_code="InternalServerError",
            message=MASKED_MESSAGE,
            detail={"exception_type": type(exc).__name__},
        ),
    )


def register_exception_handlers(
    app: FastAPI,  # type: ignore[name-defined]
    *,
    error_policy: Mapping[type[DataknobsError], ErrorPolicy] | None = None,
) -> None:
    """Register all exception handlers with a FastAPI app.

    Args:
        app: FastAPI application instance
        error_policy: Optional per-type overrides merged over
            ``DEFAULT_ERROR_POLICY``. Use it to add a policy for the
            deployment's own ``DataknobsError`` subclasses, or to disagree with
            a default — most usefully to mask ``ConfigurationError`` on a route
            serving unauthenticated traffic.

    Example:
        ```python
        from fastapi import FastAPI
        from dataknobs_bots.api.exceptions import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)
        ```

    Note:
        Handlers resolve by MRO, not by registration order, so the order below
        is only how it reads. ``APIError`` precedes ``DataknobsError`` in every
        API exception's MRO, which is what keeps ``api_error_handler``
        responsible for them.
    """
    from fastapi import HTTPException

    table = dict(DEFAULT_ERROR_POLICY)
    if error_policy:
        table.update(error_policy)

    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(DataknobsError, partial(dataknobs_error_handler, error_policy=table))
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

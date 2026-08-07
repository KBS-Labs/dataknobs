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
    ``dataknobs_common.exceptions.DataknobsError``, so an error raised at any
    depth under a route — not only in API code — returns a status matching the
    failure rather than a generic 500. The mapping is ``DEFAULT_ERROR_POLICY``,
    resolved by walking the exception's MRO, so a subclass this table has never
    heard of inherits the policy of its nearest listed ancestor instead of
    falling through.

    *Under a route*, not anywhere in the ASGI stack. Starlette builds
    ``ServerErrorMiddleware`` → user middleware → ``ExceptionMiddleware`` →
    router, and only ``ExceptionMiddleware`` consults the per-type handlers
    these register. An error raised in an ``app.add_middleware`` layer is above
    that, reaches ``ServerErrorMiddleware`` — which holds the ``Exception``
    handler alone — and comes back as a generic 500. This applies to
    :class:`APIError` equally. Middleware that wants a status should return the
    response rather than raise::

        try:
            ...
        except APIError as exc:
            return await api_error_handler(request, exc)

    Each entry also decides what is disclosed to the caller, in two parts:
    ``disclose_message`` returns the error's message, ``disclose_context``
    returns its ``context`` as ``detail``. A withheld half is replaced by
    ``MASKED_MESSAGE`` or ``{}``. Both halves are logged either way.

    **Adding a row means reading what that type's raise sites put in
    ``context``, not only what its message says** — the two halves disagree
    about which is safe, and in both directions. ``NotFoundError``'s message
    is the caller's own key echoed back while its ``context`` enumerates a
    registry's whole keyspace; ``ValidationError``'s ``context`` is the
    caller's own fields while its message can be a database driver's.

    A deployment overrides any row, including for its own subclasses:

    ```python
    from dataknobs_bots.api.exceptions import ErrorPolicy
    from dataknobs_common.exceptions import ConfigurationError

    register_exception_handlers(
        app, error_policy={ConfigurationError: ErrorPolicy(500, True, True)}
    )
    ```

    The :class:`APIError` family does not resolve through this table at all —
    see that class for why, and for the single ``client_safe`` bit it uses
    instead.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

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
from dataknobs_common.transitions import InvalidTransitionError

if TYPE_CHECKING:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

#: What a withheld message tells the caller. Shared by
#: ``general_exception_handler`` and by any policy entry with
#: ``disclose_message=False``, so the two cannot drift into describing the same
#: situation differently.
MASKED_MESSAGE = "An unexpected error occurred"

#: How deep :func:`_jsonable` walks before stringifying whatever it is looking
#: at. Bounds the cost of a deeply nested ``context`` and terminates a cyclic
#: one — ``json.dumps`` rejects a cycle outright, and an unbounded walk would
#: recurse until the stack ran out. Matches the depth bound
#: ``StructuredConfig`` uses for the same reason.
_MAX_JSON_DEPTH = 6


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Coerce a response body into something the JSON encoder accepts.

    Starlette renders with ``json.dumps(..., allow_nan=False)``, so a value it
    cannot encode raises *inside* the handler. Starlette's error middleware
    then catches that and returns the generic 500 these handlers exist to
    replace: a 404 whose ``context`` carried a ``Path`` came back as
    ``500 / "An unexpected error occurred"``, losing both the status and the
    message. ``context`` is a free ``dict[str, Any]`` and raise sites fill it
    with whatever the failure was about, so this is not an exotic input.

    Unknown values become ``str(value)`` rather than being expanded, which is
    a disclosure decision as much as a simplicity one:
    ``fastapi.encoders.jsonable_encoder`` falls back to ``dict(obj)`` and then
    ``vars(obj)``, which would put an object's whole attribute dict into a
    response body the raiser only meant to carry the object. ``str`` discloses
    what the type chose to say — for a ``StructuredConfig`` that is a repr
    with its secrets already redacted.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        # NaN and the infinities are not JSON, and `allow_nan=False` makes
        # that an error rather than the non-standard literal Python emits.
        return value if math.isfinite(value) else str(value)
    if _depth >= _MAX_JSON_DEPTH:
        return str(value)
    if isinstance(value, Mapping):
        # Keys too: json.dumps coerces int and float keys but rejects the rest.
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(v, _depth + 1) for v in value]
    return str(value)


def _disclosure_label(message: bool, context: bool) -> str:
    """Render what the caller was actually shown, for the log line.

    Both handlers log the full diagnostic whatever the policy says, so the log
    alone cannot answer "the client reports an empty error — is that the
    policy or a bug?". This puts the answer on the same line.
    """
    shown = [
        half for half, on in (("message", message), ("context", context)) if on
    ]
    return "+".join(shown) if shown else "nothing"


def _log_handled_error(
    exc: Exception, status_code: int, template: str, *args: Any
) -> None:
    """Log an error one of these handlers turned into a response.

    The level follows the status class, not the disclosure bit. A 404 or a 422
    is the caller's problem and a routine outcome of serving traffic, so
    logging it at ``warning`` made a working service look like a failing one
    and buried the 5xx that do need attention. A 5xx is the server's problem
    and is worth a traceback, which is live here because Starlette calls
    handlers from inside its ``except`` block.

    *What* is logged stays with the caller, because that is the half
    disclosure decides: a masked error's message and ``context`` appear
    nowhere else, so its line carries both.

    Except for ``__cause__``, which is appended here for every handler. A 5xx
    already got it free with the traceback; a 4xx did not, and a 4xx is exactly
    where a deliberately thin message shows up — a library that wraps a
    failure it must not disclose puts the real one on ``__cause__`` and leaves
    the message naming only what is safe. Without this the log repeats the
    thin message and the diagnosis is nowhere.
    """
    cause = exc.__cause__
    if cause is not None:
        template += " | cause=%s: %s"
        args = (*args, type(cause).__name__, cause)
    if status_code >= 500:
        logger.exception(template, *args)
    else:
        logger.info(template, *args)


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

    This family does not resolve through :data:`DEFAULT_ERROR_POLICY`:
    ``APIError`` precedes every common base in these classes' MROs, so
    Starlette hands them to :func:`api_error_handler` and the table's
    disclosure bits never see them. That is deliberate — these are the one
    family authored *for* the HTTP boundary, carrying a per-instance
    ``status_code`` rather than a per-type one and a public, overridable
    ``to_dict()``. :attr:`client_safe` is the table's disclosure decision for
    this family, declared per class.

    It stays **one** bit where the table's is two, for two reasons. A subclass
    writes its message and its ``detail`` in the same constructor for the same
    audience, so the halves have one author — unlike a table row, which
    governs a type raised across several packages by people not thinking about
    HTTP. And ``to_dict()`` returns an arbitrary dict a subclass may have
    extended, so disclosing part of it could only mean allow-listing keys,
    which would silently drop whatever an override added. A subclass wanting
    message-only authors the message and leaves ``detail`` empty.

    Attributes:
        client_safe: Whether this class's message and ``detail`` are returned
            to the caller. ``True`` here, because a class written for the HTTP
            boundary is written to be shown — a consumer's own subclass
            inherits that without opting in. A subclass whose message is built
            from text it does not control sets it ``False``; see
            :class:`BotCreationError`.
        message: Error message
        status_code: HTTP status code
        detail: Error details (maps to DataknobsError.context)
        error_code: Machine-readable error code
    """

    client_safe: ClassVar[bool] = True

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

    The only member of the family that is **not** client-safe. Its whole
    payload is one free-text ``reason``: the others put the authored part in
    ``detail`` or ``config_key`` and keep the caller's own input in the
    message, so what they disclose is bounded by construction. ``reason`` is
    not, and the pattern this package documented for it was
    ``raise BotCreationError(bot_id, str(e))``. Bots are built lazily on the
    request path, and the tool and middleware factories wrap
    ``except Exception`` into a message ending in ``{e}`` — so a tool whose
    constructor opens a database or a cache put the driver's error text, URL
    and credentials included, into an HTTP response body.

    Masking is per class rather than per raise site because the raise sites
    are the consumer's. A deployment that authors its own ``reason`` and wants
    it shown subclasses and sets ``client_safe = True``.
    """

    client_safe: ClassVar[bool] = False

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

    Message and ``context`` are gated separately because the types disagree
    about which half is safe, in both directions. ``NotFoundError``'s message
    is the caller's own key echoed back while its ``context`` enumerates the
    whole registry keyspace; ``ValidationError``'s ``context`` is the caller's
    own field names and values while its message can be a database driver's.
    One bit cannot serve both, and collapsing them forces a row to give up a
    useful half to withhold an unsafe one.

    Attributes:
        status_code: The HTTP status the error maps to.
        disclose_message: Whether the error's message is returned to the
            caller. When ``False`` the response carries ``MASKED_MESSAGE``.
        disclose_context: Whether the error's ``context`` is returned as
            ``detail``. When ``False`` the response carries ``{}``. Defaults to
            ``False`` so a row written without thinking about it fails closed,
            and so a two-argument ``ErrorPolicy(500, False)`` still reads as
            the fully-masked policy it looks like.

    Either way the full diagnostic — message and context — is logged.
    """

    status_code: int
    disclose_message: bool
    disclose_context: bool = False


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
#:
#: Read-only. This is process-global, so a consumer assigning into it would be
#: changing the disclosure policy of every app in the process — including ones
#: already registered, since ``resolve_error_policy`` reads it per request. The
#: supported route is ``register_exception_handlers(error_policy=...)``, which
#: is per app.
DEFAULT_ERROR_POLICY: Mapping[type[DataknobsError], ErrorPolicy] = MappingProxyType({
    # --- fully disclosed: message and context are both authored for the caller
    # context: field names and offending values -- the caller's own input.
    CommonValidationError: ErrorPolicy(422, True, True),
    # context: `scope`, which the caller must act on, and `user_id`. That id is
    # the caller's own, so returning it to them discloses nothing new -- unless
    # a route attempts access on behalf of a third party, where it would.
    CommonConsentRequiredError: ErrorPolicy(403, True, True),
    # context: expected and actual versions -- what the caller retries against.
    CommonConcurrencyError: ErrorPolicy(409, True, True),
    # The one row here that overrides an inherited status rather than declaring
    # a base's. `InvalidTransitionError` is an `OperationError`, which is right
    # for a library -- an invalid transition is permanent, so retry logic keyed
    # on that base correctly declines to re-attempt it -- but it inherited 500,
    # blaming the server for the caller's mistake. "Cannot go from `draft` to
    # `shipped`" is the textbook 409: the request conflicts with the resource's
    # current state and would succeed in another one. Rebasing the type onto
    # `ConcurrencyError` would have bought the same status and broken the retry
    # semantics, which is exactly the split this table exists to express.
    # context: entity, current and target status, and `allowed` -- the remedy.
    InvalidTransitionError: ErrorPolicy(409, True, True),
    # context: category, limit, interval. Also carries the `retry_after` hint,
    # which reaches the client as a header as well (see `_retry_after_headers`).
    CommonRateLimitError: ErrorPolicy(429, True, True),
    # --- message only: the diagnostic is the caller's, the context is not ---
    # message: "Item not found: <the key the caller asked for>". context:
    # `available_keys`, the registry's entire keyspace -- a "did you mean" for
    # a library caller and an inventory listing for an HTTP one.
    CommonNotFoundError: ErrorPolicy(404, True, False),
    # message: that something timed out, and usually after how long. context:
    # the timeout value -- but this type's own documented example puts the SQL
    # query there too, so a raiser following it disclosed the query.
    CommonTimeoutError: ErrorPolicy(504, True, False),
    # Masked, and the one row where that is a judgement call rather than a
    # reading of the type. Most config diagnostics are authored -- a key name,
    # a sorted list of the valid ones -- and are exactly what a deployment
    # wants back. But this type is also where the funnels that wrap a
    # third-party *constructor* or *module import* land, and that text is
    # unbounded: a database or cache client raises with its connection URL,
    # credentials included. Those funnels are bounded in-tree (they name the
    # class path and the exception type, and let `__cause__` carry the rest),
    # yet a deployment cannot audit its consumers' raise sites, and this
    # package builds bots lazily on the request path. So the default is
    # closed and the diagnostic goes to the log. Turn it back on per app --
    # `error_policy={ConfigurationError: ErrorPolicy(500, True, True)}` --
    # when the route is not public. The status stays 500 either way: a bad
    # config is a server-side fault however readable we make it.
    CommonConfigurationError: ErrorPolicy(500, False, False),
    # --- fully masked: the message may embed infrastructure detail ---
    # e.g. a connection string, with credentials, from a failed connect.
    CommonResourceError: ErrorPolicy(503, False, False),
    CommonSerializationError: ErrorPolicy(500, False, False),
    CommonOperationError: ErrorPolicy(500, False, False),
    # Terminal fallback. Here as a row rather than a `.get` default so the
    # exhaustiveness guard has nothing to special-case and a consumer can
    # override it like any other entry.
    DataknobsError: ErrorPolicy(500, False, False),
})


def resolve_error_policy(
    exc: DataknobsError,
    table: Mapping[type[DataknobsError], ErrorPolicy] | None = None,
) -> ErrorPolicy:
    """Resolve the policy for an error by walking its MRO.

    The first class in ``type(exc).__mro__`` present in the table wins — the
    same rule Starlette uses to pick a handler, so a subclass DataKnobs has
    never heard of inherits its nearest listed ancestor's policy rather than
    falling through to a masked 500.

    Args:
        exc: The error to resolve a policy for.
        table: The whole table to resolve against, *replacing*
            :data:`DEFAULT_ERROR_POLICY`. Named for what it is because
            ``register_exception_handlers``' ``error_policy=`` is the other
            thing — a set of overrides *merged over* the defaults — and two
            parameters that read alike but compose differently is how a
            deployment ends up with one row and no fallback. The merged result
            is what the registered handler passes back in here.

    Returns:
        The resolved policy.
    """
    lookup = DEFAULT_ERROR_POLICY if table is None else table
    for cls in type(exc).__mro__:
        if cls in lookup:
            return lookup[cls]
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

    The body passes through :func:`_jsonable` here rather than in each
    handler, so a consumer's overridden ``to_dict`` is covered by the same
    guarantee as a body this module built.

    Args:
        status_code: HTTP status for the response.
        content: The response body, from :func:`_error_body` or ``to_dict()``.
        headers: Optional response headers.

    Returns:
        The JSON response.
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code, content=_jsonable(content), headers=headers
    )


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

        A class declaring ``client_safe = False`` is rendered like a fully
        masked row of the policy table: ``MASKED_MESSAGE``, an empty
        ``detail``, and the diagnostic logged instead. The status still comes
        from the instance, and the ``Retry-After`` header is still emitted,
        for the same reason the table's masked branch emits it — a retry hint
        is flow control, not a diagnostic.

        This family's gate stays one bit where the table's is two. An
        ``APIError`` subclass writes its message and its ``detail`` in the
        same constructor, for the same audience, so the two halves have one
        author — unlike a table row, which governs a type raised across
        several packages by people not thinking about HTTP. And ``to_dict()``
        is overridable and returns an arbitrary dict, so disclosing *part* of
        it could only mean allow-listing keys, which would silently drop a key
        an override added. Whole-or-nothing is the fail-closed reading. A
        subclass wanting message-only authors the message and leaves ``detail``
        empty.
    """
    safe = type(exc).client_safe
    # Logged whole either way: when masked this is the only place the
    # diagnostic survives, and when disclosed the log is still the record.
    _log_handled_error(
        exc,
        exc.status_code,
        "%s handled as HTTP %d (disclosed: %s): %s | detail=%s",
        type(exc).__name__,
        exc.status_code,
        _disclosure_label(safe, safe),
        exc,
        exc.detail,
    )

    return _error_response(
        status_code=exc.status_code,
        content=exc.to_dict()
        if safe
        else _error_body(
            error_code=type(exc).__name__,
            message=MASKED_MESSAGE,
            detail={},
        ),
        headers=_retry_after_headers(exc),
    )


async def dataknobs_error_handler(
    request: Request,  # type: ignore[name-defined]
    exc: DataknobsError,
    *,
    table: Mapping[type[DataknobsError], ErrorPolicy] | None = None,
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
        table: The whole table to resolve against, replacing
            :data:`DEFAULT_ERROR_POLICY`. ``register_exception_handlers``
            binds the merged result here. Named as in
            :func:`resolve_error_policy`, and deliberately not
            ``error_policy``: that name belongs to the *overrides* parameter,
            which merges.

    Returns:
        JSON response with the error's status, disclosed per its policy
    """
    policy = resolve_error_policy(exc, table)

    # Logged whole regardless of policy: the response may carry both halves,
    # one, or neither, and the log is the one place that is always complete.
    # The label records which half the caller saw, so "the client says the
    # error was empty" is answerable without reproducing it.
    _log_handled_error(
        exc,
        policy.status_code,
        "%s handled as HTTP %d (disclosed: %s): %s | context=%s",
        type(exc).__name__,
        policy.status_code,
        _disclosure_label(policy.disclose_message, policy.disclose_context),
        exc,
        exc.context,
    )

    detail: dict[str, Any] = dict(exc.context) if policy.disclose_context else {}
    retry_after = getattr(exc, "retry_after", None)
    if policy.disclose_context and retry_after is not None:
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
            message=str(exc) if policy.disclose_message else MASKED_MESSAGE,
            detail=detail,
        ),
        # Emitted regardless of disclosure: a retry hint is a flow-control
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
    app.add_exception_handler(DataknobsError, partial(dataknobs_error_handler, table=table))
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

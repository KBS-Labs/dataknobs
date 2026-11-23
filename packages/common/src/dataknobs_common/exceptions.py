# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Common exception hierarchy for all dataknobs packages.

This module provides a unified exception framework that all dataknobs packages
can extend. It supports both simple exceptions and context-rich exceptions with
detailed error information.

The exception hierarchy supports:
- Simple error messages for straightforward cases
- Context dictionaries for rich error information
- Details dictionaries (FSM-style) for structured error data
- Package-specific extensions

Example:
    ```python
    from dataknobs_common.exceptions import ValidationError, NotFoundError

    # Simple exception
    raise ValidationError("Invalid email format")

    # Context-rich exception
    raise NotFoundError(
        "User not found",
        context={"user_id": "123", "attempted_at": "2024-11-08"}
    )

    # Catch any dataknobs error
    try:
        operation()
    except DataknobsError as e:
        logger.error(f"Error: {e}")
        if e.context:
            logger.error(f"Context: {e.context}")
    ```

Package-Specific Extensions:
    ```python
    from dataknobs_common.exceptions import DataknobsError

    class MyPackageError(DataknobsError):
        '''Base exception for mypackage.'''
        pass

    class SpecificError(MyPackageError):
        '''Specific error with custom context.'''
        def __init__(self, item_id: str, message: str):
            super().__init__(
                f"Item '{item_id}': {message}",
                context={"item_id": item_id}
            )
    ```
"""

import copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Dict, TypeVar

_E = TypeVar("_E", bound=BaseException)


class DataknobsError(Exception):
    """Base exception for all dataknobs packages.

    This is the root exception that all dataknobs packages should extend.
    It supports optional context data for rich error information, making
    debugging and error handling more effective.

    Attributes:
        context: Dictionary containing contextual information about the error
        details: Alias for context (FSM-style compatibility)

    Args:
        message: Human-readable error message
        context: Optional dictionary with error context (field names, IDs, etc.)
        details: Alternative to context. Whichever of the two is
            non-empty becomes both attributes; they are never merged

    Example:
        ```python
        error = DataknobsError(
            "Operation failed",
            context={"operation": "save", "item_id": "123"}
        )
        str(error)
        # 'Operation failed'
        error.context
        # {'operation': 'save', 'item_id': '123'}
        ```
    """

    def __init__(
        self,
        message: str,
        context: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
    ):
        """Initialize the exception with optional context.

        Args:
            message: Error message
            context: Optional context dictionary
            details: Optional details dictionary. Not merged with
                ``context`` -- the first non-empty of the two is used
        """
        super().__init__(message)
        # Support both context and details parameters
        # Neither is merged: the first non-empty wins, details first
        self.context = details or context or {}
        # Alias for FSM-style compatibility
        self.details = self.context


class ValidationError(DataknobsError):
    """Raised when validation fails.

    Use this exception when data or configuration fails validation checks.
    Common scenarios include:
    - Invalid input data
    - Schema validation failures
    - Constraint violations
    - Type mismatches

    Example:
        ```python
        raise ValidationError(
            "Email format invalid",
            context={"field": "email", "value": "not-an-email"}
        )
        ```
    """

    pass


class ConfigFieldError(ValidationError, ValueError):
    """Raised when a mapping does not fit the config class it is read into.

    :meth:`~dataknobs_common.structured_config.StructuredConfig.from_dict`
    raises it when a mapping leaves out a field the class requires, or, for a
    class that sets ``_UNKNOWN_KEYS = "raise"``, carries a key the class does
    not read. Both are one authoring error, so they share one type, and
    neither reaches the dataclass constructor's ``TypeError``.

    **Both a** :class:`ValidationError` **and a** ``ValueError``, on purpose:

    - a ``ValidationError``, because that is what a door documenting a
      dataknobs refusal promises. An ontology document missing its ``id``
      meets the loader's contract with no translation in between;
    - a ``ValueError``, because the unknown-key refusal was one before this
      type existed. An ``except ValueError`` written against it keeps
      catching it.

    The message names only what the code supplies (the class, its field names
    and the keys the caller wrote) and never a value, so a caller may show it
    to the config's author without vetting it first. That holds through a
    :class:`~dataknobs_common.registry.PluginRegistry` too: a refusal raised
    while a registry builds a plugin reaches its caller as a
    :class:`PluginConfigError` carrying this message and ``context``.

    ``context`` carries ``config_class``, and either ``missing`` (the absent
    required fields), ``unknown`` with ``accepted`` (the refused keys and the
    keys the class reads), or ``non_string_keys`` (the ``repr`` of each key
    that is not a string), each a list of names.

    The guarantee is about the refusals ``from_dict`` writes, not about the
    type: anything may construct a ``ConfigFieldError`` with its own words.
    A registry tells the two apart by a mark only ``structured_config`` sets,
    so a factory's own ``ConfigFieldError`` is bounded like any other error
    it raises. The mark records the refusal's words as ``from_dict`` wrote
    them, so a handler that later amends the message or ``context`` (as
    :func:`naming_refusals` does) adds nothing a registry will carry.

    Example:
        ```python
        try:
            config = MyConfig.from_dict(raw)
        except ConfigFieldError as e:
            report_to_author(str(e), missing=e.context.get("missing"))
        ```
    """


#: Holds what a refusal said when it was marked value-free: its message and a
#: deep copy of its ``context``. See :func:`_value_free`.
_VALUE_FREE_REFUSAL = "_dataknobs_value_free_refusal"


def _value_free(exc: ConfigFieldError) -> ConfigFieldError:
    """Mark ``exc`` as a refusal whose message and ``context`` name no value.

    A :class:`~dataknobs_common.registry.PluginRegistry` bounds the text of
    whatever a factory raises, because a factory's text can carry the
    connection URL it was handed. A refusal ``StructuredConfig`` wrote is
    text dataknobs authored to be shown to the config's author, and the type
    cannot say so: :class:`ConfigFieldError` is public, and a factory may
    raise one with its own words. The mark can, and only ``structured_config``
    sets it.

    It is a narrower claim than the registry's own mark: this text is safe to
    carry, not this error passes through. The mark is on the instance, so a
    factory re-raising the refusal passes it on marked, and one constructing
    a new ``ConfigFieldError`` does not.

    **The mark vouches for what the refusal said when it was set, not for
    the instance.** An exception is mutable: :func:`naming_refusals` prefixes
    its message and merges a caller's ``context`` into it, and any handler
    may do the same by hand. So the mark stores the message and a deep copy
    of ``context`` as they are here, and :func:`_value_free_words` returns
    that copy -- never the instance's current text, which a factory may since
    have amended with its own words.
    """
    setattr(exc, _VALUE_FREE_REFUSAL, (str(exc), copy.deepcopy(dict(exc.context))))
    return exc


def _value_free_words(exc: BaseException) -> tuple[str, dict[str, Any]] | None:
    """What a :func:`_value_free` refusal said when it was marked, if ``exc`` is one.

    A copy each time, so a caller may keep or change what it is given.
    """
    if not isinstance(exc, ConfigFieldError):
        return None
    words = getattr(exc, _VALUE_FREE_REFUSAL, None)
    if not (isinstance(words, tuple) and len(words) == 2):
        return None
    message, context = words
    if not (isinstance(message, str) and isinstance(context, dict)):
        return None
    return message, copy.deepcopy(context)


def _value_free_refusal_behind(exc: BaseException) -> tuple[str, dict[str, Any]] | None:
    """The words of the value-free refusal ``exc`` is, or was raised ``from``.

    Follows ``__cause__`` only. A ``raise ... from refusal`` is the raiser
    saying the refusal caused its error, so reporting the refusal is
    accurate. ``__context__`` says only that an error was raised while one
    was being handled, and following it would report a refusal the raiser
    caught and recovered from. Only the refusal's own words, as marked, are
    returned, so the walk cannot bring anyone else's text along with it.
    A ``__cause__`` cycle ends the walk.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        words = _value_free_words(current)
        if words is not None:
            return words
        seen.add(id(current))
        current = current.__cause__
    return None


#: Holds what an error said when dataknobs marked it as text it wrote: its
#: message and a deep copy of its ``context``. See :func:`_written_by_dataknobs`.
#:
#: **Private on purpose.** The mark vouches that dataknobs wrote this error's
#: text and that the text names only what a config's author wrote or what
#: dataknobs itself knows (a registered name, a field name, a count) -- never a
#: value a factory, a driver or a service supplied. A consumer setting it on
#: their own error would be vouching for text no dataknobs review has read,
#: which is why it is not exported and why a source-scan test pins every place
#: it is set. A public way for a consumer to mark their own refusals is a
#: separate decision, with its own design, and this is not it.
_WRITTEN_BY_DATAKNOBS = "_dataknobs_written_by_dataknobs"


def _written_by_snapshot(exc: BaseException) -> tuple[str, dict[str, Any] | None] | None:
    """What ``exc`` says now: its message and a deep copy of its ``context``.

    ``None`` when the ``context`` cannot be copied, so such an error is never
    marked -- which bounds it, the safe direction.
    """
    context = getattr(exc, "context", None)
    try:
        copied = copy.deepcopy(dict(context)) if isinstance(context, Mapping) else None
    except Exception:
        return None
    return str(exc), copied


def _written_by_dataknobs(exc: _E) -> _E:
    """Mark ``exc`` as an error whose text dataknobs wrote, and return it.

    A :class:`~dataknobs_common.registry.PluginRegistry` bounds the text of
    whatever a factory raises, because a factory's text can carry the
    connection URL it was handed. A helper in front of a registry then has
    to decide whether the factory's error may be shown after all, and the
    type cannot tell it: a ``ValueError`` dataknobs raised to say a summary
    memory has no LLM and one a driver raised with a DSN in it are the same
    type. This mark can, and :func:`~dataknobs_common.registry.surfaced_cause`
    reads it.

    **The mark vouches for what the error said when it was set, not for the
    instance.** It stores the message and a deep copy of ``context``, and
    :func:`_is_written_by_dataknobs` is true only while the error still says
    exactly that. Unlike :func:`_value_free`, the snapshot cannot be carried
    in place of the instance: a surfaced error is raised again *as itself*,
    so the only safe answer to "its words changed since" is "then it is no
    longer only ours". An error a handler amends after marking (appending to
    ``args``, changing ``context``) reads as unmarked and is bounded.
    :func:`naming_refusals` is the one amender that marks again, and only an
    error that was marked when it arrived.

    Set where the error is raised, at the sites a source-scan test lists, and
    nowhere else but :func:`naming_refusals`.
    """
    snapshot = _written_by_snapshot(exc)
    if snapshot is not None:
        setattr(exc, _WRITTEN_BY_DATAKNOBS, snapshot)
    return exc


def _is_written_by_dataknobs(exc: BaseException) -> bool:
    """Whether ``exc`` is marked and still says what it said when marked.

    Asked while another error is being handled, so it never raises: a
    ``context`` whose comparison raises, or answers something that is not a
    plain truth value, reads as unmarked. So does a ``context`` holding a
    value compared by identity, since the snapshot holds a copy of it. Both
    are bounded, the safe direction; a marking site keeps its ``context`` to
    names, counts and plain containers of them.
    """
    recorded: object = getattr(exc, _WRITTEN_BY_DATAKNOBS, None)
    if recorded is None:
        return False
    try:
        return (recorded == _written_by_snapshot(exc)) is True
    except Exception:
        return False


#: What a configuration document can write: the values YAML and JSON parse to.
_DOCUMENT_SCALARS = (str, int, float, bool, type(None))

#: How deep a document value may nest before it is named by type instead.
#: Generous for anything a config author writes; it stops a value that
#: contains itself from recursing until the interpreter gives up.
_DOCUMENT_DEPTH = 32

#: The most of a document value's ``repr`` a refusal shows.
_SHOWN_LENGTH = 200


def _is_document_value(value: object, depth: int = 0) -> bool:
    """Whether ``value`` is something a configuration document could have written.

    The types are checked exactly, not with ``isinstance``: a subclass of
    ``str`` or ``dict`` can carry a ``__repr__`` of its own, and a live
    mapping such as ``os.environ`` is a mapping of strings without being
    one an author wrote.
    """
    if depth > _DOCUMENT_DEPTH:
        return False
    kind = type(value)
    if kind in _DOCUMENT_SCALARS:
        return True
    if kind in (list, tuple) and isinstance(value, list | tuple):
        return all(_is_document_value(item, depth + 1) for item in value)
    if kind is dict and isinstance(value, dict):
        return all(
            type(key) is str and _is_document_value(item, depth + 1) for key, item in value.items()
        )
    return False


def _given(value: object) -> str:
    """``value`` as a refusal dataknobs marks may show it.

    Its ``repr`` when it is a document value -- what the config's author
    wrote, cut short past a couple of hundred characters -- and otherwise
    only its type, because a caller can hand a live
    object where a config was expected, and a live handle's ``repr`` can
    carry the connection URL it was opened with.
    """
    if not _is_document_value(value):
        return f"a {type(value).__name__}"
    shown = repr(value)
    if len(shown) <= _SHOWN_LENGTH:
        return shown
    return f"{shown[:_SHOWN_LENGTH]}... ({len(shown)} characters)"


def _given_value(value: object) -> object:
    """``value`` for a marked refusal's ``context``: itself, or its type's name."""
    return value if _is_document_value(value) else type(value).__name__


class ConfigurationError(DataknobsError):
    """Raised when configuration is invalid or missing.

    Use this exception for configuration-related errors including:
    - Missing required configuration
    - Invalid configuration values
    - Configuration file not found
    - Circular references in configuration

    Example:
        ```python
        raise ConfigurationError(
            "Database configuration missing",
            context={"config_key": "database.primary", "available_keys": ["cache", "auth"]}
        )
        ```
    """

    pass


class ResourceError(DataknobsError):
    """Raised when resource operations fail.

    Use this exception for resource management failures including:
    - Resource acquisition failures
    - Connection errors
    - Resource pool exhaustion
    - Timeout errors

    Example:
        ```python
        raise ResourceError(
            "Failed to acquire database connection",
            context={"pool_size": 10, "active_connections": 10, "timeout": 30}
        )
        ```
    """

    pass


class NotFoundError(DataknobsError):
    """Raised when a requested item is not found.

    Use this exception when looking up items by ID, name, or key and they
    don't exist. Common scenarios include:
    - Record not found in database
    - Configuration key not found
    - File not found
    - Resource not registered

    Example:
        ```python
        raise NotFoundError(
            "Record not found",
            context={"record_id": "user-123", "table": "users"}
        )
        ```
    """

    pass


class ConsentRequiredError(DataknobsError):
    """Raised when access to a consent-gated resource is refused.

    A fail-closed *policy denial*, not an operation failure: the caller has
    not granted the consent scope a resource requires, so the read or write is
    refused before it runs. A top-level sibling of :class:`ValidationError` /
    :class:`NotFoundError` (not an :class:`OperationError` — nothing failed;
    the operation was declined by policy).

    Attributes:
        scope: The consent scope that was required but not granted.
        user_id: The user the access was attempted for (optional; may be an
            opaque identifier — handle with the same care as the id itself).

    Example:
        ```python
        raise ConsentRequiredError(
            "Consent scope 'analytics' not granted",
            scope="analytics",
        )
        ```
    """

    def __init__(
        self,
        message: str = "Consent required",
        scope: str | None = None,
        user_id: str | None = None,
        context: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the consent-required error.

        Args:
            message: Error message.
            scope: The consent scope that was required but not granted.
            user_id: Optional user identifier the access was attempted for.
            context: Optional context dictionary (merged with ``scope`` /
                ``user_id``).
            details: Optional details dictionary (takes precedence per the
                base contract).
        """
        merged: Dict[str, Any] = {}
        if scope is not None:
            merged["scope"] = scope
        if user_id is not None:
            merged["user_id"] = user_id
        if context:
            merged.update(context)
        super().__init__(message, context=merged, details=details)
        self.scope = scope
        self.user_id = user_id


class OperationError(DataknobsError):
    """Raised when an operation fails.

    Use this exception for general operation failures that don't fit
    other categories. Common scenarios include:
    - Database operation failures
    - File I/O errors
    - Network operation failures
    - State transition errors

    Example:
        ```python
        raise OperationError(
            "Failed to save record",
            context={"operation": "update", "backend": "postgres", "error": "connection lost"}
        )
        ```
    """

    pass


class PluginConfigError(ConfigFieldError, OperationError):
    """A config refusal raised while a registry built a plugin.

    :class:`~dataknobs_common.registry.PluginRegistry` raises it in place of
    its bounded ``OperationError`` when what the factory raised is, or was
    caused by, a :class:`ConfigFieldError` that ``StructuredConfig`` wrote --
    a missing field, an unknown or misspelt key, a key that is not a string.
    The message is ``"Failed to create plugin '<key>': <refusal>"`` and
    ``context`` is the refusal's, plus ``key`` and ``registry``. Both are the
    refusal's words as ``from_dict`` wrote them, not as a factory may have
    amended them since. The factory's error is on ``__cause__``.

    The refusal is reported wherever it is on the ``__cause__`` chain, so one
    about settings a factory read for itself reads as the caller's config
    error. Its text still names no value; a factory that does not want that
    should catch the refusal and raise its own error, not raise ``from`` it.

    **Both a** :class:`ConfigFieldError` **and an** :class:`OperationError`,
    in that order:

    - an ``OperationError``, because that is what a registry raised for this
      failure before, so an ``except OperationError`` around ``create`` keeps
      catching it;
    - a ``ConfigFieldError`` (and so a :class:`ValidationError` and a
      ``ValueError``), because it is the config's author's error, and a
      handler choosing what to disclose by type -- the first match in the
      MRO -- should treat it as one. Listing ``ConfigFieldError`` first is
      what puts it ahead of ``OperationError`` in that walk.

    Example:
        ```python
        try:
            source = sources.create(config=spec)
        except PluginConfigError as e:
            report_to_author(str(e), unknown=e.context.get("unknown"))
        ```
    """


class FactoryValueError(ValidationError, ValueError):
    """A bounded stand-in for a factory's own ``ValueError`` or ``ValidationError``.

    Raised by :func:`~dataknobs_common.registry.create_surfacing` (and its
    async twin) when a factory behind a registry raised a ``ValueError`` or
    a :class:`ValidationError` whose text dataknobs did not write. That text may carry what the factory
    was handed -- a connection URL, a credential -- so it is not shown: the
    message is ``"Failed to create <subject> '<key>' (<type>)"`` and the
    factory's error is on ``__cause__``, for a server-side log.

    **Both a** :class:`ValidationError` **and a** ``ValueError``:

    - a ``ValueError``, because the helpers that raise it raised the
      factory's ``ValueError`` itself before, so an ``except ValueError``
      written against them keeps catching it;
    - a ``ValidationError``, so a factory's own ``ValidationError`` -- the
      dataknobs type for refused config -- is still caught as one;
      and because a factory refusing its config is the config's problem, so
      a handler choosing a status by type (the bots API's
      ``DEFAULT_ERROR_POLICY`` renders it as a 422) should treat it as one.
      The message and ``context`` are bounded, so showing them is safe.

    ``context`` carries ``key``, ``registry`` and ``cause_type``.

    Example:
        ```python
        try:
            memory = await create_memory_from_config(config)
        except FactoryValueError as e:
            logger.error("Memory backend refused its config", exc_info=e.__cause__)
            report_to_author(str(e))
        ```
    """


class ConcurrencyError(DataknobsError):
    """Raised when concurrent operation conflicts occur.

    Use this exception for concurrency-related failures including:
    - Lock acquisition failures
    - Transaction conflicts
    - Race conditions
    - Optimistic locking failures

    Example:
        ```python
        raise ConcurrencyError(
            "Record modified by another process",
            context={"record_id": "123", "expected_version": 5, "actual_version": 6}
        )
        ```
    """

    pass


class SerializationError(DataknobsError):
    """Raised when serialization or deserialization fails.

    Use this exception for data format conversion errors including:
    - JSON encoding/decoding failures
    - Invalid data format
    - Schema mismatch
    - Type conversion errors

    Example:
        ```python
        raise SerializationError(
            "Cannot deserialize data",
            context={"format": "json", "field": "created_at", "value": "invalid-date"}
        )
        ```
    """

    pass


class TimeoutError(DataknobsError):
    """Raised when an operation times out.

    Use this exception when operations exceed their time limit including:
    - Connection timeouts
    - Query timeouts
    - Resource acquisition timeouts
    - Operation execution timeouts

    Example:
        ```python
        raise TimeoutError(
            "Database query timed out",
            context={"query": "SELECT * FROM large_table", "timeout_seconds": 30}
        )
        ```
    """

    pass


class RateLimitError(OperationError):
    """Raised when a rate limit is exceeded.

    Use this exception when an operation cannot proceed because a rate limit
    has been reached. Includes an optional ``retry_after`` hint indicating
    how many seconds the caller should wait before retrying.

    Attributes:
        retry_after: Optional number of seconds to wait before retrying.

    Example:
        ```python
        raise RateLimitError(
            "API rate limit exceeded",
            retry_after=2.5,
            context={"category": "api_write", "limit": 10, "interval": 60}
        )
        ```
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        context: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: Error message.
            retry_after: Optional seconds to wait before retrying.
            context: Optional context dictionary.
            details: Optional details dictionary. Not merged with
                ``context`` -- the first non-empty of the two is used.
        """
        super().__init__(message, context=context, details=details)
        self.retry_after = retry_after


class DottedPathReason(StrEnum):
    """The complete :attr:`DottedPathError.reason` vocabulary.

    Normalized in the constructor for the same reason
    ``PackResolutionReason`` is: a plain string stays acceptable, but an
    unrecognized one is a typo rather than a new vocabulary member.
    """

    #: The reference is not of the form ``module:name`` / ``module.name``.
    MALFORMED = "malformed"
    #: A module was not found — the target itself, an ancestor package of it,
    #: or something it imports at its top level. **An environment condition:**
    #: something is not installed. This is the reason a config key documented
    #: as ``optional`` may reasonably swallow.
    #:
    #: A missing *transitive* dependency lands here rather than in
    #: :attr:`IMPORT_FAILED` deliberately. A tool whose module imports an
    #: uninstalled SDK is exactly the optional-dependency case, and telling it
    #: apart from a mistyped path would need the deployment's intent, which
    #: this layer does not have.
    MODULE_NOT_FOUND = "module_not_found"
    #: The module was found, and **executing it raised** something other than
    #: a missing module. **A defect, not an environment condition:** the code
    #: is present and broken. Split from :attr:`MODULE_NOT_FOUND` so that a
    #: caller skipping absent optional dependencies does not also skip a
    #: module that is installed and raising — the two want opposite responses,
    #: and one is never safe to swallow silently.
    IMPORT_FAILED = "import_failed"
    #: The module imported; it has no such attribute.
    ATTRIBUTE_NOT_FOUND = "attribute_not_found"
    #: The attribute resolved and is not callable.
    NOT_CALLABLE = "not_callable"


class DottedPathError(ConfigurationError):
    """Raised when a dotted path from configuration cannot be resolved.

    Carries the offending ``ref`` and a machine-readable :attr:`reason`,
    following ``PackResolutionError``'s shape.

    Deliberately a **sibling** of :class:`DottedPathTypeError`, not its
    parent. The two mean different things to a caller: a resolution failure
    is transient and environmental (a module that is not installed, a typo
    in a path), and a config key documented as ``optional`` may reasonably
    swallow it; a shape mismatch means the path resolved to the wrong kind
    of object, which is never optional. Were the shape error a subclass, the
    obvious lenient handler::

        except DottedPathError:
            if optional:
                return None

    would swallow it too, and ``optional: true`` would silently grow to
    cover misfiled specs. As siblings, that handler cannot match a shape
    mismatch at all, so the distinction holds by construction rather than by
    remembering to order the ``except`` clauses.
    """

    def __init__(
        self,
        message: str,
        *,
        ref: str,
        reason: DottedPathReason | str,
        **context: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Bounded description — see the module docstring of
                :mod:`dataknobs_common.imports` for why the underlying
                exception's text does not belong here.
            ref: The dotted path that failed, as written in configuration.
            reason: A :class:`DottedPathReason` member (or its value).
            **context: Extra context keys, merged into ``context``.
        """
        reason = DottedPathReason(reason)
        super().__init__(message, context={"ref": ref, "reason": reason, **context})
        self.ref = ref
        self.reason = reason


class DottedPathTypeError(ConfigurationError):
    """Raised when a dotted path resolves to an object of the wrong shape.

    The sibling of :class:`DottedPathError` — see that class for why the two
    are not related by inheritance. This one is **never** optional: a path
    that resolved successfully but named the wrong kind of object is a
    programmer error in the configuration's layout, and the only safe
    response is to surface it at config-load time.
    """

    def __init__(
        self,
        message: str,
        *,
        ref: str,
        expected: type,
        **context: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Bounded description of the mismatch.
            ref: The dotted path that resolved, as written in configuration.
            expected: The base class or protocol the target had to satisfy.
            **context: Extra context keys, merged into ``context``.
        """
        super().__init__(message, context={"ref": ref, "expected": expected, **context})
        self.ref = ref
        self.expected = expected


@contextmanager
def naming_refusals(origin: str, *, context: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Say where the configuration refused inside this block came from.

    A component refuses its own configuration without knowing who configured
    it: a database backend built for an ontology binding or a bot's grounded
    source names its table, and nothing a reader of the ontology document or
    the bot config would recognise. The caller that does know wraps the
    construction, and a :class:`ValidationError` or :class:`ConfigurationError`
    raised inside -- a subclass included -- is raised again **as the same
    exception**, its message prefixed with ``"<origin>: "`` and its
    ``context`` merged with ``context``, the caller's keys winning.

    Amended rather than rebuilt, so its type and attributes survive: a
    subclass may construct itself from other arguments
    (:class:`DottedPathError` takes a ``ref`` and a ``reason``), and a handler
    a caller already has for it keeps catching it. The context is replaced
    with a merged copy, never updated in place, since the raiser may still
    hold the mapping it passed. Any other exception passes through untouched.

    **A refusal dataknobs wrote stays dataknobs' text** when it was marked as
    such on entry, so a helper in front of a registry still shows it, now
    naming its origin. The origin is the caller's text: an in-tree caller
    names a source or an ontology section the author wrote, and a consumer
    passing their own ``origin`` vouches for it as they would by raising it.
    An unmarked refusal stays unmarked.

    Args:
        origin: Where the configuration came from, as its reader would name
            it (``"source 'cases'"``), in the form a schema reader's
            ``origin=`` takes.
        context: Keys that locate the origin (``{"source": "cases"}``),
            merged into the refusal's ``context``.

    Example:
        ```python
        with naming_refusals(f"source {name!r}", context={"source": name}):
            db = database_factory.create(**options)
        ```
    """
    try:
        yield
    except (ValidationError, ConfigurationError) as e:
        ours = _is_written_by_dataknobs(e)
        message = e.args[0] if e.args else ""
        e.args = (f"{origin}: {message}", *e.args[1:])
        merged = {**dict(e.context or {}), **dict(context or {})}
        e.context = merged
        e.details = merged
        if ours:
            _written_by_dataknobs(e)
        raise


__all__ = [
    "DataknobsError",
    "ValidationError",
    "ConfigFieldError",
    "ConfigurationError",
    "DottedPathError",
    "DottedPathReason",
    "DottedPathTypeError",
    "ResourceError",
    "NotFoundError",
    "ConsentRequiredError",
    "OperationError",
    "PluginConfigError",
    "FactoryValueError",
    "ConcurrencyError",
    "SerializationError",
    "TimeoutError",
    "RateLimitError",
    "naming_refusals",
]

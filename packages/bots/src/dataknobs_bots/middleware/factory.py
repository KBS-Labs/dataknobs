"""Build middleware instances from declarative specs.

A middleware spec is a plain mapping — the same shape the bot config
accepts under ``middleware:`` and ``conversation_middleware:``::

    {"class": "my_pkg.mw.AuditMiddleware", "params": {"level": "info"},
     "optional": False}

``DynaBot`` resolves its own configured specs through these functions, but
they are deliberately free-standing: anything that assembles middleware
declaratively — a composed pack of platform behavior, a deployment's
policy bundle, a test fixture — can turn specs into live instances and
hand the results to ``DynaBot.from_config(..., platform_middleware=...)``
without reaching into bot internals or reimplementing the resolution
rules.

Two flavors, wired to different layers:

- :func:`build_middleware` → :class:`~dataknobs_bots.middleware.Middleware`,
  the bot-turn lifecycle hooks (``on_turn_start`` / ``after_turn`` / ...).
- :func:`build_conversation_middleware` →
  :class:`~dataknobs_llm.conversations.ConversationMiddleware`, which wraps
  the ``llm.complete`` call itself.

Both take an *iterable* of specs and return a list of live instances, which
is the shape both install channels want — ``DynaBot`` builds its own two
configured lists through them, and a caller assembling middleware
declaratively hands the result straight to
``from_config(..., platform_middleware=...)``. Skipping the ``None`` an
``optional: true`` spec produces happens here rather than in every caller.

Both delegate to :func:`resolve_middleware_from_spec`, so there is exactly
one resolution body and the two flavors cannot drift.

.. warning::

   **Middleware specs are trusted configuration.** A spec's ``class`` is a
   dotted path that gets imported and instantiated, so resolving one
   executes whatever that module and constructor do. Specs must come from
   the same trust domain as the application's own code — a config file, a
   deployment's policy bundle, a pack a platform team authored.

   Never build a spec from end-user input, a request body, or a per-tenant
   blob supplied by the tenant. That applies with particular force to
   composed declarations: a pack field holding middleware specs concatenates
   contributions, so an attacker who can add one entry to a binding body can
   name any importable class. There is no allow-list here and no sandbox —
   import *is* execution.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from dataknobs_common.exceptions import (
    ConfigurationError,
    DottedPathError,
    DottedPathTypeError,
)
from dataknobs_common.imports import resolve_class
from dataknobs_llm.conversations import ConversationMiddleware

from .base import Middleware

logger = logging.getLogger(__name__)

__all__ = [
    "build_conversation_middleware",
    "build_middleware",
    "resolve_middleware_from_spec",
]


def resolve_middleware_from_spec(
    config: Mapping[str, Any],
    expected_base: type,
    *,
    label: str,
) -> Any | None:
    """Resolve a middleware spec to an instance, validating its class shape.

    Shared resolution body behind :func:`build_middleware` (bot-turn
    :class:`~dataknobs_bots.middleware.Middleware`) and
    :func:`build_conversation_middleware` (LLM-call
    :class:`~dataknobs_llm.conversations.ConversationMiddleware`). The two
    flavors are wired to different layers but share this construction
    shape (``class`` + ``params`` + ``optional``).

    Call this directly only for a middleware family neither wrapper
    covers; for the two built-in flavors prefer the wrappers, which supply
    the correct ``expected_base`` and ``label``.

    The class-shape check uses ``issubclass`` BEFORE instantiation so a
    wrong-shape spec never runs its ctor (avoiding network reads / file
    opens / log writes a misplaced spec's initializer might trigger).
    Type-mismatch errors raise unconditionally — ``optional: true`` covers
    transient resolution failures (module / class / params), NOT a class
    listed under the wrong field. A misplaced spec (a turn-lifecycle
    ``Middleware`` listed under ``conversation_middleware:``, or vice
    versa) is a programmer error in the config layout, and the only safe
    response is to surface it at config-load.

    Args:
        config: Middleware configuration mapping with:
            - class: Dotted import path to middleware class
            - params: Optional constructor parameters
            - optional: If True, log warning and skip on resolution
              failure (missing module / class / bad params) instead of
              raising (default: False). Does NOT apply to class-shape
              mismatches — those always raise.
        expected_base: Class the resolved middleware must subclass
            (``Middleware`` for bot turn-lifecycle hooks,
            ``ConversationMiddleware`` for LLM-call wraps).
        label: Human-readable label used in error / log messages
            (e.g. ``"middleware"``, ``"conversation_middleware"``).

    Returns:
        Instantiated middleware, or ``None`` if resolution fails
        (NOT a class-shape mismatch) and ``optional: true`` was set.

    Raises:
        ConfigurationError: If the class cannot be resolved or
            instantiation fails, unless ``optional: true``; OR if the
            resolved class is not a subclass of ``expected_base``
            (always raises, regardless of ``optional``).
    """
    optional = config.get("optional", False)
    class_path = config.get("class", "<missing>")

    # `resolve_class` validates the shape and returns the CLASS, so no
    # constructor runs for a wrong-shape spec. The two failure modes arrive
    # as two sibling exception types, which is what keeps ``optional`` from
    # reaching the shape check: `DottedPathTypeError` is not a
    # `DottedPathError`, so the clause below cannot match it.
    try:
        middleware_class = resolve_class(config["class"], expected_base)
    except DottedPathError as e:
        # Resolution failure (missing module / class / malformed spec)
        # — covered by ``optional``.
        detail = f"Failed to resolve {label} '{class_path}': {e}"
        if optional:
            logger.warning("Skipping optional %s: %s", label, detail)
            return None
        # Re-raised as the same type, not as a plain `ConfigurationError`:
        # the label says which config key the bad path was under, which is
        # worth adding, but not at the cost of the `reason` a caller can
        # branch on. Same shape as `resolve_optional_callable`'s lift.
        raise DottedPathError(
            f"Failed to resolve {label} '{class_path}' ({e.reason})",
            ref=e.ref,
            reason=e.reason,
            label=label,
        ) from e
    except KeyError as e:
        # The spec has no `class` key at all. Kept distinct from a malformed
        # path: `config["class"]` is what raises, before any resolution.
        if optional:
            logger.warning("Skipping optional %s: spec has no 'class' key", label)
            return None
        raise ConfigurationError(f"{label} spec has no 'class' key") from e
    except DottedPathTypeError as e:
        # Never optional: a class listed under the wrong field is a
        # programmer error in the config layout, not a transient environment
        # failure. Re-raised only to add the which-field hint.
        raise ConfigurationError(
            f"{label} '{class_path}' must subclass "
            f"{expected_base.__module__}.{expected_base.__qualname__} "
            f"— check whether this spec belongs under 'middleware' "
            f"(bot-turn hooks) or 'conversation_middleware' "
            f"(LLM-call wraps)."
        ) from e

    try:
        return middleware_class(**dict(config.get("params", {})))
    except Exception as e:
        # Instantiation failure (bad params, ctor raised) — covered by
        # ``optional``.
        detail = f"Failed to instantiate {label} '{class_path}': {e}"
        if optional:
            logger.warning("Skipping optional %s: %s", label, detail)
            return None
        # Bounded message: this catches ANY constructor, so `e` is
        # third-party text the deployment does not control -- a database or
        # cache client raises with its connection URL in the message, and
        # `ConfigurationError` is rendered at the HTTP boundary. Keep the
        # class path (from the config) and the exception type (a class
        # name); let __cause__ carry the rest to the logs.
        raise ConfigurationError(
            f"Failed to instantiate {label} '{class_path}' ({type(e).__name__})"
        ) from e


def build_middleware(specs: Iterable[Mapping[str, Any]]) -> list[Middleware]:
    """Build bot-turn :class:`Middleware` instances from a sequence of specs.

    See :func:`resolve_middleware_from_spec` for the resolution,
    class-shape validation, and ``optional`` semantics.

    Specs marked ``optional: true`` whose class cannot be resolved or
    instantiated are **absent** from the result (a warning is logged), so
    the returned list is directly usable as ``platform_middleware`` — the
    positional correspondence with ``specs`` is deliberately not preserved.
    Call :func:`resolve_middleware_from_spec` per spec when you need to know
    which one was skipped.

    Args:
        specs: Middleware specs, in installation order.

    Returns:
        Live middleware instances, in the order their specs appeared.

    Raises:
        ConfigurationError: On the first spec that fails to resolve without
            ``optional: true``, or whose class is not a ``Middleware``
            (which raises regardless of ``optional``).

    Example:
        ```python
        from dataknobs_bots import build_middleware

        specs = [{"class": "my_pkg.mw.AuditMiddleware"}]
        bot = await DynaBot.from_config(
            cfg, platform_middleware=build_middleware(specs)
        )
        ```
    """
    resolved = (
        resolve_middleware_from_spec(spec, Middleware, label="middleware") for spec in specs
    )
    return [mw for mw in resolved if mw is not None]


def build_conversation_middleware(
    specs: Iterable[Mapping[str, Any]],
) -> list[ConversationMiddleware]:
    """Build :class:`ConversationMiddleware` (LLM-call wraps) from specs.

    The :func:`build_middleware` sibling, one layer down; the resolution,
    ``optional``-skipping, and ordering contracts are identical. Pass the
    result to ``DynaBot.from_config(..., platform_conversation_middleware=...)``.
    """
    resolved = (
        resolve_middleware_from_spec(spec, ConversationMiddleware, label="conversation_middleware")
        for spec in specs
    )
    return [mw for mw in resolved if mw is not None]

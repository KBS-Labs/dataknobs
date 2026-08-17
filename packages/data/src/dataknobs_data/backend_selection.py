"""One answer to "which backend?", shared by every factory that asks it.

Three factories read a ``backend`` key out of a config, fall back to the
same default when it is absent, look the result up in their own registry and
raise when it names nothing. Each did all four inline, which is why a gap in
the first of them was a gap in three places: an absent key and an explicit
``backend: memory`` produced the same log line, so a config that arrived
empty could not be told from one that asked for an in-process store.

They are different events and are logged differently here. An absent key is
a WARNING naming the fallback and its consequences; a key that is present is
an INFO naming what was asked for. Nothing else about the outcome changes --
the same object is built either way, which is exactly what made the
difference invisible.

Registration lives here too, for a related reason. Backends behind an
optional dependency used two idioms that meant opposite things: some
imported their driver at module top level, so a missing driver failed the
import and the backend went unregistered, while others swallowed their own
``ImportError`` and deferred the raise to construction, so they registered
whether or not the driver was there. "Registered" therefore answered "is
this installed?" honestly for one group and dishonestly for the other.
:func:`register_backend` gives both groups one idiom -- probe the declared
driver, register when it is present and declare the backend unavailable
when it is not -- so ``registered == installed`` is true of every backend
rather than most of them.

This module imports nothing from its own package, so it is safe to import
from anywhere in it, including the modules involved in the
``vector/stores`` import cycle.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dataknobs_common.registry import PluginRegistry


logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_BACKEND",
    "available_backends",
    "backend_available",
    "backend_info",
    "build_backend",
    "is_default_backend",
    "module_installed",
    "normalize_backend",
    "register_backend",
    "select_backend",
]


#: The backend every factory here falls back to when a config names none.
#: Declared once so the construction path and the validation path that
#: mirrors it cannot drift; holding no second copy is the guarantee.
DEFAULT_BACKEND = "memory"

#: What the default costs, in the words the WARNING uses. Kept beside the
#: name rather than inlined into the message, so changing the default means
#: changing its description in the same edit -- a message that parameterises
#: the name while hard-coding "in-process and unpersisted" would go on
#: describing memory whatever it named.
DEFAULT_BACKEND_CONSEQUENCE = (
    "in-process and unpersisted -- it answers every query with zero results "
    "until something writes to it, and loses everything when the process "
    "restarts"
)


def module_installed(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it.

    Args:
        module: A top-level module name, as an optional dependency ships.

    Returns:
        True when the module is present on the import path.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A namespace package with a broken parent raises rather than
        # returning None. Unimportable either way, which is the question.
        return False


def normalize_backend(raw: Any) -> str:
    """Reduce a present ``backend`` value to a lookup key.

    Shared by the construction path and the validation path that mirrors
    it, so "what counts as a backend name" has one answer. The two used to
    normalise separately, and disagreed about ``backend: null``: one raised
    a ``ValueError`` naming the backend, the other an ``AttributeError``
    from calling ``.lower()`` on ``None``.

    Args:
        raw: The value found under the ``backend`` key. Only call this when
            the key is present -- an absent key means the default, which is
            a different event and is reported as one.

    Returns:
        The lowercased, stripped name.

    Raises:
        ValueError: The key is present but names nothing usable. Treating
            that as absent would produce the in-process default from a
            config that did try to choose.
    """
    if raw is None:
        raise ValueError(
            "The 'backend' key is present but null. Remove the key to accept "
            f"the default ('{DEFAULT_BACKEND}'), or name a backend."
        )
    if not isinstance(raw, str):
        raise ValueError(
            f"The 'backend' key must be a string naming a backend, got "
            f"{type(raw).__name__}: {raw!r}."
        )
    name = raw.strip().lower()
    if not name:
        raise ValueError(
            "The 'backend' key is present but empty. Remove the key to accept "
            f"the default ('{DEFAULT_BACKEND}'), or name a backend."
        )
    return name


def is_default_backend(config: Mapping[str, Any]) -> bool:
    """Whether this config asks for the default backend.

    True both when the config names it and when it names nothing, because
    the caller this serves is about to do the same thing either way: skip
    the factory and construct the in-process store directly, or decide not
    to build a store at all.

    Callers that *build* through a factory should not use this -- pass the
    config to :func:`select_backend`, which distinguishes the two cases and
    reports the difference. This is for the caller that branches before any
    factory is involved, where the distinction has no consequence to
    report.

    Args:
        config: A config that may carry a ``backend`` key.

    Returns:
        True when the key is absent, or present and naming
        :data:`DEFAULT_BACKEND`.

    Raises:
        ValueError: The key is present but names nothing usable, per
            :func:`normalize_backend`. Treating that as the default would
            silently give an in-process store to a config that did try to
            choose.
    """
    if "backend" not in config:
        return True
    return normalize_backend(config["backend"]) == DEFAULT_BACKEND


def register_backend(
    registry: PluginRegistry[Any],
    key: str,
    load: Callable[[], Any],
    *,
    metadata: dict[str, Any],
    aliases: Sequence[str] = (),
    installed: Callable[[str], bool] = module_installed,
    override: bool = False,
) -> None:
    """Register a backend, or record why it cannot be created here.

    The driver to probe is declared in the metadata under
    ``requires_module``; a backend without one has no optional dependency
    and always registers. When the driver is missing the backend is
    declared unavailable rather than left out, which keeps two things
    working that being left out breaks: ``get_backend_info`` can still say
    what to install, and ``create`` can say the driver is missing instead
    of reporting the name as unrecognised.

    Args:
        registry: The registry to register into.
        key: Canonical backend name.
        load: Imports and returns the backend class. Called only when the
            declared driver is present; an ``ImportError`` from it is
            treated the same way as an absent driver.
        metadata: Registry metadata for the canonical key. ``requires_module``
            names the driver (a string, or several); ``requires_install``
            says how to install it.
        aliases: Additional accepted spellings. They share the canonical
            key's factory and carry no metadata of their own, which is what
            makes them collapse in :meth:`PluginRegistry.list_canonical_keys`.
        installed: The "is this module importable?" predicate. Injectable
            so a test can describe an environment this one cannot be.
        override: Replace an existing registration under the same name.
            Forwarded to :meth:`PluginRegistry.register`, which otherwise
            raises on a second registration -- so a consumer swapping a
            built-in backend for its own had no way through this function
            and had to drop to the registry, losing the driver probe that
            makes ``registered == installed`` hold.
    """
    required = metadata.get("requires_module") or ()
    if isinstance(required, str):
        required = (required,)

    missing = [module for module in required if not installed(module)]
    reason: str | None = None
    if missing:
        reason = f"{', '.join(missing)} is not installed"
        hint = metadata.get("requires_install")
        if hint:
            reason = f"{reason}. Install with: {hint}"
    else:
        try:
            backend_class = load()
        except ImportError as exc:
            # The declared driver is present but something else the module
            # needs is not. Reported verbatim -- it names the real gap.
            reason = str(exc)

    if reason is not None:
        registry.declare_unavailable(key, metadata=metadata, reason=reason, aliases=aliases)
        logger.debug("Backend '%s' is unavailable: %s", key, reason)
        return

    registry.register(key, backend_class, metadata=metadata, override=override)
    for alias in aliases:
        registry.register(alias, backend_class, override=override)


def available_backends(registry: PluginRegistry[Any]) -> list[str]:
    """Canonical backend names, sorted, with aliases collapsed.

    A list of backends rather than a list of spellings, and of backends
    that can actually be built: one whose driver is missing is declared
    unavailable at registration, so it is absent here for the same reason
    :func:`backend_available` reports it False.

    Args:
        registry: The registry to report on.

    Returns:
        Sorted canonical names, one per available backend.
    """
    return registry.list_canonical_keys()


def backend_available(registry: PluginRegistry[Any], backend_type: str) -> bool:
    """Whether ``create`` can build this backend under this installation.

    Registration probes the backend's declared driver, so a registered name
    is one whose optional dependency is present -- which is what makes this
    answerable rather than a restatement of "is the name known".

    Args:
        registry: The registry to look the backend up in.
        backend_type: Backend name or registration alias.

    Returns:
        True when the backend is registered and its driver is installed.
    """
    return registry.is_registered(backend_type)


def backend_info(registry: PluginRegistry[Any], backend_type: str) -> dict[str, Any]:
    """Registry metadata for one backend, by any spelling of its name.

    Answers for a backend that is known but not creatable here, which is
    the case the answer is actually needed in: ``requires_install`` is only
    ever read by someone who does not have the backend installed.

    Args:
        registry: The registry to look the backend up in.
        backend_type: Backend name or alias. Case is normalised by the
            registry.

    Returns:
        The metadata dict, or a two-key dict describing the failure when
        the name is not one this registry knows.
    """
    metadata = registry.get_metadata(backend_type, follow_alias=True)
    if metadata:
        return metadata
    return {
        "description": "Unknown backend",
        "error": f"Backend '{backend_type}' not recognized",
    }


def build_backend(
    backend_class: Any,
    options: dict[str, Any],
    *,
    kind: str,
    backend_type: str,
) -> Any:
    """Construct a registered backend through its ``from_config``.

    The one place the database factories narrow what a registry handed
    back. A registry stores ``type[T] | Callable[..., T]`` -- a plain
    function is a legitimate registration, and ``PluginRegistry.create``
    supports one -- while these factories require the class form. Calling
    ``from_config`` on the union without checking produced an
    ``AttributeError`` naming ``'function' object`` from inside the
    factory, with nothing naming the registration responsible.

    Checking here rather than in each factory is what keeps the narrowing
    a single decision: both database factories reach the same union by the
    same route and required the same thing of it.

    Args:
        backend_class: Whatever the registry returned for the name.
        options: The config to build from, ``backend`` already removed.
        kind: What is being built, for the error text.
        backend_type: The resolved name, so the error names the
            registration rather than only the failure.

    Returns:
        The constructed backend.

    Raises:
        ValueError: The registered object cannot be built from a config.
    """
    from_config = getattr(backend_class, "from_config", None)
    if from_config is None:
        registered = getattr(backend_class, "__name__", type(backend_class).__name__)
        raise ValueError(
            f"The {kind} backend registered as '{backend_type}' cannot be built "
            f"from a config: {registered} has no 'from_config' classmethod. "
            "Register a backend class rather than a bare callable, or build it "
            "through the registry's own create()."
        )
    return from_config(options)


def select_backend(
    config: dict[str, Any],
    registry: PluginRegistry[Any],
    *,
    kind: str,
    unknown_message: Callable[[str, str], str] | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Read ``backend`` from ``config``, log its provenance, and resolve it.

    Args:
        config: The factory's config. Not modified -- the returned config is
            a copy without the ``backend`` key, since that key is the
            factory's own discriminator rather than something a backend
            constructor accepts.
        registry: The registry to resolve the name against.
        kind: What is being built, for the log and the default error text --
            ``"database"``, ``"async database"``, ``"vector store"``.
        unknown_message: Builds the message for an unregistered backend,
            from the requested name and the rendered list of available ones.
            Defaults to the shape the sync factories use. Supplied where a
            factory means something more specific by "not found" than "you
            typed it wrong".

    Returns:
        The registered factory, the resolved backend name, and the config
        with ``backend`` removed. The factory is returned untyped because
        the registries here are parameterised by the backend class rather
        than by an instance of it, and each caller invokes it in its own way
        -- ``from_config`` for the databases, direct construction for the
        vector stores.

    Raises:
        ValueError: The ``backend`` key names nothing usable, or names a
            backend this registry does not have.
    """
    remaining = {key: value for key, value in config.items() if key != "backend"}

    if "backend" in config:
        backend_type = normalize_backend(config["backend"])
        logger.info("Creating %s with backend: %s", kind, backend_type)
    else:
        backend_type = DEFAULT_BACKEND
        logger.warning(
            "No 'backend' key in this %s config; falling back to '%s'. That "
            "default is %s. If this config came from resolving a resource "
            "reference, check that the named resource is defined in this "
            "environment.",
            kind,
            DEFAULT_BACKEND,
            DEFAULT_BACKEND_CONSEQUENCE,
        )

    backend_class = registry.get_factory(backend_type)
    if not backend_class:
        rendered = ", ".join(available_backends(registry))
        # A backend the registry knows but cannot build says why. Sending
        # the reader to look for a typo in a correctly spelled name is the
        # one answer that helps nobody.
        declared = registry.get_metadata(backend_type, follow_alias=True)
        hint = declared.get("requires_install") if declared else None
        if hint:
            raise ValueError(
                f"Backend '{backend_type}' is known but not available here. Install with: {hint}"
            )
        if unknown_message is None:
            raise ValueError(
                f"Unknown backend type: {backend_type}. Available backends: {rendered}"
            )
        raise ValueError(unknown_message(backend_type, rendered))

    return backend_class, backend_type, remaining

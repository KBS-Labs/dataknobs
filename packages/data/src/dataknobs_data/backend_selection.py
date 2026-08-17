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

This module imports nothing from its own package, so it is safe to import
from anywhere in it, including the modules involved in the
``vector/stores`` import cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from dataknobs_common.registry import PluginRegistry


logger = logging.getLogger(__name__)


#: The backend every factory here falls back to when a config names none.
#: Declared once so the construction path and the validation path that
#: mirrors it cannot drift; holding no second copy is the guarantee.
DEFAULT_BACKEND = "memory"


def available_backends(registry: PluginRegistry[Any]) -> list[str]:
    """Canonical backend names, sorted, with aliases collapsed.

    A registry lists every spelling it accepts, so ``list_keys()`` reports
    ``postgres``, ``postgresql`` and ``pg`` as three entries for one backend.
    That is right for the lookup it serves and wrong for a list shown to
    someone choosing between backends.

    Keys registered for the same factory form one group, and the name
    reported for the group is the one carrying registry metadata -- which is
    how the aliases in this package are registered, the canonical key taking
    the metadata and the aliases taking none. A group where no key has
    metadata, or several do, reports its first-registered key, so a custom
    registration that follows a different convention still yields one name
    per backend rather than an error.

    Args:
        registry: The registry to report on.

    Returns:
        Sorted canonical names, one per registered backend.
    """
    groups: dict[int, list[str]] = {}
    for key in registry.list_keys():
        # Grouped by the factory's identity rather than by the factory
        # itself: a registered callable need not be hashable, and the
        # registry holds every one of them for the process lifetime, so an
        # id cannot be reused underneath us.
        groups.setdefault(id(registry.get_factory(key)), []).append(key)

    names = []
    for keys in groups.values():
        described = [key for key in keys if registry.get_metadata(key)]
        names.append(described[0] if described else keys[0])
    return sorted(names)


def backend_info(registry: PluginRegistry[Any], backend_type: str) -> dict[str, Any]:
    """Registry metadata for one backend, by any spelling of its name.

    An alias carries no metadata of its own, so asking about ``pg`` used to
    return an empty dict while every other question about it answered for
    postgres. The alias is resolved to the key that describes the same
    factory, so the answers agree.

    Args:
        registry: The registry to look the backend up in.
        backend_type: Backend name or alias. Case is normalised by the
            registry.

    Returns:
        The metadata dict, or a two-key dict describing the failure when the
        name is not registered at all.
    """
    if not registry.is_registered(backend_type):
        return {
            "description": "Unknown backend",
            "error": f"Backend '{backend_type}' not recognized",
        }

    metadata = registry.get_metadata(backend_type)
    if metadata:
        return metadata

    factory = registry.get_factory(backend_type)
    for key in registry.list_keys():
        if registry.get_factory(key) is factory:
            described = registry.get_metadata(key)
            if described:
                return described
    return metadata


def select_backend(
    config: dict[str, Any],
    registry: PluginRegistry[Any],
    *,
    kind: str,
    unknown_message: Callable[[str, str], str] | None = None,
) -> tuple[Any, str]:
    """Pop ``backend`` from ``config``, log its provenance, and resolve it.

    Args:
        config: The factory's config. The ``backend`` key is removed, since
            it is the factory's own discriminator rather than something a
            backend constructor accepts.
        registry: The registry to resolve the name against.
        kind: What is being built, for the log and the default error text --
            ``"database"``, ``"async database"``, ``"vector store"``.
        unknown_message: Builds the message for an unregistered backend,
            from the requested name and the rendered list of available ones.
            Defaults to the shape the sync factories use. Supplied where a
            factory means something more specific by "not found" than "you
            typed it wrong".

    Returns:
        The registered factory and the resolved backend name. The factory is
        returned untyped because the registries here are parameterised by the
        backend class rather than by an instance of it, and each caller
        invokes it in its own way -- ``from_config`` for the databases,
        direct construction for the vector stores.

    Raises:
        ValueError: The named backend is not registered.
    """
    if "backend" in config:
        backend_type = str(config.pop("backend")).lower()
        logger.info("Creating %s with backend: %s", kind, backend_type)
    else:
        backend_type = DEFAULT_BACKEND
        logger.warning(
            "No 'backend' key in this %s config; falling back to '%s'. That "
            "default is in-process and unpersisted -- it answers every query "
            "with zero results until something writes to it, and loses "
            "everything when the process restarts. If this config came from "
            "resolving a resource reference, check that the named resource is "
            "defined in this environment.",
            kind,
            DEFAULT_BACKEND,
        )

    backend_class = registry.get_factory(backend_type)
    if not backend_class:
        rendered = ", ".join(available_backends(registry))
        if unknown_message is None:
            raise ValueError(
                f"Unknown backend type: {backend_type}. Available backends: {rendered}"
            )
        raise ValueError(unknown_message(backend_type, rendered))

    return backend_class, backend_type

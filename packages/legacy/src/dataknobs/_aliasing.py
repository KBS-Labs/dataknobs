"""Publish a modular package's submodules under the legacy dotted namespace.

Each shim in this package re-exports from a modular package by importing its
submodules, which binds them as attributes of the shim. That is enough for
``from dataknobs.structures import tree``, but not for
``from dataknobs.structures.tree import Tree`` -- the form pre-split code
actually contains -- because Python resolves a dotted module path through
``sys.modules`` rather than through the parent's attributes.

Registering the alias makes both forms work, which is what backward
compatibility means for a package that exists only to provide it. Held here
rather than repeated in each shim: all three had the same shape and so had the
same gap.
"""

from __future__ import annotations

import sys
from types import ModuleType


def alias_submodules(package: str, submodules: tuple[ModuleType, ...]) -> None:
    """Register each already-imported submodule under ``package``'s namespace.

    Args:
        package: The legacy package re-exporting them, normally ``__name__``.
        submodules: Modules imported from the modular package. They are aliased
            under their own final name, so ``dataknobs_utils.json_utils``
            becomes reachable as ``dataknobs.utils.json_utils``.

    The alias is the module object itself, not a copy, so the two import paths
    share identity and module-level state.
    """
    for module in submodules:
        sys.modules[f"{package}.{module.__name__.rpartition('.')[2]}"] = module

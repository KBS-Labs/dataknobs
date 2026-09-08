"""A package door publishes by two mechanisms, and they must agree.

``__all__`` is the list a reader consults and ``from … import *`` obeys. It is
not the only thing a door publishes: the API reference renders from the
**module object**, so a name merely *imported* into ``__init__.py`` appears on
the published page whether or not the list mentions it. A construct that is
still being designed can therefore reach consumers through an import line
nobody read as an export.

These tests assert the two agree, at both doors — the package's own and the
ontology subpackage's — so that publishing a name stays a deliberate act with
one obvious spelling.

**Why submodules are excluded rather than listed.** ``from .async_iter import
…`` binds ``async_iter`` on the package as a side effect, so ``dir()`` carries
every submodule any import has reached. That set is not stable: it grows when
some *other* test imports a submodule first, so an allowlist would be red for
reasons having nothing to do with the door. Excluding by type is also exactly
what the renderer does — mkdocstrings' ``show_submodules`` is unset and
defaults to false — so the exclusion here and the omission there are one rule
stated twice.

Distinct from ``test_packs.py``'s pair, which asserts that one module's exports
are a **subset** of the top-level's. That is a re-export claim about a specific
module. This is an equality claim about the door itself.
"""

from __future__ import annotations

import pkgutil
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

import dataknobs_common
import dataknobs_common.ontology

if TYPE_CHECKING:
    from collections.abc import Iterator

DOORS = [dataknobs_common, dataknobs_common.ontology]


def _public_bindings(door: ModuleType) -> set[str]:
    """Every public name the door binds that the reference would render."""
    return {
        name
        for name in dir(door)
        if not name.startswith("_") and not isinstance(getattr(door, name), ModuleType)
    }


def _declared(door: ModuleType) -> set[str]:
    """``__all__`` without the dunders, which are metadata rather than surface."""
    return {name for name in door.__all__ if not name.startswith("_")}


def _own_submodules(door: ModuleType) -> set[str]:
    """The package's submodules, read off the filesystem.

    An *independent* source, which is the entire point of it. Deriving the
    expected set from ``isinstance(..., ModuleType)`` -- the property the
    exclusion already filters by -- is what made the check below unfalsifiable.
    """
    return {name for _, name, _ in pkgutil.iter_modules(door.__path__)}


@pytest.fixture(params=DOORS, ids=lambda door: door.__name__)
def door(request: pytest.FixtureRequest) -> Iterator[ModuleType]:
    """Each package door in turn."""
    yield request.param


def test_a_door_binds_exactly_what_it_declares(door: ModuleType) -> None:
    """Neither mechanism publishes a name the other one does not.

    The direction that motivates this is ``bound - declared``: an import line
    added to reach a collaborator publishes that collaborator on the rendered
    page, silently, and no lint rule objects. ``declared - bound`` is ruff's
    F822 and is already enforced -- asserted here anyway, because an equality
    reads as one rule where two subset assertions read as two.
    """
    bound = _public_bindings(door)
    declared = _declared(door)

    assert bound - declared == set(), (
        f"{sorted(bound - declared)} are bound on {door.__name__} but absent "
        f"from its __all__ — the reference renders from the module object, so "
        f"they publish regardless. Export them deliberately or import them "
        f"under a private alias"
    )
    assert declared - bound == set(), (
        f"{sorted(declared - bound)} are named in {door.__name__}.__all__ and bound nowhere on it"
    )


def test_the_excluded_names_are_submodules_and_nothing_else(door: ModuleType) -> None:
    """The exclusion's licence: it drops this package's own submodules, nothing else.

    Stated separately because the exclusion is the one place this guard could
    quietly stop guarding. If a name that is not a submodule of this package
    ever landed in the excluded set, the assertion above would pass while the
    name it was meant to catch went unexamined.

    **Checked against the filesystem, and it has to be.** The set is compared
    to ``pkgutil.iter_modules`` rather than to ``isinstance(..., ModuleType)``,
    because the exclusion is *defined* by that isinstance -- so an expectation
    written the same way is the same expression on both sides of an equals
    sign and cannot fail for any door, ever. That is what this test used to be,
    on both of its assertions, and
    ``test_the_exclusion_guard_can_actually_fail`` below is the input that
    proves it is no longer.

    Subset rather than equality: a submodule nothing has imported yet is on
    disk and absent from ``dir()``, which is the stable direction to assert.
    """
    excluded = {name for name in dir(door) if not name.startswith("_")} - _public_bindings(door)
    foreign = excluded - _own_submodules(door)

    assert foreign == set(), (
        f"{sorted(foreign)} are dropped from {door.__name__}'s export check but are "
        f"not submodules of it — the exclusion mirrors the reference renderer, which "
        f"omits a package's own submodules and nothing else, so these publish "
        f"unexamined"
    )


def test_the_version_is_the_only_dunder_a_door_declares() -> None:
    """``__version__`` is the reason ``__all__`` is filtered before comparing.

    A second dunder would mean the filter is doing more than this one
    documented exception, which is worth failing on rather than absorbing.
    """
    assert [n for n in dataknobs_common.__all__ if n.startswith("_")] == ["__version__"]
    assert [n for n in dataknobs_common.ontology.__all__ if n.startswith("_")] == []


def _a_door_binding_a_foreign_module() -> ModuleType:
    """A door that publicly binds a module which is not one of its submodules.

    ``import json`` at the top of an ``__init__.py`` produces exactly this: a
    public, module-typed binding that the exclusion drops. It is the input the
    exclusion's licence is a claim about, so it is the input that decides
    whether the guard above is a guard.
    """
    door = ModuleType("a_door_binding_a_foreign_module")
    door.__path__ = list(dataknobs_common.__path__)  # type: ignore[attr-defined]
    door.__all__ = ["Published"]  # type: ignore[attr-defined]
    door.Published = object()  # type: ignore[attr-defined]
    door.pytest = pytest  # type: ignore[attr-defined]
    return door


def test_the_exclusion_guard_can_actually_fail() -> None:
    """The guard above must reject a door whose exclusion drops a foreign name.

    Written because the assertion it checks was true by construction.
    ``excluded`` was ``{public names} - {public names that are not modules}``,
    which *is* ``{public names that are modules}`` -- so asserting that
    everything in it is a module could not fail for any door, ever, and neither
    could the equality beneath it, which compared the same expression to itself.
    The half of the test carrying the argument was the half that was dead.

    A guard is only a guard if some input makes it red. This is that input, and
    it is not contrived: the exclusion exists to mirror the reference renderer,
    which omits a package's *own* submodules, so a module the package merely
    imported is precisely the name that should not have been dropped silently.
    """
    with pytest.raises(AssertionError):
        test_the_excluded_names_are_submodules_and_nothing_else(_a_door_binding_a_foreign_module())

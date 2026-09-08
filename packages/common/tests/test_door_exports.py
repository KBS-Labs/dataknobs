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
    """The exclusion's licence: it drops modules, never a construct.

    Stated separately because the exclusion is the one place this guard could
    quietly stop guarding. If a public *construct* ever landed in the excluded
    set, the assertion above would pass while the name it was meant to catch
    went unexamined.
    """
    excluded = {name for name in dir(door) if not name.startswith("_")} - _public_bindings(door)

    assert all(isinstance(getattr(door, name), ModuleType) for name in excluded)
    assert excluded == {
        name
        for name in dir(door)
        if not name.startswith("_") and isinstance(getattr(door, name), ModuleType)
    }


def test_the_version_is_the_only_dunder_a_door_declares() -> None:
    """``__version__`` is the reason ``__all__`` is filtered before comparing.

    A second dunder would mean the filter is doing more than this one
    documented exception, which is worth failing on rather than absorbing.
    """
    assert [n for n in dataknobs_common.__all__ if n.startswith("_")] == ["__version__"]
    assert [n for n in dataknobs_common.ontology.__all__ if n.startswith("_")] == []

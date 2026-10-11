"""Every ``StructuredConfig`` in the workspace is unhashable, unless it says why.

``StructuredConfig.__init_subclass__`` declares each subclass unhashable before
the subclass's ``@dataclass`` runs, and leaves alone a subclass whose body
writes ``__hash__`` itself. This census asks every subclass in the six packages
that define one, by reading the class attribute rather than building a value,
so a subclass the dataclass sweeps cannot construct is covered too.

Run from the workspace root because the family spans six packages, and only
two of them carry a hashability sweep of their own.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import DataclassSweep

#: The packages that define a ``StructuredConfig`` subclass.
PACKAGES = (
    "dataknobs_common",
    "dataknobs_config",
    "dataknobs_data",
    "dataknobs_llm",
    "dataknobs_bots",
    "dataknobs_fsm",
)

#: Subclasses that write their own ``__hash__``, each with the reason.
#:
#: Fails on an entry that no longer writes one, so the list cannot outlive
#: its reasons.
OWN_HASH: dict[str, str] = {
    # A stored version is identified by its number: two records of one
    # version compare equal and hash alike whatever their payloads hold,
    # which is what a version history keyed by number needs.
    "dataknobs_bots.config.versioning.ConfigVersion": "identity is the version number",
}


def _every_subclass(root: type) -> set[type]:
    """Every class below ``root`` that exists in this process, at any depth.

    Less the classes ``@dataclass(slots=True)`` replaced: it builds a new
    class from the decorated one, and the one it replaced stays registered
    as a subclass, under the same name, though nothing can reach it.
    """
    found: set[type] = set()
    stack: list[type] = list(root.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in found:
            found.add(cls)
            stack.extend(cls.__subclasses__())
    slotted = {(cls.__module__, cls.__qualname__) for cls in found if "__slots__" in cls.__dict__}
    return {
        cls
        for cls in found
        if "__slots__" in cls.__dict__ or (cls.__module__, cls.__qualname__) not in slotted
    }


def _beyond_the_sweep(swept: Iterable[type], existing: Iterable[type]) -> list[str]:
    """Package-defined configs that exist but that the module-level sweep missed."""
    reached = set(swept)
    return sorted(
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in existing
        if cls not in reached and cls.__module__.split(".", 1)[0] in PACKAGES
    )


def _writes_its_own_eq(cls: type) -> bool:
    """Whether ``cls``'s body wrote its ``__eq__``, rather than its ``@dataclass``.

    The decorator writes the ``__eq__`` it generates into the class dict, so
    presence there says nothing. What it generates is compiled from a string,
    so its code names no source file.
    """
    eq = cls.__dict__.get("__eq__")
    code = getattr(eq, "__code__", None)
    return code is not None and not code.co_filename.startswith("<")


def _hashed_by_an_own_hash(cls: type, configs: dict[str, type]) -> bool:
    """Whether ``cls`` hashes by an ``OWN_HASH`` entry's hash, written or inherited."""
    return any(cls.__hash__ is configs[name].__hash__ for name in OWN_HASH if name in configs)


def _hash_beside_a_generated_equality(cls: type) -> bool:
    """Whether ``cls`` writes a hash and leaves its equality to the dataclass."""
    params = cls.__dict__.get("__dataclass_params__")
    return params is not None and params.eq and not _writes_its_own_eq(cls)


@pytest.fixture(scope="module")
def configs() -> dict[str, type]:
    """Every ``StructuredConfig`` subclass the six packages define, by qualified name."""
    found: dict[str, type] = {}
    failures: list[tuple[str, str]] = []
    for name in PACKAGES:
        sweep = DataclassSweep(importlib.import_module(name))
        for key, cls in sweep.every_dataclass().items():
            if issubclass(cls, StructuredConfig) and cls is not StructuredConfig:
                found[f"{name}.{key}"] = cls
        failures += sweep.import_failures()
    assert failures == [], f"modules the census could not import: {failures}"
    return found


def test_the_census_reaches_every_package(configs: dict[str, type]) -> None:
    """A package contributing nothing would make the census vacuous for it."""
    reached = {name.split(".", 1)[0] for name in configs}
    assert reached == set(PACKAGES), f"no StructuredConfig found in {set(PACKAGES) - reached}"


def test_the_sweep_reaches_every_config_that_exists(configs: dict[str, type]) -> None:
    """A config the module-level walk cannot see would escape the census.

    ``DataclassSweep`` reads module attributes, so a class defined inside a
    function or another class is invisible to it. Every subclass that exists
    once the packages are imported is listed by ``__subclasses__()``, so the
    two are compared.
    """
    missed = _beyond_the_sweep(configs.values(), _every_subclass(StructuredConfig))
    assert missed == [], f"StructuredConfig subclass(es) the census cannot see: {missed}"


def test_the_cross_check_would_report_a_config_out_of_reach() -> None:
    """Positive control: the comparison above is not vacuous."""

    class Hidden:
        __module__ = "dataknobs_common._out_of_reach"

    class Elsewhere:
        __module__ = "a_consumer.configs"

    assert _beyond_the_sweep([], [Hidden, Elsewhere]) == [
        "dataknobs_common._out_of_reach.test_the_cross_check_would_report_a_config_out_of_reach"
        ".<locals>.Hidden"
    ]


def test_every_config_is_unhashable(configs: dict[str, type]) -> None:
    hashable = sorted(
        name
        for name, cls in configs.items()
        if cls.__hash__ is not None and not _hashed_by_an_own_hash(cls, configs)
    )
    assert hashable == [], (
        f"StructuredConfig subclass(es) claim to be hashable: {hashable}. A config "
        "compares field by field, so it is unhashable unless its body writes a "
        "__hash__ consistent with its own equality; if it does, name it in OWN_HASH "
        "with the reason."
    )


def test_every_own_hash_is_still_written(configs: dict[str, type]) -> None:
    stale = sorted(
        name
        for name in OWN_HASH
        if name not in configs or configs[name].__dict__.get("__hash__") is None
    )
    assert stale == [], f"OWN_HASH names a config that no longer writes __hash__: {stale}"


def test_every_own_hash_comes_with_its_own_equality(configs: dict[str, type]) -> None:
    """A hand-written hash is stated together with the equality it serves.

    The two together say what makes two configs the same, and they agree only
    if every pair of equal configs hashes alike. Beside a generated field
    equality that holds only while the hash reads nothing but fields, which
    nothing here can check. So each entry writes ``__eq__`` too, where the two
    are read together, or declares ``eq=False`` and so compares by identity.
    """
    unpaired = sorted(
        name
        for name in OWN_HASH
        if name in configs and _hash_beside_a_generated_equality(configs[name])
    )
    assert unpaired == [], (
        f"OWN_HASH config(s) hash by hand beside a generated equality: {unpaired}"
    )


def test_the_pairing_check_would_report_a_hash_beside_a_generated_equality() -> None:
    """Positive control: the check above is not vacuous."""

    @dataclass(frozen=True)
    class HashOnly(StructuredConfig):
        version: int = 1

        def __hash__(self) -> int:
            return hash(self.version)

    @dataclass(frozen=True)
    class Paired(StructuredConfig):
        version: int = 1

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Paired) and other.version == self.version

        def __hash__(self) -> int:
            return hash(self.version)

    @dataclass(frozen=True, eq=False)
    class Identity(StructuredConfig):
        version: int = 1

        __hash__ = object.__hash__

    assert _hash_beside_a_generated_equality(HashOnly)
    assert not _hash_beside_a_generated_equality(Paired)
    assert not _hash_beside_a_generated_equality(Identity)


def test_a_class_slots_replaced_is_not_counted() -> None:
    """Positive control: the replaced class is still registered, and is dropped."""

    @dataclass(frozen=True, slots=True)
    class Slotted(StructuredConfig):
        n: int = 0

    def named(classes: Iterable[type]) -> list[type]:
        return [cls for cls in classes if cls.__qualname__ == Slotted.__qualname__]

    assert len(named(StructuredConfig.__subclasses__())) == 2
    assert named(_every_subclass(StructuredConfig)) == [Slotted]

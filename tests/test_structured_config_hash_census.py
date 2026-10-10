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


def test_every_config_is_unhashable(configs: dict[str, type]) -> None:
    hashable = sorted(
        name for name, cls in configs.items() if cls.__hash__ is not None and name not in OWN_HASH
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

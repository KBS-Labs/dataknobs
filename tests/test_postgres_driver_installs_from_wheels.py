"""The Postgres backend's extra installs from wheels, and supplies what it imports.

``dataknobs-data[postgres]`` used to require the ``psycopg2`` distribution. That
is an sdist on every platform but Windows, and building it needs ``pg_config``
from the libpq development headers, so ``pip install dataknobs-data[postgres]``
failed on a macOS or Linux machine without them. ``dataknobs-bots[postgres]``
and ``dataknobs-fsm[postgres]`` forward to that extra and failed with it.

The same module ships as the ``psycopg2-binary`` distribution, which has wheels,
and ``dataknobs-utils`` already required it as a base dependency. So the sdist
added a second copy of one module and a compiler requirement, and no capability.

A packaging failure cannot be reproduced by a unit test without a machine that
lacks the headers. What can be pinned is its cause, which a later re-pin could
restore without anyone building on such a machine:

- no workspace manifest requires the ``psycopg2`` distribution, anywhere;
- the extra the backend's ``requires_install`` hint names supplies every module
  its ``requires_module`` lists, so the hint a user is shown is true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from dataknobs_data import AsyncDatabaseFactory, DatabaseFactory
from tests._workspace import load_toml, pyprojects, rel

#: The distribution that must not be required, and the one that replaces it.
#:
#: The binary's floor is ``dataknobs-utils``' floor, for the reason that floor
#: exists: releases before 2.9.10 carry no cp312/cp313 wheels, and this
#: workspace's Python floor is 3.12, so a lower floor reopens the source build
#: under a lowest-direct resolution.
SDIST = "psycopg2"
BINARY = "psycopg2-binary"

#: Import name -> the distribution that supplies it here.
#:
#: Written out rather than looked up in installed metadata, because the
#: question is what the *manifest* promises, and an installed environment
#: answers what happens to be present. A module missing from this table fails
#: the guard rather than passing it.
DISTRIBUTION_FOR_MODULE = {
    "psycopg2": BINARY,
    "asyncpg": "asyncpg",
}


def _requirement_strings(project: dict[str, Any]) -> list[tuple[str, str]]:
    """Every requirement a manifest declares, with where it is declared."""
    found: list[tuple[str, str]] = []
    body = project.get("project", {})
    found.extend(("dependencies", r) for r in body.get("dependencies", []))
    for extra, requirements in body.get("optional-dependencies", {}).items():
        found.extend((f"optional-dependencies.{extra}", r) for r in requirements)
    for group, requirements in project.get("dependency-groups", {}).items():
        # An ``{include-group = ...}`` entry is a table, not a requirement;
        # the group it names is read under its own key.
        found.extend((f"dependency-groups.{group}", r) for r in requirements if isinstance(r, str))
    return found


@pytest.mark.parametrize("manifest", pyprojects(), ids=rel)
def test_no_manifest_requires_the_source_built_driver(manifest: Path) -> None:
    offenders = [
        f"{where}: {text}"
        for where, text in _requirement_strings(load_toml(manifest))
        if canonicalize_name(Requirement(text).name) == SDIST
    ]
    assert not offenders, (
        f"{rel(manifest)} requires the {SDIST!r} distribution, which builds from "
        f"source and needs pg_config. Require {BINARY!r} instead: it is the same "
        f"module, from wheels.\n  " + "\n  ".join(offenders)
    )


def _workspace_manifests() -> dict[str, dict[str, Any]]:
    """Each workspace distribution's manifest, by its canonical name."""
    manifests: dict[str, dict[str, Any]] = {}
    for path in pyprojects():
        data = load_toml(path)
        name = data.get("project", {}).get("name")
        if name:
            manifests[canonicalize_name(name)] = data
    return manifests


def _declared_by_extras(requirement: Requirement) -> set[str]:
    """The external distributions a workspace requirement's extras declare.

    Only extras are read, never a package's base dependencies. A base
    dependency elsewhere in the workspace supplies a module too, and that is
    the accident this guards against: ``dataknobs-utils`` requires the binary
    as a base dependency today, which is a separate decision that may move.
    An extra that forwards to another workspace package's extra, as ``bots``
    and ``fsm`` do to ``data``'s, is followed.
    """
    manifests = _workspace_manifests()
    declared: set[str] = set()
    pending = [requirement]
    seen: set[tuple[str, frozenset[str]]] = set()
    while pending:
        current = pending.pop()
        name = canonicalize_name(current.name)
        key = (name, frozenset(current.extras))
        if key in seen:
            continue
        seen.add(key)
        if name not in manifests:
            declared.add(name)
            continue
        extras = manifests[name]["project"].get("optional-dependencies", {})
        for extra in current.extras:
            assert extra in extras, f"{name} has no extra {extra!r}"
            pending.extend(Requirement(text) for text in extras[extra])
    return declared


@pytest.mark.parametrize(
    "factory", [DatabaseFactory(), AsyncDatabaseFactory()], ids=["sync", "async"]
)
def test_the_install_hint_supplies_every_module_the_backend_imports(factory: Any) -> None:
    info = factory.get_backend_info("postgres")
    hint = info["requires_install"]
    prefix = "pip install "
    assert hint.startswith(prefix), f"unrecognised install hint: {hint!r}"
    requirement = Requirement(hint.removeprefix(prefix))

    supplied = _declared_by_extras(requirement)

    unmapped = [m for m in info["requires_module"] if m not in DISTRIBUTION_FOR_MODULE]
    assert not unmapped, f"add these modules to DISTRIBUTION_FOR_MODULE: {unmapped}"
    missing = [
        f"{module} (from {DISTRIBUTION_FOR_MODULE[module]})"
        for module in info["requires_module"]
        if DISTRIBUTION_FOR_MODULE[module] not in supplied
    ]
    assert not missing, (
        f"{hint!r} does not install what the postgres backend imports: {missing}. "
        f"It installs {sorted(supplied)}."
    )

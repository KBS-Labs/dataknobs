"""The licensing surface is one statement, repeated exactly.

Apache-2.0 requires the license and the NOTICE to travel with every
distribution, and a wheel is built from one package directory — so each package
carries its own copy of both files rather than pointing at the root. Symlinks
would have been the way to avoid the duplication and cannot be used: a symlink
escaping the package directory makes the sdist unextractable, which is a build
failure rather than a style problem.

Copies drift. These tests are what keeps every copy of it one statement — and
the count is deliberately not written down here, because a number in a
docstring is one more copy to drift.

This file carries no SPDX header of its own. It never ships, and it is the file
that defines what "shipped source" means, so an exception written into it would
be the first thing to confuse a reader of that definition.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

from tests._workspace import ROOT, load_bin_module, load_toml, pyprojects, rel

#: Every directory under ``packages/`` that builds a distribution.
PACKAGE_DIRS = sorted(p for p in (ROOT / "packages").iterdir() if (p / "pyproject.toml").is_file())

#: The attribution, written once. It appears in the SPDX header on every shipped
#: source, in NOTICE, in the historical MIT text, and in the header the package
#: scaffolder emits — four places that have to say the same thing, and that a
#: year-range bump would otherwise have to find one at a time.
COPYRIGHT = "Copyright 2022-2026 KBS Labs"

#: The header every shipped source file carries, as its first two lines.
SPDX_HEADER = f"# SPDX-FileCopyrightText: {COPYRIGHT}\n# SPDX-License-Identifier: Apache-2.0\n"

#: The two files ``license-files`` names in every manifest, and that every
#: package therefore carries a copy of.
LICENSE_FILES = ("LICENSE", "NOTICE")

#: One row of the historical version table: a package label, then its version.
#: The label may carry a parenthesised qualifier — ``dataknobs (legacy)`` — so
#: the version is anchored to the end of the line rather than taken as the
#: second field.
_MIT_ROW = re.compile(r"^\s{2,}(?P<label>\S.*?)\s+(?P<version>\d+\.\d+\.\d+)\s*$")


def _shipped_sources() -> list[Path]:
    """Every ``.py`` file that ends up inside a published distribution."""
    sources: list[Path] = []
    for package in PACKAGE_DIRS:
        sources.extend(sorted((package / "src").rglob("*.py")))
    sources.extend(sorted((ROOT / "src").rglob("*.py")))
    return sources


def _registry_versions() -> dict[str, str]:
    """Each published distribution's current version, by PyPI name."""
    registry = json.loads((ROOT / ".dataknobs" / "packages.json").read_text(encoding="utf-8"))
    return {entry["pypi_name"]: entry["version"] for entry in registry["packages"]}


def _historical_versions() -> dict[str, str]:
    """The last MIT-licensed release of each package, from the historical record."""
    text = (ROOT / "LICENSES/MIT-historical.txt").read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for line in text.splitlines():
        row = _MIT_ROW.match(line)
        if row is not None:
            # "dataknobs (legacy)" names the distribution "dataknobs"; the
            # qualifier is there for a reader, not for the registry.
            found[row.group("label").split()[0]] = row.group("version")
    return found


def test_the_root_license_is_apache_2_0() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "END OF TERMS AND CONDITIONS" in text


def test_the_mit_text_is_preserved_with_its_non_retroactivity_stated() -> None:
    """The MIT grant on already-published versions is not withdrawn by this change.

    The file has to carry both halves to do its job: the licence text itself,
    and the statement that it still governs the releases made under it.
    """
    text = (ROOT / "LICENSES/MIT-historical.txt").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Permission is hereby granted, free of charge" in text
    assert "remains available under the MIT License" in text
    assert "not retroactive" in text


def test_the_historical_version_list_names_every_package() -> None:
    """A package missing from the table has no recorded last-MIT release.

    The consequence is not cosmetic: someone holding an older copy reads this
    file to learn which terms it came under, and a package absent from it
    answers nothing.
    """
    listed, published = _historical_versions(), _registry_versions()
    assert listed, "no version rows parsed from LICENSES/MIT-historical.txt"

    missing = sorted(set(published) - set(listed))
    unknown = sorted(set(listed) - set(published))
    assert not missing, f"packages with no last-MIT release recorded: {missing}"
    assert not unknown, f"the historical table names distributions that do not exist: {unknown}"


def test_no_historical_version_is_ahead_of_the_registry() -> None:
    """The frozen record cannot name a release that was never made.

    Deliberately ``<=`` rather than equality, which is the obvious assertion and
    the wrong one. The table is a *record* of the last MIT release; the registry
    is the *current* version and moves on the first Apache release after this
    lands. They are equal only until then, so an equality check would fail on
    the release rather than on the drift it was meant to catch — and the fix for
    that failure would be to edit the historical record, which is the one thing
    this file exists to prevent.
    """
    listed, published = _historical_versions(), _registry_versions()
    ahead = {
        name: (version, published[name])
        for name, version in listed.items()
        if name in published and Version(version) > Version(published[name])
    }
    assert not ahead, (
        "LICENSES/MIT-historical.txt records a last-MIT release later than the "
        f"version the registry publishes, so it names a release nobody made: {ahead}"
    )


@pytest.mark.parametrize("package", PACKAGE_DIRS, ids=lambda p: p.name)
@pytest.mark.parametrize("name", LICENSE_FILES)
def test_each_package_ships_the_root_text_verbatim(package: Path, name: str) -> None:
    copy = package / name
    assert copy.is_file(), f"{copy} is missing — its distribution would ship no {name}"
    assert not copy.is_symlink(), (
        f"{copy} is a symlink; an sdist containing one that escapes the package "
        "directory cannot be extracted"
    )
    assert copy.read_text(encoding="utf-8") == (ROOT / name).read_text(encoding="utf-8"), (
        f"{copy} has drifted from the root {name}"
    )


@pytest.mark.parametrize("manifest", pyprojects(), ids=rel)
def test_each_manifest_declares_apache_2_0(manifest: Path) -> None:
    """PEP 639 metadata, and the ``License ::`` classifier it forbids alongside it.

    Every manifest, the root one included. It publishes nothing, so its
    declaration reaches no user — but it is the file a new package's is copied
    from, and a wrong expression there propagates by being the example.
    """
    data = load_toml(manifest)["project"]
    assert data.get("license") == "Apache-2.0", f"{rel(manifest)} declares {data.get('license')!r}"
    # A superset rather than an exact match: vendoring third-party code would
    # require shipping its licence too, and the correct manifest change for that
    # is a third entry. Equality here would reject it.
    assert set(data.get("license-files", [])) >= set(LICENSE_FILES), (
        f"{rel(manifest)} declares license-files {data.get('license-files')!r}, "
        f"which omits one of {list(LICENSE_FILES)}"
    )
    classifiers = data.get("classifiers", [])
    assert not [c for c in classifiers if c.startswith("License ::")], (
        f"{rel(manifest)} carries a License classifier, which PEP 639 forbids "
        "alongside a license expression and PyPI rejects"
    )


@pytest.mark.parametrize("manifest", pyprojects(), ids=rel)
def test_each_manifest_requires_a_backend_that_understands_pep_639(manifest: Path) -> None:
    """Hatchling gained PEP 639 support in 1.27.

    Without a floor, a build resolving an older backend rejects the two fields
    above. ``uv build`` fetches the latest and so never sees it; a build with
    ``--no-build-isolation``, a lowest-resolution matrix, or a pinned offline
    cache does.
    """
    requires = load_toml(manifest)["build-system"]["requires"]
    floors = [r for r in requires if r.startswith("hatchling")]
    assert floors == ["hatchling>=1.27"], (
        f"{rel(manifest)} requires {floors or requires!r}; PEP 639 metadata needs hatchling>=1.27"
    )


def test_every_shipped_source_carries_the_spdx_header() -> None:
    sources = _shipped_sources()
    assert sources, "no shipped sources found, so this asserts nothing"
    missing = [
        rel(path)
        for path in sources
        if not path.read_text(encoding="utf-8").startswith(SPDX_HEADER)
    ]
    assert not missing, f"{len(missing)} shipped source files lack the SPDX header: {missing[:10]}"


def test_the_attribution_is_one_statement() -> None:
    """The copyright line is in five places; they have to agree.

    Three are compared here against the constant above and the fourth — the
    header on every shipped source — by the test before this one. Without this,
    a year bump moves the shipped headers (which that test reports) and silently
    leaves the NOTICE, the historical record and the scaffolder behind.
    """
    assert COPYRIGHT in (ROOT / "NOTICE").read_text(encoding="utf-8"), (
        f"NOTICE does not carry {COPYRIGHT!r}"
    )
    # The MIT text is a verbatim reproduction, so it carries the licence's own
    # "Copyright (c)" form rather than the SPDX one.
    historical = (ROOT / "LICENSES/MIT-historical.txt").read_text(encoding="utf-8")
    years = COPYRIGHT.removeprefix("Copyright ")
    assert f"Copyright (c) {years}" in historical, (
        f"LICENSES/MIT-historical.txt does not carry the years {years!r}"
    )

    creator = load_bin_module("create-package").PackageCreator(ROOT)
    assert creator._generate_init_py("probe", "0.1.0").startswith(SPDX_HEADER), (
        "bin/create-package.py emits a header that is not SPDX_HEADER, so a new "
        "package fails test_every_shipped_source_carries_the_spdx_header on day one"
    )


def test_a_scaffolded_package_satisfies_this_file() -> None:
    """The scaffolder cannot emit a package these tests would reject.

    It is the only source of new packages, so a drift between its template and
    the rules here is discovered as a failing build on somebody's first day.
    Asserted against the same constants the real packages are, so the two cannot
    be updated apart.
    """
    creator = load_bin_module("create-package").PackageCreator(ROOT)
    generated = tomllib.loads(creator._generate_pyproject_toml("probe", "A probe", "0.1.0"))

    project = generated["project"]
    assert project["license"] == "Apache-2.0"
    assert set(project["license-files"]) >= set(LICENSE_FILES)
    assert not [c for c in project.get("classifiers", []) if c.startswith("License ::")]
    assert generated["build-system"]["requires"] == ["hatchling>=1.27"]

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The licensing surface is one statement, repeated exactly.

Apache-2.0 requires the license and the NOTICE to travel with every
distribution, and a wheel is built from one package directory — so each package
carries its own copy of both files rather than pointing at the root. Symlinks
would have been the way to avoid the duplication and cannot be used: a symlink
escaping the package directory makes the sdist unextractable, which is a build
failure rather than a style problem.

Copies drift. These tests are what keeps the twelve of them one statement.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tests._workspace import ROOT

#: Every directory under ``packages/`` that builds a distribution.
PACKAGE_DIRS = sorted(p for p in (ROOT / "packages").iterdir() if (p / "pyproject.toml").is_file())

#: The header every shipped source file carries, as its first two lines.
SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)


def _shipped_sources() -> list[Path]:
    """Every ``.py`` file that ends up inside a published distribution."""
    sources: list[Path] = []
    for package in PACKAGE_DIRS:
        sources.extend(sorted((package / "src").rglob("*.py")))
    sources.extend(sorted((ROOT / "src").rglob("*.py")))
    return sources


def test_the_root_license_is_apache_2_0() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "END OF TERMS AND CONDITIONS" in text


def test_the_notice_names_the_copyright_holder() -> None:
    text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Copyright 2022-2026 KBS Labs" in text


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


@pytest.mark.parametrize("package", PACKAGE_DIRS, ids=lambda p: p.name)
@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
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


@pytest.mark.parametrize("package", PACKAGE_DIRS, ids=lambda p: p.name)
def test_each_package_declares_apache_2_0(package: Path) -> None:
    """PEP 639 metadata, and the ``License ::`` classifier it forbids alongside it."""
    data = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert data.get("license") == "Apache-2.0", f"{package.name} declares {data.get('license')!r}"
    assert sorted(data.get("license-files", [])) == ["LICENSE", "NOTICE"]
    classifiers = data.get("classifiers", [])
    assert not [c for c in classifiers if c.startswith("License ::")], (
        f"{package.name} carries a License classifier, which PEP 639 forbids "
        "alongside a license expression and PyPI rejects"
    )


def test_every_shipped_source_carries_the_spdx_header() -> None:
    sources = _shipped_sources()
    assert sources, "no shipped sources found, so this asserts nothing"
    missing = [
        str(path.relative_to(ROOT))
        for path in sources
        if not path.read_text(encoding="utf-8").startswith(SPDX_HEADER)
    ]
    assert not missing, f"{len(missing)} shipped source files lack the SPDX header: {missing[:10]}"

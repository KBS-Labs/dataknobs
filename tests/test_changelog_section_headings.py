# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""No package changelog repeats a ``###`` heading inside ``## Unreleased``.

A round of work adds its entries by *prepending* a section --- a new
``### Fixed`` above whatever is already there --- rather than appending to the
``### Fixed`` further down, which is the one already holding this release's
fixed things. Nothing notices, because the section that results is correct
markdown and each block reads correctly on its own. So the duplicates
accumulate one round at a time, and the reader who meets them is the one
writing the release notes.

What it costs is not cosmetic. ``## Unreleased`` in ``packages/data`` held
``### Changed`` four times, ``### Fixed`` three and ``### Added`` and
``### Documentation`` twice each, spread over 1,500 lines: a reader asking
"what changed?" has to find all four and merge them by hand, and any tool
reading a changelog section by section --- which is what the format is for ---
gets four answers to a question with one.

**The guard is scoped to ``## Unreleased`` and stops at the first released
section.** 32 duplicate headings survive in the released sections of six
packages, and they stay: those blocks are the published record, restructuring
them moves shipped text for a formatting gain, and a guard that failed on them
would have to carry an exemption list of version numbers --- which grows every
time someone notices another one, and which nobody can read. Scoping to the
section still being written is also where the guard actually prevents
something: the next round prepends into ``Unreleased``, never into ``v0.11.0``.

There is no allowlist for the same reason there is no exemption list. A
changelog with no ``## Unreleased`` section simply has nothing to check, which
is the honest answer rather than a skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._workspace import ROOT, rel

#: Where the section ends: the first ``## `` heading after ``## Unreleased``.
UNRELEASED = "Unreleased"


def unreleased_headings(text: str) -> list[str]:
    """The ``### `` headings under ``## Unreleased``, in the order they appear.

    Fence-aware, because a changelog entry may quote a markdown document ---
    a ``###`` inside a fenced block is sample text, not a section of this
    file.
    """
    headings: list[str] = []
    fenced = False
    inside = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if line.startswith("## "):
            if inside:
                break
            inside = UNRELEASED in line
            continue
        if inside and line.startswith("### "):
            headings.append(line.strip())
    return headings


def repeated(headings: list[str]) -> dict[str, int]:
    """Each heading that appears more than once, and how many times."""
    return {h: headings.count(h) for h in dict.fromkeys(headings) if headings.count(h) > 1}


def package_changelogs() -> list[Path]:
    paths = sorted(ROOT.glob("packages/*/CHANGELOG.md"))
    assert paths, "no package changelogs found — has the enumeration broken?"
    return paths


@pytest.mark.parametrize("path", package_changelogs(), ids=lambda p: p.parent.name)
def test_no_package_changelog_repeats_a_heading_under_unreleased(path: Path) -> None:
    """One ``### Fixed`` per release, holding everything fixed in it."""
    duplicates = repeated(unreleased_headings(path.read_text()))

    assert not duplicates, (
        f"{rel(path)} repeats "
        + ", ".join(f"'{h[4:]}' {n} times" for h, n in duplicates.items())
        + " under ## Unreleased. Add the entry to the section that is already "
        "there rather than opening a second one above it."
    )


def test_the_reader_finds_what_it_is_meant_to_and_nothing_else() -> None:
    """A positive control: a census that can find nothing always passes.

    Three properties in one sample, because each of them silently disables
    the guard on its own --- a reader that never enters the section, one that
    never leaves it, and one that counts sample text as structure.
    """
    sample = "\n".join(
        [
            "# Changelog",
            "",
            "## Unreleased",
            "",
            "### Fixed",
            "",
            "- something",
            "",
            "````",
            "### Fixed",  # inside a fence: sample text, not a section
            "````",
            "",
            "### Added",
            "",
            "- something else",
            "",
            "### Fixed",  # the defect
            "",
            "- a second block that belongs in the first",
            "",
            "## v1.0.0 - 2026-01-01",
            "",
            "### Fixed",  # released: out of scope, must not be counted
            "",
            "### Fixed",
        ]
    )

    assert unreleased_headings(sample) == ["### Fixed", "### Added", "### Fixed"]
    assert repeated(unreleased_headings(sample)) == {"### Fixed": 2}

    # ...and the clean shape it must not report.
    clean = sample.replace("### Fixed\n\n- a second block that belongs in the first", "")
    assert repeated(unreleased_headings(clean)) == {}


def test_a_changelog_with_no_unreleased_section_has_nothing_to_check() -> None:
    """Not a skip: the question is answered, and the answer is 'none'."""
    assert unreleased_headings("# Changelog\n\n## v1.0.0\n\n### Fixed\n\n### Fixed\n") == []

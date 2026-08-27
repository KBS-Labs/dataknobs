"""Reproduce-first guard: the documented tool catalog must be the catalog.

``packages/bots/docs/config-toolkit.md`` carries a table of every tool
``default_catalog`` is pre-populated with, and states the count in prose above
it; the site page states the same count with no table. The table stopped at
twelve rows and both counts said twelve while the catalog held twenty-one --
the nine wizard tools were registered by ``_register_builtin_tools`` and named
in neither document.

Nothing could fail on that, and the three checks that came closest each miss it
for a different reason. ``mkdocs build --strict`` validates links and nav. The
doc-mirror manifest records this pair as a deliberate divergence and
content-checks only the one section they share. ``test_documented_imports``
reads the names a document *names*, so a name omitted is a name it cannot see.
A table that is a correct subset passes all three.

A subset is the worst shape this particular error can take. An obviously
incomplete list invites the reader to go and look; twelve rows under the words
"all 12 built-in tools" reads as a closed set, and the reader concludes the
other nine do not exist. The count is what makes it a closed set, so the count
is checked here rather than deleted -- a number that is verified is worth more
than no number, and deleting it would leave the table's completeness resting on
nothing again.

Scope is stated rather than implied, because a guard that quietly covers less
than it appears to is worse than none -- it also reports green:

COVERED
    **The stated count, wherever a document states one.** Every markdown file a
    reader can reach is swept for the phrase, not just the two known sites, so
    a third copy of the number cannot appear unchecked. The sweep also fails if
    it matches nothing, since a check that silently stops finding its subject
    is a check that has stopped running.

    **Table membership, both directions.** A registered tool with no row and a
    row naming no registered tool are the same defect seen from two ends, and
    the second is the one that outlives a rename.

    **Per-row class name, tag set, and declared dependencies.** Tags and
    ``requires`` are frozensets on ``ToolEntry``, so the row is compared as a
    set and the order it is written in carries no claim.

NOT COVERED
    **The description and default params.** The table shows neither, and
    inventing columns to check would be a documentation decision made by a
    test.

    **The site page's own enumeration**, because it has none: it is a
    condensation that states the count and points at the table. That pointer is
    the only thing keeping the two in agreement, and it is a link, which
    ``mkdocs build --strict`` does check.

    **Tool classes that exist but are not registered.** ``ListCatalogTool``,
    ``SaveToCatalogTool`` and ``LoadFromCatalogTool`` are exported from
    ``dataknobs_bots.tools`` and documented in ``tools.md``, and their absence
    from the catalog is deliberate. This guard asks what the registry holds,
    never what the module exports -- the two are different questions and only
    the first one is what the table claims to answer.
"""

from __future__ import annotations

import re

import pytest

from dataknobs_bots.config import default_catalog
from tests._workspace import ROOT, documentation_files, rel

#: The table this guard checks, and the prose count that closes it.
CATALOG_DOC = ROOT / "packages" / "bots" / "docs" / "config-toolkit.md"

#: This file, for the one failure that asks the reader to come and edit it.
GUARD = ROOT / "tests" / "test_documented_tool_catalog.py"

#: Heading the table sits under, and the one that ends it.
TABLE_HEADING = "### Built-in Tools"

#: Any document claiming a number of built-in tools is claiming this one.
COUNT_PHRASE = re.compile(r"(\d+) built-in tools")

#: ``| `name` | `Class` | tag, tag | dep |`` -- an em dash meaning none.
ROW = re.compile(
    r"^\|\s*`(?P<name>[^`]+)`\s*"
    r"\|\s*`(?P<cls>[^`]+)`\s*"
    r"\|\s*(?P<tags>[^|]+?)\s*"
    r"\|\s*(?P<requires>[^|]+?)\s*\|\s*$"
)


def _cell(text: str) -> set[str]:
    """Read a comma-separated table cell as the set it denotes."""
    if text.strip() in {"—", "-", ""}:
        return set()
    return {part.strip().strip("`") for part in text.split(",") if part.strip()}


def _documented_rows() -> dict[str, tuple[str, set[str], set[str]]]:
    """Parse the built-in tools table into name -> (class, tags, requires)."""
    lines = CATALOG_DOC.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(TABLE_HEADING)
    except ValueError:  # pragma: no cover - the assertion below reports it
        pytest.fail(
            f"{rel(CATALOG_DOC)} no longer has a {TABLE_HEADING!r} heading. "
            f"If the table moved, point TABLE_HEADING in "
            f"{rel(GUARD)} at its new home; if it went away, this "
            f"guard has nothing left to check and should go with it."
        )
    rows: dict[str, tuple[str, set[str], set[str]]] = {}
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        match = ROW.match(line)
        if match is None:
            continue
        if match["name"].startswith("---") or match["name"] == "Name":
            continue
        rows[match["name"]] = (
            match["cls"],
            _cell(match["tags"]),
            _cell(match["requires"]),
        )
    return rows


def _registered() -> dict[str, tuple[str, set[str], set[str]]]:
    """The same shape, read from the registry the table describes."""
    return {
        entry.name: (
            entry.class_path.rsplit(".", 1)[-1],
            set(entry.tags),
            set(entry.requires),
        )
        for entry in default_catalog.list_tools()
    }


def test_every_stated_count_is_the_catalogs_size() -> None:
    """A number in prose is a claim about a registry, so ask the registry."""
    expected = len(default_catalog.list_tools())
    wrong: list[str] = []
    sites = 0
    for document in documentation_files():
        for stated in COUNT_PHRASE.findall(document.read_text(encoding="utf-8")):
            sites += 1
            if int(stated) != expected:
                wrong.append(f"  - {rel(document)}: says {stated}, catalog holds {expected}")

    assert sites, (
        "No document states a built-in tool count any more. Either the phrase "
        f"was reworded past {COUNT_PHRASE.pattern!r} -- in which case widen it, "
        "because the count is back to being unchecked -- or the counts were "
        "deleted, in which case delete this check with them."
    )
    assert not wrong, (
        "Documented built-in tool counts disagree with `default_catalog`:\n"
        + "\n".join(wrong)
        + "\n\nA tool was registered or removed and the prose did not follow."
    )


def test_the_documented_table_lists_every_registered_tool() -> None:
    """Membership, both directions: a missing row and a stale row."""
    documented = _documented_rows()
    registered = _registered()

    missing = sorted(set(registered) - set(documented))
    stale = sorted(set(documented) - set(registered))

    assert not missing, (
        f"Registered in `default_catalog` and absent from {rel(CATALOG_DOC)}:\n"
        + "\n".join(f"  - {name}" for name in missing)
        + "\n\nAdd a row. This is the shape the table drifted into last time: a "
        "correct subset, which reads as complete and fails nothing."
    )
    assert not stale, (
        f"Given a row in {rel(CATALOG_DOC)} and registered nowhere:\n"
        + "\n".join(f"  - {name}" for name in stale)
        + "\n\nEither the tool was unregistered and the row outlived it, or the "
        "row's name is misspelled."
    )


def test_each_documented_row_matches_its_registration() -> None:
    """The row's own three claims: which class, which tags, which dependencies."""
    documented = _documented_rows()
    registered = _registered()

    wrong: list[str] = []
    for name in sorted(set(documented) & set(registered)):
        doc_class, doc_tags, doc_requires = documented[name]
        reg_class, reg_tags, reg_requires = registered[name]
        if doc_class != reg_class:
            wrong.append(f"  - {name}: class {doc_class} documented, {reg_class} registered")
        if doc_tags != reg_tags:
            wrong.append(
                f"  - {name}: tags {sorted(doc_tags)} documented, {sorted(reg_tags)} registered"
            )
        if doc_requires != reg_requires:
            wrong.append(
                f"  - {name}: requires {sorted(doc_requires)} documented, "
                f"{sorted(reg_requires)} registered"
            )

    assert not wrong, (
        f"Rows in {rel(CATALOG_DOC)} that misdescribe their registration:\n"
        + "\n".join(wrong)
        + "\n\nTags and requires are compared as sets, so the order a cell is "
        "written in is not what failed."
    )

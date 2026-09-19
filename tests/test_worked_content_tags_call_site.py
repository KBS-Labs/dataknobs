"""The published content-tag call site runs, from outside the package that implements it.

A row out of somebody else's store should be readable into *which nodes is this
about* with no database, no embedder and no event loop, and the vocabulary
should then be the only thing that judges the ids. That claim is easy to make
in a design note and cheap to check exactly once: **a call site fails.** It
cannot be written around a member that does not exist, or around a name a door
does not export.

So this runs the call site the guide publishes, as written, against the
vocabulary the guide publishes beside it -- and it runs it from the workspace
root rather than from ``packages/common/tests``. That is not a filing
preference. A test living inside the package can reach anything the package
defines whether or not the door exports it, which makes it structurally unable
to fail for the one reason this test exists: a name that is implemented and not
published. ``tests/test_package_interop.py`` states the same reason for the
same location.

**One text, three consumers.** The guide carries two fenced blocks, marked
``worked-input`` and ``worked-call-site``. The input is written to disk from
the first; the code is executed from :mod:`worked_content_tags_call_site`,
which is asserted character-for-character against the second. Nothing here
paraphrases the page, so a page that goes wrong takes the suite with it.

**What this page's call site is for.** The subject is the *hop*: a hit set from
a retrieval this package did not perform, read into tags, filtered by a
vocabulary the caller names, and spent on a cursor. Four of its rows are
deliberately unlike each other -- one about two nodes, one that touched no
vocabulary, one naming an axis this file does not declare, and one a writer got
wrong -- because the read is only worth publishing if it says something useful
about each.

The assertions come after the block rather than inside it, because the block's
lines are mostly bare expressions whose values are discarded -- the form that
shows a reader what each call answers. Re-calling them to assert is only sound
because everything here is pure: the read holds no vocabulary, and every
accessor it hands off to is pure over the vocabulary's own fields.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._workspace import ROOT, door_imports, executed_source, published_fence

if TYPE_CHECKING:
    from collections.abc import Iterator

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "content-tags.md"
EXECUTED = ROOT / "tests" / "worked_content_tags_call_site.py"

INPUT_MARKER = "worked-input"
CALL_SITE_MARKER = "worked-call-site"


@pytest.fixture
def vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``mammals.yaml`` on disk, written from the guide, and made the cwd.

    The call site names the file by a bare relative path, which is what a reader
    would type. Honouring that means giving it a directory rather than rewriting
    the line to suit the harness.
    """
    (tmp_path / "mammals.yaml").write_text(
        published_fence(GUIDE, INPUT_MARKER) + "\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def ran(vocabulary: Path) -> dict[str, Any]:
    """What the call site leaves bound, having run it exactly as published."""
    return runpy.run_path(str(EXECUTED), run_name="__worked_call_site__")


def test_the_executed_copy_is_the_published_one() -> None:
    """Character for character, in both directions.

    This is what makes *verbatim* a property of the tree rather than an
    intention. Without it the guide could drift from the code, or the code from
    the guide, and every other assertion in this file would keep passing.
    """
    published = published_fence(GUIDE, CALL_SITE_MARKER)
    executed = executed_source(EXECUTED)

    assert executed == published, (
        "the executed call site and the one the guide publishes have diverged. "
        "They are one text with two homes: edit the fence in "
        f"{GUIDE.relative_to(ROOT)} and copy it to {EXECUTED.relative_to(ROOT)}, "
        "or the reverse -- but never one alone."
    )


def test_the_call_site_imports_only_through_the_doors() -> None:
    """Every import in the block is a package door, not a module path.

    The point of the increment this test arrives with is that these names are
    *exported*. A call site reaching into ``dataknobs_common.ontology.tags``
    would run identically and would prove nothing about that, so the spelling is
    the subject and not an incidental.

    ``dataknobs_common.hierarchy`` is not among them, where the anchored view
    admits it: nothing on this page reaches a bare structure. ``exceptions`` is,
    because the refusal a half-written row earns is half of what this page
    publishes --- it is caught in the fence rather than demonstrated outside it,
    so that the page runs to the end --- and every guide in this package imports
    an error from that module. The names *this* page introduces all arrive
    through ``dataknobs_common.ontology``, which is what the guard is about.
    """
    doors = {
        "dataknobs_common",
        "dataknobs_common.exceptions",
        "dataknobs_common.ontology",
    }
    reached = door_imports(published_fence(GUIDE, CALL_SITE_MARKER))

    assert reached, "the call site imports nothing from this package"
    assert reached <= doors, (
        f"{sorted(reached - doors)} is a module path rather than a door. "
        f"The call site is the increment's own acceptance that the names are "
        f"published, so reaching past a door defeats it"
    )


def test_the_row_is_read_with_no_store_and_no_loop(ran: dict[str, Any]) -> None:
    """The claim the page opens with, asserted rather than repeated.

    ``ran`` having been produced at all is most of the proof -- the module ran
    to completion outside any event loop, so nothing in it awaited. What is
    left is that the first read answered before any vocabulary was loaded,
    which is what *the read holds no vocabulary* means operationally.
    """
    tags = ran["tags"]

    assert [(t.ontology_id, t.taxonomy_id, t.node_id) for t in tags] == [
        ("mammals", "species", "golden_retriever"),
        ("mammals", "species", "beagle"),
    ]
    assert ran["onto"].entities.describe().backend == "authored"
    assert ran["onto"].entities.describe().table is None


def test_the_batch_reads_every_position_and_names_the_one_it_refused(
    ran: dict[str, Any],
) -> None:
    """Step 2: one entry per position, and the second field is what reads them.

    Two of the four entries are ``()`` and they mean opposite things -- one row
    touched no vocabulary, one was written wrong. ``malformed`` is what tells
    them apart, which is the whole reason the reading has two fields rather
    than one.
    """
    reading = ran["reading"]

    assert len(reading.tags) == 4
    assert [len(row) for row in reading.tags] == [2, 0, 1, 0]
    assert [bad.row for bad in reading.malformed] == [3]
    assert "dk_taxonomy_id" in reading.malformed[0].reason


def test_the_caller_filters_and_the_drifted_axis_is_dropped(ran: dict[str, Any]) -> None:
    """Step 3: a name off a row is not a name the caller typed.

    The third row names ``coat``, an axis this vocabulary does not declare. The
    page drops it with a membership test rather than catching a refusal,
    because a drifted corpus is not a mistake anybody can be told about --- and
    the accessor still refuses, which the package suite asserts beside this.
    """
    onto, mine = ran["onto"], ran["mine"]

    assert [t.node_id for t in mine] == ["golden_retriever", "beagle"]
    assert "coat" not in onto.taxonomies
    assert all(t.taxonomy_id in onto.taxonomies for t in mine)


def test_the_tag_reaches_a_cursor_through_the_vocabulary(ran: dict[str, Any]) -> None:
    """Step 4: the hop itself, which is what the whole page is for.

    The id travels as a pair and arrives as a key. ``qualified_id`` composes
    the one string the vocabulary's door takes, ``localize`` judges it and
    hands back what ``at()`` wants, and from there this is the cursor the
    anchored-view guide already publishes.
    """
    here, placed = ran["here"], ran["placed"]

    assert ran["mine"][0].qualified_id == "mammals:golden_retriever"
    assert here.node == "golden_retriever"
    assert here.exists() is True
    assert placed.name == "Golden Retriever"
    assert [above.node for above in here.ancestors()] == ["retriever", "dog", "mammal"]

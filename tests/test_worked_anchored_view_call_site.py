"""The published cursor call site runs, from outside the package that implements it.

A placement should be widenable into the context around it -- what is above it,
what those things are, what they say, what is still unsaid below -- with no
database, no embedder and no event loop. That claim is easy to make in a design
document and cheap to check exactly once: **a call site fails.** It cannot be
written around a member that does not exist, or around a name a door does not
export.

So this runs the call site the guide publishes, as written, against the
vocabulary the guide publishes beside it -- and it runs it from the workspace
root rather than from ``packages/common/tests``. That is not a filing
preference. A test living inside the package can reach anything the package
defines whether or not the door exports it, which makes it structurally unable
to fail for the one reason this test exists: a name that is implemented and not
published. ``tests/test_package_interop.py`` states the same reason for the same
location.

**One text, three consumers.** The guide carries two fenced blocks, marked
``worked-input`` and ``worked-call-site``. The input is written to disk from the
first; the code is executed from :mod:`worked_anchored_view_call_site`, which is
asserted character-for-character against the second. Nothing here paraphrases
the page, so a page that goes wrong takes the suite with it.

**What this page's call site is for, and what it deliberately is not.** The
subject here is the *cursor*: one axis, one node, and the questions asked from
there. Two lines therefore read differently than they do in the design note this
was drawn from. The resolver is **bound** rather than assumed, because a reader
running the page needs it to exist; and what a type inherits is asked of the
**axis**, taking an entity type, rather than of the cursor -- the cursor's node
is an *entity* and the inheritance question is about the *type* lattice, which
is a different set of edges over the same relation id. A member on the cursor
would have had to read ``entity().type``, and ``entity()`` answers ``None`` for
exactly the node the emptiness question exists to keep distinguishable.

The assertions come after the block rather than inside it, because the block's
lines are mostly bare expressions whose values are discarded -- the form that
shows a reader what each call answers. Re-calling them to assert is only sound
because every accessor here is pure over the vocabulary's own fields: nothing
opens anything and nothing is lazy.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._workspace import ROOT, executed_source, published_fence

if TYPE_CHECKING:
    from collections.abc import Iterator

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "anchored-view.md"
EXECUTED = ROOT / "tests" / "worked_anchored_view_call_site.py"

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
    *exported*. A call site reaching into ``dataknobs_common.ontology.taxonomy``
    would run identically and would prove nothing about that, so the spelling is
    the subject and not an incidental.
    """
    doors = {"dataknobs_common", "dataknobs_common.hierarchy", "dataknobs_common.ontology"}
    reached = {
        line.split()[1]
        for line in published_fence(GUIDE, CALL_SITE_MARKER).splitlines()
        if line.startswith("from dataknobs_common")
    }

    assert reached, "the call site imports nothing from this package"
    assert reached <= doors, (
        f"{sorted(reached - doors)} is a module path rather than a door. "
        f"The call site is the increment's own acceptance that the names are "
        f"published, so reaching past a door defeats it"
    )


def test_the_axis_is_reached_without_a_store_or_a_loop(ran: dict[str, Any]) -> None:
    """The claim the page opens with, asserted rather than repeated.

    ``ran`` having been produced at all is most of the proof -- the module ran to
    completion outside any event loop, so nothing in it awaited. What is left is
    that the axis came off the authored vocabulary rather than off a backend.
    """
    onto, axis = ran["onto"], ran["axis"]

    assert axis.definition.id == "species"
    assert onto.entities.describe().backend == "authored"
    assert onto.entities.describe().table is None


def test_a_placement_anchors_a_cursor(ran: dict[str, Any]) -> None:
    """Step 1 into step 2: a resolved id is what the cursor is opened at."""
    assert ran["hit"].entity_id == "golden_retriever"
    assert ran["here"].node == "golden_retriever"


def test_the_cursor_walks_up_and_answers_plurally(ran: dict[str, Any]) -> None:
    """Step 2: one step up is a tuple, and the whole way up is ordered."""
    here = ran["here"]

    assert [above.node for above in here.parents()] == ["retriever"]
    assert [above.node for above in here.ancestors()] == ["retriever", "dog", "mammal"]


def test_each_ancestor_carries_its_entity_and_its_assertions(ran: dict[str, Any]) -> None:
    """Step 2's loop: the context that folds into a prompt.

    ``find`` is asked of the **vocabulary**, not of the view, and the fence says
    so in a comment. A taxonomy holds a structure axis and an entity source and
    no assertion axis, so a cursor over one has nothing to answer this with --
    which is a fact about what a taxonomy is rather than a gap in the cursor.
    """
    onto, here = ran["onto"], ran["here"]
    above = list(here.ancestors())

    assert [node.entity().name for node in above] == ["Retriever", "Dog", "Mammal"]
    assert above[1].entity().description == "A domesticated carnivoran."
    assert [
        assertion.object.entity_id for assertion in onto.assertions.find(subject=above[1].node)
    ] == ["mammal"]


def test_inheritance_is_asked_of_the_axis_and_answers_about_the_type(
    ran: dict[str, Any],
) -> None:
    """Step 3: the *other* ``isa`` lattice, over the same relation id.

    ``golden_retriever`` is a ``Breed`` and ``Breed`` is declared ``isa:
    Species``, so the inherited set is the breed's own attribute plus the
    species' two. The entity walk in the test above and this one share a relation
    name and nothing else -- that collision is why this is asked of the axis with
    a type rather than of the cursor with a node.
    """
    axis, placed = ran["axis"], ran["placed"]

    assert placed.type == "Breed"
    assert sorted(a.name for a in axis.inherited_attributes(placed.type)) == [
        "akc_group",
        "latin_name",
        "lifespan_years",
    ]


def test_re_anchoring_shows_what_is_still_unspecified(ran: dict[str, Any]) -> None:
    """Step 4: the move, and the question a consumer asks after it.

    ``is_leaf()`` being ``False`` is the whole of use case 5: the vocabulary does
    not conclude that the placement is too coarse, it reports that the node has
    children and leaves the conclusion to the caller.
    """
    there = ran["there"]

    assert there.node == "dog"
    assert there.is_leaf() is False
    assert [below.node for below in there.children()] == ["retriever", "beagle"]


def test_the_reader_leaves_with_keys(ran: dict[str, Any]) -> None:
    """Step 5: ids in the caller's own space, from one owner of the walk."""
    axis, there = ran["axis"], ran["there"]

    assert axis.subtree_keys(there.node) == [
        "dog",
        "retriever",
        "golden_retriever",
        "beagle",
    ]

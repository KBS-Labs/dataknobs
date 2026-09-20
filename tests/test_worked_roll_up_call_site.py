"""The published roll-up call site runs, from outside the package that implements it.

A page of results out of somebody else's store should be readable into *what
these rows are about* with no database, no embedder and no event loop --- first
as a count over the tags, which needs no vocabulary at all, then as a roll-up
onto one axis of one vocabulary the caller loaded. That claim is easy to make in
a design note and cheap to check exactly once: **a call site fails.** It cannot
be written around a member that does not exist, or around a name a door does not
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
first; the code is executed from :mod:`worked_roll_up_call_site`, which is
asserted character-for-character against the second. Nothing here paraphrases
the page, so a page that goes wrong takes the suite with it.

**What this page's call site is for.** The subject is the *set*: five hits from
a retrieval this package did not perform, counted by vocabulary, rolled up onto
one axis, projected, and spent back on the cursor each entry came from. The
operation only exists at three rows or more --- over one it is an accessor ---
and two of the five are deliberately residue: one touched no vocabulary, one
names a node the vocabulary does not carry. A roll-up whose worked case contains
neither is a roll-up whose residue was never designed.

The assertions come after the block rather than inside it, because the block's
lines are mostly bare expressions whose values are discarded -- the form that
shows a reader what each call answers. Re-calling them to assert is only sound
because everything here is pure: the count holds no vocabulary, the roll-up is
pure over one it was handed, and every accessor it reaches is pure over that
vocabulary's own fields.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._workspace import ROOT, door_imports, executed_source, published_fence

if TYPE_CHECKING:
    from collections.abc import Iterator

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "roll-up.md"
EXECUTED = ROOT / "tests" / "worked_roll_up_call_site.py"

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
    *exported*. A call site reaching into ``dataknobs_common.ontology.ascent``
    would run identically and would prove nothing about that, so the spelling is
    the subject and not an incidental.

    **One door, which is the narrowest set of the seven guards in this shape**,
    and the narrowness is earned rather than incidental: this page demonstrates
    no refusal, so nothing on it catches one, so ``dataknobs_common.exceptions``
    is never reached --- where the content-tags page admits it precisely because
    the refusals it catches are half of what it publishes. The two refusals this
    operation documents are asserted in the package suite instead, which is where
    an exception belongs when the page's subject is an answer.

    The subset test means nothing already caught stops being caught.
    """
    doors = {"dataknobs_common", "dataknobs_common.ontology"}
    reached = door_imports(published_fence(GUIDE, CALL_SITE_MARKER))

    assert reached, "the call site imports nothing from this package"
    assert reached <= doors, (
        f"{sorted(reached - doors)} is a module path rather than a door. "
        f"The call site is the increment's own acceptance that the names are "
        f"published, so reaching past a door defeats it"
    )


def test_the_rows_are_counted_with_no_vocabulary_and_no_loop(ran: dict[str, Any]) -> None:
    """The claim the page opens with, asserted rather than repeated.

    ``ran`` having been produced at all is most of the proof -- the module ran to
    completion outside any event loop, so nothing in it awaited. What is left is
    that the count over vocabularies answered *before* any vocabulary was
    loaded, which is what *it holds no vocabulary* means operationally, and that
    the one it then loaded needed no store.
    """
    reading = ran["reading"]

    assert [len(row) for row in reading.tags] == [2, 0, 1, 1, 1]
    assert reading.malformed == ()
    assert ran["onto"].entities.describe().backend == "authored"
    assert ran["onto"].entities.describe().table is None


def test_the_rows_are_rolled_up_onto_what_the_vocabulary_carries(ran: dict[str, Any]) -> None:
    """The record the page publishes: three nodes, their rows, and what stands over them.

    First-seen order rather than ranked, because ``supported`` is the record and
    ``prune()`` is the presentation. ``above`` is restricted to the nodes of this
    same set --- ``golden_retriever``'s ancestors include ``retriever`` and
    ``mammal``, and neither appears, because no row named them.
    """
    answer = ran["answer"]

    assert [(s.node_id, s.rows, s.above) for s in answer.supported] == [
        ("golden_retriever", (0, 3), ("dog",)),
        ("beagle", (0,), ("dog",)),
        ("dog", (2,), ()),
    ]
    assert answer.ontology_id == "mammals"
    assert answer.taxonomy_id == "species"


def test_the_residue_is_reported_and_nothing_raised(ran: dict[str, Any]) -> None:
    """The two rows the page chose for being unlike the others.

    ``wolfhound`` is a node this vocabulary does not carry, and it arrives in
    ``unplaced`` carrying the row that named it rather than raising --- the page
    running to the end at all is half of that assertion. Row 1 touched no
    vocabulary. Both are in ``unsupported_rows``, which is what a consumer reads
    to see that the answer was drawn from three of five rows.
    """
    answer = ran["answer"]

    assert [(s.node_id, s.rows) for s in answer.unplaced] == [("wolfhound", (4,))]
    assert "wolfhound" not in {s.node_id for s in answer.supported}
    assert answer.unsupported_rows == (1, 4)


def test_the_projection_presents_and_does_not_discard(ran: dict[str, Any]) -> None:
    """The three policies the page prints, and the claim underneath them.

    ``dog`` is absent from ``MOST_SPECIFIC`` and is the whole of
    ``MOST_GENERAL``, and in both cases it is still in ``supported`` --- which is
    the page's sentence *pruning changes what is presented, never what is
    reachable*, asserted rather than repeated. The ranking is by row count
    descending, which is why ``golden_retriever`` leads with two rows.
    """
    answer = ran["answer"]
    granularity = ran["Granularity"]

    assert [s.node_id for s in answer.prune()] == ["golden_retriever", "beagle"]
    assert [s.node_id for s in answer.prune(granularity.MOST_GENERAL)] == ["dog"]
    assert [s.node_id for s in answer.prune(granularity.ALL)] == [
        "golden_retriever",
        "beagle",
        "dog",
    ]
    assert "dog" in {s.node_id for s in answer.supported}
    assert [len(s.rows) for s in answer.prune(granularity.ALL)] == [2, 1, 1]


def test_the_vocabularies_in_play_are_counted_before_one_is_loaded(ran: dict[str, Any]) -> None:
    """The step the page opens with, and the reason it comes first.

    A caller holding a page of results does not yet know which vocabulary to
    load. ``ontology_support`` answers that from the tags alone --- row 1 named
    none, so the four that did are the entry's rows, including row 4 whose node
    the vocabulary turns out not to carry. *Which vocabulary* and *does it carry
    this node* are two questions, and this is the first one.
    """
    support = ran["ontology_support"](ran["reading"].tags)

    assert [(s.ontology_id, s.rows) for s in support] == [("mammals", (0, 2, 3, 4))]
    assert support[0].ontology_id == ran["onto"].id


def test_the_tie_back_in_reaches_a_cursor_and_its_evidence(ran: dict[str, Any]) -> None:
    """The last block, which is what the whole page is for.

    Each kept entry is spent twice: on the vocabulary, which says what the node
    *is* and what stands above it, and on the caller's own sequence, which is
    where the evidence was all along. ``localize`` is not called --- the roll-up
    already answered in the axis's key space, which is the difference between
    this page's loop and the content-tags one.

    The loop leaves the last iteration bound, so what is asserted is ``beagle``:
    the second of the two entries ``prune()`` keeps.
    """
    assert ran["named"] == "Beagle"
    assert ran["context"] == ["dog", "mammal"]
    assert ran["evidence"] == ["Spay recovery in retrievers is usually uneventful."]
    assert ran["here"].node == "beagle"
    assert ran["here"].exists() is True

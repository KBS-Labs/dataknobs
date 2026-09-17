"""A published input fence and the suite constant beside it are one document.

Three guides publish a ``worked-input`` fence: a vocabulary a reader can copy,
which a workspace test writes to disk and runs the guide's published call site
against. For two of them ``_vocabularies.py`` holds the *same* vocabulary as a
module constant, which this package's fixtures and most of its suites load
instead. The third publishes a document no constant holds, so it is one copy
rather than two -- see ``UNPAIRED`` below, which is where that is said and
where it is checked.

**Nothing compared the two.** Each copy was guarded by its own suite -- a fence
that drifts takes its workspace runner red, a constant that drifts takes
``test_resolution_spans.py`` red -- so both stayed correct while what was
unguarded was that they remain the *same* document. Measured when this was
written: they already differed, by the ``# mammals.yaml`` line the fence opens
with and the constant does not, with nothing in the tree holding even that much.

**Why that matters more than it sounds.** The offsets the resolution suites
assert -- ``(3, 19)`` for ``golden retriever``, ``((0, 2), (20, 36))`` for what
the vocabulary did not account for -- are offsets into a query resolved against
the forms *the vocabulary declares*. This suite asserts them against the
constant and the workspace runner asserts them against the fence, both saying
"the same document" while nothing held them to it. A form added to one copy
alone leaves two suites agreeing on numbers that no longer describe one thing.

**The authority runs one way per copy**, which is why this compares rather than
deriving. The fence is what a reader copies, so it is the authority for the
runner; the constant is what this suite loads, so it is the authority here.
Collapsing them into one text would make this suite depend on the workspace
guards' fence reader, which it cannot reach -- and one of these fences is pinned
*verbatim* by an acceptance criterion, so it is not editable to suit a harness.

**Here rather than under** ``tests/``, which is where it was written first and
is the wrong side of a boundary the toolchain guards enforce. A workspace guard
reading ``packages/common/tests/conftest.py`` puts a *package's* file in a
workspace guard's input set, and that file belongs to no workspace hash scope --
so editing it would move this verdict while every stored hash stayed intact.
Filing it in the workspace-only tier used to be worse still: change detection
tested that tier before the package mapping and stopped, so the common conftest
would have stopped scheduling the common suite. That half has expired --
``map_files_to_packages`` accumulates now, and a file declared in two tiers
contributes to both -- so the placement rests on the hash-scope half above
rather than on both. Read from this side, the input is a package document, which
``PACKAGE_TEST_DOC_INPUTS`` already exists to declare and ``packages/*/docs/``
already hashes. ``test_packs.py`` reads its own guide the same way.

**Adding a pair is one row in the table below and one in that declaration**,
and forgetting the first is caught rather than trusted:
``test_every_published_vocabulary_is_in_the_table`` reads the guides for the
marker and fails on one neither the table nor ``UNPAIRED`` carries. A table
nobody checks against the tree is a list of the pairs somebody remembered.

**Publishing a fence is not the same as adding a pair**, which is the
distinction that assertion learned the hard way: it read the marker as a proxy
for "half of a pair" and fired on ``anchored-view.md``, which publishes a whole
vocabulary of its own and mirrors nothing. A single copy has no sameness to
guard -- it is guarded by being *executed* -- so it is declared rather than
tabled, and the declaration is checked against the tree too. What stays total
is that every published fence is accounted for as one kind or the other, so a
fourth still fails here until somebody decides which it is.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import _vocabularies
from _vocabularies import MAMMALS_DOCUMENT, MAMMALS_V11_DOCUMENT

#: Every guide that could carry a published vocabulary. Globbed rather than
#: listed, so a guide added tomorrow is inside what the table is checked
#: against rather than outside it.
#:
#: Reached through a declared guide's ``.parent`` rather than by dividing
#: ``__file__`` towards the directory, for the reason the comment above gives
#: one direction on: the declaration guard reconstructs a divided
#: ``Path(__file__)`` chain as the document it names, and a chain ending at
#: ``"guides"`` reconstructs as a directory it then fails to find.

#: Each guide is spelled as one whole ``Path(__file__)`` chain rather than a
#: shared ``GUIDES`` directory divided twice. ``PACKAGE_TEST_DOC_INPUTS`` is
#: checked against the tree by reconstructing exactly that shape, and a
#: two-step spelling reconstructs as the *directory* -- which is not a document,
#: so the declaration guard fails naming a path that cannot exist.
ONTOLOGY_GUIDE = pathlib.Path(__file__).parents[1] / "docs" / "guides" / "ontology.md"
ENTITY_RESOLUTION_GUIDE = (
    pathlib.Path(__file__).parents[1] / "docs" / "guides" / "entity-resolution.md"
)
ANCHORED_VIEW_GUIDE = pathlib.Path(__file__).parents[1] / "docs" / "guides" / "anchored-view.md"

GUIDES = sorted(ONTOLOGY_GUIDE.parent.glob("*.md"))

#: The header line a fence carries and a constant does not: a published
#: vocabulary names the file a reader should save it as, and the constant is
#: written to a path the fixture chooses. The one declared difference --
#: anything else is drift.
FILENAME_COMMENT = "# mammals.yaml"

_MARKER = "<!-- worked-input -->"
#: Anchored at the start of what follows the marker, so the fence this reads is
#: the one the marker *introduces*. An unanchored search would bind the marker
#: to the first ``yaml`` fence anywhere below it, which is a different document
#: whenever a page grows a paragraph between the two.
_FENCE = re.compile(r"\A\s*```yaml\n(?P<body>.*?)^```$", re.M | re.S)


def _carries_marker(guide: pathlib.Path) -> bool:
    """Whether ``guide`` publishes a vocabulary at all."""
    return guide.is_file() and _MARKER in guide.read_text(encoding="utf-8")


def _published_vocabulary(guide: pathlib.Path) -> str:
    """The body of ``guide``'s ``worked-input`` fence.

    **Refuses rather than returning empty**, which is the whole reason this is
    a function rather than a slice. A marker that is renamed, a fence that
    stops being ``yaml``, a guide that is moved -- each would otherwise hand
    the comparison below an empty string, and two empty strings compare equal.
    A guard reporting green because it read nothing is the failure mode of
    every reader like this one.

    **It refuses the same three things** ``tests/_workspace.published_fence``
    refuses, deliberately. That reader cannot be imported from here -- a
    package suite reaching into the workspace guards is the boundary the module
    docstring explains -- so this is a second reader by necessity, and a second
    reader whose refusals are *weaker* is the failure the first one's docstring
    warns about: a guard that stopped refusing still passes every test written
    for the guard that did. The one difference that remains is deliberate and
    narrow: this reader requires ``yaml``, because what it reads is a
    vocabulary rather than any published block.
    """
    assert guide.is_file(), (
        f"{guide} is missing. It carries the published copy of a vocabulary "
        f"this suite also holds as a constant, so its absence disables a drift "
        f"guard rather than making one inapplicable."
    )
    text = guide.read_text(encoding="utf-8")
    assert text.count(_MARKER) == 1, (
        f"{guide.name} carries {text.count(_MARKER)} {_MARKER} comments, "
        f"expected exactly one. Reading the first of several would compare a "
        f"document nobody chose"
    )

    fence = _FENCE.search(text.partition(_MARKER)[2])
    assert fence is not None, (
        f"no ```yaml fence immediately follows {_MARKER} in {guide.name}. The "
        f"marker introduces the fence below it; anything between them means "
        f"this would read a block the marker does not name"
    )
    body = fence.group("body")
    assert body.strip(), f"the {_MARKER} fence in {guide.name} is empty"
    return body


#: The pairs, named once: parametrized below and checked against the tree by
#: ``test_every_published_vocabulary_is_in_the_table``. Two readings of one
#: list, rather than a list and a copy of it.
PAIRS = [
    (ONTOLOGY_GUIDE, MAMMALS_DOCUMENT, "MAMMALS_DOCUMENT"),
    (ENTITY_RESOLUTION_GUIDE, MAMMALS_V11_DOCUMENT, "MAMMALS_V11_DOCUMENT"),
]

#: Guides that publish a vocabulary **no constant mirrors**, and why. Named so
#: the completeness assertion below stays total: a published fence is accounted
#: for as half of a pair or as a single copy with a reason, and a fence that is
#: neither fails.
#:
#: The distinction is not bookkeeping. This file exists because two copies of
#: one document were each guarded and their *sameness* was guarded by nobody.
#: A document with one copy has no sameness to guard, and declaring it here says
#: that in the place somebody adding the fourth fence will read -- where
#: omitting it from ``PAIRS`` alone would read as an oversight.
#:
#: Checked rather than believed: ``test_a_declared_single_copy_is_still_single``
#: fails if a constant ever comes to hold one of these, which is the moment the
#: reason stops being true and a row becomes owed.
UNPAIRED: dict[pathlib.Path, str] = {
    ANCHORED_VIEW_GUIDE: (
        "its only consumer is tests/worked_anchored_view_call_site.py, which "
        "writes the fence to disk and runs the guide's published call site "
        "against it -- so the fence is guarded by execution rather than by "
        "comparison. The vocabulary is its own: nearest is MAMMALS_V11_DOCUMENT, "
        "which differs substantively (latin_name is not required there, and it "
        "carries the literal-object assertion this one does not), so it is a "
        "third document rather than a drifted copy of either. It is declared in "
        "PACKAGE_TEST_DOC_INPUTS all the same, because the test below reads it "
        "to check this very claim -- a suite that reads a document has to be "
        "scheduled by an edit to it."
    ),
}


@pytest.mark.parametrize(
    ("guide", "declared", "constant"), PAIRS, ids=["ontology", "entity-resolution"]
)
def test_the_published_input_is_this_suites_own_vocabulary(
    guide: pathlib.Path, declared: str, constant: str
) -> None:
    """Character for character, less the one declared difference."""
    header, _, body = _published_vocabulary(guide).partition("\n")

    assert header == FILENAME_COMMENT, (
        f"the {_MARKER} fence in {guide.name} opens with {header!r}. A published "
        f"vocabulary names the file a reader saves it as, and that line is the "
        f"only thing this fence may carry that {constant} does not"
    )
    assert body == declared, (
        f"the vocabulary {guide.name} publishes and {constant} in "
        f"_vocabularies.py "
        f"have diverged. They are one document with two homes: the fence is "
        f"what a reader copies and the constant is what this suite loads, so "
        f"both are authorities and neither may be edited alone."
    )


def test_every_published_vocabulary_is_in_the_table() -> None:
    """The table is checked against the guides, not trusted to be complete.

    Each declared pair is compared character for character, so an *edit* to
    either copy is caught. What a hand-written table cannot catch is an
    **addition**: a third guide publishing a third vocabulary, mirrored by a
    third constant, guarded by nothing until someone remembers the row. That is
    the same shape as the defect this file was written for -- two copies, each
    guarded, and their sameness guarded by nobody.

    Read the way ``_declared_import_roots`` reads its declarations: from the
    tree, so the answer cannot be stale.
    """
    publishing = {guide.name for guide in GUIDES if _carries_marker(guide)}
    tabled = {guide.name for guide, _, _ in PAIRS}
    single = {guide.name for guide in UNPAIRED}

    assert not (tabled & single), (
        f"{sorted(tabled & single)} is both a declared pair and a declared "
        f"single copy. Those are contradictory claims about the same document, "
        f"and whichever one this file happened to read first would decide "
        f"which guard ran."
    )
    assert publishing == tabled | single, (
        f"guides publishing a {_MARKER} fence: {sorted(publishing)}; guides in "
        f"the table: {sorted(tabled)}; declared single copies: {sorted(single)}. "
        f"A published vocabulary in neither is compared to nothing and declared "
        f"as nothing, and an entry for a guide that no longer publishes one is a "
        f"guard that has quietly stopped reading. If the new fence mirrors a "
        f"constant it is a row in PAIRS; if it is the only copy of its document "
        f"it is an entry in UNPAIRED, with the reason."
    )


def test_a_declared_single_copy_is_still_single() -> None:
    """``UNPAIRED`` is checked against the tree, not taken at its word.

    Each entry claims that no constant in ``_vocabularies`` holds that guide's
    vocabulary, which is the whole reason it is exempt from the comparison
    above. The day a constant does hold it, the claim is false and a row in
    ``PAIRS`` is owed -- and nothing else in this file would notice, because
    every guard here is keyed off the table the entry is keeping it out of.

    Both the fence body and the body less its opening filename line are
    compared, since that line is the one declared difference between a fence
    and its constant. Matching either is a pair.
    """
    constants = {
        name: value
        for name, value in vars(_vocabularies).items()
        if name.isupper() and isinstance(value, str)
    }
    assert constants, (
        "no string constants found in _vocabularies -- this guard would pass "
        "by reading nothing, which is the failure mode every reader in this "
        "file refuses"
    )

    for guide, reason in UNPAIRED.items():
        body = _published_vocabulary(guide)
        held_by = sorted(
            constant
            for constant, value in constants.items()
            if value in (body, body.partition("\n")[2])
        )
        assert not held_by, (
            f"{guide.name} is declared as the only copy of its vocabulary, but "
            f"{held_by} in _vocabularies.py now holds it. The declared reason "
            f"({reason}) has stopped being true: move it from UNPAIRED to PAIRS "
            f"so the two copies are held to each other."
        )

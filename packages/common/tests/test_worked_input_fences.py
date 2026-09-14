"""A published input fence and the suite constant beside it are one document.

Two guides publish a ``worked-input`` fence: a vocabulary a reader can copy,
which a workspace test writes to disk and runs the guide's published call site
against. ``conftest.py`` holds the *same* vocabulary as a module constant,
which most of this suite loads from instead.

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
Filing it in the workspace-only tier would be worse: change detection tests that
tier before the package mapping and stops, so the common conftest would stop
scheduling the common suite. Read from this side instead, the input is a package
document, which ``PACKAGE_TEST_DOC_INPUTS`` already exists to declare and
``packages/*/docs/`` already hashes. ``test_packs.py`` reads its own guide the
same way.

Adding a third pair is one row in the table below and one in that declaration.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from conftest import MAMMALS_DOCUMENT, MAMMALS_V11_DOCUMENT

#: Each guide is spelled as one whole ``Path(__file__)`` chain rather than a
#: shared ``GUIDES`` directory divided twice. ``PACKAGE_TEST_DOC_INPUTS`` is
#: checked against the tree by reconstructing exactly that shape, and a
#: two-step spelling reconstructs as the *directory* -- which is not a document,
#: so the declaration guard fails naming a path that cannot exist.
ONTOLOGY_GUIDE = pathlib.Path(__file__).parents[1] / "docs" / "guides" / "ontology.md"
ENTITY_RESOLUTION_GUIDE = (
    pathlib.Path(__file__).parents[1] / "docs" / "guides" / "entity-resolution.md"
)

#: The header line a fence carries and a constant does not: a published
#: vocabulary names the file a reader should save it as, and the constant is
#: written to a path the fixture chooses. The one declared difference --
#: anything else is drift.
FILENAME_COMMENT = "# mammals.yaml"

_MARKER = "<!-- worked-input -->"
_FENCE = re.compile(r"^```yaml\n(?P<body>.*?)^```$", re.M | re.S)


def _published_vocabulary(guide: pathlib.Path) -> str:
    """The body of ``guide``'s ``worked-input`` fence.

    **Refuses rather than returning empty**, which is the whole reason this is
    a function rather than a slice. A marker that is renamed, a fence that
    stops being ``yaml``, a guide that is moved -- each would otherwise hand
    the comparison below an empty string, and two empty strings compare equal.
    A guard reporting green because it read nothing is the failure mode of
    every reader like this one.
    """
    assert guide.is_file(), (
        f"{guide} is missing. It carries the published copy of a vocabulary "
        f"this suite also holds as a constant, so its absence disables a drift "
        f"guard rather than making one inapplicable."
    )
    text = guide.read_text(encoding="utf-8")
    _, marker, after = text.partition(_MARKER)
    assert marker, f"{guide.name} carries no {_MARKER} comment"

    fence = _FENCE.search(after)
    assert fence is not None, f"no ```yaml fence follows {_MARKER} in {guide.name}"
    body = fence.group("body")
    assert body.strip(), f"the {_MARKER} fence in {guide.name} is empty"
    return body


@pytest.mark.parametrize(
    ("guide", "declared", "constant"),
    [
        (ONTOLOGY_GUIDE, MAMMALS_DOCUMENT, "MAMMALS_DOCUMENT"),
        (ENTITY_RESOLUTION_GUIDE, MAMMALS_V11_DOCUMENT, "MAMMALS_V11_DOCUMENT"),
    ],
    ids=["ontology", "entity-resolution"],
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
        f"the vocabulary {guide.name} publishes and {constant} in conftest.py "
        f"have diverged. They are one document with two homes: the fence is "
        f"what a reader copies and the constant is what this suite loads, so "
        f"both are authorities and neither may be edited alone."
    )

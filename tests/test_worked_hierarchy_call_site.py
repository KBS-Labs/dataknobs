"""The hierarchy guide's vocabulary loads, and its call site runs as published.

The fourth of these, and the one whose absence had already cost something.
``ontology.md``, ``anchored-view.md`` and ``entity-resolution.md`` each carry
worked markers and a guard that writes their input to disk and runs their
fence. ``hierarchy.md`` carries the same shape of content -- a full
``ontology:`` document and a call site asserting a walk over it -- and had no
guard, so nothing ran either half.

**That is how its vocabulary came to assert an attribute it never declared.**
The document's assertions carried ``relation: lifespan_years`` with no
``lifespan_years`` in any ``entity_types:`` row. Nothing reported it, because
nothing loaded the page; it surfaced only when a one-off sweep read every
ontology document in the workspace, and a sweep leaves no standing artifact.
The next such reference on this page is caught by this file or by nothing.

The claim about ``inherited_attributes`` is asserted here rather than in the
fence, following ``test_worked_ontology_call_site.py``'s split: the fence is
what the page shows a reader, and this is a property of the vocabulary that
the surrounding prose is about -- the type lattice is a different lattice
from the one ``structure`` walks -- without being a line the page publishes.
"""

from __future__ import annotations

import runpy
from typing import TYPE_CHECKING, Any

import pytest

from tests._workspace import ROOT, door_imports, executed_source, published_fence

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "hierarchy.md"
EXECUTED = ROOT / "tests" / "worked_hierarchy_call_site.py"

INPUT_MARKER = "worked-input"
CALL_SITE_MARKER = "worked-call-site"


@pytest.fixture
def vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``mammals.yaml`` on disk, written from the guide, and made the cwd.

    The call site names the file by a bare relative path, which is what a
    reader would type. Honouring that means giving it a directory rather than
    rewriting the line to suit the harness.
    """
    (tmp_path / "mammals.yaml").write_text(
        published_fence(GUIDE, INPUT_MARKER) + "\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def ran(vocabulary: Path) -> dict[str, Any]:
    """What the call site leaves bound, having run it exactly as published."""
    del vocabulary
    return runpy.run_path(str(EXECUTED), run_name="__worked_call_site__")


def test_the_executed_copy_is_the_published_one() -> None:
    """Character for character, in both directions."""
    published = published_fence(GUIDE, CALL_SITE_MARKER)
    executed = executed_source(EXECUTED)

    assert executed == published, (
        "the executed call site and the one the guide publishes have diverged. "
        "They are one text with two homes: edit the fence in "
        f"{GUIDE.relative_to(ROOT)} and copy it to {EXECUTED.relative_to(ROOT)}, "
        "or the reverse -- but never one alone."
    )


def test_the_call_site_imports_only_through_the_doors() -> None:
    """Every import in the block is a package door, not a module path."""
    doors = {"dataknobs_common", "dataknobs_common.ontology"}
    reached = door_imports(published_fence(GUIDE, CALL_SITE_MARKER))

    assert reached, "the call site imports nothing from this package"
    assert reached <= doors, f"{sorted(reached - doors)} is a module path rather than a door"


def test_the_published_vocabulary_loads(ran: dict[str, Any]) -> None:
    """The half that was broken, and the reason this file exists.

    ``ran`` having been produced at all is the assertion: the fence's own
    ``load_ontology`` is the first line of it, so a document carrying a
    reference into a section it does not declare fails here before any walk.
    """
    assert ran["onto"].id == "mammals"
    assert set(ran["onto"].entity_types) == {"Species", "Breed"}


def test_the_axis_walks_what_the_page_says_it_walks(ran: dict[str, Any]) -> None:
    """The fence's own assertion, restated so a failure names the walk.

    ``runpy`` would raise ``AssertionError`` from inside the executed copy
    with no indication of which claim broke, and this page makes several.
    """
    assert tuple(ran["species"].walk()) == ("mammal", "dog", "retriever", "beagle")


def test_the_declared_attribute_is_inherited_down_the_type_lattice(
    ran: dict[str, Any],
) -> None:
    """``lifespan_years`` answers for the type that declares it and the one below.

    The attribute the vocabulary asserts is now declared, which is what makes
    the assertion resolvable -- and declaring it on ``Species`` is what makes
    ``Breed`` answer for it too, by ``isa``. That inheritance is what the
    prose below this fence is about, and it was not previously reachable from
    this document at all: nothing declared the attribute.
    """
    onto = ran["onto"]

    assert [a.name for a in onto.inherited_attributes("Species")] == ["lifespan_years"]
    assert [a.name for a in onto.inherited_attributes("Breed")] == ["lifespan_years"]

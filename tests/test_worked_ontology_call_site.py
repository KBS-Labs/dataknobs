"""The published call site runs, from outside the package that implements it.

A vocabulary someone typed should be loadable, walkable and resolvable against
with no database, no embedder and no event loop. That claim is easy to make in
a design document and cheap to check exactly once: **a call site fails.** It
cannot be written around a class that does not exist, or around a name a door
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
the first; the code is executed from
:mod:`worked_ontology_call_site`, which is asserted character-for-character
against the second. Nothing here paraphrases the page, so a page that goes
wrong takes the suite with it.

**What that does not check, and should not claim to.** It pins the *published*
call site against the *executed* one. Whether the published one still matches
the design it was drawn from is a judgement someone made when the page was
written; after that, the page is the artifact a consumer actually reads, and it
is the copy worth guarding.

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

from tests._workspace import ROOT, code_fences

if TYPE_CHECKING:
    from collections.abc import Iterator

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "ontology.md"
EXECUTED = ROOT / "tests" / "worked_ontology_call_site.py"

INPUT_MARKER = "worked-input"
CALL_SITE_MARKER = "worked-call-site"


def _fence(marker: str) -> str:
    """The body of the guide's fence carrying ``marker``.

    Refuses rather than returns empty. A marker that has been renamed or
    deleted would otherwise hand every assertion below an empty string, and
    two empty strings compare equal -- a guard reporting green because it read
    nothing, over the one page it exists to read.
    """
    fences = [f for f in code_fences(GUIDE) if f.marker == marker]
    if len(fences) != 1:
        pytest.fail(
            f"{GUIDE.relative_to(ROOT)} carries {len(fences)} fences marked "
            f"<!-- {marker} -->, expected exactly one. The guide and this guard "
            f"agree on these markers and on nothing else."
        )
    body = fences[0].body
    if not body.strip():
        pytest.fail(f"the <!-- {marker} --> fence is empty")
    return body


def _executed_source() -> str:
    """The executed copy, less its own docstring header.

    The split is on the first blank line after the closing ``\"\"\"``, which is
    the boundary the executed module's own docstring describes. Anything else
    in that file would be an assertion a reader of the guide never sees.
    """
    text = EXECUTED.read_text(encoding="utf-8")
    _, _, after = text.partition('"""\n\n')
    if not after:
        pytest.fail(f"{EXECUTED.relative_to(ROOT)} has no docstring to split on")
    return after.rstrip("\n")


@pytest.fixture
def vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``mammals.yaml`` on disk, written from the guide, and made the cwd.

    The call site names the file by a bare relative path, which is what a
    reader would type. Honouring that means giving it a directory rather than
    rewriting the line to suit the harness.
    """
    (tmp_path / "mammals.yaml").write_text(_fence(INPUT_MARKER) + "\n", encoding="utf-8")
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
    published = _fence(CALL_SITE_MARKER)
    executed = _executed_source()

    assert executed == published, (
        "the executed call site and the one the guide publishes have diverged. "
        "They are one text with two homes: edit the fence in "
        f"{GUIDE.relative_to(ROOT)} and copy it to {EXECUTED.relative_to(ROOT)}, "
        "or the reverse -- but never one alone."
    )


def test_the_call_site_imports_only_through_the_doors() -> None:
    """Every import in the block is a package door, not a module path.

    The point of the leg this test arrives with is that these names are
    *exported*. A call site reaching into ``dataknobs_common.ontology.model``
    would run identically and would prove nothing about that, so the spelling
    is the subject and not an incidental.
    """
    doors = {"dataknobs_common", "dataknobs_common.hierarchy", "dataknobs_common.ontology"}
    reached = {
        line.split()[1]
        for line in _fence(CALL_SITE_MARKER).splitlines()
        if line.startswith("from dataknobs_common")
    }

    assert reached, "the call site imports nothing from this package"
    assert reached <= doors, (
        f"{sorted(reached - doors)} is a module path rather than a door. "
        f"The call site is the increment's own acceptance that the names are "
        f"published, so reaching past a door defeats it"
    )


def test_a_typed_surface_form_reaches_the_entity(ran: dict[str, Any]) -> None:
    """Step 1: the reader typed something that is not the entity's name."""
    onto = ran["onto"]

    assert onto.entities.by_surface_form("beagles") == frozenset({"beagle"})
    assert ran["beagle"].name == "Beagle"


def test_the_structure_walks_to_the_root(ran: dict[str, Any]) -> None:
    """Step 2: an axis of the vocabulary, walked by a module-level function.

    ``ancestors`` is generic over ``Hierarchy`` rather than a method on it, so
    what this asserts is that the concrete and the walk compose without either
    knowing the other.
    """
    assert ran["ancestors"](ran["structure"], "beagle") == ("dog", "mammal")


def test_the_vocabulary_answers_what_it_asserted(ran: dict[str, Any]) -> None:
    """Step 3: the assertion is found by subject and relation, and is asserted."""
    found = ran["onto"].assertions.find(subject="beagle", relation="isa")

    assert [a.object.entity_id for a in found] == ["dog"]
    assert {a.polarity.value for a in found} == {"asserted"}


def test_an_unbacked_origin_says_so_rather_than_inventing_one(ran: dict[str, Any]) -> None:
    """Step 4: the reference is real and the backing is not, and both are legible.

    This is the step that would be easiest to fake. ``beagle.source`` names a
    system this vocabulary has never opened, so the honest answer to
    ``fetch_origin`` is ``None`` -- and ``describe()`` is where a caller finds
    out why, which is why it is asserted here beside it rather than trusted.
    """
    onto, beagle = ran["onto"], ran["beagle"]

    assert beagle.source.source_id == "clinic_db"
    assert onto.entities.fetch_origin(beagle.source) is None
    assert onto.entities.describe().source_id != beagle.source.source_id


def test_the_resolver_ranks_the_same_placement_with_its_reason(ran: dict[str, Any]) -> None:
    """Step 5: the placement again, ranked, from a file declaring no resolver.

    The vocabulary has no ``resolver:`` section, so this also pins that the
    default cascade is built rather than refused -- without which the line is
    spellable and returns nothing worth asserting on.
    """
    result = ran["result"]

    assert result.candidates, "the default cascade placed nothing"
    assert result.candidates[0].entity_id == "beagle"
    assert result.candidates[0].evidence[0].kind.value == "declared"


def test_nothing_in_the_call_site_needs_a_loop_or_a_backend(ran: dict[str, Any]) -> None:
    """The claim the whole page opens with, asserted rather than repeated.

    ``ran`` having been produced at all is most of the proof -- the module ran
    to completion outside any event loop, so nothing in it awaited. What is
    left to check is the other half: that the vocabulary describes itself as
    authored, rather than as something with a backend behind it.
    """
    description = ran["onto"].entities.describe()

    assert description.backend == "authored"
    assert description.table is None

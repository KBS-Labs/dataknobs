"""The published registry call site runs, from outside the packages that ship it.

A cascade's rungs come from more than one distribution by design, and this is
the page where that stops being a claim: the composition is written in a
*document*, the rung that reads a vector store is registered by the package
that implements it, and the registry is what holds both ends. None of that can
be shown from inside either package --- a test living in ``packages/data``
reaches anything that package defines whether or not a door exports it, which
makes it structurally unable to fail for the one reason this file exists.

:mod:`tests.test_worked_entity_resolution_call_site` is the sibling this
copies, and states the same reason for the same location. What differs is the
door set: that page is ``dataknobs_common``'s and teaches a foreign rung
through the *mark* rather than the import, so its guard admits three ``common``
doors and no others. This page **owns** the class, so its block imports it ---
and the two guards each check their own page against their own set rather than
one guard being widened to admit both.

**The block asserts inside itself, and that is forced rather than preferred.**
``runpy.run_path`` cannot execute a module-scope ``await``, so the published
shape is ``async def main()`` / ``asyncio.run(main())`` --- and under that
shape nothing a reader would care about is left at module scope for a guard to
read. The block therefore carries its own assertions and binds what it
returns, which is the form three shipped guides already use.
"""

from __future__ import annotations

import ast
import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._workspace import ROOT, door_imports, executed_source, published_fence

if TYPE_CHECKING:
    from collections.abc import Iterator

GUIDE = ROOT / "packages" / "data" / "docs" / "guides" / "ontology-registry.md"
EXECUTED = ROOT / "tests" / "worked_ontology_registry_call_site.py"

CALL_SITE_MARKER = "worked-call-site"


@pytest.fixture
def ran() -> Iterator[dict[str, Any]]:
    """What the call site leaves bound, having run it exactly as published."""
    yield runpy.run_path(str(EXECUTED), run_name="__worked_call_site__")


def test_the_executed_copy_is_the_published_one() -> None:
    """Character for character, in both directions.

    This is what makes *verbatim* a property of the tree rather than an
    intention. Without it the guide could drift from the code, or the code
    from the guide, and every other assertion in this file would keep passing.
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

    Three doors and two distributions, which is the page's subject: a
    consumer writes one document, hands it to one registry, and the rung that
    reads their vector store arrives because the package holding it registered
    the kind. A block reaching into ``dataknobs_data.entity_resolution``
    directly would run identically and would show the opposite --- that you
    have to know where the class lives.
    """
    doors = {
        "dataknobs_common.ontology",
        "dataknobs_data.ontology",
        "dataknobs_data.testing",
    }
    fence = published_fence(GUIDE, CALL_SITE_MARKER)

    for package in ("dataknobs_common", "dataknobs_data"):
        reached = door_imports(fence, package=package)
        assert reached <= doors, (
            f"{sorted(reached - doors)} is a module path rather than a door. "
            f"The call site is this page's own acceptance that the names are "
            f"published, so reaching past a door defeats it"
        )
    assert door_imports(fence, package="dataknobs_data"), (
        "the call site imports nothing from the package whose page this is"
    )


def test_the_call_site_names_no_rung_and_builds_them_from_the_document() -> None:
    """The composition is configuration here, which is the whole claim.

    The sibling page's block constructs its rung by name, and is right to: it
    teaches the class. This one teaches the *door*, so a block that imported
    ``SemanticSignal`` and handed it to a cascade would satisfy every other
    assertion in this file while showing a reader the one path that does not
    need a registry at all.
    """
    published = published_fence(GUIDE, CALL_SITE_MARKER)

    assert "SemanticSignal" not in published, (
        "the published block constructs the rung instead of configuring it, "
        "which is the sibling page's job and not this one's"
    )
    assert '"kind": "semantic"' in published, (
        "the block no longer writes the rung as configuration, so nothing here "
        "shows that a document can reach it"
    )

    defined = [
        node.name for node in ast.walk(ast.parse(published)) if isinstance(node, ast.ClassDef)
    ]
    assert not defined, (
        f"the published call site defines {defined} instead of using what ships"
    )


def test_the_block_carries_its_own_assertions(ran: dict[str, Any]) -> None:
    """Running it *is* the check, so a block with no assertions checks nothing.

    Under the ``asyncio.run(main())`` shape a guard cannot read locals out of
    the run, so what a reader is being shown is true only if the block says so
    itself. A block whose assertions were deleted would still run clean here
    and would still be published as a worked example.
    """
    published = published_fence(GUIDE, CALL_SITE_MARKER)

    asserted = [node for node in ast.walk(ast.parse(published)) if isinstance(node, ast.Assert)]
    assert len(asserted) >= 4, (
        "the published block no longer asserts what it teaches, so running it "
        "proves only that it does not raise"
    )
    assert ran["result"] is not None, "the block binds nothing a reader can go on to use"


def test_the_cascade_reached_through_the_registry_holds_both_kinds(
    ran: dict[str, Any],
) -> None:
    """One entity two rungs reached, and one only the embedding did.

    The first is what a shared id space buys: without the rung localizing on
    the way out, ``sku-4471`` comes back twice under two keys and ``explain``
    finds each under a different one. The second is what the vector rung is
    *for* --- an entity no declared form in the query reached at all.
    """
    result = ran["result"]

    ids = [candidate.entity_id for candidate in result.candidates]
    assert ids.count("sku-4471") == 1
    assert not any(":" in entity_id for entity_id in ids), (
        "a qualified id here is the two-space failure the localization prevents"
    )
    assert sorted(evidence.signal for evidence in result.explain("sku-4471")) == [
        "exact",
        "semantic",
    ]

    inferred = [candidate for candidate in result.candidates if not candidate.declared]
    assert inferred, "no candidate came from the embedding alone, so the page shows nothing"
    assert all(
        [evidence.signal for evidence in candidate.evidence] == ["semantic"]
        for candidate in inferred
    )

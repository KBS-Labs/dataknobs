"""Reaching an axis from the ontology, walking it, and being refused one.

``taxonomy(name)`` is an accessor, not a factory: everything it needs is a
field the ontology already holds, which is why it takes only the name. The
tests here assert that in both directions -- what a caller does *not* have to
supply, and what the accessor refuses to build rather than downgrading in
silence.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.taxonomy import AsyncTaxonomy, Taxonomy
from dataknobs_common.testing import assert_twins_agree

# --------------------------------------------------------------------------
# Criterion 13 -- the axis is reachable, and it takes nothing but its name
# --------------------------------------------------------------------------


def test_the_axis_walks_the_authored_terms(mammals_v11_path: Path) -> None:
    """Every term the file declares, breadth first from the axis's root."""
    onto = load_ontology(mammals_v11_path)

    walked = tuple(onto.taxonomy("species").walk())

    assert walked == ("mammal", "dog", "retriever", "beagle", "golden_retriever")


def test_the_accessor_takes_only_the_axis_name(mammals_v11_path: Path) -> None:
    """No store, no loop, and no consumer-supplied collaborator.

    The whole of what makes this an accessor rather than a factory: every
    argument a ``Taxonomy`` needs is already a field of the ontology, so the
    caller supplies a name and nothing else.
    """
    onto = load_ontology(mammals_v11_path)

    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    parameters = inspect.signature(type(onto).taxonomy).parameters
    assert list(parameters) == ["self", "name"]

    axis = onto.taxonomy("species")
    assert isinstance(axis, Taxonomy)
    assert isinstance(axis.structure, AssertionHierarchy)
    assert axis.entities is onto.entities
    assert axis.assertions is onto.assertions
    assert axis.definition is onto.taxonomies["species"]


def test_the_walk_starts_where_it_is_told(mammals_v11_path: Path) -> None:
    """``from_id`` anchors the walk, and the anchor is included."""
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert tuple(axis.walk(from_id="dog")) == (
        "dog",
        "retriever",
        "beagle",
        "golden_retriever",
    )


def test_max_depth_bounds_the_levels_expanded(mammals_v11_path: Path) -> None:
    """Zero is the anchor alone, and each step adds one level."""
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert tuple(axis.walk(from_id="dog", max_depth=0)) == ("dog",)
    assert tuple(axis.walk(from_id="dog", max_depth=1)) == ("dog", "retriever", "beagle")


def test_the_walk_yields_each_node_once(mammals_v11_path: Path) -> None:
    """A DAG node reachable by two paths is still one node."""
    walked = tuple(load_ontology(mammals_v11_path).taxonomy("species").walk())

    assert len(walked) == len(set(walked))


@pytest.mark.asyncio
async def test_the_async_twin_walks_the_same_terms(mammals_v11_path: Path) -> None:
    """Same order, same terms, awaited -- and ``taxonomy()`` is not awaited.

    A plain ``def`` on the asynchronous twin, because it constructs over fields
    the object is already holding. Asserted rather than assumed: making it
    awaitable would cost every caller an ``await`` for a dictionary lookup.
    """
    onto = await async_load_ontology(mammals_v11_path)

    axis = onto.taxonomy("species")

    assert isinstance(axis, AsyncTaxonomy)
    assert isinstance(axis.structure, AsyncAssertionHierarchy)
    assert not inspect.iscoroutinefunction(type(onto).taxonomy)

    walked = [node async for node in axis.walk()]
    assert tuple(walked) == ("mammal", "dog", "retriever", "beagle", "golden_retriever")


# --------------------------------------------------------------------------
# Criterion 14 -- refused at the accessor, naming the axis
# --------------------------------------------------------------------------


def test_a_materialized_content_axis_is_refused_naming_the_axis(
    materialized_content_path: Path,
) -> None:
    """Refused where the caller asked, not at the first walk.

    A loader that silently downgrades is a loader whose output nobody can
    reason about, and a failure at first walk is a failure at a call site with
    no idea why. The message names the axis, so the answer is actionable
    without reading this code.
    """
    onto = load_ontology(materialized_content_path)

    with pytest.raises(ValidationError) as raised:
        onto.taxonomy("species")

    assert "species" in str(raised.value)
    assert raised.value.context["taxonomy"] == "species"
    assert raised.value.context["axis"] == "content"


def test_a_materialized_structure_axis_is_refused_naming_the_axis(
    materialized_structure_path: Path,
) -> None:
    """The snapshot branch is declared everywhere and taken by no door.

    ``MATERIALIZED`` structure is a snapshot with a build time; ``ON_DEMAND``
    is the live read. The copy is now buildable --
    ``MappingHierarchy.snapshot`` takes one from any axis -- so what a
    definition asking for the snapshot is asking for is not a branch that does
    not exist, but one this accessor does not reach for. The honest answer is
    still to say so here rather than hand back the live axis under the other
    name, and the message says which of the two it is.
    """
    onto = load_ontology(materialized_structure_path)

    with pytest.raises(ValidationError) as raised:
        onto.taxonomy("species")

    assert "species" in str(raised.value)
    assert raised.value.context["taxonomy"] == "species"
    assert raised.value.context["axis"] == "structure"


def test_the_default_axes_are_the_ones_that_exist(mammals_v11_path: Path) -> None:
    """A vocabulary that declares no ``materialization:`` still builds.

    The half a refusal is most likely to break, and it is the half the caller
    never types: both defaults must name a branch that exists, or every
    hand-edited file in the plan is refused for asking nothing.

    ``structure`` defaults to the live read because that is what the axis
    built here *is* -- an ``AssertionHierarchy``, which opens nothing and
    caches nothing. A default naming the snapshot would be a default no
    consumer could get.
    """
    definition = load_ontology(mammals_v11_path).taxonomies["species"]

    assert definition.materialization.structure.value == "on_demand"
    assert definition.materialization.content.value == "on_demand"
    assert tuple(load_ontology(mammals_v11_path).taxonomy("species").walk())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture_name",
    ["materialized_content_path", "materialized_structure_path"],
)
async def test_both_flavours_refuse_identically(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    """One rule, and neither twin carries its own copy of it.

    Over both refusals rather than one: the rule they share is
    ``_refuse_unbuildable_axis``, so a second refusal added to only one twin
    is exactly what this is here to catch.
    """
    path: Path = request.getfixturevalue(fixture_name)

    with pytest.raises(ValidationError) as sync_raised:
        load_ontology(path).taxonomy("species")
    with pytest.raises(ValidationError) as async_raised:
        (await async_load_ontology(path)).taxonomy("species")

    assert str(sync_raised.value) == str(async_raised.value)
    assert sync_raised.value.context == async_raised.value.context


def test_an_undeclared_axis_is_refused_listing_what_is_declared(
    mammals_v11_path: Path,
) -> None:
    """The useful answer to a typo is the set it was nearly one of."""
    onto = load_ontology(mammals_v11_path)

    with pytest.raises(NotFoundError) as raised:
        onto.taxonomy("speceis")

    assert "speceis" in str(raised.value)
    assert raised.value.context["declared"] == ["species"]


# --------------------------------------------------------------------------
# The twins, and what the axis carries for the cursor that comes later
# --------------------------------------------------------------------------


def test_the_taxonomy_twins_expose_the_same_annotated_surface() -> None:
    """Same fields, same walk surface; the one difference is declared.

    ``max_concurrency`` bounds a frontier read with no bulk member to use, and
    the synchronous walk issues no concurrent calls at all -- a knob that does
    nothing is worse than an asymmetry that is stated. The guard compares that
    set by equality, so a second divergence fails rather than joining it.

    No ``compare_return``: this pair is the one that streams, so its return
    annotations differ by flavour (``Iterator[str]`` against
    ``AsyncIterator[str]``) and asserting on them would pin the flavour rather
    than the contract. The hierarchy members, which do not stream, are checked
    the other way.
    """
    sync_fields = Taxonomy.__dataclass_fields__
    async_fields = AsyncTaxonomy.__dataclass_fields__

    assert list(sync_fields) == list(async_fields)

    assert_twins_agree(
        Taxonomy.walk,
        AsyncTaxonomy.walk,
        async_only={"max_concurrency"},
        label="Taxonomy.walk",
    )


def test_the_axis_carries_the_assertions_it_was_built_from(
    mammals_v11_path: Path,
) -> None:
    """What lets a cursor report the edge rather than only its endpoints.

    ``assertions is None`` is the question that tells *this edge carries no
    annotation* from *this axis has no annotations to give* -- a hierarchy over
    a ``parent_id`` column has rows and no assertions at all.
    """
    onto = load_ontology(mammals_v11_path)

    assert onto.taxonomy("species").assertions is onto.assertions

    unannotated = Taxonomy(
        definition=onto.taxonomies["species"],
        structure=AssertionHierarchy(onto.assertions, "isa"),
        entities=onto.entities,
    )
    assert unannotated.assertions is None


def test_taxonomy_does_not_import_the_ontology_package() -> None:
    """The general module stands without the specific package behind it."""
    import subprocess
    import sys

    probe = (
        "import sys, dataknobs_common.taxonomy; print('dataknobs_common.ontology' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------------
# The anchor, and the member that was declared to answer for it
# --------------------------------------------------------------------------


def test_an_anchor_the_axis_does_not_contain_is_refused(mammals_v11_path: Path) -> None:
    """An unknown ``from_id`` is refused rather than yielded back.

    The anchor is *included* in the output by design, so seeding the frontier
    with it unchecked emits an id the axis does not contain as though it were a
    term of the axis -- and the caller cannot tell the difference, because a
    walk is exactly what they asked for.

    Yielding nothing was the other candidate and is the worse one. It collapses
    *nothing below this node* into *this node is not here*, which are the two
    answers ``Hierarchy.contains`` says in its own docstring it exists to keep
    apart. An axis that owns that member and then destroys the distinction in
    the one walk that needs it is refuting itself.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert not axis.structure.contains("marmoset")

    with pytest.raises(NotFoundError) as raised:
        tuple(axis.walk(from_id="marmoset"))

    assert "marmoset" in str(raised.value)
    assert raised.value.context["anchor"] == "marmoset"
    assert raised.value.context["taxonomy"] == "species"


@pytest.mark.asyncio
async def test_both_flavours_refuse_an_unknown_anchor_alike(
    mammals_v11_path: Path,
) -> None:
    """The twins refuse with the same type and the same words.

    The differential rather than either surface alone: one algorithm behaving
    two ways by flavour is what this pair's shape exists to prevent, and a
    check written into one streaming walk and forgotten in the other is the
    specific way that happens here -- the two walks are twinned by hand.
    """
    onto = load_ontology(mammals_v11_path)
    async_onto = await async_load_ontology(mammals_v11_path)

    with pytest.raises(NotFoundError) as sync_raised:
        tuple(onto.taxonomy("species").walk(from_id="marmoset"))

    with pytest.raises(NotFoundError) as async_raised:
        [node async for node in async_onto.taxonomy("species").walk(from_id="marmoset")]

    assert str(sync_raised.value) == str(async_raised.value)
    assert sync_raised.value.context == async_raised.value.context


def test_a_known_anchor_still_walks(mammals_v11_path: Path) -> None:
    """The check costs the walk nothing it was already answering.

    Pinned beside the refusal because a containment check placed wrongly --
    against the entity source rather than the structure, say -- would refuse
    every anchor and still satisfy the test above.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert tuple(axis.walk(from_id="dog")) == (
        "dog",
        "retriever",
        "beagle",
        "golden_retriever",
    )

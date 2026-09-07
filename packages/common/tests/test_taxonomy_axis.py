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


def test_the_default_materialization_is_buildable(mammals_v11_path: Path) -> None:
    """The shipped defaults are exactly the pair that needs no store.

    Stated as its own test because it is the half the refusal could break:
    ``structure`` defaults to ``materialized`` -- ids and edges, small enough
    to hold -- and a refusal keyed on *any* materialized axis would reject
    every hand-edited vocabulary in the plan.
    """
    definition = load_ontology(mammals_v11_path).taxonomies["species"]

    assert definition.materialization.structure.value == "materialized"
    assert definition.materialization.content.value == "on_demand"
    assert tuple(load_ontology(mammals_v11_path).taxonomy("species").walk())


@pytest.mark.asyncio
async def test_both_flavours_refuse_identically(
    materialized_content_path: Path,
) -> None:
    """One rule, and neither twin carries its own copy of it."""
    path = materialized_content_path

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
    """Same fields, same parameters; the annotations differ only by flavour."""
    sync_fields = Taxonomy.__dataclass_fields__
    async_fields = AsyncTaxonomy.__dataclass_fields__

    assert list(sync_fields) == list(async_fields)

    sync_walk = inspect.signature(Taxonomy.walk)
    async_walk = inspect.signature(AsyncTaxonomy.walk)
    assert sync_walk.parameters.keys() == async_walk.parameters.keys()
    for name, parameter in sync_walk.parameters.items():
        assert parameter.default == async_walk.parameters[name].default, name
        if name != "self":
            assert parameter.annotation == async_walk.parameters[name].annotation, name


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

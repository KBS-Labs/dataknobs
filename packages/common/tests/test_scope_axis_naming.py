"""The axis a bare ``within`` scopes on is named for the wrong id space.

A source publishes one axis by default, and the values it answers with are
``Entity.type`` -- ``Breed``, ``Species``. The same package keys
:attr:`~dataknobs_common.ontology.values.Ontology.taxonomies` by
:attr:`~dataknobs_common.ontology.model.TaxonomyDefinition.id`. While that
axis was called ``taxonomy_id``, a consumer holding a genuine taxonomy id was
told by the name itself to pass it there, and passing it returned zero
candidates with nothing said.

**The refusal that exists for exactly this does not reach it.**
:func:`~dataknobs_common.entity_resolution.refuse_unknown_axes` refuses an
axis name the source will not answer for; ``taxonomy_id`` was the name it
*did* answer for. So the scope was admitted and then compared against the
wrong id space, which is the one failure the refusal cannot see -- a correctly
spelled axis carrying a value from another vocabulary.

These tests are the reproduction and the repair, kept together because the
repair is only judged by what the caller can now tell. Renaming the axis does
not make a taxonomy id work; it makes passing one *say so*, by moving the
mistake back inside the refusal's reach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataknobs_common.entity_resolution import (
    ENTITY_TYPE_KEY,
    CascadingResolver,
    ExactNormalizedSignal,
    within_axis_names,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import load_ontology

if TYPE_CHECKING:
    from pathlib import Path

    from dataknobs_common.ontology import Ontology


@pytest.fixture
def onto(mammals_v11_path: Path) -> Ontology:
    """A vocabulary whose taxonomy id and entity types are different strings.

    ``taxonomies:`` declares ``species``; the entities are typed ``Species``
    and ``Breed``. The two id spaces differ here by a single capital, which is
    what makes the confusion this module is about easy to reach and hard to
    see.
    """
    return load_ontology(mammals_v11_path)


@pytest.fixture
def resolver(onto: Ontology) -> CascadingResolver:
    """One exact rung over the vocabulary's own entities."""
    return CascadingResolver([ExactNormalizedSignal(onto.entities)], onto.entities)


def test_the_two_id_spaces_really_are_different(onto: Ontology) -> None:
    """The premise, asserted rather than assumed.

    Every test below is about a caller confusing these two. If the fixture
    ever declared a taxonomy whose id matched an entity type, the confusion
    would be unreachable and the rest of this module would pass while
    measuring nothing.
    """
    taxonomy_ids = set(onto.taxonomies)
    entity_types = {onto.entities.get(eid).type for eid in ("beagle", "dog")}

    assert taxonomy_ids == {"species"}
    assert entity_types == {"Breed", "Species"}
    assert taxonomy_ids & entity_types == set()


def test_the_published_axis_is_named_for_the_values_it_holds(onto: Ontology) -> None:
    """The default axis answers with entity types, so that is what it is called.

    The rename is the whole of item 2's fix, and this is the assertion that
    the name and the values agree. It fails if the axis is renamed back, and
    it fails if the projection starts answering with something else.
    """
    assert ENTITY_TYPE_KEY == "entity_type"
    assert within_axis_names(onto.entities) == frozenset({ENTITY_TYPE_KEY})


def test_a_genuine_taxonomy_id_is_refused_rather_than_silently_admitted(
    onto: Ontology, resolver: CascadingResolver
) -> None:
    """The defect, and the shape of its repair.

    Passing ``{"taxonomy_id": "species"}`` -- a real axis name under the old
    spelling, carrying a real taxonomy id -- used to be admitted, compared
    against ``Entity.type``, and answered with an empty candidate list. The
    caller saw the same result an empty vocabulary gives.

    It is now outside the published axis set, so the refusal that was always
    there reaches it: the error names the axis asked for and the axis this
    source does publish, which is enough to see that the value was fine and
    the key was not.
    """
    with pytest.raises(ValidationError) as raised:
        resolver.resolve("beagle", k=5, within={"taxonomy_id": "species"})

    message = str(raised.value)
    assert "taxonomy_id" in message, "the refusal must name what was asked for"
    assert ENTITY_TYPE_KEY in message, "and what this source does publish"


def test_the_same_query_under_the_right_axis_places_the_entity(
    resolver: CascadingResolver,
) -> None:
    """The positive control, so the refusal above is not passing on a dead query.

    A refusal test is satisfied by a query that could never have matched. This
    is the same query and the same vocabulary with the key corrected, and it
    returns the candidate -- so the scope, not the query, is what the previous
    test is about.
    """
    placed = resolver.resolve("beagle", k=5, within={ENTITY_TYPE_KEY: "Breed"})

    assert [c.entity_id for c in placed.candidates] == ["beagle"]


def test_the_bare_spelling_still_means_the_default_axis(
    resolver: CascadingResolver,
) -> None:
    """The rename stays sugar-preserving, which is why it costs no caller.

    ``within="Breed"`` and ``within={ENTITY_TYPE_KEY: "Breed"}`` are one
    scope. That was true of the old name too, and it is the reason a rename
    was the whole fix rather than half of one.
    """
    bare = resolver.resolve("beagle", k=5, within="Breed")
    keyed = resolver.resolve("beagle", k=5, within={ENTITY_TYPE_KEY: "Breed"})

    assert [c.entity_id for c in bare.candidates] == [c.entity_id for c in keyed.candidates]

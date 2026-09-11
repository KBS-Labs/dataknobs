"""Asking an authored assertion list what it holds.

``find_many`` is here because no acceptance criterion reaches it and it is the
member a hierarchy walk actually calls: one query per axis rather than one per
node. A bulk member with no behavioural test is where the first consumer finds
the bugs.
"""

from __future__ import annotations

import pytest

from dataknobs_common.fields import FieldType
from dataknobs_common.ontology import (
    Assertion,
    EntityRef,
    Literal,
    Polarity,
    RelationType,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.ontology.sources import (
    AsyncMappingAssertionSource,
    MappingAssertionSource,
    relation_id,
)

EDGES = [
    Assertion(id="a1", subject="beagle", relation="isa", object=EntityRef("dog")),
    Assertion(id="a2", subject="dog", relation="isa", object=EntityRef("mammal")),
    Assertion(id="a3", subject="corgi", relation="isa", object=EntityRef("dog")),
    Assertion(
        id="a4",
        subject="beagle",
        relation="weighs",
        object=Literal(value=10, type=FieldType.INTEGER, unit="kg"),
    ),
]


@pytest.fixture
def source() -> MappingAssertionSource:
    return MappingAssertionSource(EDGES)


def test_get_finds_by_id(source: MappingAssertionSource) -> None:
    assert source.get("a1") is EDGES[0]
    assert source.get("nope") is None


def test_the_id_index_is_built_rather_than_taken(
    source: MappingAssertionSource,
) -> None:
    """The input is a sequence, so the ids come from the assertions themselves.

    Trusting a caller's keys to agree with ``Assertion.id`` holds until one
    caller builds the mapping some other way, and then ``get`` answers with
    the wrong assertion rather than with None.
    """
    assert source.get("a4").subject == "beagle"


def test_find_narrows_on_each_criterion(source: MappingAssertionSource) -> None:
    assert [a.id for a in source.find(subject="beagle")] == ["a1", "a4"]
    assert [a.id for a in source.find(subject="beagle", relation="isa")] == ["a1"]
    assert [a.id for a in source.find(object=EntityRef("dog"))] == ["a1", "a3"]
    assert [a.id for a in source.find(relation="isa")] == ["a1", "a2", "a3"]


def test_find_with_no_criteria_returns_everything(
    source: MappingAssertionSource,
) -> None:
    assert len(source.find()) == len(EDGES)


def test_find_accepts_a_relation_definition_as_well_as_an_id(
    source: MappingAssertionSource,
) -> None:
    """Both forms are legal in an assertion, so both must be legal in a query."""
    isa = RelationType(id="isa", transitive=True)

    assert [a.id for a in source.find(subject="beagle", relation=isa)] == ["a1"]
    assert relation_id(isa) == relation_id("isa")


def test_find_matches_a_literal_object(source: MappingAssertionSource) -> None:
    """A literal compares by value, unit and type together."""
    matching = Literal(value=10, type=FieldType.INTEGER, unit="kg")

    assert [a.id for a in source.find(object=matching)] == ["a4"]
    assert source.find(object=Literal(value=10, type=FieldType.INTEGER)) == []


def test_find_many_keys_by_the_axis_asked_for(source: MappingAssertionSource) -> None:
    by_subject = source.find_many(subjects=["beagle", "dog"], relation="isa")
    assert {k: [a.id for a in v] for k, v in by_subject.items()} == {
        "beagle": ["a1"],
        "dog": ["a2"],
    }

    by_object = source.find_many(objects=["dog"], relation="isa")
    assert {k: [a.id for a in v] for k, v in by_object.items()} == {"dog": ["a1", "a3"]}


def test_find_many_omits_a_key_with_no_matches(
    source: MappingAssertionSource,
) -> None:
    """A key with no matches is absent, not mapped to an empty list.

    The caller asked which of these have edges; an empty list answers a
    different question, and one they would then have to filter out.
    """
    assert source.find_many(subjects=["beagle", "wombat"]) == {"beagle": [EDGES[0], EDGES[3]]}


def test_find_many_refuses_both_axes_or_neither(
    source: MappingAssertionSource,
) -> None:
    """Keyed by *the* axis: two would make the result ambiguous, none unbounded."""
    with pytest.raises(ValueError, match="exactly one"):
        source.find_many(subjects=["beagle"], objects=["dog"])

    with pytest.raises(ValueError, match="exactly one"):
        source.find_many()


@pytest.mark.asyncio
async def test_the_async_twin_answers_the_same() -> None:
    """Same index, same answers, awaited."""
    source = AsyncMappingAssertionSource(EDGES)

    assert (await source.get("a1")) is EDGES[0]
    assert [a.id for a in await source.find(subject="beagle", relation="isa")] == ["a1"]
    assert await source.find_many(objects=["dog"], relation="isa") == {"dog": [EDGES[0], EDGES[2]]}


def test_an_assertion_the_file_did_not_name_gets_a_derived_id() -> None:
    """Derived from the edge, not from the row's position.

    A positional id renumbers every assertion below the one you inserted,
    which turns adding a line into a change of identity for the rest.
    """
    onto = load_ontology(
        {
            "id": "x",
            "assertions": [
                {"subject": "dog", "relation": "isa", "object": "mammal"},
                {"subject": "beagle", "relation": "isa", "object": "dog"},
            ],
        }
    )

    assert onto.assertions.get("beagle-isa-dog") is not None

    with_a_row_inserted = load_ontology(
        {
            "id": "x",
            "assertions": [
                {"subject": "cat", "relation": "isa", "object": "mammal"},
                {"subject": "dog", "relation": "isa", "object": "mammal"},
                {"subject": "beagle", "relation": "isa", "object": "dog"},
            ],
        }
    )

    assert with_a_row_inserted.assertions.get("beagle-isa-dog") is not None


def test_an_explicit_assertion_id_is_kept() -> None:
    onto = load_ontology(
        {
            "id": "x",
            "assertions": [
                {"id": "the-edge", "subject": "dog", "relation": "isa", "object": "mammal"}
            ],
        }
    )

    assert onto.assertions.get("the-edge") is not None


def test_a_literal_object_is_read_from_a_mapping() -> None:
    """A bare string names an entity; a mapping says which it means."""
    onto = load_ontology(
        {
            "id": "x",
            "assertions": [
                {
                    "subject": "beagle",
                    "relation": "weighs",
                    "object": {"value": 10, "type": "integer", "unit": "kg"},
                },
                {"subject": "beagle", "relation": "isa", "object": {"entity": "dog"}},
            ],
        }
    )

    weight = onto.assertions.find(subject="beagle", relation="weighs")[0]
    assert weight.object == Literal(value=10, type=FieldType.INTEGER, unit="kg")

    parent = onto.assertions.find(subject="beagle", relation="isa")[0]
    assert parent.object == EntityRef("dog")


# --------------------------------------------------------------------------
# A stated negation loads, and a query can select on it
# --------------------------------------------------------------------------
#
# One criterion and not two on purpose. A loader that reads `polarity:` is
# worth nothing if no query can select on it, and a parameter is worth nothing
# if the loader discards what it would select; splitting them lets either half
# pass alone.

#: A vocabulary that says one thing it does *not* believe.
#:
#: An ontology is open world -- an assertion nobody wrote is unknown, not
#: false -- so *a whale is not a fish* is a fact a document has to be able to
#: state rather than one it can imply by omission.
NEGATION_DOCUMENT = {
    "id": "mammals",
    "assertions": [
        {"subject": "dog", "relation": "isa", "object": "mammal"},
        {"subject": "whale", "relation": "isa", "object": "fish", "polarity": "negated"},
    ],
}


def test_an_authored_negation_survives_the_load() -> None:
    """The key used to be read by nothing and dropped, which is worse than
    refusing it: the file said *not* and the vocabulary held the positive.
    """
    onto = load_ontology(NEGATION_DOCUMENT)

    stated = onto.assertions.get("whale-isa-fish")
    assert stated is not None
    assert stated.polarity is Polarity.NEGATED


def test_a_row_that_says_nothing_is_asserted() -> None:
    """The default is what every assertion written before the field meant."""
    onto = load_ontology(NEGATION_DOCUMENT)

    assert onto.assertions.get("dog-isa-mammal").polarity is Polarity.ASSERTED


def test_find_selects_on_polarity() -> None:
    """Given, it narrows; omitted, it does not constrain.

    The omitted case is the one worth pinning: a caller asking *what does this
    vocabulary say about x* wants the negation in the answer, and only a
    caller walking a structure wants it gone.
    """
    onto = load_ontology(NEGATION_DOCUMENT)

    asserted = onto.assertions.find(relation="isa", polarity=Polarity.ASSERTED)
    assert [a.subject for a in asserted] == ["dog"]

    negated = onto.assertions.find(relation="isa", polarity=Polarity.NEGATED)
    assert [a.subject for a in negated] == ["whale"]

    assert [a.subject for a in onto.assertions.find(relation="isa")] == ["dog", "whale"]


def test_find_many_selects_on_polarity_too() -> None:
    """The bulk form filters identically, and a key left with no matches goes.

    Both members take the parameter because both are read members of the
    protocol -- a bulk form that ignored it would make the answer depend on
    which shape of query a caller happened to use.
    """
    onto = load_ontology(NEGATION_DOCUMENT)

    both = onto.assertions.find_many(subjects=["dog", "whale"], relation="isa")
    assert sorted(both) == ["dog", "whale"]

    asserted = onto.assertions.find_many(
        subjects=["dog", "whale"], relation="isa", polarity=Polarity.ASSERTED
    )
    assert sorted(asserted) == ["dog"]


@pytest.mark.asyncio
async def test_the_async_twin_selects_on_polarity_the_same() -> None:
    """The parameter is on both protocol twins, awaited on one of them."""
    onto = await async_load_ontology(NEGATION_DOCUMENT)

    stated = await onto.assertions.get("whale-isa-fish")
    assert stated.polarity is Polarity.NEGATED

    asserted = await onto.assertions.find(relation="isa", polarity=Polarity.ASSERTED)
    assert [a.subject for a in asserted] == ["dog"]

    bulk = await onto.assertions.find_many(
        subjects=["dog", "whale"], relation="isa", polarity=Polarity.ASSERTED
    )
    assert sorted(bulk) == ["dog"]

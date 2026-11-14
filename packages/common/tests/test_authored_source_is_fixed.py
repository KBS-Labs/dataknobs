"""An authored source cannot change under its index.

An authored source answers ``get()`` from the objects it was handed and every
lookup from maps built once at construction. If either the objects or the
containers holding them could change afterwards, the two kinds of member would
disagree and nothing would report it. Five routes to that disagreement were
measured, and each has a test below asserting the members agree after the
change was attempted:

1. appending an alias to an entity ``get()`` returned;
2. setting that entity's ``type``;
3. setting an assertion's ``subject``;
4. adding an entity to the mapping the source was built from;
5. appending an assertion to the sequence the source was built from.

The first three are closed by the model: its value types are frozen and their
collection fields immutable, so the attempt itself raises. The last two no
freeze can reach, and are closed by each index copying the container it is
handed. Every case runs on both flavours of source.
"""

from __future__ import annotations

import copy
import inspect
import pickle
from collections.abc import Hashable
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from typing import Any

import pytest

from dataknobs_common.copying import FrozenMapping
from dataknobs_common.entity_resolution.values import (
    CompatibilityVerdict,
    EvidenceKind,
    ResolutionRef,
    Scoring,
)
from dataknobs_common.fields import FieldType
from dataknobs_common.ontology.model import (
    Assertion,
    AttributeDef,
    Entity,
    EntityRef,
    EntityType,
    Literal,
    Provenance,
    RelationType,
    SourceRef,
)
from dataknobs_common.ontology.sources import (
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    MappingAssertionSource,
    MappingEntitySource,
)

ENTITY_FLAVOURS = [MappingEntitySource, AsyncMappingEntitySource]
ASSERTION_FLAVOURS = [MappingAssertionSource, AsyncMappingAssertionSource]


async def _ask(source: Any, member: str, *args: Any, **kwargs: Any) -> Any:
    """Call ``member`` on either flavour, awaiting the asynchronous one."""
    answer = getattr(source, member)(*args, **kwargs)
    return await answer if inspect.isawaitable(answer) else answer


def _beagle() -> Entity:
    """Built with a list, as a loaded document and most callers hand one.

    A tuple here would make row 1 pass for the wrong reason: ``append`` is
    missing from a tuple whether or not the model coerces anything.
    """
    return Entity(id="beagle", type="Breed", name="Beagle", aliases=["English Beagle"])  # type: ignore[arg-type]


# --- the five measured routes -------------------------------------------------


@pytest.mark.parametrize("flavour", ENTITY_FLAVOURS)
async def test_an_alias_cannot_be_added_to_a_handed_out_entity(flavour: type) -> None:
    """Row 1: an alias ``get()`` shows is an alias ``by_surface_form`` finds."""
    source = flavour({"beagle": _beagle()})
    entity = await _ask(source, "get", "beagle")

    with pytest.raises(AttributeError):
        entity.aliases.append("Snoopy")

    shown = "Snoopy" in (await _ask(source, "get", "beagle")).aliases
    found = "beagle" in await _ask(source, "by_surface_form", "Snoopy")
    assert shown == found


@pytest.mark.parametrize("flavour", ENTITY_FLAVOURS)
async def test_a_handed_out_entitys_type_cannot_be_set(flavour: type) -> None:
    """Row 2: the type ``get()`` shows is the type ``by_type`` lists it under."""
    source = flavour({"beagle": _beagle()})
    entity = await _ask(source, "get", "beagle")

    with pytest.raises(FrozenInstanceError):
        entity.type = "Species"  # type: ignore[misc]

    shown = (await _ask(source, "get", "beagle")).type
    assert "beagle" in await _ask(source, "by_type", shown)
    assert await _ask(source, "by_type", "Species") == frozenset()


@pytest.mark.parametrize("flavour", ASSERTION_FLAVOURS)
async def test_a_handed_out_assertions_subject_cannot_be_set(flavour: type) -> None:
    """Row 3: an assertion stays reachable by the subject ``get()`` shows."""
    source = flavour([Assertion(id="a1", subject="a", relation="isa", object=EntityRef("b"))])
    assertion = await _ask(source, "get", "a1")

    with pytest.raises(FrozenInstanceError):
        assertion.subject = "c"  # type: ignore[misc]

    shown = (await _ask(source, "get", "a1")).subject
    assert [a.id for a in await _ask(source, "find", subject=shown)] == ["a1"]


@pytest.mark.parametrize("flavour", ENTITY_FLAVOURS)
async def test_the_callers_entity_mapping_is_not_the_sources(flavour: type) -> None:
    """Row 4: an entity added to the caller's dict reaches no member."""
    entities = {"beagle": _beagle()}
    source = flavour(entities)

    entities["poodle"] = Entity(id="poodle", type="Breed")

    assert await _ask(source, "get", "poodle") is None
    assert await _ask(source, "by_surface_form", "poodle") == frozenset()
    assert await _ask(source, "by_type", "Breed") == frozenset({"beagle"})


@pytest.mark.parametrize("flavour", ASSERTION_FLAVOURS)
async def test_the_callers_assertion_sequence_is_not_the_sources(flavour: type) -> None:
    """Row 5: an assertion appended to the caller's list reaches no member."""
    assertions = [Assertion(id="a1", subject="a", relation="isa", object=EntityRef("b"))]
    source = flavour(assertions)

    assertions.append(Assertion(id="a2", subject="c", relation="isa", object=EntityRef("b")))

    assert await _ask(source, "get", "a2") is None
    assert [a.id for a in await _ask(source, "find")] == ["a1"]
    assert await _ask(source, "find", subject="c") == []


# --- every frozen type --------------------------------------------------------


def _source_ref() -> SourceRef:
    return SourceRef(source_id="s", kind="table", locator={"row": {"id": [1, 2]}})


def _resolution_ref() -> ResolutionRef:
    return ResolutionRef(
        query="beagle",
        entity_id="beagle",
        score=1.0,
        scoring=Scoring.DECLARED,
        signal="exact",
        kind=EvidenceKind.DECLARED,
        compatibility=CompatibilityVerdict.COMPATIBLE,
        corpus={"embedders": ["m1"]},
    )


def _instances() -> dict[str, Any]:
    """One instance of each frozen type, holding every collection field."""
    return {
        "Entity": Entity(
            id="beagle",
            type="Breed",
            aliases=("English Beagle",),
            metadata={"tags": ["hound"]},
            source=_source_ref(),
        ),
        "EntityType": EntityType(
            id="Breed",
            attributes=(AttributeDef(name="size", value_type="enum", enum_values=("s", "m")),),
        ),
        "RelationType": RelationType(id="isa", domain=frozenset({"Breed"})),
        "AttributeDef": AttributeDef(name="size", value_type="enum", enum_values=("s", "m")),
        "SourceRef": _source_ref(),
        "Provenance": Provenance(
            source=_source_ref(),
            resolution=_resolution_ref(),
            asserted_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        "Literal": Literal(value=3, type=FieldType.INTEGER, metadata={"unit": ["kg"]}),
        "Assertion": Assertion(
            id="a1",
            subject="beagle",
            relation="isa",
            object=EntityRef("dog"),
            metadata={"note": {"by": "x"}},
            provenance=Provenance(source=_source_ref()),
        ),
        "ResolutionRef": _resolution_ref(),
    }


FROZEN = list(_instances())


@pytest.mark.parametrize("name", FROZEN)
def test_every_frozen_type_refuses_assignment(name: str) -> None:
    value = _instances()[name]
    first = next(iter(vars(value)))
    with pytest.raises(FrozenInstanceError):
        setattr(value, first, getattr(value, first))


@pytest.mark.parametrize("name", FROZEN)
def test_every_frozen_type_stays_unhashable(name: str) -> None:
    """Frozen by the route that keeps field-wise equality honest about hashing."""
    value = _instances()[name]
    assert not isinstance(value, Hashable)
    with pytest.raises(TypeError):
        hash(value)


@pytest.mark.parametrize("cls", [EntityType, RelationType])
def test_a_frozen_subclass_declares_its_own_unhashability(cls: type) -> None:
    """A frozen dataclass subclass regenerates ``__hash__`` unless it says not to.

    The hashable sweep would fail too, naming a population; this names the
    cause, which is that each subclass must write ``__hash__ = None`` itself.
    """
    assert "__hash__" in vars(cls)
    assert cls.__hash__ is None


@pytest.mark.parametrize(
    ("name", "reach"),
    [
        ("Entity", lambda v: v.aliases),
        ("EntityType", lambda v: v.attributes),
        ("AttributeDef", lambda v: v.enum_values),
    ],
)
def test_every_sequence_field_is_a_tuple(name: str, reach: Any) -> None:
    held = reach(_instances()[name])
    assert isinstance(held, tuple)
    with pytest.raises(AttributeError):
        held.append("x")


@pytest.mark.parametrize(
    ("name", "reach", "nested"),
    [
        ("Entity", lambda v: v.metadata, "tags"),
        ("Literal", lambda v: v.metadata, "unit"),
        ("Assertion", lambda v: v.metadata, "note"),
        ("SourceRef", lambda v: v.locator, "row"),
        ("ResolutionRef", lambda v: v.corpus, "embedders"),
    ],
)
def test_every_mapping_field_is_frozen_all_the_way_down(name: str, reach: Any, nested: str) -> None:
    held = reach(_instances()[name])
    assert isinstance(held, FrozenMapping)
    with pytest.raises(TypeError):
        held["new"] = 1
    inner = held[nested]
    assert isinstance(inner, (tuple, FrozenMapping))
    with pytest.raises((TypeError, AttributeError)):
        if isinstance(inner, tuple):
            inner.append("x")  # type: ignore[attr-defined]
        else:
            inner["new"] = 1  # type: ignore[index]


def test_a_list_and_a_dict_are_coerced_and_compare_equal() -> None:
    from_lists = Entity(id="e", type="T", aliases=["a"], metadata={"k": 1})  # type: ignore[arg-type]
    from_frozen = Entity(id="e", type="T", aliases=("a",), metadata=FrozenMapping({"k": 1}))

    assert from_lists.aliases == ("a",)
    assert isinstance(from_lists.metadata, FrozenMapping)
    assert from_lists == from_frozen


def test_a_metadata_list_reads_back_as_a_tuple() -> None:
    """The deep freeze's one visible cost, stated in the guide and changelog."""
    entity = Entity(id="e", type="T", metadata={"tags": ["a"]})
    assert entity.metadata["tags"] == ("a",)
    assert entity.metadata["tags"] != ["a"]


@pytest.mark.parametrize("name", FROZEN)
def test_every_frozen_type_survives_pickle_and_deepcopy(name: str) -> None:
    value = _instances()[name]
    assert pickle.loads(pickle.dumps(value)) == value
    assert copy.deepcopy(value) == value


def test_replace_still_derives_a_changed_entity() -> None:
    """``with_variants`` derives entities this way, so it has to keep working."""
    entity = _beagle()
    changed = replace(entity, aliases=("Snoopy",))

    assert changed.aliases == ("Snoopy",)
    assert entity.aliases == ("English Beagle",)


def test_replace_coerces_too() -> None:
    changed = replace(_beagle(), aliases=["Snoopy"])  # type: ignore[arg-type]
    assert changed.aliases == ("Snoopy",)


def test_assertion_still_canonicalises_its_relation() -> None:
    isa = RelationType(id="isa")
    assertion = Assertion(id="a1", subject="a", relation=isa, object=EntityRef("b"))
    assert assertion.relation == "isa"


def test_entity_still_defaults_its_name() -> None:
    assert Entity(id="beagle", type="Breed").name == "beagle"

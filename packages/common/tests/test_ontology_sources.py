"""What a source can be asked about its own contents.

The surface-form cases carry the load here. A vocabulary's aliases are how a
person's words reach the thing they meant, and the two easy mistakes -- folding
case in only one direction, and returning the alias rather than the entity that
declares it -- both produce a lookup that works in the demo and fails on real
input.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.capabilities import Capability
from dataknobs_common.ontology import Entity, SourceRef, load_ontology
from dataknobs_common.ontology.sources import MappingEntitySource


def test_an_alias_finds_the_entity_that_declares_it(mammals_path: Path) -> None:
    """The entity's own id comes back -- never the alias's.

    An alias is a string somebody typed. It is not a thing the vocabulary
    names, so it has no id, and returning one would hand the caller a key that
    resolves to nothing.
    """
    onto = load_ontology(mammals_path)

    assert onto.entities.by_surface_form("Beagles") == frozenset({"beagle"})
    assert onto.entities.get("Beagles") is None


def test_surface_forms_fold_case_in_both_directions(mammals_path: Path) -> None:
    """Whatever case the vocabulary declared, and whatever case was typed."""
    onto = load_ontology(mammals_path)

    assert onto.entities.by_surface_form("beagles") == frozenset({"beagle"})
    assert onto.entities.by_surface_form("BEAGLES") == frozenset({"beagle"})
    assert onto.entities.by_surface_form("  Beagles  ") == frozenset({"beagle"})


def test_a_name_and_an_id_are_surface_forms_too(mammals_path: Path) -> None:
    """All three of the ways a vocabulary names a thing are lookupable."""
    onto = load_ontology(mammals_path)

    assert onto.entities.by_surface_form("Dog") == frozenset({"dog"})
    assert onto.entities.by_surface_form("dog") == frozenset({"dog"})
    assert onto.entities.by_surface_form("Domestic dog") == frozenset({"dog"})


def test_an_unknown_form_returns_an_empty_frozenset(mammals_path: Path) -> None:
    """Absence is an empty vicinity, not None and not an exception."""
    assert load_ontology(mammals_path).entities.by_surface_form("wombat") == frozenset()


def test_two_entities_sharing_an_alias_both_come_back() -> None:
    """Declared ambiguity survives to the consumer.

    The vocabulary says two things answer to this word. Picking one here would
    be inventing a verdict the author did not write, and the caller is the one
    with the context to choose.
    """
    onto = load_ontology(
        {
            "id": "x",
            "entities": [
                {"id": "jaguar_animal", "type": "Species", "aliases": ["Jaguar"]},
                {"id": "jaguar_car", "type": "Marque", "aliases": ["Jaguar"]},
            ],
        }
    )

    assert onto.entities.by_surface_form("jaguar") == frozenset({"jaguar_animal", "jaguar_car"})


def test_a_custom_normalizer_replaces_the_default() -> None:
    """The normalizer belongs to the source, applied when it is bound."""
    onto = load_ontology(
        {"id": "x", "entities": [{"id": "dog", "type": "Species", "aliases": ["Hund"]}]},
        normalizer=lambda form: form.strip().upper(),
    )

    assert onto.entities.by_surface_form("hund") == frozenset({"dog"})
    assert onto.entities.by_surface_form("HUND") == frozenset({"dog"})


def test_by_type_returns_every_entity_of_that_type(mammals_path: Path) -> None:
    onto = load_ontology(mammals_path)

    assert onto.entities.by_type("Species") == frozenset({"mammal", "dog"})
    assert onto.entities.by_type("Breed") == frozenset({"beagle"})
    assert onto.entities.by_type("Wombat") == frozenset()


def test_describe_derives_what_it_declares(mammals_path: Path) -> None:
    """``declares`` is derived from the contents rather than configured.

    Over a mapping that is cheap, and a derived answer cannot drift from what
    the source actually holds the way a declared one can.
    """
    description = load_ontology(mammals_path).entities.describe()

    assert description.declares == frozenset({"Species", "Breed"})


def test_a_source_ref_travels_out_intact_while_the_origin_does_not(
    mammals_path: Path,
) -> None:
    """We cannot fetch the row. The consumer can, and leaves with the key.

    That is what makes an authored vocabulary more than a toy: a small
    hand-maintained set of terms points into a large production table without
    either owning the other.
    """
    onto = load_ontology(mammals_path)
    beagle = onto.entities.get("beagle")
    assert beagle is not None

    assert beagle.source == SourceRef(
        source_id="clinic_db",
        kind="record",
        locator={"table": "species", "key": "sp-2291"},
    )
    assert onto.entities.fetch_origin(beagle.source) is None
    assert onto.entities.fetch_origins([beagle.source]) == {}


def test_describe_says_the_origins_are_unfetchable(mammals_path: Path) -> None:
    """Stated, not left to be inferred from a None.

    A ``fetch_origin`` returning None cannot be told apart from "no such row".
    The capability's absence is the source saying which of the two it means,
    before the caller makes the round trip.
    """
    description = load_ontology(mammals_path).entities.describe()

    assert Capability.ORIGIN_FETCH not in description.capabilities
    assert description.table is None


def test_get_many_omits_the_misses() -> None:
    """A miss is absent rather than mapped to None."""
    source = MappingEntitySource({"dog": Entity(id="dog", type="Species")})

    assert set(source.get_many(["dog", "wombat"])) == {"dog"}


def test_the_ontology_reports_the_sources_it_bound(mammals_path: Path) -> None:
    """The ontology reports its bound sources directly.

    ``describes`` is the caller's route to all of the above without reaching
    through ``entities``.
    """
    onto = load_ontology(mammals_path)

    assert len(onto.describes) == 1
    assert onto.describes[0].declares == frozenset({"Species", "Breed"})


def test_the_concretes_are_recognised_as_their_protocols() -> None:
    """The runtime half of the conformance check.

    ``isinstance`` against a ``@runtime_checkable`` protocol compares method
    names only, so this catches a member that went missing and nothing subtler.
    The signatures are pinned by a type-checked assignment in the source
    module, where the checker can see them -- see
    ``_concretes_satisfy_their_protocols``.
    """
    from dataknobs_common.ontology.sources import (
        AssertionSource,
        AsyncAssertionSource,
        AsyncEntitySource,
        AsyncMappingAssertionSource,
        AsyncMappingEntitySource,
        EntitySource,
        MappingAssertionSource,
    )

    assert isinstance(MappingEntitySource({}), EntitySource)
    assert isinstance(AsyncMappingEntitySource({}), AsyncEntitySource)
    assert isinstance(MappingAssertionSource([]), AssertionSource)
    assert isinstance(AsyncMappingAssertionSource([]), AsyncAssertionSource)

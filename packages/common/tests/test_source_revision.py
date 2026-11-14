"""A source carries a revision: which state of a vocabulary is answering.

A caller that records a link made against a vocabulary -- a mention resolved
to an entity, an index built over its forms -- needs to say which state of the
vocabulary made it, so that a later change can be detected rather than
guessed. ``revision()`` is that answer. Over an authored source it is the
document's version, ``@``, then a digest of the entities the source holds,
variants included, computed when the source is built.

What this file pins:

* the twins agree, and the answer is stable across calls and mapping order;
* it moves on content -- including a ``variants:`` section, which changes no
  declared entity and every alias it admits -- and on version only in the
  prefix;
* both loader doors pass the version, so neither can forget it unnoticed;
* the digest is the construction's, so it describes what the lookups answer;
* the fold is **not** part of it, and that limit is a test rather than an
  accident;
* a value nothing identifies fails the construction, naming the field, and NaN
  does not.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common import fingerprint_value
from dataknobs_common.capabilities import Capability, supports_capability
from dataknobs_common.ontology import (
    AsyncMappingEntitySource,
    Entity,
    MappingEntitySource,
    async_load_ontology,
    load_ontology,
)

if TYPE_CHECKING:
    import pathlib

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _entities() -> dict[str, Entity]:
    return {
        "e-1": Entity(id="e-1", type="product", name="PowerPoint"),
        "e-2": Entity(id="e-2", type="product", name="Beagle", aliases=["Hound"]),
    }


def _document(**extra: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "id": "catalog",
        "entity_types": [{"id": "product"}],
        "entities": [
            {"id": "e-1", "type": "product", "name": "PowerPoint"},
            {"id": "e-2", "type": "product", "name": "Beagle", "aliases": ["Hound"]},
        ],
    }
    document.update(extra)
    return document


# --------------------------------------------------------------------------
# The twins, and stability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", [None, "2.3"])
async def test_the_twins_agree(version: str | None) -> None:
    sync = MappingEntitySource(_entities(), version=version)
    async_ = AsyncMappingEntitySource(_entities(), version=version)
    assert sync.revision() == await async_.revision()


def test_the_answer_is_stable_across_calls_and_mapping_order() -> None:
    entities = _entities()
    reversed_order = dict(reversed(list(entities.items())))
    source = MappingEntitySource(entities)
    assert source.revision() == source.revision()
    assert MappingEntitySource(reversed_order).revision() == source.revision()


# --------------------------------------------------------------------------
# What moves it, and what moves only the prefix
# --------------------------------------------------------------------------


def _changed_entities() -> dict[str, dict[str, Entity]]:
    added = _entities()
    added["e-3"] = Entity(id="e-3", type="product", name="Collie")
    aliased = _entities()
    aliased["e-1"] = Entity(id="e-1", type="product", name="PowerPoint", aliases=["PPT"])
    retyped = _entities()
    retyped["e-2"] = Entity(id="e-2", type="breed", name="Beagle", aliases=["Hound"])
    return {"an entity added": added, "an alias added": aliased, "a type changed": retyped}


@pytest.mark.parametrize("change", sorted(_changed_entities()))
def test_it_moves_on_content(change: str) -> None:
    before = MappingEntitySource(_entities()).revision()
    after = MappingEntitySource(_changed_entities()[change]).revision()
    assert before != after


def test_a_variants_section_moves_it() -> None:
    """The case the member was ruled for: no declared entity changes, every admitted alias does."""
    plain = load_ontology(_document())
    varied = load_ontology(_document(variants=["camelcase"]))
    assert "Power Point" in varied.entity("e-1").aliases
    assert plain.entities.revision() != varied.entities.revision()


def test_a_version_moves_the_prefix_and_not_the_digest() -> None:
    older = load_ontology(_document(version="2.3")).entities.revision()
    newer = load_ontology(_document(version="2.4")).entities.revision()
    assert older is not None
    assert newer is not None
    assert older != newer
    assert older.removeprefix("2.3@") == newer.removeprefix("2.4@")


# --------------------------------------------------------------------------
# The spellings, through every door
# --------------------------------------------------------------------------


def test_a_source_built_in_code_answers_the_bare_digest() -> None:
    revision = MappingEntitySource(_entities()).revision()
    assert _DIGEST.fullmatch(revision)


async def test_both_loader_doors_pass_the_version() -> None:
    """One assertion per door, so a door that forgot the keyword fails on its own."""
    sync = load_ontology(_document(version="2.3")).entities.revision()
    async_ = await (await async_load_ontology(_document(version="2.3"))).entities.revision()
    assert sync is not None
    assert sync.startswith("2.3@")
    assert async_ is not None
    assert async_.startswith("2.3@")
    assert sync == async_


async def test_a_document_without_a_version_answers_under_the_configs_default() -> None:
    """``OntologyConfig`` defaults ``version`` to ``1.0``, and the door cannot tell it was omitted."""
    onto = load_ontology(_document())
    revision = onto.entities.revision()
    assert onto.version == "1.0"
    assert revision is not None
    assert revision.startswith("1.0@")
    async_revision = await (await async_load_ontology(_document())).entities.revision()
    assert async_revision == revision


def test_the_digest_is_the_entities_fingerprint() -> None:
    """One canonical encoding: the source spends ``fingerprint_value``, not a second one."""
    entities = _entities()
    assert MappingEntitySource(entities, version="2.3").revision() == (
        f"2.3@{fingerprint_value(entities, path='entities')}"
    )


# --------------------------------------------------------------------------
# The digest is the construction's
# --------------------------------------------------------------------------


def test_a_change_to_a_handed_out_entity_leaves_the_revision_where_the_lookups_are() -> None:
    """The digest describes the state the lookup maps were built from.

    An entity ``get()`` hands out can still be changed after construction, and
    the lookups do not see it. A digest computed later would describe a state
    no lookup answers from; this one does not move, and neither do the
    lookups. Whether the change should be possible at all is a separate
    question about the model.
    """
    source = MappingEntitySource(_entities())
    before = source.revision()
    entity = source.get("e-1")
    assert entity is not None
    entity.aliases.append("Slides")
    assert source.revision() == before
    assert source.by_surface_form("Slides") == frozenset()


# --------------------------------------------------------------------------
# The fold limit
# --------------------------------------------------------------------------


def test_the_fold_is_not_part_of_the_revision() -> None:
    """Same entities, two folds: one revision, two different answers.

    A normalizer is a callable and is not digested by content, so this limit
    is documented on the member. Pinned here so that lifting it is a decision
    rather than an accident.
    """

    def keep_case(form: str) -> str:
        return form.strip()

    default = MappingEntitySource(_entities())
    case_kept = MappingEntitySource(_entities(), normalizer=keep_case)
    assert default.revision() == case_kept.revision()
    assert default.by_surface_form("beagle") == frozenset({"e-2"})
    assert case_kept.by_surface_form("beagle") == frozenset()


# --------------------------------------------------------------------------
# The capability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_cls", [MappingEntitySource, AsyncMappingEntitySource])
def test_the_capability_is_declared(source_cls: type[Any]) -> None:
    source = source_cls(_entities())
    assert Capability.SOURCE_REVISION in source.describe().capabilities
    assert supports_capability(source, Capability.SOURCE_REVISION)
    assert Capability.SOURCE_REVISION in source_cls.supported_capabilities()


# --------------------------------------------------------------------------
# What the digest refuses, and what it does not
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_cls", [MappingEntitySource, AsyncMappingEntitySource])
def test_a_value_nothing_identifies_fails_the_construction(source_cls: type[Any]) -> None:
    entities = _entities()
    entities["e-1"].metadata["hook"] = lambda form: form
    with pytest.raises(TypeError, match=r"entities\['e-1'\]\.metadata\['hook'\]"):
        source_cls(entities)


def test_nan_hashes_and_is_not_infinity() -> None:
    def with_rating(rating: float) -> dict[str, Entity]:
        entities = _entities()
        entities["e-2"].metadata["rating"] = rating
        return entities

    nan_one = MappingEntitySource(with_rating(math.nan)).revision()
    nan_two = MappingEntitySource(with_rating(float("nan"))).revision()
    infinite = MappingEntitySource(with_rating(math.inf)).revision()
    assert nan_one == nan_two
    assert nan_one != infinite


def test_a_yaml_document_holding_nan_loads_and_answers(tmp_path: pathlib.Path) -> None:
    """``.nan`` is a valid YAML value, so the door must not refuse the document for it."""
    pytest.importorskip("yaml")
    body = (
        "id: catalog\n"
        "entity_types: [{id: product}]\n"
        "entities:\n"
        "  - {id: e-1, type: product, name: Beagle, metadata: {rating: VALUE}}\n"
    )
    revisions: dict[str, str | None] = {}
    for name, value in (("nan-a", ".nan"), ("nan-b", ".nan"), ("inf", ".inf")):
        path = tmp_path / f"{name}.yaml"
        path.write_text(body.replace("VALUE", value), encoding="utf-8")
        revisions[name] = load_ontology(path).entities.revision()
    assert revisions["nan-a"] == revisions["nan-b"]
    assert revisions["nan-a"] != revisions["inf"]


def test_a_config_fingerprint_still_refuses_nan() -> None:
    """The content mode is ``fingerprint_value``'s, and does not leak into the config path."""
    from dataknobs_common.retry import RetryConfig

    assert _DIGEST.fullmatch(fingerprint_value({"rating": math.nan}))
    with pytest.raises(ValueError, match="NaN"):
        RetryConfig(backoff_multiplier=math.nan).fingerprint()

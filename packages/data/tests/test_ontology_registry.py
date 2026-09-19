"""The registry that owns a lifecycle: both doors, both config forms, teardown.

Four of these are the acceptance surface for the configuration contract --
the two doors reaching one object, the portable form round-tripping one level
above the resolved one, the three strictness levels, and the shipped refusal
that had been naming a class nobody could import. The rest are what those four
leave uncovered: a refusal is asserted by nothing else, and neither is the
distinction between closing a registry and unloading from one.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.events import Event, EventType, InMemoryEventBus, event_bus_backends
from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.ontology import (
    AUTHORED_SOURCE_ID,
    AsyncOntology,
    OntologyConfig,
    async_load_ontology,
)
from dataknobs_common.records import Record
from dataknobs_common.testing import (
    assert_no_blocking,
    assert_structured_config_consumer,
    requires_package,
)
from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig
from dataknobs_config.environment_config import ResourceNotFoundError

from dataknobs_data.backends import async_backends
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.factory import async_database_factory
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.vector.stores.memory import MemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

#: §1's app config is the pair in full again. The `taxonomies:` block was
#: lifted out while the registry bound no axis -- a row whose `kind:` nothing
#: read loaded as an assertion axis over no assertions and answered empty, so
#: leaving it in would have made the fixture assert a vocabulary that was not
#: there. The binder for it is what puts it back.
ENVIRONMENTS = {
    # The deployment §1 describes, with the backend a test can actually run.
    # Which backend a logical name reaches is the whole of what `$resource`
    # buys, so supplying a different one here is the mechanism working rather
    # than the test dodging it.
    "production": """
name: production
settings:
  strict_resources: true
resources:
  databases:
    catalog:
      backend: memory
""",
    # The same deployment with the operator's own answer to the same question,
    # so `None` -- which hands the level back to them -- has two answers to be
    # observed changing between.
    "lenient": """
name: lenient
settings:
  strict_resources: false
resources:
  databases:
    catalog:
      backend: memory
""",
}

APP = """
name: catalog
ontology:
  id: catalog
  version: "1.0"
  entity_types:
    - id: Product
  sources:
    - id: products
      kind: record
      database:
        $resource: catalog
        type: databases
      entity_projection:
        table: products
        id: sku
        name: title
        type: {const: Product}
        aliases: {column: alt_names, split: ","}
        surface_forms:
          table: product_forms
          form: folded_form
          entity: sku
      schema:
        - {name: sku, type: string}
        - {name: title, type: string}
        - {name: alt_names, type: string}
        - {name: parent_sku, type: string}
        - {name: folded_form, type: string}
  taxonomies:
    - id: categories
      kind: column
      source: products
      parent_key: parent_sku
      relation: parent
  resolver:
    rungs:
      - kind: exact
"""

#: The app config above with its one `$resource` pointing at a name no
#: environment here defines -- the vehicle for the strictness levels.
APP_WITH_A_MISSING_RESOURCE = APP.replace("$resource: catalog", "$resource: absent")


def _document(**overrides: Any) -> dict[str, Any]:
    """A minimal `kind: record` ontology, for the injected-handle door."""
    document: dict[str, Any] = {
        "id": "t",
        "entity_types": [{"id": "Product"}],
        "sources": [
            {
                "id": "products",
                "kind": "record",
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                },
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                ],
            }
        ],
    }
    document.update(overrides)
    return document


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """§1's two YAML files on disk, with the environment auto-detected."""
    (tmp_path / "config/environments").mkdir(parents=True)
    (tmp_path / "config/apps").mkdir(parents=True)
    for name, body in ENVIRONMENTS.items():
        (tmp_path / f"config/environments/{name}.yaml").write_text(body)
    (tmp_path / "config/apps/catalog.yaml").write_text(APP)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "production")
    yield tmp_path


async def _seeded() -> AsyncMemoryDatabase:
    """One store holding an entity row and the folded forms that reach it."""
    db = AsyncMemoryDatabase()
    await db.create(Record({"sku": "sku-4471", "title": "Beagle"}))
    return db


async def _catalogue() -> AsyncMemoryDatabase:
    """§1's catalogue: three rows, two of them placed by the parent column."""
    db = AsyncMemoryDatabase()
    await db.create(Record({"sku": "dogs", "title": "Dogs"}))
    await db.create(Record({"sku": "hounds", "title": "Hounds", "parent_sku": "dogs"}))
    await db.create(Record({"sku": "sku-4471", "title": "Beagle", "parent_sku": "hounds"}))
    return db


# --------------------------------------------------------------------------
# Both doors reach one object
# --------------------------------------------------------------------------


async def test_both_doors_reach_one_vocabulary(deployment: Path) -> None:
    """The configured door and the handle-with-no-logical-name reach one object.

    A handle with no logical name is only a *door* if what it produces is the
    same kind of thing the configured door produces, so both halves are here:
    two YAML files, an environment auto-detected and a `$resource` resolved on
    one side; no file, no environment and nothing to resolve on the other.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    configured = await OntologyRegistry.from_config_async(cfg.resolve_for_build("ontology"))
    injected = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded()
    )
    await injected.load()
    try:
        from_file = configured.get("catalog")
        from_handle = injected.get("t")
        assert isinstance(from_file, AsyncOntology)
        assert isinstance(from_handle, AsyncOntology)
        assert configured.list_ids() == ["catalog"]
        assert injected.list_ids() == ["t"]
        # The vocabulary each answers with is live in both cases: one source,
        # bound, reporting the table it projects.
        assert from_file.describes[0].table == "products"
        assert from_handle.describes[0].table == "products"
        entity = await from_handle.entity("sku-4471")
        assert entity is not None
        assert entity.name == "Beagle"
    finally:
        await configured.close()
        await injected.close()


# --------------------------------------------------------------------------
# The portable form round-trips, one level above the resolved one
# --------------------------------------------------------------------------


async def test_the_portable_form_round_trips_and_sits_one_level_up(deployment: Path) -> None:
    """What a deployment stores is what a registry reads back, and it is a level up.

    Both clauses, because the second is what makes the first a claim about
    this registry rather than about the config layer: the stored form is the
    *app* document, whose ontology lives under a section key, so a registry
    that did not hold the key could not read back what it stored.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    stored = OntologyRegistry.get_portable_config(cfg)

    assert stored["ontology"]["sources"][0]["database"] == {
        "$resource": "catalog",
        "type": "databases",
    }
    assert "id" not in stored, "the stored form is the app document, not the section"
    with pytest.raises(TypeError, match="id"):
        OntologyConfig.from_dict(stored)

    registry = OntologyRegistry(environment=cfg.environment)
    try:
        loaded = await registry.load(stored)
        assert loaded.id == "catalog"
        assert registry.get("catalog") is loaded
        # ...and the already-unwrapped shape resolves to the same vocabulary,
        # which is what `config.get(key, config)` at the resolution point buys.
        unwrapped = OntologyRegistry(environment=cfg.environment)
        try:
            also = await unwrapped.load(stored["ontology"])
            assert also.id == loaded.id
            assert also.describes[0].projection == loaded.describes[0].projection
        finally:
            await unwrapped.close()
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The three strictness levels, and the default, over two environments
# --------------------------------------------------------------------------


def test_the_registrys_own_default_is_strict() -> None:
    """A claim about the constructor, asserted apart from the chain below it."""
    assert OntologyRegistry()._strict_resources is True


async def test_the_three_strictness_levels_over_two_environments(
    deployment: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """True raises, False resolves to nothing, and None hands the level back.

    Two environments, because over one the third level is unobservable: an
    environment that sets `strict_resources: true` answers `None` exactly as
    it answers `True`, and a test over it alone would pass against a registry
    that ignored the setting entirely. What proves the level was *read* is
    that `None` changes answer between the two.
    """
    (deployment / "config/apps/catalog.yaml").write_text(APP_WITH_A_MISSING_RESOURCE)
    stored = OntologyRegistry.get_portable_config(EnvironmentAwareConfig.load_app("catalog"))
    strict_env = EnvironmentConfig.load("production")
    lenient_env = EnvironmentConfig.load("lenient")

    # True -- raises, and names the four things an operator reading a boot log
    # has nothing else to go on for.
    registry = OntologyRegistry(environment=strict_env, strict_resources=True)
    with pytest.raises(ResourceNotFoundError) as excinfo:
        await registry.load(stored)
    message = str(excinfo.value)
    assert "'absent'" in message
    assert "'databases'" in message
    assert "'production'" in message
    assert "sources" in message

    # False -- degrades, and the warning is the sentence the ruling rests on.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        lenient = OntologyRegistry(environment=strict_env, strict_resources=False)
        try:
            assert (await lenient.load(stored)).id == "catalog"
        finally:
            await lenient.close()
    assert "it declares no inline defaults, so this resolves to an empty config" in caplog.text

    # None -- the environment's own setting decides, so the same document with
    # the same registry keyword answers differently in the two environments.
    deferring_strict = OntologyRegistry(environment=strict_env, strict_resources=None)
    with pytest.raises(ResourceNotFoundError):
        await deferring_strict.load(stored)

    deferring_lenient = OntologyRegistry(environment=lenient_env, strict_resources=None)
    try:
        assert (await deferring_lenient.load(stored)).id == "catalog"
    finally:
        await deferring_lenient.close()


# --------------------------------------------------------------------------
# The refusal stops naming a class that does not exist
# --------------------------------------------------------------------------


async def test_the_shipped_refusal_names_a_class_that_now_loads_the_document() -> None:
    """The remedy in a shipped message is a class a consumer can now import.

    One test holding both halves, because either alone is green against the
    wrong thing: a test asserting only the refusal is what shipped while the
    class existed nowhere, and a test asserting only the load would not notice
    if the message were changed to name something else.

    It fails at the import before this leg, which is the reproduction: a
    consumer following the remedy got an ImportError, and that is the defect.
    """
    document = _document()

    with pytest.raises(ValidationError) as excinfo:
        await async_load_ontology(document)
    assert "OntologyRegistry" in str(excinfo.value)
    assert excinfo.value.context == {"source_id": "products", "kind": "record"}

    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document), database=await _seeded()
    )
    try:
        ontology = await registry.load()
        assert isinstance(ontology, AsyncOntology)
        entity = await ontology.entity("sku-4471")
        assert entity is not None and entity.type == "Product"
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The live source binds, and the vocabulary answers from it
# --------------------------------------------------------------------------


async def test_the_live_source_binds_and_the_vocabulary_answers_from_it(
    deployment: Path,
) -> None:
    """What `$resource` bought, and what the store behind it then answers.

    **Two registries and one document, because the two halves cannot be
    asserted through one door.** What `$resource` buys is *which store*: §1's
    app config names `backend: postgres` nowhere, it names the logical name
    `catalog`, and the environment here answers it with `memory`. That is the
    configured door's half, and it is a claim about the binding rather than
    about any row.

    The other half needs rows, and a `memory` handle the registry builds for
    itself cannot be seeded before it exists -- every `AsyncMemoryDatabase` is
    its own store. So the same document goes through the injected door over a
    store that *is* seeded, and answers `entity()` with the fields the
    projection names: `sku` as the id, `title` as the name. Reaching into the
    registry's handle cache to seed the first one would assert the same thing
    through a private attribute, which is a test of the cache rather than of
    the vocabulary.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    section = cfg.resolve_for_build("ontology")
    configured = await OntologyRegistry.from_config_async(section)
    injected = OntologyRegistry.from_components(
        config=OntologyConfig(**dict(section)), database=await _catalogue()
    )
    await injected.load()
    try:
        bound = configured.get("catalog")
        assert bound is not None
        # The `$resource` resolved to what this environment supplies, which is
        # not what §1's production deployment supplies -- the whole of what a
        # logical name buys. The environment's own word for it, because the
        # configured door reports the block it resolved rather than the class
        # it built.
        assert bound.describes[0].backend == "memory"
        assert bound.describes[0].table == "products"

        answering = injected.get("catalog")
        assert answering is not None
        entity = await answering.entity("sku-4471")
        assert entity is not None
        assert (entity.id, entity.name, entity.type) == ("sku-4471", "Beagle", "Product")
        # ...and the axis block §1 carries is bound rather than loaded empty.
        assert sorted(answering.taxonomies) == ["categories"]
        assert type(answering.taxonomy("categories").structure).__name__ == "ColumnHierarchy"
    finally:
        await configured.close()
        await injected.close()


# --------------------------------------------------------------------------
# What the leg owes beside the criteria
# --------------------------------------------------------------------------


def test_the_structured_config_pattern_is_applied() -> None:
    """The mixin's contracts, with the three ctor params that are not config fields.

    `environment`, `strict_resources`, `config_key` and `normalizer` are
    properties of a *registry*, not of an ontology document, so none of them
    is a field on `OntologyConfig` and none should be. Naming them is the
    guard's own documented channel for saying so. The other direction cannot
    drift: the ctor takes `**kwargs` through the mixin, so every config field
    is accepted by construction.
    """
    assert_structured_config_consumer(
        OntologyRegistry,
        ignore_params=set(OntologyRegistry.CONSTRUCTION_SETTINGS),
    )


def test_the_registrys_own_settings_are_declared_once() -> None:
    """`CONSTRUCTION_SETTINGS` and the ctor's keyword-only parameters are one fact.

    Spelled twice, they drift, and the drift is quiet in the direction that
    matters: a parameter added to the ctor and not to the set is a setting the
    published doors go on swallowing, which is the defect the set exists to
    close. The underscore-prefixed ones are the mixin's channels rather than
    this registry's settings, and are excluded here for that reason.
    """
    declared = {
        name
        for name, parameter in inspect.signature(OntologyRegistry.__init__).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not name.startswith("_")
    }
    assert declared == OntologyRegistry.CONSTRUCTION_SETTINGS


async def test_close_releases_what_it_opened_and_leaves_what_it_was_handed(
    deployment: Path,
) -> None:
    """Owned versus injected, recorded per handle at the moment it is acquired."""
    cfg = EnvironmentAwareConfig.load_app("catalog")
    configured = await OntologyRegistry.from_config_async(cfg.resolve_for_build("ontology"))
    built = [handle for handle, owned in configured._handles if owned]
    assert built, "a configured registry resolved its own handle"
    await configured.close()
    assert all(getattr(handle, "_connected", False) is False for handle in built)

    handed_over = await _seeded()
    injected = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=handed_over
    )
    await injected.load()
    await injected.close()
    # Untouched: still readable, because closing it would tear down a store
    # its owner may still be using.
    assert await handed_over.search(Query(filters=[Filter("sku", Operator.EQ, "sku-4471")]))


async def test_close_does_not_unload_and_unload_reports_what_this_registry_did() -> None:
    """Two acts, and the leg asserts the difference rather than leaving it to be found."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded()
    )
    await registry.load()
    await registry.close()
    assert registry.list_ids() == ["t"], "close() releases handles; it does not unload"

    assert await registry.unload("t") is True
    assert registry.list_ids() == []
    assert await registry.unload("t") is False, "it makes no claim beyond this registry"


async def test_the_three_events_are_a_topic_and_a_type() -> None:
    """Load, depart and rebuild announce themselves; nothing is added to EventType.

    The rebuild is asserted through the public replacement rather than by
    editing what the registry stored, because a replacement *is* what a rebuild
    is: ``reload`` is ``load(stored, replace=True)`` and nothing else.

    A rebuild is **two** announcements, at the two levels a delta can be over:
    the ontology's whole declared population, and each axis's own nodes. The
    entity below is placed on the axis by an assertion, which is what puts it
    in both.
    """
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus = InMemoryEventBus()
    await bus.connect()
    await bus.subscribe("ontology:t", record)
    await bus.subscribe("taxonomy:kinds", record)

    document = _document(
        entities=[
            {"id": "beagle", "type": "Breed", "name": "Beagle"},
            {"id": "dog", "type": "Breed", "name": "Dog"},
        ],
        entity_types=[{"id": "Breed"}],
        assertions=[{"subject": "beagle", "relation": "isa", "object": "dog"}],
        sources=[],
        taxonomies=[{"id": "kinds", "relation": "isa"}],
    )
    registry = OntologyRegistry.from_components(config=OntologyConfig(**document), event_bus=bus)
    try:
        await registry.load()
        assert [(e.topic, e.type) for e in seen] == [("ontology:t", EventType.CREATED)]

        seen.clear()
        renamed = dict(document)
        renamed["entities"] = [
            {"id": "beagle", "type": "Breed", "name": "Beagle Hound"},
            {"id": "dog", "type": "Breed", "name": "Dog"},
        ]
        await registry.load(renamed, replace=True)
        assert [(e.topic, e.type) for e in seen] == [
            ("ontology:t", EventType.DELETED),
            ("ontology:t", EventType.CREATED),
            ("ontology:t", EventType.UPDATED),
            ("taxonomy:kinds", EventType.UPDATED),
        ]
        # The third set is the point: a rename keeps its id, so a two-set
        # delta would report it as no change at all.
        for rebuilt in (seen[-2].payload, seen[-1].payload):
            assert rebuilt["gone"] == [] and rebuilt["arrived"] == []
            assert rebuilt["renamed"] == ["beagle"]
        assert seen[-1].payload["taxonomy_id"] == "kinds"
    finally:
        await registry.close()
        await bus.close()


async def test_reload_re_resolves_the_document_this_registry_stored(
    deployment: Path,
) -> None:
    """The one thing holding the vocabulary does not buy: today's environment.

    An environment whose ``$resource`` now names a different backend reaches
    the sources only through a resolution, and that is what this re-runs --
    over the document as stored, so nothing about the vocabulary is re-entered
    by hand.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = OntologyRegistry(environment=cfg.environment)
    try:
        first = await registry.load(OntologyRegistry.get_portable_config(cfg))
        again = await registry.reload("catalog")
        assert again is not first
        assert registry.list_ids() == ["catalog"]
        assert again.describes[0].projection == first.describes[0].projection
        with pytest.raises(KeyError):
            await registry.reload("never-loaded")
    finally:
        await registry.close()


async def test_a_configured_event_bus_is_built_and_closed_by_the_registry() -> None:
    """It built it, so it closes it -- and an injected one always wins untouched."""
    document = _document(event_bus={"backend": "memory"})
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**{k: v for k, v in document.items() if k != "event_bus"}),
        database=await _seeded(),
    )
    try:
        await registry.load(document)
        assert registry._event_bus is not None
        assert registry._owns_event_bus is True
    finally:
        await registry.close()

    handed_over = InMemoryEventBus()
    await handed_over.connect()
    injected = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded(), event_bus=handed_over
    )
    try:
        await injected.load(document)
        assert injected._event_bus is handed_over
        assert injected._owns_event_bus is False
    finally:
        await injected.close()
        # Left open for its owner, who is the one that connected it.
        await handed_over.publish("ontology:t", Event(type=EventType.CREATED, topic="ontology:t"))
        await handed_over.close()


async def test_the_unload_payload_carries_ids_for_an_authored_vocabulary() -> None:
    """The id set where every source is authored -- the population is in memory."""
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus = InMemoryEventBus()
    await bus.connect()
    await bus.subscribe("ontology:t", record)
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(
            id="t",
            entity_types=[{"id": "Breed"}],
            entities=[{"id": "beagle", "type": "Breed"}],
        ),
        event_bus=bus,
    )
    try:
        ontology = await registry.load()
        assert ontology.describes[0].backend == AUTHORED_SOURCE_ID
        seen.clear()
        await registry.unload("t")
        assert seen[-1].payload["entity_ids"] == ["t:beagle"]
    finally:
        await registry.close()
        await bus.close()


async def test_the_unload_payload_carries_a_prefix_once_a_source_is_live() -> None:
    """The prefix as soon as one source is live, so no scan happens on the way out."""
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus = InMemoryEventBus()
    await bus.connect()
    await bus.subscribe("ontology:t", record)
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded(), event_bus=bus
    )
    try:
        await registry.load()
        seen.clear()
        await registry.unload("t")
        assert seen[-1].payload == {"ontology_id": "t", "entity_id_prefix": "t:"}
    finally:
        await registry.close()
        await bus.close()


async def test_a_duplicate_id_is_refused_and_replace_is_how_it_is_taken() -> None:
    """The registry instance is the unit of sharing, so its ids are one namespace."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded()
    )
    try:
        await registry.load()
        with pytest.raises(ValidationError, match="already loaded"):
            await registry.load()
        assert (await registry.load(replace=True)).id == "t"
    finally:
        await registry.close()


async def test_a_live_kind_no_binder_is_registered_for_is_refused_by_name() -> None:
    """Computed from the declared kind alone, so imports cannot change the answer."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(id="t", sources=[{"id": "vectors", "kind": "vector_store"}]),
        database=await _seeded(),
    )
    try:
        with pytest.raises(ValidationError) as excinfo:
            await registry.load()
        assert "'vector_store'" in str(excinfo.value)
        assert "'record'" in str(excinfo.value)
    finally:
        await registry.close()


async def test_a_document_mixing_an_authored_vocabulary_with_a_live_one_is_refused() -> None:
    """Naming the construct that would route between them, rather than picking a winner."""
    document = _document(entities=[{"id": "beagle", "type": "Product"}])
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document), database=await _seeded()
    )
    try:
        with pytest.raises(ValidationError, match="LayeredEntitySource"):
            await registry.load()
    finally:
        await registry.close()


async def test_two_live_sources_are_refused_for_the_same_reason() -> None:
    """One vocabulary has one entity source until the router that layers them exists."""
    document = _document()
    second = dict(document["sources"][0])
    second["id"] = "also_products"
    document["sources"] = [document["sources"][0], second]
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document), database=await _seeded()
    )
    try:
        with pytest.raises(ValidationError, match="LayeredEntitySource"):
            await registry.load()
    finally:
        await registry.close()


async def test_the_index_and_the_resolver_answer_none_rather_than_raising() -> None:
    """Absence is a configuration answer, not an error -- from either side of it.

    **This still pins the absent-section branch and must.** A document
    declaring no ``resolver:`` answers ``None`` for good, however many
    cascades this registry can build --- silence is *use your own
    composition*, and inventing one would take that choice away. The positive
    twin is below; after the reader shipped, this test alone covered half the
    member.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded()
    )
    try:
        await registry.load()
        assert registry.index("t") is None
        assert registry.resolver("t") is None
    finally:
        await registry.close()


async def test_a_resolver_section_builds_the_cascade_the_document_wrote() -> None:
    """The positive twin, and the composition is asserted rather than the type.

    A member that answered *some* cascade for any document would pass an
    assertion that one came back --- so what is checked is that the rungs are
    the ones written, in the order written. The composition **is** the policy,
    which is why a document is allowed to write it at all.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(
            **_authored(resolver={"rungs": [{"kind": "alias"}, {"kind": "exact"}]})
        )
    )
    try:
        await registry.load()
        resolver = registry.resolver("t")
        assert resolver is not None
        assert [rung.name for rung in resolver.rungs] == ["alias", "exact"]

        assert [c.entity_id for c in (await resolver.resolve("Canine", k=5)).candidates] == ["dog"]
        named = await resolver.resolve("The Hound", k=5)
        assert [evidence.signal for evidence in named.explain("dog")] == ["exact"]
    finally:
        await registry.close()


async def test_an_explicitly_empty_composition_is_not_silence() -> None:
    """``rungs: []`` is a composition somebody chose, and it misses everything.

    The distinction the door one layer down draws and this reader must not
    collapse: an absent section gets ``None`` from the member above, and an
    empty list gets a cascade holding no rungs. A consumer who wants to
    resolve nothing must be able to say so, and the two answers are how they
    say it.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_authored(resolver={"rungs": []}))
    )
    try:
        await registry.load()
        resolver = registry.resolver("t")
        assert resolver is not None, "an empty composition is still a composition"
        assert list(resolver.rungs) == []
        assert list((await resolver.resolve("The Hound", k=5)).candidates) == []
    finally:
        await registry.close()


async def test_a_resolver_section_with_a_key_this_block_does_not_read_is_refused() -> None:
    """A key nothing acts on is worse than inert here, which is why it is refused.

    ``rungs`` absent is read one layer down as *a composition of nothing*, so
    a section spelling it any other way builds a cascade that matches
    everything's nothing and reports success --- a document whose author
    believes they configured three rungs and whose registry resolves to
    ``UNRESOLVED`` for every query, with nothing anywhere saying why.

    The refusal names the offending key rather than only the section, because
    the author's next question is *which one*.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_authored(resolver={"rung": [{"kind": "exact"}]}))
    )
    try:
        with pytest.raises(ValidationError, match="rung") as refused:
            await registry.load()
        assert "this block does not read" in str(refused.value)
    finally:
        await registry.close()


async def test_a_rung_needing_a_handle_this_document_did_not_declare_is_named() -> None:
    """A semantic rung over a document with no ``index:`` section.

    The handles a rung is built over are the registry's to supply and the
    index is one of them, so a composition naming that rung over a document
    that declared no index is a document missing a section --- not a registry
    missing a handle, and the message has to say which. Without this the
    failure is a ``KeyError`` out of a factory the author never named.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_authored(resolver={"rungs": [{"kind": "semantic"}]}))
    )
    try:
        with pytest.raises(ValidationError, match="semantic") as refused:
            await registry.load()
        assert "index" in str(refused.value)
    finally:
        await registry.close()


async def test_a_rung_kind_nothing_registers_is_refused_as_a_document_fault() -> None:
    """A misspelled ``kind:`` is a document to fix, so it is refused as one.

    It reaches this reader as an unknown key from a registry two packages
    down, whose own answer is the right one for a caller who asked it
    directly. Through a *document* it is a typo, and a caller catching what
    this door documents itself as raising caught it under neither name.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_authored(resolver={"rungs": [{"kind": "sematnic"}]}))
    )
    try:
        with pytest.raises(ValidationError, match="sematnic") as refused:
            await registry.load()
        assert refused.value.context.get("ontology_id") == "t"
    finally:
        await registry.close()


async def test_resolve_ref_answers_the_built_in_table_and_a_loaded_namespace() -> None:
    """A qualified id is parsed the same way whichever namespace it lands in."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded()
    )
    try:
        await registry.load()
        entity = await registry.resolve_ref("t:sku-4471")
        assert entity is not None and entity.name == "Beagle"
        builtin = await registry.resolve_ref("dk:EntityType")
        assert builtin is not None and builtin.name == "EntityType"
        assert await registry.resolve_ref("nowhere:x") is None
        assert await registry.resolve_ref("unqualified") is None
    finally:
        await registry.close()


async def test_a_configured_load_does_no_blocking_io_on_the_loop(
    deployment: Path,
) -> None:
    """Two things here block, and both are offloaded rather than reviewed.

    A registry constructed with an environment *name* has to read that
    environment's YAML, and opening a backend imports the module that
    implements it and may create a directory or a file. Both are disk I/O on
    whatever loop the caller is running -- invisible in a single-request test
    and catastrophic under concurrency.

    The second is the one review does not find: the backend registry resolves
    a name by importing, so the stall happens on the *first* load in a process
    and never again. This assertion fails against either offload removed, and
    that was checked in both directions rather than assumed.
    """
    stored = OntologyRegistry.get_portable_config(EnvironmentAwareConfig.load_app("catalog"))
    registry = OntologyRegistry(environment="production")
    try:
        with assert_no_blocking():
            assert (await registry.load(stored)).id == "catalog"
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# What a published door can carry
# --------------------------------------------------------------------------


#: An authored vocabulary with a copied structure axis, and nothing live.
#:
#: Authored so that every door below can load it with no handle, no
#: environment and no file; the axis is materialized because ``structures`` is
#: the assembly keyword most likely to be left out of a door that writes the
#: assembly itself.
AUTHORED = {
    "id": "t",
    "version": "1.1",
    "imports": ["other"],
    "entity_types": [{"id": "Species"}],
    "entities": [
        {"id": "mammal", "type": "Species", "name": "Mammal"},
        {"id": "dog", "type": "Species", "name": "The Hound", "aliases": ["Canine"]},
    ],
    "assertions": [{"subject": "dog", "relation": "isa", "object": "mammal"}],
    "taxonomies": [
        {
            "id": "species",
            "name": "Species",
            "relation": "isa",
            "materialization": {"structure": "materialized"},
        }
    ],
}


def _authored(**overrides: Any) -> dict[str, Any]:
    """:data:`AUTHORED` with keys added or replaced."""
    return {**AUTHORED, **overrides}


async def _empty(door: str, **settings: Any) -> OntologyRegistry:
    """A registry holding no document, built through the named door.

    The settings travel as loose keywords in every case, which is the point:
    three of these four doors have the shape ``(config, **components)`` and
    put everything that is not the config into the component channel, so a
    setting only arrives if the constructor takes it back out of there.
    """
    if door == "construct":
        return OntologyRegistry(**settings)
    if door == "from_config":
        return OntologyRegistry.from_config(OntologyConfig(id=""), **settings)
    if door == "from_config_async":
        return await OntologyRegistry.from_config_async(OntologyConfig(id=""), **settings)
    return OntologyRegistry.from_components(config=OntologyConfig(id=""), **settings)


@pytest.mark.parametrize(
    "door", ["construct", "from_config", "from_config_async", "from_components"]
)
async def test_every_door_carries_the_environment_this_registry_resolves_against(
    deployment: Path, door: str
) -> None:
    """`environment=` reaches the registry through whichever door it was written on.

    `construct` is the control: it is the door whose signature names the
    setting, and it passed before the other three did. The other three name
    nothing but `config` and `**components`, so an environment written on one
    of them landed on `self.components`, where the registry never looked --
    and the load below then reached `async_database_factory` holding an
    unresolved `$resource` block.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    stored = OntologyRegistry.get_portable_config(cfg)
    registry = await _empty(door, environment=cfg.environment)
    try:
        assert (await registry.load(stored)).id == "catalog"
    finally:
        await registry.close()


async def test_a_door_carries_the_strictness_level(
    deployment: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`strict_resources=False` written on a door degrades where the default raises.

    The pair, because the lenient answer alone is also what a registry that
    ignored the keyword would give if the default were lenient -- and it is
    not. What proves the level was read through the door is that two
    registries built the same way, differing in that one keyword, answer
    differently over one document.
    """
    (deployment / "config/apps/catalog.yaml").write_text(APP_WITH_A_MISSING_RESOURCE)
    environment = EnvironmentConfig.load("production")
    stored = OntologyRegistry.get_portable_config(EnvironmentAwareConfig.load_app("catalog"))

    strict = await _empty("from_config_async", environment=environment)
    try:
        with pytest.raises(ResourceNotFoundError):
            await strict.load(stored)
    finally:
        await strict.close()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        lenient = await _empty("from_config_async", environment=environment, strict_resources=False)
        try:
            assert (await lenient.load(stored)).id == "catalog"
        finally:
            await lenient.close()
    assert "it declares no inline defaults, so this resolves to an empty config" in caplog.text


async def test_a_door_carries_the_section_key() -> None:
    """`config_key=` written on a door reaches the read that uses it."""
    registry = await _empty("from_config_async", config_key="vocab")
    try:
        assert (await registry.load({"vocab": _authored()})).id == "t"
    finally:
        await registry.close()


def _drop_the(text: str) -> str:
    """A fold that differs from the default in one observable way."""
    return text.strip().casefold().removeprefix("the ")


async def test_the_registrys_normalizer_reaches_the_authored_source() -> None:
    """A fold handed to the registry is the fold the vocabulary it builds uses.

    The module-level door has taken `normalizer=` since it shipped; the
    registry loading the same document could not be given one, so the same
    vocabulary folded differently depending on which door loaded it.
    """
    registry = await _empty("from_config_async", normalizer=_drop_the)
    try:
        ontology = await registry.load(_authored())
        assert await ontology.by_surface_form("hound") == frozenset({"dog"})
    finally:
        await registry.close()

    default = await _empty("from_config_async")
    try:
        ontology = await default.load(_authored())
        # `The Hound` folds to `the hound` under the default, so the form a
        # consumer folded their own way does not reach it.
        assert await ontology.by_surface_form("hound") == frozenset()
    finally:
        await default.close()


async def test_the_registrys_normalizer_reaches_a_record_source() -> None:
    """The same fold, on the live half, over a lookup table folded with it.

    This is the half the parameter was documented for: a consumer who folded
    their `surface_forms:` table with their own callable needs the source
    reading it to fold the query the same way, and the registry is the door
    that builds that source.
    """
    document = _document(
        sources=[
            {
                "id": "products",
                "kind": "record",
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                    "surface_forms": {
                        "table": "product_forms",
                        "form": "folded_form",
                        "entity": "sku",
                    },
                },
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                    {"name": "folded_form", "type": "string"},
                ],
            }
        ]
    )
    store = AsyncMemoryDatabase()
    await store.create(Record({"sku": "sku-4471", "title": "The Beagle"}))
    await store.create(Record({"folded_form": "beagle", "sku": "sku-4471"}))

    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document), database=store, normalizer=_drop_the
    )
    try:
        ontology = await registry.load()
        assert await ontology.by_surface_form("The Beagle") == frozenset({"sku-4471"})
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# One assembly, three doors
# --------------------------------------------------------------------------


async def test_the_registry_and_the_module_door_assemble_one_vocabulary() -> None:
    """Every field that is not a bound source agrees across the two doors.

    The guard on the extraction: both doors call one assembler, so a keyword
    added to `AsyncOntology` and threaded at one of them cannot pass here. It
    is asserted field by field rather than by equality because the two
    vocabularies hold *different* source objects by construction -- which is
    the one difference between them, and the reason the rest must match.
    """
    document = _authored()
    from_module = await async_load_ontology(document)
    registry = await OntologyRegistry.from_config_async(document)
    try:
        from_registry = registry.get("t")
        assert from_registry is not None
        for field in (
            "id",
            "version",
            "entity_types",
            "relation_types",
            "taxonomies",
            "imports",
        ):
            assert getattr(from_registry, field) == getattr(from_module, field), field
        assert type(from_registry.codec) is type(from_module.codec)
        # The copied axis, by what it answers: the snapshot is a value with
        # no equality of its own, and what matters is that the registry took
        # one at all and took it over the same edges.
        assert set(from_registry.structures) == {"species"}
        copied, published = from_registry.structures["species"], from_module.structures["species"]
        assert type(copied) is type(published)
        assert await copied.roots() == await published.roots()
        assert await copied.parents("dog") == await published.parents("dog") == ("mammal",)
        assert [d.source_id for d in from_registry.describes] == [
            d.source_id for d in from_module.describes
        ]
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The bus a document configures
# --------------------------------------------------------------------------


@pytest.fixture
def probe_bus() -> Iterator[InMemoryEventBus]:
    """A bus a document can name, which the test holds a reference to.

    Registered rather than injected, because what is under test is the path a
    *configured* bus takes -- an injected one wins over it and would hide the
    question.
    """
    bus = InMemoryEventBus()
    event_bus_backends.register("probe", lambda config: bus)
    try:
        yield bus
    finally:
        event_bus_backends.unregister("probe")


async def test_the_configured_door_announces_on_the_bus_the_document_declares(
    probe_bus: InMemoryEventBus,
) -> None:
    """`from_config_async` builds the bus its document configures, and publishes to it.

    The door the guide leads with, and the one that could not configure a bus
    at all: the block was read off the raw mapping, and every published door
    coerces to `OntologyConfig` before the constructor sees anything, so what
    the constructor read was always `None`. A bus configured this way is also
    this registry's to close.
    """
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    await probe_bus.subscribe("ontology:t", record)
    registry = await OntologyRegistry.from_config_async(_authored(event_bus={"backend": "probe"}))
    try:
        assert [event.type for event in seen] == [EventType.CREATED]
        assert registry._event_bus is probe_bus
        assert registry._owns_event_bus is True
    finally:
        await registry.close()


async def test_a_configured_bus_is_built_off_the_event_loop(tmp_path: Path) -> None:
    """Building the bus does no blocking I/O on the caller's loop.

    A backend factory reads a file here, standing in for what every built-in
    one does: import its driver -- `asyncpg`, `redis`, `aioboto3` -- inside
    the factory call, so that a base install pulls none of them. That import
    is disk I/O, it happens on the first bus of its backend in a process and
    never again, and it was running on whatever loop the caller had.
    """
    probe = tmp_path / "driver.txt"
    probe.write_text("a stand-in for the driver an event bus backend imports\n")

    def _factory(config: dict[str, Any]) -> InMemoryEventBus:
        Path(config["driver"]).read_text(encoding="utf-8")
        return InMemoryEventBus()

    event_bus_backends.register("slow-probe", _factory)
    registry = None
    try:
        with assert_no_blocking():
            registry = await OntologyRegistry.from_config_async(
                _authored(event_bus={"backend": "slow-probe", "driver": str(probe)})
            )
    finally:
        if registry is not None:
            await registry.close()
        event_bus_backends.unregister("slow-probe")


# --------------------------------------------------------------------------
# What teardown leaves behind
# --------------------------------------------------------------------------


class _CountingBus(InMemoryEventBus):
    """A real in-memory bus that records how many times it was closed.

    A real bus rather than a stand-in, because what is under test is the
    registry's bookkeeping and not the bus's: ``InMemoryEventBus.close`` is
    itself idempotent, so the second close is invisible from the outside
    unless something counts it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1
        await super().close()


class _UnconnectableBus(InMemoryEventBus):
    """A bus whose ``connect()`` fails, as a backend with a bad address would."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def connect(self) -> None:
        raise OperationError("the probe bus cannot reach its broker")

    async def close(self) -> None:
        self.closed = True
        await super().close()


async def test_close_forgets_the_bus_it_built_and_keeps_the_one_it_was_handed() -> None:
    """A second ``close()`` closes an owned bus once, and an injected one never.

    ``close()`` empties ``_handles``, so the handle half is idempotent; the bus
    half was not, because the two fields recording it survived the close. A
    registry closed twice therefore closed an owned bus twice, and an
    ``unload()`` after a close published a departure to a bus this registry had
    already torn down.

    The injected half is the other direction, and it is why the reset is
    conditional: a bus this registry did not close is still live, so the
    departure event a post-close ``unload()`` announces has somewhere to go.
    """
    built = _CountingBus()
    event_bus_backends.register("counting-probe", lambda config: built)
    try:
        registry = await OntologyRegistry.from_config_async(
            _authored(event_bus={"backend": "counting-probe"})
        )
        await registry.close()
        await registry.close()
        assert built.closes == 1
        assert registry._event_bus is None
        assert registry._owns_event_bus is False
    finally:
        event_bus_backends.unregister("counting-probe")

    handed_over = _CountingBus()
    await handed_over.connect()
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    await handed_over.subscribe("ontology:t", record)
    injected = OntologyRegistry.from_components(
        config=OntologyConfig(**_authored()), event_bus=handed_over
    )
    await injected.load()
    await injected.close()
    assert handed_over.closes == 0, "it was handed over, so it is not this registry's to close"
    assert injected._event_bus is handed_over
    seen.clear()
    assert await injected.unload("t") is True
    assert [event.type for event in seen] == [EventType.DELETED]
    await handed_over.close()


async def test_a_bus_whose_connect_fails_is_closed_rather_than_leaked() -> None:
    """A backend that fails mid-connect never reaches ``close()``, so it is closed here.

    Ownership was recorded *after* the connect, so a bus that raised was
    discarded still holding whatever its constructor acquired -- and the
    registry, having recorded nothing, had no way to release it. The load
    still fails; what changes is that nothing is left behind, and that a
    retry builds a fresh bus rather than finding a broken one wired in.
    """
    unconnectable = _UnconnectableBus()
    event_bus_backends.register("unconnectable-probe", lambda config: unconnectable)
    try:
        registry = OntologyRegistry.from_components(config=OntologyConfig(**_authored()))
        with pytest.raises(OperationError, match="cannot reach its broker"):
            await registry.load(_authored(event_bus={"backend": "unconnectable-probe"}))
        assert unconnectable.closed is True
        assert registry._event_bus is None
        assert registry._owns_event_bus is False
        await registry.close()
    finally:
        event_bus_backends.unregister("unconnectable-probe")


class _UninitialisableStore(MemoryVectorStore):
    """A store whose ``initialize()`` fails, as a backend with a bad path would."""

    def __init__(self) -> None:
        super().__init__({"dimensions": 4})
        self.closed = False

    async def initialize(self) -> None:
        raise OperationError("the probe store cannot open its index")

    async def close(self) -> None:
        self.closed = True
        await super().close()


async def test_the_open_reaches_a_store_that_spells_the_step_initialize() -> None:
    """The second resource family opens under a different name, and the helper probes.

    ``_connect_or_close`` called ``resource.connect()`` outright, which is the
    only spelling the two collaborators it was written for have. A
    ``VectorStore`` -- the third collaborator, and the one an ``index:`` block
    names -- carries ``initialize()`` among its nine abstract members and no
    ``connect`` at all, so the helper raised ``AttributeError`` on the first
    store it was handed.

    The generalisation the helper was written under is still right: what it
    exists for is the window between *built* and *recorded*, and a
    ``FaissVectorStore`` or a ``ChromaVectorStore`` acquires an index or a
    client in its constructor exactly as a database does. It generalised over
    the hazard and not over the spelling, and no second spelling existed when
    it was written.

    **The probe never has to choose.** Of the 536 distinct public classes the
    falsifier below reaches across the two packages --- 1,753 name bindings,
    since a re-export binds one class twice --- none resolves both members.
    """
    registry = OntologyRegistry.from_components(config=OntologyConfig(**_authored()))
    store = MemoryVectorStore({"dimensions": 4})
    assert not hasattr(store, "connect")

    await registry._connect_or_close(store)

    assert store._initialized is True
    await store.close()
    await registry.close()


async def test_a_store_whose_initialize_fails_is_closed_rather_than_leaked() -> None:
    """The bus's case one collaborator over, and it is why the probe is in this helper.

    A second helper beside ``_connect_or_close`` would be a second spelling of
    a remedy whose whole argument is that two spellings is how one of them
    gets fixed alone. So the store inherits the close-on-failure the bus and
    the handle already have, rather than being given its own copy of it.
    """
    registry = OntologyRegistry.from_components(config=OntologyConfig(**_authored()))
    unopenable = _UninitialisableStore()

    with pytest.raises(OperationError, match="cannot open its index"):
        await registry._connect_or_close(unopenable)

    assert unopenable.closed is True
    await registry.close()


def test_no_collaborator_carries_both_spellings_of_the_open() -> None:
    """The probe's falsifier, run rather than quoted.

    ``getattr`` sees inherited members, so the question is properly asked of
    *resolved* attributes over every public class rather than of definition
    sites. A collaborator carrying both would be one the probe has to choose
    for, and choosing is a ruling this one does not make.
    """
    import importlib
    import pkgutil

    both: list[str] = []
    seen: set[str] = set()
    unreadable: dict[str, str] = {}
    for package in ("dataknobs_common", "dataknobs_data"):
        root = importlib.import_module(package)
        for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
            try:
                module = importlib.import_module(info.name)
            except Exception as exc:
                unreadable[info.name] = f"{type(exc).__name__}: {exc}"
                continue
            for name, obj in vars(module).items():
                if name.startswith("_") or not isinstance(obj, type):
                    continue
                seen.add(f"{obj.__module__}.{obj.__qualname__}")
                if hasattr(obj, "connect") and hasattr(obj, "initialize"):
                    both.append(f"{info.name}.{name}")

    # A module that will not import is a **hole in the sweep**, not a module
    # with nothing to say. This loop used to write `except Exception: continue`,
    # so the classes most likely to grow a `connect` -- the optional-backend
    # stores, which are the ones an environment can fail to import -- were
    # exactly the ones it could skip while still reporting green. Every module
    # imports today because each optional backend guards its own import, so
    # there is nothing to allow-list and an empty set is the measurement.
    assert unreadable == {}, f"the sweep could not read: {unreadable}"

    # Non-vacuous: a sweep that walked nothing would satisfy the assertion
    # below by finding nothing, which is the other way a guard reports green
    # over an unasked question.
    assert len(seen) > 400, f"the sweep resolved only {len(seen)} public classes"

    assert both == [], f"the probe would have to choose for: {sorted(set(both))}"


# --------------------------------------------------------------------------
# One handle per block, decided on the loop
# --------------------------------------------------------------------------


#: How many times the probe backend below was constructed. Module level
#: because a registered backend is a class and the registry constructs it,
#: so there is nowhere in the call to hand a counter.
_OPENED: list[str] = []


class _SlowMemoryDatabase(AsyncMemoryDatabase):
    """A real memory store that takes long enough to build to overlap with itself.

    The sleep is the whole point: without it two concurrent loads may or may
    not be inside the factory at once, and a race asserted by timing is a
    test that passes for the wrong reason roughly as often as it fails.
    """

    def _setup(self) -> None:
        super()._setup()
        _OPENED.append("built")
        time.sleep(0.05)


def _probe_document(ontology_id: str) -> dict[str, Any]:
    """A live binding over the probe backend -- no environment, no file."""
    return {
        "id": ontology_id,
        "entity_types": [{"id": "Product"}],
        "sources": [
            {
                "id": "products",
                "kind": "record",
                "database": {"backend": "probeslow"},
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                },
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                ],
            }
        ],
    }


async def test_two_concurrent_loads_over_one_block_open_one_handle() -> None:
    """The cache is checked on the loop, so two loads cannot both miss it.

    The check-then-set ran inside the worker thread the open was offloaded
    to, so two concurrent loads naming one resolved block each found the
    cache empty and each opened a handle -- two connections where the
    docstring promises one, and two entries in ``_handles`` for one store.

    The document is loaded twice under two ids rather than once, because a
    second load of the *same* id is refused before it reaches a handle.
    """
    _OPENED.clear()
    async_backends.register("probeslow", _SlowMemoryDatabase)
    registry = OntologyRegistry()
    try:
        await asyncio.gather(
            registry.load(_probe_document("first")),
            registry.load(_probe_document("second")),
        )
        assert _OPENED == ["built"], "one resolved block is one handle"
        assert [owned for _, owned in registry._handles] == [True]
    finally:
        await registry.close()
        async_backends.unregister("probeslow")


@requires_package("aiosqlite")
async def test_a_binding_naming_two_tables_opens_a_handle_for_each(tmp_path: Path) -> None:
    """The arrangement the whole two-handle path exists for, over a backend that has tables.

    ``memory``, ``file``, ``s3`` and ``elasticsearch`` declare no ``table`` on
    their config, so every test above binds one handle however many tables the
    projection names -- which left the branch that opens a second one, and the
    ``connect()`` a SQL backend needs before its first read, covered by
    nothing.

    Both halves are asserted: two handles, and a read that crosses them.
    """
    path = str(tmp_path / "catalog.db")
    entities = async_database_factory.create(backend="sqlite", path=path, table="products")
    forms = async_database_factory.create(backend="sqlite", path=path, table="product_forms")
    await entities.connect()
    await forms.connect()
    try:
        await entities.create(Record({"sku": "sku-4471", "title": "Beagle"}))
        await forms.create(Record({"folded_form": "beagle", "sku": "sku-4471"}))
    finally:
        await entities.close()
        await forms.close()

    document = _document(
        sources=[
            {
                "id": "products",
                "kind": "record",
                "database": {"backend": "sqlite", "path": path},
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                    "surface_forms": {
                        "table": "product_forms",
                        "form": "folded_form",
                        "entity": "sku",
                    },
                },
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                    {"name": "folded_form", "type": "string"},
                ],
            }
        ]
    )
    registry = OntologyRegistry()
    try:
        ontology = await registry.load(document)
        assert len(registry._handles) == 2, "a backend whose config names a table is one table's"
        assert all(owned for _, owned in registry._handles)
        entity = await ontology.entity("sku-4471")
        assert entity is not None and entity.name == "Beagle"
        assert await ontology.by_surface_form("Beagle") == frozenset({"sku-4471"})
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# What a refusal names
# --------------------------------------------------------------------------


async def test_reload_names_the_condition_rather_than_reporting_it_unloaded() -> None:
    """A vocabulary the constructor loaded is loaded; it just cannot be re-resolved.

    ``reload`` reads the **portable** document this registry stored, and only
    :meth:`load` stores one. The configured door is handed the already-resolved
    form, so there is nothing for a re-resolution to do -- and the refusal said
    ``No ontology loaded with id 't'`` about an id ``list_ids()`` reports.
    """
    registry = await OntologyRegistry.from_config_async(_authored())
    try:
        assert registry.list_ids() == ["t"]
        with pytest.raises(KeyError, match="already-resolved"):
            await registry.reload("t")
        with pytest.raises(KeyError, match="No ontology loaded"):
            await registry.reload("never-loaded")
    finally:
        await registry.close()


async def test_load_refuses_with_the_error_type_it_documents() -> None:
    """Every refusal ``load`` makes is a ``ValidationError`` carrying a context.

    Two were not: no argument over a registry constructed with no document
    raised a bare ``ValueError``, and a config of the wrong type raised
    ``TypeError`` from the coercion. Both are refusals of a *document*, which
    is what the rest of this module spells one way.
    """
    empty = OntologyRegistry()
    try:
        with pytest.raises(ValidationError, match="constructed with none"):
            await empty.load()
        with pytest.raises(ValidationError, match="OntologyConfig or a Mapping"):
            await empty.load(42)  # type: ignore[arg-type]
    finally:
        await empty.close()


# --------------------------------------------------------------------------
# A rebuild, per axis
# --------------------------------------------------------------------------


#: Two axes over two relations, with a disjoint population under each.
TWO_AXES: dict[str, Any] = {
    "id": "t",
    "entity_types": [{"id": "Thing"}],
    "entities": [
        {"id": "beagle", "type": "Thing", "name": "Beagle"},
        {"id": "dog", "type": "Thing", "name": "Dog"},
        {"id": "crimson", "type": "Thing", "name": "Crimson"},
        {"id": "red", "type": "Thing", "name": "Red"},
    ],
    "assertions": [
        {"subject": "beagle", "relation": "isa", "object": "dog"},
        {"subject": "crimson", "relation": "shade_of", "object": "red"},
    ],
    "taxonomies": [
        {"id": "kinds", "relation": "isa"},
        {"id": "colours", "relation": "shade_of"},
    ],
}


async def _delta_bus() -> tuple[InMemoryEventBus, list[Event]]:
    """A bus subscribed to the ontology topic and both axis topics."""
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus = InMemoryEventBus()
    await bus.connect()
    for topic in ("ontology:t", "taxonomy:kinds", "taxonomy:colours"):
        await bus.subscribe(topic, record)
    return bus, seen


def _delta(seen: list[Event], topic: str) -> dict[str, Any]:
    """The one ``UPDATED`` payload this topic carried."""
    updates = [e.payload for e in seen if e.topic == topic and e.type is EventType.UPDATED]
    assert len(updates) == 1, f"{topic} carried {len(updates)} deltas"
    return updates[0]


async def test_a_rebuilt_axis_carries_its_own_delta_and_not_the_ontologys() -> None:
    """A subscriber to one axis hears about that axis.

    The three sets were computed over the whole declared population and then
    published once per axis, so a rename in ``kinds`` arrived on
    ``taxonomy:colours`` as a change to the colour axis. The whole-population
    delta is still published -- an entity in no axis at all would otherwise
    change nowhere -- but on the ontology's own topic, which is the level it
    is a delta over.
    """
    bus, seen = await _delta_bus()
    registry = OntologyRegistry.from_components(config=OntologyConfig(**TWO_AXES), event_bus=bus)
    try:
        await registry.load()
        seen.clear()
        renamed = dict(TWO_AXES)
        renamed["entities"] = [
            {"id": "beagle", "type": "Thing", "name": "Beagle Hound"},
            *[e for e in TWO_AXES["entities"] if e["id"] != "beagle"],
        ]
        await registry.load(renamed, replace=True)

        assert _delta(seen, "ontology:t")["renamed"] == ["beagle"]
        assert _delta(seen, "taxonomy:kinds")["renamed"] == ["beagle"]
        colours = _delta(seen, "taxonomy:colours")
        assert (colours["gone"], colours["arrived"], colours["renamed"]) == ([], [], [])
    finally:
        await registry.close()
        await bus.close()


async def test_an_axis_a_rebuild_removes_reports_its_population_gone() -> None:
    """The loop ran over the new axes only, so a removed one announced nothing.

    Its subscribers were the ones with most to learn: the axis they read is no
    longer declared, and every node that was in it has left.
    """
    bus, seen = await _delta_bus()
    registry = OntologyRegistry.from_components(config=OntologyConfig(**TWO_AXES), event_bus=bus)
    try:
        await registry.load()
        seen.clear()
        without = dict(TWO_AXES)
        without["taxonomies"] = [{"id": "kinds", "relation": "isa"}]
        await registry.load(without, replace=True)

        colours = _delta(seen, "taxonomy:colours")
        assert colours["gone"] == ["crimson", "red"]
        assert (colours["arrived"], colours["renamed"]) == ([], [])
        kinds = _delta(seen, "taxonomy:kinds")
        assert (kinds["gone"], kinds["arrived"], kinds["renamed"]) == ([], [], [])
    finally:
        await registry.close()
        await bus.close()


# --------------------------------------------------------------------------
# The block that closes what the registry opened
# --------------------------------------------------------------------------


async def test_the_registry_closes_what_it_opened_when_the_block_ends(
    deployment: Path,
) -> None:
    """``async with`` is ``close()``, so the ownership story is enforced not remembered.

    Entry builds nothing. A registry opens handles at :meth:`load`, so there
    is no ``connect()`` here to pair the exit with -- which is the difference
    from :class:`AsyncDatabase`'s block and the reason entry only hands back
    the registry.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = await OntologyRegistry.from_config_async(cfg.resolve_for_build("ontology"))
    async with registry as entered:
        assert entered is registry, "entry hands back the registry, not a wrapper"
        built = [handle for handle, owned in registry._handles if owned]
        assert built, "a configured registry resolved its own handle"
        assert registry.list_ids() == ["catalog"]

    assert all(getattr(handle, "_connected", False) is False for handle in built)
    # The vocabulary stays listed: the block is close(), which is not unload().
    assert registry.list_ids() == ["catalog"]


async def test_the_block_closes_on_the_way_out_of_an_exception(deployment: Path) -> None:
    """The half that makes it worth having: the path a ``finally`` is forgotten on.

    A handle released only when the caller remembers is released only on the
    paths the caller thought about, and this is the other one.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = await OntologyRegistry.from_config_async(cfg.resolve_for_build("ontology"))
    built = [handle for handle, owned in registry._handles if owned]
    assert built

    with pytest.raises(RuntimeError, match="from inside the block"):
        async with registry:
            raise RuntimeError("raised from inside the block")

    assert all(getattr(handle, "_connected", False) is False for handle in built)


async def test_an_injected_handle_survives_the_block_that_the_registry_did_not_open() -> None:
    """The block is ``close()`` exactly, so it draws ownership on the same line."""
    handed_over = await _seeded()
    async with OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=handed_over
    ) as registry:
        await registry.load()

    assert await handed_over.search(Query(filters=[Filter("sku", Operator.EQ, "sku-4471")]))


async def test_a_configured_registry_is_not_missing_the_collaborators_it_resolves() -> None:
    """A registry that built everything it needs does not report itself under-wired.

    ``EXPECTED_COMPONENTS`` means *must be supplied*, and this class declared
    its two injection points there because there was no other field to declare
    them in. The cost was a live one rather than a documentation one: a fully
    loaded registry with nothing wrong with it answered
    ``{"database", "event_bus"}`` to ``missing_components()`` and raised from
    ``require_components()``, so any tooling reading either got a false
    positive on a correct object.
    """
    registry = await OntologyRegistry.from_config_async(_authored())
    try:
        assert registry.list_ids() == ["t"]
        assert registry.missing_components() == frozenset()
        registry.require_components()
    finally:
        await registry.close()


async def test_what_the_registry_may_be_handed_is_still_advertised() -> None:
    """Removing the false positive must not remove the answer tooling wanted.

    The names are real injection points -- ``from_components`` takes them --
    so a caller asking what this class accepts still has to be told. They sit
    on the field that says *may*, and the union is readable in one call.

    ``embedder`` is the one of them the configured door cannot resolve for
    itself: the construct that builds one lives in a package that depends on
    this one, so the edge runs the wrong way. It is declared here for that
    reason rather than left undeclared, because a collaborator a caller must
    supply and cannot discover is worse than one they merely may supply.
    """
    accepted = frozenset({"database", "embedder", "event_bus", "forms_database"})
    assert OntologyRegistry.expected_components() == frozenset()
    assert OntologyRegistry.optional_components() == accepted
    assert OntologyRegistry.accepted_components() == accepted


async def test_an_injected_collaborator_lands_where_from_components_can_see_it() -> None:
    """The *may* declaration is not a claim that nothing arrives.

    Guards the direction the first test could hide: a registry that really was
    handed a database has it on ``components``, so the diff answering empty
    above is "nothing is required", not "nothing is ever there".
    """
    database = AsyncMemoryDatabase()
    registry = OntologyRegistry.from_components(config=OntologyConfig(id=""), database=database)
    try:
        assert registry.components["database"] is database
        assert registry.missing_components() == frozenset()
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# The injected door, asked the question the configured one is asked
# --------------------------------------------------------------------------


def _two_table_document() -> dict[str, Any]:
    """A binding whose entity rows and form rows live in different tables."""
    return _document(
        sources=[
            {
                "id": "products",
                "kind": "record",
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                    "surface_forms": {
                        "table": "product_forms",
                        "form": "folded_form",
                        "entity": "sku",
                    },
                },
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                    {"name": "folded_form", "type": "string"},
                ],
            }
        ]
    )


@requires_package("aiosqlite")
async def test_one_injected_handle_cannot_answer_a_bindings_two_tables(tmp_path: Path) -> None:
    """A handle that reaches one table is refused a projection naming two.

    The configured door asks the backend whether a handle is one table's and
    opens a second when it is. The injected door used to ask nothing: it set
    both to the handle it was given, so a SQLite handle bound to ``products``
    was asked for ``product_forms`` rows and found none -- and *none* is the
    contract's spelling of **ran and matched nothing**, so
    ``by_surface_form`` answered ``frozenset()`` while ``describe()``
    advertised ``SURFACE_FORM_LOOKUP`` and every guard passed.

    That is the failure the capability apparatus exists to prevent, reached
    through the one door that skipped it, which is why the refusal is here
    rather than a note in the guide.
    """
    database = async_database_factory.create(
        backend="sqlite", path=str(tmp_path / "catalog.db"), table="products"
    )
    await database.connect()
    try:
        registry = OntologyRegistry.from_components(
            config=OntologyConfig(**_two_table_document()), database=database
        )
        with pytest.raises(ValidationError, match="product_forms"):
            await registry.load()
        await registry.close()
    finally:
        await database.close()


@requires_package("aiosqlite")
async def test_a_second_injected_handle_answers_the_bindings_other_table(tmp_path: Path) -> None:
    """The refusal above has a door out, and it is the one the source already had.

    ``RecordEntitySource`` has taken ``forms_database=`` since it shipped; the
    registry simply never offered it. Declaring it means the injected door can
    express what the configured door can, rather than sending a caller who
    holds two real handles back to a ``database:`` block they may not have.
    """
    path = str(tmp_path / "catalog.db")
    entities = async_database_factory.create(backend="sqlite", path=path, table="products")
    forms = async_database_factory.create(backend="sqlite", path=path, table="product_forms")
    await entities.connect()
    await forms.connect()
    try:
        await entities.create(Record({"sku": "sku-4471", "title": "Beagle"}))
        await forms.create(Record({"folded_form": "beagle", "sku": "sku-4471"}))
        registry = OntologyRegistry.from_components(
            config=OntologyConfig(**_two_table_document()),
            database=entities,
            forms_database=forms,
        )
        try:
            ontology = await registry.load()
            entity = await ontology.entity("sku-4471")
            assert entity is not None and entity.name == "Beagle"
            assert await ontology.by_surface_form("Beagle") == frozenset({"sku-4471"})
            assert not any(owned for _, owned in registry._handles), "injected handles are not ours"
        finally:
            await registry.close()
        assert await entities.search(Query(filters=[Filter("sku", Operator.EQ, "sku-4471")]))
    finally:
        await entities.close()
        await forms.close()


async def test_a_store_that_is_its_own_handle_still_needs_only_one(tmp_path: Path) -> None:
    """The refusal is the backend's answer, not a rule about two table names.

    ``memory`` declares no ``table``, so one handle *is* the store and both
    kinds of row live in it -- discriminated by the form column, which is what
    ``entity_filters`` emits. A projection naming two tables over such a
    handle is the shared-store arrangement working, not the defect above, and
    it must keep loading from one injected handle.
    """
    database = AsyncMemoryDatabase()
    await database.create(Record({"sku": "sku-1", "title": "Beagle"}))
    await database.create(Record({"folded_form": "beagle", "sku": "sku-1"}))
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_two_table_document()), database=database
    )
    try:
        ontology = await registry.load()
        assert await ontology.by_surface_form("Beagle") == frozenset({"sku-1"})
    finally:
        await registry.close()


async def test_forms_database_is_declared_as_a_collaborator_a_caller_may_inject() -> None:
    """A door tooling cannot see is a door a consumer does not find."""
    assert "forms_database" in OntologyRegistry.optional_components()
    assert "forms_database" in OntologyRegistry.accepted_components()
    assert OntologyRegistry.expected_components() == frozenset()


# --------------------------------------------------------------------------
# A scan declared over a table nothing can bound
# --------------------------------------------------------------------------


def _scanning_document(**lookup: Any) -> dict[str, Any]:
    """A record binding with a declared ``scan`` rung over it."""
    return _document(
        resolver={"rungs": [{"kind": "scan"}]},
        sources=[
            {
                "id": "products",
                "kind": "record",
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                    "surface_forms": {
                        "table": "product_forms",
                        "form": "folded_form",
                        "entity": "sku",
                        **lookup,
                    },
                },
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                    {"name": "folded_form", "type": "string"},
                ],
            }
        ],
    )


async def test_a_declared_scan_over_a_binding_that_cannot_bound_it_is_refused() -> None:
    """The combination that is quadratic in a caller's input, refused at load.

    A scanning rung probes every window of a query and takes its bound from
    the vocabulary's longest declared form. An authored index counts its own
    keys; a live table cannot be counted, so the source answers ``None`` and
    the rung enumerates in full -- *n(n+1)/2* probes, each a database round
    trip on this source. Fifty tokens is 1,275 of them.

    Refused here rather than survived, and the refusal names the key that
    fixes it: the number is one a consumer knows about their own vocabulary
    and nothing else can supply.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_scanning_document()), database=AsyncMemoryDatabase()
    )
    try:
        with pytest.raises(ValidationError, match="longest_form_tokens"):
            await registry.load()
    finally:
        await registry.close()


async def test_a_declared_scan_loads_where_the_binding_declares_its_bound() -> None:
    """The refusal's door out is the key it names."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_scanning_document(longest_form_tokens=4)),
        database=AsyncMemoryDatabase(),
    )
    try:
        ontology = await registry.load()
        assert ontology.entities.longest_form_tokens() == 4
    finally:
        await registry.close()


async def test_a_composition_with_no_scan_needs_no_bound() -> None:
    """Only the rung whose cost depends on the number is asked for it.

    ``exact`` reads folded forms too, but it compares the whole query once --
    its cost does not depend on how long the longest declared form is, so a
    binding it reads needs no bound and must keep loading without one.
    """
    document = _scanning_document()
    document["resolver"] = {"rungs": [{"kind": "exact"}, {"kind": "alias"}]}
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**document), database=AsyncMemoryDatabase()
    )
    try:
        ontology = await registry.load()
        assert ontology.entities.longest_form_tokens() is None
    finally:
        await registry.close()


# --------------------------------------------------------------------------
# Two bindings, one store that cannot tell them apart
# --------------------------------------------------------------------------


def _binding(ontology_id: str, table: str, type_id: str, **projection: Any) -> dict[str, Any]:
    """One `kind: record` ontology over a `$resource`, projecting one table."""
    entity_projection: dict[str, Any] = {
        "table": table,
        "id": "sku",
        "name": "title",
        "type": {"const": type_id},
    }
    entity_projection.update(projection)
    return {
        "id": ontology_id,
        "version": "1.0",
        "entity_types": [{"id": type_id}],
        "sources": [
            {
                "id": f"{ontology_id}-src",
                "kind": "record",
                "database": {"$resource": "catalog", "type": "databases"},
                "entity_projection": entity_projection,
                "schema": [
                    {"name": "sku", "type": "string"},
                    {"name": "title", "type": "string"},
                    {"name": "folded_form", "type": "string"},
                ],
            }
        ],
    }


async def test_two_bindings_naming_two_tables_of_one_storeless_backend_are_refused(
    deployment: Path,
) -> None:
    """The population each one reports is the other's as well.

    ``memory`` declares no ``table`` on its config, so the registry opens one
    handle for both bindings and ``table:`` narrows nothing -- it survives as a
    cache-key discriminator that :meth:`_keyed_block` discards and as a
    ``describe()`` field. Neither read is filtered by it, so ``by_type`` over
    the products binding answers with the supplier ids too, and ``get`` on a
    supplier id answers with that row projected as a Product: an entity with
    the wrong type, the wrong name and no error anywhere.

    Refused rather than narrowed because there is nothing to narrow *with*.
    The one discriminator this design has is the form column, and it separates
    form rows from entity rows rather than one binding's entities from
    another's.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = OntologyRegistry(environment=cfg.environment)
    try:
        await registry.load(_binding("catalog-products", "products", "Product"))

        with pytest.raises(ValidationError) as raised:
            await registry.load(_binding("catalog-suppliers", "suppliers", "Supplier"))

        message = str(raised.value)
        assert "suppliers" in message
        assert "products" in message
        assert "catalog-products" in message, "the refusal names who holds the store"
        assert registry.list_ids() == ["catalog-products"], "the refused load left nothing"
    finally:
        await registry.close()


async def test_two_bindings_that_name_one_table_are_left_alone(deployment: Path) -> None:
    """Sharing a table is a decision; sharing a store by accident is not.

    Two documents projecting *the same* table asked for the population they
    get -- one as a catalogue entry, one as something else, over rows that
    really are shared. Nothing here can tell that apart from a mistake, and
    refusing it would cost a composition that is reading exactly what it
    declared. What the refusal above catches is narrower: two bindings that
    named two tables and were handed one.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = OntologyRegistry(environment=cfg.environment)
    try:
        await registry.load(_binding("catalog-products", "products", "Product"))
        await registry.load(_binding("catalog-listings", "products", "Listing"))

        assert registry.list_ids() == ["catalog-products", "catalog-listings"]
    finally:
        await registry.close()


async def test_two_bindings_sharing_a_store_must_also_agree_on_the_form_table(
    deployment: Path,
) -> None:
    """The lookup contaminates on an axis of its own, and has no type to pin.

    ``by_surface_form`` emits one filter -- the folded form column -- against a
    table the backend does not separate, so two bindings whose form rows land
    in one store answer with each other's entity ids. Agreeing on the entity
    table is therefore not enough: the whole set of tables a binding names has
    to match the one already holding the store, or the two are separable only
    in the document.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = OntologyRegistry(environment=cfg.environment)
    lookup = {"table": "product_forms", "form": "folded_form", "entity": "sku"}
    try:
        await registry.load(
            _binding("catalog-products", "products", "Product", surface_forms=lookup)
        )

        with pytest.raises(ValidationError) as raised:
            await registry.load(
                _binding(
                    "catalog-listings",
                    "products",
                    "Listing",
                    surface_forms=dict(lookup, table="listing_forms"),
                )
            )

        assert "listing_forms" in str(raised.value)
    finally:
        await registry.close()


@requires_package("aiosqlite")
async def test_a_backend_that_addresses_a_table_binds_both_without_a_refusal(
    tmp_path: Path,
) -> None:
    """The negative control, and the reason the refusal asks the backend.

    ``sqlite`` declares a ``table`` on its config, so the registry opens a
    handle per table and each binding's reads reach only its own rows. The
    same two documents that cannot be told apart over ``memory`` are served
    correctly here -- which is what makes this a property of the backend
    rather than a rule about how many ontologies a registry may hold.
    """
    path = str(tmp_path / "catalog.db")
    environment = EnvironmentConfig(
        name="t", resources={"databases": {"catalog": {"backend": "sqlite", "path": path}}}
    )
    registry = OntologyRegistry(environment=environment)
    try:
        await registry.load(_binding("catalog-products", "products", "Product"))
        await registry.load(_binding("catalog-suppliers", "suppliers", "Supplier"))

        assert registry.list_ids() == ["catalog-products", "catalog-suppliers"]
    finally:
        await registry.close()


async def test_one_injected_handle_cannot_carry_two_bindings_either(deployment: Path) -> None:
    """The injected door reaches the same store by a shorter route.

    ``from_components`` holds one handle for every document the registry
    loads, so two documents through that door share a store for a reason the
    configured door reaches only through a backend that declares no table.
    Same hazard, and the refusal is the handle's rather than the block's --
    which is why it is asked of the handle and not of a ``database:`` section
    that the injected door does not have.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=AsyncMemoryDatabase()
    )
    try:
        await registry.load()

        with pytest.raises(ValidationError) as raised:
            await registry.load(
                OntologyConfig(
                    **_document(
                        id="other",
                        sources=[
                            {
                                "id": "suppliers",
                                "kind": "record",
                                "entity_projection": {
                                    "table": "suppliers",
                                    "id": "sku",
                                    "name": "title",
                                    "type": {"const": "Product"},
                                },
                                "schema": [
                                    {"name": "sku", "type": "string"},
                                    {"name": "title", "type": "string"},
                                ],
                            }
                        ],
                    )
                )
            )

        assert "suppliers" in str(raised.value)
    finally:
        await registry.close()


async def test_unloading_the_holder_frees_the_store_for_another_binding(
    deployment: Path,
) -> None:
    """The claim is the loaded vocabulary's, not the registry's for all time.

    A refusal that outlived the ontology it was taken for would make
    :meth:`unload` a partial operation -- the vocabulary gone and the store
    still spoken for, with nothing a caller could do about it but build a
    second registry. So the question is asked of what is loaded *now*, which
    is also why replacing a document with itself is not a collision with
    itself.
    """
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = OntologyRegistry(environment=cfg.environment)
    try:
        await registry.load(_binding("catalog-products", "products", "Product"))
        assert await registry.unload("catalog-products") is True

        await registry.load(_binding("catalog-suppliers", "suppliers", "Supplier"))

        assert registry.list_ids() == ["catalog-suppliers"]
    finally:
        await registry.close()


async def test_replacing_a_document_does_not_collide_with_the_one_it_replaces(
    deployment: Path,
) -> None:
    """``replace=True`` takes the id, so the departing claim is the arriving one's."""
    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = OntologyRegistry(environment=cfg.environment)
    try:
        await registry.load(_binding("catalog-products", "products", "Product"))

        again = await registry.load(
            _binding("catalog-products", "widgets", "Product"), replace=True
        )

        assert again.describes[0].table == "widgets"
    finally:
        await registry.close()


async def test_a_second_documents_event_bus_block_is_reported_rather_than_applied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One registry holds one bus, so the second block cannot be honoured.

    Moving to it would leave every vocabulary already announced on the first
    bus talking to subscribers nobody is holding, so the first wins -- and the
    consequence of *not* saying so is a consumer watching a bus their document
    configured and concluding the registry never published, which is the
    reading a silent pass-over invites. Reported at ``INFO``, like a
    ``database:`` block passed over for an injected handle.
    """
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded()
    )
    try:
        await registry.load(_document(event_bus={"backend": "memory"}))
        first = registry._event_bus
        assert first is not None

        caplog.clear()
        with caplog.at_level(logging.INFO):
            await registry.load(
                _document(id="second", event_bus={"backend": "memory"}), replace=False
            )

        assert registry._event_bus is first, "the bus the first document built is kept"
        assert "second" in caplog.text
        assert "the block is not built" in caplog.text
        assert "a configured" in caplog.text, "it says which kind of bus is already held"
    finally:
        await registry.close()


async def test_an_injected_bus_says_so_when_it_passes_over_a_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same report, naming the other reason the block was not built.

    An injected bus winning over a configured block was already the documented
    rule; what was missing is that the document declaring the block is never
    told. The two cases are one sentence with one word different, so a reader
    of the log learns *which* bus they are watching instead of theirs.
    """
    handed_over = InMemoryEventBus()
    await handed_over.connect()
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded(), event_bus=handed_over
    )
    try:
        caplog.clear()
        with caplog.at_level(logging.INFO):
            await registry.load(_document(event_bus={"backend": "memory"}))

        assert registry._event_bus is handed_over
        assert registry._owns_event_bus is False
        assert "an injected" in caplog.text
        assert "the block is not built" in caplog.text
    finally:
        await registry.close()
        await handed_over.close()

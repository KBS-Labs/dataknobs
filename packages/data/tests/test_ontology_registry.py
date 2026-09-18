"""The registry that owns a lifecycle: both doors, both config forms, teardown.

Four of these are the acceptance surface for the configuration contract --
the two doors reaching one object, the portable form round-tripping one level
above the resolved one, the three strictness levels, and the shipped refusal
that had been naming a class nobody could import. The rest are what those four
leave uncovered: a refusal is asserted by nothing else, and neither is the
distinction between closing a registry and unloading from one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.events import Event, EventType, InMemoryEventBus
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    AUTHORED_SOURCE_ID,
    AsyncOntology,
    OntologyConfig,
    async_load_ontology,
)
from dataknobs_common.records import Record
from dataknobs_common.testing import assert_no_blocking, assert_structured_config_consumer
from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig
from dataknobs_config.environment_config import ResourceNotFoundError

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.query import Filter, Operator, Query

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        - {name: folded_form, type: string}
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
# What the leg owes beside the criteria
# --------------------------------------------------------------------------


def test_the_structured_config_pattern_is_applied() -> None:
    """The mixin's contracts, with the three ctor params that are not config fields.

    `environment`, `strict_resources` and `config_key` are properties of a
    *registry*, not of an ontology document, so none of them is a field on
    `OntologyConfig` and none should be. Naming them is the guard's own
    documented channel for saying so. The other direction cannot drift: the
    ctor takes `**kwargs` through the mixin, so every config field is accepted
    by construction.
    """
    assert_structured_config_consumer(
        OntologyRegistry,
        ignore_params={"environment", "strict_resources", "config_key"},
    )


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
    """
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus = InMemoryEventBus()
    await bus.connect()
    await bus.subscribe("ontology:t", record)
    await bus.subscribe("taxonomy:kinds", record)

    document = _document(
        entities=[{"id": "beagle", "type": "Breed", "name": "Beagle"}],
        entity_types=[{"id": "Breed"}],
        sources=[],
        taxonomies=[{"id": "kinds", "relation": "isa"}],
    )
    registry = OntologyRegistry.from_components(config=OntologyConfig(**document), event_bus=bus)
    try:
        await registry.load()
        assert [(e.topic, e.type) for e in seen] == [("ontology:t", EventType.CREATED)]

        seen.clear()
        renamed = dict(document)
        renamed["entities"] = [{"id": "beagle", "type": "Breed", "name": "Beagle Hound"}]
        await registry.load(renamed, replace=True)
        assert [(e.topic, e.type) for e in seen] == [
            ("ontology:t", EventType.DELETED),
            ("ontology:t", EventType.CREATED),
            ("taxonomy:kinds", EventType.UPDATED),
        ]
        # The third set is the point: a rename keeps its id, so a two-set
        # delta would report it as no change at all.
        rebuilt = seen[-1].payload
        assert rebuilt["gone"] == [] and rebuilt["arrived"] == []
        assert rebuilt["renamed"] == ["beagle"]
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
    """Absence is a configuration answer, not an error -- from either side of it."""
    registry = OntologyRegistry.from_components(
        config=OntologyConfig(**_document()), database=await _seeded()
    )
    try:
        await registry.load()
        assert registry.index("t") is None
        assert registry.resolver("t") is None
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

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The door that owns a lifecycle.

``dataknobs_common.ontology``'s two module-level doors refuse a *live* source
kind, and the refusal is about ownership rather than about what is installed:
binding a database source creates something that must be opened and closed
again, and a function returning a value has no ``close()`` and nobody to call
it. This is the object that does.

It lives in ``dataknobs-data`` because that is where the handles are. The
vocabulary it produces stays a value -- :class:`AsyncOntology` owns no
lifecycle, every accessor on it is pure over its own fields, and it is safe to
hold and pass after the registry that built it has gone. What is *not* safe is
reading a source whose handle this registry has closed, which is why
:meth:`OntologyRegistry.close` is a separate act from :meth:`unload`.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.events import Event, EventType, create_event_bus
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import AsyncMappingHierarchy
from dataknobs_common.lifecycle import close_if_owned
from dataknobs_common.ontology import (
    AUTHORED_SOURCE_ID,
    AUTHORED_SOURCE_KINDS,
    AsyncAssertionHierarchy,
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    AsyncOntology,
    Entity,
    OntologyConfig,
    StrCodec,
    axes_to_copy,
    build_ontology,
    split_qualified,
)
from dataknobs_common.ontology.model import DK_ENTITY_TYPE, DK_RELATION_TYPE
from dataknobs_common.structured_config import StructuredConfigConsumer

from dataknobs_data.factory import async_database_factory
from dataknobs_data.fields import FieldType
from dataknobs_data.ontology.sources import (
    RECORD_SOURCE_KIND,
    EntityProjection,
    RecordEntitySource,
    validate_against_schema,
)
from dataknobs_data.schema import DatabaseSchema, FieldSchema

if TYPE_CHECKING:
    from dataknobs_common.entity_resolution.protocols import AsyncEntityResolver
    from dataknobs_common.events import EventBus
    from dataknobs_common.ontology import OntologyParts, SourceDescription
    from dataknobs_config import EnvironmentConfig

    from dataknobs_data.database import AsyncDatabase

logger = logging.getLogger(__name__)

#: The vocabulary's own two type declarations, answerable without a load.
#:
#: ``dk`` is the reserved ontology id every document is refused for claiming,
#: and these are what it names. Held here because :meth:`resolve_ref` is where
#: a caller meets them -- a qualified id is parsed the same way whichever
#: namespace it lands in.
_BUILTIN_ENTITIES: Mapping[str, Entity[str]] = {
    DK_ENTITY_TYPE: Entity(
        id=DK_ENTITY_TYPE,
        type=DK_ENTITY_TYPE,
        name="EntityType",
        description="A declaration of a kind of entity.",
    ),
    DK_RELATION_TYPE: Entity(
        id=DK_RELATION_TYPE,
        type=DK_ENTITY_TYPE,
        name="RelationType",
        description="A declaration of a kind of relation.",
    ),
}


class OntologyRegistry(StructuredConfigConsumer[OntologyConfig]):
    """Loads ontology documents, binds their live sources, and closes what it opened.

    Two doors, and which form each takes is the whole of the distinction
    between them: **a constructor takes the resolved form, a registry stores
    the portable one and resolves it.**

    - :meth:`from_config_async` takes the **resolved** form -- what
      ``cfg.resolve_for_build("ontology")`` returns -- and loads it.
    - :meth:`load` takes the **portable** form, ``$resource`` references intact
      and ``${VAR}`` unexpanded, and resolves it through this registry's own
      environment and strictness level. That is the form a deployment *stores*,
      so it is the form a registry reads back.
    - :meth:`from_components` takes handles already built, which is what a test
      with no environment and no file has. It adopts them and loads nothing;
      ``await registry.load()`` over the config it was handed is the line that
      reaches the same vocabulary the configured door reaches.

    **The registry instance is the unit of sharing.** Ids are one flat
    namespace inside one registry, so a deployment wanting per-party vocabulary
    lifecycles constructs a registry per party rather than qualifying ids
    further.

    **Teardown is owned-versus-injected, recorded per handle.** A handle this
    registry built from a resolved ``$resource`` is closed by
    :meth:`close`; one handed in through :meth:`from_components` is left
    untouched for its owner. Ownership is recorded when the handle is
    *acquired* rather than recomputed at teardown, because a registry that
    resolved one reference and was handed another has two handles with two
    different owners and a single flag cannot say that.

    **``EXPECTED_COMPONENTS`` is an advertise surface here, and that is worth
    stating.** The mixin documents the field as *what a consumer must supply*
    and pairs it with :meth:`require_components`. Neither collaborator named
    below is required: a configured registry resolves its own database from
    ``$resource`` and publishes nothing when no bus is wired. So this class
    declares the two a caller *may* inject -- which is what ``from_components``
    accepts and what tooling reading ``expected_components()`` wants to know --
    and never calls ``require_components()``, whose answer would be wrong for
    the configured door. This is the field's first adopter in the tree; if the
    two readings need separating, this is the class that found it.
    """

    CONFIG_CLS: ClassVar[type[OntologyConfig]] = OntologyConfig

    #: The collaborators a caller may inject. See the class docstring for why
    #: this is read as *may* rather than *must*, and what that costs.
    EXPECTED_COMPONENTS: ClassVar[frozenset[str]] = frozenset({"database", "event_bus"})

    #: The live source kinds this registry binds.
    #:
    #: Computed from the declared kind alone, exactly as
    #: ``dataknobs_common.ontology.loader``'s refusal is, so one config means
    #: one thing in every environment: a kind outside this set and outside
    #: ``AUTHORED_SOURCE_KINDS`` is refused by name whether or not some other
    #: package that could bind it happens to be importable.
    LIVE_SOURCE_KINDS: ClassVar[frozenset[str]] = frozenset({RECORD_SOURCE_KIND})

    def __init__(
        self,
        config: OntologyConfig | Mapping[str, Any] | None = None,
        *,
        environment: EnvironmentConfig | str | None = None,
        strict_resources: bool | None = True,
        config_key: str = "ontology",
        **kwargs: Any,
    ) -> None:
        """Build a registry.

        Args:
            config: The document this registry's construction door loads. The
                typed form, or the mapping it is built from
            environment: The environment ``$resource`` references resolve
                against -- an :class:`~dataknobs_config.EnvironmentConfig` or
                the name of one. ``None`` means references are not resolved and
                a stored document is used as it was stored. A caller whose
                environment files are not under ``config/environments`` loads
                the object themselves and passes it
            strict_resources: What a ``$resource`` naming a resource this
                environment does not define does. ``True`` -- **the default**
                -- raises, naming the resource, the environment and the config
                path; ``False`` degrades to the reference's inline defaults
                with a warning; ``None`` hands the level back to the
                environment's own ``strict_resources`` setting, then to
                ``False``. A reference's own ``$required`` overrides all three.

                Strict by default because of what leniency produces *over this
                grammar in particular*: a ``$resource`` block is marker keys
                only, so it carries no inline defaults to degrade to and a
                lenient resolution resolves it to ``{}`` -- an entity source
                built over an empty config, in an environment that was merely
                missing a definition, announced by one warning at boot
            config_key: The section of a stored document that holds the
                ontology. A document that *is* the section is used as-is, so
                both shapes a caller can hold resolve to the same answer
        """
        # A name is held, not resolved: `EnvironmentConfig.load` reads a file
        # from disk, and a constructor called inside an `async def` would do
        # that read on the event loop. It happens once, off the loop, at the
        # first resolution that needs it.
        self._environment: EnvironmentConfig | None = (
            None if isinstance(environment, str) else environment
        )
        self._environment_name = environment if isinstance(environment, str) else None
        self._strict_resources = strict_resources
        self._config_key = config_key
        self._ontologies: dict[str, AsyncOntology[str]] = {}
        self._parts: dict[str, OntologyParts] = {}
        self._documents: dict[str, dict[str, Any]] = {}
        # Every handle this registry has touched, with who owns it, in
        # acquisition order. `close()` walks it; `close_if_owned` skips the
        # injected ones.
        self._handles: list[tuple[Any, bool]] = []
        # Handles this registry built, keyed by the resolved config they were
        # built from. A reload against an unchanged environment reuses the
        # handle rather than opening a second connection and leaving the first
        # to a `close()` that may be a long way off.
        self._database_cache: dict[str, AsyncDatabase] = {}
        self._injected_database: AsyncDatabase | None = None
        self._event_bus: EventBus | None = None
        self._owns_event_bus = False
        # Read off the raw mapping, because `OntologyConfig` has no field for
        # it and `from_dict` drops what it does not declare -- rightly: a bus
        # is not part of a vocabulary. It is the registry that owns a
        # lifecycle, so it is the registry that reads this.
        self._event_bus_block = _event_bus_block(config)
        # A registry may hold no document at all -- `OntologyRegistry()` then
        # `await registry.load(stored)` is the shape a deployment reading from
        # a backend has. The mixin requires *a* config, and `id` is the one
        # field an ontology cannot do without, so the empty id is what "none
        # was configured" is spelled as: no document can claim it, which is
        # exactly what makes it available to mean this.
        super().__init__(OntologyConfig(id="") if config is None else config, **kwargs)

    # ----------------------------------------------------------------- hooks

    def _setup(self) -> None:
        """Bind injected collaborators, which every construction path delivers.

        Here rather than in :meth:`_ainit` / ``_adopt_components`` because each
        of those runs on exactly one door and this runs on all of them: a
        registry built through the synchronous ``from_config(config,
        database=db)`` would otherwise hold the handle on ``self.components``
        and never look at it.
        """
        database = self.components.get("database")
        if database is not None:
            self._injected_database = database
            self._handles.append((database, False))
        event_bus = self.components.get("event_bus")
        if event_bus is not None:
            self._event_bus = event_bus
            self._owns_event_bus = False

    async def _ainit(self, **components: Any) -> None:
        """Load the document this registry was constructed from.

        The configured door's whole convenience, and it is :meth:`load`'s work
        rather than a second way to do it -- the config handed to
        ``from_config_async`` is already resolved, so what runs is the half of
        ``load`` that comes after resolution.

        The collaborators the framework delivers here are already bound: they
        arrive on every construction path and :meth:`_setup` is the one hook
        that runs on all of them. Declared anyway, because the base declares
        them and a hook that narrows its own delivery is a hook the framework
        cannot call with what it has.
        """
        if self._prebuilt or not self._config.id:
            return
        if self._event_bus_block is not None:
            await self._event_bus_from(self._event_bus_block)
        await self._load_resolved(self._config)

    # ------------------------------------------------------------ the config

    @staticmethod
    def get_portable_config(cfg: Any) -> dict[str, Any]:
        """The storable form: ``$resource`` refs intact, ``${VAR}`` unexpanded.

        What a deployment puts in a backend, and what :meth:`load` reads back.
        An :class:`~dataknobs_config.EnvironmentAwareConfig` answers it
        directly; a mapping is already portable and passes through, copied so
        the caller's dict is not the registry's.
        """
        accessor = getattr(cfg, "get_portable_config", None)
        if callable(accessor):
            portable: dict[str, Any] = accessor()
            return portable
        if isinstance(cfg, Mapping):
            return copy.deepcopy(dict(cfg))
        raise TypeError(
            f"get_portable_config: expected an EnvironmentAwareConfig or a "
            f"Mapping, got {type(cfg).__name__}"
        )

    # ------------------------------------------------------------- lifecycle

    async def load(
        self,
        config: OntologyConfig | Mapping[str, Any] | None = None,
        *,
        replace: bool = False,
    ) -> AsyncOntology[str]:
        """Load a document, resolving it against this registry's environment.

        Takes the **portable** form. The section key is read off the document
        when it carries one and the document is used as-is when it does not, so
        a caller holding either shape gets the same answer; resolution then
        runs through this registry's environment and strictness level. With no
        environment held, the mapping is used exactly as stored.

        Args:
            config: The document, or None for the one this registry was
                constructed with -- which is the line that gives the
                ``from_components`` door the vocabulary the configured door
                loads for itself
            replace: Whether a document may take an id this registry already
                holds

        Returns:
            The loaded vocabulary

        Raises:
            ValidationError: For every refusal ``build_ontology`` makes, and
                for this registry's own: an id already loaded without
                ``replace``, a source kind nothing binds, a document mixing an
                authored vocabulary with a live one, and a live binding whose
                schema, projection or surface-form lookup does not hold up
            ResourceNotFoundError: When a ``$resource`` names a resource this
                environment does not define and the effective level is strict
        """
        if config is None and not self._config.id:
            raise ValueError(
                "load() with no argument loads the document this registry was "
                "constructed with, and this one was constructed with none. Pass "
                "the document, or construct the registry with it"
            )
        await self._ensure_environment()
        document = self._as_document(self._config if config is None else config)
        section = document.get(self._config_key, document)
        if not isinstance(section, Mapping):
            raise ValidationError(
                f"`{self._config_key}:` must be a mapping, got {type(section).__name__}",
                context={self._config_key: section},
            )
        portable = copy.deepcopy(dict(section))
        resolved = self._resolve(portable)
        bus_block = resolved.get("event_bus")
        if isinstance(bus_block, Mapping):
            await self._event_bus_from(bus_block)
        ontology = await self._load_resolved(resolved, replace=replace)
        self._documents[ontology.id] = portable
        return ontology

    async def unload(self, ontology_id: str) -> bool:
        """Drop a vocabulary this registry loaded, and say whether it was here.

        ``True`` means it was present and is now removed from **this**
        registry. It makes no claim about subscribers -- the bus cannot report
        them -- and there is no refcount: a registry removes what it loaded.

        **It releases no handle.** :meth:`get` hands out a value that outlives
        its entry, and this object cannot know whether a caller still holds
        one, so the moment a handle is released is the moment the caller says
        they are done with the registry: :meth:`close`.

        The departing ids travel with the event, in one of two forms, decided
        by the sources rather than by a threshold: the **id set** where every
        bound source is authored, and the ``{ontology_id}:`` **prefix** as soon
        as one is live. Derivable from ``describe()``, which is free because it
        is configuration -- where a size threshold would need the count the
        prefix form exists to avoid taking.
        """
        ontology = self._ontologies.get(ontology_id)
        if ontology is None:
            return False
        await self._publish_departure(ontology)
        del self._ontologies[ontology_id]
        self._parts.pop(ontology_id, None)
        self._documents.pop(ontology_id, None)
        return True

    async def reload(self, ontology_id: str) -> AsyncOntology[str]:
        """Re-resolve the document this id was loaded from, against today's environment.

        ``load(stored, replace=True)`` over what this registry stored, and
        nothing more -- a replacement is what a rebuild *is*, so this is the
        one that was already written rather than a second path through it.
        What it buys over holding the vocabulary is the resolution: an
        environment whose ``$resource`` now names a different backend, or a
        ``${VAR}`` that has moved, reaches the sources here.

        Raises:
            KeyError: When no document with this id is loaded
        """
        document = self._documents.get(ontology_id)
        if document is None:
            raise KeyError(f"No ontology loaded with id {ontology_id!r}")
        return await self.load(document, replace=True)

    def get(self, ontology_id: str) -> AsyncOntology[str] | None:
        """The vocabulary this id names, or None.

        A value: it holds sources rather than entities and owns no lifecycle,
        so holding one past an :meth:`unload` is safe. Reading one past a
        :meth:`close` is not, and that is the distinction the two members keep.
        """
        return self._ontologies.get(ontology_id)

    def list_ids(self) -> list[str]:
        """The ids this registry holds, in load order."""
        return list(self._ontologies)

    async def resolve_ref(self, qualified_id: str) -> Entity[str] | None:
        """Name resolution: a qualified id to the entity it names, or None.

        ``async`` because the vocabularies this registry builds are the lazy
        twin -- the same correction ``AsyncOntology`` itself carries, for the
        same reason. A registry that binds live sources cannot answer from a
        table it has not read.

        ``dk:`` answers from the built-in table without a load. Everything else
        is parsed ontology-first, then re-parsed against the source ids *that*
        ontology declares, because the middle segment of a three-part id is
        recognised by membership in a closed set rather than by counting
        colons.

        This is **not** the router across a binding's sources; it is name
        resolution that returns an entity. The origin address is on the
        entity's own ``source``.
        """
        builtin = _BUILTIN_ENTITIES.get(qualified_id)
        if builtin is not None:
            return builtin
        ontology_id, separator, _ = qualified_id.partition(":")
        if not separator:
            return None
        ontology = self._ontologies.get(ontology_id)
        if ontology is None:
            return None
        parts = self._parts.get(ontology_id)
        source_ids = tuple(str(spec.get("id", "")) for spec in parts.source_specs) if parts else ()
        local = split_qualified(qualified_id, source_ids).local_id
        return await ontology.entity(local)

    def index(self, ontology_id: str) -> None:
        """The semantic index built for this ontology, or None.

        ``None`` for every id this registry builds today: the index and its
        sources are a later leg's, and this member is declared here because
        absence is a configuration answer rather than an error -- an ontology
        that declared no ``index:`` section answers ``None`` for good, and one
        whose registry has no index builder answers the same thing from the
        other side.

        It **returns** what ``load()`` built and is not a second way to build
        one. The return type widens to ``SemanticIndex | None`` when that type
        exists; today ``None`` is the only value it can have, and declaring
        that is more useful than declaring a type nothing can produce.
        """
        return None

    def resolver(self, ontology_id: str) -> AsyncEntityResolver | None:
        """The configured placement cascade for this ontology, or None.

        ``None`` for every id this registry builds today, for :meth:`index`'s
        reason and with :meth:`index`'s contract: it returns what ``load()``
        built. The ``resolver:`` section *is* read at load -- an ``exact`` rung
        declared over a live binding with no folded lookup is refused there --
        so the section is not ignored, only unbuilt.
        """
        return None

    async def close(self) -> None:
        """Release every handle this registry opened, and nothing it was handed.

        A cascade, so one failing handle does not abort the rest: each close is
        error-isolated and logged. Injected handles are left untouched for
        their owner.

        **It does not unload.** The ids stay listed and the values stay
        reachable; what is gone is the ability to read through them. A caller
        who wants the departure events calls :meth:`unload` first.
        """
        for handle, owned in self._handles:
            await close_if_owned(handle, owned, on_error=self._teardown_failed)
        await close_if_owned(self._event_bus, self._owns_event_bus, on_error=self._teardown_failed)
        self._handles = []
        self._database_cache = {}

    @staticmethod
    def _teardown_failed(exc: Exception) -> None:
        logger.warning("Closing a handle failed during registry teardown: %s", exc)

    # -------------------------------------------------------------- internals

    def _as_document(self, config: OntologyConfig | Mapping[str, Any]) -> dict[str, Any]:
        """A document as a plain mapping, whichever form the caller held."""
        if isinstance(config, OntologyConfig):
            return dict(config.to_dict())
        if isinstance(config, Mapping):
            return dict(config)
        raise TypeError(
            f"load: `config` must be an OntologyConfig or a Mapping, got {type(config).__name__}"
        )

    async def _ensure_environment(self) -> None:
        """Load the environment a name stands for, once, off the event loop.

        The only blocking thing this registry does is read a YAML file, so it
        is the only thing offloaded -- and only when there is a read: a caller
        who handed over an ``EnvironmentConfig`` object, or none at all, pays
        for no thread. Same shape, and the same reason, as the one read
        ``async_load_ontology`` offloads.
        """
        if self._environment is not None or self._environment_name is None:
            return
        from dataknobs_config import EnvironmentConfig as _EnvironmentConfig

        self._environment = await asyncio.to_thread(_EnvironmentConfig.load, self._environment_name)

    def _resolve(self, section: Mapping[str, Any]) -> dict[str, Any]:
        """Late-bind ``$resource`` and ``${VAR}`` against this registry's environment.

        Through the constructor rather than
        :meth:`~dataknobs_config.EnvironmentAwareConfig.from_dict`, because
        this registry holds an ``EnvironmentConfig`` **object** and that
        classmethod takes an environment *name* and loads one.
        """
        if self._environment is None:
            return dict(section)
        from dataknobs_config import EnvironmentAwareConfig

        aware = EnvironmentAwareConfig(
            dict(section),
            environment=self._environment,
            strict_resources=self._strict_resources,
        )
        return aware.resolve_for_build()

    async def _load_resolved(
        self,
        config: OntologyConfig | Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> AsyncOntology[str]:
        """Build a vocabulary from an already-resolved document, and announce it."""
        typed = (
            config if isinstance(config, OntologyConfig) else OntologyConfig.from_dict(dict(config))
        )
        parts = build_ontology(typed)
        if parts.id in self._ontologies and not replace:
            raise ValidationError(
                f"ontology {parts.id!r} is already loaded in this registry. Pass "
                f"`replace=True` to take the id, or use a registry per party -- "
                f"the registry instance is the unit of sharing",
                context={"ontology_id": parts.id},
            )
        replaced = self._ontologies.get(parts.id)
        before = self._parts.get(parts.id)
        entities, describes = await self._bind_sources(typed, parts)
        assertions = AsyncMappingAssertionSource(parts.declared_assertions)
        ontology: AsyncOntology[str] = AsyncOntology(
            id=parts.id,
            version=parts.version,
            entity_types=parts.entity_types,
            relation_types=parts.relation_types,
            entities=entities,
            assertions=assertions,
            taxonomies=parts.taxonomies,
            describes=describes,
            codec=StrCodec(),
            structures={
                name: await AsyncMappingHierarchy.snapshot(
                    AsyncAssertionHierarchy(assertions, definition.relation)
                )
                for name, definition in axes_to_copy(parts.taxonomies)
            },
            imports=parts.imports,
        )
        # Announced before the swap, so a subscriber reading the registry on
        # the departure event still sees what departed -- and in this order,
        # rather than as a third event meaning both, because a subscriber who
        # cares about departures and one who cares about arrivals each already
        # have the event they read.
        if replaced is not None:
            await self._publish_departure(replaced)
        self._ontologies[parts.id] = ontology
        self._parts[parts.id] = parts
        await self._publish(
            topic=f"ontology:{parts.id}",
            event_type=EventType.CREATED,
            payload={
                "ontology_id": parts.id,
                "version": parts.version,
                "sources": [description.source_id for description in describes],
            },
        )
        if before is not None:
            await self._publish_taxonomy_delta(parts.id, before, parts)
        return ontology

    async def _bind_sources(
        self, config: OntologyConfig, parts: OntologyParts
    ) -> tuple[Any, tuple[SourceDescription, ...]]:
        """Dispatch each declared source kind, and refuse the ones nothing binds.

        Two refusals compose here, and they answer different questions from the
        one ``dataknobs_common``'s loader answers. That refusal is *this door
        cannot own a lifecycle* and is correct forever; these are *no binder is
        registered for this kind* and *the construct that would route between
        two sources does not exist yet*.
        """
        live = [
            spec
            for spec in parts.source_specs
            if str(spec.get("kind", "")) not in AUTHORED_SOURCE_KINDS
        ]
        for spec in live:
            kind = str(spec.get("kind", ""))
            if kind not in self.LIVE_SOURCE_KINDS:
                raise ValidationError(
                    f"source {str(spec.get('id', ''))!r} declares kind {kind!r}, "
                    f"which no binder in this registry is registered for. Live "
                    f"kinds it binds: {sorted(self.LIVE_SOURCE_KINDS)}; kinds a "
                    f"module-level loader binds: {sorted(AUTHORED_SOURCE_KINDS)}",
                    context={"source_id": spec.get("id"), "kind": kind},
                )
        if not live:
            authored = AsyncMappingEntitySource(parts.declared_entities)
            return authored, (authored.describe(),)
        if parts.declared_entities:
            raise ValidationError(
                f"ontology {parts.id!r} declares both an authored vocabulary and "
                f"a live source ({str(live[0].get('id', ''))!r}). Reads across "
                f"both go through LayeredEntitySource, which routes on the "
                f"`source_id` segment of a qualified id and is not built yet; "
                f"until it is, letting one of the two win would be a silent "
                f"choice about which entities exist",
                context={"ontology_id": parts.id, "source_id": live[0].get("id")},
            )
        if len(live) > 1:
            raise ValidationError(
                f"ontology {parts.id!r} binds {len(live)} live sources "
                f"({sorted(str(spec.get('id', '')) for spec in live)}). Reads "
                f"across more than one go through LayeredEntitySource, which is "
                f"not built yet",
                context={
                    "ontology_id": parts.id,
                    "source_ids": [spec.get("id") for spec in live],
                },
            )
        source = await self._bind_record_source(config, live[0])
        return source, (source.describe(),)

    async def _bind_record_source(
        self, config: OntologyConfig, spec: Mapping[str, Any]
    ) -> RecordEntitySource:
        """One ``kind: record`` binding, validated before a handle is opened."""
        source_id = str(spec.get("id", ""))
        projection_spec = spec.get("entity_projection")
        if not isinstance(projection_spec, Mapping):
            raise ValidationError(
                f"binding {source_id!r} declares no `entity_projection:`. The "
                f"column-to-field map is configuration, never a callable: that "
                f"is what makes it invertible, and it is the decision a return "
                f"path would rest on",
                context={"source_id": source_id},
            )
        projection = EntityProjection.from_mapping(projection_spec, binding=source_id)
        validate_against_schema(
            projection, _declared_schema(spec, binding=source_id), binding=source_id
        )
        _refuse_an_exact_rung_with_no_lookup(config, projection, binding=source_id)

        database_block = spec.get("database")
        if self._injected_database is not None:
            if database_block is not None:
                logger.info(
                    "Binding %r to the injected handle; its `database:` block is "
                    "not resolved because a handle already built needs no name.",
                    source_id,
                )
            database = self._injected_database
            forms_database = self._injected_database
            backend = type(database).__name__
        else:
            if not isinstance(database_block, Mapping):
                raise ValidationError(
                    f"binding {source_id!r} declares no `database:`, and no handle "
                    f"was injected. A live source is read through one or the "
                    f"other: a `$resource` this registry's environment resolves, "
                    f"or a handle handed to `from_components`",
                    context={"source_id": source_id},
                )
            backend = str(database_block.get("backend", "")) or "unknown"
            database = await self._database_handle(dict(database_block), projection.table)
            forms_database = (
                await self._database_handle(dict(database_block), projection.surface_forms.table)
                if projection.surface_forms is not None
                else database
            )
        return RecordEntitySource(
            database,
            projection,
            source_id=source_id,
            backend=backend,
            forms_database=forms_database,
        )

    async def _database_handle(self, block: dict[str, Any], table: str) -> AsyncDatabase:
        """A handle for one table of a resolved ``database:`` block, built once.

        **Off the event loop**, because opening a backend blocks: the backend
        registry resolves a name by *importing* the module that implements it,
        and a backend's own construction may create a directory or open a
        file. Measured rather than assumed -- the runtime detector catches the
        import, and catches it only on the first call in a process, which is
        the order-dependence that makes this the kind of stall review does not
        find.

        **Whether a handle is one table's is the backend's answer, not this
        method's**, and it is asked rather than assumed: the three SQL backends
        declare a ``table`` on their config, so a binding naming two tables
        needs two handles there; ``memory``, ``file``, ``s3`` and
        ``elasticsearch`` declare none, because for them the handle *is* the
        store and rows of both kinds live in it. A backend that names its unit
        something else -- Elasticsearch's index -- is given a second one by
        naming a second ``$resource``, which is the general escape and needs no
        special case here.

        Built once per distinct block-and-table. A second binding naming the
        same one gets neither a second handle nor a second connection, and a
        reload against an unchanged environment resolves to the same block and
        so reuses what is open -- which is what keeps a reload loop from
        accumulating connections that only :meth:`close` would release.
        """
        return await asyncio.to_thread(self._open_database, block, table)

    def _open_database(self, block: dict[str, Any], table: str) -> AsyncDatabase:
        """The blocking half of :meth:`_database_handle`, run in a worker thread."""
        if _backend_addresses_one_table(block):
            block["table"] = table
        key = json.dumps(block, sort_keys=True, default=str)
        cached = self._database_cache.get(key)
        if cached is not None:
            return cached
        handle: AsyncDatabase = async_database_factory.create(**block)
        self._database_cache[key] = handle
        self._handles.append((handle, True))
        return handle

    async def _publish_departure(self, ontology: AsyncOntology[str]) -> None:
        """Announce a vocabulary leaving, however it is leaving.

        One helper, because an unload and a replacement are the same departure
        to a subscriber and a second spelling of the payload is how the two
        come to carry different ones.
        """
        await self._publish(
            topic=f"ontology:{ontology.id}",
            event_type=EventType.DELETED,
            payload={"ontology_id": ontology.id, **self._departing(ontology)},
        )

    def _departing(self, ontology: AsyncOntology[str]) -> dict[str, Any]:
        """The departure payload's id half, in whichever of its two forms applies."""
        if all(description.backend == AUTHORED_SOURCE_ID for description in ontology.describes):
            parts = self._parts.get(ontology.id)
            declared = sorted(parts.declared_entities) if parts else []
            return {"entity_ids": [ontology.qualify(local) for local in declared]}
        return {"entity_id_prefix": f"{ontology.id}:"}

    async def _publish_taxonomy_delta(
        self, ontology_id: str, before: OntologyParts, after: OntologyParts
    ) -> None:
        """Announce what a replacement changed, in three sets rather than two.

        The third set is the point: an entity that **changed name while keeping
        its id** is in neither *gone* nor *arrived*, so a two-set delta reports
        a rename as no change at all. An empty payload therefore means a
        genuine no-op.
        """
        was, now = before.declared_entities, after.declared_entities
        renamed = sorted(
            entity_id
            for entity_id, entity in now.items()
            if entity_id in was and was[entity_id].name != entity.name
        )
        payload = {
            "ontology_id": ontology_id,
            "gone": sorted(set(was) - set(now)),
            "arrived": sorted(set(now) - set(was)),
            "renamed": renamed,
        }
        for taxonomy_id in after.taxonomies:
            await self._publish(
                topic=f"taxonomy:{taxonomy_id}",
                event_type=EventType.UPDATED,
                payload={**payload, "taxonomy_id": taxonomy_id},
            )

    async def _publish(self, *, topic: str, event_type: EventType, payload: dict[str, Any]) -> None:
        """One event, to the bus if there is one.

        The prose name of each of these -- ``ontology.loaded``,
        ``ontology.unloaded``, ``taxonomy.rebuilt`` -- describes the **pair** of
        a topic and a type. Nothing is added to ``EventType``, whose members
        are used as published.

        **Ordering is the bus's, not this registry's.** Each backend's
        guarantee is the one a subscriber gets; nothing here re-orders or
        buffers to make a weaker bus look like a stronger one.
        """
        bus = self._event_bus
        if bus is None:
            return
        await bus.publish(topic, Event(type=event_type, topic=topic, payload=payload))

    async def _event_bus_from(self, block: Mapping[str, Any]) -> None:
        """Build the bus a document configured, and own it.

        An injected bus always wins and is never closed by this registry; one
        built here is closed by :meth:`close`, because the registry built it.
        """
        if self._event_bus is not None:
            return
        bus = create_event_bus(dict(block))
        await bus.connect()
        self._event_bus = bus
        self._owns_event_bus = True


def _backend_addresses_one_table(block: Mapping[str, Any]) -> bool:
    """Whether this backend's config declares a ``table`` -- i.e. is one table's.

    Asked of the registered backend's own config class rather than decided
    from a list here, so a consumer-registered backend answers for itself and
    a backend added to this package answers without an edit in this file.
    """
    import dataclasses

    from dataknobs_data.backend_selection import normalize_backend
    from dataknobs_data.backends import async_backends

    declared = block.get("backend")
    if not declared:
        return False
    factory = async_backends.get_factory(normalize_backend(declared))
    config_cls = getattr(factory, "CONFIG_CLS", None)
    if config_cls is None or not dataclasses.is_dataclass(config_cls):
        return False
    return any(field.name == "table" for field in dataclasses.fields(config_cls))


def _event_bus_block(
    config: OntologyConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """An ``event_bus:`` block carried by a mapping handed to a constructor."""
    if isinstance(config, Mapping):
        block = config.get("event_bus")
        if isinstance(block, Mapping):
            return dict(block)
    return None


def _declared_schema(spec: Mapping[str, Any], *, binding: str) -> DatabaseSchema | None:
    """The binding's declared ``schema:``, as a :class:`DatabaseSchema`.

    The published form is a list of ``{name, type}`` rows, which is what a
    person writes in a config file. ``None`` where the binding declared none --
    the refusal for that belongs with the rest of the projection's validation
    rather than here.
    """
    declared = spec.get("schema")
    if declared is None:
        return None
    if not isinstance(declared, list):
        raise ValidationError(
            f"binding {binding!r}: `schema:` takes a list of `{{name, type}}` "
            f"rows, got {type(declared).__name__}",
            context={"source_id": binding},
        )
    schema = DatabaseSchema()
    for row in declared:
        if not isinstance(row, Mapping) or "name" not in row:
            raise ValidationError(
                f"binding {binding!r}: every `schema:` row names a column -- "
                f"`{{name: <column>, type: <type>}}` -- and this one is {row!r}",
                context={"source_id": binding, "row": row},
            )
        schema.add_field(
            FieldSchema(name=str(row["name"]), type=_field_type(row.get("type"), binding=binding))
        )
    return schema


def _field_type(declared: Any, *, binding: str) -> FieldType:
    """A declared column type, or a refusal listing the ones there are."""
    if declared is None:
        return FieldType.STRING
    try:
        return FieldType(str(declared))
    except ValueError as exc:
        raise ValidationError(
            f"binding {binding!r}: `schema:` declares type {declared!r}, which is "
            f"not a field type. Declared types: "
            f"{sorted(member.value for member in FieldType)}",
            context={"source_id": binding, "type": declared},
        ) from exc


def _refuse_an_exact_rung_with_no_lookup(
    config: OntologyConfig, projection: EntityProjection, *, binding: str
) -> None:
    """Refuse an ``exact`` rung over a binding that declares no folded lookup.

    The declared-schema rule's shape, for a second thing the loader cannot
    introspect. The first
    rung of a cascade matches surface forms, the **source** folds, and a live
    table holds the form as it was written -- so a binding under an ``exact``
    rung either declares where its folded forms live or it is rejected here,
    at load, naming the key. A binding with no exact rung over it needs none
    and loads unchanged.
    """
    if projection.surface_forms is not None:
        return
    section = config.resolver or {}
    rungs = section.get("rungs") or ()
    if not any(
        isinstance(rung, Mapping) and str(rung.get("kind", "")) == "exact" for rung in rungs
    ):
        return
    raise ValidationError(
        f"binding {binding!r} is read by an `exact` rung and declares no "
        f"`surface_forms:`. That rung matches a surface form the person typed "
        f"against forms the source folds, and a live table holds the form as it "
        f"was written -- no engine folds at query time the way `str.casefold` "
        f"does. Declare the lookup: `surface_forms: {{table: <table>, form: "
        f"<column holding the folded form>, entity: <column holding this "
        f"projection's id>}}`",
        context={"source_id": binding},
    )


__all__ = ["OntologyRegistry"]

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
import dataclasses
import json
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Self

from dataknobs_common.entity_resolution import async_signal_backends
from dataknobs_common.events import Event, EventType, create_event_bus
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.lifecycle import close_if_owned
from dataknobs_common.hierarchy import AsyncEnumerableHierarchy
from dataknobs_common.ontology import (
    AUTHORED_SOURCE_ID,
    AUTHORED_SOURCE_KINDS,
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    AsyncOntology,
    Entity,
    OntologyConfig,
    assemble_async_ontology,
    async_build_resolver,
    build_ontology,
    refuse_unbuildable_rungs,
    split_qualified,
)
from dataknobs_common.index import AliasSource, AsyncIndexSource
from dataknobs_common.ontology.index_source import EntitySourceIndexSource
from dataknobs_common.ontology.model import DK_ENTITY_TYPE, DK_RELATION_TYPE
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig

# Imported for its registration, which is the whole of what this line is for:
# the module puts `kind: semantic` in `dataknobs_common`'s asynchronous
# registry, and `_resolver_from` below builds a cascade a document may name it
# in. A consumer reaching the rung through this registry therefore never
# imports it by hand, which is the half of the `authority` precedent that does
# not apply here -- that rung's registering module is imported by the
# application because no door in this repository builds an authority stack.
import dataknobs_data.entity_resolution  # noqa: F401
from dataknobs_data.backend_selection import normalize_backend
from dataknobs_data.backends import async_backends
from dataknobs_data.factory import async_database_factory
from dataknobs_data.fields import FieldType
from dataknobs_data.ontology.hierarchy import (
    COLUMN_AXIS_KIND,
    ColumnAxisBinding,
    ColumnHierarchy,
)
from dataknobs_data.ontology.sources import (
    RECORD_SOURCE_KIND,
    EntityProjection,
    RecordEntitySource,
    refuse_undeclared_columns,
    validate_against_schema,
)
from dataknobs_data.schema import DatabaseSchema, FieldSchema
from dataknobs_data.vector.semantic_index import SemanticIndex
from dataknobs_data.vector.stores.factory import VectorStoreFactory
from dataknobs_data.vector.types import DistanceMetric

#: Every key the ``index:`` section reads.
#:
#: Declared rather than left implicit so an unread key is refused instead of
#: accepted-and-ignored: ``_index_from`` read three names off the block and
#: dropped the rest, which made ``metirc:`` a configuration the registry took,
#: reported success over, and acted on in no way. The same silent-drop failure
#: the ``embedder:`` refusal in that method exists to prevent.
INDEX_BLOCK_KEYS = frozenset({"store", "embedder", "metric", "fields", "join", "aliases"})

#: The keys a ``resolver:`` section is read for, and the only ones.
#:
#: :data:`INDEX_BLOCK_KEYS`'s argument, over a section with **one** key. A key
#: nothing acts on is configuration a consumer wrote and this registry
#: accepted while ignoring -- and here it is worse than inert, because
#: ``rungs`` absent is read one layer down as *a composition of nothing*
#: rather than as silence. So ``resolver: {rung: [...]}``, singular, would
#: build a cascade that matches nothing and report success.
RESOLVER_BLOCK_KEYS = frozenset({"rungs"})

if TYPE_CHECKING:
    from types import TracebackType

    from dataknobs_common.entity_resolution.protocols import AsyncEntityResolver
    from dataknobs_common.events import EventBus
    from dataknobs_common.hierarchy import AsyncHierarchy
    from dataknobs_common.ontology import OntologyParts, SourceDescription

    from dataknobs_data.database import AsyncDatabase
    from dataknobs_data.query import Filter
    from dataknobs_data.vector.embedding import TextEmbedder
    from dataknobs_data.vector.stores.base import VectorStore

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

    **The collaborators here are ones a caller *may* inject, and the mixin
    has a field for that.** Neither is required: a configured registry
    resolves its own database from ``$resource`` and publishes nothing when
    no bus is wired. Both are nonetheless real injection points that
    ``from_components`` takes, so tooling asking what this class accepts has
    to be told about them -- which is
    :attr:`~dataknobs_common.structured_config.StructuredConfigConsumer.OPTIONAL_COMPONENTS`,
    read by ``accepted_components()``.

    ``EXPECTED_COMPONENTS`` is the wrong field for them, and the cost of
    using it would not be a documentation one: it means *must be supplied*
    and feeds ``missing_components()`` and ``require_components()``, so a
    fully loaded registry with nothing wrong with it would answer
    ``{"database", "event_bus"}`` to the first and raise from the second.
    This class is the consumer ``OPTIONAL_COMPONENTS`` was added for, which
    is why the reading above is worth stating rather than assuming.

    **Settings are not collaborators, and both arrive through one channel.**
    Every door has the shape ``(config, **components)``, so a caller writing
    ``from_config_async(resolved, environment="production")`` sends a
    *setting* down the collaborator channel. It is taken back out in
    ``__init__`` -- see :attr:`CONSTRUCTION_SETTINGS` for the four and for why
    the lift is there rather than in each door.
    """

    CONFIG_CLS: ClassVar[type[OntologyConfig]] = OntologyConfig

    #: The collaborators a caller may inject, none of which is required:
    #: a configured registry resolves its own database from ``$resource`` and
    #: publishes nothing when no bus is wired. Declared here rather than under
    #: ``EXPECTED_COMPONENTS`` so that :meth:`missing_components` answers
    #: about this registry rather than about the field's other reading.
    #:
    #: ``forms_database`` is the second handle a binding needs when its
    #: projection's ``surface_forms:`` names a different table *and* the
    #: backend addresses one table at a time. The configured door opens that
    #: handle itself; the injected door cannot, so it is offered here rather
    #: than leaving ``from_components`` unable to express a document
    #: ``from_config_async`` accepts.
    #:
    #: ``embedder`` is the one collaborator the configured door **cannot**
    #: open, and it is here rather than resolved for a reason that is a fact
    #: about the packages rather than a preference. An ``index:`` block's
    #: ``embedder:`` resolves to a provider-and-model pair, and the only
    #: construct in the workspace that accepts one lives in
    #: ``dataknobs-llm`` --- which declares ``dataknobs-data`` as a
    #: dependency, so the edge runs the other way. Nothing here can build one.
    #:
    #: The asymmetry with the store half is real rather than an inconsistency:
    #: the stores have a registry-shaped door inside this package and the
    #: embedders have none and cannot have one without crossing that edge. And
    #: an embedder has no ``close()`` at all --- the socket is one level below,
    #: on the provider it holds --- so a registry that *built* one would claim
    #: a responsibility it could not discharge, and would say so only at
    #: DEBUG. Injection puts the lifecycle back with the only object that can
    #: end it.
    #:
    #: A document declaring ``embedder:`` with nothing injected is therefore
    #: **refused at load**, naming what to inject. Refused rather than
    #: dropped: a section parsed into a field and discarded with no error is
    #: the failure this registry has already been caught in once.
    OPTIONAL_COMPONENTS: ClassVar[frozenset[str]] = frozenset(
        {"database", "embedder", "event_bus", "forms_database"}
    )

    #: The live source kinds this registry binds.
    #:
    #: Computed from the declared kind alone, exactly as
    #: ``dataknobs_common.ontology.loader``'s refusal is, so one config means
    #: one thing in every environment: a kind outside this set and outside
    #: ``AUTHORED_SOURCE_KINDS`` is refused by name whether or not some other
    #: package that could bind it happens to be importable.
    LIVE_SOURCE_KINDS: ClassVar[frozenset[str]] = frozenset({RECORD_SOURCE_KIND})

    #: The structure-axis kinds this registry binds.
    #:
    #: :attr:`LIVE_SOURCE_KINDS` one section over, for the other half of a
    #: document. A ``taxonomies:`` row declaring no ``kind:`` asks for the
    #: assertion read, which ``dataknobs_common`` builds and this registry does
    #: not re-implement; a row declaring one is asking for a backing, and this
    #: is the set of backings there are. Computed from the declared kind alone,
    #: for the source set's reason.
    LIVE_AXIS_KINDS: ClassVar[frozenset[str]] = frozenset({COLUMN_AXIS_KIND})

    #: This registry's own construction settings, as opposed to the
    #: collaborators :attr:`OPTIONAL_COMPONENTS` names.
    #:
    #: **Why the distinction needs a name.** Every published door has the
    #: shape ``(config, **components)`` and puts everything that is not the
    #: config into the component channel, so ``from_config_async(resolved,
    #: environment="production")`` lands the environment on
    #: ``self.components`` where nothing reads it. These four are lifted back
    #: out in ``__init__``, which is the one place all four doors -- the three
    #: the mixin publishes and direct construction -- funnel through. A door
    #: override per settings would be three copies of the same sorting, and
    #: the one that got missed would be the silent one.
    #:
    #: Kept in step with the keyword-only parameters below by
    #: ``test_the_registrys_own_settings_are_declared_once``: two spellings of
    #: one fact drift, and this drift is quiet in the direction that matters
    #: -- a parameter added here and not there is a setting the doors swallow.
    CONSTRUCTION_SETTINGS: ClassVar[frozenset[str]] = frozenset(
        {"environment", "strict_resources", "config_key", "normalizer"}
    )

    def __init__(
        self,
        config: OntologyConfig | Mapping[str, Any] | None = None,
        *,
        environment: EnvironmentConfig | str | None = None,
        strict_resources: bool | None = True,
        config_key: str = "ontology",
        normalizer: Callable[[str], str] | None = None,
        _components: Mapping[str, Any] | None = None,
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
            normalizer: How a surface form is folded, for every source this
                registry binds -- the authored one and the live one alike.
                Defaults to
                :func:`~dataknobs_common.text.default_normalizer`. It is the
                same parameter the module-level doors carry and the same one
                :class:`~dataknobs_data.ontology.RecordEntitySource` takes, so
                a consumer who folded their lookup table with their own
                callable hands it here once instead of to each half
            _components: The mixin's collaborator channel. Named rather than
                left to ``**kwargs`` because the four settings above are
                lifted out of it -- see :attr:`CONSTRUCTION_SETTINGS`

        Note:
            Every keyword above may also arrive through a published door, as
            ``from_config_async(resolved, environment=...)``. The doors put
            everything that is not the config into ``_components``, so that is
            where such a keyword lands, and this constructor takes it back
            out. The channel wins where both carry a name, and that ordering
            decides nothing: a door has no parameter to pass directly with,
            and a direct call has no reason to reach for the mixin's internal
            channel, so the two are never populated at once.
        """
        settings, collaborators = self._split_settings(_components)
        environment = settings.get("environment", environment)
        strict_resources = settings.get("strict_resources", strict_resources)
        config_key = settings.get("config_key", config_key)
        normalizer = settings.get("normalizer", normalizer)
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
        self._normalizer = normalizer
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
        # The same arrangement for the other family of handle. Keyed on the
        # resolved `store:` block, for `_database_cache`'s reasons.
        self._vector_store_cache: dict[str, VectorStore] = {}
        # Every loaded ontology's axes and what each held when it was built.
        # Recorded here rather than re-read at the next load, because a live
        # axis re-read then would answer with the rows as they are now on both
        # sides of the comparison -- see `_axis_populations`.
        self._populations: dict[str, dict[str, frozenset[str] | None]] = {}
        # Which store each loaded ontology's live binding took, and the tables
        # it named there. Read to refuse a second binding over one store; not
        # pruned on `unload`, because a claim is only consulted while its
        # ontology is loaded and `_ontologies` is where that is recorded.
        self._store_claims: dict[str, _StoreClaim] = {}
        # One handle opened at a time, so that the cache lookup and the open
        # it guards are one step. Held on the loop rather than in the worker
        # thread the open is offloaded to -- see `_database_handle`.
        self._handle_lock = asyncio.Lock()
        # Every loaded ontology's semantic index, where its document declared
        # one. Dropped at `unload` beside the other per-id things; the store
        # behind it is released by `close()` and not here, for `unload`'s own
        # stated reason -- a value handed out by `index()` outlives its entry.
        self._indexes: dict[str, SemanticIndex] = {}
        # Every loaded ontology's configured cascade, where its document
        # declared a `resolver:` section. Held and dropped exactly as the
        # indexes above are, and for the same reasons -- it is a value built
        # at load, it opens nothing of its own, and a caller holding one
        # outlives its entry.
        self._resolvers: dict[str, AsyncEntityResolver] = {}
        self._injected_database: AsyncDatabase | None = None
        self._injected_forms_database: AsyncDatabase | None = None
        self._injected_embedder: TextEmbedder | None = None
        self._event_bus: EventBus | None = None
        self._owns_event_bus = False
        # A registry may hold no document at all -- `OntologyRegistry()` then
        # `await registry.load(stored)` is the shape a deployment reading from
        # a backend has. The mixin requires *a* config, and `id` is the one
        # field an ontology cannot do without, so the empty id is what "none
        # was configured" is spelled as: no document can claim it, which is
        # exactly what makes it available to mean this.
        super().__init__(
            OntologyConfig(id="") if config is None else config,
            _components=collaborators or None,
            **kwargs,
        )

    @classmethod
    def _split_settings(
        cls, components: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The component channel, split into this registry's settings and the rest.

        The settings are *removed* rather than copied out. A setting left in
        the channel is on ``self.components``, which is what
        :meth:`~dataknobs_common.structured_config.StructuredConfigConsumer.forwardable_components`
        hands to a child consumer -- and ``strict_resources=False`` arriving
        at some other object's constructor is a worse failure than the one
        this lift exists to fix.
        """
        supplied = dict(components or {})
        settings = {
            name: supplied.pop(name) for name in list(supplied) if name in cls.CONSTRUCTION_SETTINGS
        }
        return settings, supplied

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
        forms_database = self.components.get("forms_database")
        if forms_database is not None:
            self._injected_forms_database = forms_database
            self._handles.append((forms_database, False))
        # Recorded, never appended to `_handles`: an embedder is not a handle.
        # It has no `close()` -- the three members are `dimensions`,
        # `model_id` and `embed` -- so recording ownership of one would record
        # a responsibility nothing here can discharge.
        embedder = self.components.get("embedder")
        if embedder is not None:
            self._injected_embedder = embedder
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
        await self._load_resolved(self._config)

    # ------------------------------------------------------------ the config

    @staticmethod
    def get_portable_config(cfg: Any) -> dict[str, Any]:
        """The storable form: ``$resource`` refs intact, ``${VAR}`` unexpanded.

        What a deployment puts in a backend, and what :meth:`load` reads back.
        An :class:`~dataknobs_config.EnvironmentAwareConfig` answers it
        directly; a mapping is already portable and passes through, copied so
        the caller's dict is not the registry's.

        Raises:
            TypeError: For anything else. A ``TypeError`` rather than the
                ``ValidationError`` every refusal in :meth:`load` is: this
                takes a *holder* of a config rather than a config, so a
                wrong argument here is a call that cannot be made rather
                than a document that cannot be accepted
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
            raise ValidationError(
                "load() with no argument loads the document this registry was "
                "constructed with, and this one was constructed with none. Pass "
                "the document, or construct the registry with it",
                context={"ontology_id": None},
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
        self._populations.pop(ontology_id, None)
        self._indexes.pop(ontology_id, None)
        self._resolvers.pop(ontology_id, None)
        return True

    async def reload(self, ontology_id: str) -> AsyncOntology[str]:
        """Re-resolve the document this id was loaded from, against today's environment.

        ``load(stored, replace=True)`` over what this registry stored, and
        nothing more -- a replacement is what a rebuild *is*, so this is the
        one that was already written rather than a second path through it.
        What it buys over holding the vocabulary is the resolution: an
        environment whose ``$resource`` now names a different backend, or a
        ``${VAR}`` that has moved, reaches the sources here.

        **Only :meth:`load` stores a document**, because only :meth:`load` is
        handed one: the configured door takes the *resolved* form, which is
        what this method's whole value is measured against. So an id loaded
        through that door is refused here, and the refusal names that
        condition rather than reporting the vocabulary absent -- it is
        loaded, and :meth:`list_ids` says so.

        Raises:
            KeyError: When no ontology with this id is loaded, and when one
                is but came from the constructor's already-resolved config
        """
        document = self._documents.get(ontology_id)
        if document is None:
            raise KeyError(self._unreloadable(ontology_id))
        return await self.load(document, replace=True)

    def _unreloadable(self, ontology_id: str) -> str:
        """Which of the two reasons this id has no document to re-resolve."""
        if ontology_id in self._ontologies:
            return (
                f"ontology {ontology_id!r} is loaded and cannot be reloaded: it came "
                f"from the already-resolved config this registry was constructed "
                f"with, so there is no portable document here to re-resolve against "
                f"today's environment. Pass the stored document to "
                f"load(document, replace=True), which is the door reload() is "
                f"written over"
            )
        return f"No ontology loaded with id {ontology_id!r}"

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

    def index(self, ontology_id: str) -> SemanticIndex | None:
        """The semantic index this ontology declared, or ``None``.

        ``None`` where the document declared no ``index:`` section, and that
        is a configuration answer rather than an error: an ontology that
        declared none answers ``None`` for good, so a caller writes one branch
        either way and only the reason differs.

        It **returns** what ``load()`` built and is not a second way to build
        one. A caller who loaded no ``index:`` section gets ``None`` here no
        matter how many stores they hold.

        **The index is built, not connected, on every read.** The store behind
        it was opened at load and is released by :meth:`close`; this member
        hands back a value and opens nothing.

        **After ``close()`` this still answers the index**, and the store it
        holds is shut. That is :meth:`close`'s documented philosophy rather
        than an oversight --- the values stay reachable and what is gone is
        the ability to read through them --- but the sentence above says the
        store is released without saying the index outlives the release. A
        caller holding one across a ``close()`` gets a live object over a dead
        store, and the failure arrives from the store.

        Args:
            ontology_id: Which loaded vocabulary's index to return.

        Returns:
            The index, or ``None`` where this registry built none for that id.
        """
        return self._indexes.get(ontology_id)

    def resolver(self, ontology_id: str) -> AsyncEntityResolver | None:
        """The configured placement cascade for this ontology, or ``None``.

        ``None`` where the document declared no ``resolver:`` section, which
        is :meth:`index`'s answer to the same question and is a configuration
        answer rather than an error: silence is *use your own composition*,
        and this registry does not invent one. A document that wrote the
        section gets the cascade it wrote.

        It **returns** what ``load()`` built and is not a second way to build
        one, exactly as :meth:`index` does -- which is also why the two land
        together: the rung that searches an index is constructed over the
        index this registry just built, so a second builder would have to
        open a second store.

        **What it holds is the caller's to keep alive.** The cascade opens
        nothing and closes nothing; a semantic rung inside it holds this
        registry's index, so a resolver held across a :meth:`close` inherits
        what :meth:`index` records about an index held across one --- a live
        object over a dead store, one object further out.

        Args:
            ontology_id: Which loaded vocabulary's cascade to return.

        Returns:
            The cascade, or ``None`` where that document declared none.
        """
        return self._resolvers.get(ontology_id)

    async def close(self) -> None:
        """Release every handle this registry opened, and nothing it was handed.

        A cascade, so one failing handle does not abort the rest: each close is
        error-isolated and logged. Injected handles are left untouched for
        their owner.

        **It does not unload.** The ids stay listed and the values stay
        reachable; what is gone is the ability to read through them. A caller
        who wants the departure events calls :meth:`unload` first.

        **Idempotent, and that is what the resets below are for**: what this
        registry owned is closed and *forgotten*, what it was handed is
        untouched and still recorded. A second call therefore closes nothing
        twice, and an :meth:`unload` after a close still announces its
        departure when the bus is one this registry did not close -- which is
        the case where the bus is still there to hear it.

        **The lists are swapped before the first await**, so a load still in
        flight appends its handle to the list this close is no longer walking.
        That handle is the next close's rather than nobody's, which is the
        one outcome a registry whose thesis is *what it opened it closes* may
        not have.
        """
        held, self._handles = self._handles, [entry for entry in self._handles if not entry[1]]
        self._database_cache = {}
        self._vector_store_cache = {}
        # The claims name handles that are about to be closed, so they outlive
        # nothing. A load after a close opens a fresh handle and takes a fresh
        # claim, which is the arrangement that keeps the two in step.
        self._store_claims = {}
        bus, owns_bus = self._event_bus, self._owns_event_bus
        if owns_bus:
            self._event_bus = None
            self._owns_event_bus = False
        for handle, owned in held:
            await close_if_owned(handle, owned, on_error=self._teardown_failed)
        await close_if_owned(bus, owns_bus, on_error=self._teardown_failed)

    async def __aenter__(self) -> Self:
        """Enter a block whose exit is :meth:`close`.

        **Entry builds nothing, and that is the difference from
        :class:`~dataknobs_data.database.AsyncDatabase`'s block.** A handle
        connects on entry because a handle is the thing being opened; a
        registry opens handles at :meth:`load`, when a document says which
        ones, so there is nothing here to pair the exit with. Entry hands back
        the registry and the caller loads inside the block.

        **What the block buys is the path nobody writes.** ``close()`` in a
        ``finally`` releases handles on the paths its author thought about;
        the exit below releases them on the other one too, which is the whole
        of why a lifecycle owner publishes a block rather than documenting a
        call.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Leave the block by closing, however the block ended.

        :meth:`close` exactly -- so the block draws the owned-versus-injected
        line in the same place, and does not :meth:`unload`. A caller who
        wants the departure events published calls ``unload`` inside the
        block, where there is still a bus to publish them to.

        Returns ``None`` rather than ``False``, which suppresses nothing: an
        exception raised inside the block propagates, with the handles
        already released.
        """
        await self.close()

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
        raise ValidationError(
            f"load: `config` must be an OntologyConfig or a Mapping, got {type(config).__name__}",
            context={"config_type": type(config).__name__},
        )

    async def _ensure_environment(self) -> None:
        """Load the environment a name stands for, once, off the event loop.

        The only blocking thing this registry does is read a YAML file, so it
        is the only thing offloaded -- and only when there is a read: a caller
        who handed over an ``EnvironmentConfig`` object, or none at all, pays
        for no thread. Same shape, and the same reason, as the one read
        ``async_load_ontology`` offloads.

        **The class it loads through is imported at module scope**, which
        matters more in this method than in the other one that reaches for it:
        an ``import`` executed inside an ``async def`` is disk I/O on the event
        loop the first time it runs, so an in-body import here is either
        deferring nothing or blocking the loop, and only a measurement says
        which. It defers nothing. This module imports ``dataknobs_data.factory``
        at module scope, that import puts ``dataknobs_config`` in
        ``sys.modules`` before this file finishes loading, and the package binds
        both names eagerly with no module ``__getattr__`` -- so the statement
        was a dictionary lookup wearing the shape of a deferral, and the shape
        is what the next reader would have preserved.
        """
        if self._environment is not None or self._environment_name is None:
            return
        self._environment = await asyncio.to_thread(EnvironmentConfig.load, self._environment_name)

    def _resolve(self, section: Mapping[str, Any]) -> dict[str, Any]:
        """Late-bind ``$resource`` and ``${VAR}`` against this registry's environment.

        Through the constructor rather than
        :meth:`~dataknobs_config.EnvironmentAwareConfig.from_dict`, because
        this registry holds an ``EnvironmentConfig`` **object** and that
        classmethod takes an environment *name* and loads one.
        """
        if self._environment is None:
            return dict(section)
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
        """Build a vocabulary from an already-resolved document, and announce it.

        **Where the ``event_bus:``, ``index:`` and ``resolver:`` blocks are
        read, and the only place.** Both doors arrive here -- the constructor's through
        :meth:`_ainit` and :meth:`load`'s after resolution -- so reading them
        here is what makes the two agree about where a block lives. Read off
        the *typed* config rather than the mapping beside it, because a
        published door coerces before this object sees anything and a key the
        type does not declare does not survive that.

        Before the announcement below rather than after, which is the whole
        point of building it here: a document that configures a bus and loads
        a vocabulary expects the arrival of that vocabulary on that bus.
        """
        typed = (
            config if isinstance(config, OntologyConfig) else OntologyConfig.from_dict(dict(config))
        )
        if typed.event_bus is not None:
            await self._event_bus_from(typed.event_bus, ontology_id=typed.id)
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
        was_populated = self._populations.get(parts.id, {})
        bound = await self._bind_sources(typed, parts)
        ontology = await assemble_async_ontology(
            parts,
            entities=bound.entities,
            assertions=AsyncMappingAssertionSource(parts.declared_assertions),
            describes=bound.describes,
            structures=await self._bind_taxonomies(parts, bound.live),
        )
        now_populated = await self._axis_populations(ontology)
        # After the vocabulary is assembled, because the index is over it:
        # the adapter takes the ontology, so there is nothing to build until
        # there is one. Before the announcement, for the bus's reason -- a
        # subscriber reading the registry on the arrival event finds what
        # arrived, index included.
        # Before the store is opened, which is the ordering `_index_from`
        # states for its own refusals and the one this reader was outside of:
        # every way a `resolver:` section can be unreadable is a property of
        # the document and the registry, and refusing after a handle is open
        # strands it -- `from_config_async` has not returned the object whose
        # `close()` would release it.
        self._refuse_an_unbuildable_resolver_section(typed)
        index = await self._index_from(typed, ontology)
        # After the index, because a semantic rung is constructed over the one
        # that call just built -- which is also why the two readers cannot
        # land in different releases. Before the announcement, for the index's
        # reason: a subscriber reading the registry on the arrival event finds
        # what arrived, cascade included.
        resolved = await self._resolver_from(typed, ontology, index)
        # Announced before the swap, so a subscriber reading the registry on
        # the departure event still sees what departed -- and in this order,
        # rather than as a third event meaning both, because a subscriber who
        # cares about departures and one who cares about arrivals each already
        # have the event they read.
        if replaced is not None:
            await self._publish_departure(replaced)
        self._ontologies[parts.id] = ontology
        self._parts[parts.id] = parts
        self._populations[parts.id] = now_populated
        if index is None:
            self._indexes.pop(parts.id, None)
        else:
            self._indexes[parts.id] = index
        if resolved is None:
            self._resolvers.pop(parts.id, None)
        else:
            self._resolvers[parts.id] = resolved
        await self._publish(
            topic=f"ontology:{parts.id}",
            event_type=EventType.CREATED,
            payload={
                "ontology_id": parts.id,
                "version": parts.version,
                "sources": [description.source_id for description in bound.describes],
            },
        )
        if before is not None:
            await self._publish_taxonomy_delta(
                parts.id, before, parts, was_populated, now_populated
            )
        return ontology

    async def _bind_sources(self, config: OntologyConfig, parts: OntologyParts) -> _BoundSources:
        """Dispatch each declared source kind, and refuse the ones nothing binds.

        Two refusals compose here, and they answer different questions from the
        one ``dataknobs_common``'s loader answers. That refusal is *this door
        cannot own a lifecycle* and is correct forever; these are *no binder is
        registered for this kind* and *the construct that would route between
        two sources does not exist yet*.

        **It reports what it bound as well as what it built**, because a
        ``kind: column`` axis reads the same table through the same handle and
        must not open a second one. A third member rather than a wider tuple:
        the caller unpacks three things with three names, and
        :meth:`_bind_taxonomies` takes exactly one of them.
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
            authored = AsyncMappingEntitySource(
                parts.declared_entities, normalizer=self._normalizer
            )
            return _BoundSources(authored, (authored.describe(),), {})
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
        source, binding = await self._bind_record_source(
            config, live[0], ontology_id=parts.id, entity_types=parts.entity_types.keys()
        )
        return _BoundSources(source, (source.describe(),), {binding.source_id: binding})

    async def _bind_record_source(
        self,
        config: OntologyConfig,
        spec: Mapping[str, Any],
        *,
        ontology_id: str,
        entity_types: Collection[str],
    ) -> tuple[RecordEntitySource, _LiveBinding]:
        """One ``kind: record`` binding, validated before a handle is opened.

        ``ontology_id`` is carried because the store this binding takes is
        claimed on the ontology's behalf rather than the binding's: it is
        released when the ontology is unloaded, and a document replacing
        itself must not collide with the copy it replaces.

        ``entity_types`` is carried for
        :func:`_refuse_an_undeclared_projection_type`, which is this package's
        member of the reference family ``build_ontology`` refuses eight of.

        **The pair rather than the source alone**, because an axis over the
        same table needs the three things this method resolved and the source
        does not publish: the handle it opened, the projection it parsed, and
        the schema it validated against. Handing them back is what keeps
        :meth:`_bind_taxonomies` from resolving any of them a second time --
        and a second resolution of the handle in particular would be a second
        handle, which is the one thing teardown cannot absorb.
        """
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
        if not config.imports:
            # `imports:` switches the core's eight reference checks off, for
            # the reason `_owns_its_sections` gives: an import is carried and
            # never followed, so a name this document does not declare may be
            # one the import declares. This is the ninth, so it switches off
            # with them or the family disagrees with itself.
            _refuse_an_undeclared_projection_type(projection, entity_types, binding=source_id)
        declared_schema = _declared_schema(spec, binding=source_id)
        validate_against_schema(projection, declared_schema, binding=source_id)
        _refuse_a_form_reading_rung_with_no_lookup(config, projection, binding=source_id)
        _refuse_a_scan_over_a_binding_that_cannot_bound_it(config, projection, binding=source_id)

        database_block = spec.get("database")
        if self._injected_database is not None:
            if database_block is not None:
                logger.info(
                    "Binding %r to the injected handle; its `database:` block is "
                    "not resolved because a handle already built needs no name.",
                    source_id,
                )
            database = self._injected_database
            forms_database = self._injected_forms_database or self._injected_database
            if forms_database is database:
                _refuse_one_handle_for_two_tables(database, projection, binding=source_id)
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
        tables = (
            (projection.table,)
            if projection.surface_forms is None
            else (projection.table, projection.surface_forms.table)
        )
        _refuse_a_second_binding_over_one_store(
            self._store_claims,
            self._ontologies,
            database,
            tables,
            ontology_id=ontology_id,
            binding=source_id,
        )
        self._store_claims[ontology_id] = _StoreClaim(database, tables, source_id)
        source = RecordEntitySource(
            database,
            projection,
            source_id=source_id,
            backend=backend,
            normalizer=self._normalizer,
            forms_database=forms_database,
        )
        return source, _LiveBinding(
            source_id, database, projection, declared_schema, tuple(source.entity_filters())
        )

    async def _bind_taxonomies(
        self, parts: OntologyParts, live: Mapping[str, _LiveBinding]
    ) -> dict[str, AsyncHierarchy[str]]:
        """Bind every ``taxonomies:`` row that names a backing, and refuse the rest.

        :meth:`_bind_sources`' shape one section over, and written to the same
        rule: **computed from the declared kind alone.** A row declaring no
        ``kind:`` asks for the assertion read, which
        :func:`~dataknobs_common.ontology.assemble_async_ontology` builds and
        nothing here re-implements, so it is passed over rather than bound.

        Three refusals, each naming what to go and fix, and each answering a
        question the other two do not:

        * the kind names no binder registered **here** -- which is the one of
          the three that stops being true when a binder is registered, and is
          therefore the one that belongs on a registry rather than on a door;
        * ``source:`` names no source **this document declares**, listing the
          ones it does. An axis reads one document's table and a source id is
          resolved in the document that wrote it;
        * ``source:`` names a source this registry bound as an **authored**
          one. A column axis has a column to read and an authored vocabulary
          has no table under it, so the answer is a refusal rather than a
          downgrade to the assertion read -- which would be the silent
          substitution this whole leg exists to end, arriving one step later.

        **It opens nothing.** The handle is the one :meth:`_bind_record_source`
        already resolved for the entity table, handed over rather than asked
        for again -- so ``self._handles`` gains no entry, ``close()``'s cascade
        is unchanged, and the axis cannot outlive the source it reads beside.

        Which is also why this ``async def`` awaits nothing, and it is stated
        rather than left for the next reader to go looking for the I/O: binding
        an axis is parsing a row and constructing over handles already open.
        The flavour is :meth:`_bind_sources`', whose own binding does open one,
        so that the two halves of a document are dispatched the same way and a
        backing whose binding *is* asynchronous needs no signature change here.
        """
        bound: dict[str, AsyncHierarchy[str]] = {}
        declared_sources = tuple(str(spec.get("id", "")) for spec in parts.source_specs)
        for spec in parts.taxonomy_specs:
            if "kind" not in spec:
                continue
            kind = str(spec.get("kind", ""))
            taxonomy_id = str(spec.get("id", ""))
            if kind not in self.LIVE_AXIS_KINDS:
                raise ValidationError(
                    f"taxonomy {taxonomy_id!r} declares kind {kind!r}, which no "
                    f"axis binder in this registry is registered for. Kinds it "
                    f"binds: {sorted(self.LIVE_AXIS_KINDS)}; an axis over this "
                    f"document's own assertions declares no `kind:` at all",
                    context={"taxonomy": taxonomy_id, "kind": kind},
                )
            axis = ColumnAxisBinding.from_mapping(spec, binding=taxonomy_id)
            if axis.source not in declared_sources:
                raise ValidationError(
                    f"taxonomy {taxonomy_id!r} reads column {axis.parent_key!r} of "
                    f"source {axis.source!r}, which this document does not "
                    f"declare. Declared: {sorted(declared_sources)}",
                    context={
                        "taxonomy": taxonomy_id,
                        "source_id": axis.source,
                        "declared": sorted(declared_sources),
                    },
                )
            binding = live.get(axis.source)
            if binding is None:
                raise ValidationError(
                    f"taxonomy {taxonomy_id!r} declares `kind: {COLUMN_AXIS_KIND}` "
                    f"over source {axis.source!r}, which is an authored source: it "
                    f"has no table, so there is no {axis.parent_key!r} column to "
                    f"read. An axis over an authored vocabulary's own edges "
                    f"declares no `kind:`",
                    context={"taxonomy": taxonomy_id, "source_id": axis.source},
                )
            refuse_undeclared_columns(
                (axis.parent_key,),
                binding.schema,
                binding=binding.source_id,
                named_by=f"taxonomy {taxonomy_id!r}",
            )
            bound[taxonomy_id] = ColumnHierarchy(
                binding.database,
                binding.projection.table,
                binding.projection.id,
                axis.parent_key,
                binding.narrowing,
            )
        return bound

    async def _axis_populations(
        self, ontology: AsyncOntology[str]
    ) -> dict[str, frozenset[str] | None]:
        """Every declared axis's node set, as the axis a subscriber reads reports it.

        **Recorded at load, which is the whole mechanism.** A delta between two
        loads cannot be computed from two reads taken at the second one: a live
        axis reads the table, so reading the outgoing axis at the moment the
        incoming one is built asks the same rows the same question twice and
        answers *nothing changed* however much did. So the population is taken
        when the vocabulary is built and kept until the next build replaces it.

        Asked of ``taxonomy(name).structure`` rather than derived here, because
        *which edges count* is the axis's rule -- asserted, of this relation,
        between entities for one backing; a row with both ends for the other --
        and a second copy of it is one that can disagree with the axis a
        subscriber is holding. ``parent_edges`` is the member that answers for
        the whole axis rather than the part a descent from the roots reaches,
        and it gives every node an entry including one that is only ever a
        parent.

        Asked through :meth:`~dataknobs_common.ontology.AsyncOntology.structure_for`
        rather than through ``taxonomy(name)``, and that is a correctness
        matter rather than a saving. The accessor refuses a definition
        declaring ``materialization.content: materialized`` -- a refusal about
        the axis's *content*, which this method never touches -- so asking
        through it made such a document raise out of :meth:`load`, and only
        where a bus was held. A refusal whose own reason is that it belongs
        where a caller asked for the axis had been moved to a call site that
        did not.

        ``None`` where the population is **not known**, which is a different
        answer from ``frozenset()`` and the one
        :meth:`_publish_taxonomy_delta` has a form for. Two things produce it:

        * **no bus at this load.** The read is a query, which a registry
          publishing nothing has no reason to pay -- but a registry can
          acquire a bus *after* a load, since any document may declare one, and
          the next reload then compares a real reading against this one. Read
          as an empty population that comparison reports every node as
          ``arrived`` on an axis where nothing changed;
        * **a backing that cannot enumerate itself.** The two optional
          hierarchy protocols are opt-in by member presence, so an axis with
          only the four singular members offers no enumeration but a descent
          from the roots, which misses a cyclic component entirely. Every
          backing a door files here answers ``parent_edges``, so the branch is
          the type checker's guarantee rather than a case reached today.
        """
        unknown = dict.fromkeys(ontology.taxonomies)
        if self._event_bus is None:
            return unknown
        populations: dict[str, frozenset[str] | None] = {}
        for name in ontology.taxonomies:
            axis = ontology.structure_for(name)
            populations[name] = (
                frozenset(await axis.parent_edges())
                if isinstance(axis, AsyncEnumerableHierarchy)
                else None
            )
        return populations

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
        store and rows of both kinds live in it.

        **One handle for two tables is a supported arrangement, not a
        degenerate one**, and what makes it work is on the source rather than
        here: the surface-form column is what tells the two kinds of row
        apart, in both directions. See
        :meth:`~dataknobs_data.ontology.sources.RecordEntitySource.entity_filters`
        for the invariant that rests on -- an entity row must not carry the
        form column, which is what the lookup direction already required.

        Built once per distinct block-and-table. A second binding naming the
        same one gets neither a second handle nor a second connection, and a
        reload against an unchanged environment resolves to the same block and
        so reuses what is open -- which is what keeps a reload loop from
        accumulating connections that only :meth:`close` would release.

        **Against a *changed* environment it does accumulate, and that is the
        deliberate half.** A reload resolving to a different block is a
        different key, so it opens a handle and the superseded one stays in
        ``_handles`` until :meth:`close`. Releasing it at ``replace=True``
        would be releasing a handle that a vocabulary handed out by
        :meth:`get` may still be reading through -- the value outlives its
        entry and this object cannot know who holds one, which is
        :meth:`unload`'s stated reason for releasing nothing either. So a
        long-lived loop reloading against an environment that keeps moving
        grows one handle per distinct resolution, and the way to bound that
        is to close the registry rather than to reload it forever.

        **The bookkeeping stays on the loop, and only the blocking calls are
        offloaded.** The cache lookup, the cache write and the append to
        ``_handles`` are a check-then-set over this registry's own state: run
        inside the worker thread, two concurrent loads naming one resolved
        block each found the cache empty and each opened a handle. On the
        loop, under :attr:`_handle_lock`, they cannot -- the second waits and
        then finds what the first opened. Two thread hops rather than one is
        what that costs, and a handle is opened once.

        **Connected here, because this registry built it.** ``memory`` and
        ``file`` answer a read either way, which is why nothing reported
        this; every other backend -- the three SQL ones, ``s3`` and
        ``elasticsearch`` -- raises *Database not connected* on its first
        query, so a ``$resource`` naming one produced a source that loaded
        clean and failed on its first read. An injected handle is connected
        by whoever handed it over, which is the same line ownership is drawn
        on everywhere else here.

        **Not through**
        :meth:`~dataknobs_data.database.AsyncDatabase.from_backend`, **for
        three reasons, none of them the loop.** That method offloads its own
        resolve-and-build to a thread exactly as this does -- its comment
        names this one as the precedent -- so a paragraph here once said this
        went its own way to keep the import off the loop, and by the time both
        had landed that was the one reason that had stopped being true. What
        it cannot do is the rest: it takes a backend name and a config, so it
        has nowhere to put the table :meth:`_keyed_block` writes into the
        resolved block; it builds an instance per call, where a handle here is
        cached under that keyed block and shared by every projection resolving
        to it; and it connects without owning the failure, where
        :meth:`_connect_or_close` closes a handle that raises mid-connect
        rather than losing it between *built* and *recorded*.
        """
        async with self._handle_lock:
            keyed = await asyncio.to_thread(self._keyed_block, block, table)
            key = json.dumps(keyed, sort_keys=True, default=str)
            cached = self._database_cache.get(key)
            if cached is not None:
                return cached
            handle: AsyncDatabase = await asyncio.to_thread(async_database_factory.create, **keyed)
            await self._connect_or_close(handle)
            self._database_cache[key] = handle
            self._handles.append((handle, True))
            return handle

    @staticmethod
    def _refuse_an_unbuildable_resolver_section(config: OntologyConfig) -> None:
        """Every ``resolver:`` refusal that needs nothing open, taken before anything is.

        **Split out of :meth:`_resolver_from` because of when it runs, not
        because of what it checks.** That reader is called after
        :meth:`_index_from`, which opens a vector store and records it --- so
        an undeclared key or a misspelled ``kind:``, neither of which needs an
        index to judge, was refused with a store already open. Inside
        :meth:`load` the caller still holds a registry to :meth:`close`;
        inside :meth:`from_config_async` they hold nothing, and the handle is
        stranded rather than leaked to somebody.

        :meth:`_index_from` states the rule for its own block --- *refused
        before the store is opened, so a document that cannot build an index
        does not leave a handle behind proving it tried* --- and the sibling
        was outside it.

        **The rung half is asked of the door rather than restated here.**
        :func:`~dataknobs_common.ontology.refuse_unbuildable_rungs` is the
        same check the door runs before it builds, published so a caller that
        opens resources on the way there can run it first. A copy on this side
        would be a second thing to keep in step with the registry it reads.
        """
        section = config.resolver
        if section is None:
            return
        _refuse_undeclared_keys(
            section,
            ontology_id=config.id,
            section="resolver",
            allowed=RESOLVER_BLOCK_KEYS,
            why=(
                "and here it is worse than inert: a section with no `rungs:` key is "
                "read as a composition of nothing, so `rung:` builds a cascade that "
                "matches nothing and reports success"
            ),
        )
        try:
            refuse_unbuildable_rungs(section, registry=async_signal_backends)
        except ValidationError as exc:
            # Re-raised naming the ontology, which the door cannot: it is
            # handed a section and does not know whose. The wording is
            # `_resolver_from`'s, so the two refusals a document author can
            # hit read as one voice, and `ontology_id` stays on the context
            # where a caller reads it.
            raise ValidationError(
                f"ontology {config.id!r} declares a `resolver:` section this registry "
                f"cannot build: {exc}",
                context={**exc.context, "ontology_id": config.id},
            ) from exc

    async def _index_from(
        self, config: OntologyConfig, ontology: AsyncOntology[str]
    ) -> SemanticIndex | None:
        """Build the semantic index an ``index:`` block asks for, or answer None.

        **Absence is an answer here and a refusal one level in.** A document
        with no ``index:`` section gets ``None`` -- the member that reads it
        says so and a shipped test pins it. A document *with* one and no
        embedder to build it with is **refused**, naming what to inject,
        because a section parsed into a field and quietly dropped is the
        failure that produces a registry reporting success while holding no
        store, no embedder and no index.

        The store is opened here and recorded as this registry's. The embedder
        is never opened: it is injected or the load fails.

        Args:
            config: The typed, already-resolved document.
            ontology: The vocabulary just assembled, which the index is over.

        Returns:
            The index, or ``None`` where the document declared no section.

        Raises:
            ValidationError: When the section carries a key this block does
                not read, is malformed, declares an embedder with nothing
                injected to satisfy it (or with one injected, so that two
                sources name the model), names entity fields an entity does
                not carry, misspells ``metric:``, or claims a metric the
                configured store is not serving.

                **One exception type, and the last two used to escape it.**
                ``DistanceMetric.resolve`` and the index's own claim check
                both raise ``ValueError``, so a caller catching what this
                docstring named did not catch a misspelled or disagreeing
                metric. What is being refused in every one of these cases is
                a *document*, which is what ``ValidationError`` means here.
        """
        block = config.index
        if not block:
            return None

        _refuse_undeclared_keys(
            block,
            ontology_id=config.id,
            section="index",
            allowed=INDEX_BLOCK_KEYS,
            why="`metirc:` builds an index with no metric check and reports success",
        )

        store_block = block.get("store")
        if not isinstance(store_block, Mapping) or not store_block:
            raise ValidationError(
                f"ontology {config.id!r} declares an `index:` section with no usable "
                f"`store:` block. The store is where the vectors go and there is no "
                f"default for it",
                context={"ontology_id": config.id},
            )

        # Refused before the store is opened, so a document that cannot build
        # an index does not leave a handle behind proving it tried. That is
        # true of everything down to `_vector_store_handle` below -- the two
        # embedder refusals, the source the document configures, and the
        # spelling of `metric:` -- and it is why they are ordered this way
        # rather than written in the order the constructor takes them. The one
        # check that cannot move is the metric *claim*, which is a comparison
        # against `store.metric` and so needs the store it is about.
        #
        # **Both directions**, because a section parsed and dropped is the
        # failure named above: with an embedder injected, a declared
        # `embedder:` used to be read by nothing at all, so a document naming
        # one model and a process injecting another agreed on nothing and
        # reported success. Every row records the injected model, so the
        # staleness contract stays self-consistent while the document is
        # silently untrue.
        declared_embedder = block.get("embedder")
        if declared_embedder is not None and self._injected_embedder is not None:
            raise ValidationError(
                f"ontology {config.id!r} declares an `index:` section with an `embedder:` "
                f"block **and** an embedder was injected. Both name the model every row is "
                f"written and judged stale against, and this registry cannot build the "
                f"declared one to compare -- so it would silently index under the injected "
                f"model while the document named another. Drop one",
                context={"ontology_id": config.id},
            )
        if declared_embedder is not None:
            raise ValidationError(
                f"ontology {config.id!r} declares an `index:` section with an `embedder:` "
                f"block, and no embedder was injected. An embedder is built in "
                f"`dataknobs-llm` -- which depends on this package, so this one cannot "
                f"build it -- and it holds no closeable resource of its own, so it is "
                f"the caller's to own. Pass `embedder=...` to `from_components` or to "
                f"`from_config_async`",
                context={"ontology_id": config.id},
            )
        if self._injected_embedder is None:
            raise ValidationError(
                f"ontology {config.id!r} declares an `index:` section and no embedder was "
                f"injected. Pass `embedder=...`; see the `embedder:` block's refusal for "
                f"why this registry does not build one",
                context={"ontology_id": config.id},
            )

        source = self._index_source_from(block, ontology)

        metric = block.get("metric")
        try:
            claimed = DistanceMetric.resolve(metric) if metric is not None else None
        except ValueError as exc:
            raise ValidationError(
                f"ontology {config.id!r} declares `index.metric: {metric!r}`, which is not "
                f"a spelling this library resolves: {exc}",
                context={"ontology_id": config.id, "metric": metric},
            ) from exc

        store = await self._vector_store_handle(dict(store_block))
        try:
            return SemanticIndex(source, self._injected_embedder, store, metric=claimed)
        except ValueError as exc:
            # A claim about a store that is not serving it. Raised as what
            # every sibling refusal in this block raises, because the thing
            # being refused is a document.
            raise ValidationError(
                f"ontology {config.id!r} declares an `index:` section whose store "
                f"disagrees with it: {exc}",
                context={"ontology_id": config.id, "metric": metric},
            ) from exc

    async def _resolver_from(
        self,
        config: OntologyConfig,
        ontology: AsyncOntology[str],
        index: SemanticIndex | None,
    ) -> AsyncEntityResolver | None:
        """Build the cascade a ``resolver:`` section asks for, or answer ``None``.

        :meth:`_index_from`'s shape one section over, because that is the
        sibling and a second shape here would be a second thing to keep in
        step. Where the two differ, they differ for a stated reason:

        * an absent section answers ``None`` here as it does there ---
          silence is a configuration answer;
        * an **empty** section does not. ``index: {}`` is read there as no
          index at all (``if not block``), and ``resolver: {}`` is read here
          as a composition somebody wrote: the door one layer down rules an
          absent ``rungs:`` key *a composition of nothing*, identically to
          ``rungs: []``, and both are different from an absent **section**.
          So a document writing an empty section gets a cascade with no
          rungs and :meth:`resolver` answers it rather than ``None``;
        * a section carrying no ``rungs:`` is therefore **not** an error;
        * **nothing is opened.** The index is already built and the store
          inside it is already recorded, so this reader takes a handle rather
          than taking a resource;
        * **its document-only refusals are not here.** They run before
          :meth:`_index_from` --- see
          :meth:`_refuse_an_unbuildable_resolver_section` --- because that is
          where the sibling's own ordering rule puts them, and this reader
          runs after a store is open.

        **The rungs are constructed through**
        :func:`~dataknobs_common.ontology.loader.async_build_resolver`, which
        is the one door that turns a ``resolver:`` section into rungs. A
        registry assembling its own list beside that door would be a second
        implementation of it, drifting silently, and the door takes a
        *handles* channel precisely so this reader does not have to.

        **What the rung it may construct searches is what the caller built.**
        A ``kind: semantic`` rung is handed the index this load just
        assembled, and ``load()`` does not fill it: building an index is the
        caller's line, for the reason :meth:`index` states. So a cascade built
        here over a store nobody has written to resolves through its declared
        rungs and reports the empty index rather than silently answering
        without it --- the rung says so itself, once, the first time it finds
        nothing.

        Args:
            config: The typed, already-resolved document.
            ontology: The vocabulary just assembled, which the rungs match
                against.
            index: The index this load built, or ``None``. Passed rather than
                read back off ``self``, because the entry is not stored until
                after both readers have run.

        Returns:
            The cascade, or ``None`` where the document declared no section.

        Raises:
            ValidationError: When a rung's own factory refuses the
                configuration --- the case that needs the index, and so the
                only one left here: a rung asking for a handle the document
                did not declare.

                **One exception type, and four used to escape it.** An
                unknown ``kind:`` arrived as ``NotFoundError``, a factory's
                refusal wrapped in ``OperationError``, a rung entry with no
                ``kind:`` as ``ValueError`` and a non-mapping entry as
                ``TypeError`` --- so a caller catching what this docstring
                named caught none of the four. They are fixed **at the door**
                rather than converted here: both
                :func:`~dataknobs_common.ontology.async_build_resolver` and
                its synchronous twin promise ``ValidationError`` in their own
                ``Raises:``, and a conversion on this side served this caller
                while every other caller of those doors kept the old types.
                The unwrapping of an authored cause moved with it.
        """
        section = config.resolver
        if section is None:
            return None

        # **The section, not the document**, and the difference is a second
        # parse rather than a convenience. The door takes a document because
        # its other callers hold one; this caller holds a document that has
        # already been resolved, typed and accepted, and handing it back would
        # have it read again by a reader that can refuse what the first one
        # passed. What the door consumes is `read.resolver` and nothing else,
        # so the minimal document carrying that section is the whole of what
        # there is to hand over -- and `id` travels with it because a config
        # cannot be built without one and a refusal that names the ontology is
        # worth more than one that does not.
        #
        # The handles are forwarded whatever the composition asks for, because
        # a factory reads the keys it names and ignores the rest: a cascade of
        # declared rungs is unaffected by their presence, and one naming a
        # semantic rung finds them there. `index` may be `None`, which the
        # rung's own factory refuses by name -- a document asking for a rung
        # over an index it never declared is a document missing a section,
        # not a registry missing a handle.
        try:
            return await async_build_resolver(
                {"id": config.id, "resolver": dict(section)},
                ontology,
                handles={"index": index, "ontology": ontology},
            )
        except ValidationError as exc:
            raise ValidationError(
                f"ontology {config.id!r} declares a `resolver:` section this registry "
                f"cannot build: {exc}",
                context={"ontology_id": config.id},
            ) from exc

    def _index_source_from(
        self, block: Mapping[str, Any], ontology: AsyncOntology[str]
    ) -> AsyncIndexSource:
        """The source an ``index:`` block describes, decorated if it asks to be.

        The block used to build ``EntitySourceIndexSource(ontology)`` and
        nothing else, so a document could not name the fields its text is
        composed from, could not set the separator between them, and could not
        reach :class:`~dataknobs_common.index.AliasSource` at all --- which is
        a published class whose only route was Python. A reference
        implementation shipped beside a configured door that cannot name it is
        a seam rather than a feature.

        Args:
            block: The ``index:`` section, already checked for unread keys.
            ontology: The vocabulary the source enumerates.

        Returns:
            The source, wrapped in the surface-form decorator where the block
            asked for one.

        Raises:
            ValidationError: When ``fields:``, ``join:`` or ``aliases:`` is
                malformed, when the block names fields an entity does not
                carry, or when the source cannot be enumerated. Every refusal
                reachable from here is one, which is the guarantee the door
                above it makes.
        """
        # Built as kwargs rather than passed positionally so the adapter keeps
        # ownership of its own defaults: a block naming neither key gets
        # whatever `EntitySourceIndexSource` declares, and this method does not
        # restate them where they would drift.
        configured: dict[str, Any] = {}

        fields = block.get("fields")
        if fields is not None:
            # A bare scalar in YAML is a string, so `fields: name` is what a
            # consumer writes by hand -- and `tuple("name")` is four
            # one-character field names, so the adapter's refusal would name
            # 'a', 'e', 'm', 'n' rather than the mistake. A non-sequence raised
            # `TypeError` past a door whose every other refusal is a
            # `ValidationError` about a document.
            if isinstance(fields, str) or not isinstance(fields, Sequence):
                raise ValidationError(
                    f"ontology {ontology.id!r} declares `index.fields: {fields!r}`; it takes "
                    f"a list of entity field names, and a bare string is one name spelled "
                    f"as its characters rather than a list of one",
                    context={"ontology_id": ontology.id, "fields": fields},
                )
            configured["fields"] = tuple(fields)

        join = block.get("join")
        if join is not None:
            # Not validated, this reaches `str.join`'s receiver slot and fails
            # at the first read rather than at load.
            if not isinstance(join, str):
                raise ValidationError(
                    f"ontology {ontology.id!r} declares `index.join: {join!r}`; it is what "
                    f"goes between two field values, so it is a string",
                    context={"ontology_id": ontology.id, "join": join},
                )
            configured["join"] = join

        leaf = EntitySourceIndexSource(ontology, **configured)

        aliases = block.get("aliases")
        # Checked for being a boolean rather than for truthiness: `aliases: "no"`
        # is truthy, so the value that most obviously means *off* turned it on.
        if aliases is not None and not isinstance(aliases, bool):
            raise ValidationError(
                f"ontology {ontology.id!r} declares `index.aliases: {aliases!r}`; it is on "
                f"or off, so it is `true` or `false`",
                context={"ontology_id": ontology.id, "aliases": aliases},
            )
        if not aliases:
            return leaf
        # Pointed at the leaf's own key rather than at the constant, so the
        # two ends of a one-key contract cannot be spelled apart here.
        return AliasSource(leaf, leaf.aliases_key)

    async def _vector_store_handle(self, block: dict[str, Any]) -> VectorStore:
        """A vector store for one resolved ``store:`` block, built once.

        :meth:`_database_handle`'s reasons transfer verbatim and are not
        restated: **off the event loop**, because the backend registry
        resolves a name by *importing* the module that implements it and a
        store's own constructor may build an index or open a client; **under
        the handle lock**, because the cache lookup and the open it guards are
        one check-then-set over this registry's state and two concurrent loads
        naming one block must not each open a store; and **cached on the
        resolved block**, so a reload against an unchanged environment reuses
        what is open rather than accumulating stores only :meth:`close` would
        release.

        Opened through :meth:`_connect_or_close`, which probes for the open by
        member presence -- a store spells the step ``initialize()`` where a
        handle spells it ``connect()``, and the window between *built* and
        *recorded* is the same window in both.
        """
        async with self._handle_lock:
            key = json.dumps(block, sort_keys=True, default=str)
            cached = self._vector_store_cache.get(key)
            if cached is not None:
                return cached
            store: VectorStore = await asyncio.to_thread(VectorStoreFactory().create, **block)
            await self._connect_or_close(store)
            self._vector_store_cache[key] = store
            self._handles.append((store, True))
            return store

    @staticmethod
    def _keyed_block(block: dict[str, Any], table: str) -> dict[str, Any]:
        """The resolved block, carrying the table it addresses where that applies.

        The blocking half of the decision, and the reason it is one: asking
        the backend registry for a name resolves it by *importing* the module
        that implements it, on the first such question in a process.
        """
        keyed = dict(block)
        if _backend_addresses_one_table(keyed):
            keyed["table"] = table
        return keyed

    async def _connect_or_close(self, resource: Any) -> None:
        """Connect something this registry just built, and close it if that fails.

        The window between *built* and *recorded* is where a collaborator
        gets lost: a backend that raises mid-connect has already acquired
        whatever its constructor acquired, and a registry that records
        ownership afterwards never learns it exists. So the close happens
        here, where the only reference is, and the caller records only what
        connected.

        One helper for the handle, the bus **and the store**, because the
        hazard is the acquire-then-connect shape rather than any one
        collaborator: they had the same gap and a second spelling of the
        remedy is one that can be fixed on one of them.

        **The open is probed by member presence, because the third
        collaborator spells it differently.** A database and a bus both carry
        ``connect()``; a
        :class:`~dataknobs_data.vector.stores.base.VectorStore` carries
        ``initialize()`` among its nine abstract members and no ``connect``
        at all, so calling one name outright raised ``AttributeError`` on the
        first store an ``index:`` block named. The generalisation above was
        right about the hazard and silent about the spelling, and no second
        spelling existed when it was written.

        ``close_if_owned`` already probes this way, for this reason, in this
        family. **The probe never has to choose**: of the public classes in
        ``common``, ``data`` and ``llm``, none resolves both members, which a
        test in the registry's suite asserts rather than this comment
        claiming it. A collaborator that carried both would be a ruling of
        its own.

        ``BaseException`` rather than ``Exception``, so a cancellation
        between the two steps releases what it interrupted. The close itself
        is error-isolated -- a teardown that raises must not replace the
        failure the caller is about to see.
        """
        opener = getattr(resource, "connect", None) or resource.initialize
        try:
            await opener()
        except BaseException:
            await close_if_owned(resource, True, on_error=self._teardown_failed)
            raise

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
        self,
        ontology_id: str,
        before: OntologyParts,
        after: OntologyParts,
        was: Mapping[str, frozenset[str] | None],
        now: Mapping[str, frozenset[str] | None],
    ) -> None:
        """Announce what a replacement changed -- once for the whole, once per axis.

        Three sets rather than two, and the third is the point: an entity
        that **changed name while keeping its id** is in neither *gone* nor
        *arrived*, so a two-set delta reports a rename as no change at all.
        An empty payload therefore means a genuine no-op.

        **A topic carries a delta over what that topic is about.** The sets are
        computed over one population per event: the declared entities for the
        ontology's own topic, and *that axis's nodes* for an axis topic. One
        delta published to every axis told a subscriber to ``taxonomy:colours``
        about a rename in ``taxonomy:sizes``, which is indistinguishable from a
        change to the axis they read.

        **The axis populations are the ones recorded at each load**, and that
        is what makes an axis over rows answerable at all. Computed here from
        the two documents, they would be the *declared* assertions either side
        -- which is ``{}`` for every vocabulary this registry exists to bind,
        so the three sets were empty whatever had changed and the guarantee
        stated above -- that an empty payload means a genuine no-op -- was
        false in the common case. Re-reading the outgoing axis here does not
        fix it either: a live axis reads the table, so both sides would answer
        with today's rows. See :meth:`_axis_populations`.

        **Two forms, for the axis whose population is not known.** Where either
        side's is unknown the payload carries ``axis_unenumerable: true`` and
        **none of the three sets** -- a subscriber reading ``payload["gone"]``
        gets a ``KeyError`` rather than an empty list, which is the point: it
        tells them to re-read the axis rather than telling them nothing
        happened. It is the shape :meth:`_departing` takes for the same reason
        one method along: a payload that cannot carry the ids says so, rather
        than carrying an empty list that already means something else.

        The reachable producer of that form is a registry that acquired its
        bus **between** two loads -- the first recorded nothing, having nobody
        to tell -- rather than an exotic backing; see
        :meth:`_axis_populations` for both.

        **Both levels, because neither contains the other.** An entity in no
        axis at all changes nothing on any axis topic and would otherwise be
        announced nowhere; an axis node that is not a declared entity -- an
        id an assertion places and no ``entities:`` row names -- is in no
        ontology-level set. The ontology topic already carries the pair of a
        departure and an arrival for the replacement itself; this is what
        *within* it moved.

        **Every axis either side declares**, in the order the new document
        writes them and then the ones it no longer does. An axis a rebuild
        removed reports its whole population gone, which is the strongest
        thing its subscribers can be told: the axis they read is not there
        any more.
        """
        departed = [t for t in before.taxonomies if t not in after.taxonomies]
        await self._publish(
            topic=f"ontology:{ontology_id}",
            event_type=EventType.UPDATED,
            payload={
                "ontology_id": ontology_id,
                **_three_sets(
                    before, after, set(before.declared_entities), set(after.declared_entities)
                ),
            },
        )
        for taxonomy_id in (*after.taxonomies, *departed):
            # An axis either side no longer declares held nothing, rather than
            # holding something unknown: the whole of its population is `gone`,
            # which is the strongest thing its subscribers can be told. An axis
            # it *did* declare and no population was recorded for is the other
            # case, and `.get` answering None is what keeps the two apart --
            # `frozenset()` there would report a whole axis as newly arrived.
            was_nodes = was.get(taxonomy_id) if taxonomy_id in before.taxonomies else frozenset()
            now_nodes = now.get(taxonomy_id) if taxonomy_id in after.taxonomies else frozenset()
            await self._publish(
                topic=f"taxonomy:{taxonomy_id}",
                event_type=EventType.UPDATED,
                payload={
                    "ontology_id": ontology_id,
                    "taxonomy_id": taxonomy_id,
                    **(
                        {"axis_unenumerable": True}
                        if was_nodes is None or now_nodes is None
                        else _three_sets(before, after, set(was_nodes), set(now_nodes))
                    ),
                },
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

    async def _event_bus_from(self, block: Mapping[str, Any], *, ontology_id: str) -> None:
        """Build the bus a document configured, and own it.

        **The first bus wins, whatever built it, and a later block is
        reported rather than applied.** An injected bus always wins and is
        never closed by this registry; one built here is closed by
        :meth:`close`, because the registry built it. A registry holds *one*
        bus for every vocabulary it loads, so a second document declaring
        ``event_bus:`` is asking for something the registry cannot give it
        without moving the ontologies already announced on the first one onto
        a bus their subscribers are not holding. It is logged at ``INFO``, the
        way a ``database:`` block passed over for an injected handle is: a
        block that did not take effect is exactly the kind of silence that
        reads as a broken bus rather than as a decision.

        **Off the event loop**, for :meth:`_database_handle`'s reason and the
        same measured one: every built-in backend factory imports its driver
        --- ``asyncpg``, ``redis``, ``aioboto3`` --- inside the factory call,
        so that it is not imported at all in a base install. An import is
        disk I/O, and it happens on the first bus of that backend in a
        process and never again, which is the order-dependence that makes
        this the kind of stall review does not find.

        **Through the synchronous door, deliberately.**
        ``create_event_bus_async`` exists and awaits a factory whose
        *construction* is asynchronous, on the caller's loop -- which is
        right for such a factory and is exactly what must not happen to a
        synchronous one whose first act is an import. ``EventBusFactory``,
        the shape the registry publishes for a consumer's own backend, is
        synchronous, so the door taken here is the one the extension point
        describes. A consumer whose bus really is built asynchronously builds
        it themselves and hands it over: that is the ``event_bus``
        collaborator :attr:`OPTIONAL_COMPONENTS` names, and an injected bus
        wins over a configured one at the line above.
        """
        if self._event_bus is not None:
            logger.info(
                "Ontology %r declares `event_bus:` and this registry already holds "
                "%s bus; the block is not built, and this vocabulary's events are "
                "announced on the bus already here.",
                ontology_id,
                "an injected" if not self._owns_event_bus else "a configured",
            )
            return
        bus = await asyncio.to_thread(create_event_bus, dict(block))
        await self._connect_or_close(bus)
        self._event_bus = bus
        self._owns_event_bus = True


def _three_sets(
    before: OntologyParts, after: OntologyParts, was: set[str], now: set[str]
) -> dict[str, list[str]]:
    """Gone, arrived and renamed over one population, from two documents.

    The population is the caller's -- a declared-entity set for the
    ontology, an axis's nodes for an axis -- and the *names* are always the
    two documents', because that is the only place a name is written. An id
    in the population with no declared row cannot be renamed and is reported
    only by its arrival or departure.
    """
    names_before, names_after = before.declared_entities, after.declared_entities
    return {
        "gone": sorted(was - now),
        "arrived": sorted(now - was),
        "renamed": sorted(
            entity_id
            for entity_id in was & now
            if entity_id in names_before
            and entity_id in names_after
            and names_before[entity_id].name != names_after[entity_id].name
        ),
    }


def _backend_addresses_one_table(block: Mapping[str, Any]) -> bool:
    """Whether this backend's config declares a ``table`` -- i.e. is one table's.

    Asked of the registered backend's own config class rather than decided
    from a list here, so a backend added to this package answers without an
    edit in this file.

    **Two limits, both of which read as "the handle is the store".** Neither
    is a defect in the seven backends this package ships -- all seven register
    a class carrying a ``CONFIG_CLS`` dataclass, and the three that address a
    table spell the field ``table`` -- but both are properties of the
    *mechanism* rather than of those seven, so a reader must not take a
    ``False`` here as *asked and answered no*.

    A :class:`~dataknobs_common.registry.PluginRegistry` accepts a plain
    callable as a factory, and a function carries no ``CONFIG_CLS``. Such a
    registration therefore cannot answer yes, whatever it builds: a
    consumer-registered backend answers for itself only where it is
    registered as a *class* whose ``CONFIG_CLS`` is a dataclass.

    And the field is matched by the name ``table``. Elasticsearch's ``index``
    and S3's ``prefix`` are the same concept spelled differently, so both are
    read here as store-is-handle -- which for Elasticsearch is what routes it
    onto the shared-store path the guide describes, and is load-bearing there
    rather than incidental. Widening the match is not a rename: the caller
    that acts on a ``True`` writes the *projection's* table into the block
    under a key, so an addressing backend has to say which key rather than
    only that it has one.

    **Imported at module scope, and the laziness it dropped was buying
    nothing.** All three names here were once imported in the body, which
    reads as a deferral this function needs. It does not: this module already
    imports ``dataknobs_data.factory`` at module scope, and that import alone
    puts ``backend_selection`` and ``backends`` in ``sys.modules`` before this
    file finishes loading -- so the statements deferred an import that had
    already happened. What genuinely stays off the event loop is the
    ``get_factory`` call below, whose ``on_first_access`` hook imports each
    backend implementation at *call* time; that is offloaded by
    :meth:`OntologyRegistry._database_handle`, which runs this whole function
    in a worker thread, and it would be offloaded from wherever the ``import``
    statement sat. An in-body import implies a constraint, and the next reader
    preserves it.
    """
    declared = block.get("backend")
    if not declared:
        return False
    factory = async_backends.get_factory(normalize_backend(declared))
    return _config_class_addresses_one_table(getattr(factory, "CONFIG_CLS", None))


class _LiveBinding(NamedTuple):
    """What a bound live source leaves behind for an axis over the same table.

    Not state: it lives for one load, is handed from :meth:`_bind_sources` to
    :meth:`_bind_taxonomies`, and is dropped. What it carries is the three
    things a column axis needs and a :class:`RecordEntitySource` does not
    publish -- the handle the registry opened for the entity table, the parsed
    projection whose ``id:`` is the axis's child column, and the schema the
    ``parent_key:`` is checked against.

    **The handle above all.** ``_database_handle`` is cached per resolved block
    and table, so an axis naming the source's table would get the same object
    back from it -- but *getting it back from the cache* and *being handed
    it* differ where they matter: the first is a code path that could open one,
    and the second cannot. See :meth:`_bind_taxonomies`.
    """

    source_id: str
    database: AsyncDatabase
    projection: EntityProjection
    schema: DatabaseSchema | None
    #: What narrows a read through this handle to the binding's **entity**
    #: rows, as :meth:`RecordEntitySource.entity_filters` computes it. Empty
    #: where the two tables have handles of their own; the form-column
    #: discriminator where the handle *is* the store. Carried rather than
    #: re-derived, because a second reader deriving it from a different
    #: invariant is how the axis and the entity source came to disagree about
    #: which rows of one store are this binding's.
    narrowing: tuple[Filter, ...]


class _BoundSources(NamedTuple):
    """What one document's ``sources:`` section came to, in three named parts."""

    entities: Any
    describes: tuple[SourceDescription, ...]
    live: Mapping[str, _LiveBinding]


class _StoreClaim(NamedTuple):
    """One loaded ontology's hold on a store, and what it named there.

    ``database`` is compared by **identity**, which is the whole mechanism:
    two bindings collide exactly when the registry handed them one object,
    and that is true for a different reason at each door. A backend declaring
    a ``table`` gets one handle per table from
    :meth:`OntologyRegistry._database_handle`, so two tables are two objects
    and never meet here. A backend declaring none gets one handle for its
    whole store. An injected handle is one object for every document the
    registry loads, and where that handle is itself table-addressed it is
    stuck on the *one* table it was built for -- so two bindings naming two
    tables through it are further wrong, not less. Asking the object rather
    than asking the backend is what covers all three with one question.
    """

    database: AsyncDatabase
    tables: tuple[str, ...]
    binding: str


def _refuse_a_second_binding_over_one_store(
    claims: Mapping[str, _StoreClaim],
    loaded: Mapping[str, Any],
    database: AsyncDatabase,
    tables: tuple[str, ...],
    *,
    ontology_id: str,
    binding: str,
) -> None:
    """Refuse a binding that would share a store with one already loaded.

    **Nothing narrows a read to a binding's table on a store that has no
    table dimension.** ``table:`` survives as a cache-key discriminator that
    :meth:`OntologyRegistry._keyed_block` drops for such a backend and as a
    ``describe()`` field; no query carries it. So two bindings over one store
    read each other's rows on both axes at once --
    :meth:`~...RecordEntitySource.by_type` answers with the other binding's
    ids, :meth:`~...RecordEntitySource.get` answers with the other binding's
    row projected under this binding's type and name, and
    :meth:`~...RecordEntitySource.by_surface_form` filters a folded-form
    column that the other binding's form rows also carry.

    **Refused rather than narrowed, because there is nothing to narrow
    with.** The one discriminator this design has is the form column, and it
    separates *form rows from entity rows* rather than one binding's entities
    from another's. Giving it a second job would need a column on the
    consumer's own table saying which binding a row belongs to -- and this
    source only ever reads, so such a column is theirs to add and backfill
    rather than ours to write. The shape that would lift this refusal is a
    projection naming a type column it already has, alongside the ``const``
    it already declares, so that ``declares`` stays a complete enumeration and
    the reads gain a filter. Until a binding can say that, a composition this
    refuses is one no arrangement of these handles can serve.

    **Declaring the same tables is left alone.** Two documents projecting one
    table asked for the population they get -- the same rows read as a
    catalogue entry by one and as something else by the other -- and nothing
    here can tell that apart from a mistake. What is caught is narrower: two
    bindings that named *different* tables and were handed one store.

    **A claim is consulted only while its ontology is loaded**, which is why
    ``loaded`` is passed rather than the claims being pruned. An id that was
    unloaded releases its store with no bookkeeping, a document replacing
    itself does not collide with itself, and a load that failed after binding
    leaves a claim that can never be reached.
    """
    for other_id, claim in claims.items():
        if other_id == ontology_id or other_id not in loaded:
            continue
        if claim.database is not database or claim.tables == tables:
            continue
        raise ValidationError(
            f"binding {binding!r} names {list(tables)} in a store that "
            f"ontology {other_id!r} already binds as {list(claim.tables)} "
            f"(binding {claim.binding!r}). One handle serves both, and no read "
            f"narrows to a binding's table there, so each would answer with "
            f"the other's rows. Give this binding a `$resource` of its own, or "
            f"a backend that addresses a table",
            context={
                "source_id": binding,
                "ontology_id": ontology_id,
                "tables": list(tables),
                "held_by": other_id,
                "held_tables": list(claim.tables),
                "handle": type(database).__name__,
            },
        )


def _refuse_one_handle_for_two_tables(
    database: AsyncDatabase, projection: EntityProjection, *, binding: str
) -> None:
    """Refuse an injected handle that cannot reach both of a projection's tables.

    The configured door asks
    :func:`_backend_addresses_one_table` and opens a second handle when the
    answer is yes. The injected door has no block to ask about, so it asks the
    handle -- and where one handle addresses one table and the projection names
    two, there is no arrangement of it that reads them both.

    **Why this is a refusal rather than a miss.** Nothing downstream can
    notice. ``_shared_store`` is true because the two handles are one object,
    so ``entity_filters`` emits the form-column discriminator and the entity
    reads are correct; ``describe()`` advertises ``SURFACE_FORM_LOOKUP``
    because the projection declares a lookup; and ``by_surface_form`` filters
    the form column against a table that does not carry it, finding nothing.
    ``frozenset()`` is the contract's spelling of *ran and matched nothing*,
    so the caller is told the vocabulary has no such form rather than that the
    binding cannot be served.
    """
    lookup = projection.surface_forms
    if lookup is None or lookup.table == projection.table:
        return
    if not _handle_addresses_one_table(database):
        return
    raise ValidationError(
        f"binding {binding!r} projects entity rows from {projection.table!r} and "
        f"surface forms from {lookup.table!r}, but the injected handle "
        f"({type(database).__name__}) addresses one table. Inject the second "
        f"handle as `forms_database=`, or bind through a `database:` block and "
        f"let the registry open both",
        context={
            "source_id": binding,
            "table": projection.table,
            "surface_forms_table": lookup.table,
            "handle": type(database).__name__,
        },
    )


def _config_class_addresses_one_table(config_cls: Any) -> bool:
    """Whether a backend's config dataclass declares a ``table``.

    The shared half of :func:`_backend_addresses_one_table`, which asks it of
    a name in a ``database:`` block, and
    :func:`_handle_addresses_one_table`, which asks it of a handle already
    built. One question, two ways of reaching the class that answers it -- and
    a second copy of the ``dataclasses.fields`` walk would be a second answer
    to drift from, which is what the configured and injected doors disagreeing
    about a shared store cost the first time.
    """
    if config_cls is None or not dataclasses.is_dataclass(config_cls):
        return False
    return any(field.name == "table" for field in dataclasses.fields(config_cls))


def _handle_addresses_one_table(database: AsyncDatabase) -> bool:
    """Whether this handle reaches one table, asked of the handle's own class.

    Every backend in this package carries its config dataclass as a
    ``CONFIG_CLS`` class attribute, so a handle answers the same question a
    ``database:`` block does without the registry keeping a list. A handle
    whose class declares none -- a consumer's own ``AsyncDatabase``
    implementation, or a test double -- answers False, which is the reading
    that refuses nothing: it says *this handle is the store*, which is what an
    implementation with no table concept is.

    **That default is a guess where the handle is a consumer's own**, and it
    guesses in the direction that stays quiet. An implementation that really
    does address one table, and carries no ``CONFIG_CLS`` dataclass to say so,
    is read as a shared store and passes a refusal it should have failed --
    leaving :meth:`~...RecordEntitySource.by_surface_form` filtering the form
    column against a table that does not carry it. The remedy is on the
    caller's side and needs no classification: hand the second handle in as
    ``forms_database=``, and the question never arises.
    """
    return _config_class_addresses_one_table(getattr(type(database), "CONFIG_CLS", None))


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


def _refuse_a_scan_over_a_binding_that_cannot_bound_it(
    config: OntologyConfig, projection: EntityProjection, *, binding: str
) -> None:
    """Refuse a **declared** scanning rung over a binding declaring no bound.

    :func:`_refuse_a_form_reading_rung_with_no_lookup`'s shape for the second
    number a live binding cannot supply, and refused for the same reason: it
    is a property of the vocabulary that no ``Database`` in this package can
    be asked about.

    **What the bound is for.** A scanning rung probes every contiguous window
    of a query and takes its width from the vocabulary's longest declared form
    -- see
    :meth:`~dataknobs_common.ontology.sources.EntitySource.longest_form_tokens`.
    An authored index counts its own keys and answers. A live table cannot,
    so the source answers ``None``, the rung enumerates in full, and the cost
    is *n(n+1)/2* probes for *n* tokens: 1,275 for a fifty-token utterance,
    20,100 for a two-hundred-token paste. Over an in-memory source those are
    dictionary lookups and the quadratic is affordable -- which is why the
    refusal is *here*, where the source is known to reach for data, rather
    than in the rung, which is also composed over sources where it is fine.

    **Which rungs those are is asked, not listed**, exactly as the sibling
    refusal asks which rungs read folded forms. Each kind declares
    ``bounded_by_longest_form`` on its class and the signal registry derives
    it, so a rung added to ``dataknobs-common`` and a consumer's own are both
    covered without an edit here.

    **Declared rungs only**, which is the sibling's scope and for its reason:
    a document with no ``resolver:`` section gets the default composition,
    whose scan nothing in the document names. That case belongs where the rung
    is constructed, by the door holding both the composition and the source.
    It is not this registry even now that :meth:`OntologyRegistry.resolver`
    answers a cascade, and the reason is the same silence: a document with no
    section gets ``None`` from :meth:`OntologyRegistry._resolver_from`, so the
    default composition is the value-level doors' to build and this registry
    constructs no rung a document did not name. What this function does is
    refuse the case a document *does* state.
    """
    lookup = projection.surface_forms
    if lookup is None or lookup.longest_form_tokens is not None:
        return
    declared = _declared_rung_kinds(config)
    scanning = sorted(
        kind
        for kind in declared
        if async_signal_backends.get_metadata(kind).get("bounded_by_longest_form")
    )
    if not scanning:
        return
    raise ValidationError(
        f"binding {binding!r} is read by {scanning}, which probes every window "
        f"of a query and bounds that enumeration by the vocabulary's longest "
        f"declared form. A live table cannot be counted, so the binding must "
        f"declare `surface_forms.longest_form_tokens:` -- without it the scan "
        f"spends n(n+1)/2 reads per query, each one a round trip",
        context={
            "source_id": binding,
            "rungs": scanning,
            "surface_forms_table": lookup.table,
        },
    )


def _refuse_an_undeclared_projection_type(
    projection: EntityProjection, entity_types: Collection[str], *, binding: str
) -> None:
    """A projection's ``type: {const: ...}`` names a type the document declares.

    The ninth member of the family
    :func:`~dataknobs_common.ontology.loader._refuse_an_unresolved_reference`
    refuses eight of, and the one that cannot live beside them: the core has
    no notion of ``entity_projection:``, which is this package's schema, so
    reading it there would make ``dataknobs_common`` parse a section only a
    live binding understands. The rule travels; the reading stays here.

    **It is the worst-consequence member of the family.** An ``entities:``
    row whose ``type:`` names nothing mistypes one entity; a projection's
    ``const:`` types *every row of the table*, so an index enumerating the
    vocabulary by type silently finds none of them -- over a source whose
    whole purpose is that nobody enumerated its rows by hand.

    The same guard the other eight carry: an empty ``entity_types:`` is *no*
    schema rather than an empty one, and a document binding a live table
    while leaving its type vocabulary elsewhere is not making a claim this
    registry can check. Its caller applies the family's other exemption, for
    a document declaring ``imports:``.

    After :meth:`EntityProjection.from_mapping`, which is where ``type:``
    being a column or a malformed block is refused. A row has to be well
    formed before it is worth telling its author which of its keys resolves.
    """
    if not entity_types:
        return
    if projection.const_type in entity_types:
        return
    raise ValidationError(
        f"binding {binding!r} declares `entity_projection.type: "
        f"{{const: {projection.const_type!r}}}`, which no entity type in this "
        f"document declares -- so every row of {projection.table!r} would be "
        f"typed as something the vocabulary does not hold. "
        f"Declared: {sorted(entity_types)}",
        context={
            "source_id": binding,
            "section": "entity_projection",
            "field": "type",
            "value": projection.const_type,
        },
    )


def _refuse_a_form_reading_rung_with_no_lookup(
    config: OntologyConfig, projection: EntityProjection, *, binding: str
) -> None:
    """Refuse a **declared** rung that reads folded forms this binding has none of.

    The declared-schema rule's shape, for a second thing the loader cannot
    introspect. A rung reading
    :meth:`~dataknobs_common.ontology.sources.EntitySource.by_surface_form`
    matches a surface form the person typed against forms the *source* folds,
    and a live table holds the form as it was written -- so a binding under
    one either declares where its folded forms live or it is rejected here,
    at load, naming the key.

    **Which rungs those are is asked, not listed.** ``exact`` is one of
    three: ``scan`` and ``lexical`` reach the same member, and a refusal
    naming only the first let a document declaring either of the others load
    clean and fail on the first resolve. Each kind declares the fact in the
    registry it is registered under, so this covers a rung added to
    ``dataknobs-common`` and a consumer's own without an edit here.

    **Declared rungs only, which is narrower than every rung that will be
    built.** A document with no ``resolver:`` section gets the *default*
    composition, two of whose three rungs read the member, and nothing in the
    document says so. That case is not this function's: the rung is refused
    where it is constructed, by the door that holds both the composition and
    the source. Refusing it here would refuse every record binding a
    consumer loads to read entities from and never resolves against --
    :func:`~dataknobs_common.ontology.loader._refuse_async_only_rungs` states
    that precedent. This registry does build rungs now, but only the ones a
    document writes: silence gets ``None`` rather than the default
    composition, so the case this paragraph excludes is still nobody's here.
    """
    if projection.surface_forms is not None:
        return
    declared = _declared_rung_kinds(config)
    reading = sorted(
        kind
        for kind in declared
        if async_signal_backends.get_metadata(kind).get("reads_surface_forms")
    )
    if not reading:
        return
    raise ValidationError(
        f"binding {binding!r} is read by the {reading} rung(s) and declares no "
        f"`surface_forms:`. Such a rung matches a surface form the person typed "
        f"against forms the source folds, and a live table holds the form as it "
        f"was written -- no engine folds at query time the way `str.casefold` "
        f"does. Declare the lookup: `surface_forms: {{table: <table>, form: "
        f"<column holding the folded form>, entity: <column holding this "
        f"projection's id>}}`",
        context={"source_id": binding, "rungs": reading},
    )


def _refuse_undeclared_keys(
    block: Mapping[str, Any], *, ontology_id: str, section: str, allowed: frozenset[str], why: str
) -> None:
    """Refuse a configured section carrying a key nothing reads.

    **One body, because the second copy was written by pasting the first.**
    Two sections declare a key set and refuse what is outside it, and the
    argument is the same for both: a key nothing acts on is configuration a
    consumer wrote and this registry accepted while ignoring, so ``metirc:``
    builds an index with no metric check and ``rung:`` builds a cascade that
    matches nothing --- each reporting success. Only the example differs,
    which is why it is the parameter.

    Args:
        block: The section as the document wrote it.
        ontology_id: Whose document it is, for the message.
        section: The section's own name, spelled as a document spells it.
            Carried into the message without an article, so one body serves
            ``index:`` and ``resolver:`` without a ``a(n)``.
        allowed: Every key the reader acts on.
        why: What accepting an unread key costs *this* section, as a clause
            the message continues into.

    Raises:
        ValidationError: When the block carries a key outside *allowed*.
    """
    undeclared = sorted(set(block) - allowed)
    if not undeclared:
        return
    raise ValidationError(
        f"ontology {ontology_id!r} declares `{section}:` with key(s) "
        f"{undeclared}, which this block does not read. A key nothing acts on is "
        f"configuration a consumer wrote and the registry accepted while ignoring "
        f"-- {why}. This block reads: {', '.join(sorted(allowed))}",
        context={"ontology_id": ontology_id, "undeclared": undeclared, "section": section},
    )


def _declared_rung_kinds(config: OntologyConfig) -> tuple[str, ...]:
    """The rung kinds this document **writes**, in the order it writes them.

    ``None`` and ``{}`` are different answers upstream -- silence gets the
    default composition, an empty section is a composition somebody chose --
    and both are *written* rungs of nothing, so both answer ``()`` here. The
    distinction is kept in the read rather than collapsed in a ``or {}``,
    because a later caller asking a different question of this section needs
    it back and would not find it.
    """
    section = config.resolver
    if section is None:
        return ()
    rungs = section.get("rungs")
    if not isinstance(rungs, list):
        return ()
    return tuple(str(rung.get("kind", "")) for rung in rungs if isinstance(rung, Mapping))


__all__ = ["OntologyRegistry"]

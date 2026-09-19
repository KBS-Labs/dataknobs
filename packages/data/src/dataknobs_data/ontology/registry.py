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
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

from dataknobs_common.entity_resolution import async_signal_backends
from dataknobs_common.events import Event, EventType, create_event_bus
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.lifecycle import close_if_owned
from dataknobs_common.ontology import (
    AUTHORED_SOURCE_ID,
    AUTHORED_SOURCE_KINDS,
    AssertionHierarchy,
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    AsyncOntology,
    Entity,
    MappingAssertionSource,
    OntologyConfig,
    assemble_async_ontology,
    build_ontology,
    split_qualified,
)
from dataknobs_common.ontology.model import DK_ENTITY_TYPE, DK_RELATION_TYPE
from dataknobs_common.structured_config import StructuredConfigConsumer

from dataknobs_data.backend_selection import normalize_backend
from dataknobs_data.backends import async_backends
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
    from types import TracebackType

    from dataknobs_common.entity_resolution.protocols import AsyncEntityResolver
    from dataknobs_common.events import EventBus
    from dataknobs_common.ontology import OntologyParts, RelationRef, SourceDescription
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

    **The collaborators here are ones a caller *may* inject, and the mixin
    has a field for that.** Neither is required: a configured registry
    resolves its own database from ``$resource`` and publishes nothing when
    no bus is wired. Both are nonetheless real injection points that
    ``from_components`` takes, so tooling asking what this class accepts has
    to be told about them -- which is
    :attr:`~dataknobs_common.structured_config.StructuredConfigConsumer.OPTIONAL_COMPONENTS`,
    read by ``accepted_components()``.

    They were declared under ``EXPECTED_COMPONENTS`` first, because that was
    the only field there was, and the cost was not a documentation one: a
    fully loaded registry with nothing wrong with it answered
    ``{"database", "event_bus"}`` to ``missing_components()`` and raised from
    ``require_components()``. This class was that field's first adopter in the
    tree and is the reason the second spelling exists.

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
    OPTIONAL_COMPONENTS: ClassVar[frozenset[str]] = frozenset(
        {"database", "event_bus", "forms_database"}
    )

    #: The live source kinds this registry binds.
    #:
    #: Computed from the declared kind alone, exactly as
    #: ``dataknobs_common.ontology.loader``'s refusal is, so one config means
    #: one thing in every environment: a kind outside this set and outside
    #: ``AUTHORED_SOURCE_KINDS`` is refused by name whether or not some other
    #: package that could bind it happens to be importable.
    LIVE_SOURCE_KINDS: ClassVar[frozenset[str]] = frozenset({RECORD_SOURCE_KIND})

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
        # One handle opened at a time, so that the cache lookup and the open
        # it guards are one step. Held on the loop rather than in the worker
        # thread the open is offloaded to -- see `_database_handle`.
        self._handle_lock = asyncio.Lock()
        self._injected_database: AsyncDatabase | None = None
        self._injected_forms_database: AsyncDatabase | None = None
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

    def index(self, ontology_id: str) -> None:
        """``None``: the semantic index this ontology declared, which is never one yet.

        The summary says ``None`` rather than *the index, or None* because
        the annotation does, and a docstring promising a value its signature
        cannot return is the half a caller reads. The member exists anyway,
        because absence is a configuration answer rather than an error -- an
        ontology that declared no ``index:`` section answers ``None`` for
        good, and one whose registry has no index builder answers the same
        thing from the other side, so a caller writes the same branch either
        way and only the reason differs.

        It **returns** what ``load()`` built and is not a second way to build
        one. The return type widens to ``SemanticIndex | None`` when that type
        exists; today ``None`` is the only value it can have, and declaring
        that is more useful than declaring a type nothing can produce.

        ``ontology_id`` is unread for the same reason, and is in the signature
        rather than out of it because :meth:`resolver` -- which answers the
        same shape of question and *will* read it -- takes it too. A member
        that grows a parameter when its body starts needing one is a
        signature change a consumer pays for.
        """
        return None

    def resolver(self, ontology_id: str) -> AsyncEntityResolver | None:
        """The configured placement cascade for this ontology, or None.

        ``None`` for every id this registry builds today, for :meth:`index`'s
        reason and with :meth:`index`'s contract: it returns what ``load()``
        built. The ``resolver:`` section *is* read at load -- a rung that
        reads folded surface forms, declared over a live binding with none,
        is refused there -- so the section is not ignored, only unbuilt. A
        composition the document does not write is refused where its rungs
        are constructed instead, which is the only place it can be seen.
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
        """Build a vocabulary from an already-resolved document, and announce it.

        **Where an ``event_bus:`` block is read, and the only place.** Both
        doors arrive here -- the constructor's through :meth:`_ainit` and
        :meth:`load`'s after resolution -- so reading it here is what makes
        the two agree about where the block lives. Read off the *typed*
        config rather than the mapping beside it, because a published door
        coerces before this object sees anything and a key the type does not
        declare does not survive that.

        Before the announcement below rather than after, which is the whole
        point of building it here: a document that configures a bus and loads
        a vocabulary expects the arrival of that vocabulary on that bus.
        """
        typed = (
            config if isinstance(config, OntologyConfig) else OntologyConfig.from_dict(dict(config))
        )
        if typed.event_bus is not None:
            await self._event_bus_from(typed.event_bus)
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
        ontology = await assemble_async_ontology(
            parts,
            entities=entities,
            assertions=AsyncMappingAssertionSource(parts.declared_assertions),
            describes=describes,
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
            authored = AsyncMappingEntitySource(
                parts.declared_entities, normalizer=self._normalizer
            )
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
        return RecordEntitySource(
            database,
            projection,
            source_id=source_id,
            backend=backend,
            normalizer=self._normalizer,
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
        store and rows of both kinds live in it.

        **One handle for two tables is a supported arrangement, not a
        degenerate one**, and what makes it work is on the source rather than
        here: the surface-form column is what tells the two kinds of row
        apart, in both directions. See
        :meth:`~dataknobs_data.ontology.sources.RecordEntitySource._entity_filters`
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
        on everywhere else here. Not through
        :meth:`~dataknobs_data.database.AsyncDatabase.from_backend`, which
        does both halves in one call: it resolves and builds on the caller's
        loop, which is the import this offload exists to keep off it.
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

        One helper for the handle and the bus, because the hazard is the
        acquire-then-connect shape rather than either collaborator: the two
        had the same gap and a second spelling of the remedy is one that can
        be fixed on one of them.

        ``BaseException`` rather than ``Exception``, so a cancellation
        between the two steps releases what it interrupted. The close itself
        is error-isolated -- a teardown that raises must not replace the
        failure the caller is about to see.
        """
        try:
            await resource.connect()
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
        self, ontology_id: str, before: OntologyParts, after: OntologyParts
    ) -> None:
        """Announce what a replacement changed -- once for the whole, once per axis.

        Three sets rather than two, and the third is the point: an entity
        that **changed name while keeping its id** is in neither *gone* nor
        *arrived*, so a two-set delta reports a rename as no change at all.
        An empty payload therefore means a genuine no-op.

        **A topic carries a delta over what that topic is about.** The three
        sets are computed over one population per event: the declared
        entities for the ontology's own topic, and *that axis's nodes* for an
        axis topic. One delta published to every axis told a subscriber to
        ``taxonomy:colours`` about a rename in ``taxonomy:sizes``, which is
        indistinguishable from a change to the axis they read.

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
            was_axis = before.taxonomies.get(taxonomy_id)
            now_axis = after.taxonomies.get(taxonomy_id)
            await self._publish(
                topic=f"taxonomy:{taxonomy_id}",
                event_type=EventType.UPDATED,
                payload={
                    "ontology_id": ontology_id,
                    "taxonomy_id": taxonomy_id,
                    **_three_sets(
                        before,
                        after,
                        _axis_nodes(before, was_axis.relation) if was_axis else set(),
                        _axis_nodes(after, now_axis.relation) if now_axis else set(),
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

    async def _event_bus_from(self, block: Mapping[str, Any]) -> None:
        """Build the bus a document configured, and own it.

        An injected bus always wins and is never closed by this registry; one
        built here is closed by :meth:`close`, because the registry built it.

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
            return
        bus = await asyncio.to_thread(create_event_bus, dict(block))
        await self._connect_or_close(bus)
        self._event_bus = bus
        self._owns_event_bus = True


def _axis_nodes(parts: OntologyParts, relation: RelationRef) -> set[str]:
    """Every node this relation's edges place, as the axis over them reads it.

    Asked of :class:`~dataknobs_common.ontology.AssertionHierarchy` rather
    than derived here, because *which edges count* is that class's rule --
    asserted, of this relation, between entities rather than to a literal --
    and a second copy of it is one that can disagree with the axis a
    subscriber is holding. ``parent_edges`` is the member that answers for
    the whole axis rather than the part a descent from the roots reaches,
    and it gives every node an entry including one that is only ever a
    parent.

    Synchronous and over declared rows: an axis a *document* wrote is what a
    delta between two documents is computed over, which is the same
    population :meth:`_departing` reads and the same reason -- a live
    binding's rows change underneath the registry with no reload at all.
    """
    axis: AssertionHierarchy[str] = AssertionHierarchy(
        MappingAssertionSource(parts.declared_assertions), relation
    )
    return set(axis.parent_edges())


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
    from a list here, so a consumer-registered backend answers for itself and
    a backend added to this package answers without an edit in this file.

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
    so ``_entity_filters`` emits the form-column discriminator and the entity
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
    is constructed, by the door holding both the composition and the source --
    a door this registry does not have yet, since :meth:`OntologyRegistry.resolver`
    builds nothing. What this function does is make the key exist for that
    door to check, and refuse the case a document *does* state.
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
    that precedent, and this registry builds no rung at all.
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

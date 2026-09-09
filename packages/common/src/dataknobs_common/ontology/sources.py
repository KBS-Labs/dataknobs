"""Where an ontology's entities and assertions come from, and two concretes.

Four protocols, twinned. The twins exist because a taxonomy over an in-memory
backing is synchronous all the way down: a vocabulary a person typed has
nothing to await, and forcing an event loop on it would make the cheapest case
pay for the most expensive one.

The concretes here back an *authored* vocabulary -- a file someone edited --
and each flavour is a thin façade over one shared index. The index is where the
lookups live, so a fix to how a surface form is matched lands once rather than
twice; the flavour classes hold only the difference, which is the ``async``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dataknobs_common.capabilities import Capability
from dataknobs_common.ontology.model import (
    Assertion,
    Entity,
    EntityRef,
    Polarity,
    RelationRef,
    RelationType,
    SourceRef,
    Term,
)
from dataknobs_common.text import default_normalizer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from dataknobs_common.records import Record

#: The ``source_id`` and ``backend`` an authored vocabulary reports.
#:
#: An authored source has no ``database_factory`` key behind it, and saying so
#: is more useful than an empty string: a caller reading ``backend`` learns
#: what kind of thing this is rather than that the field was not filled in.
AUTHORED_SOURCE_ID = "authored"


@dataclass(frozen=True)
class SourceDescription:
    """What a source can say about itself without touching its backend.

    Free, because it is configuration -- which is what makes it the level a
    caller may consult on every read rather than once at startup.
    """

    source_id: str
    backend: str
    table: str | None
    projection: Mapping[str, Any]
    capabilities: frozenset[Capability]
    declares: frozenset[str] = frozenset()


@runtime_checkable
class EntitySource(Protocol):
    """Read access to entities, synchronously."""

    def get(self, entity_id: str) -> Entity | None: ...

    def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]: ...

    def fetch_origin(self, ref: SourceRef) -> Record | None: ...

    def fetch_origins(self, refs: Sequence[SourceRef]) -> dict[SourceRef, Record]: ...

    def describe(self) -> SourceDescription: ...

    def by_surface_form(self, form: str) -> frozenset[str]: ...

    def by_alias_form(self, form: str) -> frozenset[str]: ...

    def by_type(self, type_id: str) -> frozenset[str]: ...


@runtime_checkable
class AsyncEntitySource(Protocol):
    """The same members with ``async`` added to those that reach for data.

    ``describe`` stays synchronous on both twins: it answers from
    configuration and touches no backend, so making it awaitable would buy
    nothing and cost every caller an ``await``.
    """

    async def get(self, entity_id: str) -> Entity | None: ...

    async def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]: ...

    async def fetch_origin(self, ref: SourceRef) -> Record | None: ...

    async def fetch_origins(self, refs: Sequence[SourceRef]) -> dict[SourceRef, Record]: ...

    def describe(self) -> SourceDescription: ...

    async def by_surface_form(self, form: str) -> frozenset[str]: ...

    async def by_alias_form(self, form: str) -> frozenset[str]: ...

    async def by_type(self, type_id: str) -> frozenset[str]: ...


@runtime_checkable
class AssertionSource(Protocol):
    """Read access to assertions, synchronously.

    ``polarity`` is on both read members from the start, and that is the one
    decision here a later release could not take back. This is a
    ``@runtime_checkable`` protocol a consumer satisfies structurally, so
    widening it afterwards leaves ``isinstance`` passing while every call that
    passes the new keyword raises ``TypeError`` against their implementation --
    a migration for code we never see, to add a keyword they can ignore.
    """

    def get(self, assertion_id: str) -> Assertion | None: ...

    def find(
        self,
        *,
        subject: str | None = None,
        relation: RelationRef | None = None,
        object: Term | None = None,
        polarity: Polarity | None = None,
    ) -> list[Assertion]: ...

    def find_many(
        self,
        *,
        subjects: Sequence[str] | None = None,
        objects: Sequence[str] | None = None,
        relation: RelationRef | None = None,
        polarity: Polarity | None = None,
    ) -> dict[str, list[Assertion]]: ...


@runtime_checkable
class AsyncAssertionSource(Protocol):
    """The asynchronous twin."""

    async def get(self, assertion_id: str) -> Assertion | None: ...

    async def find(
        self,
        *,
        subject: str | None = None,
        relation: RelationRef | None = None,
        object: Term | None = None,
        polarity: Polarity | None = None,
    ) -> list[Assertion]: ...

    async def find_many(
        self,
        *,
        subjects: Sequence[str] | None = None,
        objects: Sequence[str] | None = None,
        relation: RelationRef | None = None,
        polarity: Polarity | None = None,
    ) -> dict[str, list[Assertion]]: ...


def object_entity_id(term: Term) -> str | None:
    """The entity id an assertion object points at, or None for a literal.

    Public beside :func:`relation_id` for the same reason: two readers ask this
    question -- the index below, and the assertion-backed hierarchy that walks
    one relation's edges -- and a second copy of *what counts as an entity
    object* is a rule that can disagree with itself.
    """
    if isinstance(term, EntityRef):
        return term.entity_id
    return None


def relation_id(relation: RelationRef) -> str:
    """The id of a relation given either an id or the definition itself.

    Both forms are legal in an assertion, so every comparison goes through
    here rather than each site deciding what it was handed.
    """
    if isinstance(relation, RelationType):
        return relation.id
    return relation


@dataclass
class _EntityIndex:
    """The lookups both entity flavours share.

    Built once at construction. A surface form maps to the ids of the entities
    that *declare* it -- never to an alias's own id, which is not an entity --
    and to a frozenset rather than one id, because two entities may
    legitimately share a form. That ambiguity is declared by the vocabulary,
    so it survives to the caller rather than being resolved here by a rule
    nobody wrote down.

    **Aliases are indexed twice**, into the folded map every form goes into
    and into a map of their own. The first is what an exact-form lookup wants
    and cannot distinguish; the second is what a rung matching *alias forms
    specifically* needs, and it cannot be reconstructed from the first --
    ``by_form`` cannot report which of an entity's three kinds of form
    matched. Reconstructing it in the caller, by fetching each hit and folding
    its aliases again, would put the vocabulary's own answer in the matcher's
    hands, which is the arrangement the frozenset above exists to avoid.
    """

    entities: Mapping[str, Entity]
    normalizer: Callable[[str], str] = default_normalizer
    by_form: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    by_alias: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    by_type: dict[str, frozenset[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        forms: dict[str, set[str]] = {}
        aliases: dict[str, set[str]] = {}
        types: dict[str, set[str]] = {}
        for entity_id, entity in self.entities.items():
            for form in (entity_id, entity.name):
                if not form:
                    continue
                forms.setdefault(self.normalizer(form), set()).add(entity_id)
            for form in entity.aliases:
                if not form:
                    continue
                folded = self.normalizer(form)
                forms.setdefault(folded, set()).add(entity_id)
                aliases.setdefault(folded, set()).add(entity_id)
            types.setdefault(entity.type, set()).add(entity_id)
        self.by_form = {form: frozenset(ids) for form, ids in forms.items()}
        self.by_alias = {form: frozenset(ids) for form, ids in aliases.items()}
        self.by_type = {type_id: frozenset(ids) for type_id, ids in types.items()}

    def get(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        found = {}
        for entity_id in entity_ids:
            entity = self.entities.get(entity_id)
            if entity is not None:
                found[entity_id] = entity
        return found

    def surface_form(self, form: str) -> frozenset[str]:
        return self.by_form.get(self.normalizer(form), frozenset())

    def alias_form(self, form: str) -> frozenset[str]:
        return self.by_alias.get(self.normalizer(form), frozenset())

    def of_type(self, type_id: str) -> frozenset[str]:
        return self.by_type.get(type_id, frozenset())

    def describe(self) -> SourceDescription:
        return SourceDescription(
            source_id=AUTHORED_SOURCE_ID,
            backend=AUTHORED_SOURCE_ID,
            table=None,
            projection={},
            # ORIGIN_FETCH is absent, which is how this source states that
            # `fetch_origin` will answer None for every ref it is handed. The
            # `SourceRef` still travels out intact: we cannot reach the row,
            # the caller can, and saying so costs one capability rather than
            # a caller's failed round trip.
            capabilities=frozenset(),
            declares=frozenset(self.by_type),
        )


class MappingEntitySource:
    """An :class:`EntitySource` over entities already in memory.

    Backs a hand-edited vocabulary. ``fetch_origin`` always answers ``None``
    and ``describe()`` says so, which is the honest arrangement: an authored
    file has no database behind it, and an entity's ``source:`` is a reference
    the *consumer* can spend on their own data even though we cannot.
    """

    def __init__(
        self,
        entities: Mapping[str, Entity],
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._index = _EntityIndex(entities, normalizer or default_normalizer)

    def get(self, entity_id: str) -> Entity | None:
        """The entity with this id, or None."""
        return self._index.get(entity_id)

    def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        """Those of these ids that name an entity. Misses are absent."""
        return self._index.get_many(entity_ids)

    def fetch_origin(self, ref: SourceRef) -> Record | None:
        """Always None -- see the class docstring, and ``describe()``."""
        return None

    def fetch_origins(self, refs: Sequence[SourceRef]) -> dict[SourceRef, Record]:
        """Always empty, for ``fetch_origin``'s reason."""
        return {}

    def describe(self) -> SourceDescription:
        """What this source is, including that its origins are unfetchable.

        ``declares`` is derived from the entity types actually held, rather
        than configured: over a mapping that is cheap, and a derived answer
        cannot drift from the contents the way a declared one can.
        """
        return self._index.describe()

    def by_surface_form(self, form: str) -> frozenset[str]:
        """The ids of entities whose id, name or alias matches this form.

        The entity's own id, never an alias's -- an alias is a string a person
        typed, not a thing the vocabulary names. Which of the three matched is
        not reported here; :meth:`by_alias_form` is the member that asks the
        narrower question.
        """
        return self._index.surface_form(form)

    def by_alias_form(self, form: str) -> frozenset[str]:
        """The ids of entities carrying this form as an **alias**.

        A subset of :meth:`by_surface_form`, and a subset the caller could not
        compute: that member folds an entity's id, name and aliases together
        and cannot say which matched. A rung matching alias forms
        specifically asks this instead of fetching each hit and folding its
        aliases a second time -- the vocabulary declares which forms are
        aliases, so the matcher does not have to decide it again.
        """
        return self._index.alias_form(form)

    def by_type(self, type_id: str) -> frozenset[str]:
        """The ids of every entity of this type."""
        return self._index.of_type(type_id)


class AsyncMappingEntitySource:
    """:class:`MappingEntitySource` with ``async`` on the members that read.

    Nothing here awaits anything -- the data is already in memory. The class
    exists so an ``AsyncOntology`` over an authored file satisfies the same
    protocol as one over a database, which is what lets a consumer swap the
    backing without rewriting the calls.
    """

    def __init__(
        self,
        entities: Mapping[str, Entity],
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._index = _EntityIndex(entities, normalizer or default_normalizer)

    async def get(self, entity_id: str) -> Entity | None:
        """The entity with this id, or None."""
        return self._index.get(entity_id)

    async def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        """Those of these ids that name an entity. Misses are absent."""
        return self._index.get_many(entity_ids)

    async def fetch_origin(self, ref: SourceRef) -> Record | None:
        """Always None -- see :class:`MappingEntitySource`."""
        return None

    async def fetch_origins(self, refs: Sequence[SourceRef]) -> dict[SourceRef, Record]:
        """Always empty, for ``fetch_origin``'s reason."""
        return {}

    def describe(self) -> SourceDescription:
        """Synchronous on both twins -- it answers from configuration."""
        return self._index.describe()

    async def by_surface_form(self, form: str) -> frozenset[str]:
        """The ids of entities whose id, name or alias matches this form."""
        return self._index.surface_form(form)

    async def by_alias_form(self, form: str) -> frozenset[str]:
        """The ids of entities carrying this form as an alias.

        See :meth:`MappingEntitySource.by_alias_form` for why this is a member
        rather than something a caller reconstructs.
        """
        return self._index.alias_form(form)

    async def by_type(self, type_id: str) -> frozenset[str]:
        """The ids of every entity of this type."""
        return self._index.of_type(type_id)


@dataclass
class _AssertionIndex:
    """The lookups both assertion flavours share.

    Indexed three ways at construction -- by id, by subject, by object entity
    -- because every question asked of an authored assertion list is one of
    those three, and a list scan per question turns a walk of a hierarchy into
    a quadratic one.

    The id index is *built* rather than taken from a caller's keys: the input
    is a sequence, and trusting keys to agree with ``Assertion.id`` is the kind
    of assumption that holds until one caller passes a dict built some other
    way.

    **Matching lives here and nowhere else.** Both concretes below are
    one-line forwards, so a criterion written into one of them -- ``polarity``
    is the newest -- would have to be written into the other and would then be
    two rules that can disagree. Neither flavour holds a matcher, which is why
    a filter added to this class is added to both sources at once.
    """

    assertions: Sequence[Assertion]
    by_id: dict[str, Assertion] = field(default_factory=dict, init=False)
    by_subject: dict[str, list[Assertion]] = field(default_factory=dict, init=False)
    by_object: dict[str, list[Assertion]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for assertion in self.assertions:
            self.by_id[assertion.id] = assertion
            self.by_subject.setdefault(assertion.subject, []).append(assertion)
            object_id = object_entity_id(assertion.object)
            if object_id is not None:
                self.by_object.setdefault(object_id, []).append(assertion)

    def get(self, assertion_id: str) -> Assertion | None:
        return self.by_id.get(assertion_id)

    def find(
        self,
        subject: str | None,
        relation: RelationRef | None,
        object_: Term | None,
        polarity: Polarity | None = None,
    ) -> list[Assertion]:
        if subject is not None:
            candidates: Sequence[Assertion] = self.by_subject.get(subject, [])
        elif object_ is not None and (oid := object_entity_id(object_)) is not None:
            candidates = self.by_object.get(oid, [])
        else:
            candidates = self.assertions

        wanted = relation_id(relation) if relation is not None else None
        return [
            assertion
            for assertion in candidates
            if (subject is None or assertion.subject == subject)
            and (wanted is None or relation_id(assertion.relation) == wanted)
            and (object_ is None or assertion.object == object_)
            and (polarity is None or assertion.polarity is polarity)
        ]

    def find_many(
        self,
        subjects: Sequence[str] | None,
        objects: Sequence[str] | None,
        relation: RelationRef | None,
        polarity: Polarity | None = None,
    ) -> dict[str, list[Assertion]]:
        if (subjects is None) == (objects is None):
            raise ValueError(
                "find_many takes exactly one of `subjects` or `objects`; "
                f"got subjects={subjects!r}, objects={objects!r}"
            )

        wanted = relation_id(relation) if relation is not None else None
        if subjects is not None:
            index, keys = self.by_subject, subjects
        else:
            # Narrowed by the guard above: exactly one of the two is set, so
            # `objects` is not None here. Written as a branch rather than an
            # `assert` so the narrowing is the control flow rather than a
            # claim about it.
            index, keys = self.by_object, objects or ()

        found: dict[str, list[Assertion]] = {}
        for key in keys:
            matches = [
                assertion
                for assertion in index.get(key, [])
                if (wanted is None or relation_id(assertion.relation) == wanted)
                and (polarity is None or assertion.polarity is polarity)
            ]
            # A key with no matches is absent rather than empty: the caller
            # asked which of these have edges, and an empty list would answer
            # a question they did not ask.
            if matches:
                found[key] = matches
        return found


class MappingAssertionSource:
    """An :class:`AssertionSource` over the authored ``assertions:`` rows."""

    def __init__(self, assertions: Sequence[Assertion]) -> None:
        self._index = _AssertionIndex(assertions)

    def get(self, assertion_id: str) -> Assertion | None:
        """The assertion with this id, or None."""
        return self._index.get(assertion_id)

    def find(
        self,
        *,
        subject: str | None = None,
        relation: RelationRef | None = None,
        object: Term | None = None,
        polarity: Polarity | None = None,
    ) -> list[Assertion]:
        """Every assertion matching all of the criteria given.

        A criterion left as None does not constrain -- ``polarity=None``
        therefore returns negations alongside assertions, which is the right
        default for a caller asking *what does this vocabulary say about x*
        and the wrong one for a caller walking a structure. The walk asks for
        ``ASSERTED``; see
        :meth:`~dataknobs_common.ontology.hierarchy.AssertionHierarchy._find`.

        ``relation`` accepts either an id or the definition itself, and matches
        on id either way.
        """
        return self._index.find(subject, relation, object, polarity)

    def find_many(
        self,
        *,
        subjects: Sequence[str] | None = None,
        objects: Sequence[str] | None = None,
        relation: RelationRef | None = None,
        polarity: Polarity | None = None,
    ) -> dict[str, list[Assertion]]:
        """The bulk form: one call per axis, never one per node.

        Exactly one of ``subjects`` or ``objects``; the result is keyed by that
        axis, and a key with no matches is absent.
        """
        return self._index.find_many(subjects, objects, relation, polarity)


class AsyncMappingAssertionSource:
    """:class:`MappingAssertionSource` with ``async`` on every member."""

    def __init__(self, assertions: Sequence[Assertion]) -> None:
        self._index = _AssertionIndex(assertions)

    async def get(self, assertion_id: str) -> Assertion | None:
        """The assertion with this id, or None."""
        return self._index.get(assertion_id)

    async def find(
        self,
        *,
        subject: str | None = None,
        relation: RelationRef | None = None,
        object: Term | None = None,
        polarity: Polarity | None = None,
    ) -> list[Assertion]:
        """Every assertion matching all of the criteria given."""
        return self._index.find(subject, relation, object, polarity)

    async def find_many(
        self,
        *,
        subjects: Sequence[str] | None = None,
        objects: Sequence[str] | None = None,
        relation: RelationRef | None = None,
        polarity: Polarity | None = None,
    ) -> dict[str, list[Assertion]]:
        """The bulk form -- see :meth:`MappingAssertionSource.find_many`."""
        return self._index.find_many(subjects, objects, relation, polarity)


if TYPE_CHECKING:  # pragma: no cover - checked by the type checker, not run

    def _concretes_satisfy_their_protocols() -> None:
        """Each concrete is assignable to the protocol it claims.

        Written here rather than as a test because this file is type-checked
        and the test tree is not.

        What it catches, measured rather than assumed: a **missing member**,
        an incompatible **type**, and a renamed **keyword-only** parameter --
        which covers ``find`` and ``find_many``, whose arguments are all
        keyword-only. What it does not catch is a renamed positional
        parameter: a positional call still works, so the checker allows it,
        and ``by_surface_form(form)`` could be renamed here without complaint.

        ``isinstance`` catches strictly less -- it compares method *names* and
        nothing else -- which is why this is not left to the runtime check the
        ``@runtime_checkable`` decorator provides.
        """
        entities: EntitySource = MappingEntitySource({})
        async_entities: AsyncEntitySource = AsyncMappingEntitySource({})
        assertions: AssertionSource = MappingAssertionSource([])
        async_assertions: AsyncAssertionSource = AsyncMappingAssertionSource([])
        del entities, async_entities, assertions, async_assertions

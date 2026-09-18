# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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
from dataknobs_common.hierarchy import K
from dataknobs_common.ontology.model import (
    Assertion,
    Entity,
    EntityRef,
    Polarity,
    RelationRef,
    SourceRef,
    Term,
    relation_id,
)
from dataknobs_common.text import default_normalizer, token_spans

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from dataknobs_common.entity_resolution.protocols import (
        AliasFormSource,
        AsyncAliasFormSource,
        AsyncSurfaceFormCatalog,
        SurfaceFormCatalog,
    )
    from dataknobs_common.records import Record

#: The ``source_id`` and ``backend`` an authored vocabulary reports.
#:
#: An authored source has no ``database_factory`` key behind it, and saying so
#: is more useful than an empty string: a caller reading ``backend`` learns
#: what kind of thing this is rather than that the field was not filled in.
AUTHORED_SOURCE_ID = "authored"


@dataclass(eq=True, frozen=False)
class SourceDescription:
    """What a source can say about itself without touching its backend.

    Free, because it is configuration -- which is what makes it the level a
    caller may consult on every read rather than once at startup.

    **Compared field-wise, and therefore unhashable.** ``describe()`` builds a
    fresh value on every call, so identity comparison would leave a pure
    derived value unequal to itself. ``projection`` is a mapping, so a hash
    over the fields would raise.
    """

    source_id: str
    backend: str
    table: str | None
    projection: Mapping[str, Any]
    capabilities: frozenset[Capability]
    declares: frozenset[str] = frozenset()


@runtime_checkable
class EntitySource(Protocol[K]):
    """Read access to entities, synchronously.

    **Generic in the entity key, defaulted to ``str``.** The structure axis is
    generic because a walk only ever hashes a node id; this is generic because
    an axis carries both axes at once, and a hierarchy over a caller's own key
    beside a lookup that could only be asked about a ``str`` was a pair that
    could not be used together. A source written before the parameter existed
    is an ``EntitySource[str]`` and conforms unchanged.

    **Four of its members move with the key and four do not**, which is the
    distinction a reader cannot get from the diff: ``get``, ``get_many``,
    ``by_surface_form`` and ``by_type`` each carry an entity id somewhere. What
    stays ``str`` is a *surface form* a person typed, an entity **type** id the
    document authored, and the origin members' record types.
    ``by_surface_form(form: str) -> frozenset[K]`` is the member with one of
    each in one signature.

    **Reporting alias forms is not here**, although
    :class:`MappingEntitySource` answers for them:
    :class:`~dataknobs_common.entity_resolution.AliasFormSource` carries that
    member, and a rung wanting it checks for that protocol. This one is
    ``@runtime_checkable`` and consumers satisfy it structurally, so every
    member added to it turns an implementation we never see from conforming
    into non-conforming at once -- which is the migration the warning on
    :class:`AssertionSource` refuses to impose, and it is stricter for a
    member than for a keyword.

    **:meth:`longest_form_tokens` is here anyway, and the exception is dated.**
    That argument is about a population of structural conformers, and when the
    member was added that population was empty: ``git ls-tree -r
    common/v3.2.0 -- packages/common/src/dataknobs_common/ontology`` lists no
    files, so this protocol had not been in a release and the migration it
    describes cost nothing. It is the only member that will ever be able to
    say that, and the tag is named so the next author can re-run the check
    rather than inherit the conclusion.

    The alternative -- a second optional protocol, on the
    :class:`AliasFormSource` pattern -- buys a source the right not to answer,
    and what it charges for that is a *slower* scan rather than a lossier one:
    a rung told nothing enumerates every window, which is exactly what this
    member's own ``None`` now means. So the choice was never between a bound
    and a guess; it was between asking every source and asking some. Asking
    every source is what keeps the linear enumeration the common case instead
    of the configured one. A source whose backing makes the answer expensive
    may cache it; it is fixed for the life of the vocabulary.
    """

    def get(self, entity_id: K) -> Entity[K] | None: ...

    def get_many(self, entity_ids: Sequence[K]) -> dict[K, Entity[K]]: ...

    def fetch_origin(self, ref: SourceRef) -> Record | None: ...

    def fetch_origins(self, refs: Sequence[SourceRef]) -> dict[SourceRef, Record]: ...

    def describe(self) -> SourceDescription: ...

    def by_surface_form(self, form: str) -> frozenset[K]:
        """The ids of entities carrying this form, **folded by the source**.

        The fold is the source's own and is not the caller's to apply: a
        rung hands this member the query as the person typed it, and the
        source folds both sides with the normalizer it was built with --
        :func:`~dataknobs_common.text.default_normalizer` unless one was
        supplied. Folding in the caller instead would put the vocabulary's
        own answer in the matcher's hands, and two matchers would fold two
        ways over one vocabulary.

        ``frozenset()`` means **ran and matched nothing**, which is what
        every cascade reads it as before falling through to a guessing
        rung. A source that cannot fold must not answer it: it withholds
        :attr:`~dataknobs_common.capabilities.Capability.SURFACE_FORM_LOOKUP`
        from :meth:`describe` and raises
        :class:`~dataknobs_common.capabilities.CapabilityNotSupportedError`
        when asked anyway, because an unfolded answer is indistinguishable
        from a genuine miss and the caller takes its fallback for the wrong
        reason.

        Stated here rather than left to each implementation, and stated
        once this protocol had a second implementor: a silent protocol is
        how two packages came to answer one question two ways before
        anyone noticed.
        """
        ...

    def by_type(self, type_id: str) -> frozenset[K]: ...

    def longest_form_tokens(self) -> int | None: ...


@runtime_checkable
class AsyncEntitySource(Protocol[K]):
    """The same members with ``async`` added to those that reach for data.

    Alias forms are absent here too, for the reason :class:`EntitySource`
    gives; :class:`~dataknobs_common.entity_resolution.AsyncAliasFormSource`
    is where that member lives.

    ``describe`` and ``longest_form_tokens`` stay synchronous on both twins:
    each answers from what the source already holds and touches no backend, so
    making either awaitable would buy nothing and cost every caller an
    ``await``.

    :meth:`longest_form_tokens` widened this protocol too, under the dated
    exception :class:`EntitySource` records and for the same reason -- neither
    twin had been in a release. Read that note before adding a second member
    on its authority: what it licensed was a widening with no conformers to
    break, and that is a fact about ``common/v3.2.0``, not a standing
    allowance.
    """

    async def get(self, entity_id: K) -> Entity[K] | None: ...

    async def get_many(self, entity_ids: Sequence[K]) -> dict[K, Entity[K]]: ...

    async def fetch_origin(self, ref: SourceRef) -> Record | None: ...

    async def fetch_origins(self, refs: Sequence[SourceRef]) -> dict[SourceRef, Record]: ...

    def describe(self) -> SourceDescription: ...

    async def by_surface_form(self, form: str) -> frozenset[K]:
        """The ids of entities carrying this form, **folded by the source**.

        The synchronous twin's contract, unchanged by the ``await``:
        the source folds with its own normalizer, ``frozenset()`` means
        *ran and matched nothing*, and a source that cannot fold withholds
        :attr:`~dataknobs_common.capabilities.Capability.SURFACE_FORM_LOOKUP`
        and raises
        :class:`~dataknobs_common.capabilities.CapabilityNotSupportedError`
        rather than answering over an unfolded column. See
        :meth:`EntitySource.by_surface_form`.

        This twin is where the contract costs something to keep. A live
        table holds the form as it was written and no engine folds the way
        :meth:`str.casefold` does at query time, so a source over one
        reads a lookup whose rows were folded when they were written, or
        it declines.
        """
        ...

    async def by_type(self, type_id: str) -> frozenset[K]: ...

    def longest_form_tokens(self) -> int | None: ...


@runtime_checkable
class AssertionSource(Protocol[K]):
    """Read access to assertions, synchronously.

    **Generic in the entity key**, for :class:`EntitySource`'s reason and with
    the same split: ``find`` and ``find_many`` name entities, and ``get`` names
    an **assertion**, whose id is minted where the assertion is written rather
    than supplied by a consumer. So two of the three members move.

    ``polarity`` is on both read members from the start, and that is the one
    decision here a later release could not take back. This is a
    ``@runtime_checkable`` protocol a consumer satisfies structurally, so
    widening it afterwards leaves ``isinstance`` passing while every call that
    passes the new keyword raises ``TypeError`` against their implementation --
    a migration for code we never see, to add a keyword they can ignore.
    """

    def get(self, assertion_id: str) -> Assertion[K] | None: ...

    def find(
        self,
        *,
        subject: K | None = None,
        relation: RelationRef | None = None,
        object: Term[K] | None = None,
        polarity: Polarity | None = None,
    ) -> list[Assertion[K]]: ...

    def find_many(
        self,
        *,
        subjects: Sequence[K] | None = None,
        objects: Sequence[K] | None = None,
        relation: RelationRef | None = None,
        polarity: Polarity | None = None,
    ) -> dict[K, list[Assertion[K]]]: ...


@runtime_checkable
class AsyncAssertionSource(Protocol[K]):
    """The asynchronous twin."""

    async def get(self, assertion_id: str) -> Assertion[K] | None: ...

    async def find(
        self,
        *,
        subject: K | None = None,
        relation: RelationRef | None = None,
        object: Term[K] | None = None,
        polarity: Polarity | None = None,
    ) -> list[Assertion[K]]: ...

    async def find_many(
        self,
        *,
        subjects: Sequence[K] | None = None,
        objects: Sequence[K] | None = None,
        relation: RelationRef | None = None,
        polarity: Polarity | None = None,
    ) -> dict[K, list[Assertion[K]]]: ...


def object_entity_id(term: Term[K]) -> K | None:
    """The entity id an assertion object points at, or None for a literal.

    Generic in the key, so an edge read out of an assertion comes back in the
    space the assertion was written in rather than in ``str``.

    Public for the reason :func:`~dataknobs_common.ontology.model.relation_id`
    is: two readers ask this question -- the index below, and the
    assertion-backed hierarchy that walks one relation's edges -- and a second
    copy of *what counts as an entity object* is a rule that can disagree with
    itself. That one now lives beside the alias it resolves, because the
    dataclasses holding a ``RelationRef`` call it from ``__post_init__``.
    """
    if isinstance(term, EntityRef):
        return term.entity_id
    return None


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
    longest_form: int | None = field(default=0, init=False)

    def __post_init__(self) -> None:
        forms: dict[str, set[str]] = {}
        aliases: dict[str, set[str]] = {}
        types: dict[str, set[str]] = {}
        merges = False
        for entity_id, entity in self.entities.items():
            for form in (entity_id, entity.name):
                if not form:
                    continue
                folded = self.normalizer(form)
                merges = merges or self._merges_tokens(form, folded)
                forms.setdefault(folded, set()).add(entity_id)
            for form in entity.aliases:
                if not form:
                    continue
                folded = self.normalizer(form)
                merges = merges or self._merges_tokens(form, folded)
                forms.setdefault(folded, set()).add(entity_id)
                aliases.setdefault(folded, set()).add(entity_id)
            types.setdefault(entity.type, set()).add(entity_id)
        self.by_form = {form: frozenset(ids) for form, ids in forms.items()}
        self.by_alias = {form: frozenset(ids) for form, ids in aliases.items()}
        self.by_type = {type_id: frozenset(ids) for type_id, ids in types.items()}
        # Measured over the *folded* keys, because a folded form is what a
        # lookup compares against -- and once here rather than per query,
        # since the forms cannot change after construction. ``None`` where
        # this fold was seen to merge token boundaries, because then no
        # number is an upper bound: see :meth:`_merges_tokens`.
        self.longest_form = (
            None if merges else max((len(token_spans(form)) for form in self.by_form), default=0)
        )

    @staticmethod
    def _merges_tokens(form: str, folded: str) -> bool:
        """Whether folding ``form`` ran two of its tokens together.

        The window bound a scan spends rests on one implication: a query
        window of *L* tokens can only match a key of *L* tokens or more, so
        the widest key bounds the widest useful window. That holds exactly
        while the fold preserves token boundaries, and a fold which deletes
        them breaks it in a direction no wider number repairs -- under
        ``re.sub(r"[^0-9a-z]", "", ...)`` the two-token ``Golden Retriever``
        and the fifteen-token ``g o l d e n r e t r i e v e r`` both reach one
        one-token key, and nothing stops a query spelling it wider still.

        So this reports the condition rather than trying to price it, and a
        source that sees it declines to bound at all. That costs the scan its
        linearity over such a vocabulary and costs it no answers, which is the
        right way round: a bound that is wrong is a declared form the scan
        silently stops finding, which is the failure
        :meth:`MappingEntitySource.longest_form_tokens` exists to prevent.

        **Seen, not proven.** The check reads the forms the vocabulary
        declares, so it cannot see a fold that merges only on some query no
        declared form resembles -- ``default_normalizer`` casefolds the
        non-alphanumeric ``U+0345`` to the alphanumeric ``iota``, so a
        vocabulary declaring the folded spelling and a query using the
        combining one part here undetected. A rung that folds for itself is
        outside this entirely, and declines the bound for that reason rather
        than this one.
        """
        return len(token_spans(folded)) < len(token_spans(form))

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

    def forms(self) -> Iterable[str]:
        """Every folded form this index holds, in the order it built them.

        A view rather than a copy: the index is immutable after construction,
        and a rung reading every form on every query is the caller this
        exists for.
        """
        return self.by_form.keys()

    def alias_form(self, form: str) -> frozenset[str]:
        return self.by_alias.get(self.normalizer(form), frozenset())

    def of_type(self, type_id: str) -> frozenset[str]:
        return self.by_type.get(type_id, frozenset())

    def longest_form_tokens(self) -> int | None:
        return self.longest_form

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

    def surface_forms(self) -> Iterable[str]:
        """Every form this source carries, folded the way its index holds them.

        Satisfies
        :class:`~dataknobs_common.entity_resolution.SurfaceFormCatalog` for free -- the
        shared index already keys its lookup map by the folded form, so this
        is a view over what was built at construction rather than a second
        pass over the vocabulary.

        **Folded rather than as written**, which is the spelling a comparison
        wants: a rung scoring a query window against these forms and then
        resolving the winner through :meth:`by_surface_form` would otherwise
        score one spelling and look up another. What it costs is that a
        caller cannot recover the display spelling from this member; they
        fetch the entity, which is where the vocabulary keeps it.

        The order is the vocabulary's own, so two runs over the same source
        enumerate identically -- a rung breaking a tie by arrival would
        otherwise return a different winner on a different process.
        """
        return self._index.forms()

    def by_type(self, type_id: str) -> frozenset[str]:
        """The ids of every entity of this type."""
        return self._index.of_type(type_id)

    def longest_form_tokens(self) -> int | None:
        """How many tokens the longest form this source declares occupies.

        **What a scanning rung needs to stop enumerating.** A rung that looks
        up every contiguous window of a query spends *n(n+1)/2* lookups for
        *n* tokens, and every window longer than this answer is one no
        declared form could fill -- so the bound turns a cost quadratic in the
        caller's input into one linear in it, without changing an answer. See
        :class:`~dataknobs_common.entity_resolution.ScanningSignal`.

        Counted with :func:`~dataknobs_common.text.token_spans`, which is the
        same boundary policy a probe is sliced on: an answer counted any other
        way would bound the wrong enumeration.

        ``0`` for a vocabulary declaring no forms at all, which is the honest
        answer and stops a scan before its first lookup.

        **``None`` where this source's own fold was seen to merge token
        boundaries**, which is the case no number describes. The bound rests
        on a window of *L* tokens needing a key of *L* tokens or more, and a
        fold deleting the characters :func:`~dataknobs_common.text.token_spans`
        reads as boundaries breaks that implication without putting any other
        number in its place: under one, ``Golden Retriever`` and
        ``g o l d e n r e t r i e v e r`` reach the same one-token key and a
        query may spell it wider still. So the source says it cannot bound,
        the scan enumerates in full, and the vocabulary keeps every answer it
        had at the cost of the linearity it never could have had. See
        :meth:`~dataknobs_common.ontology.sources._EntityIndex._merges_tokens`
        for what the check sees and what it cannot.
        """
        return self._index.longest_form_tokens()


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

    async def surface_forms(self) -> Iterable[str]:
        """Every form this source carries, folded the way its index holds them.

        Satisfies
        :class:`~dataknobs_common.entity_resolution.AsyncSurfaceFormCatalog` for free -- the
        shared index already keys its lookup map by the folded form, so this
        is a view over what was built at construction rather than a second
        pass over the vocabulary.

        **Folded rather than as written**, which is the spelling a comparison
        wants: a rung scoring a query window against these forms and then
        resolving the winner through :meth:`by_surface_form` would otherwise
        score one spelling and look up another. What it costs is that a
        caller cannot recover the display spelling from this member; they
        fetch the entity, which is where the vocabulary keeps it.

        The order is the vocabulary's own, so two runs over the same source
        enumerate identically -- a rung breaking a tie by arrival would
        otherwise return a different winner on a different process.
        """
        return self._index.forms()

    async def by_type(self, type_id: str) -> frozenset[str]:
        """The ids of every entity of this type."""
        return self._index.of_type(type_id)

    def longest_form_tokens(self) -> int | None:
        """Synchronous on both twins, and for ``describe``'s reason.

        The answer is fixed at construction and reaches for nothing, so an
        ``await`` here would cost every caller a suspension to read a number
        this object already holds. See
        :meth:`MappingEntitySource.longest_form_tokens`.
        """
        return self._index.longest_form_tokens()


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
        # The optional protocols the mapping sources also satisfy. They are
        # here for the reason the four above are, and for one more: a source
        # satisfies an optional protocol *structurally*, so nothing would
        # otherwise notice a member renamed out from under the rung that
        # checks for it -- the rung would simply stop finding it and report
        # an empty vocabulary.
        catalogue: SurfaceFormCatalog = MappingEntitySource({})
        async_catalogue: AsyncSurfaceFormCatalog = AsyncMappingEntitySource({})
        aliases: AliasFormSource = MappingEntitySource({})
        async_aliases: AsyncAliasFormSource = AsyncMappingEntitySource({})
        del entities, async_entities, assertions, async_assertions
        del catalogue, async_catalogue, aliases, async_aliases

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The resolution rung that searches a semantic index.

``dataknobs_common`` ships four rungs that read an entity source directly, and
``dataknobs_xization`` ships one that reads an authority stack. This is the
rung whose backing is a **vector store**: it proposes entities whose
*description* sits near the query in embedding space, which is how a
vocabulary answers a phrasing nobody enumerated and no near-spelling reaches.

**It lives here rather than in** ``dataknobs_common`` **because it holds a**
:class:`~dataknobs_data.vector.SemanticIndex`, and an index holds a store.
That is also why there is no synchronous twin, and the absence is a design
rather than a gap: every member of a ``VectorStore`` is an ``async def``, so
nothing about a query *string* can be answered without a round trip. A rung
taking a query **vector** could be natively synchronous over the backends
that support it, and would be a different class with a different constructor
rather than this one's twin.

**Top level rather than under** ``vector/``, because this is a member of the
*resolution* family --- whose other members are in
``dataknobs_common.entity_resolution`` and
``dataknobs_xization.entity_resolution`` --- and filing it under ``vector/``
would name its backing instead of its role. The index it searches is under
``vector/`` for the mirror reason: that one *is* a vector construct.

Importing this module registers the rung under ``kind: "semantic"`` in
``dataknobs_common``'s asynchronous registry, which is what clears the mark
that package leaves for the key. A consumer reaching the rung through an
:class:`~dataknobs_data.ontology.OntologyRegistry` never imports it by hand:
that registry imports it to build a cascade.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_common.entity_resolution.registry import (
    async_signal_backends,
    declared_signal_metadata,
)
from dataknobs_common.entity_resolution.signals import refuse_negative_k
from dataknobs_common.entity_resolution.values import (
    EntityCandidate,
    EvidenceKind,
    MatchEvidence,
    Scoring,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import ONTOLOGY_ID_KEY, AsyncOntology

from dataknobs_data.vector import SemanticIndex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataknobs_common.entity_resolution.protocols import AsyncMatchSignal

    from dataknobs_data.vector.stores.base import VectorSearchResult

__all__ = ["SemanticSignal"]

logger = logging.getLogger(__name__)

#: The registered key, and the string a consumer writes as ``kind:``.
_KEY = "semantic"


class SemanticSignal:
    """Propose entities whose indexed text sits near the query.

    Example:
        ```python
        rung = SemanticSignal(registry.index("catalog"), registry.get("catalog"))
        found = await rung.candidates("the orange one", k=5)
        ```

    **Written against the bare protocol rather than against**
    :class:`~dataknobs_common.entity_resolution.AsyncDeclaredSignal`, and the
    three reasons are structural rather than stylistic --- which is the case
    :class:`~dataknobs_common.entity_resolution.DeclaredSignal`'s own
    docstring reserves the protocol for, *a rung whose backing is not a
    dictionary lookup*:

    * that base's constructor takes an
      :class:`~dataknobs_common.ontology.sources.AsyncEntitySource`; this rung
      takes an index;
    * its pipeline runs through
      :class:`~dataknobs_common.entity_resolution.values.FormHit`, whose
      ``span`` is **required** --- and a cosine neighbour has no position in
      the query, which is what :attr:`EvidenceKind.INFERRED` with ``span=None``
      says here;
    * its ``_hits`` hook answers a ``frozenset[str]`` and discards order, and
      a ranked answer *is* an order.

    **It holds the vocabulary as well as the index, and that is the id
    space.** Every row an
    :class:`~dataknobs_common.ontology.EntitySourceIndexSource` writes carries
    a **qualified** id, and
    :attr:`~dataknobs_common.entity_resolution.values.EntityCandidate.entity_id`
    is in the *resolver's ontology's* space --- so a rung answering the
    store's ids puts two id spaces in one candidate list, where the cascade
    de-duplicates neither and one entity is returned twice under two keys.
    The conversion is :meth:`~dataknobs_common.ontology.AsyncOntology.localize`
    and nothing else: re-spelling the parse here would be a second spelling of
    a rule that is already published, and wrong besides for a consumer key
    with a codec.

    **It scopes every read to its own vocabulary**, and that is a separate
    guarantee from the localization above rather than a restatement of it.
    Localizing fixes the id space a hit is *answered* in; the scope fixes
    which rows are searched at all, because a store may hold more than one
    vocabulary's and nothing about holding an index makes it exclusively
    this ontology's. See :meth:`_search`.

    **It opens nothing and closes nothing.** The index is the caller's --- in
    the configured path, the registry's, released by
    :meth:`~dataknobs_data.ontology.OntologyRegistry.close` --- and so is the
    store inside it. A rung held across that ``close()`` inherits what
    :meth:`~dataknobs_data.ontology.OntologyRegistry.index` already records
    about an index held across one: a live object over a dead store.
    """

    #: The registered key, and the string the evidence carries as its
    #: ``signal``, so a caller reading ``evidence.signal`` can correlate a hit
    #: back to the ``kind:`` they configured.
    key = _KEY

    #: Declared rather than inherited, because this rung subclasses no base
    #: that would declare it --- and **written down rather than left to the
    #: default**, because two load-time refusals in
    #: :mod:`dataknobs_data.ontology.registry` read this key off the registry
    #: by name. A fact that is true only because nobody set it is a fact no
    #: reader can find.
    #:
    #: ``False``: this rung matches an *embedding*, not a folded surface form,
    #: so a binding that publishes no folded-form lookup is no obstacle to it.
    reads_surface_forms = False

    #: ``False``, for :attr:`reads_surface_forms`'s reason and the other
    #: refusal's question: this rung embeds the query whole and probes no
    #: windows, so its cost does not depend on the vocabulary's longest
    #: declared form.
    bounded_by_longest_form = False

    #: A neighbourhood rather than a form the vocabulary carries.
    kind = EvidenceKind.INFERRED

    #: A cosine number is backend-defined, which is what this member means ---
    #: and why
    #: :data:`~dataknobs_common.entity_resolution.values._NORMALIZING` declines
    #: it for a distribution.
    scoring = Scoring.NATIVE

    def __init__(
        self,
        index: SemanticIndex,
        ontology: AsyncOntology[str],
        *,
        threshold: float | None = None,
    ) -> None:
        """Args:
        index: What to search. Already built and already configured --- this
            rung neither builds it nor opens it. An index whose store holds
            no rows answers nothing, which :meth:`candidates` reports rather
            than returning as an ordinary empty answer.
        ontology: The vocabulary the index was built over, and the id space
            this rung answers in. The **same object** the cascade decides a
            scope against, which is the point of taking it here rather than
            an injected conversion: a candidate must be in the cascade's
            source's space, and only the ontology knows what that is. It is a
            value with no lifecycle, so holding one costs nothing.
        threshold: Drop hits scoring below this. ``None`` --- the default ---
            keeps every hit the store returns, which is what an unthresholded
            cosine search means: the *k* nearest rows, whatever the query.
            Supplying one is how a document writing ``kind: semantic`` says
            how near is near enough.

        Raises:
            ValidationError: When the index's ids are not this ontology's ---
                measured at construction off the source's own ``declares()``,
                because the alternative is a rung that constructs cleanly and
                fails on every hit it finds.

                **Membership, which is not exclusivity.** This settles that
                the index holds *these* ids and says nothing about rows some
                other vocabulary wrote into the same store --- a question no
                constructor can answer, because the answer changes with every
                write. That one is settled per read, by the scope
                :meth:`_search` sends.
        """
        self._index = index
        self._ontology = ontology
        self._threshold = threshold
        self._rows_checked = False

        # Refused here rather than left to `localize`, which would raise on
        # the first hit and name an id rather than the mismatch. The member
        # read is the one written for this question: *"a caller asking for
        # results within a named set needs the source to have said which sets
        # it writes, or membership cannot be decided"*. An empty answer is a
        # source over a bare table, whose ids are local and fall in no named
        # space -- a complete answer, and one this rung cannot localize.
        declared = index.source.declares()
        if ontology.id not in declared:
            raise ValidationError(
                f"this index does not hold ontology {ontology.id!r}'s ids: its source "
                f"declares {sorted(declared) or 'no named set at all'}. A rung answers "
                f"in the resolver's ontology's id space and localizes on the way out, "
                f"so an index over another vocabulary -- or over a bare table, whose "
                f"ids are local -- has nothing this rung can convert",
                context={"ontology_id": ontology.id, "declares": sorted(declared)},
            )

    @property
    def name(self) -> str:
        """The key this rung is registered under."""
        return self.key

    def narrows(self) -> bool:
        """False: no row this index holds carries a scope axis.

        **This is about the cascade's scope and not about filtering at all.**
        The rung does send the store a filter --- its own vocabulary's tag,
        see :meth:`_search` --- and the two are different questions over
        different keys. What is declined here is the *scope* a caller names,
        which is rendered as ``{entity_type: [...]}``; what is always sent is
        ``ONTOLOGY_ID_KEY``, which every row carries because the source that
        wrote them declares it.

        **Not a property of vector search, and not permanent.** A store's
        metadata filter is key-equality with a documented rule --- *a missing
        metadata key fails the filter* --- and a scope is rendered as
        ``{entity_type: [...]}``. The rows an
        :class:`~dataknobs_common.ontology.EntitySourceIndexSource` writes
        carry the ontology id and the alias forms and nothing else, so a
        filter naming ``entity_type`` matches **every backend's nothing**: a
        rung that forwarded it would answer the empty list under every scope
        and read as *not in the corpus*. That is the same rule the ontology
        tag *passes*, and why one is sent and the other is not.

        Answering ``False`` costs this rung its ``k`` slots on candidates the
        cascade may drop, and buys that the cascade rules rather than the
        store silently emptying. That trade is the one the cascade is built
        to make ---
        :class:`~dataknobs_common.entity_resolution.values.Coverage` is where
        a candidate outside the scope lands --- and a rung declining is a
        cost rather than a correctness question.

        It becomes ``True``, per axis, when a row carries an axis key.
        """
        return False

    async def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate[str]]:
        """At most ``k`` entities whose indexed text sits nearest ``query``.

        A *caller-supplied* ``filter`` is accepted for the protocol's shape
        and is **not forwarded**, which :meth:`narrows` is the published
        statement of: a rung answering ``False`` is never offered one by a
        cascade, and the scope a cascade would name matches no row this index
        holds. The rung's own vocabulary filter is a different key and is
        always sent --- see :meth:`_search`.
        """
        [found] = await self._search([query], k)
        return found

    async def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate[str]]]:
        """One answer per query, in the order asked.

        **A real batch path rather than the protocol's loop.** The index
        publishes :meth:`~dataknobs_data.vector.SemanticIndex.search_batch`
        with ``search``'s keywords, so a batch embeds every query in one call
        instead of one call each --- which is the whole of what a batch
        resolve over a corpus buys.

        ``filter`` is read exactly as :meth:`candidates` reads it, because
        both are wrappers over one search: a caller's is not forwarded, and
        this rung's own vocabulary scope always is.
        """
        return await self._search(list(queries), k)

    async def _search(self, queries: list[str], k: int) -> list[list[EntityCandidate[str]]]:
        """The one search both public members are thin wrappers over.

        Written once because the two differ in arity and in nothing else,
        which is :meth:`~dataknobs_data.vector.SemanticIndex._search_many`'s
        argument one layer up --- and because the two things this method does
        *besides* searching are exactly the two that must not differ between
        them.

        **The read is scoped to this rung's own vocabulary, always.** A store
        is not this rung's, and nothing makes it this ontology's either: the
        registry caches a vector store on its resolved ``store:`` block alone,
        so two documents writing the same block share one store, and a
        ``table:`` or a collection two deployments name is shared by
        construction. An unscoped read over such a store answers rows the
        cascade's ontology never declared, whose ids
        :meth:`~dataknobs_common.ontology.AsyncOntology.localize` then refuses
        --- so the rung that *constructed* cleanly failed on every hit it
        found, which is the outcome the constructor's own guard names and
        cannot prevent by itself: membership is not exclusivity, and
        ``declares()`` answers the first question.

        The key is the one the source writes.
        :meth:`~dataknobs_common.ontology.EntitySourceIndexSource.declares`
        states the two as one fact --- *an ontology id declared, and
        ``ONTOLOGY_ID_KEY`` stored on the row, one for a scope filter to
        translate against and one to match it* --- and the protocol's own
        ``declares()`` says a named set is *what a scope filter is translated
        against*. This is that translation, and it is the first one in the
        tree.

        ``k`` is refused here rather than at each public member, and through
        the published check rather than a fourth spelling of it: every rung
        assembled by
        :func:`~dataknobs_common.entity_resolution.declared_candidates` calls
        the same function, and this rung does its own assembly.
        """
        refuse_negative_k(k)
        batches = await self._index.search_batch(
            queries, k=k, threshold=self._threshold, filter=self._scope
        )
        found = [
            [self._candidate(hit, query) for hit in hits]
            for query, hits in zip(queries, batches, strict=True)
        ]
        if queries and not any(found):
            await self._report_an_empty_answer()
        return found

    @property
    def _scope(self) -> dict[str, str]:
        """The filter every read of this rung carries: its own vocabulary's tag."""
        return {ONTOLOGY_ID_KEY: self._ontology.id}

    def _candidate(self, hit: VectorSearchResult, query: str) -> EntityCandidate[str]:
        """One hit, in the ontology's id space, with this rung's reason for it.

        One piece of evidence rather than several: the index writes one row
        per entity, so an entity is found in one place or not at all --- which
        is the difference from a rung over surface forms, where one entity is
        reachable at several spans.

        ``matched_text`` is the query itself and ``span`` is ``None``,
        together: a cosine neighbour was proposed by the *whole* query and sits
        at no position in it, so a span here would be invented and a
        ``matched_text`` sliced from one would be a slice of nothing.

        **A hit with no id is refused rather than dropped or repaired.** The
        published read door types a record's id as ``str | None``, and both
        other readings are the ones this family refuses everywhere else:
        skipping loses a candidate silently, and coercing turns an absence
        into an id that resolves to nothing. Refusing names the store while
        the caller can still act on it -- where the alternative, measured, is
        an ``AttributeError`` raised inside the ontology's id parser, three
        frames from anything naming a rung.

        Raises:
            ValidationError: For a hit whose record carries no id.
        """
        if hit.record.id is None:
            raise ValidationError(
                f"the store behind this index answered a hit with no record id, so "
                f"there is nothing to place in the id space of ontology "
                f"{self._ontology.id!r}. Every row an index writes is keyed by its "
                f"source's item id; a hit without one is a row this rung cannot have "
                f"written",
                context={"ontology_id": self._ontology.id, "score": hit.score},
            )
        return EntityCandidate(
            entity_id=self._ontology.localize(hit.record.id),
            score=hit.score,
            evidence=(
                MatchEvidence(
                    signal=self.key,
                    kind=self.kind,
                    score=hit.score,
                    scoring=self.scoring,
                    matched_text=query,
                    span=None,
                ),
            ),
        )

    async def _report_an_empty_answer(self) -> None:
        """Say so when this rung answered nothing because nothing was there to find.

        **The count is what separates the cases, and nothing else does.** An
        unthresholded search answers the *k* nearest rows whatever the query,
        so a rung returning nothing over a populated store is already
        anomalous --- but a rung carrying a ``threshold`` may legitimately
        answer nothing over a full one. Two states are worth a sentence and
        neither is visible from the answer:

        * **the store holds no rows at all** --- ``load()`` assembles an index
          and returns, and filling it is the caller's line, so a cascade
          resolving before that call is the ordinary first-run mistake;
        * **the store holds rows and none of them are this vocabulary's** ---
          which is the shared-store case :meth:`_search` scopes against. The
          scope is doing its job and the answer is correct; what it means is
          that *this* ontology's index was never built, into a store where
          another one's was. Without this the two are one silent empty list.

        **Probed once per rung instance, and the flag is set whatever the
        answer is.** It used to be set only on the empty branch, so a store
        with rows was re-counted on every empty answer for the life of the
        process --- and a ``count()`` is a ``SELECT COUNT(*)`` on pgvector and
        a metadata walk on the others whenever a filter applies, which is a
        per-query sequential scan under any threshold a real deployment sets.
        Nothing here is O(1): the earlier claim that it was named the memory
        backend's unfiltered length and generalised it, and neither half
        survives a ``domain_id`` or a real database.

        Racy in the harmless direction: two concurrent resolves can both find
        the flag unset and both probe, which costs a second count and cannot
        produce a second report of a different thing.

        **Why this is a log and not a refusal.** The index a registry hands a
        rung has never been built: ``load()`` assembles it and returns, and
        building it is the caller's line --- so refusing here would refuse
        every first load. The earliest moment this state is distinguishable
        from the ordinary one is the first resolve that finds nothing, which
        is here.
        """
        if self._rows_checked:
            return
        self._rows_checked = True
        if await self._index.store.count() == 0:
            logger.warning(
                "the %r rung over ontology %r found nothing and its index holds no rows: "
                "an index is built by its caller, not by load() -- call "
                "`await registry.index(%r).build()` once before resolving",
                self.key,
                self._ontology.id,
                self._ontology.id,
            )
            return
        if await self._index.store.count(self._scope) == 0:
            logger.warning(
                "the %r rung over ontology %r found nothing and its store holds no row "
                "tagged %s=%r, though it holds rows for something else: this store is "
                "shared -- two documents naming one `store:` block, or one table -- and "
                "this vocabulary's index was never built into it. Call "
                "`await registry.index(%r).build()`, or give this ontology a store of "
                "its own",
                self.key,
                self._ontology.id,
                ONTOLOGY_ID_KEY,
                self._ontology.id,
                self._ontology.id,
            )


def _make_semantic(config: dict[str, Any]) -> AsyncMatchSignal[str]:
    """Build the rung from a document's spec plus the handles a door forwarded.

    **The two handles are not a document's to write**, which is why their
    absence is refused here with a sentence naming the door that supplies
    them: a live index and a loaded vocabulary are objects, and a YAML file
    cannot hold one. :func:`~dataknobs_common.ontology.loader.async_build_resolver`
    forwards whatever its caller hands it; the caller that has both is
    :class:`~dataknobs_data.ontology.OntologyRegistry`.

    **Presence is not the check, and it used to be.** Handles and document
    keys share one namespace after that door merges them --- handles win where
    both spell a key, which is the published rule --- so a key a document
    wrote under a handle's name reaches this factory untouched whenever no
    handle was supplied for it. ``index: my-index`` in a document then built a
    rung over the *string*, and the failure was an ``AttributeError`` from
    ``"my-index".source``, wrapped as ``Failed to create plugin 'semantic'``:
    the same shape as the missing-id defect one layer over, and the same
    remedy. So each handle is checked for what it has to be, and the refusal
    says which one arrived as what.

    Raises:
        ValidationError: When either handle is missing or is not the type it
            has to be, naming both and the door that supplies them. Without
            this the failure is a ``KeyError`` or an ``AttributeError`` out of
            a factory the caller never named.
    """
    required: tuple[tuple[str, type], ...] = (("index", SemanticIndex), ("ontology", AsyncOntology))
    missing = sorted(handle for handle, _ in required if config.get(handle) is None)
    if missing:
        raise ValidationError(
            f"rung kind {_KEY!r} needs {missing} and the configuration supplied "
            f"neither a value nor a handle for it. A document cannot write a live "
            f"index or a loaded vocabulary, so these arrive from the door that holds "
            f"both. Loading through OntologyRegistry? The handle it forwards is the "
            f"index your own document declares, so declare one: add an `index:` "
            f"section beside the `resolver:` section. Calling async_build_resolver "
            f"directly? Pass `handles={{'index': ..., 'ontology': ...}}`",
            context={"kind": _KEY, "missing": missing},
        )
    wrong = sorted(
        f"{handle}={type(config[handle]).__name__}"
        for handle, expected in required
        if not isinstance(config[handle], expected)
    )
    if wrong:
        raise ValidationError(
            f"rung kind {_KEY!r} was given {wrong} where a live handle belongs. These "
            f"two keys are not a document's to write -- a YAML file cannot hold an "
            f"index or a vocabulary -- so a value here is a document naming a key the "
            f"door was going to supply. Drop it from the `resolver:` section and let "
            f"OntologyRegistry forward the real one, or pass it through `handles=`",
            context={"kind": _KEY, "wrong": wrong},
        )
    return SemanticSignal(
        config["index"],
        config["ontology"],
        threshold=config.get("threshold"),
    )


#: What a door reads before building anything. ``needs_io`` is ``True``
#: because every answer this rung gives is a round trip to a store, which is
#: the fact that sends a document naming this kind to a registry rather than
#: to the value-level door. The two derived keys are read off the class by
#: :func:`~dataknobs_common.entity_resolution.declared_signal_metadata` rather
#: than restated, so a fact this rung gains is one both load-time refusals see
#: without an edit here.
_SEMANTIC_METADATA = declared_signal_metadata(
    SemanticSignal,
    {
        "flavour": "async",
        "needs_io": True,
        "requires_install": "pip install dataknobs-data",
    },
)


def _register_rung(*, override: bool = False) -> None:
    """Put the rung in ``dataknobs_common``'s asynchronous registry.

    Run at this module's import, which is what clears the mark that package
    leaves for this kind. The synchronous registry is left alone: there is no
    synchronous form of this rung to register, here or anywhere, which is what
    its mark there says.

    **A key a consumer already holds is left alone**, for the reason the
    ``authority`` registration one distribution over states: ``register``
    refuses a key it already holds, so without the check a consumer who
    registered their own rung first made any later ``import
    dataknobs_data.entity_resolution`` raise out of the import statement ---
    taking every other thing that module offers down with it. Skipping rather
    than overriding is the other half: a consumer's own rung winning is what
    that registry's own prose promises.

    Args:
        override: Replace whatever holds the key. For restoring this package's
            own registration --- a test that stood a rung of its own in the
            way and is putting ours back.
    """
    if override or not async_signal_backends.is_registered(_KEY):
        async_signal_backends.register(
            _KEY, _make_semantic, metadata=_SEMANTIC_METADATA, override=override
        )


_register_rung()

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A text-producing view of some data, bound to a vector store.

Three collaborators and no storage of its own: a source that streams text, an
embedder that turns text into vectors, and a store that holds them. What this
class adds is that the three agree --- the ids the source minted are the ids
the store is keyed on, and a hit comes back carrying the id the source emitted
rather than one the store invented.

**Not a store, and filed away from ``stores/`` so it does not look like one.**
A class under ``stores/`` invites a caller to reach for the store handle it
holds, and the whole value of this one is that the caller does not have to
know which backend is underneath.

**It does not synchronise.** Keeping stored vectors current as their sources
change is a contract that already ships, with two mechanisms and a document
describing them; a third implementation of it here would be a third thing to
keep in step. Rebuild, or use the synchroniser.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_common.exceptions import OperationError

from dataknobs_data.vector.content import foreign_model_names
from dataknobs_data.vector.types import DistanceMetric

if TYPE_CHECKING:
    from dataknobs_common.index import AsyncIndexSource

    from dataknobs_data.vector.embedding import TextEmbedder
    from dataknobs_data.vector.stores.base import VectorStore
    from dataknobs_data.vector.types import VectorSearchResult

logger = logging.getLogger(__name__)

__all__ = ["SemanticIndex"]

#: How many items one write to the store carries.
#:
#: The build streams, so this is what bounds what it holds at once. The store
#: batches its own embedding under a configured size; this bounds the list
#: handed to it, which is the half a store cannot bound for a caller.
BUILD_BATCH_SIZE = 1000


class SemanticIndex:
    """Binds a text-producing view of some data to a vector store.

    Example:
        ```python
        index = SemanticIndex(
            EntitySourceIndexSource(onto), embedder, store, metric=DistanceMetric.COSINE
        )
        written = await index.build()
        hits = await index.search("acme widget", k=5)
        ```
    """

    def __init__(
        self,
        source: AsyncIndexSource,
        embedder: TextEmbedder,
        store: VectorStore,
        *,
        metric: DistanceMetric | None = None,
    ) -> None:
        """Bind the three, and check the one claim that can be checked.

        Args:
            source: What streams the items. Its ids are this index's ids ---
                nothing here rewrites one or infers a namespace.
            embedder: What turns text into vectors. Its ``model_id`` is
                recorded on every row written, which is what makes a stored
                vector's staleness judgeable by something that never saw this
                object.
            store: Where the vectors go. Already built and already configured;
                this class neither opens it nor closes it.
            metric: **A claim about the store, not a setting on it.** The
                store owns its metric --- it resolves one at construction and
                ``search`` takes no such argument --- so a value here cannot
                change what the store does. What it can do is refuse a store
                that is not doing what the caller thought, which is worth more
                than a parameter that reads as a setting and silently is not.

                Compared in **canonical** form, because six member names name
                four metrics: a store configured ``l2`` and an index declaring
                ``EUCLIDEAN`` agree, and an assertion that fired on the
                spelling would refuse a correctly configured store while
                reading as the index catching a mistake.

                **No default**, because the only safe default is the one that
                checks nothing: defaulting to cosine would refuse every
                euclidean store a caller deliberately configured. ``None``
                means *do not check*.

        Raises:
            ValueError: When *metric* names a family the store is not serving.
        """
        self.source = source
        self.embedder = embedder
        self.store = store
        self.metric = metric

        #: Whether the staleness *log* has already fired on this instance.
        #:
        #: The read path carries no other once-per-instance state, and this
        #: is why it has any: a mismatch is a property of the store and the
        #: embedder, not of the query, so a caller running a thousand
        #: searches against a stale store has one fact to be told and would
        #: otherwise be told it a thousand times.
        #:
        #: It gates the log alone, not the comparison. Stopping the
        #: comparison too would cap :attr:`mismatched_model_ids` at whichever
        #: model happened to be seen first, which is the answer a caller
        #: deciding what to re-embed can least use.
        self._reported_stale = False

        #: Every foreign ``model_name`` this index's searches have ranked
        #: against, published through :attr:`mismatched_model_ids`.
        self._mismatched_model_ids: set[str] = set()

        if metric is not None:
            claimed = DistanceMetric.resolve(metric).canonical()
            actual = DistanceMetric.resolve(store.metric).canonical()
            if claimed is not actual:
                raise ValueError(
                    f"index declares {claimed.value!r} but its store is configured for "
                    f"{actual.value!r}. The store owns the metric; change the store's "
                    f"configuration, or drop the claim"
                )

    @property
    def mismatched_model_ids(self) -> list[str]:
        """Foreign model identities this index's searches have ranked against.

        Sorted, distinct, and empty until a search returns a row whose
        ``model_name`` differs from this index's embedder. Non-empty means
        the rankings rested on vectors from more than one embedding space,
        so scores --- and therefore any ``threshold`` applied to them --- are
        arithmetic on incomparable quantities.

        **Handed back as well as logged, because a log line is not an answer
        a program can act on.** The warning fires once per instance, so a
        service that starts before its log sink, or reads its logs nowhere,
        has the fact pass it by; this member is still there afterwards. The
        sibling is :attr:`~dataknobs_data.dedup.DedupResult.mismatched_model_ids`,
        which carries the same fact under the same name for the same stated
        reason --- *"every candidate is still the best answer available; what
        changes is that the caller can now tell the answer is untrustworthy"*.

        **Not a scan.** It reads nothing the searches did not already read,
        so it answers *which foreign models has this index been ranking
        against* and not *what wrote the rows in this store*. The second is
        the more general question and a different shape: undefined over a
        store several vocabularies share, and on some backends a full scan.

        Returns:
            A fresh sorted list, so a caller cannot edit this index's state
            by holding onto it.
        """
        return sorted(self._mismatched_model_ids)

    async def build(self) -> int:
        """Embed everything the source streams and write it to the store.

        A full build, not an update: what the source streams now is what the
        store holds after, for every id the source emits. Ids the source no
        longer emits are **left alone** --- this class does not decide that a
        row it did not just write is stale, because it cannot tell an entity
        that was deleted from a source that was narrowed.

        **Written in batches as the source yields them**, never collected
        whole. A source exists to be larger than memory, and a ``build`` that
        materialised the stream first would take that property away from every
        source at the one call site that consumes them all.

        **So it is not atomic, and a failure says how far it got.** Batching
        and all-or-nothing are incompatible without a transaction no vector
        store offers, and streaming is the property worth keeping. What that
        costs is that a raise partway through leaves earlier batches
        committed --- so the count travels on the exception rather than
        dying in a local. Without it a caller cannot tell a store holding
        nothing from one holding most of the corpus, and the two want
        opposite responses: retry from scratch is free over the first and
        wasteful over the second, while reporting the index unbuilt is true
        of the first and a lie about the second. The store cannot answer this
        --- it never saw the stream --- which is what makes it this method's
        to report.

        The ids go to the store explicitly. Without that every backend mints a
        ``uuid4`` per row, identically, and the id the source took care to
        emit would be discarded one call below the decision to emit it ---
        leaving every hit carrying a uuid that resolves to nothing.

        **The store writes the source-field pair, not this method.** Where the
        source can say what its text was composed from, that travels as a call
        argument and the store writes ``source_field`` beside the
        ``source_text`` it already writes. Read off the source with a
        fall-back rather than through a protocol member: the protocol has two
        members and a required third would break ``isinstance`` for every
        structural implementor outside this tree, which is a cost this buys
        nothing worth.

        Without it a row this index wrote and a row some other caller wrote
        through the raw vector door are **indistinguishable** at the published
        read door --- both answer ``vector_field=None`` --- so a consumer
        cannot tell an index hit from anything else in the same store.

        **An item may answer for itself**, through
        :attr:`~dataknobs_common.index.IndexItem.source_field`, and that
        answer travels by the store's own per-row route: a ``source_field``
        key in the row's metadata, which the store keeps over the call
        argument. It is needed where one source emits items of more than one
        kind --- ``AliasSource`` yields the inner's composed text and then
        each surface form, and one aggregate value describes only the first.
        The precedence is the item's own field, then a ``source_field`` the
        item's metadata already carried, then the source's answer; the item
        wins over its metadata because a decorator copies the inner item's
        metadata onto every form, and an inherited key would label an alias
        with the field the canonical text came from.

        Returns:
            How many items were **handed to the store**. ``0`` over an empty
            source is a legitimate answer and not an error: a vocabulary that
            holds nothing is a vocabulary.

            Not a row count, and the difference is observable: a decorator
            that emits several items under one id leaves one row, so a build
            answering ``3`` over a store holding ``1`` is the documented
            behaviour of that composition rather than a discrepancy.

        Raises:
            OperationError: When the source, the embedder or the store fails
                partway. ``context["written"]`` is how many items had already
                reached the store, and the original failure is the cause.
        """
        source_field = getattr(self.source, "source_field", None)
        written = 0
        ids: list[str] = []
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []

        async def flush() -> int:
            if not texts:
                return 0
            await self.store.bulk_embed_and_store(
                list(texts),
                ids=list(ids),
                metadata=list(metadata),
                embedder=self.embedder,
                source_field=source_field,
            )
            count = len(texts)
            ids.clear()
            texts.clear()
            metadata.clear()
            return count

        try:
            # `aclosing_iter` rather than a bare `async for`, because every exit
            # from this loop but the last one leaves the source's generator
            # suspended. An abandoned async generator runs its cleanup when
            # the interpreter finalizes it -- a later turn of the loop --
            # so what the source was holding was still held while the caller
            # decided what to do about the failure. Two shipped sources hold
            # something real there: `RecordFieldSource` over PostgreSQL
            # yields from inside `pool.acquire()` and an open transaction,
            # and over Elasticsearch from inside a scroll. A source's
            # end-of-stream report is the same question -- it arrives at the
            # close, so a build that never closes never sees it.
            async with aclosing_iter(self.source.stream_items()) as items:
                async for item in items:
                    ids.append(item.id)
                    texts.append(item.text)
                    row = dict(item.metadata)
                    if item.source_field is not None:
                        row["source_field"] = item.source_field
                    metadata.append(row)
                    if len(texts) >= BUILD_BATCH_SIZE:
                        written += await flush()
                written += await flush()
        except Exception as exc:
            # `written` is the last completed flush, so it is exactly what the
            # store holds from this build -- the partial batch in hand was
            # never sent. Logged as well as raised: the raise reaches the
            # caller, and the log reaches whoever is reading the process's
            # output when the caller swallows it.
            logger.warning(
                "semantic index build failed after %d item(s) were written; the store "
                "holds a partial build",
                written,
                exc_info=exc,
            )
            raise OperationError(
                f"semantic index build failed after it wrote {written} item(s); the "
                f"store holds a partial build. Rebuild, or resume from what is there -- "
                f"the ids are the source's, so a rebuild overwrites rather than duplicates",
                context={"written": written},
            ) from exc

        if written == 0:
            logger.info("semantic index build over an empty source; nothing written")
        return written

    async def search(
        self,
        text: str,
        *,
        k: int = 10,
        threshold: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """The rows nearest this text, best first.

        **No record fetcher is passed**, so a hit's ``record`` is the minimal
        one synthesised from stored metadata. Getting back to the unprojected
        row is the source's job through its own origin lookup, and having two
        routes to it is how the two start disagreeing.

        **An empty answer is worth pairing with
        :attr:`mismatched_model_ids`.** Searching a store built under another
        model produces scores a *threshold* tuned under this one will tend to
        reject, so "no results" and "the wrong embedder" look identical from
        the return value. The comparison behind that member runs on what the
        store answered with, before *threshold*, so it is populated in
        exactly the case the result list is not.

        Args:
            text: What to search for.
            k: How many hits to return at most.
            threshold: Drop hits scoring below this. ``None`` keeps them all.
            filter: A metadata filter, in the store's dialect.

        Returns:
            Hits, best first, at most *k* of them.
        """
        [results] = await self._search_many([text], k=k, threshold=threshold, filter=filter)
        return results

    async def search_batch(
        self,
        texts: list[str],
        *,
        k: int = 10,
        threshold: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[list[VectorSearchResult]]:
        """:meth:`search` over several texts, one result list each, in order.

        **The same keywords as :meth:`search`, with the same defaults**, and
        that parity is asserted by a test rather than maintained by care: this
        signature has lost a parameter its sibling kept once already, and a
        batch member that quietly drops an option is the kind of divergence
        nothing reports.

        Args:
            texts: What to search for, one query each.
            k: As :meth:`search`.
            threshold: As :meth:`search`.
            filter: As :meth:`search`.

        Returns:
            One list of hits per query, positionally aligned with *texts*.
        """
        return await self._search_many(texts, k=k, threshold=threshold, filter=filter)

    async def _search_many(
        self,
        texts: list[str],
        *,
        k: int,
        threshold: float | None,
        filter: dict[str, Any] | None,
    ) -> list[list[VectorSearchResult]]:
        """The one search, which both public members are thin wrappers over.

        Written once because the two differ in arity and in nothing else. A
        second copy would be where the two signatures drift, which is the
        failure the parity test exists to catch and this method exists to make
        impossible.
        """
        if not texts:
            return []
        vectors = await self.embedder.embed(texts)
        out: list[list[VectorSearchResult]] = []
        for vector in vectors:
            hits = await self.store.search_similar_records(
                np.asarray(vector, dtype=np.float32),
                k=k,
                filter=filter,
            )
            # Before the threshold, not after. What the store answered with
            # is the evidence; what survives *threshold* is the answer. A
            # mismatch depresses exactly the scores a threshold is compared
            # against, so filtering first is how the check goes silent on the
            # case it exists for -- see `_note_foreign_models`.
            self._note_foreign_models(hits)
            if threshold is not None:
                hits = [hit for hit in hits if hit.score >= threshold]
            out.append(hits)
        return out

    def _note_foreign_models(self, hits: list[VectorSearchResult]) -> None:
        """Record, and once per index report, rows written by another model.

        ``build`` writes the embedder's ``model_id`` onto every row, and the
        docstring on that parameter says what for: *"which is what makes a
        stored vector's staleness judgeable by something that never saw this
        object"*. Nothing compared it. Measured: a store built under one
        model and searched through another returned three ranked hits and
        said nothing, at any level --- so the datum was recorded and there
        was nowhere to stand to read it.

        **The evidence is what the store answered with, which is why this
        runs before the caller's threshold.** A threshold is applied to
        scores, and a score computed across two embedding spaces is the
        thing a mismatch corrupts --- so the likeliest presentation of this
        defect is an empty result list, and filtering first made that the
        one presentation the check could not see. Measured on the unfixed
        code: the store answered a cross-model query with three rows, every
        one of them carrying the foreign name, and a threshold removed them
        from the answer and from the comparison together.

        **The real limit is narrower, and it stands.** The name is legible in
        the metadata *of a hit*, so nothing can be said about a query the
        **store** answered with nothing, and nothing can be said before a
        query at all. Answering *what model wrote these rows* without one is
        the more general fix and a different shape: a scan on some backends,
        and undefined over a store several vocabularies share.

        **Absent is not disagreement, and that rule is not this method's.**
        ``add_records`` omits the key when the field carries no name, so a
        row written before the key existed, or by an embedder with no
        ``model_id``, has nothing to compare and is passed over --- as does
        an index whose own embedder is unnamed. Which rows those are is
        ``foreign_model_names``' answer rather than a loop here: this rule
        was restated at five sites, and two of the copies had already come to
        disagree about an empty name on either side. ``DedupChecker`` is one
        of the two and now asks the same function, so *one* name for the
        fact is matched by one rule behind it.

        **The log stops at one; the record does not.** A caller has one fact
        to be told and would otherwise be told it on every query. But a store
        can hold rows from several earlier models, and *which ones* is what
        a caller deciding what to re-embed needs --- so every foreign name
        is kept for :attr:`mismatched_model_ids` even after the warning has
        fired. The line names every model *this* search ranked against;
        anything later reaches only the member, which is what the line says.

        Args:
            hits: One query's results **as the store returned them**, before
                any threshold filtering.
        """
        mine = getattr(self.embedder, "model_id", None)
        foreign = foreign_model_names((hit.metadata for hit in hits), mine)
        if not foreign:
            return
        self._mismatched_model_ids.update(foreign)
        if self._reported_stale:
            return
        self._reported_stale = True
        logger.warning(
            "semantic index searched with embedder %r returned rows written by %s; "
            "the stored vectors and the query vector come from different models, so "
            "the ranking is arithmetic on incomparable quantities. Rebuild the index, "
            "or search with the model that wrote it. Logged once per index; every "
            "such model, including any seen after this line, is named by "
            "`mismatched_model_ids`",
            mine,
            ", ".join(repr(name) for name in foreign),
        )

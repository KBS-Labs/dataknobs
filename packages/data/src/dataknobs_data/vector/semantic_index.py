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

from dataknobs_common.exceptions import OperationError

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

        if metric is not None:
            claimed = DistanceMetric.resolve(metric).canonical()
            actual = DistanceMetric.resolve(store.metric).canonical()
            if claimed is not actual:
                raise ValueError(
                    f"index declares {claimed.value!r} but its store is configured for "
                    f"{actual.value!r}. The store owns the metric; change the store's "
                    f"configuration, or drop the claim"
                )

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
            async for item in self.source.stream_items():
                ids.append(item.id)
                texts.append(item.text)
                metadata.append(dict(item.metadata))
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
            if threshold is not None:
                hits = [hit for hit in hits if hit.score >= threshold]
            out.append(hits)
        return out

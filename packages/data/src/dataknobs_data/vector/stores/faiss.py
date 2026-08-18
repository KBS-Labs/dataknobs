"""Faiss vector store implementation."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..types import DistanceMetric
from .base import VectorStore
from .config import FaissVectorStoreConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import ClassVar

    import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissVectorStore(VectorStore):
    """Faiss-based vector store for efficient similarity search.

    Faiss is a library for efficient similarity search and clustering of dense vectors.
    It provides various index types optimized for different use cases:
    - Flat: Exact search, best for small datasets
    - IVF: Inverted file index, good for medium datasets
    - HNSW: Hierarchical navigable small world, good for large datasets
    """

    CONFIG_CLS: ClassVar[type[FaissVectorStoreConfig]] = FaissVectorStoreConfig

    def _setup(self) -> None:
        """Initialize Faiss-specific derived config and runtime state."""
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed. Install with: pip install faiss-cpu")

        super()._setup()

        # Determine index type. The explicit ``index_type`` config key
        # wins; otherwise fall back to ``index_params["type"]`` (default
        # ``"auto"``), preserving the legacy dual-source precedence.
        self.index_type = (
            self.config.index_type
            if self.config.index_type is not None
            else self.index_params.get("type", "auto")
        )

        self.nlist = self.index_params.get("nlist", 100)  # For IVF
        self.m = self.index_params.get("m", 32)  # For HNSW
        self.ef_construction = self.index_params.get("ef_construction", 200)  # For HNSW
        self.ef_search = self.index_params.get("ef_search", 50)  # For HNSW search
        self.nprobe = self.search_params.get("nprobe", 10)  # For IVF search

        # ``faiss`` ships no type information, so the live index is typed
        # the way every helper that builds one already is: the concrete
        # class varies by index type and swaps under ``_build_deferred_ivf``.
        self.index: Any = None
        self.id_map = {}  # Map from our IDs to Faiss internal indices
        self.metadata_store = {}  # Store metadata separately
        # internal_id -> (created_at, updated_at). Aware UTC datetimes.
        # Keyed by internal id to match ``metadata_store`` so the
        # shared update_metadata_where helper, get_vectors, search,
        # delete, and clear all stay consistent. Pickle-persisted in
        # the ``.meta`` side-car; legacy pickles without this key load
        # as empty (rows return None/None on include_timestamps).
        self.timestamps: dict[int, tuple[datetime, datetime]] = {}
        # internal_id -> stored (already-normalized) vector row. Same
        # key space and side-car pattern as ``metadata_store`` /
        # ``timestamps``: ``get_vectors`` serves from here rather than
        # ``faiss`` ``reconstruct``. FAISS reconstruct-by-id is not
        # usable across all index types (IVF needs a maintained direct
        # map, which is mutually exclusive with ``remove_ids`` in this
        # faiss build), so the index is kept purely for ``search`` and
        # the authoritative vectors live in this side-car. Pickle-
        # persisted in the ``.meta`` side-car; legacy pickles without
        # this key load empty (rows return None until re-ingested).
        self.vectors: dict[int, np.ndarray] = {}
        self.next_idx = 0
        # True when the configured index type is IVF (``ivfflat`` /
        # ``ivfpq``) but fewer than ``nlist`` vectors have been added,
        # so ``self.index`` is currently a temporary flat index. FAISS
        # IVF k-means training requires >= ``nlist`` points, so the
        # real IVF index is built and migrated to (from the side-car)
        # the first time the corpus reaches that threshold. Persisted
        # in the ``.meta`` side-car; legacy pickles without the key
        # load ``False`` (a persisted IVF index is necessarily trained,
        # since pre-fix a sub-``nlist`` first batch could not be added).
        self._deferred_ivf: bool = False
        # Whether the "this store's vector side-car is short" warning has
        # already been emitted. The condition is a property of the loaded
        # file, not of a query, so it holds for every filtered search
        # this instance serves — warning per call would put one line per
        # user turn in the log of a RAG read path, all of them naming the
        # same one-off remedy. Same per-instance shape, and the same
        # reason for it, as ``_timestamp_collision_warned``.
        self._sidecar_shortfall_warned: bool = False

    async def initialize(self) -> None:
        """Initialize Faiss index."""
        if self._initialized:
            return

        # Create index based on type and metric. For a persisted store
        # this is only the fresh-store fallback: load() below overwrites
        # both self.index (via read_index) and self._deferred_ivf (from
        # the pickle), so a reloaded trained IVF correctly ends with
        # _deferred_ivf=False even though _create_index set it True.
        self.index = self._create_index()

        # Load any existing persisted index. ``load`` offloads its own
        # existence check + disk read off the loop and is a no-op when no
        # file exists, so the blocking ``os.path.exists`` stat that used
        # to run here is gone.
        if self.persist_path:
            await self.load()

        self._initialized = True

    def _faiss_metric(self) -> Any:
        """Map the configured distance metric to a FAISS metric const."""
        if self.metric == DistanceMetric.COSINE:
            # Cosine: vectors are normalized, then inner product.
            return faiss.METRIC_INNER_PRODUCT
        if self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            return faiss.METRIC_L2
        if self.metric in (
            DistanceMetric.DOT_PRODUCT,
            DistanceMetric.INNER_PRODUCT,
        ):
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2

    def _new_raw_index(self, index_type: str) -> Any:
        """Build an unwrapped FAISS index of the given concrete type.

        Single construction site for every index type so the temporary
        flat fallback and the deferred IVF build cannot drift apart.
        """
        dimensions = self.dimensions
        metric = self._faiss_metric()

        if index_type == "flat":
            if metric == faiss.METRIC_INNER_PRODUCT:
                return faiss.IndexFlatIP(dimensions)
            return faiss.IndexFlatL2(dimensions)

        if index_type == "ivfflat":
            # The coarse quantizer always uses L2 for k-means clustering
            # (standard FAISS practice) even when the search metric is
            # inner-product; the search metric is passed separately below.
            quantizer = faiss.IndexFlatL2(dimensions)
            if metric == faiss.METRIC_INNER_PRODUCT:
                return faiss.IndexIVFFlat(quantizer, dimensions, self.nlist, metric)
            return faiss.IndexIVFFlat(quantizer, dimensions, self.nlist)

        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimensions, self.m, metric)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index

        if index_type == "ivfpq":
            m = 8  # Number of subquantizers
            quantizer = faiss.IndexFlatL2(dimensions)
            return faiss.IndexIVFPQ(quantizer, dimensions, self.nlist, m, 8)

        raise ValueError(f"Unknown index type: {index_type}")

    def _create_index(self) -> Any:
        """Create the live FAISS search index from configuration.

        For the IVF family (``ivfflat`` / ``ivfpq``) the live index
        starts as a temporary **flat** index: FAISS IVF training needs
        >= ``nlist`` points, so a sub-``nlist`` cold start would
        otherwise crash on ``add_with_ids``. ``_build_deferred_ivf``
        swaps in the real IVF (rebuilt from the side-car) once the
        corpus reaches ``nlist``. ``self._deferred_ivf`` tracks which
        mode the live index is in.
        """
        # Auto-select index type based on expected dataset size.
        if self.index_type == "auto":
            self.index_type = "flat" if self.dimensions < 100 else "ivfflat"

        if self.index_type in ("ivfflat", "ivfpq"):
            # Defer the real IVF until >= nlist vectors exist; serve a
            # temporary flat index (correct search + add) until then.
            raw = self._new_raw_index("flat")
            self._deferred_ivf = True
        else:
            raw = self._new_raw_index(self.index_type)
            self._deferred_ivf = False

        # Wrap with IDMap2 to maintain our own internal IDs for
        # ``add_with_ids`` / ``search`` / ``remove_ids``. The index is
        # used only for similarity ``search``; ``get_vectors`` serves
        # from the ``self.vectors`` side-car rather than FAISS
        # ``reconstruct`` (reconstruct-by-id requires a maintained IVF
        # direct map, which this faiss build refuses to combine with
        # ``remove_ids``). IDMap2 round-trips through
        # ``faiss.write_index`` / ``read_index`` unchanged; keeping it
        # (vs plain IndexIDMap) leaves the on-disk index format stable.
        return faiss.IndexIDMap2(raw)

    def _build_deferred_ivf(self) -> None:
        """Swap the temporary flat index for the real trained IVF.

        Called once the side-car reaches ``nlist`` vectors. Trains the
        IVF on, and re-adds, exactly the side-car contents (the
        authoritative stored rows, keyed by internal id), then replaces
        the live index. The discarded temporary flat index held the
        same rows, so search/get_vectors stay correct across the swap.
        """
        import numpy as np

        items = list(self.vectors.items())  # (internal_id, row)
        internal_ids = np.array([iid for iid, _ in items], dtype=np.int64)
        matrix = np.ascontiguousarray(np.vstack([row for _, row in items]), dtype=np.float32)

        raw = self._new_raw_index(self.index_type)
        raw.train(matrix)
        ivf = faiss.IndexIDMap2(raw)
        ivf.add_with_ids(matrix, internal_ids)

        self.index = ivf
        self._deferred_ivf = False
        logger.info(
            "FAISS: migrated deferred %s index to trained IVF (%d vectors, nlist=%d)",
            self.index_type,
            len(items),
            self.nlist,
        )

    async def close(self) -> None:
        """Persist any unsaved changes, then release the store.

        Only a store that was *mutated* is persisted. An instance opened
        to read and closed again writes nothing — which is not merely an
        optimization: its write would move the file's identity, and the
        instance actually holding new rows would then find the file
        changed underneath it and refuse to save them.

        Releasing the store and persisting it are separate obligations,
        and a failure of the second must not skip the first. The save
        runs under ``try``/``finally`` so a refusal (see
        :meth:`save`) still leaves a closed store rather than one stuck
        re-raising on every further attempt — the exception propagates,
        as it must, because rows are on the floor.
        """
        try:
            if self.persist_path and self._initialized and self._dirty:
                await self.save()
        finally:
            self._initialized = False

    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to the index."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Prepare vectors using common method
        vectors = self._prepare_vector(vectors, normalize=(self.metric == DistanceMetric.COSINE))

        # For Faiss, we need to ensure vectors are C-contiguous
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Per-row metadata: fresh dicts with config-level domain_id
        # defaulted in (caller's dicts never aliased — Items #8 / 131).
        rows = self._apply_domain_default(metadata, len(ids))

        # No inline training: the live index is always immediately
        # addable — flat / hnsw, or the temporary flat standing in for
        # a deferred IVF, or an already-migrated trained IVF. IVF
        # training happens once in ``_build_deferred_ivf`` below, after
        # the side-car reaches ``nlist``.

        # Upsert support: when an external ID re-appears, evict its
        # prior internal ID from the FAISS index and metadata_store
        # BEFORE assigning the new internal ID. Without this,
        # ``id_map[ext_id] = internal_id`` below overwrites the only
        # external→internal pointer, leaving the prior internal_id
        # as an unreachable orphan — silent residual under filtered
        # ``clear`` and ``get_vectors`` (both walk ``id_map``), but
        # still scored by FAISS ``search``. ``index.remove_ids``
        # is the same call used by ``delete_vectors``.
        # Upsert timestamp semantics: re-adding an external id creates a
        # NEW internal id (the old one is evicted below), so carry the
        # original created_at across the internal-id change. Mirrors
        # MemoryVectorStore.add_vectors (preserve created, refresh
        # updated).
        now = datetime.now(UTC)
        prior_created: dict[str, datetime] = {}
        for ext_id in ids:
            if ext_id in self.id_map:
                old_internal = self.id_map[ext_id]
                if old_internal in self.timestamps:
                    prior_created[ext_id] = self.timestamps[old_internal][0]

        orphan_internal_ids = [self.id_map[ext_id] for ext_id in ids if ext_id in self.id_map]
        if orphan_internal_ids:
            orphan_array = np.array(orphan_internal_ids, dtype=np.int64)
            self.index.remove_ids(orphan_array)
            for orphan_id in orphan_internal_ids:
                self.metadata_store.pop(orphan_id, None)
                self.timestamps.pop(orphan_id, None)
                self.vectors.pop(orphan_id, None)

        # Map IDs to internal indices
        internal_ids = []
        for i, ext_id in enumerate(ids):
            internal_id = self.next_idx
            self.next_idx += 1
            self.id_map[ext_id] = internal_id
            self.metadata_store[internal_id] = rows[i]
            self.timestamps[internal_id] = (
                prior_created.get(ext_id, now),
                now,
            )
            # Store the prepared (normalized when cosine) row, matching
            # exactly what is added to the index. Copy so the caller's
            # array and the big batch buffer are not aliased; pin float32
            # so the side-car cannot silently drift from the index dtype.
            self.vectors[internal_id] = np.array(vectors[i], dtype=np.float32)
            internal_ids.append(internal_id)

        # Add to index with internal IDs
        internal_ids_array = np.array(internal_ids, dtype=np.int64)
        self.index.add_with_ids(vectors, internal_ids_array)

        # First time a deferred-IVF store reaches nlist: build and
        # migrate to the real trained IVF (from the side-car, which now
        # holds every row including the ones just added).
        if self._deferred_ivf and len(self.vectors) >= self.nlist:
            self._build_deferred_ivf()

        if ids:
            self._mark_dirty()
        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID."""
        if not self._initialized:
            await self.initialize()

        inject = include_timestamps and include_metadata
        results: list[tuple[np.ndarray | None, dict[str, Any] | None]] = []
        for ext_id in ids:
            if ext_id not in self.id_map:
                results.append((None, None))
                continue

            internal_id = self.id_map[ext_id]

            stored = self.vectors.get(internal_id)
            if stored is None:
                # ``ext_id`` resolved but its internal id has no stored
                # vector — a genuine desync (e.g. a post-delete
                # internal-id reuse race), not an absent id (handled
                # before this point). Surface it at WARNING rather than
                # silently collapsing to indistinguishable-from-absent.
                logger.warning(
                    "FAISS get_vectors: no stored vector for id %s (internal id %s)",
                    ext_id,
                    internal_id,
                )
                results.append((None, None))
                continue

            # Copy so the caller cannot mutate the stored array — and the
            # same for the metadata dict beside it, which was handed out
            # live whenever timestamps were not being injected.
            vector = stored.copy()
            created, updated = self.timestamps.get(internal_id, (None, None))
            results.append(
                (
                    vector,
                    self._outbound_metadata(
                        self.metadata_store.get(internal_id) if include_metadata else None,
                        inject=inject,
                        created=created,
                        updated=updated,
                    ),
                )
            )

        return results

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Get internal IDs
        internal_ids = []
        for ext_id in ids:
            if ext_id in self.id_map:
                internal_id = self.id_map[ext_id]
                internal_ids.append(internal_id)
                del self.id_map[ext_id]
                if internal_id in self.metadata_store:
                    del self.metadata_store[internal_id]
                self.timestamps.pop(internal_id, None)
                self.vectors.pop(internal_id, None)

        if internal_ids:
            # Remove from index
            internal_ids_array = np.array(internal_ids, dtype=np.int64)
            removed = self.index.remove_ids(internal_ids_array)
            self._mark_dirty()
            return removed

        return 0

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors.

        An unfiltered search is answered by the FAISS index. A filtered
        one is not, and cannot be: the index has no way to express the
        filter, so it returns a top-``k`` window that the filter then
        empties — a small co-tenant occupying the window is enough to
        return nothing at all from a store holding hundreds of matching
        rows. The matching rows are selected from ``metadata_store`` and
        scored directly instead, which is exact on every index type and
        is the answer ``MemoryVectorStore`` gives for the same corpus.
        """
        if not self._initialized:
            await self.initialize()

        # Asking for no rows is answered the same way whichever path
        # would have served it. Normalized here rather than in each
        # branch: the filtered path used to return ``[]`` for a negative
        # ``k`` while the unfiltered one passed it to ``index.search``.
        if k <= 0:
            return []

        inject = include_timestamps and include_metadata

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        # Prepare query vector using common method
        query = self._prepare_vector(query_vector, normalize=(self.metric == DistanceMetric.COSINE))

        # Set search parameters for IVF. self.index is always an
        # IndexIDMap2, which does not proxy nprobe, so it must be set on
        # the unwrapped inner index. The inner is a flat index while the
        # IVF is still deferred (no nprobe) — the hasattr guard skips it.
        inner = faiss.downcast_index(self.index.index)
        if hasattr(inner, "nprobe"):
            inner.nprobe = self.nprobe

        # An empty filter drops nothing, so it takes the index path with
        # everything else that is unfiltered.
        if filter:
            return self._search_filtered(query, k, filter, include_metadata, inject)

        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than we have
        if k == 0:
            return []

        scores, indices = self.index.search(query, k)

        # Convert results
        reverse_id_map = {v: k for k, v in self.id_map.items()}
        results = []

        for i in range(len(indices[0])):
            internal_id = int(indices[0][i])
            if internal_id == -1:  # No result
                continue
            results.append(
                self._result_row(
                    internal_id,
                    float(scores[0][i]),
                    reverse_id_map,
                    include_metadata,
                    inject,
                )
            )

        return results

    def _score_from_raw(self, raw: float) -> float:
        """Convert a raw FAISS metric value into the score callers see.

        Cosine is an inner product of normalized vectors, which already
        *is* the similarity. L2 arrives as a distance and is inverted.
        Every other configured metric maps onto one of those two FAISS
        metrics (:meth:`_faiss_metric`) and its raw value is reported
        unchanged, which is what callers have always received.

        Both search paths convert here, so a filtered search and an
        unfiltered one report the same number for the same row.
        """
        if self.metric == DistanceMetric.COSINE:
            # Inner product of normalized vectors = cosine similarity
            return raw
        if self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            return 1.0 / (1.0 + raw)
        return raw

    def _result_row(
        self,
        internal_id: int,
        raw: float,
        reverse_id_map: dict[int, str],
        include_metadata: bool,
        inject: bool,
    ) -> tuple[str, float, dict[str, Any] | None]:
        """Assemble one result tuple. Shared by both search paths."""
        ext_id = reverse_id_map.get(internal_id, str(internal_id))
        created, updated = self.timestamps.get(internal_id, (None, None))
        # Copied, not handed out live — and only for the ``k`` rows that
        # are actually returned, since this runs per result row rather
        # than per scored candidate.
        out_meta = self._outbound_metadata(
            self.metadata_store.get(internal_id) if include_metadata else None,
            inject=inject,
            created=created,
            updated=updated,
        )
        return (ext_id, self._score_from_raw(raw), out_meta)

    def _raw_index_scores(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Reproduce over ``matrix`` the values ``index.search`` returns.

        Because :meth:`_score_from_raw` converts for both paths, this
        has to produce what FAISS itself produces: an inner product
        under ``METRIC_INNER_PRODUCT``, a **squared** L2 distance under
        ``METRIC_L2``. Computing an unsquared distance here would give a
        filtered search different scores from an unfiltered one over the
        same rows.
        """
        import numpy as np

        row = query.reshape(-1)
        if self._faiss_metric() == faiss.METRIC_INNER_PRODUCT:
            return np.asarray(matrix @ row, dtype=np.float64)
        diff = matrix - row
        return np.asarray(np.einsum("ij,ij->i", diff, diff), dtype=np.float64)

    def _search_filtered(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, Any],
        include_metadata: bool,
        inject: bool,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Score the rows matching ``filter`` and return the best ``k``.

        Walks ``metadata_store`` for the matches — the same walk
        ``count(filter=...)`` already performs — then scores only those
        rows from the ``self.vectors`` side-car. A filtered search
        therefore leaves the approximate index and becomes exact over
        the matching subset. Unfiltered search keeps the index and is
        untouched.

        ``k`` is already positive: :meth:`search` normalizes it for both
        paths before dispatching here.
        """
        matching = [
            internal_id
            for internal_id, metadata in self.metadata_store.items()
            if self._match_metadata_filter(metadata, filter)
        ]
        if not matching:
            return []

        scorable = [internal_id for internal_id in matching if internal_id in self.vectors]
        if len(scorable) == len(matching):
            return self._score_stored(query, matching, k, include_metadata, inject)

        # Some matching rows have no stored vector — see
        # :meth:`_warn_sidecar_shortfall` for how a store gets that way.
        self._warn_sidecar_shortfall(matched=len(matching), scorable=len(scorable))
        return self._score_partial(query, matching, scorable, k, include_metadata, inject)

    def _score_stored(
        self,
        query: np.ndarray,
        internal_ids: list[int],
        k: int,
        include_metadata: bool,
        inject: bool,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Rank ``internal_ids`` against ``query`` from the side-car.

        The exact path, used when every matching row has a stored vector.
        Kept array-shaped end to end — the ids stay a list and the scores
        an ndarray, zipped only for the surviving ``k`` — because this is
        the path a large scoped store takes on every call.
        """
        import numpy as np

        if not internal_ids:
            return []

        raw = self._raw_index_scores(
            query, np.vstack([self.vectors[internal_id] for internal_id in internal_ids])
        )
        # Inner product ranks high-to-low, L2 distance low-to-high;
        # ``stable`` keeps ties in insertion order so repeated searches
        # over one corpus agree with each other.
        nearest_first = -raw if self._faiss_metric() == faiss.METRIC_INNER_PRODUCT else raw
        order = np.argsort(nearest_first, kind="stable")

        reverse_id_map = {internal: ext for ext, internal in self.id_map.items()}
        return [
            self._result_row(
                internal_ids[position],
                float(raw[position]),
                reverse_id_map,
                include_metadata,
                inject,
            )
            for position in order[:k]
        ]

    def _warn_sidecar_shortfall(self, *, matched: int, scorable: int) -> None:
        """Report, once per instance, that the vector side-car is short.

        ``_load_from_disk`` reads the side-car with a default, so a
        ``.meta`` pickle written before the side-car existed loads empty
        against a fully populated index. Recoverable by re-ingesting,
        which is why this is a WARNING and not silence: a shortfall is
        indistinguishable to a caller from a store that genuinely holds
        no more matches.

        Once, because the condition belongs to the loaded file rather
        than to any one query — every filtered search this instance
        serves meets it, and repeating a message whose remedy is a
        one-off re-ingest just fills the log of a read path.
        """
        if self._sidecar_shortfall_warned:
            return
        self._sidecar_shortfall_warned = True
        logger.warning(
            "FAISS search: %d of the %d rows matching this filter have no "
            "stored vector (index holds %d). Those rows are ranked from the "
            "index instead of scored directly, which can miss the ones an "
            "approximate index does not route to — so a filtered search may "
            "return fewer than k rows. Re-ingest the store to restore exact "
            "filtered search. Reported once per store.",
            matched - scorable,
            matched,
            self.index.ntotal,
        )

    def _score_partial(
        self,
        query: np.ndarray,
        matching: list[int],
        scorable: list[int],
        k: int,
        include_metadata: bool,
        inject: bool,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Rank a filter's matches when only some have stored vectors.

        Neither source alone is the right answer. Scoring only the
        side-car silently omits a row that has no stored vector but would
        have outranked everything returned; scanning only the index
        answers approximately for rows the side-car could have scored
        exactly. So both are used and merged.

        The merge is sound because :meth:`_raw_index_scores` reproduces
        the values ``index.search`` itself returns — the two halves are
        on one scale, not two, which is the same property that makes a
        filtered search's scores match an unfiltered one's.

        This still under-returns where the *index* cannot reach an
        unscorable row: an HNSW graph does not route to every node, and
        an IVF probe covers only its own lists. That residue is why the
        caller reports the shortfall rather than treating this as a
        repair.
        """
        import numpy as np

        scored: list[tuple[int, float]] = []
        if scorable:
            raw = self._raw_index_scores(
                query, np.vstack([self.vectors[internal_id] for internal_id in scorable])
            )
            scored.extend((internal_id, float(raw[i])) for i, internal_id in enumerate(scorable))

        scored.extend(self._raw_scores_via_index(query, k, set(matching) - set(scorable)))
        if not scored:
            return []

        values = np.fromiter((value for _, value in scored), dtype=np.float64, count=len(scored))
        nearest_first = -values if self._faiss_metric() == faiss.METRIC_INNER_PRODUCT else values
        order = np.argsort(nearest_first, kind="stable")

        reverse_id_map = {internal: ext for ext, internal in self.id_map.items()}
        return [
            self._result_row(
                scored[position][0],
                scored[position][1],
                reverse_id_map,
                include_metadata,
                inject,
            )
            for position in order[:k]
        ]

    def _raw_scores_via_index(
        self,
        query: np.ndarray,
        k: int,
        wanted: set[int],
    ) -> list[tuple[int, float]]:
        """Ask the index for the ``wanted`` rows it can reach.

        The only way to score a row whose vector the side-car does not
        hold. Over-fetches and escalates to the whole index, so it falls
        short only where the index genuinely cannot route to a row rather
        than because the window was too small.
        """
        if not wanted:
            return []

        found: dict[int, float] = {}
        for fetch in self._overfetch_sizes(k, has_post_filter=True, ceiling=self.index.ntotal):
            if fetch <= 0:
                break
            scores, indices = self.index.search(query, fetch)
            found = {}
            for i in range(len(indices[0])):
                internal_id = int(indices[0][i])
                if internal_id in wanted:
                    found[internal_id] = float(scores[0][i])
            if len(found) >= k:
                break

        return list(found.items())

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors."""
        if not self._initialized:
            await self.initialize()

        now = datetime.now(UTC)
        updated = 0
        for ext_id, meta in zip(ids, metadata, strict=False):
            if ext_id in self.id_map:
                internal_id = self.id_map[ext_id]
                # Stored as a copy: the caller keeps its own dict and
                # must not keep a handle on the row through it. The
                # ``add_vectors`` path gets this from
                # ``_apply_domain_default``; this one has no equivalent.
                self.metadata_store[internal_id] = self._copy_metadata(meta) or {}
                if internal_id in self.timestamps:
                    created, _ = self.timestamps[internal_id]
                    self.timestamps[internal_id] = (created, now)
                updated += 1

        if updated:
            self._mark_dirty()
        return updated

    async def update_metadata_where(
        self,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Merge ``set_`` into metadata of every filter-matched vector.

        Operates purely on the ``metadata_store`` side-car (keyed by
        internal id) that :meth:`search` selects its matches from — the
        FAISS index is pure-vector, knows nothing about metadata, and
        so has nothing to invalidate here. Same mechanism as
        ``clear(filter=)`` / ``count(filter=)``.
        """
        if not self._initialized:
            await self.initialize()

        updated = self._update_metadata_where_filtered(
            self.metadata_store.items(),
            self.timestamps,
            self._effective_filter(filter),
            set_,
        )
        if updated:
            self._mark_dirty()
        return updated

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors in the store."""
        if not self._initialized:
            await self.initialize()

        filter = self._effective_filter(filter)
        if filter is None:
            return self.index.ntotal

        # Count with filter
        count = 0
        for metadata in self.metadata_store.values():
            if self._match_metadata_filter(metadata, filter):
                count += 1

        return count

    async def metadata_fields(self) -> set[str]:
        """Discover metadata field names across all stored vectors."""
        if not self._initialized:
            await self.initialize()

        fields: set[str] = set()
        for meta in self.metadata_store.values():
            fields.update(meta.keys())
        return fields

    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        """Clear vectors, optionally filtered by metadata.

        Filtered clear iterates ``metadata_store`` to find matching
        IDs, then delegates to :meth:`delete_vectors`.  The walk is
        O(N) over stored vectors; FAISS has no native filtered
        delete.  Acceptable for typical knowledge-base sizes;
        workloads at scale where filtered clear is hot should prefer
        pgvector or Chroma.
        """
        if not self._initialized:
            await self.initialize()

        filter = self._effective_filter(filter)
        if filter is None:
            self.index = self._create_index()
            self.id_map.clear()
            self.metadata_store.clear()
            self.timestamps.clear()
            self.vectors.clear()
            self.next_idx = 0
            self._mark_dirty()
            # Nothing is left for the side-car to be short of.
            self._sidecar_shortfall_warned = False
            return

        # ``metadata_store`` is keyed by internal id; ``id_map`` maps
        # external -> internal. Walk external ids and check each
        # corresponding metadata entry against the filter.
        matching_ext_ids: list[str] = [
            ext_id
            for ext_id, internal_id in self.id_map.items()
            if self._match_metadata_filter(
                self.metadata_store.get(internal_id),
                filter,
            )
        ]
        if matching_ext_ids:
            await self.delete_vectors(matching_ext_ids)

    async def save(self, *, force: bool = False) -> None:
        """Save index and metadata to disk.

        Offloads the entire disk body (``os.makedirs``,
        ``faiss.write_index``, ``open`` + ``pickle.dump`` of the
        ``.meta`` side-car) onto a worker thread via
        :func:`asyncio.to_thread` so the event loop is never blocked.
        In-memory ``add`` / ``search`` are CPU-bound C++ (the GIL is
        released inside FAISS) and remain on the loop — only the disk
        I/O is offloaded.

        Raises :class:`~dataknobs_common.exceptions.ConcurrencyError`
        when the file changed since this instance read or wrote it,
        rather than replacing another writer's rows with this instance's
        snapshot.

        Args:
            force: Write regardless of what is on disk. This is the way
                out of a refusal — the store keeps its rows in memory
                after one, but every further save raises too, because
                what it compares against has not moved. Passing ``True``
                accepts the loss of whatever the other writer persisted;
                the alternative that loses nothing is to open a second
                store on the file and re-add these rows to it.
        """
        if not self.persist_path:
            return
        if self.index is None:
            # The index is created in initialize(); a save() before that
            # has nothing to persist. Skip rather than crash downstream in
            # faiss.write_index(None) (the dicts are necessarily empty too,
            # since they are only populated through add_vectors).
            return
        # Snapshot the mutable in-memory state on the event loop BEFORE
        # handing off to the worker thread, so a concurrent add_vectors /
        # delete_vectors can't race the serialization:
        #   * The dicts are shallow-copied — values (ndarrays, metadata
        #     dicts, timestamp tuples) are replaced by reference on
        #     mutation, never mutated in place, so the worker reads them
        #     safely without "dictionary changed size during iteration".
        #   * The FAISS index is deep-cloned because ``faiss.write_index``
        #     would otherwise serialize an index being mutated by a
        #     concurrent ``add_with_ids`` / ``remove_ids`` on the loop.
        # Both snapshots are taken synchronously with no ``await`` between
        # them, so no mutation can interleave — the persisted index and
        # ``.meta`` side-car are mutually consistent.
        #
        # The lock covers snapshot *and* write together. Taking it later
        # would leave the staleness check and the write it guards on
        # opposite sides of a scheduling point, which is how one
        # instance's two overlapping saves end up racing each other.
        async with self._save_lock:
            index_snapshot = faiss.clone_index(self.index)
            meta_snapshot = {
                "id_map": dict(self.id_map),
                "metadata_store": dict(self.metadata_store),
                "timestamps": dict(self.timestamps),
                "vectors": dict(self.vectors),
                "deferred_ivf": self._deferred_ivf,
                "next_idx": self.next_idx,
                "config": {
                    "dimensions": self.dimensions,
                    "metric": self.metric.value,
                    "index_type": self.index_type,
                },
            }
            await asyncio.to_thread(self._save_to_disk, index_snapshot, meta_snapshot, force)

    def _save_to_disk(self, index: Any, meta: dict[str, Any], force: bool = False) -> None:
        """Synchronous disk write — run via ``to_thread`` from :meth:`save`.

        Receives a loop-side snapshot (a cloned index and a dict of
        shallow-copied mappings); reads only those, never the live
        ``self.*`` state, so a concurrent mutation cannot corrupt the
        write. That protects against this instance's own event loop; the
        staleness check protects against a different instance, which the
        snapshot cannot see at all.
        """
        # Convert Path to string for FAISS
        persist_path_str = str(self.persist_path)
        metadata_path = persist_path_str + ".meta"

        self._guard_persisted_identity(persist_path_str, force=force)

        # Create directory if needed
        parent_dir = os.path.dirname(persist_path_str)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        def write_index(path: str) -> None:
            faiss.write_index(index, path)

        def write_meta(path: str) -> None:
            with open(path, "wb") as f:
                pickle.dump(meta, f)

        # The index and its ``.meta`` side-car describe one corpus, so
        # neither is written directly over its target: a ``.meta`` that
        # fails to serialize would otherwise leave a *new* index beside a
        # *stale* side-car, and leave this instance's identity stamp
        # pointing at a file it had already replaced — after which every
        # later save of its own raises.
        self._write_then_publish([(persist_path_str, write_index), (metadata_path, write_meta)])

        # This instance is now the file's last writer, and is in step
        # with what is on disk: a further save of its own must not trip
        # the check above, and ``close()`` has nothing left to persist.
        self._stamp_persisted_identity(persist_path_str)

    async def load(self) -> None:
        """Load index and metadata from disk.

        Offloads the entire disk body (the existence ``os.path.exists``
        stat, ``faiss.read_index``, ``open`` + ``pickle.load`` of the
        ``.meta`` side-car) onto a worker thread via
        :func:`asyncio.to_thread`. A no-op when ``persist_path`` is unset
        or no file exists.
        """
        if not self.persist_path:
            return
        await asyncio.to_thread(self._load_from_disk)

    def _load_from_disk(self) -> None:
        """Synchronous disk read — run via ``to_thread`` from :meth:`load`."""
        # Convert Path to string for FAISS
        persist_path_str = str(self.persist_path)

        if not os.path.exists(persist_path_str):
            logger.debug(
                "FAISS: No persist path or file not found: %s",
                self.persist_path,
            )
            return

        # Load index
        self.index = faiss.read_index(persist_path_str)
        logger.info(
            "FAISS: Loaded index from %s with %d vectors",
            persist_path_str,
            self.index.ntotal,
        )

        # Load metadata and mappings
        metadata_path = persist_path_str + ".meta"
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.id_map = data["id_map"]
                self.metadata_store = data["metadata_store"]
                # .get() for backward-compat with older .meta pickles
                # that predate a key: timestamps return None/None on
                # include_timestamps; a missing ``vectors`` side-car
                # means get_vectors returns None until the index is
                # re-ingested (parity with memory.py and pgvector
                # pre-migration NULL rows).
                self.timestamps = data.get("timestamps", {})
                self.vectors = data.get("vectors", {})
                # Legacy IVF pickles have no flag but are necessarily a
                # trained IVF (a sub-nlist first batch could not have
                # been persisted pre-fix), so False is correct.
                self._deferred_ivf = data.get("deferred_ivf", False)
                self.next_idx = data["next_idx"]
            logger.info("FAISS: Loaded metadata with %d entries", len(self.id_map))

        # Stamped only once the whole read has succeeded: this is both
        # what a later save() compares against to tell its own writes
        # from another instance's, and the flag saying memory and disk
        # agree. A partial load agrees with nothing.
        self._stamp_persisted_identity(persist_path_str)
        # A reload can only have replaced the side-car this warns about.
        self._sidecar_shortfall_warned = False

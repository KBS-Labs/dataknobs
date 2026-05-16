"""Faiss vector store implementation."""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..types import DistanceMetric
from .base import VectorStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Faiss vector store."""
        if not FAISS_AVAILABLE:
            raise ImportError(
                "Faiss is not installed. Install with: pip install faiss-cpu"
            )

        super().__init__(config)
        self.index = None
        self.id_map = {}  # Map from our IDs to Faiss internal indices
        self.metadata_store = {}  # Store metadata separately
        # internal_id -> (created_at, updated_at). Aware UTC datetimes.
        # Keyed by internal id to match ``metadata_store`` so the
        # shared update_metadata_where helper, get_vectors, search,
        # delete, and clear all stay consistent. Pickle-persisted in
        # the ``.meta`` side-car; legacy pickles without this key load
        # as empty (rows return None/None on include_timestamps).
        self.timestamps: dict[int, tuple[datetime, datetime]] = {}
        self.next_idx = 0
        # IVF index types (ivfflat/ivfpq) need an explicit reverse map
        # before ``reconstruct``-by-id works; built lazily once the
        # index is trained and populated, and rebuilt after ``load()``
        # (``faiss.read_index`` does not restore it).
        self._direct_map_built: bool = False

    def _parse_backend_config(self) -> None:
        """Parse Faiss-specific configuration."""
        # Determine index type
        self.index_type = self.index_params.get("type", "auto")
        if "index_type" in self.config:
            self.index_type = self.config["index_type"]

        self.nlist = self.index_params.get("nlist", 100)  # For IVF
        self.m = self.index_params.get("m", 32)  # For HNSW
        self.ef_construction = self.index_params.get("ef_construction", 200)  # For HNSW
        self.ef_search = self.index_params.get("ef_search", 50)  # For HNSW search
        self.nprobe = self.search_params.get("nprobe", 10)  # For IVF search

    async def initialize(self) -> None:
        """Initialize Faiss index."""
        if self._initialized:
            return

        # Create index based on type and metric
        self.index = self._create_index()

        # Load existing index if persist path exists
        if self.persist_path and os.path.exists(self.persist_path):
            await self.load()

        self._initialized = True

    def _create_index(self) -> Any:
        """Create Faiss index based on configuration."""
        dimensions = self.dimensions

        # Map distance metrics
        if self.metric == DistanceMetric.COSINE:
            # For cosine similarity, we'll normalize vectors and use inner product
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            metric = faiss.METRIC_L2
        elif self.metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        # Auto-select index type based on expected dataset size
        if self.index_type == "auto":
            # Use flat for small dimensions/datasets
            if dimensions < 100:
                self.index_type = "flat"
            else:
                self.index_type = "ivfflat"

        # Create index
        if self.index_type == "flat":
            if metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dimensions)
            else:
                index = faiss.IndexFlatL2(dimensions)

        elif self.index_type == "ivfflat":
            # Create quantizer
            quantizer = faiss.IndexFlatL2(dimensions)
            if metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexIVFFlat(quantizer, dimensions, self.nlist, metric)
            else:
                index = faiss.IndexIVFFlat(quantizer, dimensions, self.nlist)

        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimensions, self.m, metric)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search

        elif self.index_type == "ivfpq":
            # Product quantization for compression
            m = 8  # Number of subquantizers
            quantizer = faiss.IndexFlatL2(dimensions)
            index = faiss.IndexIVFPQ(quantizer, dimensions, self.nlist, m, 8)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Wrap with IDMap2 to maintain our own IDs. IDMap2 is a strict
        # superset of IDMap: identical add/search/remove behavior, plus
        # it maintains the reverse map required for reconstruct-by-id.
        # ``get_vectors`` reconstructs the stored vector by internal id
        # (see ``self.index.reconstruct`` in ``get_vectors``); plain
        # ``IndexIDMap`` raises ``RuntimeError`` ("reconstruct not
        # implemented") for that call, so ``get_vectors`` returned
        # ``(None, None)`` for every id. IDMap2 round-trips through
        # ``faiss.write_index``/``read_index`` unchanged. Note: for the
        # IVF index types (ivfflat/ivfpq) the wrapped index additionally
        # needs ``make_direct_map()`` before reconstruct succeeds — a
        # separate pre-existing limitation tracked independently; the
        # flat/hnsw paths (auto-selected for typical and test configs)
        # reconstruct directly.
        index = faiss.IndexIDMap2(index)

        return index

    async def close(self) -> None:
        """Save and close the index."""
        if self.persist_path and self._initialized:
            await self.save()
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
        if not vectors.flags['C_CONTIGUOUS']:
            vectors = np.ascontiguousarray(vectors)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Per-row metadata: fresh dicts with config-level domain_id
        # defaulted in (caller's dicts never aliased — Items #8 / 131).
        rows = self._apply_domain_default(metadata, len(ids))

        # Train index if needed (for IVF types)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            if len(vectors) >= self.nlist:
                self.index.train(vectors)
            else:
                # Not enough vectors to train, use flat index temporarily
                pass

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
        now = datetime.now(timezone.utc)
        prior_created: dict[str, datetime] = {}
        for ext_id in ids:
            if ext_id in self.id_map:
                old_internal = self.id_map[ext_id]
                if old_internal in self.timestamps:
                    prior_created[ext_id] = self.timestamps[old_internal][0]

        orphan_internal_ids = [
            self.id_map[ext_id] for ext_id in ids if ext_id in self.id_map
        ]
        if orphan_internal_ids:
            orphan_array = np.array(orphan_internal_ids, dtype=np.int64)
            self.index.remove_ids(orphan_array)
            for orphan_id in orphan_internal_ids:
                self.metadata_store.pop(orphan_id, None)
                self.timestamps.pop(orphan_id, None)

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
            internal_ids.append(internal_id)

        # Add to index with internal IDs
        internal_ids_array = np.array(internal_ids, dtype=np.int64)
        self.index.add_with_ids(vectors, internal_ids_array)

        # IVF reconstruct-by-id (used by get_vectors) needs a direct
        # map, valid only once the index is trained and populated.
        self._ensure_ivf_direct_map()

        return ids

    def _ensure_ivf_direct_map(self) -> None:
        """Build the IVF reverse map so ``reconstruct``-by-id works.

        Only the IVF index types (``ivfflat``/``ivfpq``, auto-selected
        for ``dimensions >= 100``) need this. ``IndexIDMap2.reconstruct``
        delegates to the wrapped index; ``IndexIVF`` raises
        ``RuntimeError`` for reconstruct until ``make_direct_map()`` has
        been called, and the map is valid only after train+populate.
        Idempotent via ``_direct_map_built``; ``load()`` resets the flag
        because ``faiss.read_index`` does not restore the map.
        """
        if self._direct_map_built:
            return
        if self.index_type not in ("ivfflat", "ivfpq"):
            return
        if self.index is None or self.index.ntotal <= 0:
            return
        # The live faiss object is authoritative — ``index_type`` is
        # only the configured/auto-selected hint. Reach the wrapped
        # index through ``IndexIDMap2.index`` and downcast to its
        # concrete type rather than ``faiss.extract_index_ivf`` (which
        # *raises* for a degenerate/untrained persisted IVF, breaking
        # ``load()``). A non-IVF or untrained inner index has nothing
        # to reconstruct, so there is simply no direct map to build.
        inner = getattr(self.index, "index", None)
        if inner is None:
            return
        inner = faiss.downcast_index(inner)
        if not isinstance(inner, faiss.IndexIVF):
            return
        if not inner.is_trained:
            return
        inner.make_direct_map()
        self._direct_map_built = True

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

            # Reconstruct vector from index
            try:
                vector = self.index.reconstruct(int(internal_id))
                metadata = (
                    self.metadata_store.get(internal_id)
                    if include_metadata
                    else None
                )
                if inject:
                    created, updated = self.timestamps.get(
                        internal_id, (None, None)
                    )
                    metadata = self._inject_timestamps(
                        metadata, created=created, updated=updated
                    )
                results.append((vector, metadata))
            except Exception as exc:
                # The "id absent" case is handled before the ``try``
                # (continue above), so reaching here is a genuine
                # reconstruct failure (e.g. a post-delete internal-id
                # reuse race). Surface it at WARNING rather than
                # silently collapsing it to indistinguishable-from-
                # absent.
                logger.warning(
                    "FAISS reconstruct failed for id %s: %s", ext_id, exc
                )
                results.append((None, None))

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

        if internal_ids:
            # Remove from index
            internal_ids_array = np.array(internal_ids, dtype=np.int64)
            removed = self.index.remove_ids(internal_ids_array)
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
        """Search for similar vectors."""
        if not self._initialized:
            await self.initialize()

        inject = include_timestamps and include_metadata

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        # Prepare query vector using common method
        query = self._prepare_vector(query_vector, normalize=(self.metric == DistanceMetric.COSINE))

        # Set search parameters for IVF
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than we have
        if k == 0:
            return []

        scores, indices = self.index.search(query, k)

        # Convert results
        results = []
        reverse_id_map = {v: k for k, v in self.id_map.items()}

        for i in range(len(indices[0])):
            internal_id = indices[0][i]
            if internal_id == -1:  # No result
                continue

            score = float(scores[0][i])

            # Convert score based on metric
            if self.metric == DistanceMetric.COSINE:
                # Inner product of normalized vectors = cosine similarity
                score = score  # noqa: PLW0127 - Keep for clarity
            elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
                # Convert distance to similarity score
                score = 1.0 / (1.0 + score)

            # Get external ID
            ext_id = reverse_id_map.get(internal_id, str(internal_id))

            # Apply metadata filter if provided. Fetch metadata even
            # when not returning it to the caller so the filter can be
            # evaluated post-top-k.
            metadata = self.metadata_store.get(internal_id)
            if filter and not self._match_metadata_filter(metadata, filter):
                continue

            out_meta = metadata if include_metadata else None
            if inject:
                created, updated = self.timestamps.get(
                    internal_id, (None, None)
                )
                out_meta = self._inject_timestamps(
                    out_meta, created=created, updated=updated
                )
            results.append((ext_id, score, out_meta))

        return results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors."""
        if not self._initialized:
            await self.initialize()

        now = datetime.now(timezone.utc)
        updated = 0
        for ext_id, meta in zip(ids, metadata, strict=False):
            if ext_id in self.id_map:
                internal_id = self.id_map[ext_id]
                self.metadata_store[internal_id] = meta
                if internal_id in self.timestamps:
                    created, _ = self.timestamps[internal_id]
                    self.timestamps[internal_id] = (created, now)
                updated += 1

        return updated

    async def update_metadata_where(
        self,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Merge ``set_`` into metadata of every filter-matched vector.

        Operates purely on the ``metadata_store`` side-car (keyed by
        internal id) that :meth:`search` already post-filters against
        (``faiss.py`` filtering is post-retrieval; the FAISS index is
        pure-vector and has nothing to invalidate), so this is the same
        mechanism as ``clear(filter=)`` / ``count(filter=)``.
        """
        if not self._initialized:
            await self.initialize()

        return self._update_metadata_where_filtered(
            self.metadata_store.items(),
            self.timestamps,
            self._effective_filter(filter),
            set_,
        )

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
            self.next_idx = 0
            return

        # ``metadata_store`` is keyed by internal id; ``id_map`` maps
        # external -> internal. Walk external ids and check each
        # corresponding metadata entry against the filter.
        matching_ext_ids: list[str] = [
            ext_id
            for ext_id, internal_id in self.id_map.items()
            if self._match_metadata_filter(
                self.metadata_store.get(internal_id), filter,
            )
        ]
        if matching_ext_ids:
            await self.delete_vectors(matching_ext_ids)

    async def save(self) -> None:
        """Save index and metadata to disk."""
        if not self.persist_path:
            return

        # Convert Path to string for FAISS
        persist_path_str = str(self.persist_path)

        # Create directory if needed
        parent_dir = os.path.dirname(persist_path_str)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Save index
        faiss.write_index(self.index, persist_path_str)

        # Save metadata and mappings
        metadata_path = persist_path_str + ".meta"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "metadata_store": self.metadata_store,
                "timestamps": dict(self.timestamps),
                "next_idx": self.next_idx,
                "config": {
                    "dimensions": self.dimensions,
                    "metric": self.metric.value,
                    "index_type": self.index_type,
                }
            }, f)

    async def load(self) -> None:
        """Load index and metadata from disk."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            logger.debug(
                "FAISS: No persist path or file not found: %s",
                self.persist_path,
            )
            return

        # Convert Path to string for FAISS
        persist_path_str = str(self.persist_path)

        # Load index
        self.index = faiss.read_index(persist_path_str)
        logger.info(
            "FAISS: Loaded index from %s with %d vectors",
            persist_path_str,
            self.index.ntotal,
        )

        # ``faiss.read_index`` restores the index but not the IVF
        # direct map, so reconstruct-by-id would fail after a reload.
        # Reset the guard and rebuild it for the loaded population.
        self._direct_map_built = False
        self._ensure_ivf_direct_map()

        # Load metadata and mappings
        metadata_path = persist_path_str + ".meta"
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.id_map = data["id_map"]
                self.metadata_store = data["metadata_store"]
                # .get() for backward-compat with pre-Item-36 .meta
                # pickles — those have no tracked timestamps, so rows
                # return None/None on include_timestamps=True (parity
                # with memory.py and pgvector pre-migration NULL rows).
                self.timestamps = data.get("timestamps", {})
                self.next_idx = data["next_idx"]
            logger.info(
                "FAISS: Loaded metadata with %d entries", len(self.id_map)
            )

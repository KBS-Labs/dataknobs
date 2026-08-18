"""In-memory vector store implementation."""

from __future__ import annotations

import asyncio
import os
import pickle
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from .base import VectorStore
from .config import MemoryVectorStoreConfig

if TYPE_CHECKING:
    from typing import ClassVar


class MemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing and development.

    This implementation stores vectors in memory using numpy arrays
    and performs brute-force search. Suitable for small datasets
    and testing scenarios.
    """

    CONFIG_CLS: ClassVar[type[MemoryVectorStoreConfig]] = MemoryVectorStoreConfig

    def _setup(self) -> None:
        """Initialize in-memory runtime state."""
        super()._setup()
        self.vectors: dict[str, np.ndarray] = {}  # id -> vector
        self.metadata_store: dict[str, dict[str, Any]] = {}  # id -> metadata
        # id -> (created_at, updated_at). Aware UTC datetimes. Pickle-
        # persisted alongside vectors/metadata. Legacy pickles without
        # this key load as an empty dict — matches pgvector's
        # pre-migration NULL-row semantics.
        self.timestamps: dict[str, tuple[datetime, datetime]] = {}

    async def initialize(self) -> None:
        """Initialize the store."""
        if self._initialized:
            return

        # Load existing data if any. ``load`` self-guards on the persist
        # path and the file's existence (off the event loop), so there is
        # no on-loop ``os.path.exists`` stat here.
        await self.load()

        self._initialized = True

    async def close(self) -> None:
        """Persist any unsaved changes, then release the store.

        Only a store that was *mutated* is persisted. An instance opened
        to read and closed again writes nothing — which is not merely an
        optimization: its write would move the file's identity, and the
        instance actually holding new rows would then find the file
        changed underneath it and refuse to save them.

        The save runs under ``try``/``finally`` so a refusal still leaves
        a closed store; the exception propagates, as it must, because
        rows are on the floor.
        """
        try:
            if self.persist_path and self._initialized and self._dirty:
                await self.save()
        finally:
            self._initialized = False

    async def save(self, *, force: bool = False) -> None:
        """Save vectors and metadata to disk (offloaded off the event loop).

        Raises :class:`~dataknobs_common.exceptions.ConcurrencyError`
        when the file changed since this instance read or wrote it: a
        save serializes this store's whole state over the file, so
        writing over another instance's rows would lose every one of
        them silently.

        Args:
            force: Write regardless of what is on disk, accepting the
                loss of whatever the other writer persisted. This is the
                way out of a refusal, which otherwise repeats on every
                further save because what it compares against has not
                moved.
        """
        if not self.persist_path:
            return
        # Snapshot the mutable in-memory state on the event loop BEFORE
        # handing off to the worker thread. ``add_vectors`` /
        # ``delete_vectors`` / ``update_metadata`` mutate these dicts
        # directly on the loop; iterating them live in the worker would
        # race a concurrent mutation (``RuntimeError: dictionary changed
        # size during iteration`` or a torn write). Shallow copies suffice
        # — the values (ndarrays, metadata dicts, timestamp tuples) are
        # replaced by reference on mutation, never mutated in place, so the
        # worker can read them safely. Mirrors ``FaissVectorStore.save``.
        #
        # The lock spans snapshot and write so this instance's own
        # overlapping saves cannot straddle the staleness check.
        async with self._save_lock:
            await asyncio.to_thread(
                self._save_to_disk,
                dict(self.vectors),
                dict(self.metadata_store),
                dict(self.timestamps),
                force,
            )

    def _save_to_disk(
        self,
        vectors: dict[str, np.ndarray],
        metadata_store: dict[str, dict[str, Any]],
        timestamps: dict[str, tuple[datetime, datetime]],
        force: bool = False,
    ) -> None:
        """Synchronous disk write — run via ``to_thread`` from :meth:`save`.

        Receives a loop-side snapshot of the store's mutable dicts; reads
        only that snapshot plus immutable config (``persist_path``,
        ``dimensions``, ``metric``), never the live ``self.*`` dicts.
        That covers this instance's own event loop; the staleness check
        covers a second instance, which the snapshot cannot see at all.
        """
        persist_path_str = str(self.persist_path)

        self._guard_persisted_identity(persist_path_str, force=force)

        # Create directory if needed. ``os.path.dirname`` is "" for a
        # bare filename (no directory component); ``makedirs("")`` raises
        # FileNotFoundError, so guard it (parity with FaissVectorStore).
        parent_dir = os.path.dirname(persist_path_str)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        payload = {
            "vectors": {k: v.tolist() for k, v in vectors.items()},
            "metadata_store": metadata_store,
            "timestamps": timestamps,
            "config": {
                "dimensions": self.dimensions,
                "metric": self.metric.value if hasattr(self.metric, "value") else str(self.metric),
            },
        }

        def write_pickle(path: str) -> None:
            with open(path, "wb") as f:
                pickle.dump(payload, f)

        # Written to a scratch sibling and renamed, so a pickle that
        # fails midway leaves the previous state intact rather than a
        # truncated file that no longer loads.
        self._write_then_publish([(persist_path_str, write_pickle)])

        # In step with disk: a further save of this instance's own must
        # not trip the check above, and ``close()`` has nothing left to
        # persist.
        self._stamp_persisted_identity(persist_path_str)

    async def load(self) -> None:
        """Load vectors and metadata from disk (offloaded off the event loop)."""
        if not self.persist_path:
            return
        await asyncio.to_thread(self._load_from_disk)

    def _load_from_disk(self) -> None:
        """Synchronous disk read — run via ``to_thread`` from :meth:`load`."""
        # ``load()`` guards on ``persist_path`` before dispatching here;
        # naming it locally is what lets the rest of the body be typed.
        persist_path_str = str(self.persist_path)

        if not os.path.exists(persist_path_str):
            return

        with open(persist_path_str, "rb") as f:
            data = pickle.load(f)
            # Convert lists back to numpy arrays
            self.vectors = {k: np.array(v, dtype=np.float32) for k, v in data["vectors"].items()}
            self.metadata_store = data["metadata_store"]
            # .get() for backward-compat with pickle files written before
            # timestamp tracking existed — those files have no tracked
            # timestamps, so existing rows
            # return None/None on include_timestamps=True (analogous to
            # pgvector's pre-migration NULL rows).
            self.timestamps = data.get("timestamps", {})

        # Stamped only once the whole read has succeeded: this is both
        # what a later save() compares against to tell its own writes
        # from another instance's, and the flag saying memory and disk
        # agree. A partial load agrees with nothing.
        self._stamp_persisted_identity(persist_path_str)

    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to memory."""
        if not self._initialized:
            await self.initialize()

        # An empty batch is a no-op, not an error: see
        # ``VectorStoreBase._is_empty_batch``.
        if self._is_empty_batch(vectors):
            return []

        # Convert to numpy array
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Store vectors, metadata, and timestamps. Upsert semantics:
        # preserve created_at across re-adds of the same id; refresh
        # updated_at every time. ``_apply_domain_default`` returns
        # fresh per-row dicts (config-level domain_id defaulted in,
        # caller's dicts never aliased — see Items #8 / 131).
        rows = self._apply_domain_default(metadata, len(ids))
        now = datetime.now(UTC)
        for i, vector_id in enumerate(ids):
            self.vectors[vector_id] = vectors[i]
            self.metadata_store[vector_id] = rows[i]
            if vector_id in self.timestamps:
                created, _ = self.timestamps[vector_id]
                self.timestamps[vector_id] = (created, now)
            else:
                self.timestamps[vector_id] = (now, now)

        if ids:
            self._mark_dirty()
        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Get vectors by ID."""
        if not self._initialized:
            await self.initialize()

        results: list[tuple[np.ndarray | None, dict[str, Any] | None]] = []
        inject = include_timestamps and include_metadata
        for vector_id in ids:
            if vector_id in self.vectors:
                vector = self.vectors[vector_id]
                results.append((vector, self._out_metadata(vector_id, include_metadata, inject)))
            else:
                results.append((None, None))

        return results

    def _out_metadata(
        self, vector_id: str, include_metadata: bool, inject: bool
    ) -> dict[str, Any] | None:
        """The metadata dict a result carries — never the stored one.

        Copied on the way out, at every depth. Chroma and pgvector
        reconstruct each row from its serialized form, so returning the
        stored object here made "mutate a result" quietly rewrite the
        store on two backends of four — a swap-visible difference, and a
        way to corrupt a store without calling a mutator.
        """
        created, updated = self.timestamps.get(vector_id, (None, None))
        return self._outbound_metadata(
            self.metadata_store.get(vector_id) if include_metadata else None,
            inject=inject,
            created=created,
            updated=updated,
        )

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        deleted = 0
        for vector_id in ids:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
                self.metadata_store.pop(vector_id, None)
                self.timestamps.pop(vector_id, None)
                deleted += 1

        if deleted:
            self._mark_dirty()
        return deleted

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors using brute force."""
        if not self._initialized:
            await self.initialize()

        if not self.vectors:
            return []

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        # Prepare query
        query = query_vector.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Filter candidates
        candidates = []
        for vector_id, vector in self.vectors.items():
            # Apply metadata filter
            if filter:
                meta = self.metadata_store.get(vector_id, {})
                if not self._match_metadata_filter(meta, filter):
                    continue

            candidates.append((vector_id, vector))

        if not candidates:
            return []

        # Calculate distances using common method
        scores = []
        for vector_id, vector in candidates:
            score = self._calculate_similarity(query[0], vector)
            scores.append((vector_id, score))

        # Sort by score (descending for similarity)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        inject = include_timestamps and include_metadata
        for vector_id, score in scores[:k]:
            results.append(
                (vector_id, score, self._out_metadata(vector_id, include_metadata, inject))
            )

        return results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for vectors."""
        if not self._initialized:
            await self.initialize()

        now = datetime.now(UTC)
        updated = 0
        for vector_id, meta in zip(ids, metadata, strict=False):
            if vector_id in self.vectors:
                # Stored as a copy: the caller keeps its own dict and
                # must not keep a handle on the row through it. The
                # ``add_vectors`` path gets this from
                # ``_apply_domain_default``; this one has no equivalent.
                self.metadata_store[vector_id] = self._copy_metadata(meta) or {}
                # Legacy pickles: IDs written before timestamp
                # tracking was introduced exist in ``self.vectors`` but
                # not in ``self.timestamps``. Leave ``updated_at`` as
                # None for those rows, consistent with pgvector's
                # pre-migration NULL semantics (see vector-timestamps
                # docs).
                if vector_id in self.timestamps:
                    created, _ = self.timestamps[vector_id]
                    self.timestamps[vector_id] = (created, now)
                updated += 1

        if updated:
            self._mark_dirty()
        return updated

    async def update_metadata_where(
        self,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Merge ``set_`` into metadata of every filter-matched vector."""
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
        """Count vectors."""
        if not self._initialized:
            await self.initialize()

        filter = self._effective_filter(filter)
        if filter is None:
            return len(self.vectors)

        # Count with filter
        count = 0
        for vector_id in self.vectors:
            meta = self.metadata_store.get(vector_id, {})
            if self._match_metadata_filter(meta, filter):
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
        """Clear vectors, optionally filtered by metadata."""
        if not self._initialized:
            await self.initialize()

        filter = self._effective_filter(filter)
        if filter is None:
            self.vectors.clear()
            self.metadata_store.clear()
            self.timestamps.clear()
            self._mark_dirty()
            return

        matching_ids = [
            vid
            for vid, meta in self.metadata_store.items()
            if self._match_metadata_filter(meta, filter)
        ]
        for vid in matching_ids:
            self.vectors.pop(vid, None)
            self.metadata_store.pop(vid, None)
            self.timestamps.pop(vid, None)
        if matching_ids:
            self._mark_dirty()

"""Chroma vector store implementation."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..types import DistanceMetric
from .base import VectorStore
from .config import ChromaVectorStoreConfig

if TYPE_CHECKING:
    from typing import ClassVar

    import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError as ChromaNotFoundError
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ChromaVectorStore(VectorStore):
    """Chroma-based vector store for semantic search.
    
    Chroma is a vector database designed for AI applications with features like:
    - Built-in embedding functions
    - Metadata filtering
    - Persistent storage
    - Multi-tenancy support
    """

    CONFIG_CLS: ClassVar[type[ChromaVectorStoreConfig]] = ChromaVectorStoreConfig

    def _setup(self) -> None:
        """Initialize Chroma-specific derived config and runtime state."""
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        super()._setup()
        cfg = self.config

        # ``dimensions`` defaults to 384 in ChromaVectorStoreConfig, so
        # ``self.dimensions`` is already resolved by the base ``_setup``.
        self.collection_name = cfg.collection_name

        # Opt-in declaration of metadata keys whose stored values are
        # always scalar (never list-valued). For declared scalar keys
        # the partitioner pushes a Chroma-native ``$eq`` predicate
        # instead of falling back to the post-filter, eliminating
        # metadata materialization for the common multi-tenant
        # scoping pattern (e.g. ``{"domain_id": "x"}``).
        # Defaults to empty (current post-filter behavior preserved).
        # See ``VECTOR_FILTER_SEMANTICS.md`` for the partition rules.
        self.scalar_metadata_keys: frozenset[str] = (
            cfg.scalar_metadata_keys or frozenset()
        )

        # Handle embedding function
        self.embedding_function = None
        ef = cfg.embedding_function
        if ef is not None:
            if isinstance(ef, str):
                # Map string to Chroma embedding functions
                if ef == "default":
                    from chromadb.utils import embedding_functions
                    self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                elif ef == "openai":
                    from chromadb.utils import embedding_functions
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=cfg.openai_api_key)
                # Add more as needed
            else:
                self.embedding_function = ef

        # Map distance metrics
        metric_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.L2: "l2",
            DistanceMetric.DOT_PRODUCT: "ip",
            DistanceMetric.INNER_PRODUCT: "ip",
        }
        self.chroma_metric = metric_map.get(self.metric, "cosine")

        # Typed ``Any``: the chromadb client/collection types are
        # untyped, and these start ``None`` until ``initialize`` builds
        # them — annotating ``Any`` keeps the post-init attribute accesses
        # (and the ``to_thread`` offload call sites) type-clean.
        self.client: Any = None
        self.collection: Any = None

    # chromadb's metadata contract is scalar-only (str/int/float/bool).
    # It rejects an empty/``None`` metadata dict outright, and — worse —
    # chromadb 1.x *silently accepts* a list-valued metadata value and
    # then corrupts it: the value bleeds positionally across unrelated
    # collections sharing chromadb's process-wide in-memory System
    # (reproduced as cross-test ``metadata_fields`` contamination).
    #
    # So every non-scalar value (any list — including ``[]`` — and any
    # dict) is encoded to a reversible scalar string on write and
    # restored on read, keeping chromadb scalar-only while preserving the
    # cross-backend round-trip contract (Memory/FAISS preserve
    # ``{"k": []}`` and ``{"k": [...]}`` as real values). The NUL-
    # delimited prefixes make a real-value collision infeasible.
    #
    # ``_EMPTY_LIST_SENTINEL`` is retained for backward-compatible decode
    # of data written by earlier versions (which sentinelled only ``[]``);
    # new writes use the JSON form uniformly for all non-scalars.
    _EMPTY_LIST_SENTINEL = "\x00dk\x00empty_list\x00"
    _NONSCALAR_PREFIX = "\x00dk\x00json\x00"

    @classmethod
    def _encode_metadata(
        cls, meta: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Adapt one row's metadata to chromadb's scalar-only contract.

        Map an empty/``None`` dict to ``None`` (the only "no metadata"
        form chromadb accepts). JSON-encode every list/dict value behind
        :attr:`_NONSCALAR_PREFIX` so chromadb only ever stores scalars
        (lists corrupt across collections otherwise). Scalars pass
        through. Inverse of :meth:`_decode_metadata`.
        """
        if not meta:
            return None
        encoded: dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, (list, dict)):
                encoded[key] = cls._NONSCALAR_PREFIX + json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
            else:
                encoded[key] = value
        return encoded or None

    @classmethod
    def _decode_metadata(
        cls, meta: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Reverse :meth:`_encode_metadata`.

        chromadb returns ``None`` for rows stored without metadata;
        surface ``{}`` to match the Memory/FAISS contract. JSON-prefixed
        values are parsed back to their list/dict form; the legacy
        empty-list sentinel still decodes to ``[]``.
        """
        if not meta:
            return {}
        decoded: dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, str) and value.startswith(
                cls._NONSCALAR_PREFIX
            ):
                decoded[key] = json.loads(
                    value[len(cls._NONSCALAR_PREFIX):]
                )
            elif value == cls._EMPTY_LIST_SENTINEL:
                decoded[key] = []
            else:
                decoded[key] = value
        return decoded

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Coerce a chromadb result field to a plain list.

        chromadb 1.x returns result fields (e.g. ``embeddings``) as
        numpy arrays; bare truthiness or ``x or []`` on an ndarray
        raises ``ValueError: ambiguous truth value``. ``None`` → ``[]``.
        """
        if value is None:
            return []
        return list(value)

    async def initialize(self) -> None:
        """Initialize Chroma client and collection.

        chromadb's client/collection API is synchronous; every call here
        (client construction, ``get_collection`` / ``create_collection``)
        is offloaded via :func:`asyncio.to_thread` so the on-disk sqlite
        load and index setup do not block the event loop.
        """
        if self._initialized:
            return

        # Create client. ``Settings(...)`` is built inside the worker
        # thread too: pydantic-settings reads ``.env`` files on
        # construction (a blocking ``os.stat``), so evaluating it as an
        # argument on the loop would defeat the offload.
        if self.persist_path:
            persist_path = self.persist_path

            def _make_client() -> Any:
                return chromadb.PersistentClient(
                    path=persist_path,
                    settings=Settings(anonymized_telemetry=False),
                )
        else:
            def _make_client() -> Any:
                return chromadb.Client(
                    settings=Settings(anonymized_telemetry=False)
                )

        self.client = await asyncio.to_thread(_make_client)

        # Get or create collection
        try:
            self.collection = await asyncio.to_thread(
                self.client.get_collection,
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        except ChromaNotFoundError:
            # Collection doesn't exist yet — create it. Only a genuine
            # "not found" triggers creation; transport / auth / internal
            # errors propagate rather than being misread as absence (which
            # would surface the real error obscurely on create_collection).
            self.collection = await asyncio.to_thread(
                self.client.create_collection,
                name=self.collection_name,
                metadata={"hnsw:space": self.chroma_metric},
                embedding_function=self.embedding_function,
            )

        self._initialized = True

    async def close(self) -> None:
        """Close Chroma client."""
        # Chroma handles persistence automatically
        self._initialized = False

    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to the collection."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Convert to list format for Chroma
        if isinstance(vectors, np.ndarray):
            if vectors.ndim == 1:
                vectors = [vectors.tolist()]
            else:
                vectors = vectors.tolist()
        elif isinstance(vectors, list) and len(vectors) > 0:
            if isinstance(vectors[0], np.ndarray):
                vectors = [v.tolist() for v in vectors]

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Per-row metadata: fresh dicts with config-level domain_id
        # defaulted in (caller's dicts never aliased — Items #8 / 131).
        metadata = self._apply_domain_default(metadata, len(ids))

        # Add to collection. chromadb 1.x rejects empty dict / empty-list
        # metadata; encode per row (decoded back on read). Offloaded:
        # chromadb's add is a synchronous native call.
        await asyncio.to_thread(
            self.collection.add,
            embeddings=vectors,
            ids=ids,
            metadatas=[self._encode_metadata(m) for m in metadata],
        )

        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Get from collection
        include = (
            ["embeddings", "metadatas"]
            if include_metadata
            else ["embeddings"]
        )
        result = await asyncio.to_thread(
            self.collection.get, ids=ids, include=include
        )

        # chromadb 1.x returns ndarrays — coerce before truthiness/index.
        result_ids = self._as_list(result.get("ids"))
        embeddings = self._as_list(result.get("embeddings"))
        metadatas = self._as_list(result.get("metadatas"))
        index_of = {rid: i for i, rid in enumerate(result_ids)}

        vectors: list[tuple[np.ndarray | None, dict[str, Any] | None]] = []
        for id_val in ids:
            idx = index_of.get(id_val)
            if idx is None:
                vectors.append((None, None))
                continue
            emb = embeddings[idx] if idx < len(embeddings) else None
            if emb is not None:
                emb = np.array(emb, dtype=np.float32)
            meta = (
                self._decode_metadata(metadatas[idx])
                if include_metadata and idx < len(metadatas)
                else None
            )
            vectors.append((emb, meta))

        return vectors

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        # Check which IDs exist
        existing = await asyncio.to_thread(
            self.collection.get, ids=ids, include=[]
        )
        existing_ids = self._as_list(existing.get("ids"))

        if existing_ids:
            await asyncio.to_thread(self.collection.delete, ids=existing_ids)
            return len(existing_ids)

        return 0

    @staticmethod
    def _filter_is_unsatisfiable(filter: dict[str, Any] | None) -> bool:
        """Return True when ``filter`` can never match any record.

        A filter element with an empty-list value rejects everything
        under four-quadrant semantics (intersection with the empty set
        is empty). Used at every Chroma read entry point — ``search``,
        ``search_documents``, ``count`` — to short-circuit before
        Chroma is touched, avoiding a pointless ``k * 4`` over-fetch
        that would be entirely rejected by the post-filter.

        The optimization is Chroma-specific: Memory/FAISS post-filter
        only (no over-fetch), and pgvector emits ``FALSE`` in SQL so
        Postgres short-circuits the scan itself.
        """
        if not filter:
            return False
        return any(isinstance(v, list) and not v for v in filter.values())

    def _partition_filter_for_chroma(
        self, filter: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Split filter into (native_where, python_postfilter).

        Chroma's metadata match operates on scalar metadata fields
        only. For list-valued metadata, ``$eq`` returns zero rows — a
        real bug for consumers whose metadata carries tag/category/
        domain lists. This helper partitions the filter so:

        * Scalar filter values for keys NOT declared in
          ``scalar_metadata_keys``: **not** pushed into Chroma
          ``$eq`` (which would mis-match list metadata). Kept for
          Python-side post-filtering via ``_match_metadata_filter``.
          Chroma still ranks the top-k by similarity — we only relax
          the metadata gate.
        * Scalar filter values for keys declared in
          ``scalar_metadata_keys``: pushed as Chroma-native ``$eq``.
          The consumer's declaration is the contract: stored values
          for these keys are always scalar, so ``$eq`` is correct
          and no post-filter is needed.
        * List filter values: pushed as ``$in`` ONLY for keys declared
          in ``scalar_metadata_keys``. chromadb's where-engine returns
          zero rows for any predicate against list-valued metadata, so
          for undeclared (possibly list-valued) keys the list filter is
          post-filter only — ``_match_metadata_filter`` applies the
          non-empty-intersection four-quadrant semantics.

        An empty/``None`` filter returns ``(None, {})``.
        """
        if not filter:
            return None, {}
        native: dict[str, Any] = {}
        post: dict[str, Any] = {}
        for key, value in filter.items():
            if isinstance(value, list):
                # Empty-list filter never matches under four-quadrant
                # semantics (the unsatisfiable short-circuit handles it
                # upstream). For a non-empty list filter, push a native
                # ``$in`` ONLY when the consumer declared the key
                # always-scalar: chromadb's where-engine returns zero
                # rows for ANY predicate against list-valued metadata
                # (verified on chromadb 1.x), so pushing ``$in`` for a
                # possibly list-valued key over-restricts to nothing.
                # Undeclared keys post-filter only — correctness via
                # ``_match_metadata_filter``.
                if value and key in self.scalar_metadata_keys:
                    native[key] = {"$in": value}
                post[key] = value
            else:
                # Scalar filter value. Push down ``$eq`` only when
                # the consumer has declared the key as always-scalar
                # in metadata. Otherwise post-filter to handle
                # potential list-valued metadata correctly.
                if key in self.scalar_metadata_keys:
                    native[key] = {"$eq": value}
                else:
                    post[key] = value
        return (native or None), post

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors."""
        if not self._initialized:
            await self.initialize()

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        # Convert query vector
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        if self._filter_is_unsatisfiable(filter):
            return []

        where, post_filter = self._partition_filter_for_chroma(filter or {})

        # Over-fetch when post-filtering: the native ``where`` is looser
        # than the effective filter, so naive ``n_results=k`` may return
        # zero post-filter matches. 4x is a pragmatic default; not made
        # configurable in this item.
        over_k = k * 4 if post_filter else k

        # Always fetch metadata when post-filtering (we need it for the
        # Python-side check) even if the caller didn't ask for it.
        need_metadata = include_metadata or bool(post_filter)
        include = ["metadatas", "distances"] if need_metadata else ["distances"]

        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_vector],
            n_results=over_k,
            where=where,
            include=include,
        )

        # chromadb 1.x returns nested ndarrays — coerce before any
        # truthiness/index, then decode metadata for parity with
        # Memory/FAISS (sentinel → [], no-metadata → {}).
        ids_groups = self._as_list(results.get("ids"))
        if not ids_groups:
            return []
        ids = self._as_list(ids_groups[0])
        if not ids:
            return []
        dist_groups = self._as_list(results.get("distances"))
        distances = (
            self._as_list(dist_groups[0])
            if dist_groups
            else [0.0] * len(ids)
        )
        meta_groups = self._as_list(results.get("metadatas"))
        metadatas = (
            self._as_list(meta_groups[0])
            if need_metadata and meta_groups
            else [None] * len(ids)
        )

        search_results: list[tuple[str, float, dict[str, Any] | None]] = []
        for id_val, distance, raw_meta in zip(
            ids, distances, metadatas, strict=False
        ):
            metadata = self._decode_metadata(raw_meta)
            if post_filter and not self._match_metadata_filter(
                metadata, post_filter
            ):
                continue

            if self.metric == DistanceMetric.COSINE:
                score = 1.0 - distance
            elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
                score = 1.0 / (1.0 + distance)
            else:
                score = float(distance)

            search_results.append(
                (id_val, score, metadata if include_metadata else None)
            )
            if len(search_results) >= k:
                break

        return search_results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors."""
        if not self._initialized:
            await self.initialize()

        # Check which IDs exist
        existing = await asyncio.to_thread(
            self.collection.get, ids=ids, include=[]
        )
        existing_ids = set(self._as_list(existing.get("ids")))

        if existing_ids:
            # Filter metadata to only update existing vectors
            filtered_ids = []
            filtered_metadata = []
            for id_val, meta in zip(ids, metadata, strict=False):
                if id_val in existing_ids:
                    filtered_ids.append(id_val)
                    filtered_metadata.append(self._encode_metadata(meta))

            if filtered_ids:
                await asyncio.to_thread(
                    self.collection.update,
                    ids=filtered_ids,
                    metadatas=filtered_metadata,
                )
                return len(filtered_ids)

        return 0

    async def update_metadata_where(
        self,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Merge ``set_`` into metadata of every filter-matched vector.

        Mirrors the filtered :meth:`clear` path: partition the filter
        into a Chroma-native ``where`` plus a Python post-filter, fetch
        matching rows with their metadata, merge ``set_`` into each
        (Chroma ``update`` replaces a row's metadata wholesale, so the
        merge is done here), then write back via ``collection.update``.
        """
        if not self._initialized:
            await self.initialize()

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        if filter is not None and self._filter_is_unsatisfiable(filter):
            return 0

        where, post_filter = self._partition_filter_for_chroma(filter or {})
        result = await asyncio.to_thread(
            self.collection.get,
            where=where if where else None,
            include=["metadatas"],
        )
        ids = self._as_list(result.get("ids"))
        metadatas = self._as_list(result.get("metadatas"))

        update_ids: list[str] = []
        update_metadatas: list[dict[str, Any] | None] = []
        for cid, raw_meta in zip(ids, metadatas, strict=False):
            existing = self._decode_metadata(raw_meta)
            if post_filter and not self._match_metadata_filter(
                existing, post_filter
            ):
                continue
            existing.update(set_)
            update_ids.append(cid)
            update_metadatas.append(self._encode_metadata(existing))

        if update_ids:
            await asyncio.to_thread(
                self.collection.update,
                ids=update_ids,
                metadatas=update_metadatas,
            )
        return len(update_ids)

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors in the collection.

        Uses ``collection.get(where=...)`` to enumerate matching
        IDs and post-filter through ``_match_metadata_filter`` only
        when the partitioned filter has a post-filter remainder.
        Replaces the previous dummy-vector query path which was capped
        at one result and therefore fundamentally wrong as a count.

        Memory profile:

        * No filter: native ``collection.count()`` — O(1).
        * Filter fully push-down (all values list-typed, or all
          scalar values for keys declared in
          ``scalar_metadata_keys``): ``collection.get(where=...,
          include=[])`` returns IDs only — no metadata
          materialization.
        * Filter partially or fully post-filter (scalar values for
          undeclared keys): ``collection.get(where=...,
          include=["metadatas"])`` materializes matching metadata
          for Python-side narrowing. A first-class filtered-count
          API is a Chroma upstream limitation. See
          ``VECTOR_FILTER_SEMANTICS.md`` for details and the
          ``scalar_metadata_keys`` opt-in.
        """
        if not self._initialized:
            await self.initialize()

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        if filter is None:
            return await asyncio.to_thread(self.collection.count)

        if self._filter_is_unsatisfiable(filter):
            return 0

        where, post_filter = self._partition_filter_for_chroma(filter)

        if not post_filter:
            # Filter fully pushed down. Skip metadata materialization
            # — IDs are sufficient for the count.
            result = await asyncio.to_thread(
                self.collection.get, where=where, include=[]
            )
            return len(self._as_list(result.get("ids")))

        result = await asyncio.to_thread(
            self.collection.get, where=where, include=["metadatas"]
        )
        metadatas = self._as_list(result.get("metadatas"))
        return sum(
            1
            for m in metadatas
            if self._match_metadata_filter(
                self._decode_metadata(m), post_filter
            )
        )

    async def metadata_fields(self) -> set[str]:
        """Discover metadata field names across all stored vectors."""
        if not self._initialized:
            await self.initialize()

        # Fetch all metadata from the collection
        result = await asyncio.to_thread(
            self.collection.get, include=["metadatas"]
        )
        fields: set[str] = set()
        for meta in self._as_list(result.get("metadatas")):
            fields.update(self._decode_metadata(meta).keys())
        return fields

    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        """Clear vectors, optionally filtered by metadata.

        Unfiltered clear keeps the existing
        ``delete_collection`` + ``create_collection`` shape (cheaper
        than scanning IDs).  Filtered clear partitions the filter via
        :meth:`_partition_filter_for_chroma`: a Chroma-native
        ``where`` narrows the candidate set, then Python-side
        post-filtering through :meth:`_match_metadata_filter` matches
        the four-quadrant semantics of every other backend before
        ``collection.delete(ids=...)``.
        """
        if not self._initialized:
            await self.initialize()

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        if not filter:
            # Delete and recreate collection
            await asyncio.to_thread(
                self.client.delete_collection, name=self.collection_name
            )
            self.collection = await asyncio.to_thread(
                self.client.create_collection,
                name=self.collection_name,
                metadata={"hnsw:space": self.chroma_metric},
                embedding_function=self.embedding_function,
            )
            return

        if self._filter_is_unsatisfiable(filter):
            # An empty-list filter element matches nothing under
            # four-quadrant semantics — clear is a no-op.
            return

        where, post_filter = self._partition_filter_for_chroma(filter)
        result = await asyncio.to_thread(
            self.collection.get,
            where=where if where else None,
            include=["metadatas"],
        )
        ids = self._as_list(result.get("ids"))
        metadatas = self._as_list(result.get("metadatas"))
        ids_to_delete = [
            cid
            for cid, meta in zip(ids, metadatas, strict=False)
            if self._match_metadata_filter(
                self._decode_metadata(meta), post_filter
            )
        ]
        if ids_to_delete:
            await asyncio.to_thread(self.collection.delete, ids=ids_to_delete)

    async def add_documents(
        self,
        documents: list[str],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to the collection (uses Chroma's embedding)."""
        if not self._initialized:
            await self.initialize()

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]

        # Ensure metadata is provided
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        # Add documents (Chroma will embed them if embedding_function is set)
        await asyncio.to_thread(
            self.collection.add,
            documents=documents,
            ids=ids,
            metadatas=[self._encode_metadata(m) for m in metadata],
        )

        return ids

    async def search_documents(
        self,
        query_text: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, str, dict[str, Any] | None]]:
        """Search using text query (uses Chroma's embedding)."""
        if not self._initialized:
            await self.initialize()

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        if self._filter_is_unsatisfiable(filter):
            return []

        where, post_filter = self._partition_filter_for_chroma(filter or {})

        over_k = k * 4 if post_filter else k

        # Always need metadata when post-filtering — caller-visible
        # surface still respects include_metadata.
        results = await asyncio.to_thread(
            self.collection.query,
            query_texts=[query_text],
            n_results=over_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[tuple[str, float, str, dict[str, Any] | None]] = []
        ids_groups = self._as_list(results.get("ids"))
        if not ids_groups:
            return []
        ids = self._as_list(ids_groups[0])
        if not ids:
            return []
        dist_groups = self._as_list(results.get("distances"))
        distances = (
            self._as_list(dist_groups[0])
            if dist_groups
            else [0.0] * len(ids)
        )
        doc_groups = self._as_list(results.get("documents"))
        documents = (
            self._as_list(doc_groups[0])
            if doc_groups
            else [None] * len(ids)
        )
        meta_groups = self._as_list(results.get("metadatas"))
        metadatas = (
            self._as_list(meta_groups[0])
            if meta_groups
            else [None] * len(ids)
        )

        for id_val, distance, doc, raw_meta in zip(
            ids, distances, documents, metadatas, strict=False
        ):
            metadata = self._decode_metadata(raw_meta)
            if post_filter and not self._match_metadata_filter(
                metadata, post_filter
            ):
                continue
            score = 1.0 - distance  # Cosine distance to similarity
            search_results.append(
                (id_val, score, doc, metadata if include_metadata else None)
            )
            if len(search_results) >= k:
                break

        return search_results

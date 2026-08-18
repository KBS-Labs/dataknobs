"""Chroma vector store implementation."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from ..types import DistanceMetric
from .base import VectorStore
from .config import ChromaVectorStoreConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import ClassVar

    import numpy as np

_RowT = TypeVar("_RowT")
"""One assembled result row — a shape each search method decides."""

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
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")

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
        self.scalar_metadata_keys: frozenset[str] = cfg.scalar_metadata_keys or frozenset()

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

                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=cfg.openai_api_key
                    )
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

    # chromadb rejects an empty/``None`` metadata dict outright, and its
    # value domain is narrower than Python's. Every non-scalar value
    # (any list — including ``[]`` — and any dict) is therefore encoded
    # to a reversible scalar string on write and restored on read,
    # preserving the cross-backend round-trip contract (Memory/FAISS
    # preserve ``{"k": []}`` and ``{"k": [...]}`` as real values). The
    # NUL-delimited prefixes make a real-value collision infeasible.
    #
    # Two things justified the blanket rule, and only one still does.
    #
    # Early chromadb 1.x *silently accepted* a list-valued metadata
    # value and then corrupted it — the value bled positionally across
    # unrelated collections sharing chromadb's process-wide in-memory
    # System (reproduced as cross-test ``metadata_fields``
    # contamination). That is fixed as of 1.5.9, which has first-class
    # list metadata: a homogeneous non-empty list of str/int/float/bool
    # round-trips intact, does not bleed, and supports a native
    # ``$contains`` predicate. The declared floor is ``chromadb>=1.0.0``,
    # so the hazard is still reachable for a consumer on an older 1.x.
    #
    # What no version accepts is an empty list, a heterogeneous list, or
    # a dict — so the encoding cannot retire, only narrow.
    #
    # Narrowing it is worth doing: a natively-stored list would let the
    # scalar-filter-against-list-metadata quadrant push down as
    # ``$contains`` instead of driving the post-filter escalation that
    # dominates this backend's query cost. It cannot be done as a code
    # change alone. chromadb's operator set
    # (``$gt $gte $lt $lte $ne $eq $in $nin $contains $not_contains``)
    # has no prefix or pattern match, so an encoded value is invisible
    # to every filter — pushing ``$contains`` down against a collection
    # that still holds sentinel-encoded rows would silently miss them.
    # It therefore needs a collection-level format marker gating the
    # pushdown, plus a migration for existing collections.
    #
    # ``_EMPTY_LIST_SENTINEL`` is retained for backward-compatible decode
    # of data written by earlier versions (which sentinelled only ``[]``);
    # new writes use the JSON form uniformly for all non-scalars.
    _EMPTY_LIST_SENTINEL = "\x00dk\x00empty_list\x00"
    _NONSCALAR_PREFIX = "\x00dk\x00json\x00"

    # Row timestamps live in the metadata dict itself, because a Chroma
    # collection is the only per-row storage this backend has — the
    # other backends keep a side-car keyed by row and never face this.
    # In-band storage means the keys share a namespace the consumer
    # owns, so they take the same NUL-delimited form as the encoding
    # prefixes above and for the same reason: a real-value collision is
    # infeasible.
    #
    # The storage keys are deliberately NOT the configured output keys
    # (``timestamps.created_key`` / ``updated_key``). Keeping them
    # distinct is what preserves the documented collision policy — a
    # consumer's own ``_created_at`` stays ordinary metadata, and
    # ``_inject_timestamps`` arbitrates between the two exactly as it
    # does on every other backend. Were they the same key, a stored
    # value and a consumer-supplied one would be indistinguishable and
    # the policy could not be honoured here at all.
    #
    # The stored value is an epoch float regardless of the configured
    # ``timestamps.format``: format is an output concern, and a store
    # whose format changes must still be able to read rows written
    # under the old one.
    _TS_CREATED_KEY = "\x00dk\x00created\x00"
    _TS_UPDATED_KEY = "\x00dk\x00updated\x00"
    _RESERVED_KEYS = frozenset({_TS_CREATED_KEY, _TS_UPDATED_KEY})

    @classmethod
    def _encode_metadata(cls, meta: dict[str, Any] | None) -> dict[str, Any] | None:
        """Adapt one row's metadata to chromadb's scalar-only contract.

        Map an empty/``None`` dict to ``None`` (the only "no metadata"
        form chromadb accepts). JSON-encode every list/dict value behind
        :attr:`_NONSCALAR_PREFIX` so chromadb only ever stores scalars
        (lists corrupt across collections otherwise). Scalars pass
        through. Inverse of :meth:`_decode_metadata`, which strips the
        same reserved keys on the way back.

        The reserved timestamp keys are dropped here rather than left to
        :meth:`_stamp` to overwrite. ``_stamp`` returns early for a row
        the store does not yet track, so on a pre-tracking row nothing
        displaced a consumer value and a numeric one was read back as
        that row's real creation date. Every write path encodes, so this
        is the one place the exclusion holds structurally instead of as
        a side effect of stamping having something to do.
        """
        if not meta:
            return None
        encoded: dict[str, Any] = {}
        for key, value in meta.items():
            if key in cls._RESERVED_KEYS:
                continue
            if isinstance(value, (list, dict)):
                encoded[key] = cls._NONSCALAR_PREFIX + json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
            else:
                encoded[key] = value
        return encoded or None

    @classmethod
    def _decode_metadata(cls, meta: dict[str, Any] | None) -> dict[str, Any]:
        """Reverse :meth:`_encode_metadata`.

        chromadb returns ``None`` for rows stored without metadata;
        surface ``{}`` to match the Memory/FAISS contract. JSON-prefixed
        values are parsed back to their list/dict form; the legacy
        empty-list sentinel still decodes to ``[]``.

        Reserved timestamp keys are dropped here rather than at each
        caller. Every public read path decodes, so one strip at the
        boundary is what keeps them out of ``get_vectors``, ``search``,
        ``search_documents``, ``metadata_fields``, the residual
        post-filter, and the merge round trip in
        :meth:`update_metadata_where` — seven sites that would
        otherwise each have to remember. Read the values with
        :meth:`_reserved_timestamps`, which takes the raw dict.
        """
        if not meta:
            return {}
        decoded: dict[str, Any] = {}
        for key, value in meta.items():
            if key in cls._RESERVED_KEYS:
                continue
            if isinstance(value, str) and value.startswith(cls._NONSCALAR_PREFIX):
                decoded[key] = json.loads(value[len(cls._NONSCALAR_PREFIX) :])
            elif value == cls._EMPTY_LIST_SENTINEL:
                decoded[key] = []
            else:
                decoded[key] = value
        return decoded

    @classmethod
    def _reserved_timestamps(
        cls, raw: dict[str, Any] | None
    ) -> tuple[datetime | None, datetime | None]:
        """The row's tracked timestamps, read from *raw* chromadb metadata.

        Takes the undecoded dict on purpose: :meth:`_decode_metadata`
        strips these keys, so a decoded row no longer has them.

        Returns ``(None, None)`` for a row written before this backend
        tracked timestamps. No backfill — that is the same answer the
        other backends give for their own pre-tracking rows (a pgvector
        row from before the migration, a Memory/FAISS pickle written
        without a side-car), and the value repopulates on the row's
        next write.
        """

        def _at(key: str) -> datetime | None:
            value = (raw or {}).get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return datetime.fromtimestamp(float(value), UTC)
            return None

        return _at(cls._TS_CREATED_KEY), _at(cls._TS_UPDATED_KEY)

    @classmethod
    def _stamp(
        cls,
        encoded: dict[str, Any] | None,
        *,
        created: datetime | None,
        updated: datetime,
        start_tracking: bool,
    ) -> dict[str, Any]:
        """Attach the reserved timestamp keys to an encoded payload.

        ``created`` of ``None`` is ambiguous on its own — it means
        either "brand new row" or "row that predates tracking" — and the
        two want opposite answers, so the caller disambiguates with
        *start_tracking* rather than this method guessing.

        ``start_tracking=True`` is the write paths (``add_vectors``,
        ``add_documents``): a row being written begins its life now, and
        a ``created`` read back from an existing row is preserved, which
        is what makes an upsert keep the original date.

        ``start_tracking=False`` is the update paths. An update does not
        begin tracking a row that was never tracked: ``created_at`` of
        ``None`` means *not known*, and inventing one would silently
        record the update time as a creation date with nothing left to
        distinguish it from a real one. The other three backends already
        behave this way — Memory and FAISS guard on the row having a
        side-car entry, pgvector leaves a NULL ``created_at`` alone — so
        the row simply stays untracked until a write establishes it.

        Returns a dict even when *encoded* is ``None``. A row the
        consumer gave no metadata still has timestamps, and chromadb's
        "no metadata" form cannot carry them.
        """
        payload = dict(encoded or {})
        if created is None and not start_tracking:
            return payload
        payload[cls._TS_CREATED_KEY] = (created or updated).timestamp()
        payload[cls._TS_UPDATED_KEY] = updated.timestamp()
        return payload

    @classmethod
    def _replacement_payload(
        cls,
        stored: dict[str, Any] | None,
        new_meta: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """The ``update`` payload that makes ``new_meta`` the whole row.

        chromadb's ``update`` merges the dict it is given into what is
        already stored, and a key it does not mention survives. So
        replacement is expressed by naming the departing keys with a
        ``None`` value, which deletes them.

        ``stored`` is the raw chromadb metadata, not a decoded one:
        only its key set is used, and encoding does not change that.

        Returns ``None`` when neither the stored row nor the
        replacement has any keys, matching the "no metadata" form
        :meth:`_encode_metadata` produces. It does *not* reach chromadb
        in that form: every caller passes the result through
        :meth:`_stamp`, which may add the reserved timestamp keys and
        always returns a dict. Deciding whether the final payload is
        writable is therefore the caller's job, not this method's —
        chromadb rejects an empty update dict, so a payload that is
        still empty after stamping is skipped rather than sent.
        """
        payload: dict[str, Any] = dict(cls._encode_metadata(new_meta) or {})
        for key in stored or {}:
            if key not in payload:
                payload[key] = None
        return payload or None

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
                return chromadb.Client(settings=Settings(anonymized_telemetry=False))

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

        # An empty batch is a no-op, not an error: see
        # ``VectorStoreBase._is_empty_batch``.
        if self._is_empty_batch(vectors):
            return []

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

        # Per-row metadata (fresh dicts, config-level domain_id defaulted
        # in — caller's dicts never aliased) is built inside
        # ``_stamped_payloads``, which is the only place the stored rows
        # the default has to preserve are in hand.

        # Add to collection. chromadb 1.x rejects empty dict / empty-list
        # metadata; encode per row (decoded back on read). Offloaded:
        # chromadb's upsert is a synchronous native call.
        #
        # ``upsert``, not ``add``: re-adding an id a store already holds
        # replaces that row on every other backend, and chromadb's
        # ``add`` instead discards the write silently — no exception,
        # no warning, the original vector and metadata retained. A
        # consumer correcting an embedding kept the stale one here and
        # had no way to tell.
        await asyncio.to_thread(
            self.collection.upsert,
            embeddings=vectors,
            ids=ids,
            metadatas=await self._stamped_payloads(ids, metadata),
        )

        return ids

    async def _stamped_payloads(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Encoded metadata for a write, carrying the reserved timestamps.

        One ``collection.get`` for the whole batch, not one per row, and
        it answers two questions at once: which of these ids already
        exist and when they were created, and which keys each of them
        currently holds. Re-adding an existing id is an upsert on every
        backend, and an upsert preserves ``created_at`` while advancing
        ``updated_at`` — so the prior value has to be read before it is
        overwritten.

        The second question is why the whole stored dict is kept rather
        than just the timestamp. chromadb's ``upsert`` **merges** its
        metadata into what is already stored, exactly as ``update``
        does — a key the caller omits survives. Re-adding an id with
        ``{"rev": 2}`` over a stored ``{"tenant": "A", "rev": 1}``
        therefore left ``tenant`` behind here while Memory, FAISS and
        pgvector all replaced the row outright, and re-adding with no
        metadata at all kept the entire prior dict. So the payload goes
        through :meth:`_replacement_payload`, which names the departing
        keys with ``None`` to delete them — the same mechanism
        ``update_metadata`` uses, for the same reason.

        The reserved keys are stored keys the caller never supplies, so
        that tombstoning covers them too; ``_stamp`` re-sets both
        immediately afterwards, and with ``start_tracking=True`` it
        always returns a non-empty payload, so no row here can reach
        the empty-update form chromadb rejects.

        One ``now`` for the batch, so rows written together carry the
        same instant rather than drifting across the loop.
        """
        existing = await asyncio.to_thread(self.collection.get, ids=ids, include=["metadatas"])
        existing_ids = self._as_list(existing.get("ids"))
        existing_metas = self._as_list(existing.get("metadatas"))
        stored_by_id: dict[str, dict[str, Any] | None] = {
            rid: (existing_metas[i] if i < len(existing_metas) else None)
            for i, rid in enumerate(existing_ids)
        }

        # A scoped store may not write an id another domain owns: this
        # path upserts, so that write would capture the row rather than
        # add one. It lives here rather than in each caller because both
        # write paths reach it, and the ``get`` above has already paid
        # for the stored metadata the check needs. Raising before the
        # ``upsert`` is what keeps a rejected batch from landing
        # partially — chromadb has no transaction to roll back.
        decoded_by_id = {rid: self._decode_metadata(raw) for rid, raw in stored_by_id.items()}
        self._reject_out_of_scope_ids(decoded_by_id)

        now = datetime.now(UTC)
        # The configured ``domain_id`` is defaulted in here rather than
        # in each caller, for the same reason the guard above is: both
        # write paths reach this method, and the decoded stored rows the
        # default needs to preserve a co-owned scope are the ones the
        # guard just read. Applying it in the callers meant the default
        # could not see what it was overwriting, so a ``t1`` store's
        # silent write re-stamped ``"t1"`` over a stored ``["t1", "t2"]``
        # and evicted the co-owner.
        rows = self._apply_domain_default(metadata, len(ids), ids=ids, stored=decoded_by_id)
        return [
            self._stamp(
                self._replacement_payload(stored_by_id.get(id_val), meta),
                created=self._reserved_timestamps(stored_by_id.get(id_val))[0],
                updated=now,
                # A write establishes tracking: a genuinely new id has no
                # stored ``created`` and starts its life now.
                start_tracking=True,
            )
            for id_val, meta in zip(ids, rows, strict=False)
        ]

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID.

        ``include_timestamps`` injects the configured timestamp keys
        from the values this store tracks per row. A row written before
        this backend tracked them reports ``None`` for both until its
        next write — the same answer the contract defines for a
        pgvector row from before the migration or a Memory/FAISS pickle
        written without a side-car.
        """
        if not self._initialized:
            await self.initialize()

        # chromadb's ``validate_ids`` rejects an empty list before the
        # query runs, so this has to be answered here rather than by
        # the backend. An empty id list is a well-formed question with
        # an empty answer — the id-keyed counterpart of
        # ``_is_empty_batch``, and the same contract the other three
        # backends have always had.
        if not ids:
            return []

        import numpy as np

        inject = include_timestamps and include_metadata

        # Get from collection. Metadata is fetched whenever a scope is
        # configured even if the caller did not ask for it: on this
        # backend the row's ``domain_id`` lives in that dict, so it is
        # what the scope check has to read.
        want_meta = include_metadata or self._is_scoped
        include = ["embeddings", "metadatas"] if want_meta else ["embeddings"]
        result = await asyncio.to_thread(self.collection.get, ids=ids, include=include)

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
            raw_meta = metadatas[idx] if idx < len(metadatas) else None
            # Decoded once and reused: the scope check and the returned
            # value want the same dict, and decoding parses every
            # sentinel-encoded value in the row.
            decoded = self._decode_metadata(raw_meta)
            # Out-of-domain rows answer exactly as absent ones do, so a
            # caller cannot distinguish "not here" from "not yours".
            if not self._in_configured_domain(decoded):
                vectors.append((None, None))
                continue
            meta = decoded if include_metadata else None
            if inject:
                created, updated = self._reserved_timestamps(raw_meta)
                meta = self._inject_timestamps(meta, created=created, updated=updated)
            vectors.append((emb, meta))

        return vectors

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        # chromadb's ``validate_ids`` rejects an empty list before the
        # query runs, so this has to be answered here rather than by
        # the backend. An empty id list is a well-formed question with
        # an empty answer — the id-keyed counterpart of
        # ``_is_empty_batch``, and the same contract the other three
        # backends have always had.
        if not ids:
            return 0

        # Metadata, not just ids: it carries the scope each candidate
        # has to be checked against before it can be deleted.
        existing = await asyncio.to_thread(self.collection.get, ids=ids, include=["metadatas"])
        existing_metas = self._as_list(existing.get("metadatas"))
        existing_ids = [
            rid
            for i, rid in enumerate(self._as_list(existing.get("ids")))
            if self._in_configured_domain(
                self._decode_metadata(existing_metas[i] if i < len(existing_metas) else None)
            )
        ]

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
            if self._is_scoped and key == "domain_id":
                # The configured scope key never pushes down, whatever
                # the consumer declared. ``scalar_metadata_keys`` is a
                # promise about stored values, and this is the one key
                # the write path cannot keep it for: the scope is a
                # *default*, so a caller can store a list here through
                # the public API, and a co-owned row is a documented
                # shape. A list is stored sentinel-encoded, so a native
                # ``$eq`` against the encoded string matches nothing and
                # the filter-keyed half goes blind to a row the id-keyed
                # half still returns — the split that resolving scope
                # through one evaluator exists to prevent.
                post[key] = value
                continue
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

    def _score_from_distance(self, distance: float) -> float:
        """Convert a Chroma distance into the score callers see.

        The collection is created with ``hnsw:space`` set from the
        configured metric, so the distances it returns are cosine, L2 or
        inner-product accordingly — and the conversion has to follow.
        Shared by :meth:`search` and :meth:`search_documents`, which
        previously each decided this for themselves and disagreed:
        ``search_documents`` applied the cosine conversion unconditionally
        and so reported wrong scores on any store configured
        ``euclidean``, ``l2``, ``dot_product`` or ``inner_product``.
        """
        if self.metric == DistanceMetric.COSINE:
            return 1.0 - distance
        if self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            return 1.0 / (1.0 + distance)
        return float(distance)

    async def _search_with_escalation(
        self,
        k: int,
        post_filter: dict[str, Any],
        run_query: Callable[[int], Awaitable[Any]],
        build_rows: Callable[[Any], list[_RowT]],
    ) -> list[_RowT]:
        """Query with a widening ``n_results`` until ``k`` rows survive.

        Chroma truncates to ``n_results`` before this store's residual
        Python filter runs, so every candidate that filter drops is a row
        the caller asked for and does not get. Asking for ``k *
        POST_FILTER_OVERFETCH`` compensates for the common case and no
        more: a filter matching fewer than one candidate in
        ``POST_FILTER_OVERFETCH`` still under-returns, and can return
        nothing at all from a collection holding many matches — the same
        "count says N, search says zero" shape the FAISS filtered path
        was fixed for, at a wider window.

        So the fetch escalates instead of settling. ``collection.count()``
        is native and O(1), which makes it a usable ceiling: the sequence
        doubles up to the whole collection, at which point Chroma has
        returned every row and the post-filter's answer is exact rather
        than merely over-fetched. The count is only taken when there *is*
        a residual filter — a fully pushed-down one needs no compensation
        and pays nothing for this.
        """
        if k <= 0:
            return []

        ceiling: int | None = None
        if post_filter:
            ceiling = int(await asyncio.to_thread(self.collection.count))
            if ceiling <= 0:
                return []

        rows: list[_RowT] = []
        for n_results in self._overfetch_sizes(
            k, has_post_filter=bool(post_filter), ceiling=ceiling
        ):
            if n_results <= 0:
                break
            rows = build_rows(await run_query(n_results))
            if len(rows) >= k:
                break
        return rows[:k]

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_timestamps: bool = False,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors.

        See :meth:`get_vectors` for what ``include_timestamps`` exposes
        and what a row written before tracking reports.
        """
        if not self._initialized:
            await self.initialize()

        inject = include_timestamps and include_metadata

        # Apply config-level domain_id scoping (no-op when unset).
        filter = self._effective_filter(filter)

        # Convert query vector
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        if self._filter_is_unsatisfiable(filter):
            return []

        where, post_filter = self._partition_filter_for_chroma(filter or {})

        # Always fetch metadata when post-filtering (we need it for the
        # Python-side check) even if the caller didn't ask for it.
        need_metadata = include_metadata or bool(post_filter)
        include = ["metadatas", "distances"] if need_metadata else ["distances"]

        async def run_query(n_results: int) -> Any:
            return await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where,
                include=include,
            )

        def build_rows(results: Any) -> list[tuple[str, float, dict[str, Any] | None]]:
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
            distances = self._as_list(dist_groups[0]) if dist_groups else [0.0] * len(ids)
            meta_groups = self._as_list(results.get("metadatas"))
            metadatas = (
                self._as_list(meta_groups[0])
                if need_metadata and meta_groups
                else [None] * len(ids)
            )

            rows: list[tuple[str, float, dict[str, Any] | None]] = []
            for id_val, distance, raw_meta in zip(ids, distances, metadatas, strict=False):
                metadata = self._decode_metadata(raw_meta)
                if post_filter and not self._match_metadata_filter(metadata, post_filter):
                    continue
                out_meta = metadata if include_metadata else None
                if inject:
                    created, updated = self._reserved_timestamps(raw_meta)
                    out_meta = self._inject_timestamps(out_meta, created=created, updated=updated)
                rows.append((id_val, self._score_from_distance(distance), out_meta))
            return rows

        return await self._search_with_escalation(k, post_filter, run_query, build_rows)

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Replace metadata for existing vectors.

        The supplied dict becomes the row's metadata outright: a key
        the caller omits is removed, matching every other backend and
        the "new metadata for each vector" the base class documents.

        chromadb's own ``update`` merges rather than replaces, so a
        removed key has to be named to be dropped — ``None`` as a
        value deletes that key. The stored metadata is fetched first
        for exactly that reason: the keys to tombstone are the ones
        the row has and the caller's dict does not.

        One consequence, and the only place this backend cannot match
        the others: a ``None`` **value** is not storable here. Removing
        a key and setting it to ``None`` are the same request in
        chromadb's update API, so ``{"reviewer": None}`` drops the key
        rather than storing it — Memory and FAISS keep it, and pgvector
        stores JSON ``null``. A consumer needing "present but empty" as
        a portable value should pick one this backend can hold, such as
        ``""`` or ``False``.

        This is not new — chromadb always deleted on ``None`` — but the
        replacement mechanism now *depends* on that behaviour, which
        turns an incidental limitation into a structural one.

        Replacing the consumer's keys does not disturb the row's
        timestamps: ``created_at`` is preserved and ``updated_at``
        advances, as on every other backend.
        """
        if not self._initialized:
            await self.initialize()

        # chromadb's ``validate_ids`` rejects an empty list before the
        # query runs, so this has to be answered here rather than by
        # the backend. An empty id list is a well-formed question with
        # an empty answer — the id-keyed counterpart of
        # ``_is_empty_batch``, and the same contract the other three
        # backends have always had.
        if not ids:
            return 0

        # Metadata, not just ids: the tombstone set is derived from the
        # keys the row currently holds.
        existing = await asyncio.to_thread(self.collection.get, ids=ids, include=["metadatas"])
        existing_ids = self._as_list(existing.get("ids"))
        existing_metas = self._as_list(existing.get("metadatas"))
        stored_by_id = {
            rid: (existing_metas[i] if i < len(existing_metas) else None)
            for i, rid in enumerate(existing_ids)
        }

        now = datetime.now(UTC)
        matched = 0
        # The configured ``domain_id`` is defaulted back in, exactly as
        # ``add_vectors`` does it: this path replaces the metadata dict
        # outright and the scope lives inside that dict, so a caller
        # updating one field would otherwise push the row out of its own
        # domain. Applied before ``_replacement_payload`` so the scope
        # is one of the keys being written rather than one of the keys
        # being tombstoned.
        rows = self._apply_domain_default(
            metadata,
            len(metadata),
            ids=ids,
            stored={rid: self._decode_metadata(raw) for rid, raw in stored_by_id.items()},
        )
        filtered_ids: list[str] = []
        filtered_metadata: list[dict[str, Any]] = []
        for id_val, meta in zip(ids, rows, strict=False):
            if id_val not in stored_by_id:
                continue
            stored = stored_by_id[id_val]
            # A scoped store may not rewrite another domain's row — and
            # since the replacement carries the configured scope, an
            # unguarded write would capture it rather than merely edit it.
            if not self._in_configured_domain(self._decode_metadata(stored)):
                continue
            matched += 1
            # Stamp after the replacement payload, not before: that
            # payload tombstones every stored key the caller omitted,
            # and the reserved keys are stored keys the caller never
            # supplies. Re-setting them here is what keeps a metadata
            # replacement from also erasing the row's timestamps.
            payload = self._stamp(
                self._replacement_payload(stored, meta),
                created=self._reserved_timestamps(stored)[0],
                updated=now,
                start_tracking=False,
            )
            # An empty payload is a row with nothing to write: an
            # untracked row, carrying no metadata, replaced by none.
            # chromadb rejects an empty update dict, and there is
            # nothing to send anyway — the requested state already
            # holds, so it counts as updated but is not written.
            if payload:
                filtered_ids.append(id_val)
                filtered_metadata.append(payload)

        if filtered_ids:
            await asyncio.to_thread(
                self.collection.update,
                ids=filtered_ids,
                metadatas=filtered_metadata,
            )
        return matched

    async def update_metadata_where(
        self,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Merge ``set_`` into metadata of every filter-matched vector.

        Mirrors the filtered :meth:`clear` path: partition the filter
        into a Chroma-native ``where`` plus a Python post-filter, fetch
        matching rows with their metadata, merge ``set_`` into each,
        then write back via ``collection.update``.

        Merging here rather than leaning on chromadb is deliberate on
        two counts. The rows have to be fetched and decoded regardless,
        because the residual post-filter decides which of them match;
        and doing the merge locally keeps this method's result
        independent of what chromadb's ``update`` does with the keys it
        is not given. It merges them today, but that is its choice, not
        this contract — and the sibling :meth:`update_metadata`
        deliberately overrides it to replace.
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

        now = datetime.now(UTC)
        matched = 0
        update_ids: list[str] = []
        update_metadatas: list[dict[str, Any]] = []
        for cid, raw_meta in zip(ids, metadatas, strict=False):
            existing = self._decode_metadata(raw_meta)
            if post_filter and not self._match_metadata_filter(existing, post_filter):
                continue
            existing.update(set_)
            matched += 1
            # ``existing`` came back from a decode, which strips the
            # reserved keys — so the re-encoded payload carries none,
            # and a merge-update would leave the stored ones untouched.
            # ``updated_at`` has to advance on a matched row, so both
            # are written back explicitly.
            payload = self._stamp(
                self._encode_metadata(existing),
                created=self._reserved_timestamps(raw_meta)[0],
                updated=now,
                start_tracking=False,
            )
            # See ``update_metadata``: an empty payload is a matched row
            # with nothing to write, and chromadb refuses one.
            if payload:
                update_ids.append(cid)
                update_metadatas.append(payload)

        if update_ids:
            await asyncio.to_thread(
                self.collection.update,
                ids=update_ids,
                metadatas=update_metadatas,
            )
        return matched

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
            result = await asyncio.to_thread(self.collection.get, where=where, include=[])
            return len(self._as_list(result.get("ids")))

        result = await asyncio.to_thread(self.collection.get, where=where, include=["metadatas"])
        metadatas = self._as_list(result.get("metadatas"))
        return sum(
            1
            for m in metadatas
            if self._match_metadata_filter(self._decode_metadata(m), post_filter)
        )

    async def metadata_fields(self) -> set[str]:
        """Discover metadata field names across all stored vectors."""
        if not self._initialized:
            await self.initialize()

        # Fetch all metadata from the collection
        result = await asyncio.to_thread(self.collection.get, include=["metadatas"])
        fields: set[str] = set()
        for meta in self._as_list(result.get("metadatas")):
            decoded = self._decode_metadata(meta)
            if self._in_configured_domain(decoded):
                fields.update(decoded.keys())
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
            await asyncio.to_thread(self.client.delete_collection, name=self.collection_name)
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
            if self._match_metadata_filter(self._decode_metadata(meta), post_filter)
        ]
        if ids_to_delete:
            await asyncio.to_thread(self.collection.delete, ids=ids_to_delete)

    async def add_documents(
        self,
        documents: list[str],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to the collection (uses Chroma's embedding).

        The rows land in the same collection :meth:`add_vectors` writes
        to, and every read path treats them identically — so this path
        prepares them identically too: an empty batch is a no-op, the
        configured ``domain_id`` is defaulted in, the row is timestamped,
        and an id already present is replaced rather than silently
        discarded. Diverging on any of the four produced rows the store
        itself could not see, could not date, or could not write.
        """
        if not self._initialized:
            await self.initialize()

        # An empty batch is a no-op, not an error: see
        # ``VectorStoreBase._is_empty_batch``. A chunker handed a blank
        # document describes this path more literally than the vector
        # one, and without the guard the empty list reached
        # ``_stamped_payloads`` and died inside chromadb's id validator.
        if self._is_empty_batch(documents):
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]

        # Per-row metadata carries the config-level domain_id, applied
        # inside ``_stamped_payloads`` exactly as the vector write path
        # gets it. Without it a scoped store wrote rows carrying no
        # domain_id, which every scoped read then filtered back out.

        # Add documents (Chroma will embed them if embedding_function is set)
        await asyncio.to_thread(
            self.collection.upsert,
            documents=documents,
            ids=ids,
            metadatas=await self._stamped_payloads(ids, metadata),
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

        # Always need metadata when post-filtering — caller-visible
        # surface still respects include_metadata.
        async def run_query(n_results: int) -> Any:
            return await asyncio.to_thread(
                self.collection.query,
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        def build_rows(results: Any) -> list[tuple[str, float, str, dict[str, Any] | None]]:
            ids_groups = self._as_list(results.get("ids"))
            if not ids_groups:
                return []
            ids = self._as_list(ids_groups[0])
            if not ids:
                return []
            dist_groups = self._as_list(results.get("distances"))
            distances = self._as_list(dist_groups[0]) if dist_groups else [0.0] * len(ids)
            doc_groups = self._as_list(results.get("documents"))
            documents = self._as_list(doc_groups[0]) if doc_groups else [None] * len(ids)
            meta_groups = self._as_list(results.get("metadatas"))
            metadatas = self._as_list(meta_groups[0]) if meta_groups else [None] * len(ids)

            rows: list[tuple[str, float, str, dict[str, Any] | None]] = []
            for id_val, distance, doc, raw_meta in zip(
                ids, distances, documents, metadatas, strict=False
            ):
                metadata = self._decode_metadata(raw_meta)
                if post_filter and not self._match_metadata_filter(metadata, post_filter):
                    continue
                rows.append(
                    (
                        id_val,
                        self._score_from_distance(distance),
                        doc,
                        metadata if include_metadata else None,
                    )
                )
            return rows

        # Same escalation as ``search`` above, for the same reason.
        return await self._search_with_escalation(k, post_filter, run_query, build_rows)

"""Content deduplication checking.

Provides a general-purpose utility for checking content uniqueness against
an existing dataset, combining exact hash matching with optional semantic
similarity via vector stores.

Example:
    >>> from dataknobs_data import DedupChecker, DedupConfig
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> db = AsyncMemoryDatabase()
    >>> checker = DedupChecker(db=db, config=DedupConfig())
    >>> await checker.register({"content": "Hello world"}, record_id="doc-1")
    >>> result = await checker.check({"content": "Hello world"})
    >>> result.is_exact_duplicate
    True
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from dataknobs_data.database import AsyncDatabase
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.records import Record
from dataknobs_data.vector.content import MODEL_NAME_KEY
from dataknobs_data.vector.embedding import (
    TextEmbedder,
    embed_text,
    require_embedding_source,
)

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    """Configuration for deduplication checking.

    Attributes:
        hash_fields: Field names used for computing the content hash.
        hash_algorithm: Hash algorithm to use (``"md5"`` or ``"sha256"``).
        semantic_check: Whether to perform semantic similarity search.
        semantic_fields: Fields concatenated for embedding. Defaults to
            ``hash_fields`` if not set.
        similarity_threshold: Minimum similarity score to consider a match.
        max_similar_results: Maximum number of similar items to return.
    """

    hash_fields: list[str] = field(default_factory=lambda: ["content"])
    hash_algorithm: str = "md5"
    semantic_check: bool = False
    semantic_fields: list[str] | None = None
    similarity_threshold: float = 0.92
    max_similar_results: int = 5


@dataclass
class SimilarItem:
    """A record that is semantically similar to the candidate.

    Attributes:
        record_id: The ID of the similar record.
        score: Similarity score (higher is more similar).
        matched_text: The text that was matched against.
    """

    record_id: str
    score: float
    matched_text: str = ""


@dataclass
class DedupResult:
    """Result of a deduplication check.

    Attributes:
        is_exact_duplicate: Whether an exact hash match was found.
        exact_match_id: The record ID of the exact match, if any.
        similar_items: Semantically similar items found.
        recommendation: One of ``"unique"``, ``"possible_duplicate"``,
            or ``"exact_duplicate"``.
        content_hash: The computed hash of the checked content.
        mismatched_model_ids: Model identities found on candidate vectors
            that differ from the embedder that produced the query vector.
            Non-empty means the scores this result rests on compared vectors
            from different spaces, so both ``similar_items`` and
            ``recommendation`` are unreliable --- in the direction of missing
            duplicates rather than inventing them.
    """

    is_exact_duplicate: bool
    exact_match_id: str | None = None
    similar_items: list[SimilarItem] = field(default_factory=list)
    recommendation: str = "unique"
    content_hash: str = ""
    mismatched_model_ids: list[str] = field(default_factory=list)


class DedupChecker:
    """Checks content uniqueness via hash matching and optional semantic similarity.

    Uses an ``AsyncDatabase`` for hash-based exact matching and an optional
    ``VectorStore`` for semantic similarity search.

    Example:
        >>> checker = DedupChecker(db=dedup_db, config=DedupConfig())
        >>> await checker.register({"content": "A question about math"}, "q-1")
        >>> result = await checker.check({"content": "A question about math"})
        >>> result.recommendation
        'exact_duplicate'
    """

    def __init__(
        self,
        db: AsyncDatabase,
        config: DedupConfig,
        vector_store: Any | None = None,
        embedding_fn: Callable[[str], Awaitable[list[float]]] | None = None,
        *,
        embedder: TextEmbedder | None = None,
    ) -> None:
        """Initialize the dedup checker.

        Args:
            db: Database for storing content hashes.
            config: Deduplication configuration.
            vector_store: Optional vector store for semantic similarity search.
                Expects a ``VectorStore``-compatible interface.
            embedding_fn: Async function that takes text and returns an
                embedding vector. Required, along with *vector_store*, when
                ``config.semantic_check`` is True --- unless *embedder* is
                given instead.
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`, in place
                of *embedding_fn*. Exact-hash matching needs neither, so
                having no source is a supported construction.

        Raises:
            ValueError: Both *embedding_fn* and *embedder* were given.
        """
        require_embedding_source(embedder, embedding_fn, allow_neither=True)

        self._db = db
        self._config = config
        self._vector_store = vector_store
        self._embedder = embedder
        self._embedding_fn = embedding_fn

    def _semantic_store(self) -> Any | None:
        """The vector store to use for a semantic pass, or ``None`` if there is none.

        Three conditions, asked in two places --- :meth:`check` before
        searching and :meth:`register` before storing --- which must agree or
        one half of the pair runs without the other. They were written out
        twice, and adding a second embedding source to both copies is what
        made that worth naming.

        It returns the store rather than a bool because the caller needs the
        store anyway. A predicate would leave every site re-reading
        ``self._vector_store`` as an ``Any | None``, which is how a guard and
        the thing it guards drift apart.
        """
        if not self._config.semantic_check or self._vector_store is None:
            return None
        if self._embedder is None and self._embedding_fn is None:
            return None
        return self._vector_store

    async def _embed(self, text: str) -> Any:
        """One embedding, from whichever source was configured.

        Returns ``Any`` rather than ``list[float]`` because the callable path
        deliberately hands back what the caller's own function returned,
        unconverted --- an ``np.ndarray`` as readily as a list. Both callers
        pass the result straight to ``np.array``.
        """
        return await embed_text(text, embedder=self._embedder, embedding_fn=self._embedding_fn)

    @property
    def config(self) -> DedupConfig:
        """The dedup configuration."""
        return self._config

    def compute_hash(self, content: dict[str, Any]) -> str:
        """Compute a deterministic content hash from configured fields.

        Fields are joined with ``|`` separator to avoid collisions between
        values like ``("a b", "c")`` and ``("a", "b c")``. Missing fields
        are treated as empty strings.

        Args:
            content: Content dictionary to hash.

        Returns:
            Hex digest of the content hash.
        """
        parts: list[str] = []
        for field_name in self._config.hash_fields:
            value = content.get(field_name, "")
            parts.append(str(value))

        combined = "|".join(parts)

        if self._config.hash_algorithm == "sha256":
            return hashlib.sha256(combined.encode()).hexdigest()
        return hashlib.md5(combined.encode()).hexdigest()

    async def check(self, content: dict[str, Any]) -> DedupResult:
        """Check content for duplicates.

        Performs an exact hash match first, then optionally checks semantic
        similarity if configured.

        Args:
            content: Content dictionary to check.

        Returns:
            DedupResult with match information and recommendation.
        """
        content_hash = self.compute_hash(content)

        # Step 1: Exact hash match
        exact_match = await self._find_exact_match(content_hash)
        if exact_match:
            return DedupResult(
                is_exact_duplicate=True,
                exact_match_id=exact_match,
                recommendation="exact_duplicate",
                content_hash=content_hash,
            )

        # Step 2: Semantic similarity (optional)
        similar_items: list[SimilarItem] = []
        mismatched_model_ids: list[str] = []
        store = self._semantic_store()
        if store is not None:
            similar_items, mismatched_model_ids = await self._find_similar(content, store)

        # Step 3: Build recommendation
        recommendation = "unique"
        if similar_items:
            recommendation = "possible_duplicate"

        return DedupResult(
            is_exact_duplicate=False,
            similar_items=similar_items,
            recommendation=recommendation,
            content_hash=content_hash,
            mismatched_model_ids=mismatched_model_ids,
        )

    async def register(
        self,
        content: dict[str, Any],
        record_id: str,
    ) -> None:
        """Register content for future duplicate lookups.

        Stores the content hash in the database and optionally the embedding
        in the vector store.

        Args:
            content: Content dictionary to register.
            record_id: The record ID to associate with this content.
        """
        content_hash = self.compute_hash(content)

        # Store hash record in database
        record = Record({"content_hash": content_hash, "record_id": record_id})
        await self._db.create(record)

        # Store embedding in vector store (if semantic check enabled)
        store = self._semantic_store()
        if store is not None:
            text = self._build_semantic_text(content)
            embedding = await self._embed(text)
            # The model that produced this vector, so a later `check` can
            # tell "these texts are unrelated" from "these vectors are in
            # different spaces". Written only when it is knowable: the
            # `embedding_fn` lane carries no identity, and the key is then
            # ABSENT rather than `None` --- the same rule the vector lane
            # follows, so that a store predating this key reads as unknown
            # instead of turning every existing vector into a warning.
            metadata: dict[str, Any] = {"text": text, "content_hash": content_hash}
            if self._embedder is not None:
                metadata[MODEL_NAME_KEY] = self._embedder.model_id
            await store.add_vectors(
                vectors=np.array([embedding], dtype=np.float32),
                ids=[record_id],
                metadata=[metadata],
            )

        logger.debug(
            "Registered content for dedup: record_id=%s, hash=%s",
            record_id,
            content_hash[:8],
        )

    async def _find_exact_match(self, content_hash: str) -> str | None:
        """Find a record with an exact hash match.

        Args:
            content_hash: Hash to search for.

        Returns:
            The matching record_id, or None if no match.
        """
        query = Query(filters=[Filter("content_hash", Operator.EQ, content_hash)])
        results = await self._db.search(query)
        if results:
            # `get_value` answers `Any`. The field is written by `register`,
            # whose `record_id` is declared `str`, so the cast states what the
            # writer already guarantees rather than converting anything --- a
            # `str(...)` here would silently stringify a value that got in by
            # another route instead of leaving it visible.
            return cast("str | None", results[0].get_value("record_id"))
        return None

    async def _find_similar(
        self, content: dict[str, Any], store: Any
    ) -> tuple[list[SimilarItem], list[str]]:
        """Find semantically similar content via vector search.

        Args:
            content: Content dictionary to check.
            store: The vector store to search, as returned by
                :meth:`_semantic_store`. Passed in rather than re-read so the
                caller's guard and this search cannot disagree about which
                store --- or whether one exists at all.

        Returns:
            The similar items above the configured threshold, and any model
            identities on the candidates that differ from this checker's own.
            The second half is returned rather than raised because every
            candidate is still the best answer available; what changes is that
            the caller can now tell the answer is untrustworthy.
        """
        text = self._build_semantic_text(content)
        embedding = await self._embed(text)
        results = await store.search(
            query_vector=np.array(embedding, dtype=np.float32),
            k=self._config.max_similar_results,
        )

        similar: list[SimilarItem] = []
        mismatched: list[str] = []
        current_model = self._embedder.model_id if self._embedder is not None else None
        for record_id, score, meta in results:
            # A candidate embedded by another model is not a weak match, it is
            # a meaningless comparison: the two vectors are in different
            # spaces. An ABSENT key is unknown, not a mismatch -- vectors
            # written before this key existed, and every vector from the
            # `embedding_fn` lane, must not all read as stale.
            stored_model = meta.get(MODEL_NAME_KEY) if meta else None
            if (
                current_model is not None
                and stored_model is not None
                and stored_model != current_model
                and stored_model not in mismatched
            ):
                mismatched.append(str(stored_model))
            if score >= self._config.similarity_threshold:
                similar.append(
                    SimilarItem(
                        record_id=record_id,
                        score=score,
                        matched_text=meta.get("text", "") if meta else "",
                    )
                )
        if mismatched:
            logger.warning(
                "Semantic dedup compared vectors from different models: stored %s, "
                "querying with %s. Similarity scores across an embedding-model change "
                "are meaningless, and this check fails OPEN -- duplicates are admitted "
                "as unique. Re-embed the store, or treat `mismatched_model_ids` on the "
                "result as an error.",
                ", ".join(sorted(mismatched)),
                current_model,
            )
        return similar, sorted(mismatched)

    def _build_semantic_text(self, content: dict[str, Any]) -> str:
        """Build text for semantic embedding from configured fields.

        Args:
            content: Content dictionary.

        Returns:
            Concatenated text from semantic (or hash) fields.
        """
        semantic_fields = self._config.semantic_fields or self._config.hash_fields
        text_parts = [str(content.get(f, "")) for f in semantic_fields]
        return " ".join(text_parts)

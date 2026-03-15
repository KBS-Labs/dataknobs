"""Caching embedding provider that wraps any AsyncLLMProvider.

Embeddings are deterministic: same (model, text) always produces the same
vector. This module provides a wrapper that caches ``embed()`` results
persistently, passing all other methods through to the inner provider.

Cache backends:
    - ``MemoryEmbeddingCache``: In-memory dict — for testing.
    - ``SqliteEmbeddingCache``: SQLite with WAL mode — for persistent caching.
      Requires ``aiosqlite`` (install via ``pip install 'dataknobs-llm[sqlite-cache]'``).
"""

import asyncio
import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

from ..base import (
    AsyncLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    ModelCapability,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache abstraction
# ---------------------------------------------------------------------------


def _cache_key(model: str, text: str) -> str:
    """Compute a deterministic cache key for (model, text).

    Uses SHA-256 with a null-byte separator so that ``("model", "text")``
    never collides with ``("mode", "ltext")``.
    """
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class EmbeddingCache(ABC):
    """Cache for embedding vectors, keyed by (model, text)."""

    @abstractmethod
    async def get(self, model: str, text: str) -> list[float] | None:
        """Retrieve a cached vector, or ``None`` on miss."""

    @abstractmethod
    async def put(self, model: str, text: str, vector: list[float]) -> None:
        """Store a vector in the cache."""

    @abstractmethod
    async def get_batch(
        self, model: str, texts: list[str]
    ) -> list[list[float] | None]:
        """Retrieve cached vectors for a batch of texts.

        Returns a list parallel to *texts*: each element is either the
        cached vector or ``None`` on a miss.
        """

    @abstractmethod
    async def put_batch(
        self, model: str, texts: list[str], vectors: list[list[float]]
    ) -> None:
        """Store a batch of vectors in the cache."""

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the cache backend (create tables, open files, etc.)."""

    @abstractmethod
    async def close(self) -> None:
        """Release cache resources."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all cached entries."""

    @abstractmethod
    async def count(self) -> int:
        """Return the number of cached entries."""


# ---------------------------------------------------------------------------
# In-memory backend (testing)
# ---------------------------------------------------------------------------


class MemoryEmbeddingCache(EmbeddingCache):
    """In-memory cache backend for testing."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    async def get(self, model: str, text: str) -> list[float] | None:
        return self._store.get(_cache_key(model, text))

    async def put(self, model: str, text: str, vector: list[float]) -> None:
        self._store[_cache_key(model, text)] = vector

    async def get_batch(
        self, model: str, texts: list[str]
    ) -> list[list[float] | None]:
        return [self._store.get(_cache_key(model, t)) for t in texts]

    async def put_batch(
        self, model: str, texts: list[str], vectors: list[list[float]]
    ) -> None:
        for text, vector in zip(texts, vectors):
            self._store[_cache_key(model, text)] = vector

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def clear(self) -> None:
        self._store.clear()

    async def count(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# SQLite backend (persistent)
# ---------------------------------------------------------------------------


def _vector_to_blob(vector: list[float]) -> bytes:
    """Pack a float vector into a compact binary blob (float32)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes) -> list[float]:
    """Unpack a binary blob into a float vector."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


class SqliteEmbeddingCache(EmbeddingCache):
    """SQLite-backed persistent cache with WAL mode.

    Vectors are stored as compact float32 binary blobs for efficiency
    (768 floats = 3 KB vs ~6 KB for JSON text).

    Requires ``aiosqlite``. Install via::

        pip install 'dataknobs-llm[sqlite-cache]'

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: Any = None  # aiosqlite.Connection

    async def initialize(self) -> None:
        """Open the database and create the embeddings table if needed."""
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "SqliteEmbeddingCache requires aiosqlite. "
                "Install it with: pip install 'dataknobs-llm[sqlite-cache]'"
            ) from None

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def get(self, model: str, text: str) -> list[float] | None:
        key = _cache_key(model, text)
        cursor = await self._conn.execute(
            "SELECT vector FROM embeddings WHERE cache_key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _blob_to_vector(row[0])

    async def put(self, model: str, text: str, vector: list[float]) -> None:
        key = _cache_key(model, text)
        blob = _vector_to_blob(vector)
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (cache_key, model, vector)
            VALUES (?, ?, ?)
            """,
            (key, model, blob),
        )
        await self._conn.commit()

    async def get_batch(
        self, model: str, texts: list[str]
    ) -> list[list[float] | None]:
        keys = [_cache_key(model, t) for t in texts]
        placeholders = ",".join("?" * len(keys))
        cursor = await self._conn.execute(
            f"SELECT cache_key, vector FROM embeddings "
            f"WHERE cache_key IN ({placeholders})",
            keys,
        )
        rows = {row[0]: _blob_to_vector(row[1]) for row in await cursor.fetchall()}
        return [rows.get(k) for k in keys]

    async def put_batch(
        self, model: str, texts: list[str], vectors: list[list[float]]
    ) -> None:
        rows = [
            (_cache_key(model, t), model, _vector_to_blob(v))
            for t, v in zip(texts, vectors)
        ]
        await self._conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings (cache_key, model, vector)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        await self._conn.commit()

    async def clear(self) -> None:
        await self._conn.execute("DELETE FROM embeddings")
        await self._conn.commit()

    async def count(self) -> int:
        cursor = await self._conn.execute("SELECT COUNT(*) FROM embeddings")
        row = await cursor.fetchone()
        return row[0]


# ---------------------------------------------------------------------------
# CachingEmbedProvider
# ---------------------------------------------------------------------------


class CachingEmbedProvider(AsyncLLMProvider):
    """Provider wrapper that caches ``embed()`` results persistently.

    Embeddings are deterministic: same (model, text) produces the same
    vector. This wrapper caches them once and reuses them across scenarios
    and runs. ``complete()``, ``stream_complete()``, and ``function_call()``
    pass through to the inner provider unchanged.

    Args:
        inner: The real provider to delegate to.
        cache: The cache backend for storing embeddings.

    Example::

        from dataknobs_llm import EchoProvider
        from dataknobs_llm.llm.providers.caching import (
            CachingEmbedProvider,
            MemoryEmbeddingCache,
        )

        inner = EchoProvider({"provider": "echo", "model": "test"})
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        vec1 = await provider.embed("hello")   # cache miss → inner
        vec2 = await provider.embed("hello")   # cache hit → no inner call
    """

    def __init__(self, inner: AsyncLLMProvider, cache: EmbeddingCache) -> None:
        # Skip LLMProvider.__init__ — config is delegated to inner.
        # Initialize attributes that the base class hierarchy references
        # so that inherited methods (render_and_complete, close, etc.)
        # don't hit AttributeError.
        self._inner = inner
        self._cache = cache
        self._is_initialized = False
        self._is_closing = False
        self._in_flight: set[asyncio.Task[Any]] = set()
        self._client = None
        self.prompt_builder = None

    # -- Config / capability forwarding ------------------------------------

    @property
    def config(self) -> LLMConfig:  # type: ignore[override]
        """Forward config from the inner provider."""
        return self._inner.config

    @config.setter
    def config(self, value: LLMConfig) -> None:
        """Allow config assignment (required by LLMProvider.__init__)."""
        # No-op — config is always delegated to inner.

    async def validate_model(self) -> bool:
        """Delegate model validation to inner provider."""
        return await self._inner.validate_model()

    def _detect_capabilities(self) -> List[ModelCapability]:
        """Delegate capability detection to inner provider."""
        return self._inner._detect_capabilities()

    def get_capabilities(self) -> List[ModelCapability]:
        """Delegate capabilities to inner provider."""
        return self._inner.get_capabilities()

    # -- Lifecycle ---------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize both the inner provider and the cache."""
        await self._inner.initialize()
        await self._cache.initialize()
        self._is_initialized = True

    async def close(self) -> None:
        """Close both the inner provider and the cache."""
        if self._is_closing:
            return
        self._is_closing = True
        try:
            await self._inner.close()
        except Exception:
            logger.exception("Error closing inner provider")
        try:
            await self._cache.close()
        except Exception:
            logger.exception("Error closing embedding cache")
        self._is_initialized = False
        self._is_closing = False

    # -- Passthrough methods -----------------------------------------------

    async def complete(
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Delegate to inner provider."""
        return await self._inner.complete(
            messages, config_overrides=config_overrides, tools=tools, **kwargs
        )

    async def stream_complete(  # type: ignore[override]
        self,
        messages: Union[str, List[LLMMessage]],
        config_overrides: Dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to inner provider (async generator passthrough)."""
        async for chunk in self._inner.stream_complete(
            messages, config_overrides=config_overrides, tools=tools, **kwargs
        ):
            yield chunk

    async def function_call(
        self,
        messages: List[LLMMessage],
        functions: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Delegate to inner provider."""
        return await self._inner.function_call(messages, functions, **kwargs)

    # -- Cached embed ------------------------------------------------------

    async def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[List[float], List[List[float]]]:
        """Return cached embeddings, delegating to inner on cache miss.

        Handles both single-text and batch forms. For batches, only the
        uncached texts are sent to the inner provider. The inner provider
        always receives a ``list[str]`` and always returns
        ``list[list[float]]``.
        """
        model = self.config.model
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)

        # Batch cache lookup
        cached = await self._cache.get_batch(model, text_list)
        results: list[list[float]] = [[] for _ in text_list]
        misses: list[tuple[int, str]] = []

        for i, (text, cached_vec) in enumerate(zip(text_list, cached)):
            if cached_vec is not None:
                results[i] = cached_vec
            else:
                misses.append((i, text))

        # Delegate misses to inner provider
        if misses:
            miss_texts = [t for _, t in misses]
            miss_result = await self._inner.embed(miss_texts, **kwargs)

            # miss_texts is always list[str], so inner returns
            # list[list[float]]. Cast for type safety.
            miss_vectors: list[list[float]] = miss_result  # type: ignore[assignment]

            for (idx, _text), vector in zip(misses, miss_vectors):
                results[idx] = vector

            # Store all misses in cache
            await self._cache.put_batch(
                model,
                miss_texts,
                miss_vectors,
            )

        logger.debug(
            "Embedding cache: %d hits, %d misses (model=%s)",
            len(text_list) - len(misses),
            len(misses),
            model,
        )

        return results[0] if single else results


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


async def create_caching_provider(
    inner: AsyncLLMProvider,
    cache_path: str | Path | None = None,
    *,
    cache_backend: str = "sqlite",
) -> CachingEmbedProvider:
    """Create and initialize a ``CachingEmbedProvider``.

    Args:
        inner: The provider to wrap.
        cache_path: Path for the SQLite cache file. Required when
            *cache_backend* is ``"sqlite"``. Ignored for ``"memory"``.
        cache_backend: ``"sqlite"`` (default) or ``"memory"``.
            The ``"sqlite"`` backend requires ``aiosqlite``
            (install via ``pip install 'dataknobs-llm[sqlite-cache]'``).

    Returns:
        An initialized ``CachingEmbedProvider``.

    Raises:
        ValueError: If *cache_backend* is ``"sqlite"`` and no *cache_path*
            is provided, or if *cache_backend* is unknown.
    """
    if cache_backend == "memory":
        cache: EmbeddingCache = MemoryEmbeddingCache()
    elif cache_backend == "sqlite":
        if cache_path is None:
            raise ValueError(
                "cache_path is required for the 'sqlite' cache backend"
            )
        cache = SqliteEmbeddingCache(cache_path)
    else:
        raise ValueError(
            f"Unknown cache backend: {cache_backend!r}. "
            f"Use 'sqlite' or 'memory'."
        )

    provider = CachingEmbedProvider(inner, cache)
    await provider.initialize()
    return provider

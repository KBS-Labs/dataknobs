"""Knowledge ingestion manager for coordinating file storage to vector storage.

This module provides the KnowledgeIngestionManager which coordinates loading
files from a KnowledgeResourceBackend into a RAGKnowledgeBase.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from .storage import IngestionStatus, InvalidVersionError

if TYPE_CHECKING:
    from dataknobs_common.events import EventBus
    from dataknobs_xization.ingestion import KnowledgeBaseConfig

    from .rag import RAGKnowledgeBase
    from .storage import KnowledgeFile, KnowledgeResourceBackend

logger = logging.getLogger(__name__)


class IngestSwapMode(Enum):
    """How a re-ingest replaces the destination's existing chunks.

    The swap policy is a property of the shared apply-core
    (:meth:`KnowledgeIngestionManager._apply_file_ops`), *not* of the
    entry point — full-domain (:meth:`~KnowledgeIngestionManager.ingest`)
    and per-file (:meth:`~KnowledgeIngestionManager.ingest_changes`)
    re-ingests honor the same modes through the same code path.

    Members ship incrementally with the same value space:

    - ``CLEAR_FIRST`` — delete the (scoped) existing chunks, then
      ingest. The legacy ``clear_existing=True`` behavior; carries a
      temporal zero-results window for concurrent reads.
    - ``APPEND`` — ingest without a preceding full-domain clear (the
      legacy ``clear_existing=False`` behavior). Per-file deletions
      requested explicitly still happen.
    - ``TOMBSTONE`` — crash-safe re-ingest: mark the existing
      (scoped) chunks ``_stale`` (reads stop seeing them via the
      :class:`RAGKnowledgeBase` read chokepoint), then ingest the new
      generation under **distinct, generation-keyed chunk ids** so it
      never overwrites the tombstoned old rows — both generations
      coexist physically. On a clean commit the old generation is
      physically retired; on a raised error or partial-error ingest
      the rollback drops exactly the new generation by its token,
      restores the modified files' old generation to visibility, and
      unconditionally purges files deleted at the source. The old
      generation is never overwritten or deleted until the new one
      commits cleanly — so unlike ``CLEAR_FIRST`` (delete-then-insert,
      where a failed insert leaves nothing) a crash, a raised error,
      or a racing same-domain re-ingest always leaves a fully
      restorable previous generation. A crash mid-swap leaves the
      domain in :attr:`IngestionStatus.SWAPPING`, auto-reconciled by
      the next ingest (or :meth:`KnowledgeIngestionManager.reconcile`).
      This is the documented production default for multi-replica
      re-ingest.

      A transient in-swap window remains: between marking the old
      generation stale and the new one being committed, a concurrent
      reader sees the new generation only (and, briefly, nothing while
      the new one is still embedding). Closing *that* window requires
      a generation pointer-flip (``SHADOW_GENERATION``); ``TOMBSTONE``
      deliberately trades it for a far simpler, crash-safe mechanism.

    ``SHADOW_GENERATION`` (a pointer-flip variant with no in-swap read
    window) is a future, stronger-atomicity mode; it is intentionally
    not a member until it is implemented, so the enum never carries a
    value the apply-core cannot honor.
    """

    CLEAR_FIRST = "clear_first"
    APPEND = "append"
    TOMBSTONE = "tombstone"


@dataclass
class IngestionResult:
    """Result of an ingestion operation.

    Contains statistics about files processed, chunks created, and any errors
    encountered during ingestion. ``files_deleted`` is the count of
    source files whose chunks were removed because the file no longer
    exists at the source (populated by
    :meth:`KnowledgeIngestionManager.ingest_changes`; ``0`` for a
    full :meth:`~KnowledgeIngestionManager.ingest`).

    Invariants:
        ``completed_at`` is populated on every terminal state — call
        :meth:`finish` at every return site (or in a ``finally`` block)
        to enforce this. ``IngestionResult`` is the lower-level sibling
        of :class:`EnsureIngestionResult` and shares this invariant.
    """

    domain_id: str
    files_processed: int = 0
    chunks_created: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def finish(self) -> IngestionResult:
        """Stamp ``completed_at`` on the first call; idempotent thereafter.

        Returns ``self`` so it can be chained at return sites or used
        from a ``finally`` block.
        """
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)
        return self

    @property
    def success(self) -> bool:
        """Check if ingestion completed without errors."""
        return len(self.errors) == 0

    @property
    def duration_seconds(self) -> float | None:
        """Get duration in seconds if completed."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain_id": self.domain_id,
            "files_processed": self.files_processed,
            "chunks_created": self.chunks_created,
            "files_skipped": self.files_skipped,
            "files_deleted": self.files_deleted,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
        }


class KnowledgeIngestionManager:
    """Coordinates loading files from storage backend into RAG knowledge base.

    This manager bridges :class:`KnowledgeResourceBackend` (file storage)
    and :class:`RAGKnowledgeBase` (vector storage). It delegates the
    actual document processing (pattern matching, chunking, streaming
    JSON, exclude patterns, per-pattern metadata) to
    :meth:`RAGKnowledgeBase.ingest_from_backend`, and owns:

    - Ingestion status tracking on the source backend
    - Event-bus publishing for hot-reload consumers
    - Version-based skip (`ingest_if_changed`)

    When ``config`` is omitted, :meth:`ingest_from_backend` reads any
    ``_metadata/knowledge_base.(yaml|yml|json)`` in the backend's domain
    namespace and falls back to defaults. This is a strict superset of
    the pre-unification behavior, which silently ignored patterns and
    excludes.

    Example:
        ```python
        from dataknobs_bots.knowledge import (
            KnowledgeIngestionManager,
            create_knowledge_backend,
            RAGKnowledgeBase,
        )

        # Create components
        file_backend = create_knowledge_backend("file", {"path": "./data/kb"})
        rag_kb = await RAGKnowledgeBase.from_config(rag_config)

        # Create manager
        manager = KnowledgeIngestionManager(
            source=file_backend,
            destination=rag_kb,
        )

        # Ingest all files for a domain
        result = await manager.ingest("my-domain")
        print(f"Ingested {result.chunks_created} chunks from {result.files_processed} files")
        ```
    """

    def __init__(
        self,
        source: KnowledgeResourceBackend,
        destination: RAGKnowledgeBase,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the ingestion manager.

        Args:
            source: Backend storing knowledge files
            destination: RAG knowledge base for vector storage
            event_bus: Optional event bus for publishing ingestion events
        """
        self._source = source
        self._destination = destination
        self._event_bus = event_bus

    @staticmethod
    def _resolve_swap_mode(
        clear_existing: bool | None,
        swap_mode: IngestSwapMode | None,
    ) -> IngestSwapMode:
        """Resolve the effective swap mode from the two knobs.

        ``swap_mode`` is authoritative when set. The legacy
        ``clear_existing`` is honored (with a ``DeprecationWarning``)
        only when ``swap_mode`` is not given. With neither set the
        default is ``CLEAR_FIRST`` — identical to the pre-deprecation
        ``clear_existing=True`` default, so existing callers are
        unaffected.
        """
        if clear_existing is not None:
            warnings.warn(
                "KnowledgeIngestionManager.ingest(clear_existing=) is "
                "deprecated; pass swap_mode= (IngestSwapMode) instead. "
                "clear_existing=True maps to CLEAR_FIRST, False to "
                "APPEND. For zero-downtime re-ingest use "
                "IngestSwapMode.TOMBSTONE.",
                DeprecationWarning,
                stacklevel=3,
            )
            if swap_mode is None:
                return (
                    IngestSwapMode.CLEAR_FIRST
                    if clear_existing
                    else IngestSwapMode.APPEND
                )
        if swap_mode is not None:
            return swap_mode
        return IngestSwapMode.CLEAR_FIRST

    async def ingest(
        self,
        domain_id: str,
        clear_existing: bool | None = None,
        *,
        swap_mode: IngestSwapMode | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        config: KnowledgeBaseConfig | None = None,
    ) -> IngestionResult:
        """Ingest all documents from a domain into the knowledge base.

        Delegates to :meth:`RAGKnowledgeBase.ingest_from_backend` for
        the actual processing — pattern-based chunking, exclude
        patterns, per-pattern metadata, streaming JSON — and wraps the
        result in an :class:`IngestionResult` for backward compatibility.

        Args:
            domain_id: Domain to ingest
            clear_existing: **Deprecated** — use ``swap_mode``.
                ``True`` maps to :attr:`IngestSwapMode.CLEAR_FIRST`,
                ``False`` to :attr:`IngestSwapMode.APPEND`. ``None``
                (default) defers entirely to ``swap_mode``.
            swap_mode: How the re-ingest replaces existing chunks.
                Defaults to :attr:`IngestSwapMode.CLEAR_FIRST` (the
                historical behavior). :attr:`IngestSwapMode.TOMBSTONE`
                gives a zero-downtime swap and is the recommended
                production default for multi-replica deployments.
            progress_callback: Optional callback invoked as
                ``(file_path, num_chunks)`` after each ingested document
            config: Optional :class:`KnowledgeBaseConfig` overriding any
                backend-hosted ``_metadata/knowledge_base.(yaml|yml|json)``

        Returns:
            :class:`IngestionResult` with aggregate statistics
        """
        effective_mode = self._resolve_swap_mode(clear_existing, swap_mode)
        result = IngestionResult(domain_id=domain_id)

        try:
            # Recover a domain stuck in SWAPPING from a crashed prior
            # swap *before* the INGESTING status write clears the
            # persisted token. No-op when not interrupted.
            await self._reconcile_interrupted_swap(domain_id)
            await self._source.set_ingestion_status(
                domain_id, IngestionStatus.INGESTING
            )
            logger.info("Starting ingestion for domain: %s", domain_id)

            await self._apply_file_ops(
                domain_id,
                upsert_filter=None,
                delete_paths=None,
                purely_deleted_paths=None,
                swap_mode=effective_mode,
                config=config,
                progress_callback=progress_callback,
                result=result,
            )
            await self._finalize(domain_id, result)

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", domain_id, e)
            await self._source.set_ingestion_status(domain_id, "error", str(e))
            raise
        finally:
            result.finish()

        return result

    async def _apply_file_ops(
        self,
        domain_id: str,
        *,
        upsert_filter: Callable[[KnowledgeFile], bool] | None,
        delete_paths: list[str] | None,
        purely_deleted_paths: list[str] | None = None,
        swap_mode: IngestSwapMode,
        config: KnowledgeBaseConfig | None,
        progress_callback: Callable[[str, int], None] | None,
        result: IngestionResult,
    ) -> None:
        """The single place a re-ingest mutates the destination.

        Used by **both** :meth:`ingest` (full domain) and
        :meth:`ingest_changes` (per-file delta) so swap semantics never
        diverge between the two entry points.

        ``swap_mode`` governs how the existing (scoped) chunks are
        replaced:

        - ``CLEAR_FIRST`` — scoped
          ``clear(filter={"domain_id": ...})`` before the upsert (the
          legacy ``clear_existing=True`` behavior). The filter scopes
          the wipe to this domain so a shared multi-tenant store keeps
          every other tenant's chunks. Carries a temporal
          zero-results window for concurrent reads.
        - ``APPEND`` — no full-domain clear.
        - ``TOMBSTONE`` — crash-safe re-ingest: mark the old (scoped)
          chunks ``_stale`` (reads stop seeing them via the
          :class:`RAGKnowledgeBase` read chokepoint), ingest the new
          generation under distinct generation-keyed ids so it never
          overwrites the old rows, then physically retire the old
          generation only on a clean commit — on failure or partial
          error the rollback drops the new generation by its token and
          restores the old one. The old generation is never
          overwritten or deleted before the new one commits, so a
          crash or a racing re-ingest always leaves a fully restorable
          previous generation (unlike the ``CLEAR_FIRST``
          delete-then-insert). A transient in-swap read window
          remains — see :attr:`IngestSwapMode.TOMBSTONE` and
          :meth:`_apply_tombstone`.

        ``delete_paths`` (when non-empty) additionally purges every
        chunk whose ``source_path`` matches — used by
        :meth:`ingest_changes` to drop chunks for *deleted and
        modified* files before re-embedding. ``source_path`` is the
        backend-stable source-relative path
        (``DocumentFileRef.path``); ``source`` is a display path and
        is not used as the delete key. Under ``TOMBSTONE`` the same
        paths define the swap scope (tombstoned, not eagerly deleted).

        ``purely_deleted_paths`` (internal; ``ingest_changes`` only) is
        the subset of ``delete_paths`` whose source files were
        *deleted* (not modified) — they are not re-embedded. Under
        ``TOMBSTONE`` this lets :meth:`_apply_tombstone` distinguish
        "restore on rollback" (modified files: the old generation must
        come back) from "purge unconditionally" (deleted files: the
        old generation must stay gone even on rollback, never
        resurrected). :meth:`ingest` passes ``None`` (a full re-ingest
        has no per-file deleted scope).

        The upsert re-embeds the source files selected by
        ``upsert_filter`` (``None`` = every file) via
        :meth:`RAGKnowledgeBase.ingest_from_backend`. ``result`` is
        populated in place from the returned statistics;
        ``files_deleted`` is owned by the caller, which knows the
        delete semantics for its entry point.
        """
        if swap_mode is IngestSwapMode.TOMBSTONE:
            await self._apply_tombstone(
                domain_id,
                upsert_filter=upsert_filter,
                delete_paths=delete_paths,
                purely_deleted_paths=purely_deleted_paths,
                config=config,
                progress_callback=progress_callback,
                result=result,
            )
            return

        if swap_mode is IngestSwapMode.CLEAR_FIRST:
            await self._destination.clear(filter={"domain_id": domain_id})
            logger.debug(
                "Cleared existing vectors for domain: %s", domain_id
            )

        for path in delete_paths or ():
            await self._destination.clear(
                filter={"domain_id": domain_id, "source_path": path}
            )
        if delete_paths:
            logger.debug(
                "Purged chunks for %d source path(s) in domain: %s",
                len(delete_paths),
                domain_id,
            )

        await self._upsert(
            domain_id,
            upsert_filter=upsert_filter,
            config=config,
            progress_callback=progress_callback,
            result=result,
        )

    async def _upsert(
        self,
        domain_id: str,
        *,
        upsert_filter: Callable[[KnowledgeFile], bool] | None,
        config: KnowledgeBaseConfig | None,
        progress_callback: Callable[[str, int], None] | None,
        result: IngestionResult,
        generation: str | None = None,
    ) -> None:
        """Re-embed the selected source files into the destination.

        The one place :meth:`RAGKnowledgeBase.ingest_from_backend` is
        invoked and its statistics are mapped onto ``result`` — shared
        by every swap mode so the embed + stats handling never drifts
        between them.

        ``generation`` (TOMBSTONE only) is threaded into
        ``extra_metadata`` as ``_generation``. When set,
        :meth:`RAGKnowledgeBase._embed_and_store_chunks` folds it into
        the chunk-id prefix so the new generation gets *distinct* ids
        instead of upserting (overwriting) the tombstoned old rows in
        place, and stamps it on every new chunk's metadata so the
        rollback can target exactly this swap's rows. When ``None``
        (APPEND / CLEAR_FIRST) the id derivation is byte-for-byte
        unchanged.
        """
        extra_metadata: dict[str, Any] = {"domain_id": domain_id}
        if generation is not None:
            extra_metadata["_generation"] = generation
        stats = await self._destination.ingest_from_backend(
            self._source,
            domain_id,
            config=config,
            progress_callback=progress_callback,
            extra_metadata=extra_metadata,
            file_filter=upsert_filter,
        )

        result.files_processed = int(stats.get("total_files", 0))
        result.chunks_created = int(stats.get("total_chunks", 0))
        result.files_skipped = int(stats.get("files_skipped", 0))
        result.errors = list(stats.get("errors", []))

    async def _apply_tombstone(
        self,
        domain_id: str,
        *,
        upsert_filter: Callable[[KnowledgeFile], bool] | None,
        delete_paths: list[str] | None,
        purely_deleted_paths: list[str] | None = None,
        config: KnowledgeBaseConfig | None,
        progress_callback: Callable[[str, int], None] | None,
        result: IngestionResult,
    ) -> None:
        """Crash-safe, generation-distinct re-ingest for a scope.

        Each swap gets a fresh ``generation = uuid4().hex`` threaded
        into the new chunks' ids (via :meth:`_upsert` →
        ``extra_metadata``). Because the new generation has *distinct*
        ids it never overwrites the tombstoned old rows in place — both
        generations coexist physically until the swap commits cleanly.
        This is what makes the swap genuinely crash-safe: the old
        generation is fully restorable until the new one is proven
        good.

        Sequence:

        1. ``update_metadata_where(tombstone_scope, {"_stale": True})``
           — mark the existing generation. The read chokepoint
           (:meth:`RAGKnowledgeBase._resolve_read_filter`) hides
           ``_stale``-true chunks, so reads stop seeing the old
           generation *without anything being deleted*.
        2. Status → :attr:`IngestionStatus.SWAPPING`.
        3. Ingest the new generation (distinct ids, no ``_stale`` key,
           stamped ``_generation``; immediately read-visible).
        4. On a clean commit, physically delete the tombstoned old
           generation. On a raised failure **or** a partial-error
           ingest, :meth:`_rollback_swap` drops the (possibly partial)
           new generation by its ``_generation`` token, un-tombstones
           the modified files' old generation, and unconditionally
           purges files deleted at the source — so a crash never leaks
           partial chunks and never resurrects deleted files.

        The old generation is never *removed* unless the new one
        committed cleanly. A crash mid-swap leaves status=SWAPPING with
        the old generation tombstoned-but-restorable (recovered by the
        next ingest's reconciliation). A transient in-swap read window
        remains (old hidden from step 1, new not visible until step 3
        completes); closing it needs a generation pointer-flip
        (``SHADOW_GENERATION``), which ``TOMBSTONE`` deliberately trades
        away for this simpler crash-safe mechanism.

        Scopes (only the per-file delta distinguishes them):

        - ``tombstone_scope`` — every old chunk to hide then retire
          (``{"domain_id": ...}`` for a full re-ingest; the
          changed+deleted ``source_path`` list for a delta).
        - ``modified_scope`` — old chunks of files being re-embedded;
          restored to visibility on rollback.
        - ``deleted_scope`` — old chunks of files *deleted at the
          source*; purged unconditionally, never resurrected.
        """
        generation = uuid.uuid4().hex
        base: dict[str, Any] = {"domain_id": domain_id}
        deleted_set = set(purely_deleted_paths or ())

        if delete_paths:
            tombstone_scope: dict[str, Any] = {
                **base,
                "source_path": list(delete_paths),
            }
            modified_paths = sorted(set(delete_paths) - deleted_set)
            modified_scope = (
                {**base, "source_path": modified_paths}
                if modified_paths
                else None
            )
            deleted_scope = (
                {**base, "source_path": sorted(deleted_set)}
                if deleted_set
                else None
            )
        else:
            # Full-domain swap: the whole domain is re-embedded, so the
            # entire old generation is the modified scope; nothing is
            # purely deleted.
            tombstone_scope = dict(base)
            modified_scope = dict(base)
            deleted_scope = None

        await self._destination.update_metadata_where(
            tombstone_scope, {"_stale": True}
        )
        await self._source.set_ingestion_status(
            domain_id, IngestionStatus.SWAPPING, generation=generation
        )
        logger.debug(
            "Tombstoned existing chunks for domain %s "
            "(scope=%s, gen=%s)",
            domain_id,
            tombstone_scope,
            generation,
        )

        try:
            await self._upsert(
                domain_id,
                upsert_filter=upsert_filter,
                config=config,
                progress_callback=progress_callback,
                result=result,
                generation=generation,
            )
        except Exception:
            await self._rollback_swap(
                domain_id, generation, modified_scope, deleted_scope
            )
            logger.warning(
                "Tombstone swap failed for domain %s; dropped the "
                "partial new generation and restored the previous one",
                domain_id,
            )
            raise

        if result.success:
            # Clean commit — physically retire the tombstoned old
            # generation. The new generation has distinct ids and no
            # ``_stale`` key, so it is untouched.
            await self._destination.clear(
                filter={**tombstone_scope, "_stale": True}
            )
            logger.debug(
                "Retired tombstoned chunks for domain %s (gen=%s)",
                domain_id,
                generation,
            )
        else:
            # Partial-error ingest: the new generation is incomplete
            # and must not be kept. Identical undo to a raised failure
            # — the only difference is the caller does not re-raise;
            # _finalize reports "error".
            await self._rollback_swap(
                domain_id, generation, modified_scope, deleted_scope
            )
            logger.warning(
                "Tombstone swap for domain %s completed with %d "
                "error(s); dropped the incomplete new generation and "
                "restored the previous one",
                domain_id,
                len(result.errors),
            )

    async def _rollback_swap(
        self,
        domain_id: str,
        generation: str,
        modified_scope: dict[str, Any] | None,
        deleted_scope: dict[str, Any] | None,
    ) -> None:
        """Undo an uncommitted TOMBSTONE swap.

        Shared by the raised-exception and partial-error paths — the
        undo is byte-identical; only whether the caller re-raises
        differs (the caller owns that). Three precise,
        order-significant steps:

        1. ``clear({"domain_id", "_generation": generation})`` —
           delete exactly the rows *this* swap wrote. The new
           generation carries ``_generation``; the old one never did,
           and ``generation`` is a fresh uuid, so no other generation
           can match. Precise and scope-independent regardless of how
           many files committed before the failure.
        2. ``update_metadata_where(modified_scope | _stale,
           {"_stale": False})`` — un-tombstone the modified files' old
           generation so it is visible again (the swap degraded to
           "kept the previous generation", not data loss).
        3. ``clear(deleted_scope | _stale)`` — physically purge the old
           generation of files deleted at the source. The rollback
           must not resurrect them; their absence is the intended end
           state.
        """
        await self._destination.clear(
            filter={"domain_id": domain_id, "_generation": generation}
        )
        if modified_scope is not None:
            await self._destination.update_metadata_where(
                {**modified_scope, "_stale": True}, {"_stale": False}
            )
        if deleted_scope is not None:
            await self._destination.clear(
                filter={**deleted_scope, "_stale": True}
            )

    async def _reconcile_interrupted_swap(self, domain_id: str) -> bool:
        """Recover a domain stuck in SWAPPING from a crashed swap.

        A process crash between :meth:`_upsert` and the commit (a
        SIGKILL bypasses Python ``except``/``finally``, so
        :meth:`_rollback_swap` never ran) leaves the old generation
        tombstoned, orphan new-generation chunks written, and status
        :attr:`IngestionStatus.SWAPPING` with the in-flight token
        persisted on :attr:`KnowledgeBaseInfo.generation`. This undoes
        exactly that, idempotently:

        1. Un-tombstone the whole domain's old generation
           (``update_metadata_where({"domain_id", "_stale": True},
           {"_stale": False})``) — it was never deleted, so the
           previous generation is restored to visibility. Domain-wide
           because a crashed full-domain swap tombstones the whole
           domain; a per-file follow-up would otherwise leave
           unrelated files hidden forever.
        2. Drop exactly the crashed swap's orphans by its persisted
           token (``clear({"domain_id", "_generation": token})``). A
           ``None`` token (crash before the SWAPPING status carried
           one) means there is nothing precise to delete — step 1
           already restored service.
        3. Set status READY: the previous generation is now the live,
           valid state. A following ingest immediately overwrites this
           with INGESTING; the public :meth:`reconcile` relies on it as
           the terminal state.

        Returns ``True`` if a reconcile was performed, ``False`` if the
        domain was not in an interrupted-swap state (a safe no-op).
        """
        info = await self._source.get_info(domain_id)
        if (
            info is None
            or info.ingestion_status is not IngestionStatus.SWAPPING
        ):
            return False

        token = info.generation
        await self._destination.update_metadata_where(
            {"domain_id": domain_id, "_stale": True}, {"_stale": False}
        )
        if token:
            await self._destination.clear(
                filter={"domain_id": domain_id, "_generation": token}
            )
        await self._source.set_ingestion_status(
            domain_id, IngestionStatus.READY
        )
        logger.warning(
            "Reconciled an interrupted TOMBSTONE swap for domain %s "
            "(restored previous generation; dropped orphan gen=%s)",
            domain_id,
            token,
        )
        return True

    async def reconcile(self, domain_id: str) -> bool:
        """Recover a domain left in SWAPPING by an interrupted swap.

        The real recovery mechanism for the case where no further
        ingest is scheduled (auto-reconciliation otherwise runs at the
        head of :meth:`ingest` / :meth:`ingest_changes`). Restores the
        previous generation and drops the crashed swap's orphan chunks
        by its persisted token. Idempotent — safe to call when nothing
        is interrupted (a no-op returning ``False``).

        Returns ``True`` if a reconcile was performed.
        """
        return await self._reconcile_interrupted_swap(domain_id)

    async def _finalize(
        self, domain_id: str, result: IngestionResult
    ) -> str:
        """Set the terminal ingestion status, log, and publish.

        Shared tail of :meth:`ingest` and :meth:`ingest_changes` — the
        success/error decision, backend status write, completion log,
        and event publish are one behavior, not duplicated per entry
        point. Returns the resolved status string.
        """
        status = "ready" if result.success else "error"
        error_msg = str(result.errors) if result.errors else None
        await self._source.set_ingestion_status(
            domain_id, status, error_msg
        )

        logger.info(
            "Ingestion completed for %s: %d files, %d chunks, "
            "%d deleted, %d errors",
            domain_id,
            result.files_processed,
            result.chunks_created,
            result.files_deleted,
            len(result.errors),
        )

        if self._event_bus:
            await self._publish_ingestion_event(domain_id, result, status)
        return status

    async def _publish_ingestion_event(
        self,
        domain_id: str,
        result: IngestionResult,
        status: str,
    ) -> None:
        """Publish an ingestion completed event."""
        if not self._event_bus:
            return

        from dataknobs_common.events import Event, EventType

        await self._event_bus.publish(
            "knowledge:ingestion",
            Event(
                type=EventType.UPDATED,
                topic="knowledge:ingestion",
                payload={
                    "domain_id": domain_id,
                    "files_processed": result.files_processed,
                    "chunks_created": result.chunks_created,
                    "files_deleted": result.files_deleted,
                    "status": status,
                },
                timestamp=result.completed_at or datetime.now(timezone.utc),
            ),
        )

    async def ingest_if_changed(
        self,
        domain_id: str,
        last_version: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        config: KnowledgeBaseConfig | None = None,
    ) -> IngestionResult | None:
        """Ingest only if the knowledge base has changed.

        Useful for hot-reload scenarios where re-ingestion should be
        skipped if nothing has changed.

        Args:
            domain_id: Domain to check and potentially ingest
            last_version: Last known version string; if ``None``, always
                ingests
            progress_callback: Optional callback ``(file_path, chunks)``
                passed through to :meth:`ingest`
            config: Optional :class:`KnowledgeBaseConfig` passed through
                to :meth:`ingest`

        Returns:
            :class:`IngestionResult` if ingestion occurred, ``None`` if
            skipped
        """
        if last_version is not None:
            try:
                has_changes = await self._source.has_changes_since(domain_id, last_version)
                if not has_changes:
                    logger.debug(
                        "No changes since version %s for domain: %s",
                        last_version,
                        domain_id,
                    )
                    return None
            except ValueError:
                logger.warning("Domain not found: %s", domain_id)
                return None

        return await self.ingest(
            domain_id,
            progress_callback=progress_callback,
            config=config,
        )

    async def ingest_changes(
        self,
        domain_id: str,
        since_version: str,
        *,
        swap_mode: IngestSwapMode = IngestSwapMode.APPEND,
        progress_callback: Callable[[str, int], None] | None = None,
        config: KnowledgeBaseConfig | None = None,
    ) -> IngestionResult:
        """Ingest only the files changed since ``since_version``.

        ``since_version`` must be a canonical snapshot id previously
        returned by :meth:`get_current_version` (equivalently
        ``backend.get_checksum``) — *not* the monotonic
        ``KnowledgeBaseInfo.version`` counter. The source backend's
        :meth:`~dataknobs_bots.knowledge.storage.KnowledgeResourceBackend.list_changes_since`
        computes the file-level delta; chunks for **deleted and
        modified** files are purged, then **added and modified** files
        are re-embedded through the shared apply-core, so a per-file
        re-ingest gets the same swap semantics as a full re-ingest.

        If ``since_version`` predates the backend's snapshot retention
        (the backend raises
        :class:`~dataknobs_bots.knowledge.storage.InvalidVersionError`),
        this falls back to a full :meth:`ingest` (``CLEAR_FIRST``)
        after a warning — it never silently skips the update.

        Args:
            domain_id: Domain to ingest changes for
            since_version: Canonical snapshot id to diff against
            swap_mode: How the changed files' chunks are replaced.
                Defaults to :attr:`IngestSwapMode.APPEND` (the
                per-file purge-then-reembed behavior).
                :attr:`IngestSwapMode.TOMBSTONE` scopes a zero-downtime
                swap to exactly the changed/deleted files' chunks, so
                a per-file delta gets the same crash-safe guarantee as
                a full re-ingest — through the same apply-core.
            progress_callback: Optional ``(file_path, num_chunks)``
                callback, passed through to the re-embed
            config: Optional :class:`KnowledgeBaseConfig` overriding
                any backend-hosted ``knowledge_base.*``

        Returns:
            :class:`IngestionResult` where ``files_processed`` counts
            added + modified files and ``files_deleted`` carries the
            count of files removed because they no longer exist at the
            source.
        """
        # Resolve the file-level delta *before* entering this method's
        # ingest lifecycle. A ``since_version`` that predates the
        # backend's snapshot retention is a full handoff to
        # :meth:`ingest`, which runs its own complete lifecycle (status
        # write, ``_finalize``, ``result.finish()``). That handoff must
        # stay outside the lifecycle ``try`` below — otherwise a failure
        # inside the delegated ``ingest`` would get its terminal "error"
        # status and ``result.finish()`` written a second time here.
        try:
            change = await self._source.list_changes_since(
                domain_id, since_version
            )
        except InvalidVersionError:
            logger.warning(
                "Version %s for domain %s predates snapshot "
                "retention; falling back to a full re-ingest",
                since_version,
                domain_id,
            )
            return await self.ingest(
                domain_id,
                progress_callback=progress_callback,
                config=config,
            )
        except Exception as e:
            logger.error(
                "Incremental ingestion failed for %s: %s", domain_id, e
            )
            await self._source.set_ingestion_status(
                domain_id, "error", str(e)
            )
            raise

        result = IngestionResult(domain_id=domain_id)
        try:
            # Recover a domain stuck in SWAPPING from a crashed prior
            # swap *before* the INGESTING status write clears the
            # persisted token. No-op when not interrupted.
            await self._reconcile_interrupted_swap(domain_id)
            await self._source.set_ingestion_status(domain_id, "ingesting")

            changed_paths = {f.path for f in change.added} | {
                f.path for f in change.modified
            }
            # Modified files' stale chunks must be purged before
            # re-embedding, alongside the deleted ones.
            delete_paths = sorted(
                set(change.deleted)
                | {f.path for f in change.modified}
            )
            # Files deleted at the source (not modified) — under
            # TOMBSTONE these must stay gone even on rollback (never
            # resurrected), unlike modified files whose old generation
            # is restored.
            purely_deleted_paths = sorted(set(change.deleted))
            logger.info(
                "Ingesting changes for %s: +%d ~%d -%d (since %s)",
                domain_id,
                len(change.added),
                len(change.modified),
                len(change.deleted),
                since_version,
            )

            if not changed_paths and not change.deleted:
                # Nothing added, modified, or deleted — a successful
                # no-op (still records terminal status + event).
                await self._finalize(domain_id, result)
                return result

            await self._apply_file_ops(
                domain_id,
                upsert_filter=lambda f: f.path in changed_paths,
                delete_paths=delete_paths,
                purely_deleted_paths=purely_deleted_paths,
                swap_mode=swap_mode,
                config=config,
                progress_callback=progress_callback,
                result=result,
            )
            # Must precede _finalize: _finalize publishes files_deleted
            # in the completion event payload (it is owned by this
            # caller, not by _apply_file_ops).
            result.files_deleted = len(change.deleted)
            await self._finalize(domain_id, result)

        except Exception as e:
            logger.error(
                "Incremental ingestion failed for %s: %s", domain_id, e
            )
            await self._source.set_ingestion_status(
                domain_id, "error", str(e)
            )
            raise
        finally:
            result.finish()

        return result

    async def get_current_version(self, domain_id: str) -> str | None:
        """Get the current version of a knowledge base.

        Returns the canonical content-snapshot identity
        (``backend.get_checksum``) — the value
        :meth:`ingest_if_changed` compares via
        ``has_changes_since``. Capturing this and passing it back as
        ``last_version`` is now a correct round-trip (previously it
        returned the monotonic ``info.version`` counter, which lived in
        a different space than ``has_changes_since`` compared against,
        causing every checksum-keyed check to spuriously re-ingest).

        Args:
            domain_id: Domain to get version for

        Returns:
            Canonical snapshot identity, or ``None`` if domain doesn't
            exist
        """
        if await self._source.get_info(domain_id) is None:
            return None
        return await self._source.get_checksum(domain_id)

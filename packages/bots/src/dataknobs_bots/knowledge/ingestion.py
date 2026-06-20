"""Knowledge ingestion manager for coordinating file storage to vector storage.

This module provides the KnowledgeIngestionManager which coordinates loading
files from a KnowledgeResourceBackend into a RAGKnowledgeBase.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.callbacks import CallbackRegistry
from dataknobs_common.capabilities import (
    Capability,
    CapabilityLike,
    DynamicCapabilityMixin,
)
from dataknobs_common.tenancy import BoundTenantContext

from .events import INGEST_DOMAIN_END, INGEST_DOMAIN_START
from .storage import IngestionStatus, InvalidVersionError

if TYPE_CHECKING:
    from dataknobs_common.events import EventBus
    from dataknobs_common.ratelimit import RateLimiter
    from dataknobs_common.tenancy import TenantContext
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

      A transient in-swap window remains. Once the old generation is
      marked stale it is hidden from reads; the new generation is then
      written incrementally with no transaction boundary, so a
      concurrent reader first sees nothing, then sees the growing new
      generation as chunks land, and finally the complete new
      generation once embedding finishes — the old generation is never
      visible again from the moment it is marked stale. Closing *that*
      window (so a reader always sees one complete generation) requires
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


class KnowledgeIngestionManager(DynamicCapabilityMixin):
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

    # Capability advertisement: the manager threads the bound
    # ``tenant_id`` (when set) into every chunk's identity tags via
    # :meth:`_compose_extra_metadata` AND into every destination write/
    # delete filter via :meth:`_scope_for_tenant`, so two managers bound
    # to distinct tenants but sharing a destination cannot collide.
    #
    # A tenant-bound manager also routes every backend state operation
    # (``set_ingestion_status`` / ``get_info`` / ``get_checksum`` /
    # ``has_changes_since`` / ``list_changes_since``) through a
    # per-tenant :class:`~dataknobs_common.tenancy.BoundTenantContext`
    # (see :meth:`_resolve_context`), so the backend isolates the
    # tenant's ingestion-status metadata under the tenant's state-key
    # prefix on a shared backend. ``Capability.TENANT_SCOPED_STATE`` and
    # ``Capability.SNAPSHOT_ISOLATION`` therefore hold structurally: the
    # class always has the ctx-routing code path; whether a given
    # instance is currently isolating is the ``self._tenant_id is not
    # None`` binding check (mirroring how :class:`RAGKnowledgeBase`
    # declares ``TENANT_SCOPED_CHUNKS`` structurally). Change detection
    # resolves against the shared domain *content* lineage — content,
    # and the snapshot lineage derived from it, stay domain-keyed; only
    # ingest status is per-tenant.
    #
    # ``TENANT_SCOPED_LOCKS`` / ``TRANSACTIONAL_METADATA`` are
    # deliberately NOT declared: the manager and backends are lock-free.
    # Concurrent same-tenant ingests serialize through the
    # :class:`~dataknobs_bots.knowledge.orchestration.IngestOrchestrator`'s
    # tenant-scoped :class:`~dataknobs_common.locks.DistributedLock`, not
    # a manager/backend lock.
    #
    # The static set holds only the always-true capabilities.
    # ``EVENT_BUS_EMISSION`` is config-dependent — a manager constructed
    # without an ``event_bus`` never fans out to a bus — so it is added
    # per-instance in :meth:`_compute_instance_capabilities`, honoring
    # the "advertised ⇒ the contract guarantees the behaviour" rule.
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset(
        {
            Capability.TENANT_SCOPED_CHUNKS,
            Capability.TENANT_SCOPED_STATE,
            Capability.SNAPSHOT_ISOLATION,
            Capability.CALLBACK_REGISTRY,
            Capability.INGEST_EVENT_PUBLICATION,
        }
    )

    def __init__(
        self,
        source: KnowledgeResourceBackend,
        destination: RAGKnowledgeBase,
        event_bus: EventBus | None = None,
        rate_limiter: RateLimiter | None = None,
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Initialize the ingestion manager.

        Args:
            source: Backend storing knowledge files
            destination: RAG knowledge base for vector storage
            event_bus: Optional event bus for publishing ingestion events
            rate_limiter: Optional :class:`~dataknobs_common.ratelimit.RateLimiter`
                paced over the ingest-path embedder. When set, every
                per-chunk embed call during ingest is preceded by
                ``await rate_limiter.acquire("embed")`` so a
                rate-limited embedding provider (e.g. a hosted API)
                cannot fail the whole ingest under burst. ``None``
                (default) is today's behaviour exactly — no pacing,
                correct for a local Ollama embedder.
            tenant_id: Optional keyword-only tenant identity. When set,
                every chunk this manager writes auto-stamps
                ``tenant_id`` into the chunk-metadata fold (and into
                the chunk-id prefix via
                :attr:`RAGKnowledgeBase._CHUNK_ID_PREFIX_KEYS`) so two
                managers bound to distinct tenants but pointing at the
                same shared :class:`RAGKnowledgeBase` produce disjoint
                chunk-id namespaces. Auto-derived wins over any
                ``extra_metadata={"tenant_id": ...}`` a caller supplies
                — identity is sacred at the write boundary. ``None``
                (default) is today's single-tenant byte-identical
                posture.
        """
        self._source = source
        self._destination = destination
        self._event_bus = event_bus
        self._rate_limiter = rate_limiter
        self._tenant_id = tenant_id
        # Lazily constructed in-process callback registry for ingest
        # lifecycle events (see :attr:`lifecycle_callbacks`).
        self._lifecycle_callbacks: CallbackRegistry | None = None
        # DynamicCapabilityMixin: EVENT_BUS_EMISSION is computed from
        # whether an event_bus was supplied (see
        # :meth:`_compute_instance_capabilities`).
        self._init_capability_cache()

    def _compute_instance_capabilities(self) -> frozenset[CapabilityLike]:
        """Add ``EVENT_BUS_EMISSION`` only when an event bus is bound.

        A busless manager fires only the in-process
        :attr:`lifecycle_callbacks`; it never fans out to a bus, so it
        must not advertise :attr:`Capability.EVENT_BUS_EMISSION` (a
        consumer ``require_capability``-guarding on it would otherwise
        get a false positive). The lifecycle registry auto-composes
        ``also_publish_to(event_bus)`` exactly when ``self._event_bus``
        is set, so that field is the authoritative signal.
        """
        caps = type(self).SUPPORTED_CAPABILITIES
        if self._event_bus is not None:
            caps = caps | {Capability.EVENT_BUS_EMISSION}
        return caps

    @property
    def source(self) -> KnowledgeResourceBackend:
        """The backend this manager ingests from.

        Exposed read-only so callers (e.g. an
        :class:`~dataknobs_bots.knowledge.orchestration.IngestOrchestrator`
        resolving a per-tenant manager) can reach the backend to
        classify a trigger key without coupling to the private field.
        """
        return self._source

    @property
    def lifecycle_callbacks(self) -> CallbackRegistry:
        """In-process registry receiving ingest-lifecycle events.

        Fires:
            ``INGEST_DOMAIN_START`` — at the beginning of every
                :meth:`ingest` and :meth:`ingest_changes` call.
            ``INGEST_DOMAIN_END`` — at the end of every :meth:`ingest`
                and :meth:`ingest_changes` call (success OR failure).

        Consumers register callbacks for in-process observability;
        compose with
        :meth:`~dataknobs_common.callbacks.CallbackRegistry.also_publish_to`
        for cross-replica fan-out. When this manager was constructed
        with an ``event_bus``, the registry auto-composes
        ``also_publish_to(event_bus)`` on first access, so a consumer
        passing an event bus gets cross-replica fan-out for free. The
        event payload is a ``dict[str, Any]`` matching the canonical
        shape documented on the topic constants in
        :mod:`dataknobs_bots.knowledge.events`.
        """
        if self._lifecycle_callbacks is None:
            self._lifecycle_callbacks = CallbackRegistry()
            if self._event_bus is not None:
                self._lifecycle_callbacks.also_publish_to(self._event_bus)
        return self._lifecycle_callbacks

    def _resolve_context(self, domain_id: str) -> TenantContext | None:
        """The :class:`TenantContext` for a per-domain state operation.

        Returns ``BoundTenantContext(self._tenant_id, domain_id)`` when
        this manager is tenant-bound, else ``None`` — which every
        backend treats as the single-tenant case (byte-identical state
        paths to an unbound manager). Returning ``None`` (rather than a
        throwaway ``SingleTenantContext``) keeps the unbound path's
        backend state calls literally unchanged: an unbound manager
        passes ``ctx=None``, exactly as before this routing existed.

        Threaded into every backend state call so a tenant-bound
        manager isolates its ingestion **status** under the tenant's
        state-key prefix on a shared backend. Content — and the snapshot
        lineage derived from it — stays domain-keyed; a tenant's change
        detection resolves against the shared content lineage and stays
        minimal.
        """
        if self._tenant_id is not None:
            return BoundTenantContext(self._tenant_id, domain_id)
        return None

    def _scope_for_tenant(self, base: Mapping[str, Any]) -> dict[str, Any]:
        """AND-compose the bound ``tenant_id`` into a write/delete filter.

        Every filter the manager uses to clear, tombstone, or
        otherwise mutate rows on the destination passes through here
        so a tenant-bound manager cannot accidentally delete another
        tenant's rows under the same ``domain_id``. ``tenant_id``
        absent (unbound manager) is a no-op pass-through — single-
        tenant byte-identical posture preserved.

        Returns a fresh dict so the caller's mapping is never mutated.
        """
        scoped = dict(base)
        if self._tenant_id is not None:
            scoped["tenant_id"] = self._tenant_id
        return scoped

    def _compose_extra_metadata(
        self,
        extra_metadata: Mapping[str, Any] | None,
        *,
        domain_id: str,
        generation: str | None = None,
    ) -> dict[str, Any]:
        """Compose the ``extra_metadata`` dict handed to the destination.

        Auto-derived identity tags (the bound ``tenant_id``, the per-
        call ``domain_id``, the TOMBSTONE-swap ``_generation`` token)
        win over caller-supplied keys on collision — identity is sacred
        at the write boundary, so a malicious or mistaken caller cannot
        re-tag chunks for another tenant or another domain via the
        ``extra_metadata`` channel. Non-identity keys are preserved.
        Returns a fresh dict so the caller's mapping is never mutated.

        Mirrors :meth:`RAGKnowledgeBase._compose_extra_metadata` — same
        precedence rule, same name, the two call sites share the
        ``auto-derived wins`` invariant.
        """
        composed: dict[str, Any] = dict(extra_metadata or {})
        composed["domain_id"] = domain_id
        if self._tenant_id is not None:
            composed["tenant_id"] = self._tenant_id
        if generation is not None:
            composed["_generation"] = generation
        return composed

    @staticmethod
    def _resolve_swap_mode(
        clear_existing: bool | None,
        swap_mode: IngestSwapMode | None,
    ) -> IngestSwapMode:
        """Resolve the effective swap mode from the two knobs.

        Passing the legacy ``clear_existing`` at all (i.e. not
        ``None``) always emits a ``DeprecationWarning`` — even when
        ``swap_mode`` is also given and overrides it — because the
        caller is still using a deprecated parameter. ``swap_mode`` is
        authoritative when set; ``clear_existing`` only *determines*
        the mode when ``swap_mode`` is not given
        (``True`` → ``CLEAR_FIRST``, ``False`` → ``APPEND``). With
        neither set the default is ``CLEAR_FIRST`` — identical to the
        pre-deprecation ``clear_existing=True`` default, so existing
        callers are unaffected.
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
        extra_metadata: Mapping[str, Any] | None = None,
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
            extra_metadata: Optional keyword-only mapping merged into
                every chunk's metadata before identity auto-derivation.
                Auto-derived identity tags (``domain_id``, the bound
                ``tenant_id``, the TOMBSTONE-swap ``_generation`` token)
                win on collision — a caller cannot silently re-tag
                chunks for another tenant or domain via this channel.
                Non-identity keys (``cohort``, ``region``, any custom
                tag) are preserved as-is.

        Returns:
            :class:`IngestionResult` with aggregate statistics
        """
        effective_mode = self._resolve_swap_mode(clear_existing, swap_mode)
        result = IngestionResult(domain_id=domain_id)
        ctx = self._resolve_context(domain_id)
        await self._publish_ingest_start(domain_id, result.started_at)
        failed = False

        try:
            # Recover a domain stuck in SWAPPING from a crashed prior
            # swap *before* the INGESTING status write clears the
            # persisted token. No-op when not interrupted.
            await self._reconcile_interrupted_swap(domain_id)
            await self._source.set_ingestion_status(
                domain_id, IngestionStatus.INGESTING, ctx=ctx
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
                extra_metadata=extra_metadata,
            )
            await self._finalize(domain_id, result)

        except Exception as e:
            failed = True
            logger.error("Ingestion failed for %s: %s", domain_id, e)
            await self._source.set_ingestion_status(
                domain_id, IngestionStatus.ERROR, str(e), ctx=ctx
            )
            raise
        finally:
            result.finish()
            await self._publish_ingest_end(
                domain_id, result, failed=failed
            )

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
        extra_metadata: Mapping[str, Any] | None = None,
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
                extra_metadata=extra_metadata,
            )
            return

        if swap_mode is IngestSwapMode.CLEAR_FIRST:
            await self._destination.clear(
                filter=self._scope_for_tenant({"domain_id": domain_id})
            )
            logger.debug(
                "Cleared existing vectors for domain: %s", domain_id
            )

        for path in delete_paths or ():
            await self._destination.clear(
                filter=self._scope_for_tenant(
                    {"domain_id": domain_id, "source_path": path}
                )
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
            extra_metadata=extra_metadata,
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
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Re-embed the selected source files into the destination.

        The one place :meth:`RAGKnowledgeBase.ingest_from_backend` is
        invoked and its statistics are mapped onto ``result`` — shared
        by every swap mode so the embed + stats handling never drifts
        between them.

        ``generation`` (TOMBSTONE only) and the bound ``tenant_id`` are
        folded into the chunk metadata via
        :meth:`_compose_extra_metadata`. Auto-derived identity tags
        (``domain_id``, the bound ``tenant_id``, the ``_generation``
        token) win over any caller-supplied ``extra_metadata`` entry —
        identity is sacred at the write boundary.
        :meth:`RAGKnowledgeBase._embed_and_store_chunks` then folds the
        present identity keys into the chunk-id prefix per
        :attr:`RAGKnowledgeBase._CHUNK_ID_PREFIX_KEYS`, so two tenants
        ingesting the same ``domain_id`` produce disjoint chunk-id
        namespaces and a generation swap produces distinct ids
        (instead of UPSERTing the tombstoned old rows in place). With
        no bound tenant and no ``generation`` (APPEND / CLEAR_FIRST,
        single-tenant) the id derivation is byte-for-byte unchanged.
        """
        composed = self._compose_extra_metadata(
            extra_metadata,
            domain_id=domain_id,
            generation=generation,
        )
        stats = await self._destination.ingest_from_backend(
            self._source,
            domain_id,
            config=config,
            progress_callback=progress_callback,
            extra_metadata=composed,
            file_filter=upsert_filter,
            rate_limiter=self._rate_limiter,
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
        extra_metadata: Mapping[str, Any] | None = None,
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
           — mark the existing generation. ``_stale`` exclusion is not
           expressible as an equality filter, so it is enforced as a
           post-filter on the read path:
           :meth:`RAGKnowledgeBase._fetch_drop_stale_truncate`
           over-fetches and drops ``_stale``-true rows via
           :meth:`RAGKnowledgeBase._is_stale`. Reads stop seeing the
           old generation *without anything being deleted*.
        2. Status → :attr:`IngestionStatus.SWAPPING`.
        3. Ingest the new generation (distinct ids, no ``_stale`` key,
           stamped ``_generation``). There is no transaction boundary:
           each new chunk is read-visible the moment its
           ``add_vectors`` write lands, so the new generation becomes
           visible incrementally as step 3 progresses (not atomically
           at the end).
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
        remains: from step 1 the old generation is hidden, and the new
        generation is only partially visible until step 3 finishes
        writing it (a reader during step 3 sees whatever subset has
        been embedded so far). Closing that window needs a generation
        pointer-flip (``SHADOW_GENERATION``), which ``TOMBSTONE``
        deliberately trades away for this simpler crash-safe mechanism.

        Scopes (selected by whether this is a full re-ingest or a
        delta — see the scope-selection block below):

        - ``tombstone_scope`` — every old chunk to hide then retire:
          ``{"domain_id": ...}`` for a full re-ingest; the
          changed+deleted ``source_path`` list for a per-file delta;
          ``None`` for a purely additive delta (new files only — no
          prior generation exists, so nothing is tombstoned or
          retired).
        - ``modified_scope`` — old chunks of files being re-embedded;
          restored to visibility on rollback (``None`` when no existing
          file is re-embedded).
        - ``deleted_scope`` — old chunks of files *deleted at the
          source*; purged unconditionally, never resurrected.
        """
        generation = uuid.uuid4().hex
        base: dict[str, Any] = self._scope_for_tenant({"domain_id": domain_id})
        deleted_set = set(purely_deleted_paths or ())

        # The swap scope must match the *upsert* scope, not whether any
        # file was deleted/modified. ``upsert_filter is None`` is the
        # one true discriminator: only :meth:`ingest` (full re-ingest)
        # passes it, and only then is every file re-embedded — so the
        # whole old generation is being replaced. Any non-None filter
        # is a delta (:meth:`ingest_changes`): a *subset* is re-embedded,
        # so only the changed/deleted files' old chunks may be swapped.
        # Keying off ``delete_paths`` truthiness conflated a purely
        # additive delta (new files only) with a full re-ingest and
        # retired every untouched file's chunks.
        tombstone_scope: dict[str, Any] | None
        if upsert_filter is None:
            # Full re-ingest: the whole domain is re-embedded, so the
            # entire old generation is the modified scope; nothing is
            # purely deleted.
            tombstone_scope = dict(base)
            modified_scope = dict(base)
            deleted_scope = None
        elif delete_paths:
            # Per-file delta touching existing chunks (modified and/or
            # deleted files).
            tombstone_scope = {
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
            # Purely additive delta: only brand-new files are embedded.
            # They have no prior generation, so there is nothing to
            # tombstone or retire — the generation token is still
            # threaded so a failed/crashed add is cleaned up precisely
            # by token (rollback step 1 / reconcile), but no existing
            # chunk is touched.
            tombstone_scope = None
            modified_scope = None
            deleted_scope = None

        if tombstone_scope is not None:
            await self._destination.update_metadata_where(
                tombstone_scope, {"_stale": True}
            )
        await self._source.set_ingestion_status(
            domain_id,
            IngestionStatus.SWAPPING,
            generation=generation,
            ctx=self._resolve_context(domain_id),
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
                extra_metadata=extra_metadata,
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
            # ``_stale`` key, so it is untouched. ``None`` scope is a
            # purely additive delta: nothing was tombstoned, so there
            # is nothing to retire.
            if tombstone_scope is not None:
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
            filter=self._scope_for_tenant(
                {"domain_id": domain_id, "_generation": generation}
            )
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
        ctx = self._resolve_context(domain_id)
        info = await self._source.get_info(domain_id, ctx=ctx)
        if (
            info is None
            or info.ingestion_status is not IngestionStatus.SWAPPING
        ):
            return False

        token = info.generation
        await self._destination.update_metadata_where(
            self._scope_for_tenant({"domain_id": domain_id, "_stale": True}),
            {"_stale": False},
        )
        if token:
            await self._destination.clear(
                filter=self._scope_for_tenant(
                    {"domain_id": domain_id, "_generation": token}
                )
            )
        await self._source.set_ingestion_status(
            domain_id, IngestionStatus.READY, ctx=ctx
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
        """Set the terminal ingestion status and log.

        Shared tail of :meth:`ingest` and :meth:`ingest_changes` — the
        success/error decision, backend status write, and completion
        log are one behavior, not duplicated per entry point. Returns
        the resolved status string. The ``INGEST_DOMAIN_END`` lifecycle
        event is fired by the entry methods themselves (in their
        ``finally`` block) so it covers the failure path too, not just
        this success tail.
        """
        status = "ready" if result.success else "error"
        error_msg = str(result.errors) if result.errors else None
        await self._source.set_ingestion_status(
            domain_id,
            IngestionStatus.READY
            if result.success
            else IngestionStatus.ERROR,
            error_msg,
            ctx=self._resolve_context(domain_id),
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
        return status

    def _build_lifecycle_payload(
        self,
        domain_id: str,
        **extras: Any,
    ) -> dict[str, Any]:
        """Compose a lifecycle-event payload.

        Adds the bound ``tenant_id`` only when this manager is
        tenant-bound, so single-tenant subscribers see byte-identical
        payloads modulo the ``tenant_id`` key.
        """
        payload: dict[str, Any] = {"domain_id": domain_id, **extras}
        if self._tenant_id is not None:
            payload["tenant_id"] = self._tenant_id
        return payload

    async def _publish_ingest_start(
        self,
        domain_id: str,
        started_at: datetime,
    ) -> None:
        """Fire ``INGEST_DOMAIN_START`` for an ingest run."""
        payload = self._build_lifecycle_payload(
            domain_id,
            started_at=started_at.isoformat(),
        )
        await self.lifecycle_callbacks.fire_async(
            INGEST_DOMAIN_START, payload
        )

    async def _publish_ingest_end(
        self,
        domain_id: str,
        result: IngestionResult,
        *,
        failed: bool,
    ) -> None:
        """Fire ``INGEST_DOMAIN_END`` for an ingest run.

        ``failed`` is ``True`` when the run raised; otherwise the status
        reflects ``result.success`` (recorded errors without an
        exception still mark the run failed).
        """
        status = "completed" if (result.success and not failed) else "failed"
        payload = self._build_lifecycle_payload(
            domain_id,
            files_processed=result.files_processed,
            chunks_created=result.chunks_created,
            files_deleted=result.files_deleted,
            status=status,
            completed_at=(
                result.completed_at or datetime.now(timezone.utc)
            ).isoformat(),
        )
        await self.lifecycle_callbacks.fire_async(
            INGEST_DOMAIN_END, payload
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
                has_changes = await self._source.has_changes_since(
                    domain_id,
                    last_version,
                    ctx=self._resolve_context(domain_id),
                )
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
        extra_metadata: Mapping[str, Any] | None = None,
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
            extra_metadata: Optional mapping merged into every chunk's
                metadata before identity auto-derivation. Same
                precedence rule as :meth:`ingest` — auto-derived
                identity tags (``domain_id``, the bound ``tenant_id``,
                the ``_generation`` token) win on collision; non-
                identity keys are preserved.

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
        ctx = self._resolve_context(domain_id)
        try:
            change = await self._source.list_changes_since(
                domain_id, since_version, ctx=ctx
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
                domain_id, IngestionStatus.ERROR, str(e), ctx=ctx
            )
            raise

        result = IngestionResult(domain_id=domain_id)
        await self._publish_ingest_start(domain_id, result.started_at)
        failed = False
        try:
            # Recover a domain stuck in SWAPPING from a crashed prior
            # swap *before* the INGESTING status write clears the
            # persisted token. No-op when not interrupted.
            await self._reconcile_interrupted_swap(domain_id)
            await self._source.set_ingestion_status(
                domain_id, IngestionStatus.INGESTING, ctx=ctx
            )

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
                extra_metadata=extra_metadata,
            )
            # Must precede the INGEST_DOMAIN_END fire: the lifecycle
            # payload reports files_deleted (it is owned by this caller,
            # not by _apply_file_ops).
            result.files_deleted = len(change.deleted)
            await self._finalize(domain_id, result)

        except Exception as e:
            failed = True
            logger.error(
                "Incremental ingestion failed for %s: %s", domain_id, e
            )
            await self._source.set_ingestion_status(
                domain_id, IngestionStatus.ERROR, str(e), ctx=ctx
            )
            raise
        finally:
            result.finish()
            await self._publish_ingest_end(
                domain_id, result, failed=failed
            )

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
        ctx = self._resolve_context(domain_id)
        if await self._source.get_info(domain_id, ctx=ctx) is None:
            return None
        return await self._source.get_checksum(domain_id, ctx=ctx)

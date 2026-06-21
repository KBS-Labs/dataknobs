"""Shared change-detection core for knowledge resource backends.

`get_checksum` / `has_changes_since` / `list_changes_since` are one
algorithm, not three per-backend reimplementations. Historically each
backend copy-pasted ``return info.version != version`` for
``has_changes_since`` while ``get_checksum`` returned a content-snapshot
MD5 — different value spaces, so the documented
``get_checksum``→``has_changes_since`` pairing always reported "changed".
This mixin makes the canonical content
snapshot *the* version: ``get_checksum`` produces it, and both
``has_changes_since`` and ``list_changes_since`` derive from it. One
reviewed implementation; every backend inherits correct behaviour.

`_load_snapshot` is the only backend-specific seam. The default
(no stored snapshot ⇒ empty snapshot) is correct but non-minimal: when
the requested version differs from the current one every current file is
reported ``added`` (a full, not delta, re-ingest). ``is_empty`` stays
correct via the version-equality short-circuit, so change *detection* is
right for every backend even without an override. All three in-tree
backends now override ``_load_snapshot`` with a real per-version store
for minimal diffs: memory (in-process map), file
(``_snapshots/<version>.json``), and S3 (snapshot objects, or the
metadata object's own S3 version history in ``s3_versioning`` mode). The
base default remains for out-of-tree backends.
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, BinaryIO, ClassVar

from dataknobs_common.callbacks import CallbackRegistry
from dataknobs_common.capabilities import (
    Capability,
    CapabilityLike,
    CapabilityMixin,
)

from ..events import INGEST_METADATA_WRITE, INGEST_SNAPSHOT_WRITE
from .key_layout import KnowledgeKeyKind
from .models import ChangeSet, InvalidVersionError

if TYPE_CHECKING:
    from dataknobs_common.events import Event, EventBus, Subscription
    from dataknobs_common.tenancy import TenantContext

    from .models import KnowledgeBaseInfo, KnowledgeFile


class KnowledgeResourceBackendMixin(CapabilityMixin):
    """Canonical change detection + key-layout classification.

    Mixed into every in-tree backend. Relies only on the backend's own
    :meth:`get_info` and :meth:`list_files` (both part of the
    ``KnowledgeResourceBackend`` protocol), so it is backend-agnostic.

    Declares the three layout constants (``METADATA_FILE`` /
    ``CONTENT_DIR`` / ``SNAPSHOTS_DIR``) once at the contract layer:
    any backend honoring the documented layout inherits identical
    values and the default :meth:`classify_key` implementation for
    free. A backend with a non-standard layout overrides the constants
    (and inherits the same :meth:`classify_key` against the overridden
    values) or overrides :meth:`classify_key` entirely.

    Async transport contract: the shared async storage methods MUST NOT block
    the event loop — use an async transport or offload blocking disk I/O via
    ``asyncio.to_thread`` / ``aiter_sync_in_thread`` (ruff ``ASYNC`` enforces;
    ``assert_no_blocking()`` proves).
    """

    # --- Layout constants (consumed by classify_key + each backend) ---

    METADATA_FILE: ClassVar[str] = "_metadata.json"
    CONTENT_DIR: ClassVar[str] = "content"
    SNAPSHOTS_DIR: ClassVar[str] = "_snapshots"

    # --- Capability declaration ---
    #
    # These four are implemented by this mixin uniformly, so every
    # in-tree backend inherits them honestly:
    #   - KEY_PATTERN_FILTERING / CHANGE_SUBSCRIPTION — every backend
    #     implements ``key_pattern`` + the mixin's ``classify_key`` /
    #     ``subscribe_to_changes`` surfaces.
    #   - BACKEND_STATE_OBSERVABILITY / CALLBACK_REGISTRY — every
    #     backend fires metadata / snapshot state-write events through
    #     the shared ``_fire_state_write`` helper on
    #     ``state_write_callbacks``.
    # Capabilities still landing incrementally as their behaviour ships
    # (``STREAMING_READS`` on backends that implement ``stream_file``;
    # ``TENANT_SCOPED_STATE`` once state methods are tenant-scoped at
    # the contract layer) are deliberately NOT declared here — adopters
    # get the honest "not advertised" answer until the behaviour
    # exists. A backend widening its own set unions onto this base.
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset({
        Capability.KEY_PATTERN_FILTERING,
        Capability.CHANGE_SUBSCRIPTION,
        Capability.BACKEND_STATE_OBSERVABILITY,
        Capability.CALLBACK_REGISTRY,
    })

    # --- Tenant-context scoping (shared) ---

    @staticmethod
    def _state_prefix(ctx: TenantContext | None) -> str:
        """State-key prefix for ``ctx`` (``""`` for the no-tenant case).

        When ``ctx`` is ``None`` (every single-tenant call site) the
        prefix is empty, so the tenant-scoped state keys/paths are
        byte-identical to the pre-tenancy layout. A bound tenant context
        contributes its
        :meth:`~dataknobs_common.tenancy.TenantContext.state_key_prefix`,
        isolating per-tenant ingest **state** (metadata + snapshot
        lineage). **Content** stays keyed by ``domain_id`` alone — a
        backend routes only its state key/path construction through this
        helper, never its content paths.
        """
        return ctx.state_key_prefix() if ctx is not None else ""

    # --- Required of any backend (supplied by the concrete class) ---

    async def get_info(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> KnowledgeBaseInfo | None:
        """Return KB metadata, or ``None`` if it does not exist.

        ``ctx`` scopes the per-tenant ingest **state** view (ingestion
        status / generation token); ``None`` preserves the single-tenant
        view. KB existence/identity stays keyed by ``domain_id``.
        """
        raise NotImplementedError  # pragma: no cover - overridden by backends

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[KnowledgeFile]:
        """Return all files in the KB (used to compute the snapshot)."""
        raise NotImplementedError  # pragma: no cover - overridden by backends

    async def get_state_version(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> str | None:
        """Opaque state-version token of the KB metadata document.

        Declared abstract here rather than defaulted: the token is
        backend-native (S3 ETag / file content hash / in-memory counter),
        and — more importantly — a real token is only honest when the
        backend also enforces it on the write side (the
        ``expected_version`` guard in :meth:`set_ingestion_status`). A
        mixin default would hand an out-of-tree backend a read token with
        no enforcement, advertising a conditional-write contract it does
        not keep. Backends that realize both halves override this method
        and advertise :attr:`~dataknobs_common.Capability.TRANSACTIONAL_METADATA`.
        See the protocol for the full contract.
        """
        raise NotImplementedError  # pragma: no cover - overridden by backends

    def key_pattern(
        self,
        kind: KnowledgeKeyKind = KnowledgeKeyKind.CONTENT,
        domain_id: str | None = None,
    ) -> str:
        """Backend-native pattern for keys of ``kind`` (see the protocol)."""
        raise NotImplementedError  # pragma: no cover - overridden by backends

    # --- Canonical change-detection algorithm (shared) ---

    @staticmethod
    def _guess_content_type(path: str) -> str:
        """Best-effort MIME type for ``path`` (default octet-stream).

        The first ``mimetypes`` lookup in a process lazily reads the
        system MIME database from disk, a blocking call — invoke this via
        ``asyncio.to_thread`` from an async method so the event loop stays
        free. Shared by the file and S3 backends.
        """
        guessed_type, _ = mimetypes.guess_type(path)
        return guessed_type or "application/octet-stream"

    @staticmethod
    async def _read_content_bytes(content: bytes | BinaryIO) -> bytes:
        """Normalize :meth:`put_file` content to bytes.

        ``bytes`` pass straight through. A file-like object is read via
        ``asyncio.to_thread`` so a real OS file handle's blocking
        ``read()`` does not stall the event loop. Shared by every
        backend's :meth:`put_file` so the read happens off-loop in one
        reviewed place rather than three inline copies.
        """
        if isinstance(content, bytes):
            return content
        return await asyncio.to_thread(content.read)

    @staticmethod
    def _identity_of_snapshot(snapshot: dict[str, str]) -> str:
        """Canonical content-snapshot identity of a ``{path: checksum}`` map.

        MD5 over the sorted ``path:checksum`` pairs; ``""`` for an empty
        map. This is *the* version-identity formula — every other
        identity entry point (:meth:`_snapshot_identity`,
        :meth:`get_checksum`, and the S3-versioning backend's
        history-matching fast path) routes through here so a stored
        snapshot and a freshly-listed one can never disagree on identity.
        """
        if not snapshot:
            return ""
        combined = ":".join(sorted(f"{p}:{c}" for p, c in snapshot.items()))
        return hashlib.md5(combined.encode()).hexdigest()

    @classmethod
    def _snapshot_identity(cls, files: list[KnowledgeFile]) -> str:
        """Canonical content-snapshot identity of a file list.

        Thin adapter over :meth:`_identity_of_snapshot` — pure function
        of the supplied files so it can be applied to an already-fetched
        ``list_files()`` result without a second backend round-trip.
        """
        return cls._identity_of_snapshot({f.path: f.checksum for f in files})

    async def get_checksum(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> str:
        """Canonical content-snapshot identity of the whole KB.

        MD5 over the sorted ``path:checksum`` of every file. The empty
        KB has identity ``""``. This value *is* the version: capture it,
        pass it back to :meth:`has_changes_since` /
        :meth:`list_changes_since`.

        ``ctx`` scopes only the KB-existence check (via
        :meth:`get_info`); the identity itself is a **content** hash over
        the shared ``list_files`` result, so it is the same across
        tenants of the same ``domain_id``.

        Raises:
            ValueError: If ``domain_id`` does not exist.
        """
        if await self.get_info(domain_id, ctx=ctx) is None:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")
        return self._snapshot_identity(await self.list_files(domain_id))

    async def list_changes_since(
        self,
        domain_id: str,
        version: str,
        *,
        ctx: TenantContext | None = None,
    ) -> ChangeSet:
        """Diff the current KB against the snapshot identified by ``version``.

        ``version`` is a value previously returned by
        :meth:`get_checksum`. If it equals the current canonical
        identity the result is empty without needing a stored snapshot
        (the equality short-circuit — correct for every backend).
        Otherwise the current files are diffed against
        :meth:`_load_snapshot`.

        Raises:
            ValueError: If ``domain_id`` does not exist.
            InvalidVersionError: If ``version`` differs from the current
                identity and the backend cannot resolve it to a snapshot
                (predates retention / unknown). Consumers fall back to a
                full re-ingest.
        """
        if await self.get_info(domain_id, ctx=ctx) is None:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")
        files = await self.list_files(domain_id)
        current_version = self._snapshot_identity(files)
        current = {f.path: f for f in files}
        if version == current_version:
            return ChangeSet(
                added=[], modified=[], deleted=[], version=current_version
            )

        snapshot = await self._load_content_snapshot(domain_id, version, ctx)
        added: list[KnowledgeFile] = []
        modified: list[KnowledgeFile] = []
        for path, file in current.items():
            if path not in snapshot:
                added.append(file)
            elif snapshot[path] != file.checksum:
                modified.append(file)
        deleted = [path for path in snapshot if path not in current]
        return ChangeSet(
            added=added,
            modified=modified,
            deleted=deleted,
            version=current_version,
        )

    async def has_changes_since(
        self,
        domain_id: str,
        version: str,
        *,
        ctx: TenantContext | None = None,
    ) -> bool:
        """``True`` if the KB differs from the snapshot at ``version``.

        The degenerate case of :meth:`list_changes_since`:
        one algorithm, no sibling to drift. An unresolvable version is
        treated as "assume changed" so callers safely re-ingest.

        Raises:
            ValueError: If ``domain_id`` does not exist.
        """
        try:
            return not (
                await self.list_changes_since(domain_id, version, ctx=ctx)
            ).is_empty
        except InvalidVersionError:
            return True

    async def _load_content_snapshot(
        self,
        domain_id: str,
        version: str,
        ctx: TenantContext | None,
    ) -> dict[str, str]:
        """Resolve a content ``version`` to its snapshot, tenant-first.

        The change-detection policy layer over :meth:`_load_snapshot`.
        Snapshot *versions* are content identities (the value
        :meth:`get_checksum` returns is the same for every tenant of a
        domain) and the snapshot *map* is shared domain content state.
        Content mutations (``put_file`` / ``delete_file``) therefore
        record snapshots only under the domain-keyed store; the
        per-tenant snapshot store is populated solely by an upper layer
        that may not be wired.

        So a tenant whose own scope has no retained snapshot for a
        content version falls back to the shared domain-keyed lineage
        (``ctx=None``) rather than treating the version as unresolvable
        and forcing a full re-ingest. This keeps per-tenant change
        detection minimal while leaving the strict, scope-local
        semantics of :meth:`_load_snapshot` itself unchanged. The
        fallback only engages for a context that actually carries a
        state prefix; single-tenant callers are byte-identical to a
        direct :meth:`_load_snapshot` call.
        """
        try:
            return await self._load_snapshot(domain_id, version, ctx=ctx)
        except InvalidVersionError:
            if not self._state_prefix(ctx):
                raise
            return await self._load_snapshot(domain_id, version, ctx=None)

    async def _load_snapshot(
        self,
        domain_id: str,
        version: str,
        *,
        ctx: TenantContext | None = None,
    ) -> dict[str, str]:
        """Resolve ``version`` to a ``{path: checksum}`` snapshot map.

        Strict and scope-local: reads only the store selected by ``ctx``
        (the domain-keyed store for ``ctx=None`` / an empty prefix, the
        per-tenant store otherwise) and raises
        :class:`InvalidVersionError` for a version not retained *in that
        scope*. The tenant-first fallback across scopes lives in
        :meth:`_load_content_snapshot`, the change-detection caller.

        Default: no per-version store ⇒ the empty snapshot. Combined
        with the equality short-circuit in :meth:`list_changes_since`
        this yields correct change *detection* for every backend (a
        differing version reports every current file as ``added`` — a
        full, not minimal, re-ingest). Backends with a real store
        override this and raise :class:`InvalidVersionError` for an
        unretained version.
        """
        return {}

    # --- Key-layout classification (shared) ---

    def classify_key(self, key: str) -> KnowledgeKeyKind:
        """Classify a key by its position in the backend's layout.

        Inspects the key's path segments (split on ``"/"``) against the
        declared layout constants:

        - Any segment equal to :attr:`CONTENT_DIR` →
          :attr:`KnowledgeKeyKind.CONTENT` (so a content file named
          ``content/_metadata.json`` is still ``CONTENT`` — the
          leading ``content/`` segment wins, preventing false-positives
          on consumer filenames that coincidentally match a state-key
          name). The ``content`` segment wins at *any* depth: a
          consumer-uploaded file at
          ``{domain}/content/a/b/content/x.md`` still classifies as
          ``CONTENT``, matching the :meth:`put_file` key structure
          (which always writes under the ``content/`` subtree, never
          alongside it).
        - Terminal segment equal to :attr:`METADATA_FILE` and no
          ``CONTENT_DIR`` ancestor → :attr:`KnowledgeKeyKind.METADATA`.
        - Any segment equal to :attr:`SNAPSHOTS_DIR` and no
          ``CONTENT_DIR`` ancestor → :attr:`KnowledgeKeyKind.SNAPSHOT`.
        - Anything else → :attr:`KnowledgeKeyKind.UNKNOWN`.

        Prefer :meth:`key_pattern` (per-backend) for source-level
        filtering when the event source supports patterns — filtering
        upstream avoids paying the message-receive cost for state
        writes. Use :meth:`classify_key` for per-event filtering when
        patterns are not supported.
        """
        segments = [s for s in key.split("/") if s]
        if not segments:
            return KnowledgeKeyKind.UNKNOWN
        if self.CONTENT_DIR in segments:
            return KnowledgeKeyKind.CONTENT
        if segments[-1] == self.METADATA_FILE:
            return KnowledgeKeyKind.METADATA
        if self.SNAPSHOTS_DIR in segments:
            return KnowledgeKeyKind.SNAPSHOT
        return KnowledgeKeyKind.UNKNOWN

    # --- State-write observability (shared) ---

    @property
    def state_write_callbacks(self) -> CallbackRegistry:
        """In-process registry receiving backend state-write events.

        Fires:
            ``INGEST_METADATA_WRITE`` — after every metadata state write.
            ``INGEST_SNAPSHOT_WRITE`` — after every snapshot state write.

        Consumers register callbacks for in-process observability;
        compose with
        :meth:`~dataknobs_common.callbacks.CallbackRegistry.also_publish_to`
        for cross-replica fan-out. The event payload is a
        ``dict[str, Any]`` with keys ``domain_id``, ``key``, ``kind``
        (:class:`KnowledgeKeyKind`), ``byte_size``.

        Lazily constructed (the mixin has no ``__init__`` of its own, so
        the registry is created on first access and cached on the
        instance). Backends fire through :meth:`_fire_state_write`.
        """
        reg: CallbackRegistry | None = getattr(
            self, "_state_write_callbacks", None
        )
        if reg is None:
            reg = CallbackRegistry()
            self._state_write_callbacks = reg
        return reg

    async def _fire_state_write(
        self,
        *,
        domain_id: str,
        key: str,
        kind: KnowledgeKeyKind,
        byte_size: int,
    ) -> None:
        """Fire a state-write event on the per-backend registry.

        Called from a backend's metadata / snapshot write paths on every
        successful write. Zero-overhead when no callbacks were ever
        registered (the registry is not constructed until first access
        via :attr:`state_write_callbacks`). Async so EventBus fan-out
        configured via ``also_publish_to`` is awaited correctly from the
        backend's running event loop (a sync ``fire`` would be rejected
        by the substrate's fan-out-in-running-loop guard).
        """
        reg: CallbackRegistry | None = getattr(
            self, "_state_write_callbacks", None
        )
        if reg is None:
            return  # No registry constructed — zero overhead.

        if kind is KnowledgeKeyKind.METADATA:
            topic = INGEST_METADATA_WRITE
        elif kind is KnowledgeKeyKind.SNAPSHOT:
            topic = INGEST_SNAPSHOT_WRITE
        else:  # pragma: no cover - CONTENT / UNKNOWN never reach here
            return

        await reg.fire_async(
            topic,
            {
                "domain_id": domain_id,
                "key": key,
                "kind": kind,
                "byte_size": byte_size,
            },
        )

    # --- Change subscription convenience (shared) ---

    def _kind_to_topic_pattern(
        self,
        kinds: Iterable[KnowledgeKeyKind],
        domain_id: str | None,
    ) -> str:
        """Derive an fnmatch pattern from the requested kinds.

        Single-kind: returns the backend's :meth:`key_pattern` output
        directly. Multi-kind is not expressible as a single fnmatch
        pattern (the in-tree :class:`InMemoryEventBus` uses Python
        ``fnmatch``, which has no ``{a,b}`` alternation), so it raises
        with consumer guidance to subscribe once per kind. The
        single-kind path is the load-bearing intent.
        """
        kinds_set = frozenset(kinds)
        if len(kinds_set) == 1:
            only_kind = next(iter(kinds_set))
            return self.key_pattern(only_kind, domain_id)
        raise ValueError(
            "subscribe_to_changes does not support multi-kind "
            "subscription via a single fnmatch pattern. Call "
            "subscribe_to_changes(kinds={kind}, ...) once per kind."
        )

    async def subscribe_to_changes(
        self,
        bus: EventBus,
        *,
        kinds: Iterable[KnowledgeKeyKind] | None = None,
        domain_id: str | None = None,
        handler: Callable[[Event], Awaitable[None]],
    ) -> Subscription:
        """Subscribe ``handler`` to content-key change events on ``bus``.

        Composition convenience: wraps this backend's :meth:`key_pattern`
        with :meth:`EventBus.subscribe`.

        Args:
            bus: the event bus the external source publishes to.
            kinds: which :class:`KnowledgeKeyKind` values to subscribe
                to. Default ``None`` resolves to
                ``frozenset({KnowledgeKeyKind.CONTENT})`` — the
                load-bearing intent (observe consumer writes, skip
                DK-managed state writes). Consumers auditing state
                changes opt in by passing ``{METADATA}`` / ``{SNAPSHOT}``.
            domain_id: scope to a single domain. Default ``None`` matches
                every domain.
            handler: async callback invoked on every matching event.

        Returns the :class:`~dataknobs_common.events.Subscription`
        handle; the consumer awaits ``sub.cancel()`` to tear down. See
        :meth:`changes_subscription` for an async-context-manager variant.
        """
        resolved_kinds = kinds or frozenset({KnowledgeKeyKind.CONTENT})
        pattern = self._kind_to_topic_pattern(resolved_kinds, domain_id)
        # The key pattern IS the subscription topic here: passed as the
        # ``topic`` positional (what to match) AND as ``pattern=`` (which
        # engages the bus's wildcard/fnmatch matching rather than an
        # exact-topic match). The deliberate double-pass is required —
        # without ``pattern=`` the bus would treat the wildcard string as
        # a literal topic and never match a concrete published key.
        return await bus.subscribe(pattern, handler, pattern=pattern)

    @asynccontextmanager
    async def changes_subscription(
        self,
        bus: EventBus,
        *,
        kinds: Iterable[KnowledgeKeyKind] | None = None,
        domain_id: str | None = None,
        handler: Callable[[Event], Awaitable[None]],
    ) -> AsyncIterator[Subscription]:
        """Async context manager wrapping :meth:`subscribe_to_changes`.

        Subscribes on entry and cancels on exit (even when the body
        raises)::

            async with backend.changes_subscription(
                bus, kinds={KnowledgeKeyKind.CONTENT}, handler=my_handler,
            ) as sub:
                ...  # subscriber is live
            # subscriber torn down
        """
        sub = await self.subscribe_to_changes(
            bus,
            kinds=kinds,
            domain_id=domain_id,
            handler=handler,
        )
        try:
            yield sub
        finally:
            await sub.cancel()

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

import hashlib
from typing import TYPE_CHECKING, ClassVar

from dataknobs_common.capabilities import CapabilityLike, CapabilityMixin

from .key_layout import KnowledgeKeyKind
from .models import ChangeSet, InvalidVersionError

if TYPE_CHECKING:
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
    """

    # --- Layout constants (consumed by classify_key + each backend) ---

    METADATA_FILE: ClassVar[str] = "_metadata.json"
    CONTENT_DIR: ClassVar[str] = "content"
    SNAPSHOTS_DIR: ClassVar[str] = "_snapshots"

    # --- Capability declaration (declaration-only today) ---
    #
    # Backends inherit :class:`CapabilityMixin` via this mixin so the
    # capability-contract surface is present uniformly. The declared
    # set is empty today: per-backend capability widening
    # (``STREAMING_READS`` on backends that implement ``stream_file``;
    # ``CHANGE_SUBSCRIPTION`` / ``EVENT_BUS_EMISSION`` /
    # ``KEY_PATTERN_FILTERING`` once subscribe/emit surfaces ship;
    # ``TENANT_SCOPED_STATE`` once ``set_ingestion_status`` /
    # ``get_checksum`` / ``has_changes_since`` are tenant-scoped at
    # the contract layer) lands incrementally as each capability's
    # underlying behaviour is implemented, rather than being declared
    # speculatively here. Adopters checking ``backend.supports(...)``
    # for any specific capability get the honest "not advertised"
    # answer today; this preserves the capability identifiers'
    # meaning (advertised ⇒ the contract guarantees the behaviour,
    # not just that the method exists).
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset()

    # --- Required of any backend (supplied by the concrete class) ---

    async def get_info(self, domain_id: str) -> KnowledgeBaseInfo | None:
        """Return KB metadata, or ``None`` if it does not exist."""
        raise NotImplementedError  # pragma: no cover - overridden by backends

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[KnowledgeFile]:
        """Return all files in the KB (used to compute the snapshot)."""
        raise NotImplementedError  # pragma: no cover - overridden by backends

    # --- Canonical change-detection algorithm (shared) ---

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

    async def get_checksum(self, domain_id: str) -> str:
        """Canonical content-snapshot identity of the whole KB.

        MD5 over the sorted ``path:checksum`` of every file. The empty
        KB has identity ``""``. This value *is* the version: capture it,
        pass it back to :meth:`has_changes_since` /
        :meth:`list_changes_since`.

        Raises:
            ValueError: If ``domain_id`` does not exist.
        """
        if await self.get_info(domain_id) is None:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")
        return self._snapshot_identity(await self.list_files(domain_id))

    async def list_changes_since(
        self, domain_id: str, version: str
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
        if await self.get_info(domain_id) is None:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")
        files = await self.list_files(domain_id)
        current_version = self._snapshot_identity(files)
        current = {f.path: f for f in files}
        if version == current_version:
            return ChangeSet(
                added=[], modified=[], deleted=[], version=current_version
            )

        snapshot = await self._load_snapshot(domain_id, version)
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

    async def has_changes_since(self, domain_id: str, version: str) -> bool:
        """``True`` if the KB differs from the snapshot at ``version``.

        The degenerate case of :meth:`list_changes_since`:
        one algorithm, no sibling to drift. An unresolvable version is
        treated as "assume changed" so callers safely re-ingest.

        Raises:
            ValueError: If ``domain_id`` does not exist.
        """
        try:
            return not (
                await self.list_changes_since(domain_id, version)
            ).is_empty
        except InvalidVersionError:
            return True

    async def _load_snapshot(
        self, domain_id: str, version: str
    ) -> dict[str, str]:
        """Resolve ``version`` to a ``{path: checksum}`` snapshot map.

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

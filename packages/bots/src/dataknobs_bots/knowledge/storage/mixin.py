"""Shared change-detection core for knowledge resource backends.

`get_checksum` / `has_changes_since` / `list_changes_since` are one
algorithm, not three per-backend reimplementations. Historically each
backend copy-pasted ``return info.version != version`` for
``has_changes_since`` while ``get_checksum`` returned a content-snapshot
MD5 — different value spaces, so the documented
``get_checksum``→``has_changes_since`` pairing always reported "changed"
(RC1; see Items 125+126 Phase 0). This mixin makes the canonical content
snapshot *the* version: ``get_checksum`` produces it, and both
``has_changes_since`` and ``list_changes_since`` derive from it. One
reviewed implementation; every backend inherits correct behaviour.

`_load_snapshot` is the only backend-specific seam. The default
(no stored snapshot ⇒ empty snapshot) is correct but non-minimal: when
the requested version differs from the current one every current file is
reported ``added`` (a full, not delta, re-ingest). ``is_empty`` stays
correct via the version-equality short-circuit, so change *detection* is
right for every backend; minimal *diffs* arrive when a backend overrides
``_load_snapshot`` with a real per-version store (memory does so now;
file/S3 in Phase 3).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .models import ChangeSet, InvalidVersionError

if TYPE_CHECKING:
    from .models import KnowledgeBaseInfo, KnowledgeFile


class KnowledgeResourceBackendMixin:
    """Canonical change detection built on ``list_files()`` + a snapshot.

    Mixed into every in-tree backend. Relies only on the backend's own
    :meth:`get_info` and :meth:`list_files` (both part of the
    ``KnowledgeResourceBackend`` protocol), so it is backend-agnostic.
    """

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
    def _snapshot_identity(files: list[KnowledgeFile]) -> str:
        """Canonical content-snapshot identity of a file list.

        MD5 over the sorted ``path:checksum`` of every file; ``""`` for
        an empty list. Pure function of the supplied files so it can be
        applied to an already-fetched ``list_files()`` result without a
        second backend round-trip.
        """
        if not files:
            return ""
        combined = ":".join(sorted(f"{f.path}:{f.checksum}" for f in files))
        return hashlib.md5(combined.encode()).hexdigest()

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
        (the equality short-circuit — correct for every backend, and the
        RC1 fix). Otherwise the current files are diffed against
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

        The degenerate case of :meth:`list_changes_since` (decision 5):
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

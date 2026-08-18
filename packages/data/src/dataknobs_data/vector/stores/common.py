"""Common base implementation for vector stores."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from dataknobs_common.exceptions import ConcurrencyError
from dataknobs_common.locks import FileLock
from dataknobs_common.structured_config import StructuredConfigConsumer

from ..exceptions import VectorDomainScopeError

from ..types import DistanceMetric
from .config import VectorStoreConfig, VectorStoreTimestampConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    import numpy as np


logger = logging.getLogger(__name__)


def _forced_save_effect(
    current: tuple[int, int, int] | None,
    persisted: tuple[int, int, int] | None,
) -> str:
    """What ``save(force=True)`` is about to cost, for the WARNING log.

    Three outcomes, not two. A file that is *gone* differs from one that
    *changed*: both fail the identity comparison, but only one of them
    has another writer's rows in it. Reporting a loss that did not happen
    sends an operator looking for rows nothing discarded.
    """
    if current == persisted:
        return "The file is unchanged, so nothing was discarded."
    if current is None:
        return "The file is no longer there, so nothing was discarded."
    return (
        "The file changed since this store read or wrote it, so another "
        "writer's rows are being discarded."
    )


def _flush_to_disk(path: str) -> None:
    """Force ``path``'s contents out of the page cache before it is published.

    ``os.replace`` is atomic with respect to *readers*, not with respect
    to power loss: on a journalled filesystem the rename metadata can
    reach the disk while the data it names has not, leaving a truncated
    file that has already replaced a known-good one. The whole point of
    staging is that a failure leaves the previous state intact, and
    without this the guarantee stops at process death.

    Best-effort by design. A filesystem that refuses ``fsync`` on a
    read-only descriptor is not a reason to fail a save that has
    otherwise succeeded.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("Could not fsync %s before publishing it", path)
    finally:
        os.close(fd)


def _flush_directory(published_path: str) -> None:
    """Force the *rename* durable, having forced the contents durable.

    A file's own ``fsync`` says nothing about the directory entry
    pointing at it, so a crash can lose the rename while keeping the
    data. Not available on every platform — Windows cannot open a
    directory as a file — so failure here is expected rather than
    exceptional.
    """
    try:
        fd = os.open(os.path.dirname(published_path) or ".", os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("Could not fsync the directory holding %s", published_path)
    finally:
        os.close(fd)


POST_FILTER_OVERFETCH = 4
"""Candidates fetched per requested row when a post-filter follows.

A backend whose index cannot express the whole filter drops some of
what it fetched, so asking the index for exactly ``k`` guarantees
fewer than ``k`` survivors whenever anything is dropped. Over-fetching
is the compensation, and this is how much of it there is.

The value is a heuristic, not a bound: a filter matching fewer than
one candidate in four still under-returns. Where the number of rows
available to search is known, :meth:`VectorStoreBase._overfetch_sizes`
escalates from here instead of settling for it.
"""


class VectorStoreBase(StructuredConfigConsumer[VectorStoreConfig]):
    """Base implementation with common functionality for all vector stores.

    Constructed through a :class:`VectorStoreConfig` subclass via
    :class:`~dataknobs_common.structured_config.StructuredConfigConsumer`:
    each concrete store declares its leaf ``CONFIG_CLS`` and the typed
    config drives construction. ``store.config`` is the typed config (not
    a dict). This class provides the shared derived-attribute computation
    (in :meth:`_setup`) plus the common similarity / filter / timestamp
    helpers.
    """

    CONFIG_CLS: ClassVar[type[VectorStoreConfig]] = VectorStoreConfig

    def _setup(self) -> None:
        """Derive shared attributes from the typed config.

        Runs once during construction (the mixin calls it after
        ``self._config`` is established). Subclasses override and call
        ``super()._setup()`` first, then compute their backend-specific
        derived attributes. Field parsing already happened in
        :meth:`VectorStoreConfig.from_dict`; this only computes attributes
        that are not pure field storage (metric→enum, path expansion,
        timestamp-key resolution) and initializes shared runtime state.
        """
        cfg = self.config

        self.dimensions = cfg.dimensions

        # Distance metric: keep the string in config, derive the enum here.
        self.metric = (
            cfg.metric if isinstance(cfg.metric, DistanceMetric) else DistanceMetric(cfg.metric)
        )

        # Expand ~ to home directory for persistent storage.
        self.persist_path = Path(cfg.persist_path).expanduser() if cfg.persist_path else None
        self.batch_size = cfg.batch_size

        if cfg.persist_path:
            logger.info(
                "VectorStore persist_path: %s -> %s (exists: %s)",
                cfg.persist_path,
                self.persist_path,
                os.path.exists(self.persist_path) if self.persist_path else False,
            )

        self.index_params = cfg.index_params
        self.search_params = cfg.search_params
        self.metadata = cfg.metadata

        # Config-level multi-tenant scoping. When set, every
        # read/count/clear/update is implicitly scoped to this domain
        # and add_vectors defaults a row's "domain_id" to it. Scoping
        # applies however a row is addressed: the filter-keyed surfaces
        # get it from _effective_filter, the id-keyed ones
        # (get_vectors, delete_vectors, update_metadata) and
        # metadata_fields from _in_configured_domain. This mirrors
        # PgVectorStore's long-standing config-level domain_id
        # behavior; Memory/FAISS/Chroma honor it through the shared
        # helpers so a runtime backend swap preserves isolation
        # semantics. None ⇒ no implicit scoping (prior behavior).
        self.domain_id = cfg.domain_id

        # Timestamp exposure config. All vector stores expose
        # created_at / updated_at metadata via include_timestamps=True
        # on get_vectors() and search(). Where the values live is the
        # backend's business: pgvector has real columns, MVS and FAISS
        # keep a side-car keyed by row, and Chroma — whose only per-row
        # storage is the metadata dict the consumer also owns — keeps
        # them in-band under reserved keys stripped from every read. Format and key names are configurable; defaults are
        # consistent across backends so runtime-swap produces identical
        # metadata surfaces. The format is validated in
        # VectorStoreTimestampConfig.__post_init__.
        ts = cfg.timestamps or VectorStoreTimestampConfig()
        self.timestamps_format: str = ts.format
        self.timestamps_created_key: str = ts.created_key
        self.timestamps_updated_key: str = ts.updated_key

        self._initialized = False
        # Per-instance set of configured timestamp keys for which a
        # collision warning has already been emitted. Lives on the
        # instance (not module scope) so lifetime matches the store —
        # avoids the CPython id() reuse hazard where a new store could
        # inherit a dead store's warning state at the same memory
        # address.
        self._timestamp_collision_warned: set[str] = set()

        # --- Single-file persistence state (file-backed stores only) ---
        # Identity of the persisted file as this instance last saw it —
        # from a load, or from its own most recent save. A save refuses
        # to write over a file that no longer matches, because it
        # serializes this instance's whole in-memory state and would drop
        # every row another instance wrote meanwhile. None until a file
        # has been read or written. See :meth:`_file_identity`.
        self._persisted_identity: tuple[int, int, int] | None = None
        # True when in-memory state has diverged from the persisted file.
        # ``close()`` persists only a dirty store: an instance that only
        # read would otherwise write a snapshot on teardown, which both
        # costs a pointless serialization and — since that write moves
        # the file's identity — makes the *real* writer's later save
        # raise and lose its rows. Set by every mutator, cleared by a
        # successful save or load (:meth:`_stamp_persisted_identity`).
        self._dirty: bool = False
        # Serializes this instance's own saves. The staleness check and
        # the write that follows it are two operations on a worker
        # thread; without this, two overlapping ``save()`` calls on one
        # store (an autosave task racing ``close()``, or a bare
        # ``asyncio.gather``) either both pass the check and race on the
        # file, or one stats a half-written file and raises
        # ``ConcurrencyError`` against itself. Cross-*instance* conflict
        # is what ``_persisted_identity`` detects; this covers the one
        # case a single instance can create on its own.
        self._save_lock = asyncio.Lock()

    def _validate_dimensions(self) -> None:
        """Validate vector dimensions.

        Raises:
            ValueError: If dimensions are invalid
        """
        if self.dimensions <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.dimensions}")
        if self.dimensions > 65536:
            raise ValueError(f"Dimensions {self.dimensions} exceeds maximum (65536)")

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector for cosine similarity.

        Args:
            vector: Vector to normalize

        Returns:
            Normalized vector
        """
        import numpy as np

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate similarity between two vectors based on configured metric.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score
        """
        import numpy as np

        if self.metric == DistanceMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            # Convert distance to similarity
            distance = float(np.linalg.norm(vec1 - vec2))
            return 1.0 / (1.0 + distance)

        elif self.metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            # Dot product
            return float(np.dot(vec1, vec2))

        elif self.metric == DistanceMetric.L1:
            # Manhattan distance to similarity
            distance = np.sum(np.abs(vec1 - vec2))
            return 1.0 / (1.0 + distance)

        else:
            # Default to cosine
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _convert_distance_to_score(self, distance: float) -> float:
        """Convert a distance to a similarity score based on metric.

        Args:
            distance: Distance value

        Returns:
            Similarity score (higher is more similar)
        """
        if self.metric == DistanceMetric.COSINE:
            # Cosine distance is 1 - similarity
            return 1.0 - distance
        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            # Convert distance to similarity
            return 1.0 / (1.0 + distance)
        elif self.metric == DistanceMetric.L1:
            # Manhattan distance to similarity
            return 1.0 / (1.0 + distance)
        else:
            # For dot product and others, higher is better
            return distance

    def _prepare_vector(
        self, vector: np.ndarray | list[float] | list[np.ndarray], normalize: bool = False
    ) -> np.ndarray:
        """Prepare a vector for storage or search.

        Args:
            vector: Input vector (numpy array, list of floats, or list of arrays)
            normalize: Whether to normalize for cosine similarity

        Returns:
            Prepared numpy array
        """
        import numpy as np

        # Convert to numpy array
        if isinstance(vector, list):
            if len(vector) > 0 and isinstance(vector[0], np.ndarray):
                # List of arrays - stack them
                vector = np.vstack(vector).astype(np.float32)
            else:
                # List of floats
                vector = np.array(vector, dtype=np.float32)
        else:
            vector = np.asarray(vector, dtype=np.float32)

        # Ensure vector is an ndarray at this point
        assert isinstance(vector, np.ndarray)

        # Ensure correct shape
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Normalize if needed (e.g., for cosine similarity)
        if normalize or self.metric == DistanceMetric.COSINE:
            # Apply normalization for cosine similarity
            norms = np.linalg.norm(vector, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vector = vector / norms

        return cast("np.ndarray", vector)

    def _effective_filter(self, filter: dict[str, Any] | None) -> dict[str, Any] | None:
        """AND-merge the config-level ``domain_id`` scope into ``filter``.

        When no config ``domain_id`` is set this is the identity
        function (prior behavior — ``None`` stays ``None``). When set,
        the returned filter additionally constrains ``domain_id`` to
        the intersection of the configured scope and any caller-supplied
        ``domain_id`` constraint:

        * caller did not constrain ``domain_id`` → require
          ``domain_id == self.domain_id``;
        * caller asked for the same scalar (or a list containing it) →
          require ``domain_id == self.domain_id`` (no widening);
        * caller asked for a *different* domain (scalar or a list not
          containing the configured scope) → an **empty-list** filter
          value, which ``_match_metadata_filter`` can never satisfy, so
          the result set is empty.

        The in-scope case *collapses* rather than intersecting, which is
        visible only for a row belonging to several domains: a store
        scoped to ``t1`` hands back a row reading ``["t1", "t2"]`` and
        then answers ``0`` for ``{"domain_id": "t2"}``. Recorded as a
        follow-up rather than settled here, because the two readings —
        the scope as a ceiling on what a filter may ask for, or as one
        more constraint ANDed with it — are a semantics choice and not a
        defect in either direction.

        What is *not* an argument for the collapse is parity with
        ``PgVectorStore``, which this docstring used to claim. That
        backend keeps ``domain_id`` in a column and stores caller
        metadata JSONB verbatim, so an explicit ``domain_id`` filter is
        a containment probe against a key the column consumed: it
        answers ``0`` for the configured scope too, not merely for a
        cross-domain one. The behaviours differ in both directions, and
        the note above ``domain_scoped_store`` in the filter-semantics
        suite is where that divergence is described.

        Callers pass the result straight to ``_match_metadata_filter`` /
        the filtered count/clear/update paths; a returned dict is never
        ``None`` when scoping is active, so ``filter is None`` fast
        paths must consult this first.
        """
        if not self._is_scoped:
            return filter

        eff: dict[str, Any] = dict(filter) if filter else {}
        if "domain_id" not in eff:
            eff["domain_id"] = self.domain_id
            return eff

        caller = eff["domain_id"]
        if isinstance(caller, list):
            in_scope = self.domain_id in caller
        else:
            in_scope = caller == self.domain_id
        # In scope ⇒ collapse to the configured scope (no widening).
        # Out of scope ⇒ unsatisfiable empty-list value.
        eff["domain_id"] = self.domain_id if in_scope else []
        return eff

    def _apply_domain_default(
        self,
        metadata: list[dict[str, Any]] | None,
        count: int,
        *,
        ids: Sequence[str] | None = None,
        stored: Mapping[str, dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Return per-row metadata with ``domain_id`` defaulted.

        Mirrors ``PgVectorStore``'s ``meta.get("domain_id",
        self.domain_id)`` write-path default: when a config-level
        ``domain_id`` is set, any row whose metadata omits
        ``domain_id`` is tagged with the configured scope so the
        read-side ``_effective_filter`` can find it. Returns fresh
        per-row dicts (never mutates or aliases the caller's). A no-op
        passthrough (still copied) when no scope is configured.

        The copy is deep, and the sentence above is why it has to be: it
        already promised the result does not alias the caller's, and a
        ``dict(...)`` made that true of the outer dict alone. Every value
        inside stayed shared with whatever the caller went on holding.

        *ids* and *stored* let the default preserve the row's **own**
        scope rather than re-stamping the configured one. The rule being
        implemented is "a caller who does not mention ``domain_id`` must
        not change it", and re-stamping the configured scalar implements
        that only where the stored scope already *is* that scalar. A row
        tagged ``["t1", "t2"]`` belongs to both domains under the
        four-quadrant rule, so a ``t1`` store's write is admitted — and
        re-stamping ``"t1"`` over it silently evicted ``t2``.

        Passing them is what makes the write path agree with the guard
        that admitted it: anything still in *stored* has already been
        checked by :meth:`_reject_out_of_scope_ids`, so carrying its
        value forward cannot widen the row into a domain the caller does
        not hold. Omit them and the prior scalar behaviour stands, which
        is correct wherever a stored list cannot arise — ``PgVectorStore``
        keeps ``domain_id`` in a scalar column and does its own default
        inline.
        """
        rows = (
            [
                (self._copy_metadata(metadata[i]) or {}) if i < len(metadata) else {}
                for i in range(count)
            ]
            if metadata is not None
            else [{} for _ in range(count)]
        )
        if self._is_scoped:
            for i, row in enumerate(rows):
                if "domain_id" not in row:
                    row["domain_id"] = self._preserved_domain(ids, stored, i)
        return rows

    def _preserved_domain(
        self,
        ids: Sequence[str] | None,
        stored: Mapping[str, dict[str, Any] | None] | None,
        index: int,
    ) -> Any:
        """The ``domain_id`` a silent row should keep, or the config default.

        Returns the value already stored for the row when there is one,
        so a write that never mentions ``domain_id`` leaves it alone.
        Falls back to the configured scope for a genuine insert, for a
        stored row that carries no scope of its own, and for callers
        that do not supply the stored map at all.
        """
        if ids is None or stored is None or index >= len(ids):
            return self.domain_id
        prior = stored.get(ids[index])
        if not prior:
            return self.domain_id
        existing = prior.get("domain_id")
        return self.domain_id if existing is None else existing

    @staticmethod
    def _is_empty_batch(vectors: Any) -> bool:
        """Whether an ``add_vectors`` batch carries no rows.

        An empty batch is something a caller *produces* rather than
        intends — a comprehension that filtered everything out, a
        chunker handed a blank document — so it is a no-op, not an
        error. Each backend guards with this and returns ``[]``.

        Both spellings count: ``[]`` and ``np.array([])`` are the same
        intent, and a caller assembling a batch numerically produces the
        latter. ``len`` answers for both, and for a well-formed 2-D
        batch it is the row count, so a single un-nested vector
        (``ndim == 1``, ``len == dimensions``) is correctly *not* empty.

        A 0-d input — ``np.array(5.0)``, ``np.float32(1.0)`` — has no
        length and is not a batch in any reading, empty or otherwise.
        It answers ``False`` rather than raising, so that the caller
        sees the backend's dimension error, which can say what shape was
        expected, instead of a bare ``len() of unsized object`` coming
        out of an emptiness check. A predicate that raises on part of
        its input domain is not one a guard can be built from.

        Left to each backend rather than hoisted into a concrete
        ``add_vectors`` on the base: that method is abstract, and giving
        it a body would rename the abstract half out from under every
        out-of-tree store.
        """
        try:
            return len(vectors) == 0
        except TypeError:
            return False

    @property
    def _is_scoped(self) -> bool:
        """Whether a configured domain scope is in force.

        One test, because ``domain_id`` is optional rather than
        truthy-optional and the two spellings disagree on ``""``.
        ``PgVectorStore`` guarded its column predicates with
        ``if not self.domain_id`` while the metadata-carrying backends
        used ``is None``, so a store configured with an empty-string
        domain isolated on three backends and ran unscoped on the
        fourth — a tenant boundary that disappeared on a config-selected
        backend swap, which is the one thing this scope exists to
        survive.
        """
        return self.domain_id is not None

    def _in_configured_domain(self, meta: dict[str, Any] | None) -> bool:
        """Whether a stored row falls inside the configured scope.

        The id-keyed half of what :meth:`_effective_filter` does for the
        filter-keyed half. ``get_vectors``, ``delete_vectors`` and
        ``update_metadata`` address rows by id, so no filter is built and
        that helper never runs — which left the configured scope binding
        only the surfaces that happen to take a ``filter``. Isolation
        that depends on *how* a caller asks is not isolation: ids are
        routinely derived from content and are guessable.

        ``metadata_fields`` uses it too. Field names are data — the union
        over every stored row discloses the shape of a neighbouring
        domain's metadata without returning any of its rows.

        Identity when no scope is configured, so an unscoped store keeps
        its prior behaviour exactly. Backends whose ``domain_id`` is a
        real column (``PgVectorStore``) express the same predicate in
        SQL instead of calling this.

        Delegates to :meth:`_match_metadata_filter` rather than
        comparing with ``==``: ``domain_id`` is an ordinary metadata key
        on the backends that carry it in metadata, so the four-quadrant
        rule applies to it like any other, and a scalar scope against a
        list value is *membership*. Comparing directly read a row tagged
        ``["t1", "t2"]`` as belonging to neither domain, while the
        filter-keyed half — which resolves the same scope through that
        method — read it as belonging to both. One evaluator is what
        keeps the two halves from answering differently about the same
        row; the split had ``count()`` reporting a row that
        ``get_vectors`` called absent and ``delete_vectors`` refused,
        which ``clear()`` then removed anyway.
        """
        if not self._is_scoped:
            return True
        return self._match_metadata_filter(meta, {"domain_id": self.domain_id})

    def _reject_out_of_scope_ids(self, stored: Mapping[str, dict[str, Any] | None]) -> None:
        """Fail closed when a write would capture out-of-domain rows.

        The write-side counterpart of :meth:`_in_configured_domain`.
        ``add_vectors`` and ``add_documents`` are id-keyed like the read
        verbs — this store upserts on id conflict — but they cannot
        answer "absent" the way a read does. The row they would write
        carries the configured scope (``_apply_domain_default`` puts it
        there), so writing an id another domain owns does not insert
        alongside it or edit it: it takes it. On pgvector the capture is
        explicit in the SQL, whose ``ON CONFLICT`` clause assigns
        ``domain_id`` from the incoming row.

        *stored* maps id to the metadata already held, for the subset of
        the batch that already exists — an id with no stored row is a
        genuine insert and is not the caller's to lose. Backends whose
        ``domain_id`` is a column pass a synthesized ``{"domain_id":
        ...}`` so one predicate decides for every backend.

        Raises before anything is written, so a rejected batch leaves no
        partial state on the backends that have no transaction to roll
        back.
        """
        if not self._is_scoped:
            return
        foreign = sorted(
            rid for rid, meta in stored.items() if not self._in_configured_domain(meta)
        )
        if foreign:
            # ``_is_scoped`` is exactly "domain_id is not None", but it
            # is a property, so the narrowing does not carry here.
            raise VectorDomainScopeError(foreign, cast("str", self.domain_id))

    # ------------------------------------------------------------------
    # Single-file persistence: dirty tracking, identity, atomic publish.
    #
    # Shared by every store that persists by serializing its whole
    # in-memory state over one file — ``MemoryVectorStore`` and
    # ``FaissVectorStore`` today. Nothing here is backend-specific: the
    # hazard is the *whole-state rewrite*, not the format written, so a
    # store that adopts that persistence shape inherits the hazard and
    # should adopt these with it. Stores whose backing service handles
    # its own concurrency (``chroma``, ``pgvector``) never touch them.
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        """Record that in-memory state has diverged from the file.

        Every mutator calls this. A store that has not is byte-identical
        to what is on disk, so ``close()`` can skip persisting it — see
        ``_dirty`` in :meth:`_setup` for why that matters beyond the
        wasted write.
        """
        self._dirty = True

    @staticmethod
    def _file_identity(path: str | os.PathLike[str]) -> tuple[int, int, int] | None:
        """Best-effort identity of the file at ``path``, ``None`` if absent.

        Modification time, size and inode. Explicitly *best-effort*, not
        a collision-free fingerprint:

        * ``st_ino`` discriminates only a replace-by-rename or a
          delete-and-recreate. A writer that truncates the file in place
          leaves it unchanged.
        * ``st_mtime_ns`` is nanosecond-granular on APFS and modern
          ext4, but one **second** on HFS+, ext3, and many network and
          overlay mounts. Two writes inside one tick are indistinguishable
          there.
        * ``st_size`` repeats whenever two snapshots serialize to the
          same length — two indexes with the same row count and
          dimension, for instance.

        So this detects the overwhelmingly common accident (two live
        instances over one path) and does not pretend to be a lock. A
        content hash would be exact and is deliberately not used: this
        runs before every save, and a file large enough for the race to
        matter is a file too large to re-read to check.

        Returns ``None`` for any ``OSError`` — an absent file, but also
        an unreadable directory. That is safe rather than lenient: the
        write that follows fails on its own and reports the real errno,
        which is a better error than one about concurrent writers.

        Blocking ``stat``; every caller already runs on a worker thread.
        """
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size, stat.st_ino)

    def _guard_persisted_identity(self, path: str | os.PathLike[str], *, force: bool) -> None:
        """Refuse to overwrite a file that changed since this store saw it.

        A whole-state rewrite replaces the file with this instance's
        snapshot, so proceeding over another instance's write discards
        every row that instance persisted — total and silent, for the
        writer that got there first.

        ``force`` skips the check. It is the deliberate way out of a
        refusal, and it accepts the loss — so it is logged at WARNING
        every time it is used, per ``rules/security.md`` §8 and because
        it is the line an operator wants when asking where the rows
        went. The identity is read even under ``force``, at the cost of
        one ``stat`` before a whole-state serialization, so the log can
        say whether anything was actually discarded.
        """
        current = self._file_identity(path)
        if force:
            logger.warning(
                "%s: save(force=True) bypassed the staleness check on %s. %s",
                type(self).__name__,
                path,
                _forced_save_effect(current, self._persisted_identity),
            )
            return
        if current == self._persisted_identity:
            return
        raise ConcurrencyError(
            f"{type(self).__name__}: the persisted file changed since this "
            "store read or wrote it, so saving would replace another "
            "writer's rows with this instance's snapshot. A persist_path is "
            "single-writer: give each store instance its own path, keep "
            "their lifetimes sequential, or use a backend that supports "
            "concurrent writers, such as pgvector. To overwrite anyway — "
            "accepting the loss of whatever the other writer persisted — "
            "call save(force=True).",
            context={
                "persist_path": str(path),
                "loaded": self._persisted_identity is not None,
                "exists_now": current is not None,
            },
        )

    def _refresh_persisted_identity(self, path: str | os.PathLike[str]) -> None:
        """Record ``path``'s identity without declaring the store saved.

        Split out from :meth:`_stamp_persisted_identity` because the two
        facts that method maintains come apart on exactly one path. A
        publish that fails partway can still have replaced the tracked
        file, which makes this instance its last writer — but nothing was
        persisted, so the store is still dirty. Stamping there would tell
        ``close()`` it had nothing left to write, which is how a failed
        save turns into a silent one.
        """
        self._persisted_identity = self._file_identity(path)

    def _stamp_persisted_identity(self, path: str | os.PathLike[str]) -> None:
        """Record ``path`` as the state this instance is now in step with.

        Called after a successful load and after a successful save, which
        are exactly the two moments memory and disk agree. Clearing
        ``_dirty`` here rather than at each call site is what keeps the
        two facts from drifting apart.
        """
        self._refresh_persisted_identity(path)
        self._dirty = False

    @contextlib.contextmanager
    def _persisted_save(self, path: str, *, force: bool) -> Iterator[None]:
        """Hold ``path``'s file lock across staleness check → write → stamp.

        The check and the write it guards are two operations with a
        scheduling point between them: both run on a worker thread, with
        a whole serialization in between. Without a lock spanning them,
        two instances over one path both pass the check before either
        writes, and the second replaces the first's file with a snapshot
        that never saw its rows — **and neither raises**, because the
        refusal that exists to prevent exactly that was passed by both
        while the file still looked untouched.

        ``_save_lock`` cannot close this. It is an ``asyncio.Lock`` on
        one instance, and the conflicting writer is a *different*
        instance, in this process or another. :class:`FileLock` covers
        both — it is the same lock ``AsyncFileDatabase`` already takes
        for the same hazard on the same kind of single-file store.

        The bracket lives here rather than in each store because the
        sequence is identical in both, and a third store adopting
        ``persist_path`` should inherit it rather than rediscover it.
        That is also why the directory is created here. It was written
        out in each store, and it is a *precondition of the lock* — the
        lockfile is a sibling of the target, so its directory has to
        exist before ``FileLock`` can open it. A third store inheriting
        the bracket without it would get ``FileNotFoundError`` out of the
        lock on its first save.
        """
        # ``os.path.dirname`` is "" for a bare filename (no directory
        # component), and ``makedirs("")`` raises FileNotFoundError.
        # The guard is why each store had this line; it survives the
        # move rather than being rediscovered a third time.
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with FileLock(path):
            self._sweep_orphaned_scratch(path)
            self._guard_persisted_identity(path, force=force)
            yield
            self._stamp_persisted_identity(path)

    @contextlib.contextmanager
    def _persisted_load(self, path: str) -> Iterator[bool]:
        """Hold ``path``'s file lock across the read and the stamp after it.

        Yields whether the file exists. A body told ``False`` must not
        read, and nothing is stamped in that case — a store that read
        nothing is in step with nothing.

        The lock is not only about this instance's own save.
        :meth:`_stamp_persisted_identity` writes two fields the save path
        owns, so a load running inside a save can declare the store in
        step with a file the save has not written yet, after which
        ``close()`` skips persisting a mutation nobody wrote. And a store
        that publishes two files — an index beside its side-car — renames
        them one after the other, so an unlocked reader landing between
        the two renames gets a new index with a stale side-car.
        """
        if not Path(path).parent.is_dir():
            # A ``persist_path`` under a directory that does not exist
            # yet is the ordinary first-run shape: ``save()`` creates the
            # directory, ``load()`` finds nothing. Taking the lock first
            # would mean opening a lockfile in a directory that is not
            # there and failing the load outright.
            yield False
            return

        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(FileLock(path))
            except OSError as exc:
                # Taking the lock means creating or opening
                # ``<path>.lock``, so a directory this process cannot
                # write is a directory it cannot lock — an index baked
                # into a read-only image layer, or served from a
                # read-only mount. Failing here would refuse a load that
                # worked before the lock existed.
                #
                # Degrading is sound rather than merely convenient, and
                # only on this side: publishing is ``os.replace`` into
                # this same directory, so a writer to exclude cannot
                # exist here either. ``_persisted_save`` keeps the hard
                # lock, because there the write *is* the thing that
                # needs excluding.
                logger.warning(
                    "%s: could not take the persist lock on %s (%s). Reading "
                    "without it: nothing can be published into a directory "
                    "this process cannot write, so there is no concurrent "
                    "writer to exclude.",
                    type(self).__name__,
                    path,
                    exc,
                )
            exists = Path(path).exists()
            yield exists
            if exists:
                self._stamp_persisted_identity(path)

    def _write_then_publish(
        self,
        writes: list[tuple[str, Callable[[str], None]]],
    ) -> None:
        """Write each file to a scratch sibling, then rename them into place.

        Each ``(final_path, write)`` pair is written to a uniquely named
        scratch file in ``final_path``'s own directory; only once *every*
        write has succeeded are the renames performed. A write that fails — out of space, permissions, a
        pickle that will not serialize — therefore leaves the existing
        files untouched, instead of replacing one of them and abandoning
        the caller with a half-updated pair whose two halves describe
        different corpora.

        This is not a transaction across several files. ``os.replace`` is
        atomic per file, so a crash *between* two renames still leaves a
        new file beside an old sibling — a concurrent *reader* is covered
        by :meth:`_persisted_load` taking the same lock the publish is
        under, but a crash is not. Making that impossible needs a
        single-file format or a write-ahead log, neither of which this
        change introduces. What it does remove is the far likelier
        failure: an error raised on the second write after the first has
        already overwritten its target.

        What it must not do is leave the store unable to try again. A
        rename that fails after an earlier one landed makes this instance
        the tracked file's last writer, while the caller never reaches
        its stamp — so the identity check would go on comparing against
        the file this store had itself replaced and refuse every later
        save, naming a conflicting writer that does not exist. The only
        way out of that would be ``save(force=True)``, whose whole
        purpose is discarding somebody else's rows; using it to recover
        from a self-inflicted failure is not a recovery. The identity is
        therefore refreshed on the way out, while ``_dirty`` stays set
        because nothing was persisted.

        Each scratch file gets a name of its own. A fixed
        ``<final>.tmp`` sibling is shared by every writer of that path,
        so two of them stage over each other's bytes and the loser's
        cleanup can unlink a file the winner is about to rename —
        turning a silent clobber into a spurious ``FileNotFoundError``.
        The cost of unique names is that a process killed mid-save
        leaves its scratch file behind instead of having it overwritten
        by the next save; one stray file is the better failure.

        Permissions follow the file being replaced, where there is one.
        ``mkstemp`` creates owner-only, and publishing that over an
        existing file would silently narrow whatever mode the consumer
        had set — as the previous fixed-name scratch silently *widened*
        it to the umask default on every save.

        Every scratch file is removed on any failure, including one
        whose ``write`` raised — which is the likeliest failure of the
        three the paragraph above names, and the one a list built from
        *successful* writes cannot clean up. Blocking I/O; callers run on
        a worker thread.
        """
        # Appended before ``write`` rather than after it, because the
        # write is what fails. A published scratch no longer exists, so
        # the unlink below finds nothing and suppresses the ENOENT;
        # tracking the two cases apart would buy nothing.
        created: list[str] = []
        staged: list[tuple[str, str]] = []
        published: list[str] = []
        try:
            for final_path, write in writes:
                tmp = self._scratch_sibling(final_path)
                created.append(tmp)
                write(tmp)
                _flush_to_disk(tmp)
                self._carry_mode(tmp, final_path)
                staged.append((tmp, final_path))
            for tmp, final_path in staged:
                os.replace(tmp, final_path)
                published.append(final_path)
            staged.clear()
            if published:
                # One entry per directory, and in practice one directory:
                # a store's files are siblings, and the rename is only
                # atomic because the scratch is too.
                _flush_directory(published[0])
        finally:
            # Emptied above exactly when every rename landed, so a
            # non-empty ``staged`` here *is* the failure signal.
            failed = bool(staged)
            for tmp in created:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
            if failed and self.persist_path is not None and str(self.persist_path) in published:
                self._refresh_persisted_identity(str(self.persist_path))

    @staticmethod
    def _canonical_persist_path(path: str | os.PathLike[str]) -> str:
        """Resolve ``path`` to the file it actually names.

        A ``persist_path`` is often a stable name pointing at versioned
        storage, and publishing is ``os.replace`` — which replaces the
        *symlink* rather than writing through it. The first save then
        turns the alias into a regular file holding this store's
        snapshot, while the versioned file it used to point at keeps the
        old one, and nothing says so.

        Resolving here fixes that and keeps the lock honest. ``FileLock``
        takes its lockfile beside the resolved target so two spellings of
        one file contend; a save that destroys the symlink would move the
        lockfile out from under that agreement after exactly one write,
        leaving two writers serialized until the moment it stopped
        mattering.

        Every derived path follows from the result, which is why the
        stores resolve once at the top rather than each consumer
        resolving its own: FAISS's ``.meta`` side-car is not itself a
        symlink, so deriving it from an unresolved path would strand it
        in a different directory from the index it describes.

        Non-strict: a path whose directory does not exist yet resolves to
        itself, which is the ordinary first-run shape.
        """
        return os.path.realpath(path)

    @staticmethod
    def _sweep_orphaned_scratch(final_path: str) -> None:
        """Remove scratch siblings of ``final_path`` left by a dead writer.

        Called with the file lock held, which is what makes it safe: a
        live writer publishing this target holds that lock, so every
        ``<name>.*.tmp`` visible here belongs to a process that died
        between creating one and renaming it.

        Unique scratch names made this necessary. The previous fixed
        ``<name>.tmp`` was overwritten by the next save, so a hard kill
        cost one stray file forever; unique names cost one *per kill*,
        which is unbounded over a process that is restarted.
        """
        target = Path(final_path)
        try:
            orphans = list(target.parent.glob(target.name + ".*.tmp"))
        except OSError:
            return
        for orphan in orphans:
            with contextlib.suppress(OSError):
                orphan.unlink()
            logger.debug("Removed an orphaned scratch file: %s", orphan)

    @staticmethod
    def _scratch_sibling(final_path: str) -> str:
        """An empty, uniquely named file beside ``final_path``.

        Beside it rather than in a temp directory so the publishing
        ``os.replace`` stays within one filesystem, which is what makes
        it atomic.
        """
        target = Path(final_path)
        fd, tmp = tempfile.mkstemp(
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".tmp",
        )
        os.close(fd)
        return tmp

    @staticmethod
    def _carry_mode(scratch: str, final_path: str) -> None:
        """Give ``scratch`` the permissions of the file it will replace.

        A no-op when there is nothing to replace: a file created here
        keeps ``mkstemp``'s owner-only mode, which is the right default
        for a store's own data and the one thing this cannot inherit.

        Only the permission bits carry. ``setgid`` and the sticky bit are
        masked off deliberately: they are rarely meaningful on a regular
        file, and reproducing ``setgid`` from a file this process may not
        own is the kind of thing a save should not be doing silently. A
        directory relying on ``setgid`` for group inheritance is
        unaffected — that bit lives on the directory, not on the file
        being replaced.
        """
        try:
            mode = Path(final_path).stat().st_mode & 0o777
        except OSError:
            return
        with contextlib.suppress(OSError):
            Path(scratch).chmod(mode)

    def _overfetch_sizes(
        self,
        k: int,
        *,
        has_post_filter: bool,
        ceiling: int | None = None,
    ) -> Iterator[int]:
        """Yield the sizes a search should fetch to still return ``k`` rows.

        Every candidate a post-filter drops is a row the caller asked
        for and does not get, so a backend that filters after its index
        has already truncated to ``k`` must ask for more than ``k``.
        This is the single place that decides how much more.

        Without a post-filter the sequence is just ``k``: the index's
        own truncation is already exact and over-fetching is waste.

        With one, the first size is ``k * POST_FILTER_OVERFETCH``. That
        alone is a heuristic and not a bound: a filter matching fewer
        than one candidate in ``POST_FILTER_OVERFETCH`` still
        under-returns. A caller that can say how many rows exist to
        search passes ``ceiling``; the sequence then doubles from there,
        capped at ``ceiling`` and ending on it, so the caller keeps
        asking until enough rows survive its filter or the corpus is
        exhausted — at which point the answer is exact rather than
        merely over-fetched. A caller that cannot cheaply establish that
        bound omits ``ceiling`` and gets the single over-fetched size,
        keeping the heuristic's shortfall.

        Args:
            k: Rows the caller asked for.
            has_post_filter: Whether candidates are dropped after the
                index returns them.
            ceiling: Rows available to fetch, when the caller knows.

        Yields:
            Fetch sizes in increasing order. Callers stop as soon as
            enough rows survive their filter.
        """
        if not has_post_filter:
            yield k if ceiling is None else min(k, ceiling)
            return

        fetch = k * POST_FILTER_OVERFETCH
        if ceiling is None:
            yield fetch
            return

        while True:
            capped = min(fetch, ceiling)
            yield capped
            if capped >= ceiling:
                return
            # ``capped + 1`` keeps a non-positive ``k`` from stalling the
            # escalation at zero forever.
            fetch = max(capped * 2, capped + 1)

    def _match_metadata_filter(
        self,
        metadata: dict[str, Any] | None,
        filter: dict[str, Any],
    ) -> bool:
        """Check whether a record's metadata satisfies every filter key.

        Per-key semantics:

        * ``scalar`` filter, ``scalar`` metadata — equality.
        * ``scalar`` filter, ``list`` metadata — membership (is the
          scalar in the list?).
        * ``list`` filter, ``scalar`` metadata — IN (is the scalar any
          filter element?).
        * ``list`` filter, ``list`` metadata — non-empty intersection.

        A missing metadata key fails the filter (``None`` is treated as
        absence). All keys must match (AND across keys). An empty
        filter dict matches everything.

        Empty-list contract: an empty-list filter value is unsatisfiable —
        ``{key: []}`` matches no record on any backend (here, neither the
        list/list intersection nor the list/scalar IN branch can succeed
        against ``[]``). Backends that translate filters natively (chroma,
        pgvector) MUST preserve this; consumers (e.g.
        ``VectorMemory.clear()`` and :meth:`_effective_filter`) rely on it
        to express a deliberate no-op / unsatisfiable cross-tenant
        request.

        Elements of list filter values and list metadata values must be
        hashable. Nested dicts or lists are unsupported; consumers
        storing such values should compose a separate filter source.
        A ``TypeError`` from ``set()`` propagates as caller error.
        """
        if not filter:
            return True
        if metadata is None:
            return False
        for key, filter_val in filter.items():
            meta_val = metadata.get(key)
            if meta_val is None:
                return False
            filter_is_list = isinstance(filter_val, list)
            meta_is_list = isinstance(meta_val, list)
            if filter_is_list and meta_is_list:
                if not set(filter_val).intersection(meta_val):
                    return False
            elif filter_is_list:
                if meta_val not in filter_val:
                    return False
            elif meta_is_list:
                if filter_val not in meta_val:
                    return False
            else:
                if meta_val != filter_val:
                    return False
        return True

    def _apply_metadata_filter(
        self,
        candidates: list[tuple[Any, dict]],
        filter: dict[str, Any],
    ) -> list[tuple[Any, dict]]:
        """Apply metadata filter to (id, metadata) candidate tuples.

        Delegates to ``_match_metadata_filter`` for the per-record
        decision. Retained as a separate method because the filter +
        candidate-list shape is convenient for post-hoc filtering paths.
        """
        if not filter:
            return candidates
        return [
            (item_id, metadata)
            for item_id, metadata in candidates
            if self._match_metadata_filter(metadata, filter)
        ]

    def _update_metadata_where_filtered(
        self,
        metadata_items: Iterable[tuple[Any, dict[str, Any]]],
        timestamps: dict[Any, tuple[datetime, datetime]] | None,
        filter: dict[str, Any] | None,
        set_: dict[str, Any],
    ) -> int:
        """Shared post-filter + in-place merge for in-process backends.

        The byte-identical loop that Memory and FAISS
        ``update_metadata_where`` previously duplicated. Each
        ``(key, meta)`` pair is matched against ``filter`` (``None``
        matches all, parity with :meth:`_match_metadata_filter`); on a
        match ``set_`` is merged into ``meta`` in place (existing keys
        overwritten, others preserved). When ``timestamps`` is provided
        and contains ``key``, that row's ``updated_at`` is refreshed
        while ``created_at`` is preserved — the same upsert-timestamp
        semantics as ``add_vectors``/``update_metadata``. ``key`` must
        index ``timestamps`` the same way the backend keys it (Memory:
        external id; FAISS: internal id) — the caller passes matching
        ``metadata_items`` and ``timestamps``.

        ``set_`` is copied per matched row rather than merged by
        reference. One ``set_`` is merged into every match, so a nested
        value inside it would otherwise be shared by the caller *and* by
        every row the filter selected — a single ``append`` reaching an
        unbounded number of rows at once, and the rows unable to diverge
        from each other afterwards.

        Returns the number of rows whose metadata was merged.
        """
        now = datetime.now(UTC)
        updated = 0
        for key, meta in metadata_items:
            if filter is not None and not self._match_metadata_filter(meta, filter):
                continue
            meta.update(self._copy_metadata(set_) or {})
            if timestamps is not None and key in timestamps:
                created, _ = timestamps[key]
                timestamps[key] = (created, now)
            updated += 1
        return updated

    def _format_timestamp(self, dt: datetime | None) -> Any:
        """Format a timestamp per the configured ``timestamps.format``.

        Supported formats:

        * ``"iso"`` — ISO-8601 string (e.g. ``"2026-04-22T14:23:45.123456+00:00"``)
        * ``"epoch"`` — seconds since epoch as a ``float``
        * ``"datetime"`` — native ``datetime`` object

        Returns ``None`` when input is ``None`` (e.g. a pgvector row
        with ``updated_at IS NULL`` from pre-migration data or an
        MVS/FAISS legacy pickle without tracked timestamps).
        """
        if dt is None:
            return None
        if self.timestamps_format == "datetime":
            return dt
        if self.timestamps_format == "iso":
            return dt.isoformat()
        if self.timestamps_format == "epoch":
            # .timestamp() on naive datetimes treats as local time; on
            # aware datetimes uses the tzinfo. Documented as backend-
            # dependent — pgvector uses naive server time, MVS/FAISS
            # use aware UTC.
            return dt.timestamp()
        # Unreachable — validated in VectorStoreTimestampConfig.__post_init__.
        raise ValueError(f"Unknown timestamps.format: {self.timestamps_format!r}")

    def _inject_timestamps(
        self,
        meta: dict[str, Any] | None,
        created: datetime | None,
        updated: datetime | None,
    ) -> dict[str, Any]:
        """Return a new dict with timestamp keys injected.

        Uses the configured ``timestamps_created_key`` /
        ``timestamps_updated_key`` as the injection keys and
        ``_format_timestamp`` for the values.

        Collision policy: if ``meta`` already contains one of the
        configured keys, the consumer's value wins and framework
        injection for that key is skipped. A WARNING is logged once
        per store instance per colliding key (tracked on the instance,
        so warning state is GC'd with the store).

        Args:
            meta: Consumer metadata (may be ``None``).
            created: Created timestamp from the backend (may be ``None``).
            updated: Updated timestamp from the backend (may be ``None``).

        Returns:
            New dict — never mutates the input.
        """
        result: dict[str, Any] = dict(meta) if meta else {}
        for key, value in (
            (self.timestamps_created_key, created),
            (self.timestamps_updated_key, updated),
        ):
            if key in result:
                if key not in self._timestamp_collision_warned:
                    self._timestamp_collision_warned.add(key)
                    logging.getLogger(__name__).warning(
                        "VectorStore timestamp injection skipped — "
                        "consumer metadata already contains key %r. "
                        "Rename via timestamps.created_key / "
                        "timestamps.updated_key config to avoid collision.",
                        key,
                    )
                continue
            result[key] = self._format_timestamp(value)
        return result

    # ------------------------------------------------------------------
    # Consumer metadata ownership.
    #
    # A store must not share a mutable object with its caller in either
    # direction: not hand out a reference into its own storage, and not
    # keep one to a dict the caller still holds. Chroma and pgvector get
    # this for free — both serialize at their boundary, so what they
    # store and what they return are reconstructions. The in-process
    # backends have to do it deliberately, and these are where they do.
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_metadata(meta: dict[str, Any] | None) -> dict[str, Any] | None:
        """An independent copy of consumer metadata, ``None`` passed through.

        Deep, because shallow is not independence: a shallow copy of
        ``{"tags": ["a"]}`` still shares the list, and appending to it
        through the copy reaches whatever else holds the original. That
        is the whole defect this exists to prevent, one level down —
        and it is exactly the level at which it was missed the first
        time.

        ``deepcopy`` rather than a JSON round-trip: the in-process
        backends persist by pickle and so accept values JSON cannot
        express, and reconstructing through JSON would quietly change
        them (a tuple returning as a list). The cost is bounded — this
        runs per stored row and per *returned* row, never per scored
        candidate.
        """
        return copy.deepcopy(meta) if meta is not None else None

    def _outbound_metadata(
        self,
        stored: dict[str, Any] | None,
        *,
        inject: bool,
        created: datetime | None = None,
        updated: datetime | None = None,
    ) -> dict[str, Any] | None:
        """The metadata dict a result carries — never the stored one.

        Shared by every read path of the in-process backends, both the
        ranked one and the id-keyed one. Those three call sites each
        used to decide the copy for themselves, which is how two of them
        ended up copying only the outer dict.

        ``inject`` adds the configured timestamp keys, and is the
        caller's already-resolved ``include_timestamps and
        include_metadata`` — a store with no metadata for the row still
        returns a dict when injecting, and ``None`` when not.
        """
        copied = self._copy_metadata(stored)
        if inject:
            return self._inject_timestamps(copied, created=created, updated=updated)
        return copied

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(dimensions={self.dimensions}, metric={self.metric.value})"
        )

"""Common base implementation for vector stores."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dataknobs_config import ConfigurableBase

from ..types import DistanceMetric

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np




class VectorStoreBase(ConfigurableBase):
    """Base implementation with common functionality for all vector stores.
    
    This class provides:
    - Configuration parsing following the database pattern
    - Common parameter extraction
    - Shared utility methods
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize vector store with configuration.
        
        Args:
            config: Configuration dictionary with backend-specific parameters
        """
        # ConfigurableBase doesn't have __init__, so don't call super().__init__()
        self.config = config or {}
        self._parse_common_config()
        self._parse_backend_config()
        self._initialized = False
        # Per-instance set of configured timestamp keys for which a
        # collision warning has already been emitted. Lives on the
        # instance (not module scope) so lifetime matches the store —
        # avoids the CPython id() reuse hazard where a new store could
        # inherit a dead store's warning state at the same memory
        # address.
        self._timestamp_collision_warned: set[str] = set()

    def _parse_common_config(self) -> None:
        """Parse common configuration parameters.
        
        Extracts parameters that are common to all vector stores:
        - dimensions: Vector dimensions
        - metric: Distance metric
        - persist_path: Path for persistent storage
        - batch_size: Batch size for operations
        - index_params: Index-specific parameters
        - search_params: Search-specific parameters
        """
        # Extract dimensions (required for most stores)
        self.dimensions = self.config.get("dimensions", 0)

        # Extract and parse metric
        metric = self.config.get("metric", "cosine")
        if isinstance(metric, str):
            self.metric = DistanceMetric(metric)
        else:
            self.metric = metric

        # Extract paths and sizes (expand ~ to home directory)
        persist_path = self.config.get("persist_path")
        self.persist_path = Path(persist_path).expanduser() if persist_path else None
        self.batch_size = self.config.get("batch_size", 100)

        # Debug logging for path resolution
        import logging
        logger = logging.getLogger(__name__)
        if persist_path:
            logger.info(f"VectorStore persist_path: {persist_path} -> {self.persist_path} (exists: {os.path.exists(self.persist_path) if self.persist_path else False})")

        # Extract parameter dictionaries
        self.index_params = self.config.get("index_params", {})
        self.search_params = self.config.get("search_params", {})

        # Store any additional metadata
        self.metadata = self.config.get("metadata", {})

        # Config-level multi-tenant scoping. When set, every
        # read/count/clear/update_metadata_where is implicitly scoped
        # to this domain (via _effective_filter) and add_vectors
        # defaults a row's "domain_id" to it. This mirrors
        # PgVectorStore's long-standing config-level domain_id
        # behavior; Memory/FAISS/Chroma honor it through the shared
        # helpers so a runtime backend swap preserves isolation
        # semantics. None ⇒ no implicit scoping (prior behavior).
        self.domain_id = self.config.get("domain_id")

        # Timestamp exposure config. All vector stores expose
        # created_at / updated_at metadata via include_timestamps=True
        # on get_vectors() and search(). Backends that don't natively
        # persist timestamps (MVS, FAISS) track them in-process. Format
        # and key names are configurable; defaults are consistent
        # across backends so runtime-swap produces identical metadata
        # surfaces.
        ts_cfg = self.config.get("timestamps", {})
        self.timestamps_format: str = ts_cfg.get("format", "iso")
        if self.timestamps_format not in ("iso", "epoch", "datetime"):
            raise ValueError(
                "timestamps.format must be 'iso', 'epoch', or 'datetime'; "
                f"got {self.timestamps_format!r}"
            )
        self.timestamps_created_key: str = ts_cfg.get(
            "created_key", "_created_at"
        )
        self.timestamps_updated_key: str = ts_cfg.get(
            "updated_key", "_updated_at"
        )

    def _parse_backend_config(self) -> None:
        """Parse backend-specific configuration.
        
        Override this method in subclasses to handle backend-specific parameters.
        """
        pass

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

    def _prepare_vector(self, vector: np.ndarray | list[float] | list[np.ndarray], normalize: bool = False) -> np.ndarray:
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

    def _effective_filter(
        self, filter: dict[str, Any] | None
    ) -> dict[str, Any] | None:
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
          the result set is empty. This matches ``PgVectorStore``,
          where the column predicate ``domain_id = $scope`` is ANDed
          with the caller filter and a cross-domain request yields no
          rows.

        Callers pass the result straight to ``_match_metadata_filter`` /
        the filtered count/clear/update paths; a returned dict is never
        ``None`` when scoping is active, so ``filter is None`` fast
        paths must consult this first.
        """
        if self.domain_id is None:
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
        self, metadata: list[dict[str, Any]] | None, count: int
    ) -> list[dict[str, Any]]:
        """Return per-row metadata with ``domain_id`` defaulted.

        Mirrors ``PgVectorStore``'s ``meta.get("domain_id",
        self.domain_id)`` write-path default: when a config-level
        ``domain_id`` is set, any row whose metadata omits
        ``domain_id`` is tagged with the configured scope so the
        read-side ``_effective_filter`` can find it. Returns fresh
        per-row dicts (never mutates or aliases the caller's). A no-op
        passthrough (still copied) when no scope is configured.
        """
        rows = (
            [dict(metadata[i]) if i < len(metadata) else {} for i in range(count)]
            if metadata is not None
            else [{} for _ in range(count)]
        )
        if self.domain_id is not None:
            for row in rows:
                row.setdefault("domain_id", self.domain_id)
        return rows

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

        Returns the number of rows whose metadata was merged.
        """
        now = datetime.now(timezone.utc)
        updated = 0
        for key, meta in metadata_items:
            if filter is not None and not self._match_metadata_filter(
                meta, filter
            ):
                continue
            meta.update(set_)
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
        # Unreachable — validated in _parse_common_config.
        raise ValueError(
            f"Unknown timestamps.format: {self.timestamps_format!r}"
        )

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

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"dimensions={self.dimensions}, "
            f"metric={self.metric.value})"
        )

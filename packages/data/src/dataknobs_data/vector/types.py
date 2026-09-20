# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Core types and data structures for vector operations."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..records import Record


class DistanceMetric(Enum):
    """The vector distance metrics, and the one place their names are read.

    Six members name **four** metrics: ``INNER_PRODUCT`` is ``DOT_PRODUCT``
    and ``L2`` is ``EUCLIDEAN``. That was stated in a trailing comment on each
    member and nowhere a program could read it, so every site mapping a metric
    to something had to restate the aliasing by hand. There were eight ---
    two choosing a pgvector operator, two choosing an index operator class,
    four converting a distance to a score --- and the four that branched on
    the metric each ended in a cosine default, so a caller asking for a
    family that site had missed was answered in cosine distances with no
    error and no log line. They disagreed about which families those were:
    the operator table knew ``inner_product`` and not ``dot_product``, the
    operator-class table knew both, and none of the four knew ``l1``.

    :meth:`canonical` is the fix for that class of divergence: a table keyed
    on the canonical member has four entries and covers all six, where a
    table keyed on the member has six entries and can be written with four.

    ``get_aliases`` described a further vocabulary --- ``cos``, ``manhattan``,
    ``euclidean_distance`` and five more --- that nothing resolved: six of the
    eight raised ``ValueError`` from ``DistanceMetric(...)``, and the two that
    did not were member values rather than anything the alias table achieved.
    :meth:`resolve` is what makes the published list true, and it reads the
    same table :meth:`get_aliases` prints, so the two cannot drift.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    INNER_PRODUCT = "inner_product"  # Alias for dot_product
    L2 = "l2"  # Alias for euclidean
    L1 = "l1"  # Manhattan distance

    def canonical(self) -> DistanceMetric:
        """The member that stands for this metric's family.

        ``INNER_PRODUCT`` canonicalises to ``DOT_PRODUCT`` and ``L2`` to
        ``EUCLIDEAN``; every other member is its own canonical form. Key a
        lookup table on this and a spelling cannot be missed, because there
        are only four keys to miss.

        Returns:
            The canonical member for this metric.
        """
        return _CANONICAL.get(self, self)

    def get_aliases(self) -> list[str]:
        """Alternative names for this metric, all of which :meth:`resolve` accepts.

        Returns:
            The alias list, which may be empty for a member that is itself an
            alternative spelling (``INNER_PRODUCT``, ``L2``).
        """
        return list(_ALIASES.get(self, ()))

    @classmethod
    def resolve(cls, metric: DistanceMetric | str) -> DistanceMetric:
        """Settle any accepted spelling of a metric into a member.

        Accepts a member, a member value, or any name :meth:`get_aliases`
        publishes, in any case. A member value wins over an alias where the
        two collide (``"l2"`` is both), which changes nothing: the two answers
        share a :meth:`canonical` form.

        Args:
            metric: A member, a member value, or a published alias.

        Returns:
            The member that spelling names.

        Raises:
            ValueError: If the name is none of those. Refusing is the point:
                the tables this replaces answered an unrecognised name with
                cosine.
        """
        if isinstance(metric, cls):
            return metric

        name = str(metric).strip().lower()
        try:
            return cls(name)
        except ValueError:
            pass

        for member, aliases in _ALIASES.items():
            if name in aliases:
                return member

        accepted = sorted({m.value for m in cls} | {a for al in _ALIASES.values() for a in al})
        raise ValueError(f"Unknown distance metric {metric!r}; accepted: {', '.join(accepted)}")


# Defined beside the enum rather than inside it: a plain mapping in an ``Enum``
# body becomes a member. Both are read by ``DistanceMetric`` methods and by
# nothing else, which is what keeps this the single vocabulary.
_CANONICAL: dict[DistanceMetric, DistanceMetric] = {
    DistanceMetric.INNER_PRODUCT: DistanceMetric.DOT_PRODUCT,
    DistanceMetric.L2: DistanceMetric.EUCLIDEAN,
}

_ALIASES: dict[DistanceMetric, tuple[str, ...]] = {
    DistanceMetric.COSINE: ("cosine_similarity", "cos"),
    DistanceMetric.EUCLIDEAN: ("l2", "euclidean_distance"),
    DistanceMetric.DOT_PRODUCT: ("inner_product", "ip"),
    DistanceMetric.L1: ("manhattan", "l1_distance"),
}


# What a batch embedding callable may return. `np.ndarray` is the legacy
# shape; the list arm is what a `TextEmbedder` produces and what these sites
# always accepted at runtime --- `pair_records_with_vectors` requires only
# "something indexable and sized".
#
# Named rather than inlined because `AsyncBulkEmbedMixin` and
# `AsyncVectorOperationsMixin` are mixed into the same four backends. Widening
# one declaration and not the other is not an error at either site: it is a
# `[misc]` "incompatible definitions in base classes" finding on every backend
# inheriting both, four files away from the edit that caused it.
#
# Not in `embedding.py`, whose freedom from numpy is deliberate and argued in
# its own docstring --- the protocol returns `list[list[float]]` precisely so
# the `llm` boundary needs no numpy.
BatchVectors = np.ndarray | list[list[float]]


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search operation."""

    record: Record
    score: float
    source_text: str | None = None
    vector_field: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: VectorSearchResult) -> bool:
        """Enable sorting by score."""
        return self.score < other.score

    def __repr__(self) -> str:
        """String representation of the result."""
        return (
            f"VectorSearchResult(score={self.score:.4f}, "
            f"record_id={self.record.id}, "
            f"vector_field={self.vector_field})"
        )


@dataclass
class VectorConfig:
    """Configuration for vector operations."""

    dimensions: int
    metric: DistanceMetric = DistanceMetric.COSINE
    normalize: bool = False
    source_field: str | None = None
    model_name: str | None = None
    model_version: str | None = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.dimensions <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.dimensions}")

        if self.dimensions > 65536:  # Common maximum for vector databases
            raise ValueError(f"Dimensions {self.dimensions} exceeds maximum supported (65536)")


@dataclass
class VectorIndexConfig:
    """Configuration for vector index creation."""

    index_type: str = "auto"  # auto, flat, ivfflat, hnsw
    lists: int | None = None  # For IVFFlat
    m: int | None = None  # For HNSW
    ef_construction: int | None = None  # For HNSW
    ef_search: int | None = None  # For HNSW search
    probes: int | None = None  # For IVFFlat search
    quantization: str | None = None  # none, scalar, product

    def get_optimal_params(self, num_vectors: int) -> dict[str, Any]:
        """Get optimal index parameters based on dataset size."""
        # Annotated, not inferred: the first assignment is a `str` and every
        # tuning value after it is an `int`, so an inferred `dict[str, str]`
        # made all ten of those a finding. The return type already says
        # `dict[str, Any]`.
        params: dict[str, Any] = {}

        if self.index_type == "auto":
            # Auto-select based on dataset size
            if num_vectors < 10_000:
                params["type"] = "flat"
            elif num_vectors < 1_000_000:
                params["type"] = "ivfflat"
                params["lists"] = self.lists or max(num_vectors // 1000, 100)
                params["probes"] = self.probes or 10
            else:
                params["type"] = "hnsw"
                params["m"] = self.m or 16
                params["ef_construction"] = self.ef_construction or 200
                params["ef_search"] = self.ef_search or 64
        else:
            params["type"] = self.index_type
            if self.index_type == "ivfflat":
                params["lists"] = self.lists or 100
                params["probes"] = self.probes or 10
            elif self.index_type == "hnsw":
                params["m"] = self.m or 16
                params["ef_construction"] = self.ef_construction or 200
                params["ef_search"] = self.ef_search or 64

        if self.quantization:
            params["quantization"] = self.quantization

        return params


@dataclass
class VectorMetadata:
    """Metadata associated with vector fields.

    Not the whole of what a ``{field}_metadata`` sidecar holds.
    ``IncrementalVectorizer`` merges the staleness digest in beside
    :meth:`to_dict`'s output --- the three keys ``content_hash_metadata``
    writes --- because a vector nothing can judge is treated as current
    forever. So round-tripping such a sidecar through :meth:`from_dict` drops
    the digest and silently restores that exemption; read the dict, or merge
    the digest back. Nothing in this package does that round trip, which is
    the only reason this is a note rather than a defect.
    """

    dimensions: int
    source_field: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    index_type: str | None = None
    metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimensions": self.dimensions,
            "source_field": self.source_field,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
            }
            if self.model_name
            else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "index_type": self.index_type,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorMetadata:
        """Create from dictionary representation."""
        model_info = data.get("model", {})
        return cls(
            dimensions=data["dimensions"],
            source_field=data.get("source_field"),
            model_name=model_info.get("name") if model_info else None,
            model_version=model_info.get("version") if model_info else None,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            index_type=data.get("index_type"),
            metric=data.get("metric"),
        )

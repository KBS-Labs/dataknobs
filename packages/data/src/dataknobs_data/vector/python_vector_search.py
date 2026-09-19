# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Python-based vector search implementation for databases without native vector support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np

    from ..query import Query
    from ..records import Record
    from .types import DistanceMetric, VectorSearchResult

logger = logging.getLogger(__name__)


class PythonVectorSearchMixin:
    """Mixin providing Python-based vector similarity search.

    Used by the eight backends with no native k-NN --- memory, file, sqlite
    and s3, in both lanes --- as the body of their ``_vector_search`` hook.

    The backend must provide:

    - a record-fetch method, named by ``fetch_all_method`` /
      ``fetch_filtered_method`` (both default to ``search``)
    - ``_compute_similarity``, which memory, file, sqlite and s3 all inherit
      from ``SQLiteVectorSupport``
    """

    if TYPE_CHECKING:
        # Declared for the type checker only. A runtime stub here would sit on
        # the MRO ahead of a host that supplies the real one if a backend ever
        # listed the bases the other way round, and shadow it silently.
        #
        # Spelled exactly as ``SQLiteVectorSupport`` spells it, including the
        # optional ``metric`` and the ``| None`` arms: a narrower declaration
        # here is a ``[misc]`` "incompatible definitions in base classes" on
        # every backend that inherits both, reported against *their* files.
        def _compute_similarity(
            self,
            vec1: np.ndarray | None,
            vec2: np.ndarray | None,
            metric: DistanceMetric = ...,
        ) -> float: ...

    async def python_vector_search_async(
        self,
        query_vector: np.ndarray | Sequence[float],
        *,
        vector_field: str = "embedding",
        k: int = 10,
        filter: Query | None = None,
        metric: DistanceMetric | str | None = None,
        fetch_all_method: str = "search",
        fetch_filtered_method: str = "search",
    ) -> list[VectorSearchResult]:
        """Perform async vector search using Python calculations.

        Args:
            query_vector: Query vector
            vector_field: Name of the vector field to search
            k: Number of results to return
            filter: Optional filter conditions
            metric: Distance metric, resolved against the database's
                configured one when absent
            fetch_all_method: Name of method to fetch all records
            fetch_filtered_method: Name of method to fetch filtered records

        Returns:
            List of VectorSearchResult objects, nearest first
        """
        from ..query import Query
        from .mixins import resolve_metric

        records = (
            await getattr(self, fetch_filtered_method)(filter)
            if filter
            else await getattr(self, fetch_all_method)(Query())
        )
        return self._score_and_rank(
            records,
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            metric=resolve_metric(self, metric),
        )

    def python_vector_search_sync(
        self,
        query_vector: np.ndarray | Sequence[float],
        *,
        vector_field: str = "embedding",
        k: int = 10,
        filter: Query | None = None,
        metric: DistanceMetric | str | None = None,
        fetch_all_method: str = "search",
        fetch_filtered_method: str = "search",
    ) -> list[VectorSearchResult]:
        """Perform sync vector search using Python calculations.

        Args:
            query_vector: Query vector
            vector_field: Name of the vector field to search
            k: Number of results to return
            filter: Optional filter conditions
            metric: Distance metric, resolved against the database's
                configured one when absent
            fetch_all_method: Name of method to fetch all records
            fetch_filtered_method: Name of method to fetch filtered records

        Returns:
            List of VectorSearchResult objects, nearest first
        """
        from ..query import Query
        from .mixins import resolve_metric

        records = (
            getattr(self, fetch_filtered_method)(filter)
            if filter
            else getattr(self, fetch_all_method)(Query())
        )
        return self._score_and_rank(
            records,
            query_vector=query_vector,
            vector_field=vector_field,
            k=k,
            metric=resolve_metric(self, metric),
        )

    def _score_and_rank(
        self,
        records: Iterable[Record | dict[str, Any] | Any],
        *,
        query_vector: np.ndarray | Sequence[float],
        vector_field: str,
        k: int,
        metric: DistanceMetric,
    ) -> list[VectorSearchResult]:
        """Score a fetched page of records and return the nearest ``k``.

        Everything the two search methods do after the fetch, which is
        everything except the ``await``. They carried a copy each --- the
        metric resolution, the numpy coercion, the format handling, the sort
        --- about forty-five lines that had to be changed twice or diverge.

        Args:
            records: Whatever the backend's fetch method yielded: ``Record``
                objects, raw row dicts, or plain data dicts.
            query_vector: The query, coerced here if it is not already an array.
            vector_field: The field holding each record's vector.
            k: How many to return.
            metric: The metric, already resolved.

        Returns:
            The ``k`` highest-scoring results, nearest first.
        """
        import numpy as np

        from ..records import Record
        from .types import VectorSearchResult

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        results: list[VectorSearchResult] = []
        for record_data in records:
            if isinstance(record_data, dict):
                data = self._extract_record_data(record_data)
            elif isinstance(record_data, Record):
                data = record_data.data
            else:
                data = record_data

            if not (
                isinstance(data, dict) and data.get(vector_field) is not None
            ):  # pragma: no branch
                continue

            stored_vector = data[vector_field]
            # The ``VectorField`` dict form, as ``to_dict()`` writes it.
            if isinstance(stored_vector, dict) and "value" in stored_vector:
                stored_vector = stored_vector["value"]
            if not isinstance(stored_vector, np.ndarray):
                stored_vector = np.array(stored_vector, dtype=np.float32)

            score = self._compute_similarity(query_vector, stored_vector, metric)
            record = (
                record_data
                if isinstance(record_data, Record)
                else self._create_record_from_data(record_data, data)
            )
            results.append(
                VectorSearchResult(record=record, score=float(score), vector_field=vector_field)
            )

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:k]

    def _extract_record_data(self, record_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract the actual data from a record dictionary.

        Handles different storage formats like:
        - Direct data storage
        - Data in a 'data' column (JSON)
        - Double-nested data structures

        Args:
            record_dict: Raw record dictionary from database

        Returns:
            Extracted data dictionary, or an empty one where the column held
            something that is not a mapping --- which the callers already
            treated as "no vector here", by testing the result's type.
        """
        import json

        if "data" not in record_dict:
            return record_dict

        data = record_dict["data"]
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return {}

        nested = data.get("data")
        return nested if isinstance(nested, dict) else data

    def _create_record_from_data(self, record_dict: dict[str, Any], data: dict[str, Any]) -> Record:
        """Create a Record object from raw data.

        Args:
            record_dict: Original record dictionary (may contain metadata)
            data: Extracted data dictionary

        Returns:
            Record object
        """
        import json

        from ..records import Record

        metadata = record_dict.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                metadata = {}

        return Record(data=data, id=record_dict.get("id"), metadata=metadata)

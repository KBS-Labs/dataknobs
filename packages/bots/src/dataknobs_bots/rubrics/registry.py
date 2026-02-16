"""Rubric registry for storing and retrieving rubrics.

This module provides persistent rubric storage backed by AsyncDatabase,
supporting versioned rubric management and target-type lookups.

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> db = AsyncMemoryDatabase()
    >>> registry = RubricRegistry(db)
    >>> rubric_id = await registry.register(rubric)
    >>> retrieved = await registry.get(rubric_id)
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_data import AsyncDatabase, Filter, Operator, Query, Record

from .models import Rubric

logger = logging.getLogger(__name__)


class RubricRegistry:
    """Persistent rubric storage backed by AsyncDatabase.

    Rubrics are stored as versioned records with composite keys
    ``"{rubric_id}:{version}"``. A pointer record at ``"{rubric_id}"``
    tracks the latest version.

    Args:
        db: The async database backend for storage.
    """

    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db

    async def register(self, rubric: Rubric) -> str:
        """Store a rubric and return its ID.

        Creates both a versioned record and a latest-pointer record.

        Args:
            rubric: The rubric to store.

        Returns:
            The rubric ID.
        """
        version_key = f"{rubric.id}:{rubric.version}"
        data = rubric.to_dict()
        data["_version_key"] = version_key

        await self._db.upsert(version_key, Record(data))
        await self._db.upsert(rubric.id, Record(data))

        logger.info(
            "Registered rubric '%s' version '%s'",
            rubric.id,
            rubric.version,
        )
        return rubric.id

    async def get(
        self, rubric_id: str, version: str | None = None
    ) -> Rubric | None:
        """Retrieve a rubric by ID, optionally at a specific version.

        Args:
            rubric_id: The rubric identifier.
            version: If provided, retrieves that specific version.
                If None, retrieves the latest version.

        Returns:
            The rubric, or None if not found.
        """
        key = f"{rubric_id}:{version}" if version else rubric_id
        record = await self._db.read(key)
        if record is None:
            return None
        return Rubric.from_dict(record.data)

    async def get_for_target(self, target_type: str) -> list[Rubric]:
        """Get all rubrics applicable to a target type.

        This searches the latest version of each rubric. Returns only
        rubrics whose target_type matches the given type.

        Args:
            target_type: The type to filter by (e.g., "content", "rubric").

        Returns:
            List of matching rubrics.
        """
        query = Query(
            filters=[
                Filter("target_type", Operator.EQ, target_type),
            ]
        )
        records = await self._db.search(query)

        rubrics: list[Rubric] = []
        seen_ids: set[str] = set()
        for record in records:
            rubric = Rubric.from_dict(record.data)
            # Skip versioned duplicates — only include each rubric once
            if rubric.id not in seen_ids:
                seen_ids.add(rubric.id)
                rubrics.append(rubric)

        return rubrics

    async def update(self, rubric: Rubric) -> str:
        """Store a new version of an existing rubric.

        The rubric should have its version already bumped. This creates
        a new versioned record and updates the latest pointer.

        Args:
            rubric: The rubric with updated version.

        Returns:
            The rubric ID.
        """
        return await self.register(rubric)

    async def delete(self, rubric_id: str) -> bool:
        """Remove a rubric's latest pointer record.

        Note: Versioned records are retained for audit history.

        Args:
            rubric_id: The rubric to remove.

        Returns:
            True if the rubric was found and deleted, False otherwise.
        """
        record = await self._db.read(rubric_id)
        if record is None:
            return False
        return await self._db.delete(rubric_id)

    async def list_all(self) -> list[Rubric]:
        """List the latest version of all registered rubrics.

        Returns:
            List of all rubrics (latest versions only).
        """
        all_records = await self._db.search(Query())

        rubrics: list[Rubric] = []
        seen_ids: set[str] = set()
        for record in all_records:
            data = record.data
            # Skip versioned records — only return latest pointers
            version_key = data.get("_version_key", "")
            record_key = record.storage_id or record.id
            if version_key and record_key == version_key:
                continue
            rubric = Rubric.from_dict(data)
            if rubric.id not in seen_ids:
                seen_ids.add(rubric.id)
                rubrics.append(rubric)

        return rubrics

    @classmethod
    async def from_config(
        cls, config: dict[str, Any], db: AsyncDatabase
    ) -> RubricRegistry:
        """Create a registry and load rubrics from a config dict.

        The config should have a ``"rubrics"`` key containing a list
        of rubric definitions (each as a dict matching Rubric.from_dict).

        Args:
            config: Configuration dict with rubric definitions.
            db: The async database backend.

        Returns:
            A populated RubricRegistry.
        """
        registry = cls(db)
        rubric_defs = config.get("rubrics", [])
        for rubric_data in rubric_defs:
            rubric = Rubric.from_dict(rubric_data)
            await registry.register(rubric)
            logger.info(
                "Loaded rubric '%s' from config",
                rubric.id,
            )
        return registry

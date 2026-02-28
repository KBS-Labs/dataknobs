"""ArtifactBankCatalog — data-store-backed collection of artifacts.

Manages multiple compiled artifacts (e.g. recipes) in a ``SyncDatabase``,
keyed by artifact name.  Tools interact with the catalog via list/get/save
operations; file import/export is a separate concern handled by
``artifact_io``.

Example::

    >>> from dataknobs_data.backends.memory import SyncMemoryDatabase
    >>> from dataknobs_bots.memory.catalog import ArtifactBankCatalog
    >>> catalog = ArtifactBankCatalog(SyncMemoryDatabase())
    >>> catalog.save(artifact)
    >>> catalog.list()
    [{'name': 'recipe', 'sections': ['ingredients', 'instructions'], 'field_count': 1}]
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_data import Record, SyncDatabase

from .artifact_bank import ArtifactBank

logger = logging.getLogger(__name__)


class ArtifactBankCatalog:
    """Collection of compiled artifacts backed by a ``SyncDatabase``.

    Each artifact is stored as a ``Record`` keyed by its ``_artifact_name``
    field.  The record data is the full compiled dict from
    ``ArtifactBank.compile()``.

    Args:
        db: Sync database backend for storage.
        artifact_config: Optional artifact configuration dict passed to
            ``ArtifactBank.from_config`` when loading entries back into
            an ``ArtifactBank``.
    """

    def __init__(
        self,
        db: SyncDatabase,
        artifact_config: dict[str, Any] | None = None,
    ) -> None:
        self._db = db
        self._artifact_config = artifact_config

    # -----------------------------------------------------------------
    # Read operations
    # -----------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        """List summary info for all catalog entries.

        Returns:
            List of dicts with ``name``, ``sections`` (list of section
            names), and ``field_count`` keys.
        """
        records = self._db.all()
        entries: list[dict[str, Any]] = []
        for record in records:
            data = record.data
            compiled = data.get("compiled", {})
            name = compiled.get("_artifact_name", record.storage_id or "")
            sections: list[str] = []
            field_count = 0
            for key, value in compiled.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, list) and all(
                    isinstance(v, dict) for v in value
                ):
                    sections.append(key)
                else:
                    field_count += 1
            entries.append({
                "name": name,
                "sections": sections,
                "field_count": field_count,
            })
        return entries

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a compiled artifact dict by name.

        Args:
            name: Artifact name (the ``_artifact_name`` value).

        Returns:
            The compiled dict, or ``None`` if not found.
        """
        record = self._db.read(name)
        if record is None:
            return None
        return record.data.get("compiled")

    def count(self) -> int:
        """Return the number of entries in the catalog."""
        return len(self._db.all())

    # -----------------------------------------------------------------
    # Write operations
    # -----------------------------------------------------------------

    def save(self, artifact: ArtifactBank) -> None:
        """Validate, compile, and upsert an artifact into the catalog.

        The artifact is stored under its ``name`` as the record key.

        Args:
            artifact: The artifact to save.

        Raises:
            ValueError: If the artifact fails validation.
        """
        errors = artifact.validate()
        if errors:
            raise ValueError(
                f"Cannot save artifact '{artifact.name}': "
                + "; ".join(errors)
            )
        compiled = artifact.compile()
        record = Record({"compiled": compiled}, storage_id=artifact.name)
        self._db.upsert(artifact.name, record)
        logger.info(
            "Saved artifact '%s' to catalog (%d sections)",
            artifact.name,
            len(artifact.sections),
        )

    def delete(self, name: str) -> bool:
        """Remove an entry by name.

        Args:
            name: Artifact name to remove.

        Returns:
            ``True`` if the entry was found and deleted.
        """
        if not self._db.exists(name):
            return False
        self._db.delete(name)
        logger.info("Deleted artifact '%s' from catalog", name)
        return True

    # -----------------------------------------------------------------
    # Load into an existing ArtifactBank
    # -----------------------------------------------------------------

    def load_into(self, name: str, target: ArtifactBank) -> bool:
        """Load a catalog entry into an existing ``ArtifactBank``.

        Replaces the target's fields and section records with the
        catalog entry via ``replace_from_compiled``.

        Args:
            name: Artifact name to load.
            target: The ``ArtifactBank`` to populate.

        Returns:
            ``True`` if the entry was found and loaded.
        """
        compiled = self.get(name)
        if compiled is None:
            return False
        target.replace_from_compiled(compiled, source_stage="catalog")
        logger.info(
            "Loaded artifact '%s' from catalog into target",
            name,
        )
        return True

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ArtifactBankCatalog:
        """Create a catalog from a configuration dict.

        Config keys:
            backend: Database backend type (default ``"memory"``).
            backend_config: Dict passed to the database factory.
            artifact_config: Optional artifact configuration for loads.

        Args:
            config: Configuration dict.

        Returns:
            Configured ``ArtifactBankCatalog`` instance.
        """
        from dataknobs_data import database_factory

        backend = config.get("backend", "memory")
        backend_config = dict(config.get("backend_config", {}))
        backend_config["backend"] = backend
        db = database_factory.create(**backend_config)
        artifact_config = config.get("artifact_config")
        return cls(db=db, artifact_config=artifact_config)

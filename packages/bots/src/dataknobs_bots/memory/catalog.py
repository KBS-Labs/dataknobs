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
    >>> catalog.list()  # doctest: +SKIP
    [{'name': 'recipe', 'artifact_type': 'recipe', 'fields': {...}, 'sections': {...}}]
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_data import Record, SyncDatabase

from .artifact_bank import ArtifactBank

logger = logging.getLogger(__name__)


class ArtifactBankCatalog:
    """Collection of compiled artifacts backed by a ``SyncDatabase``.

    Each artifact is stored as a ``Record`` keyed by either a
    configurable field value (``entry_name_field``) or the artifact's
    type name.  The record data contains the compiled dict and an
    optional ``"previous"`` field for single-level undo.

    Args:
        db: Sync database backend for storage.
        artifact_config: Optional artifact configuration dict passed to
            ``ArtifactBank.from_config`` when loading entries back into
            an ``ArtifactBank``.
        entry_name_field: Optional artifact field name whose value is
            used as the catalog entry key.  When set, each distinct
            field value gets its own entry (e.g. each recipe name).
            Falls back to ``artifact.name`` if unset or the field is
            empty.
    """

    def __init__(
        self,
        db: SyncDatabase,
        artifact_config: dict[str, Any] | None = None,
        entry_name_field: str | None = None,
    ) -> None:
        self._db = db
        self._artifact_config = artifact_config
        self._entry_name_field = entry_name_field

    # -----------------------------------------------------------------
    # Entry name resolution
    # -----------------------------------------------------------------

    def resolve_entry_name(self, artifact: ArtifactBank) -> str:
        """Resolve the catalog entry key for this artifact.

        If ``entry_name_field`` is configured and the artifact has a
        non-empty value for that field, the field value is used as the
        key.  Otherwise falls back to ``artifact.name``.

        Args:
            artifact: The artifact to resolve a key for.

        Returns:
            The catalog entry key string.
        """
        if self._entry_name_field:
            value = artifact.field(self._entry_name_field)
            if value:
                return str(value)
        return artifact.name

    # -----------------------------------------------------------------
    # Read operations
    # -----------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        """List summary info for all catalog entries.

        Returns:
            List of dicts with:

            - ``name``: The catalog entry key (``storage_id``), usable
              with ``get()``, ``load_into()``, ``delete()``, and
              ``revert()``.
            - ``artifact_type``: The artifact type name (e.g.
              ``"recipe"``).
            - ``fields``: Dict of scalar field names to their values.
            - ``sections``: Dict of section names to record counts.
        """
        records = self._db.all()
        entries: list[dict[str, Any]] = []
        for record in records:
            data = record.data
            compiled = data.get("compiled", {})
            name = record.storage_id or ""
            artifact_type = compiled.get("_artifact_name", "unknown")
            fields: dict[str, Any] = {}
            sections: dict[str, int] = {}
            for key, value in compiled.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, list) and all(
                    isinstance(v, dict) for v in value
                ):
                    sections[key] = len(value)
                else:
                    fields[key] = value
            entries.append({
                "name": name,
                "artifact_type": artifact_type,
                "fields": fields,
                "sections": sections,
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

    def save(self, artifact: ArtifactBank) -> str:
        """Validate, compile, and upsert an artifact into the catalog.

        The artifact is stored under a key resolved by
        ``resolve_entry_name()`` (either the configured field value or
        the artifact type name).

        Before overwriting, the existing compiled data is preserved as a
        ``"previous"`` field in the new record, enabling single-level
        undo via ``revert()``.

        Args:
            artifact: The artifact to save.

        Returns:
            The catalog entry key under which the artifact was stored.

        Raises:
            ValueError: If the artifact fails validation.
        """
        errors = artifact.validate()
        if errors:
            raise ValueError(
                f"Cannot save artifact '{artifact.name}': "
                + "; ".join(errors)
            )
        key = self.resolve_entry_name(artifact)
        compiled = artifact.compile()

        # Preserve previous version for single-level undo.
        existing = self._db.read(key)
        previous = existing.data.get("compiled") if existing else None

        record = Record(
            {"compiled": compiled, "previous": previous},
            storage_id=key,
        )
        self._db.upsert(key, record)
        logger.info(
            "Saved artifact '%s' to catalog as '%s' (%d sections)",
            artifact.name,
            key,
            len(artifact.sections),
        )
        return key

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

    def revert(self, name: str) -> bool:
        """Restore the previous version of a catalog entry.

        Reads the ``"previous"`` field from the DB record and overwrites
        ``"compiled"`` with it.  Sets ``"previous"`` to ``None`` — there
        is no cascading undo.

        Args:
            name: Catalog entry name to revert.

        Returns:
            ``True`` if the entry was reverted, ``False`` if no entry or
            no previous version exists.
        """
        record = self._db.read(name)
        if record is None:
            return False
        previous = record.data.get("previous")
        if previous is None:
            return False
        reverted = Record(
            {"compiled": previous, "previous": None},
            storage_id=name,
        )
        self._db.upsert(name, reverted)
        logger.info("Reverted catalog entry '%s' to previous version", name)
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
        db.connect()
        artifact_config = config.get("artifact_config")
        entry_name_field = config.get("entry_name_field")
        return cls(
            db=db,
            artifact_config=artifact_config,
            entry_name_field=entry_name_field,
        )

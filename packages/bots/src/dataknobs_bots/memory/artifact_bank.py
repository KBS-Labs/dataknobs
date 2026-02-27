"""ArtifactBank — composition layer over MemoryBank sections and scalar fields.

ArtifactBank ties together top-level fields (e.g. ``recipe_name``) and named
``MemoryBank`` sections (e.g. ``ingredients``, ``instructions``) into a single
artifact with compile, validation, and finalization semantics.

This is **composition, not inheritance** — it owns MemoryBank sections and
manages scalar fields alongside them.

Example:
    >>> from dataknobs_data.backends.memory import SyncMemoryDatabase
    >>> from dataknobs_bots.memory.bank import MemoryBank
    >>> ingredients = MemoryBank("ingredients", schema={"required": ["name"]},
    ...                          db=SyncMemoryDatabase())
    >>> instructions = MemoryBank("instructions", schema={"required": ["instruction"]},
    ...                           db=SyncMemoryDatabase())
    >>> artifact = ArtifactBank(
    ...     name="recipe",
    ...     field_defs={"recipe_name": {"required": True}},
    ...     sections={"ingredients": ingredients, "instructions": instructions},
    ... )
    >>> artifact.set_field("recipe_name", "Chocolate Chip Cookies")
    >>> ingredients.add({"name": "flour", "amount": "2 cups"})
    '...'
    >>> compiled = artifact.compile()
    >>> compiled["recipe_name"]
    'Chocolate Chip Cookies'
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from dataknobs_data import SyncDatabase

from .bank import EmptyBankProxy, MemoryBank

logger = logging.getLogger(__name__)


class ArtifactBank:
    """Composition of scalar fields and named MemoryBank sections.

    Args:
        name: Artifact identifier (e.g. ``"recipe"``).
        field_defs: Field definitions.  Keys are field names, values are
            dicts with optional ``required`` key (bool).
        sections: Named ``MemoryBank`` instances comprising the artifact.
        section_configs: Optional per-section configuration dicts, preserved
            for serialization so ``from_dict`` can reconstruct sections
            with the correct backend.
    """

    def __init__(
        self,
        name: str,
        field_defs: dict[str, dict[str, Any]],
        sections: dict[str, MemoryBank],
        section_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._name = name
        self._field_defs = dict(field_defs)
        self._field_values: dict[str, Any] = dict.fromkeys(field_defs)
        self._sections = dict(sections)
        self._section_configs = dict(section_configs or {})
        self._finalized = False

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        """Artifact identifier."""
        return self._name

    @property
    def field_defs(self) -> dict[str, dict[str, Any]]:
        """Field definitions (name -> config dict)."""
        return dict(self._field_defs)

    @property
    def fields(self) -> dict[str, Any]:
        """Snapshot copy of current field values."""
        return dict(self._field_values)

    @property
    def sections(self) -> dict[str, MemoryBank]:
        """Direct references to section MemoryBank instances."""
        return dict(self._sections)

    # -----------------------------------------------------------------
    # Field management
    # -----------------------------------------------------------------

    def field(self, name: str) -> Any:
        """Get a single field value.  Returns ``None`` if unset."""
        return self._field_values.get(name)

    def set_field(self, name: str, value: Any) -> None:
        """Set a field value.

        Args:
            name: Field name (must be defined in ``field_defs``).
            value: Value to set.

        Raises:
            ValueError: If the artifact is finalized or the field is unknown.
        """
        if self._finalized:
            raise ValueError(
                f"Cannot modify field '{name}': artifact '{self._name}' "
                "is finalized. Call unfinalize() first."
            )
        if name not in self._field_defs:
            raise ValueError(
                f"Unknown field '{name}' for artifact '{self._name}'. "
                f"Defined fields: {list(self._field_defs)}"
            )
        self._field_values[name] = value

    def clear_fields(self) -> None:
        """Reset all field values to ``None``."""
        self._field_values = dict.fromkeys(self._field_defs)

    # -----------------------------------------------------------------
    # Section access
    # -----------------------------------------------------------------

    def section(self, name: str) -> MemoryBank | EmptyBankProxy:
        """Get a section by name.

        Returns an ``EmptyBankProxy`` if the section does not exist,
        preventing crashes in templates and conditions.
        """
        return self._sections.get(name, EmptyBankProxy(name))

    # -----------------------------------------------------------------
    # Compilation
    # -----------------------------------------------------------------

    def compile(self) -> dict[str, Any]:
        """Compile all fields and sections into a single dict.

        Returns:
            Dict with artifact metadata, field values, and section records.
        """
        result: dict[str, Any] = {
            "_artifact_name": self._name,
            "_compiled_at": time.time(),
        }
        # Add field values
        for field_name, value in self._field_values.items():
            result[field_name] = value
        # Add section records
        for section_name, bank in self._sections.items():
            records = bank.all()
            result[section_name] = [dict(r.data) for r in records]
        return result

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the artifact is complete.

        Returns:
            List of error strings.  Empty list means valid.
        """
        errors: list[str] = []
        # Check required fields
        for field_name, field_def in self._field_defs.items():
            if field_def.get("required") and self._field_values.get(field_name) is None:
                errors.append(
                    f"Required field '{field_name}' is not set."
                )
        # Check sections have records
        for section_name, bank in self._sections.items():
            if bank.count() == 0:
                errors.append(
                    f"Section '{section_name}' has no records."
                )
        return errors

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    @property
    def is_finalized(self) -> bool:
        """Whether the artifact has been finalized."""
        return self._finalized

    def finalize(self) -> dict[str, Any]:
        """Validate, compile, and lock the artifact.

        Returns:
            The compiled artifact dict.

        Raises:
            ValueError: If validation fails.
        """
        errors = self.validate()
        if errors:
            raise ValueError(
                f"Cannot finalize artifact '{self._name}': "
                + "; ".join(errors)
            )
        compiled = self.compile()
        self._finalized = True
        logger.info(
            "Finalized artifact '%s' with %d fields and %d sections",
            self._name,
            len(self._field_values),
            len(self._sections),
        )
        return compiled

    def unfinalize(self) -> None:
        """Re-open the artifact for edits."""
        self._finalized = False
        logger.debug("Unfinalized artifact '%s'", self._name)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact to a plain dict."""
        return {
            "name": self._name,
            "field_defs": self._field_defs,
            "field_values": dict(self._field_values),
            "finalized": self._finalized,
            "section_configs": self._section_configs,
            "sections": {
                name: bank.to_dict()
                for name, bank in self._sections.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        db_factory: Callable[[str, dict[str, Any]], tuple[SyncDatabase, str]] | None = None,
    ) -> ArtifactBank:
        """Deserialize an artifact from a plain dict.

        Args:
            data: Serialized artifact dict (from ``to_dict()``).
            db_factory: Optional factory for creating database backends
                per-section.  Takes ``(bank_name, config)`` and returns
                ``(SyncDatabase, storage_mode)``.  Without ``db_factory``,
                sections default to ``SyncMemoryDatabase`` (inline restore).
        """
        section_configs = data.get("section_configs", {})
        sections: dict[str, MemoryBank] = {}
        for name, bank_dict in data.get("sections", {}).items():
            cfg = section_configs.get(name, {})
            if db_factory and cfg.get("backend", "memory") != "memory":
                db, _mode = db_factory(name, cfg)
                sections[name] = MemoryBank.from_dict(bank_dict, db=db)
            else:
                sections[name] = MemoryBank.from_dict(bank_dict)

        artifact = cls(
            name=data["name"],
            field_defs=data.get("field_defs", {}),
            sections=sections,
            section_configs=section_configs,
        )
        # Restore field values
        for field_name, value in data.get("field_values", {}).items():
            if field_name in artifact._field_defs:
                artifact._field_values[field_name] = value
        # Restore finalization state
        artifact._finalized = data.get("finalized", False)
        return artifact

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        db_factory: Callable[[str, dict[str, Any]], tuple[SyncDatabase, str]] | None = None,
    ) -> ArtifactBank:
        """Create an ArtifactBank from a configuration dict.

        Args:
            config: Artifact configuration with ``name``, ``fields``, and
                ``sections`` keys.
            db_factory: Optional factory for creating database backends
                per-section.  Takes ``(bank_name, config)`` and returns
                ``(SyncDatabase, storage_mode)``.  Enables persistent
                backends (SQLite, PostgreSQL, etc.) for sections.

        Returns:
            Configured ``ArtifactBank`` instance.
        """
        name = config.get("name", "artifact")
        field_defs = config.get("fields", {})
        section_configs = config.get("sections", {})

        sections: dict[str, MemoryBank] = {}
        for section_name, cfg in section_configs.items():
            # Support both flat keys and nested duplicate_detection
            dup_cfg = cfg.get("duplicate_detection", {})
            dup_strategy = (
                cfg.get("duplicate_strategy")
                or dup_cfg.get("strategy", "allow")
            )
            match_fields = (
                cfg.get("match_fields")
                or dup_cfg.get("match_fields")
            )

            if db_factory:
                db, storage_mode = db_factory(section_name, cfg)
            else:
                from dataknobs_data.backends.memory import SyncMemoryDatabase

                db = SyncMemoryDatabase()
                storage_mode = "inline"

            sections[section_name] = MemoryBank(
                name=section_name,
                schema=cfg.get("schema", {}),
                db=db,
                max_records=cfg.get("max_records"),
                duplicate_strategy=dup_strategy,
                match_fields=match_fields,
                storage_mode=storage_mode,
            )

        return cls(
            name=name,
            field_defs=field_defs,
            sections=sections,
            section_configs=section_configs,
        )

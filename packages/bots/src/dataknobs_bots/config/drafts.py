"""Draft management for interactive configuration creation.

Provides file-based draft persistence during wizard-driven config building.
Drafts are saved incrementally as users progress through stages, with
automatic cleanup of stale drafts.

Example:
    ```python
    from pathlib import Path
    from dataknobs_bots.config.drafts import ConfigDraftManager

    manager = ConfigDraftManager(output_dir=Path("/data/configs"))

    # Create a draft during wizard flow
    draft_id = manager.create_draft({"llm": {"provider": "ollama"}}, stage="configure_llm")

    # Update as user progresses
    manager.update_draft(draft_id, updated_config, stage="configure_memory")

    # Finalize when complete
    final_config = manager.finalize(draft_id, final_name="my-bot")

    # Cleanup old drafts
    cleaned = manager.cleanup_stale()
    ```
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DraftMetadata:
    """Metadata for a configuration draft.

    Attributes:
        draft_id: Unique identifier for the draft.
        created_at: ISO 8601 creation timestamp.
        last_updated: ISO 8601 last update timestamp.
        stage: Current wizard stage when draft was saved.
        complete: Whether the draft represents a complete config.
        config_name: Optional name for the final config file.
    """

    draft_id: str
    created_at: str
    last_updated: str
    stage: str | None = None
    complete: bool = False
    config_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {
            "id": self.draft_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "complete": self.complete,
        }
        if self.stage is not None:
            result["stage"] = self.stage
        if self.config_name is not None:
            result["config_name"] = self.config_name
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DraftMetadata:
        """Create DraftMetadata from a dictionary.

        Args:
            data: Dictionary with metadata fields.

        Returns:
            A new DraftMetadata instance.
        """
        return cls(
            draft_id=data.get("id", data.get("draft_id", "")),
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
            stage=data.get("stage"),
            complete=data.get("complete", False),
            config_name=data.get("config_name"),
        )


class ConfigDraftManager:
    """File-based draft manager for interactive config creation.

    Manages the lifecycle of configuration drafts: creation, incremental
    updates, finalization, and cleanup of stale drafts.

    Draft files are named ``{prefix}{draft_id}.yaml`` and stored in
    the output directory. When a config_name is provided, a named alias
    file ``{config_name}.yaml`` is also maintained.
    """

    def __init__(
        self,
        output_dir: Path,
        draft_prefix: str = "_draft-",
        max_age_hours: float = 24.0,
        metadata_key: str = "_draft",
    ) -> None:
        """Initialize the draft manager.

        Args:
            output_dir: Directory for draft and config files.
            draft_prefix: Prefix for draft file names.
            max_age_hours: Default maximum age for stale draft cleanup.
            metadata_key: Key used to store draft metadata in config files.
        """
        self._output_dir = output_dir
        self._draft_prefix = draft_prefix
        self._max_age_hours = max_age_hours
        self._metadata_key = metadata_key

    @property
    def output_dir(self) -> Path:
        """The output directory for drafts."""
        return self._output_dir

    def create_draft(
        self,
        config: dict[str, Any],
        stage: str | None = None,
    ) -> str:
        """Create a new draft from a config dict.

        Args:
            config: Configuration dictionary to save as draft.
            stage: Current wizard stage.

        Returns:
            The generated draft ID.
        """
        draft_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()

        metadata = DraftMetadata(
            draft_id=draft_id,
            created_at=now,
            last_updated=now,
            stage=stage,
        )

        self._write_draft(draft_id, config, metadata)
        logger.info(
            "Created draft %s at stage '%s'",
            draft_id,
            stage,
            extra={"draft_id": draft_id, "stage": stage},
        )
        return draft_id

    def update_draft(
        self,
        draft_id: str,
        config: dict[str, Any],
        stage: str | None = None,
        config_name: str | None = None,
    ) -> None:
        """Update an existing draft.

        Args:
            draft_id: The draft ID to update.
            config: Updated configuration dictionary.
            stage: Current wizard stage.
            config_name: Optional name for the config file alias.

        Raises:
            FileNotFoundError: If the draft file does not exist.
        """
        draft_path = self._draft_path(draft_id)
        if not draft_path.exists():
            raise FileNotFoundError(f"Draft not found: {draft_id}")

        existing = self._read_file(draft_path)
        existing_meta = existing.get(self._metadata_key, {})
        now = datetime.now(timezone.utc).isoformat()

        metadata = DraftMetadata(
            draft_id=draft_id,
            created_at=existing_meta.get("created_at", now),
            last_updated=now,
            stage=stage or existing_meta.get("stage"),
            config_name=config_name or existing_meta.get("config_name"),
        )

        self._write_draft(draft_id, config, metadata)

        # Also write named alias file if config_name is set
        if metadata.config_name:
            self._write_named_file(metadata.config_name, config, metadata)

        logger.info(
            "Updated draft %s at stage '%s'",
            draft_id,
            stage,
            extra={"draft_id": draft_id, "stage": stage},
        )

    def get_draft(
        self, draft_id: str
    ) -> tuple[dict[str, Any], DraftMetadata] | None:
        """Retrieve a draft and its metadata.

        Args:
            draft_id: The draft ID to retrieve.

        Returns:
            Tuple of (config_dict, metadata), or None if not found.
        """
        draft_path = self._draft_path(draft_id)
        if not draft_path.exists():
            return None

        data = self._read_file(draft_path)
        meta_dict = data.pop(self._metadata_key, {})
        metadata = DraftMetadata.from_dict(meta_dict)
        return data, metadata

    def finalize(
        self,
        draft_id: str,
        final_name: str | None = None,
    ) -> dict[str, Any]:
        """Finalize a draft into a completed configuration.

        Strips draft metadata, writes the final config file, and
        removes the draft file.

        Args:
            draft_id: The draft ID to finalize.
            final_name: Name for the final config file. If not provided,
                uses the config_name from draft metadata.

        Returns:
            The finalized configuration dict (without draft metadata).

        Raises:
            FileNotFoundError: If the draft does not exist.
            ValueError: If no final name can be determined.
        """
        result = self.get_draft(draft_id)
        if result is None:
            raise FileNotFoundError(f"Draft not found: {draft_id}")

        config, metadata = result
        name = final_name or metadata.config_name
        if not name:
            raise ValueError(
                "No final_name provided and draft has no config_name set"
            )

        # Write final file without metadata
        self._ensure_output_dir()
        final_path = self._output_dir / f"{name}.yaml"
        self._write_yaml(final_path, config)

        # Remove draft file
        draft_path = self._draft_path(draft_id)
        if draft_path.exists():
            draft_path.unlink()

        logger.info(
            "Finalized draft %s as '%s'",
            draft_id,
            name,
            extra={"draft_id": draft_id, "final_name": name},
        )
        return config

    def discard(self, draft_id: str) -> bool:
        """Discard a draft by removing its file.

        Args:
            draft_id: The draft ID to discard.

        Returns:
            True if the draft was found and removed, False otherwise.
        """
        draft_path = self._draft_path(draft_id)
        if draft_path.exists():
            draft_path.unlink()
            logger.info("Discarded draft %s", draft_id)
            return True
        return False

    def list_drafts(self) -> list[DraftMetadata]:
        """List all current drafts.

        Returns:
            List of DraftMetadata for all draft files.
        """
        result: list[DraftMetadata] = []
        if not self._output_dir.exists():
            return result

        for path in sorted(self._output_dir.glob(f"{self._draft_prefix}*.yaml")):
            try:
                data = self._read_file(path)
                meta_dict = data.get(self._metadata_key, {})
                if meta_dict:
                    result.append(DraftMetadata.from_dict(meta_dict))
            except Exception:
                logger.exception("Failed to read draft: %s", path)
        return result

    def cleanup_stale(self, max_age_hours: float | None = None) -> int:
        """Remove drafts older than the specified age.

        Also strips stale draft metadata blocks from named config files.

        Args:
            max_age_hours: Maximum age in hours. Defaults to the
                manager's configured max_age_hours.

        Returns:
            Number of stale drafts removed.
        """
        age_limit = max_age_hours if max_age_hours is not None else self._max_age_hours
        cutoff = time.time() - (age_limit * 3600)
        cleaned = 0

        if not self._output_dir.exists():
            return 0

        # Clean draft files
        for path in self._output_dir.glob(f"{self._draft_prefix}*.yaml"):
            try:
                data = self._read_file(path)
                meta = data.get(self._metadata_key, {})
                last_updated = meta.get("last_updated", "")
                if last_updated and _parse_timestamp(last_updated) < cutoff:
                    path.unlink()
                    cleaned += 1
                    logger.info("Cleaned stale draft: %s", path.name)
            except Exception:
                logger.exception("Failed to cleanup draft: %s", path)

        # Strip stale metadata from named config files
        for path in self._output_dir.glob("*.yaml"):
            if path.name.startswith(self._draft_prefix):
                continue
            try:
                data = self._read_file(path)
                meta = data.get(self._metadata_key, {})
                if not meta:
                    continue
                last_updated = meta.get("last_updated", "")
                if last_updated and _parse_timestamp(last_updated) < cutoff:
                    data.pop(self._metadata_key, None)
                    self._write_yaml(path, data)
                    logger.info(
                        "Stripped stale metadata from %s", path.name
                    )
            except Exception:
                logger.exception(
                    "Failed to strip metadata from: %s", path
                )

        return cleaned

    # -- Private helpers --

    def _draft_path(self, draft_id: str) -> Path:
        """Get the file path for a draft ID."""
        return self._output_dir / f"{self._draft_prefix}{draft_id}.yaml"

    def _ensure_output_dir(self) -> None:
        """Ensure the output directory exists."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _write_draft(
        self,
        draft_id: str,
        config: dict[str, Any],
        metadata: DraftMetadata,
    ) -> None:
        """Write a draft file with metadata."""
        self._ensure_output_dir()
        data = dict(config)
        data[self._metadata_key] = metadata.to_dict()
        self._write_yaml(self._draft_path(draft_id), data)

    def _write_named_file(
        self,
        name: str,
        config: dict[str, Any],
        metadata: DraftMetadata,
    ) -> None:
        """Write a named config file with draft metadata."""
        self._ensure_output_dir()
        data = dict(config)
        data[self._metadata_key] = metadata.to_dict()
        path = self._output_dir / f"{name}.yaml"
        self._write_yaml(path, data)

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        """Write a dict to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _read_file(path: Path) -> dict[str, Any]:
        """Read a YAML file and return its contents."""
        with open(path) as f:
            result = yaml.safe_load(f)
        if result is None:
            return {}
        if not isinstance(result, dict):
            return {}
        return result


def _parse_timestamp(iso_str: str) -> float:
    """Parse an ISO 8601 timestamp string to epoch seconds.

    Args:
        iso_str: ISO 8601 format timestamp string.

    Returns:
        Epoch seconds as float.
    """
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0

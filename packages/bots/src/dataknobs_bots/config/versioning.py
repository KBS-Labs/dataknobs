"""Configuration versioning for bot configurations.

This module provides version tracking and management for bot configurations,
enabling version history, rollback, and audit trails.

The versioning system supports:
- Immutable version records
- Replace-only updates (no incremental edits)
- Version history with timestamps and reasons
- Rollback to previous versions

Example:
    ```python
    manager = ConfigVersionManager()

    # Create initial version
    v1 = manager.create(
        config={"name": "MyBot", "llm": "gpt-4"},
        reason="Initial configuration",
    )

    # Update creates new version
    v2 = manager.update(
        config={"name": "MyBot", "llm": "gpt-4o"},
        reason="Upgrade to GPT-4o",
    )

    # Rollback to previous version
    v3 = manager.rollback(to_version=1, reason="Reverting upgrade")

    # Get history
    history = manager.get_history()
    ```
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class VersionConflictError(Exception):
    """Raised when there's a version conflict during update."""

    def __init__(
        self,
        message: str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        """Initialize VersionConflictError.

        Args:
            message: Error message
            expected_version: Version that was expected
            actual_version: Actual current version
        """
        super().__init__(message)
        self.expected_version = expected_version
        self.actual_version = actual_version


@dataclass
class ConfigVersion:
    """A single version record for a configuration.

    Each ConfigVersion is immutable once created. Updates create new versions
    rather than modifying existing ones.

    Attributes:
        version: Version number (1-indexed)
        config: The configuration data at this version
        timestamp: When this version was created (Unix timestamp)
        reason: Why this version was created
        previous_version: Version number this was derived from
        created_by: Who/what created this version
        metadata: Additional metadata about this version
    """

    version: int
    config: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    previous_version: int | None = None
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "version": self.version,
            "config": self.config,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "previous_version": self.previous_version,
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigVersion:
        """Create from dictionary.

        Args:
            data: Dictionary containing version data

        Returns:
            ConfigVersion instance
        """
        return cls(
            version=data["version"],
            config=data["config"],
            timestamp=data.get("timestamp", time.time()),
            reason=data.get("reason", ""),
            previous_version=data.get("previous_version"),
            created_by=data.get("created_by"),
            metadata=data.get("metadata", {}),
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on version number."""
        if not isinstance(other, ConfigVersion):
            return NotImplemented
        return self.version == other.version

    def __hash__(self) -> int:
        """Hash based on version number."""
        return hash(self.version)


class ConfigVersionManager:
    """Manages versioned configurations with history tracking.

    Provides create, update, rollback, and history operations for
    configuration versioning.

    This implementation uses replace-only semantics: each update replaces
    the entire configuration rather than applying incremental changes.

    Attributes:
        _versions: List of all version records
        _current_version: Current version number

    Example:
        ```python
        manager = ConfigVersionManager()

        # Create initial config
        v1 = manager.create(
            {"name": "Bot", "llm": "gpt-4"},
            reason="Initial",
        )
        assert v1.version == 1

        # Update config
        v2 = manager.update(
            {"name": "Bot", "llm": "gpt-4o"},
            reason="Upgrade LLM",
        )
        assert v2.version == 2
        assert v2.previous_version == 1

        # Rollback
        v3 = manager.rollback(to_version=1)
        assert v3.version == 3
        assert v3.config == v1.config
        ```
    """

    def __init__(self) -> None:
        """Initialize ConfigVersionManager with empty history."""
        self._versions: list[ConfigVersion] = []
        self._current_version: int = 0

    @property
    def current_version(self) -> int:
        """Get the current version number."""
        return self._current_version

    @property
    def current_config(self) -> dict[str, Any] | None:
        """Get the current configuration.

        Returns:
            Current config dict, or None if no versions exist
        """
        if not self._versions:
            return None
        return copy.deepcopy(self._versions[-1].config)

    def create(
        self,
        config: dict[str, Any],
        reason: str = "Initial configuration",
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConfigVersion:
        """Create the initial configuration version.

        Args:
            config: Configuration data
            reason: Reason for creation
            created_by: Who/what created this
            metadata: Additional metadata

        Returns:
            The created ConfigVersion

        Raises:
            ValueError: If versions already exist
        """
        if self._versions:
            raise ValueError(
                "Configuration already exists. Use update() to create new versions."
            )

        version = ConfigVersion(
            version=1,
            config=copy.deepcopy(config),
            reason=reason,
            previous_version=None,
            created_by=created_by,
            metadata=metadata or {},
        )

        self._versions.append(version)
        self._current_version = 1

        logger.info(
            "Created configuration version 1: %s",
            reason,
        )

        return version

    def update(
        self,
        config: dict[str, Any],
        reason: str = "Configuration update",
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> ConfigVersion:
        """Create a new version with updated configuration.

        Uses replace-only semantics: the entire configuration is replaced.

        Args:
            config: New configuration data
            reason: Reason for update
            created_by: Who/what created this
            metadata: Additional metadata
            expected_version: If provided, verify current version matches

        Returns:
            The new ConfigVersion

        Raises:
            ValueError: If no versions exist yet
            VersionConflictError: If expected_version doesn't match current
        """
        if not self._versions:
            raise ValueError(
                "No configuration exists. Use create() to create the initial version."
            )

        # Check for version conflict
        if expected_version is not None and expected_version != self._current_version:
            raise VersionConflictError(
                f"Version conflict: expected {expected_version}, "
                f"but current is {self._current_version}",
                expected_version=expected_version,
                actual_version=self._current_version,
            )

        new_version_num = self._current_version + 1

        version = ConfigVersion(
            version=new_version_num,
            config=copy.deepcopy(config),
            reason=reason,
            previous_version=self._current_version,
            created_by=created_by,
            metadata=metadata or {},
        )

        self._versions.append(version)
        self._current_version = new_version_num

        logger.info(
            "Created configuration version %d: %s",
            new_version_num,
            reason,
        )

        return version

    def rollback(
        self,
        to_version: int,
        reason: str | None = None,
        created_by: str | None = None,
    ) -> ConfigVersion:
        """Rollback to a previous version.

        Creates a new version with the configuration from the target version.

        Args:
            to_version: Version number to rollback to
            reason: Reason for rollback
            created_by: Who/what performed the rollback

        Returns:
            The new ConfigVersion (with new version number)

        Raises:
            ValueError: If target version doesn't exist
        """
        target = self.get_version(to_version)
        if target is None:
            raise ValueError(f"Version {to_version} not found")

        rollback_reason = reason or f"Rollback to version {to_version}"

        return self.update(
            config=target.config,
            reason=rollback_reason,
            created_by=created_by,
            metadata={"rollback_from": self._current_version, "rollback_to": to_version},
        )

    def get_version(self, version: int) -> ConfigVersion | None:
        """Get a specific version.

        Args:
            version: Version number to retrieve

        Returns:
            ConfigVersion if found, None otherwise
        """
        for v in self._versions:
            if v.version == version:
                return v
        return None

    def get_history(
        self,
        limit: int | None = None,
        since_version: int | None = None,
    ) -> list[ConfigVersion]:
        """Get version history.

        Args:
            limit: Maximum number of versions to return
            since_version: Only return versions after this version

        Returns:
            List of ConfigVersion records (newest first)
        """
        versions = list(reversed(self._versions))

        if since_version is not None:
            versions = [v for v in versions if v.version > since_version]

        if limit is not None:
            versions = versions[:limit]

        return versions

    def diff(
        self,
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        """Get differences between two versions.

        Returns a simple diff showing added, removed, and changed keys.

        Args:
            from_version: Starting version
            to_version: Ending version

        Returns:
            Dict with 'added', 'removed', and 'changed' keys

        Raises:
            ValueError: If either version doesn't exist
        """
        from_v = self.get_version(from_version)
        to_v = self.get_version(to_version)

        if from_v is None:
            raise ValueError(f"Version {from_version} not found")
        if to_v is None:
            raise ValueError(f"Version {to_version} not found")

        from_config = from_v.config
        to_config = to_v.config

        from_keys = set(from_config.keys())
        to_keys = set(to_config.keys())

        added = {k: to_config[k] for k in (to_keys - from_keys)}
        removed = {k: from_config[k] for k in (from_keys - to_keys)}

        changed = {}
        for key in from_keys & to_keys:
            if from_config[key] != to_config[key]:
                changed[key] = {
                    "from": from_config[key],
                    "to": to_config[key],
                }

        return {
            "from_version": from_version,
            "to_version": to_version,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize manager state to dictionary.

        Returns:
            Dictionary with all version history
        """
        return {
            "current_version": self._current_version,
            "versions": [v.to_dict() for v in self._versions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigVersionManager:
        """Create manager from serialized state.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored ConfigVersionManager
        """
        manager = cls()
        manager._current_version = data.get("current_version", 0)
        manager._versions = [
            ConfigVersion.from_dict(v) for v in data.get("versions", [])
        ]
        return manager

    def __len__(self) -> int:
        """Return number of versions."""
        return len(self._versions)

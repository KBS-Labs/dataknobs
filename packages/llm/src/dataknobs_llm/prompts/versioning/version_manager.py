"""Version management for prompts.

This module provides version control capabilities including:
- Creating and retrieving prompt versions
- Semantic version management
- Version history tracking
- Tagging and status management
"""

import re
import uuid
from typing import Any, Dict, List
from datetime import datetime

from dataknobs_llm.exceptions import VersioningError

from .types import (
    PromptVersion,
    VersionStatus,
)


class VersionManager:
    """Manages prompt versions with semantic versioning.

    Handles version creation, retrieval, and lifecycle management.
    Supports semantic versioning (major.minor.patch) and version tagging.

    Example:
        ```python
        manager = VersionManager(storage_backend)

        # Create a version
        v1 = await manager.create_version(
            name="greeting",
            prompt_type="system",
            template="Hello {{name}}!",
            version="1.0.0"
        )

        # Get latest version
        latest = await manager.get_version(
            name="greeting",
            prompt_type="system"
        )

        # Tag a version
        await manager.tag_version(v1.version_id, "production")
        ```
    """

    # Semantic version pattern: major.minor.patch
    VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

    def __init__(self, storage: Any | None = None):
        """Initialize version manager.

        Args:
            storage: Backend storage (dict for in-memory, database for persistence)
                    If None, uses in-memory dictionary
        """
        self.storage = storage if storage is not None else {}
        self._versions: Dict[str, PromptVersion] = {}  # version_id -> PromptVersion
        self._version_index: Dict[str, List[str]] = {}  # "{name}:{type}" -> [version_ids]

    async def create_version(
        self,
        name: str,
        prompt_type: str,
        template: str,
        version: str | None = None,
        defaults: Dict[str, Any] | None = None,
        validation: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
        created_by: str | None = None,
        parent_version: str | None = None,
        tags: List[str] | None = None,
        status: VersionStatus = VersionStatus.ACTIVE,
    ) -> PromptVersion:
        """Create a new prompt version.

        Args:
            name: Prompt name
            prompt_type: Prompt type ("system", "user", "message")
            template: Template content
            version: Semantic version (e.g., "1.2.3"). If None, auto-increments from latest
            defaults: Default parameter values
            validation: Validation configuration
            metadata: Additional metadata
            created_by: Creator username/ID
            parent_version: Previous version ID for history tracking
            tags: List of tags
            status: Initial version status

        Returns:
            Created PromptVersion

        Raises:
            VersioningError: If version format is invalid or version already exists
        """
        # Auto-increment version if not provided
        if version is None:
            version = await self._auto_increment_version(name, prompt_type)
            # If no parent_version specified, use the latest version
            if parent_version is None:
                latest = await self.get_version(name, prompt_type)
                if latest:
                    parent_version = latest.version_id
        else:
            # Validate version format
            if not self.VERSION_PATTERN.match(version):
                raise VersioningError(
                    f"Invalid version format: {version}. "
                    f"Expected semantic version (e.g., '1.0.0')"
                )

        # Check if version already exists
        key = self._make_key(name, prompt_type)
        existing_versions = await self.list_versions(name, prompt_type)
        if any(v.version == version for v in existing_versions):
            raise VersioningError(
                f"Version {version} already exists for {name} ({prompt_type})"
            )

        # Generate unique version ID
        version_id = str(uuid.uuid4())

        # Create version object
        prompt_version = PromptVersion(
            version_id=version_id,
            name=name,
            prompt_type=prompt_type,
            version=version,
            template=template,
            defaults=defaults or {},
            validation=validation,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            created_by=created_by,
            parent_version=parent_version,
            tags=tags or [],
            status=status,
        )

        # Store version
        self._versions[version_id] = prompt_version

        # Update index
        if key not in self._version_index:
            self._version_index[key] = []
        self._version_index[key].append(version_id)

        # Persist to backend if available
        if hasattr(self.storage, "set"):
            await self._persist_version(prompt_version)

        return prompt_version

    async def get_version(
        self,
        name: str,
        prompt_type: str,
        version: str = "latest",
        version_id: str | None = None,
    ) -> PromptVersion | None:
        """Retrieve a prompt version.

        Args:
            name: Prompt name
            prompt_type: Prompt type
            version: Version string or "latest" for most recent
            version_id: Specific version ID (takes precedence over version)

        Returns:
            PromptVersion if found, None otherwise
        """
        # Direct lookup by version_id
        if version_id:
            return self._versions.get(version_id)

        # Get all versions for this prompt
        versions = await self.list_versions(name, prompt_type)
        if not versions:
            return None

        # Return latest version
        if version == "latest":
            return self._get_latest_version(versions)

        # Find specific version
        for v in versions:
            if v.version == version:
                return v

        return None

    async def list_versions(
        self,
        name: str,
        prompt_type: str,
        tags: List[str] | None = None,
        status: VersionStatus | None = None,
    ) -> List[PromptVersion]:
        """List all versions of a prompt.

        Args:
            name: Prompt name
            prompt_type: Prompt type
            tags: Filter by tags (returns versions with ANY of these tags)
            status: Filter by status

        Returns:
            List of PromptVersion objects, sorted by version (newest first)
        """
        key = self._make_key(name, prompt_type)
        version_ids = self._version_index.get(key, [])

        versions = [self._versions[vid] for vid in version_ids]

        # Apply filters
        if tags:
            versions = [v for v in versions if any(t in v.tags for t in tags)]

        if status:
            versions = [v for v in versions if v.status == status]

        # Sort by version (newest first)
        return sorted(versions, key=lambda v: self._parse_version(v.version), reverse=True)

    async def tag_version(
        self,
        version_id: str,
        tag: str,
    ) -> PromptVersion:
        """Add a tag to a version.

        Args:
            version_id: Version ID to tag
            tag: Tag to add (e.g., "production", "deprecated")

        Returns:
            Updated PromptVersion

        Raises:
            VersioningError: If version not found
        """
        version = self._versions.get(version_id)
        if not version:
            raise VersioningError(f"Version not found: {version_id}")

        if tag not in version.tags:
            version.tags.append(tag)

            # Persist if backend available
            if hasattr(self.storage, "set"):
                await self._persist_version(version)

        return version

    async def untag_version(
        self,
        version_id: str,
        tag: str,
    ) -> PromptVersion:
        """Remove a tag from a version.

        Args:
            version_id: Version ID
            tag: Tag to remove

        Returns:
            Updated PromptVersion

        Raises:
            VersioningError: If version not found
        """
        version = self._versions.get(version_id)
        if not version:
            raise VersioningError(f"Version not found: {version_id}")

        if tag in version.tags:
            version.tags.remove(tag)

            # Persist if backend available
            if hasattr(self.storage, "set"):
                await self._persist_version(version)

        return version

    async def update_status(
        self,
        version_id: str,
        status: VersionStatus,
    ) -> PromptVersion:
        """Update version status.

        Args:
            version_id: Version ID
            status: New status

        Returns:
            Updated PromptVersion

        Raises:
            VersioningError: If version not found
        """
        version = self._versions.get(version_id)
        if not version:
            raise VersioningError(f"Version not found: {version_id}")

        version.status = status

        # Persist if backend available
        if hasattr(self.storage, "set"):
            await self._persist_version(version)

        return version

    async def delete_version(
        self,
        version_id: str,
    ) -> bool:
        """Delete a version.

        Note: This permanently removes the version. Consider using
        update_status() with ARCHIVED instead.

        Args:
            version_id: Version ID to delete

        Returns:
            True if deleted, False if not found
        """
        version = self._versions.get(version_id)
        if not version:
            return False

        # Remove from index
        key = self._make_key(version.name, version.prompt_type)
        if key in self._version_index:
            self._version_index[key] = [
                vid for vid in self._version_index[key]
                if vid != version_id
            ]

        # Remove from storage
        del self._versions[version_id]

        # Persist deletion if backend available
        if hasattr(self.storage, "delete"):
            await self.storage.delete(f"version:{version_id}")

        return True

    # ===== Helper Methods =====

    def _make_key(self, name: str, prompt_type: str) -> str:
        """Create index key for prompt name and type."""
        return f"{name}:{prompt_type}"

    def _parse_version(self, version: str) -> tuple:
        """Parse semantic version into (major, minor, patch) tuple."""
        match = self.VERSION_PATTERN.match(version)
        if not match:
            return (0, 0, 0)
        return tuple(int(x) for x in match.groups())

    def _get_latest_version(self, versions: List[PromptVersion]) -> PromptVersion | None:
        """Get the latest version from a list."""
        if not versions:
            return None

        # Filter out archived/deprecated if active versions exist
        active = [v for v in versions if v.status in (VersionStatus.ACTIVE, VersionStatus.PRODUCTION)]
        if active:
            versions = active

        # Sort by version number
        sorted_versions = sorted(
            versions,
            key=lambda v: self._parse_version(v.version),
            reverse=True
        )
        return sorted_versions[0]

    async def _auto_increment_version(
        self,
        name: str,
        prompt_type: str,
    ) -> str:
        """Auto-increment version number from latest version.

        Increments patch version by default (1.0.0 -> 1.0.1).
        If no versions exist, returns "1.0.0".
        """
        versions = await self.list_versions(name, prompt_type)
        if not versions:
            return "1.0.0"

        latest = self._get_latest_version(versions)
        if not latest:
            return "1.0.0"

        major, minor, patch = self._parse_version(latest.version)
        return f"{major}.{minor}.{patch + 1}"

    async def _persist_version(self, version: PromptVersion):
        """Persist version to backend storage."""
        if hasattr(self.storage, "set"):
            key = f"version:{version.version_id}"
            await self.storage.set(key, version.to_dict())

    async def get_version_history(
        self,
        name: str,
        prompt_type: str,
    ) -> List[PromptVersion]:
        """Get version history with parent relationships.

        Returns versions in chronological order (oldest first).

        Args:
            name: Prompt name
            prompt_type: Prompt type

        Returns:
            List of versions in chronological order
        """
        versions = await self.list_versions(name, prompt_type)
        return sorted(versions, key=lambda v: v.created_at)

"""Configuration schema for knowledge base ingestion.

This module provides configuration classes for loading and processing
documents from a directory into a knowledge base.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IngestionConfigError(Exception):
    """Error related to ingestion configuration."""

    pass


@dataclass
class FilePatternConfig:
    """Configuration for a specific file pattern.

    Allows overriding chunking and metadata settings for files
    matching a glob pattern.

    Attributes:
        pattern: Glob pattern to match files (e.g., "api/**/*.json")
        enabled: Whether to process files matching this pattern
        chunking: Override chunking settings for matched files
        text_template: Jinja2 template for JSON text generation
        text_fields: Fields to use for text generation (JSON)
        metadata_fields: Fields to include in chunk metadata
    """

    pattern: str
    enabled: bool = True
    chunking: dict[str, Any] | None = None
    text_template: str | None = None
    text_fields: list[str] | None = None
    metadata_fields: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {"pattern": self.pattern}
        if not self.enabled:
            result["enabled"] = False
        if self.chunking:
            result["chunking"] = self.chunking
        if self.text_template:
            result["text_template"] = self.text_template
        if self.text_fields:
            result["text_fields"] = self.text_fields
        if self.metadata_fields:
            result["metadata_fields"] = self.metadata_fields
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilePatternConfig:
        """Create from dictionary representation."""
        return cls(
            pattern=data["pattern"],
            enabled=data.get("enabled", True),
            chunking=data.get("chunking"),
            text_template=data.get("text_template"),
            text_fields=data.get("text_fields"),
            metadata_fields=data.get("metadata_fields"),
        )


@dataclass
class KnowledgeBaseConfig:
    r"""Configuration for knowledge base ingestion from a directory.

    Defines how documents in a directory should be processed, chunked,
    and prepared for embedding. Supports glob-pattern based overrides
    for different file types.

    Attributes:
        name: Name of the knowledge base
        default_chunking: Default chunking settings for all files
        default_quality_filter: Default quality filter settings
        patterns: List of file pattern configurations with overrides
        exclude_patterns: Glob patterns for files to skip
        default_metadata: Metadata to attach to all chunks

    Example:
        ```yaml
        name: product-docs
        default_chunking:
          max_chunk_size: 500
          chunk_overlap: 50

        patterns:
          - pattern: "api/**/*.json"
            text_template: "API: {{ method }} {{ path }}\\n{{ description }}"
            metadata_fields: [method, path, auth_required]

          - pattern: "guides/**/*.md"
            chunking:
              max_chunk_size: 800

        exclude_patterns:
          - "**/drafts/**"
          - "**/.git/**"
        ```
    """

    name: str
    default_chunking: dict[str, Any] = field(default_factory=lambda: {
        "max_chunk_size": 500,
        "chunk_overlap": 50,
    })
    default_quality_filter: dict[str, Any] | None = None
    patterns: list[FilePatternConfig] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    default_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: str | Path) -> KnowledgeBaseConfig:
        """Load configuration from a directory.

        Looks for `knowledge_base.yaml`, `knowledge_base.yml`, or
        `knowledge_base.json` in the directory.

        Args:
            directory: Directory containing the config file

        Returns:
            Loaded KnowledgeBaseConfig instance

        Raises:
            IngestionConfigError: If config file is invalid or missing
        """
        directory = Path(directory)
        config_path = cls._find_config_file(directory)

        if config_path is None:
            # Return default config with directory name
            logger.debug(
                f"No knowledge_base config found in {directory}, using defaults"
            )
            return cls(name=directory.name)

        try:
            data = cls._load_file(config_path)
        except Exception as e:
            raise IngestionConfigError(
                f"Failed to load config from {config_path}: {e}"
            ) from e

        return cls.from_dict(data, default_name=directory.name)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        default_name: str = "knowledge_base",
    ) -> KnowledgeBaseConfig:
        """Create from dictionary representation.

        Args:
            data: Configuration dictionary
            default_name: Default name if not specified in data

        Returns:
            KnowledgeBaseConfig instance
        """
        patterns = [
            FilePatternConfig.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("patterns", [])
        ]

        return cls(
            name=data.get("name", default_name),
            default_chunking=data.get("default_chunking", {
                "max_chunk_size": 500,
                "chunk_overlap": 50,
            }),
            default_quality_filter=data.get("default_quality_filter"),
            patterns=patterns,
            exclude_patterns=data.get("exclude_patterns", []),
            default_metadata=data.get("default_metadata", {}),
        )

    @classmethod
    def _find_config_file(cls, directory: Path) -> Path | None:
        """Find the config file in a directory.

        Args:
            directory: Directory to search

        Returns:
            Path to config file, or None if not found
        """
        for name in ["knowledge_base.yaml", "knowledge_base.yml", "knowledge_base.json"]:
            path = directory / name
            if path.exists():
                return path
        return None

    @classmethod
    def _load_file(cls, path: Path) -> dict[str, Any]:
        """Load and parse a config file.

        Args:
            path: Path to config file

        Returns:
            Parsed configuration dictionary
        """
        with open(path, encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError as err:
                    raise IngestionConfigError(
                        "PyYAML is required to load YAML config files. "
                        "Install with: pip install pyyaml"
                    ) from err
            else:
                data = json.load(f)

        if not isinstance(data, dict):
            raise IngestionConfigError(
                f"Config file must contain a dictionary: {path}"
            )

        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {"name": self.name}

        if self.default_chunking:
            result["default_chunking"] = self.default_chunking

        if self.default_quality_filter:
            result["default_quality_filter"] = self.default_quality_filter

        if self.patterns:
            result["patterns"] = [p.to_dict() for p in self.patterns]

        if self.exclude_patterns:
            result["exclude_patterns"] = self.exclude_patterns

        if self.default_metadata:
            result["default_metadata"] = self.default_metadata

        return result

    def get_pattern_config(self, filepath: str | Path) -> FilePatternConfig | None:
        """Get the pattern config that matches a file path.

        Returns the first matching pattern config, or None if no pattern matches.
        Patterns are checked in order, so more specific patterns should come first.

        Args:
            filepath: Path to check (relative to knowledge base root)

        Returns:
            Matching FilePatternConfig, or None
        """
        filepath = Path(filepath)

        for pattern_config in self.patterns:
            if pattern_config.enabled and self._matches_pattern(filepath, pattern_config.pattern):
                return pattern_config

        return None

    def is_excluded(self, filepath: str | Path) -> bool:
        """Check if a file path matches any exclude pattern.

        Args:
            filepath: Path to check (relative to knowledge base root)

        Returns:
            True if file should be excluded
        """
        filepath = Path(filepath)

        for pattern in self.exclude_patterns:
            if self._matches_pattern(filepath, pattern):
                return True

        return False

    def _matches_pattern(self, filepath: Path, pattern: str) -> bool:
        """Check if a filepath matches a glob pattern.

        Handles both fnmatch-style and glob-style patterns including `**`.

        Args:
            filepath: Path to check
            pattern: Glob pattern

        Returns:
            True if path matches pattern
        """
        from fnmatch import fnmatch

        filepath_str = str(filepath)

        # Handle ** patterns by using Path.match for recursive matching
        if "**" in pattern:
            # Path.match handles ** as recursive glob
            return filepath.match(pattern)
        else:
            # Use fnmatch for simple patterns
            return fnmatch(filepath_str, pattern)

    def get_chunking_config(self, filepath: str | Path) -> dict[str, Any]:
        """Get the effective chunking config for a file.

        Merges default chunking with any pattern-specific overrides.

        Args:
            filepath: Path to file

        Returns:
            Merged chunking configuration
        """
        config = self.default_chunking.copy()

        pattern_config = self.get_pattern_config(filepath)
        if pattern_config and pattern_config.chunking:
            config.update(pattern_config.chunking)

        return config

    def get_metadata(self, filepath: str | Path) -> dict[str, Any]:
        """Get the effective metadata for a file.

        Includes default metadata plus source file info.

        Args:
            filepath: Path to file

        Returns:
            Metadata dictionary
        """
        filepath = Path(filepath)
        metadata = self.default_metadata.copy()
        metadata["source"] = str(filepath)
        metadata["filename"] = filepath.name
        return metadata

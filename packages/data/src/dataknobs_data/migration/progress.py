"""Migration progress tracking, separate from migration logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MigrationProgress:
    """Track migration progress and statistics.
    
    Provides a clean separation between migration logic and progress tracking,
    allowing for flexible reporting without cluttering the migration code.
    """

    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> MigrationProgress:
        """Mark migration as started.
        
        Returns:
            Self for chaining
        """
        self.start_time = time.time()
        return self

    def finish(self) -> MigrationProgress:
        """Mark migration as finished.
        
        Returns:
            Self for chaining
        """
        self.end_time = time.time()
        return self

    @property
    def duration(self) -> float:
        """Get migration duration in seconds.
        
        Returns:
            Duration in seconds, or 0 if not started
        """
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def percent(self) -> float:
        """Get completion percentage.
        
        Returns:
            Percentage complete (0-100)
        """
        if self.total == 0:
            return 0.0
        return (self.processed / self.total) * 100

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage.
        
        Returns:
            Success rate (0-100)
        """
        if self.processed == 0:
            return 0.0
        return (self.succeeded / self.processed) * 100

    @property
    def is_complete(self) -> bool:
        """Check if migration is complete.
        
        Returns:
            True if all records have been processed
        """
        return self.total > 0 and self.processed >= self.total

    @property
    def has_errors(self) -> bool:
        """Check if migration had any errors.
        
        Returns:
            True if there were any failures
        """
        return self.failed > 0 or len(self.errors) > 0

    def record_success(self, record_id: str | None = None) -> MigrationProgress:
        """Record a successful migration.
        
        Args:
            record_id: Optional ID of successfully migrated record
            
        Returns:
            Self for chaining
        """
        self.processed += 1
        self.succeeded += 1
        return self

    def record_failure(
        self,
        error: str,
        record_id: str | None = None,
        exception: Exception | None = None
    ) -> MigrationProgress:
        """Record a failed migration.
        
        Args:
            error: Error message
            record_id: Optional ID of failed record
            exception: Optional exception that caused failure
            
        Returns:
            Self for chaining
        """
        self.processed += 1
        self.failed += 1

        error_info = {
            "error": error,
            "record_id": record_id,
            "timestamp": time.time()
        }

        if exception:
            error_info["exception"] = str(exception)
            error_info["exception_type"] = type(exception).__name__

        self.errors.append(error_info)
        return self

    def record_skip(self, reason: str, record_id: str | None = None) -> MigrationProgress:
        """Record a skipped record.
        
        Args:
            reason: Reason for skipping
            record_id: Optional ID of skipped record
            
        Returns:
            Self for chaining
        """
        self.processed += 1
        self.skipped += 1
        self.warnings.append(f"Skipped record {record_id}: {reason}")
        return self

    def add_warning(self, warning: str) -> MigrationProgress:
        """Add a warning message.
        
        Args:
            warning: Warning message
            
        Returns:
            Self for chaining
        """
        self.warnings.append(warning)
        return self

    def set_metadata(self, key: str, value: Any) -> MigrationProgress:
        """Store metadata about the migration.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Self for chaining
        """
        self.metadata[key] = value
        return self

    def merge(self, other: MigrationProgress) -> MigrationProgress:
        """Merge another progress object into this one.
        
        Useful for combining progress from parallel migrations.
        
        Args:
            other: Another MigrationProgress to merge
            
        Returns:
            Self for chaining
        """
        self.total += other.total
        self.processed += other.processed
        self.succeeded += other.succeeded
        self.failed += other.failed
        self.skipped += other.skipped
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

        # Merge metadata
        for key, value in other.metadata.items():
            if key in self.metadata:
                # Combine values if both exist
                if isinstance(self.metadata[key], list):
                    if isinstance(value, list):
                        self.metadata[key].extend(value)
                    else:
                        self.metadata[key].append(value)
                else:
                    # Convert to list if merging different values
                    self.metadata[key] = [self.metadata[key], value]
            else:
                self.metadata[key] = value

        return self

    def get_summary(self) -> str:
        """Get a human-readable summary of the migration progress.
        
        Returns:
            Summary string
        """
        lines = [
            f"Migration Progress: {self.percent:.1f}% complete",
            f"Total: {self.total} | Processed: {self.processed}",
            f"Succeeded: {self.succeeded} | Failed: {self.failed} | Skipped: {self.skipped}",
        ]

        if self.duration > 0:
            rate = self.processed / self.duration if self.duration > 0 else 0
            lines.append(f"Duration: {self.duration:.2f}s | Rate: {rate:.1f} records/s")

        if self.has_errors:
            lines.append(f"Errors: {len(self.errors)}")

        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert progress to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "total": self.total,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "percent": self.percent,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "is_complete": self.is_complete,
            "has_errors": self.has_errors,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.get_summary()

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"MigrationProgress(total={self.total}, processed={self.processed}, "
            f"succeeded={self.succeeded}, failed={self.failed})"
        )

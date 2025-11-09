"""Core type definitions for prompt versioning and A/B testing.

This module defines:
- Version data structures
- Experiment configurations
- Metrics tracking types
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from enum import Enum


class VersionStatus(Enum):
    """Status of a prompt version.

    Attributes:
        DRAFT: Version is in development
        ACTIVE: Version is active and can be used
        PRODUCTION: Version is deployed in production
        DEPRECATED: Version is deprecated but still available
        ARCHIVED: Version is archived and should not be used
    """
    DRAFT = "draft"
    ACTIVE = "active"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class PromptVersion:
    """Represents a versioned prompt.

    Attributes:
        version_id: Unique identifier for this version (auto-generated)
        name: Name of the prompt
        prompt_type: Type of prompt ("system", "user", "message")
        version: Semantic version string (e.g., "1.2.3")
        template: The prompt template content
        defaults: Default parameter values
        validation: Validation configuration
        metadata: Additional metadata (author, description, etc.)
        created_at: Timestamp when version was created
        created_by: Username/ID of creator
        parent_version: Previous version ID (for history tracking)
        tags: List of tags (e.g., ["production", "experiment-A"])
        status: Current status of this version
    """
    version_id: str
    name: str
    prompt_type: str
    version: str
    template: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str | None = None
    parent_version: str | None = None
    tags: List[str] = field(default_factory=list)
    status: VersionStatus = VersionStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "version_id": self.version_id,
            "name": self.name,
            "prompt_type": self.prompt_type,
            "version": self.version,
            "template": self.template,
            "defaults": self.defaults,
            "validation": self.validation,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "parent_version": self.parent_version,
            "tags": self.tags,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        data = data.copy()
        # Parse datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        # Parse status enum
        if isinstance(data.get("status"), str):
            data["status"] = VersionStatus(data["status"])
        return cls(**data)


@dataclass
class PromptVariant:
    """A variant in an A/B test experiment.

    Attributes:
        version: Version string of this variant
        weight: Traffic allocation weight (relative weight, must be > 0.0)
                Weights are normalized to sum to 1.0 when creating experiment
        description: Human-readable description
        metadata: Additional variant metadata
    """
    version: str
    weight: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate weight is positive."""
        if self.weight <= 0.0:
            raise ValueError(f"Variant weight must be positive, got {self.weight}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "version": self.version,
            "weight": self.weight,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVariant":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PromptExperiment:
    """Configuration for an A/B test experiment.

    Attributes:
        experiment_id: Unique identifier for this experiment
        name: Name of the prompt being tested
        prompt_type: Type of prompt ("system", "user", "message")
        variants: List of variants in this experiment
        traffic_split: Mapping of version to traffic percentage
        start_date: When experiment started
        end_date: When experiment ended (None if still running)
        status: Current status ("running", "paused", "completed")
        metrics: Aggregated metrics for the experiment
        metadata: Additional experiment metadata
    """
    experiment_id: str
    name: str
    prompt_type: str
    variants: List[PromptVariant]
    traffic_split: Dict[str, float]
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime | None = None
    status: str = "running"
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate traffic split sums to 1.0."""
        total = sum(self.traffic_split.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Traffic split must sum to 1.0, got {total}. "
                f"Split: {self.traffic_split}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "prompt_type": self.prompt_type,
            "variants": [v.to_dict() for v in self.variants],
            "traffic_split": self.traffic_split,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptExperiment":
        """Create from dictionary."""
        data = data.copy()
        # Parse datetimes
        if isinstance(data.get("start_date"), str):
            data["start_date"] = datetime.fromisoformat(data["start_date"])
        if isinstance(data.get("end_date"), str):
            data["end_date"] = datetime.fromisoformat(data["end_date"])
        # Parse variants
        if data.get("variants"):
            data["variants"] = [
                PromptVariant.from_dict(v) if isinstance(v, dict) else v
                for v in data["variants"]
            ]
        return cls(**data)


@dataclass
class PromptMetrics:
    """Performance metrics for a prompt version.

    Attributes:
        version_id: Version ID these metrics belong to
        total_uses: Total number of times this version was used
        success_count: Number of successful uses
        error_count: Number of errors/failures
        total_response_time: Total response time across all uses (seconds)
        total_tokens: Total tokens used across all uses
        user_ratings: List of user ratings (1-5 scale)
        last_used: Timestamp of last use
        metadata: Additional custom metrics
    """
    version_id: str
    total_uses: int = 0
    success_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    total_tokens: int = 0
    user_ratings: List[float] = field(default_factory=list)
    last_used: datetime | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_uses == 0:
            return 0.0
        return self.total_response_time / self.total_uses

    @property
    def avg_tokens(self) -> float:
        """Calculate average tokens per use."""
        if self.total_uses == 0:
            return 0.0
        return self.total_tokens / self.total_uses

    @property
    def avg_rating(self) -> float:
        """Calculate average user rating."""
        if not self.user_ratings:
            return 0.0
        return sum(self.user_ratings) / len(self.user_ratings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "version_id": self.version_id,
            "total_uses": self.total_uses,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_response_time": self.total_response_time,
            "total_tokens": self.total_tokens,
            "user_ratings": self.user_ratings,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "metadata": self.metadata,
            # Include computed properties
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "avg_tokens": self.avg_tokens,
            "avg_rating": self.avg_rating,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptMetrics":
        """Create from dictionary."""
        data = data.copy()
        # Parse datetime
        if isinstance(data.get("last_used"), str):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        # Remove computed properties (they're recalculated)
        for key in ["success_rate", "avg_response_time", "avg_tokens", "avg_rating"]:
            data.pop(key, None)
        return cls(**data)


@dataclass
class MetricEvent:
    """Single event for metrics tracking.

    Attributes:
        version_id: Version ID this event belongs to
        timestamp: When the event occurred
        success: Whether the use was successful
        response_time: Response time in seconds (None if not applicable)
        tokens: Number of tokens used (None if not applicable)
        user_rating: User rating 1-5 (None if not provided)
        metadata: Additional event metadata
    """
    version_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    response_time: float | None = None
    tokens: int | None = None
    user_rating: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "response_time": self.response_time,
            "tokens": self.tokens,
            "user_rating": self.user_rating,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricEvent":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

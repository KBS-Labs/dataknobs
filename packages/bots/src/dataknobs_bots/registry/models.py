"""Registration model for bot registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Registration:
    """Bot registration with metadata.

    Stores a bot configuration along with lifecycle metadata like
    timestamps and status. Used by registry backends to persist
    bot configurations.

    Attributes:
        bot_id: Unique bot identifier
        config: Bot configuration dictionary (should be portable)
        status: Registration status (active, inactive, error)
        created_at: When the registration was created
        updated_at: When the registration was last updated
        last_accessed_at: When the bot was last accessed

    Example:
        ```python
        reg = Registration(
            bot_id="my-bot",
            config={"bot": {"llm": {"$resource": "default", "type": "llm_providers"}}},
        )
        print(f"Bot {reg.bot_id} created at {reg.created_at}")

        # Serialize for storage
        data = reg.to_dict()

        # Restore from storage
        restored = Registration.from_dict(data)
        ```
    """

    bot_id: str
    config: dict[str, Any]
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with ISO format timestamps
        """
        return {
            "bot_id": self.bot_id,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Registration:
        """Create from dictionary.

        Args:
            data: Dictionary with registration data

        Returns:
            Registration instance
        """
        return cls(
            bot_id=data["bot_id"],
            config=data["config"],
            status=data.get("status", "active"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now(timezone.utc)
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else datetime.now(timezone.utc)
            ),
            last_accessed_at=(
                datetime.fromisoformat(data["last_accessed_at"])
                if data.get("last_accessed_at")
                else datetime.now(timezone.utc)
            ),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Registration(bot_id={self.bot_id!r}, status={self.status!r}, "
            f"created_at={self.created_at.isoformat() if self.created_at else None})"
        )

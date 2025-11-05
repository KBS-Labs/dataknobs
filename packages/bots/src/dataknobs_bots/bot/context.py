"""Bot execution context."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BotContext:
    """Runtime context for bot execution.

    Attributes:
        conversation_id: Unique identifier for the conversation
        client_id: Identifier for the client/tenant
        user_id: Optional user identifier
        session_metadata: Metadata for the session
        request_metadata: Metadata for the current request
    """

    conversation_id: str
    client_id: str
    user_id: str | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)

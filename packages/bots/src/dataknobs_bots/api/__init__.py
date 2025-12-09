"""FastAPI integration components for dataknobs_bots."""

from .dependencies import (
    BotManagerDep,
    get_bot_manager,
    init_bot_manager,
    reset_bot_manager,
)
from .exceptions import (
    APIError,
    BotCreationError,
    BotNotFoundError,
    ConfigurationError,
    ConversationNotFoundError,
    RateLimitError,
    ValidationError,
    api_error_handler,
    general_exception_handler,
    http_exception_handler,
    register_exception_handlers,
)

__all__ = [
    # Dependencies
    "get_bot_manager",
    "init_bot_manager",
    "reset_bot_manager",
    "BotManagerDep",
    # Exceptions
    "APIError",
    "BotNotFoundError",
    "BotCreationError",
    "ConversationNotFoundError",
    "ValidationError",
    "ConfigurationError",
    "RateLimitError",
    # Handlers
    "api_error_handler",
    "http_exception_handler",
    "general_exception_handler",
    "register_exception_handlers",
]

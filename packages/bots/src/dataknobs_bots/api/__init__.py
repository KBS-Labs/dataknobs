"""FastAPI integration components for dataknobs_bots."""

from .dependencies import (
    BotManagerDep,
    BotRegistryDep,
    get_bot_manager,
    get_bot_registry,
    init_bot_manager,
    init_bot_registry,
    reset_bot_manager,
    reset_bot_registry,
)
from .exceptions import (
    DEFAULT_ERROR_POLICY,
    MASKED_MESSAGE,
    APIError,
    BotCreationError,
    BotNotFoundError,
    ConfigurationError,
    ConversationNotFoundError,
    ErrorPolicy,
    RateLimitError,
    ValidationError,
    api_error_handler,
    dataknobs_error_handler,
    general_exception_handler,
    http_exception_handler,
    register_exception_handlers,
    resolve_error_policy,
)

__all__ = [
    # Dependencies
    "get_bot_registry",
    "init_bot_registry",
    "reset_bot_registry",
    "BotRegistryDep",
    # Dependencies (deprecated -- the four above replace these)
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
    # Error policy
    "ErrorPolicy",
    "DEFAULT_ERROR_POLICY",
    "MASKED_MESSAGE",
    "resolve_error_policy",
    # Handlers
    "api_error_handler",
    "dataknobs_error_handler",
    "http_exception_handler",
    "general_exception_handler",
    "register_exception_handlers",
]

"""Tests for API exceptions and dependencies."""

import pytest

from dataknobs_bots.api.exceptions import (
    APIError,
    BotCreationError,
    BotNotFoundError,
    ConfigurationError,
    ConversationNotFoundError,
    RateLimitError,
    ValidationError,
)
from dataknobs_bots.api.dependencies import (
    _BotManagerSingleton,
    get_bot_manager,
    init_bot_manager,
    reset_bot_manager,
)
from dataknobs_bots.bot.manager import BotManager
from dataknobs_common.exceptions import (
    ConfigurationError as CommonConfigurationError,
)
from dataknobs_common.exceptions import (
    DataknobsError,
)
from dataknobs_common.exceptions import (
    NotFoundError as CommonNotFoundError,
)
from dataknobs_common.exceptions import (
    ValidationError as CommonValidationError,
)


class TestAPIError:
    """Tests for APIError base class."""

    def test_basic_error(self):
        """Test creating a basic API error."""
        error = APIError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code == 500
        assert error.error_code == "APIError"

    def test_error_with_status_code(self):
        """Test creating an error with custom status code."""
        error = APIError("Not found", status_code=404)

        assert error.status_code == 404

    def test_error_with_detail(self):
        """Test creating an error with detail."""
        detail = {"field": "email", "reason": "invalid format"}
        error = APIError("Validation failed", detail=detail)

        assert error.detail == detail
        # Also accessible via context (inherited from DataknobsError)
        assert error.context == detail

    def test_error_with_error_code(self):
        """Test creating an error with custom error code."""
        error = APIError("Custom error", error_code="CUSTOM_ERROR")

        assert error.error_code == "CUSTOM_ERROR"

    def test_to_dict(self):
        """Test converting error to dictionary."""
        error = APIError(
            "Test error",
            status_code=400,
            detail={"key": "value"},
            error_code="TEST_ERROR",
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["detail"] == {"key": "value"}
        assert "timestamp" in error_dict

    def test_inherits_from_dataknobs_error(self):
        """Test that APIError inherits from DataknobsError."""
        error = APIError("Test")
        assert isinstance(error, DataknobsError)


class TestBotNotFoundError:
    """Tests for BotNotFoundError."""

    def test_basic_error(self):
        """Test creating a bot not found error."""
        error = BotNotFoundError("my-bot-id")

        assert "my-bot-id" in str(error)
        assert error.status_code == 404
        assert error.detail["bot_id"] == "my-bot-id"

    def test_inherits_from_common_not_found(self):
        """Test that BotNotFoundError inherits from CommonNotFoundError."""
        error = BotNotFoundError("my-bot-id")
        assert isinstance(error, CommonNotFoundError)


class TestBotCreationError:
    """Tests for BotCreationError."""

    def test_basic_error(self):
        """Test creating a bot creation error."""
        error = BotCreationError("my-bot-id", "Invalid configuration")

        assert "my-bot-id" in str(error)
        assert "Invalid configuration" in str(error)
        assert error.status_code == 500
        assert error.detail["bot_id"] == "my-bot-id"
        assert error.detail["reason"] == "Invalid configuration"


class TestConversationNotFoundError:
    """Tests for ConversationNotFoundError."""

    def test_basic_error(self):
        """Test creating a conversation not found error."""
        error = ConversationNotFoundError("conv-123")

        assert "conv-123" in str(error)
        assert error.status_code == 404
        assert error.detail["conversation_id"] == "conv-123"

    def test_inherits_from_common_not_found(self):
        """Test that ConversationNotFoundError inherits from CommonNotFoundError."""
        error = ConversationNotFoundError("conv-123")
        assert isinstance(error, CommonNotFoundError)


class TestValidationError:
    """Tests for ValidationError."""

    def test_basic_error(self):
        """Test creating a validation error."""
        error = ValidationError("Invalid input")

        assert str(error) == "Invalid input"
        assert error.status_code == 422

    def test_error_with_detail(self):
        """Test creating a validation error with detail."""
        detail = {"field": "email", "constraint": "must be valid email"}
        error = ValidationError("Validation failed", detail=detail)

        assert error.detail == detail

    def test_inherits_from_common_validation(self):
        """Test that ValidationError inherits from CommonValidationError."""
        error = ValidationError("Test")
        assert isinstance(error, CommonValidationError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_error(self):
        """Test creating a configuration error."""
        error = ConfigurationError("Invalid config")

        assert str(error) == "Invalid config"
        assert error.status_code == 500

    def test_error_with_config_key(self):
        """Test creating a configuration error with config key."""
        error = ConfigurationError("Invalid value", config_key="llm.model")

        assert error.detail["config_key"] == "llm.model"

    def test_inherits_from_common_configuration(self):
        """Test that ConfigurationError inherits from CommonConfigurationError."""
        error = ConfigurationError("Test")
        assert isinstance(error, CommonConfigurationError)


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_basic_error(self):
        """Test creating a rate limit error."""
        error = RateLimitError()

        assert "Rate limit exceeded" in str(error)
        assert error.status_code == 429

    def test_error_with_custom_message(self):
        """Test creating a rate limit error with custom message."""
        error = RateLimitError("Too many requests")

        assert str(error) == "Too many requests"

    def test_error_with_retry_after(self):
        """Test creating a rate limit error with retry_after."""
        error = RateLimitError(retry_after=60)

        assert error.detail["retry_after"] == 60


class TestBotManagerSingleton:
    """Tests for BotManager singleton management."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_bot_manager()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_bot_manager()

    def test_get_creates_default(self):
        """Test that get creates a default BotManager if none exists."""
        manager = get_bot_manager()

        assert manager is not None
        assert isinstance(manager, BotManager)

    def test_get_returns_same_instance(self):
        """Test that get returns the same instance."""
        manager1 = get_bot_manager()
        manager2 = get_bot_manager()

        assert manager1 is manager2

    def test_init_with_config_loader(self):
        """Test initializing with a config loader."""

        def loader(bot_id: str) -> dict:
            return {}

        manager = init_bot_manager(config_loader=loader)

        assert manager is not None
        assert get_bot_manager() is manager

    def test_reset_clears_instance(self):
        """Test that reset clears the singleton."""
        manager1 = get_bot_manager()
        reset_bot_manager()
        manager2 = get_bot_manager()

        assert manager1 is not manager2

    def test_singleton_class_get(self):
        """Test _BotManagerSingleton.get()."""
        manager = _BotManagerSingleton.get()
        assert isinstance(manager, BotManager)

    def test_singleton_class_init(self):
        """Test _BotManagerSingleton.init()."""
        manager = _BotManagerSingleton.init()
        assert isinstance(manager, BotManager)
        assert _BotManagerSingleton.get() is manager

    def test_singleton_class_reset(self):
        """Test _BotManagerSingleton.reset()."""
        _BotManagerSingleton.get()  # Create instance
        _BotManagerSingleton.reset()
        assert _BotManagerSingleton._instance is None

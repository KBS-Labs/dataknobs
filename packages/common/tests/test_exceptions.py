"""Tests for the exception framework."""

import pytest

from dataknobs_common.exceptions import (
    ConcurrencyError,
    ConfigurationError,
    DataknobsError,
    NotFoundError,
    OperationError,
    ResourceError,
    SerializationError,
    TimeoutError,
    ValidationError,
)


class TestDataknobsError:
    """Test the base DataknobsError class."""

    def test_basic_exception(self):
        """Test basic exception without context."""
        error = DataknobsError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.context == {}
        assert error.details == {}

    def test_exception_with_context(self):
        """Test exception with context dictionary."""
        error = DataknobsError(
            "Operation failed",
            context={"operation": "save", "item_id": "123"}
        )
        assert str(error) == "Operation failed"
        assert error.context == {"operation": "save", "item_id": "123"}
        assert error.details == {"operation": "save", "item_id": "123"}

    def test_exception_with_details(self):
        """Test exception with details dictionary (FSM-style)."""
        error = DataknobsError(
            "State transition failed",
            details={"state": "processing", "error_code": 500}
        )
        assert str(error) == "State transition failed"
        assert error.context == {"state": "processing", "error_code": 500}
        assert error.details == {"state": "processing", "error_code": 500}

    def test_details_takes_precedence(self):
        """Test that details parameter takes precedence over context."""
        error = DataknobsError(
            "Error",
            context={"key": "context_value"},
            details={"key": "details_value"}
        )
        assert error.context == {"key": "details_value"}
        assert error.details == {"key": "details_value"}

    def test_exception_inheritance(self):
        """Test that exception can be caught as Exception."""
        with pytest.raises(Exception):
            raise DataknobsError("Test error")

    def test_exception_catchable_as_base(self):
        """Test that specific exceptions can be caught as base."""
        with pytest.raises(DataknobsError):
            raise ValidationError("Invalid data")


class TestValidationError:
    """Test ValidationError."""

    def test_basic_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Invalid email format")
        assert str(error) == "Invalid email format"
        assert isinstance(error, DataknobsError)

    def test_validation_error_with_context(self):
        """Test validation error with field context."""
        error = ValidationError(
            "Field validation failed",
            context={"field": "email", "value": "not-an-email", "rule": "email_format"}
        )
        assert error.context["field"] == "email"
        assert error.context["value"] == "not-an-email"


class TestConfigurationError:
    """Test ConfigurationError."""

    def test_basic_configuration_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Missing required configuration")
        assert str(error) == "Missing required configuration"
        assert isinstance(error, DataknobsError)

    def test_configuration_error_with_context(self):
        """Test configuration error with key context."""
        error = ConfigurationError(
            "Configuration key not found",
            context={"key": "database.host", "available": ["database.port"]}
        )
        assert error.context["key"] == "database.host"
        assert "available" in error.context


class TestResourceError:
    """Test ResourceError."""

    def test_basic_resource_error(self):
        """Test basic resource error."""
        error = ResourceError("Failed to acquire connection")
        assert str(error) == "Failed to acquire connection"
        assert isinstance(error, DataknobsError)

    def test_resource_error_with_pool_context(self):
        """Test resource error with pool information."""
        error = ResourceError(
            "Connection pool exhausted",
            context={"pool_size": 10, "active": 10, "waiting": 5}
        )
        assert error.context["pool_size"] == 10
        assert error.context["active"] == 10


class TestNotFoundError:
    """Test NotFoundError."""

    def test_basic_not_found_error(self):
        """Test basic not found error."""
        error = NotFoundError("Item not found")
        assert str(error) == "Item not found"
        assert isinstance(error, DataknobsError)

    def test_not_found_error_with_id(self):
        """Test not found error with item identifier."""
        error = NotFoundError(
            "Record not found",
            context={"record_id": "user-123", "table": "users"}
        )
        assert error.context["record_id"] == "user-123"
        assert error.context["table"] == "users"


class TestOperationError:
    """Test OperationError."""

    def test_basic_operation_error(self):
        """Test basic operation error."""
        error = OperationError("Operation failed")
        assert str(error) == "Operation failed"
        assert isinstance(error, DataknobsError)

    def test_operation_error_with_details(self):
        """Test operation error with operation details."""
        error = OperationError(
            "Database operation failed",
            context={
                "operation": "update",
                "backend": "postgres",
                "error": "connection lost"
            }
        )
        assert error.context["operation"] == "update"


class TestConcurrencyError:
    """Test ConcurrencyError."""

    def test_basic_concurrency_error(self):
        """Test basic concurrency error."""
        error = ConcurrencyError("Lock acquisition failed")
        assert str(error) == "Lock acquisition failed"
        assert isinstance(error, DataknobsError)

    def test_concurrency_error_with_versions(self):
        """Test concurrency error with version information."""
        error = ConcurrencyError(
            "Optimistic lock failure",
            context={
                "record_id": "123",
                "expected_version": 5,
                "actual_version": 6
            }
        )
        assert error.context["expected_version"] == 5
        assert error.context["actual_version"] == 6


class TestSerializationError:
    """Test SerializationError."""

    def test_basic_serialization_error(self):
        """Test basic serialization error."""
        error = SerializationError("Failed to serialize data")
        assert str(error) == "Failed to serialize data"
        assert isinstance(error, DataknobsError)

    def test_serialization_error_with_format(self):
        """Test serialization error with format details."""
        error = SerializationError(
            "JSON encoding failed",
            context={"format": "json", "field": "created_at", "value": "invalid"}
        )
        assert error.context["format"] == "json"


class TestTimeoutError:
    """Test TimeoutError."""

    def test_basic_timeout_error(self):
        """Test basic timeout error."""
        error = TimeoutError("Operation timed out")
        assert str(error) == "Operation timed out"
        assert isinstance(error, DataknobsError)

    def test_timeout_error_with_duration(self):
        """Test timeout error with timeout details."""
        error = TimeoutError(
            "Query timeout exceeded",
            context={"query": "SELECT * FROM large_table", "timeout_seconds": 30}
        )
        assert error.context["timeout_seconds"] == 30


class TestCustomExceptions:
    """Test creating custom exceptions from base."""

    def test_custom_exception_basic(self):
        """Test creating custom exception."""
        class MyPackageError(DataknobsError):
            pass

        error = MyPackageError("Custom error")
        assert str(error) == "Custom error"
        assert isinstance(error, DataknobsError)

    def test_custom_exception_with_custom_init(self):
        """Test custom exception with custom __init__."""
        class ItemError(DataknobsError):
            def __init__(self, item_id: str, message: str):
                super().__init__(
                    f"Item '{item_id}': {message}",
                    context={"item_id": item_id}
                )

        error = ItemError("item-123", "not found")
        assert str(error) == "Item 'item-123': not found"
        assert error.context["item_id"] == "item-123"

    def test_custom_exception_catchable_as_base(self):
        """Test custom exception can be caught as base."""
        class MyError(DataknobsError):
            pass

        with pytest.raises(DataknobsError):
            raise MyError("Test")


class TestExceptionHierarchy:
    """Test exception hierarchy and catching."""

    def test_catch_all_dataknobs_errors(self):
        """Test catching all dataknobs errors with base class."""
        exceptions_to_test = [
            ValidationError("test"),
            ConfigurationError("test"),
            ResourceError("test"),
            NotFoundError("test"),
            OperationError("test"),
            ConcurrencyError("test"),
            SerializationError("test"),
            TimeoutError("test"),
        ]

        for exc in exceptions_to_test:
            with pytest.raises(DataknobsError):
                raise exc

    def test_specific_exception_catching(self):
        """Test catching specific exception types."""
        with pytest.raises(ValidationError):
            raise ValidationError("test")

        with pytest.raises(NotFoundError):
            raise NotFoundError("test")

    def test_exception_context_preserved(self):
        """Test that context is preserved when catching."""
        context_data = {"key": "value", "number": 42}

        try:
            raise ValidationError("Test", context=context_data)
        except DataknobsError as e:
            assert e.context == context_data
            assert e.details == context_data

"""Tests for custom exceptions in dataknobs_data package."""

import pytest
from dataknobs_data.exceptions import (
    DataknobsDataError,
    RecordNotFoundError,
    RecordValidationError,
    FieldTypeError,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseOperationError,
    QueryError,
    SerializationError,
    DataFormatError,
    BackendNotFoundError,
    ConfigurationError,
    ConcurrencyError,
    TransactionError,
    MigrationError,
)


class TestDataknobsDataError:
    """Tests for base exception class."""

    def test_base_exception(self):
        """Test base exception can be instantiated and raised."""
        with pytest.raises(DataknobsDataError, match="Test error"):
            raise DataknobsDataError("Test error")

    def test_inheritance(self):
        """Test that base exception inherits from Exception."""
        assert issubclass(DataknobsDataError, Exception)


class TestRecordNotFoundError:
    """Tests for RecordNotFoundError."""

    def test_record_not_found_with_id(self):
        """Test exception with record ID."""
        error = RecordNotFoundError("test-id-123")
        assert error.id == "test-id-123"
        assert str(error) == "Record with ID 'test-id-123' not found"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(RecordNotFoundError, DataknobsDataError)


class TestRecordValidationError:
    """Tests for RecordValidationError."""

    def test_validation_error_without_field(self):
        """Test validation error without field name."""
        error = RecordValidationError("Invalid data format")
        assert error.field_name is None
        assert str(error) == "Invalid data format"

    def test_validation_error_with_field(self):
        """Test validation error with field name."""
        error = RecordValidationError("must be positive", field_name="age")
        assert error.field_name == "age"
        assert str(error) == "Field 'age': must be positive"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(RecordValidationError, DataknobsDataError)


class TestFieldTypeError:
    """Tests for FieldTypeError."""

    def test_field_type_error(self):
        """Test field type error with all parameters."""
        error = FieldTypeError("age", "int", "str")
        assert error.field_name == "age"
        assert error.expected_type == "int"
        assert error.actual_type == "str"
        assert str(error) == "Field 'age' type mismatch: expected int, got str"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(FieldTypeError, DataknobsDataError)


class TestDatabaseError:
    """Tests for DatabaseError."""

    def test_database_error(self):
        """Test general database error."""
        with pytest.raises(DatabaseError, match="Database is locked"):
            raise DatabaseError("Database is locked")

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(DatabaseError, DataknobsDataError)


class TestDatabaseConnectionError:
    """Tests for DatabaseConnectionError."""

    def test_connection_error(self):
        """Test database connection error."""
        error = DatabaseConnectionError("PostgreSQL", "Connection refused")
        assert error.backend == "PostgreSQL"
        assert str(error) == "Failed to connect to PostgreSQL backend: Connection refused"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(DatabaseConnectionError, DataknobsDataError)


class TestDatabaseOperationError:
    """Tests for DatabaseOperationError."""

    def test_operation_error(self):
        """Test database operation error."""
        error = DatabaseOperationError("INSERT", "Duplicate key violation")
        assert error.operation == "INSERT"
        assert str(error) == "Database operation 'INSERT' failed: Duplicate key violation"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(DatabaseOperationError, DataknobsDataError)


class TestQueryError:
    """Tests for QueryError."""

    def test_query_error_without_query_object(self):
        """Test query error without query object."""
        error = QueryError("Invalid field name")
        assert error.query is None
        assert str(error) == "Query error: Invalid field name"

    def test_query_error_with_query_object(self):
        """Test query error with query object."""
        # Mock query object
        class MockQuery:
            pass

        query = MockQuery()
        error = QueryError("Syntax error", query=query)
        assert error.query is query
        assert str(error) == "Query error: Syntax error"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(QueryError, DataknobsDataError)


class TestSerializationError:
    """Tests for SerializationError."""

    def test_serialization_error(self):
        """Test serialization error."""
        error = SerializationError("JSON", "Invalid character at position 42")
        assert error.format == "JSON"
        assert str(error) == "Serialization error (JSON): Invalid character at position 42"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(SerializationError, DataknobsDataError)


class TestDataFormatError:
    """Tests for DataFormatError."""

    def test_data_format_error(self):
        """Test data format error."""
        error = DataFormatError("CSV", "Missing required header")
        assert error.format == "CSV"
        assert str(error) == "Data format error (CSV): Missing required header"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(DataFormatError, DataknobsDataError)


class TestBackendNotFoundError:
    """Tests for BackendNotFoundError."""

    def test_backend_not_found_without_available(self):
        """Test backend not found without available list."""
        error = BackendNotFoundError("redis")
        assert error.backend == "redis"
        assert error.available == []
        assert str(error) == "Backend 'redis' not found"

    def test_backend_not_found_with_available(self):
        """Test backend not found with available list."""
        available = ["memory", "file", "postgres"]
        error = BackendNotFoundError("redis", available=available)
        assert error.backend == "redis"
        assert error.available == available
        assert str(error) == "Backend 'redis' not found. Available backends: memory, file, postgres"

    def test_backend_not_found_with_empty_available(self):
        """Test backend not found with empty available list."""
        error = BackendNotFoundError("redis", available=[])
        assert error.backend == "redis"
        assert error.available == []
        assert str(error) == "Backend 'redis' not found"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(BackendNotFoundError, DataknobsDataError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError("max_connections", "Value must be positive")
        assert error.parameter == "max_connections"
        assert str(error) == "Configuration error for 'max_connections': Value must be positive"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(ConfigurationError, DataknobsDataError)


class TestConcurrencyError:
    """Tests for ConcurrencyError."""

    def test_concurrency_error(self):
        """Test concurrency error."""
        error = ConcurrencyError("Resource locked by another process")
        assert str(error) == "Concurrency error: Resource locked by another process"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(ConcurrencyError, DataknobsDataError)


class TestTransactionError:
    """Tests for TransactionError."""

    def test_transaction_error(self):
        """Test transaction error."""
        error = TransactionError("Rollback due to deadlock")
        assert str(error) == "Transaction error: Rollback due to deadlock"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(TransactionError, DataknobsDataError)


class TestMigrationError:
    """Tests for MigrationError."""

    def test_migration_error(self):
        """Test migration error."""
        error = MigrationError("PostgreSQL", "S3", "Schema incompatibility")
        assert error.source == "PostgreSQL"
        assert error.target == "S3"
        assert str(error) == "Migration from PostgreSQL to S3 failed: Schema incompatibility"

    def test_is_dataknobs_error(self):
        """Test that it's a DataknobsDataError."""
        assert issubclass(MigrationError, DataknobsDataError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy and relationships."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from DataknobsDataError."""
        exceptions = [
            RecordNotFoundError,
            RecordValidationError,
            FieldTypeError,
            DatabaseError,
            DatabaseConnectionError,
            DatabaseOperationError,
            QueryError,
            SerializationError,
            DataFormatError,
            BackendNotFoundError,
            ConfigurationError,
            ConcurrencyError,
            TransactionError,
            MigrationError,
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, DataknobsDataError)
            assert issubclass(exc_class, Exception)

    def test_exception_catching_hierarchy(self):
        """Test that catching base exception catches all derived exceptions."""
        test_cases = [
            (RecordNotFoundError, "test-id"),
            (RecordValidationError, "Invalid"),
            (DatabaseError, "Error"),
            (ConcurrencyError, "Locked"),
            (TransactionError, "Failed"),
        ]
        
        for exc_class, *args in test_cases:
            try:
                raise exc_class(*args)
            except DataknobsDataError as e:
                # Should catch all derived exceptions
                assert isinstance(e, exc_class)
            except Exception:
                pytest.fail(f"{exc_class.__name__} was not caught as DataknobsDataError")
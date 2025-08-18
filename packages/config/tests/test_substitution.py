"""Test environment variable substitution."""

import os
import pytest

from dataknobs_config.substitution import VariableSubstitution


class TestVariableSubstitution:
    """Test environment variable substitution functionality."""
    
    @pytest.fixture
    def substitution(self):
        """Create a VariableSubstitution instance."""
        return VariableSubstitution()
    
    def test_simple_substitution(self, substitution, monkeypatch):
        """Test simple variable substitution."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        result = substitution.substitute("${TEST_VAR}")
        assert result == "test_value"
    
    def test_substitution_with_default(self, substitution, monkeypatch):
        """Test substitution with default value."""
        # Ensure variable doesn't exist
        monkeypatch.delenv("MISSING_VAR", raising=False)
        
        result = substitution.substitute("${MISSING_VAR:default_value}")
        assert result == "default_value"
    
    def test_substitution_with_dash_default(self, substitution, monkeypatch):
        """Test bash-style substitution with default."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        
        result = substitution.substitute("${MISSING_VAR:-default_value}")
        assert result == "default_value"
    
    def test_missing_variable_error(self, substitution, monkeypatch):
        """Test that missing variable without default raises error."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        
        with pytest.raises(ValueError, match="Environment variable 'MISSING_VAR' not found"):
            substitution.substitute("${MISSING_VAR}")
    
    def test_mixed_content(self, substitution, monkeypatch):
        """Test substitution in mixed content."""
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        
        result = substitution.substitute("postgresql://${HOST}:${PORT}/mydb")
        assert result == "postgresql://localhost:5432/mydb"
    
    def test_type_conversion(self, substitution, monkeypatch):
        """Test that single variables can be converted to appropriate types."""
        monkeypatch.setenv("INT_VAR", "42")
        monkeypatch.setenv("FLOAT_VAR", "3.14")
        monkeypatch.setenv("BOOL_TRUE", "true")
        monkeypatch.setenv("BOOL_FALSE", "false")
        monkeypatch.setenv("STRING_VAR", "hello")
        
        assert substitution.substitute("${INT_VAR}") == 42
        assert substitution.substitute("${FLOAT_VAR}") == 3.14
        assert substitution.substitute("${BOOL_TRUE}") is True
        assert substitution.substitute("${BOOL_FALSE}") is False
        assert substitution.substitute("${STRING_VAR}") == "hello"
    
    def test_dict_substitution(self, substitution, monkeypatch):
        """Test substitution in dictionary."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_NAME", "testdb")
        
        config = {
            "host": "${DB_HOST}",
            "port": "${DB_PORT}",
            "database": "${DB_NAME}",
            "ssl": True
        }
        
        result = substitution.substitute(config)
        assert result == {
            "host": "localhost",
            "port": 5432,  # Converted to int
            "database": "testdb",
            "ssl": True
        }
    
    def test_list_substitution(self, substitution, monkeypatch):
        """Test substitution in list."""
        monkeypatch.setenv("HOST1", "server1")
        monkeypatch.setenv("HOST2", "server2")
        
        config = ["${HOST1}", "${HOST2}", "server3"]
        
        result = substitution.substitute(config)
        assert result == ["server1", "server2", "server3"]
    
    def test_nested_substitution(self, substitution, monkeypatch):
        """Test substitution in nested structures."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("DB_HOST", "prod.db.com")
        monkeypatch.setenv("CACHE_SIZE", "1000")
        
        config = {
            "environment": "${ENV}",
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT:5432}",
                "options": {
                    "timeout": 30,
                    "cache_size": "${CACHE_SIZE}"
                }
            },
            "servers": ["${HOST1:server1}", "${HOST2:server2}"]
        }
        
        result = substitution.substitute(config)
        assert result == {
            "environment": "production",
            "database": {
                "host": "prod.db.com",
                "port": 5432,
                "options": {
                    "timeout": 30,
                    "cache_size": 1000
                }
            },
            "servers": ["server1", "server2"]
        }
    
    def test_empty_default(self, substitution, monkeypatch):
        """Test substitution with empty default value."""
        monkeypatch.delenv("OPTIONAL_VAR", raising=False)
        
        result = substitution.substitute("${OPTIONAL_VAR:}")
        assert result == ""
    
    def test_has_variables(self, substitution):
        """Test detection of variable patterns."""
        assert substitution.has_variables("${VAR}") is True
        assert substitution.has_variables("text ${VAR} text") is True
        assert substitution.has_variables("${VAR:default}") is True
        assert substitution.has_variables("no variables") is False
        assert substitution.has_variables({"key": "${VAR}"}) is True
        assert substitution.has_variables(["${VAR}", "text"]) is True
        assert substitution.has_variables(42) is False
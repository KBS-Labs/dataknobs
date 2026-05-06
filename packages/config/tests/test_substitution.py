"""Test environment variable substitution."""

import warnings

import pytest

from dataknobs_config.substitution import VariableSubstitution


class TestVariableSubstitution:
    """Test environment variable substitution functionality."""

    @pytest.fixture
    def substitution(self):
        """Create a VariableSubstitution instance.

        Suppresses the deprecation warning emitted on construction so the
        behavioral-parity tests below do not flood pytest output. The
        deprecation warning itself is asserted in
        ``test_emits_deprecation_warning``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

    def test_emits_deprecation_warning(self):
        """Constructing VariableSubstitution emits a DeprecationWarning that
        points at the canonical helper.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VariableSubstitution()
        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecation_warnings, "expected a DeprecationWarning"
        assert any(
            "substitute_env_vars" in str(w.message) for w in deprecation_warnings
        )

    def test_question_mark_msg_passes_through_canonical(self, substitution, monkeypatch):
        """${VAR:?multi word msg} preserves the canonical helper's wording.

        The shim only rewrites errors that originated from the bare
        ``${VAR}`` form. Bash-style ``${VAR:?error_msg}`` errors carry a
        ``bash_form`` flag on the underlying ``RequiredEnvVarError`` so
        we can pass them through unchanged regardless of whether the
        message text happens to look like an identifier.
        """
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(
            ValueError,
            match=r"Required environment variable not set: DB password is required",
        ):
            substitution.substitute("${MISSING_VAR:?DB password is required}")

    def test_question_mark_msg_single_word_not_rewritten(self, substitution, monkeypatch):
        """${VAR:?Required} with a single-word custom msg is NOT rewritten.

        Regression test: an earlier version of the shim rewrote the
        canonical "Required environment variable not set: X" message back
        to "Environment variable 'X' not found" whenever ``X`` matched
        ``[A-Za-z_][A-Za-z0-9_]*``. That guard misfired for single-word
        custom error messages: ``${PORT:?Required}`` produced an error
        like ``Environment variable 'Required' not found``, making the
        user's chosen error word look like a variable name. The shim now
        keys off the typed exception's ``bash_form`` attribute, so any
        ``${VAR:?msg}`` error is passed through verbatim — even when
        ``msg`` is a single identifier.
        """
        monkeypatch.delenv("PORT", raising=False)
        with pytest.raises(
            ValueError,
            match=r"^Required environment variable not set: Required$",
        ):
            substitution.substitute("${PORT:?Required}")

    def test_question_mark_empty_msg_uses_var_name_canonical(
        self, substitution, monkeypatch
    ):
        """${VAR:?} with empty msg falls back to the var name + canonical wording.

        ``${FOO:?}`` is bash-style with an empty error message. The
        canonical helper substitutes the variable name as the message,
        producing ``Required environment variable not set: FOO``. The
        shim must preserve this canonical wording (it is bash-form),
        rather than rewriting it to the historical
        ``Environment variable 'FOO' not found`` as the bare ``${FOO}``
        form would.
        """
        monkeypatch.delenv("FOO", raising=False)
        with pytest.raises(
            ValueError,
            match=r"^Required environment variable not set: FOO$",
        ):
            substitution.substitute("${FOO:?}")

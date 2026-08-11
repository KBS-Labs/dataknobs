"""Test environment variable substitution."""

import warnings

import pytest

from dataknobs_config.binding_resolver import ConfigBindingResolver
from dataknobs_config.environment_aware import EnvironmentAwareConfig
from dataknobs_config.environment_config import EnvironmentConfig
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

        config = {"host": "${DB_HOST}", "port": "${DB_PORT}", "database": "${DB_NAME}", "ssl": True}

        result = substitution.substitute(config)
        assert result == {
            "host": "localhost",
            "port": 5432,  # Converted to int
            "database": "testdb",
            "ssl": True,
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
                "options": {"timeout": 30, "cache_size": "${CACHE_SIZE}"},
            },
            "servers": ["${HOST1:server1}", "${HOST2:server2}"],
        }

        result = substitution.substitute(config)
        assert result == {
            "environment": "production",
            "database": {
                "host": "prod.db.com",
                "port": 5432,
                "options": {"timeout": 30, "cache_size": 1000},
            },
            "servers": ["server1", "server2"],
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
        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings, "expected a DeprecationWarning"
        assert any("substitute_env_vars" in str(w.message) for w in deprecation_warnings)

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

    def test_question_mark_empty_msg_uses_var_name_canonical(self, substitution, monkeypatch):
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


# ---------------------------------------------------------------------------
# Substitute-once-per-source (the item-199 defect class)
# ---------------------------------------------------------------------------

#: A secret whose *value* contains a ``${...}`` sequence. Nothing about this
#: is exotic — generated passwords routinely contain ``$`` and ``{``.
SECRET_WITH_VAR_SYNTAX = "p${GUARD_INNER}ss"

#: The value ``GUARD_INNER`` would expand to if the secret's own text were
#: ever re-interpreted as a template. Its presence in a result is the defect.
INNER_VALUE = "INJECTED"


def _environment(*, substitute_vars: bool = True) -> EnvironmentConfig:
    """An environment whose ``password`` is read from ``${GUARD_PW}``."""
    return EnvironmentConfig.from_dict(
        {
            "name": "test",
            "resources": {
                "databases": {
                    "main": {
                        "backend": "postgres",
                        "password": "${GUARD_PW}",
                    }
                }
            },
        },
        substitute_vars=substitute_vars,
    )


def _via_direct_read(env: EnvironmentConfig) -> str:
    """Access path: ``EnvironmentConfig.get_resource`` directly."""
    return str(env.get_resource("databases", "main")["password"])


def _via_resource_ref(env: EnvironmentConfig) -> str:
    """Access path: a ``$resource`` ref through ``resolve_for_build``."""
    app = EnvironmentAwareConfig(
        config={"db": {"$resource": "main", "type": "databases"}},
        environment=env,
    )
    return str(app.resolve_for_build()["db"]["password"])


def _via_binding_resolver(env: EnvironmentConfig) -> str:
    """Access path: ``ConfigBindingResolver``, read **at the factory**.

    The value asserted here is the one a live resource is actually
    constructed from — there is no intermediate config artifact on this
    path for anyone to inspect.
    """
    built: dict[str, object] = {}

    def factory(**config: object) -> object:
        built.update(config)
        return object()

    resolver = ConfigBindingResolver(env)
    resolver.register_factory("databases", factory)
    resolver.resolve("databases", "main")
    return str(built["password"])


#: The two paths that run substitution *themselves*, downstream of the
#: environment. These are the compositions a caller cannot avoid, and the
#: ones the double expansion lived in.
RESOLUTION_PATHS = [
    pytest.param(_via_resource_ref, id="resource-ref"),
    pytest.param(_via_binding_resolver, id="binding-resolver"),
]

#: Every path, including the direct read that performs no substitution of
#: its own and simply returns what the environment holds.
ACCESS_PATHS = [pytest.param(_via_direct_read, id="direct"), *RESOLUTION_PATHS]


class TestSubstituteOncePerSource:
    """A value's *content* is never re-interpreted as a template.

    ``${VAR}`` substitution used to run in two layers that did not know
    about each other, so a value pulled from an already-substituted
    ``EnvironmentConfig`` was substituted a **second** time. The second
    pass expanded the *output* of the first, which means the content of a
    secret was read as a template and replaced with the value of whatever
    unrelated variable that content happened to name.

    The property below is the whole item, stated once: whatever access path
    a config value travels, it is substituted exactly once.
    """

    @pytest.fixture(autouse=True)
    def _guard_vars(self, monkeypatch):
        monkeypatch.setenv("GUARD_PW", SECRET_WITH_VAR_SYNTAX)
        monkeypatch.setenv("GUARD_INNER", INNER_VALUE)

    @pytest.mark.parametrize("access_path", ACCESS_PATHS)
    def test_value_containing_var_syntax_is_substituted_exactly_once(self, access_path):
        """The literal value of ``$GUARD_PW`` arrives intact on every path."""
        resolved = access_path(_environment())

        assert resolved == SECRET_WITH_VAR_SYNTAX
        assert INNER_VALUE not in resolved, (
            "the secret's own text was re-expanded as a template — "
            f"an unrelated variable's value leaked into it: {resolved!r}"
        )

    @pytest.mark.parametrize("access_path", RESOLUTION_PATHS)
    def test_unsubstituted_environment_still_substitutes_exactly_once(self, access_path):
        """``substitute_vars=False`` keeps the downstream pass load-bearing.

        A directly-constructed (or explicitly unsubstituted) environment has
        never had substitution applied, so the resolution layers must still
        run it — exactly once. This cell was already correct before the fix
        and is the one most at risk from a naive "just skip the second pass"
        change, which is why it is asserted rather than assumed.
        """
        resolved = access_path(_environment(substitute_vars=False))

        assert resolved == SECRET_WITH_VAR_SYNTAX
        assert INNER_VALUE not in resolved

    def test_direct_read_of_unsubstituted_environment_stays_raw(self):
        """The one cell that is *meant* to stay raw stays raw.

        Reading straight off an environment built with
        ``substitute_vars=False`` yields the unexpanded ref: the caller
        asked for raw refs and holds a config with none applied. The fix
        must not start substituting on this path — ``substituted_view()``
        is non-mutating precisely so it cannot, and a caller holding an
        unsubstituted config keeps the config it asked for even after a
        resolution layer has read through it.
        """
        env = _environment(substitute_vars=False)

        assert env.get_resource("databases", "main")["password"] == "${GUARD_PW}"

        # Reading through a resolution layer must not have mutated it.
        _via_resource_ref(env)
        _via_binding_resolver(env)

        assert env.get_resource("databases", "main")["password"] == "${GUARD_PW}"
        assert env.substituted is False

    def test_ordinary_values_are_unaffected(self):
        """A value with no ``${`` in it behaves identically to before.

        The second pass was always a no-op for these; removing it must be
        observable only for the defect class.
        """
        env = EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {
                    "databases": {
                        "main": {
                            "backend": "postgres",
                            "dsn": "postgresql://u:p@h/db",
                            "port": "${GUARD_PORT:5432}",
                        }
                    }
                },
            }
        )

        built = EnvironmentAwareConfig(
            config={"db": {"$resource": "main", "type": "databases"}},
            environment=env,
        ).resolve_for_build()["db"]

        assert built["dsn"] == "postgresql://u:p@h/db"
        assert built["port"] == "5432"
        assert env.get_resource("databases", "main")["dsn"] == ("postgresql://u:p@h/db")


class TestToDictRoundTripHazard:
    """``from_dict(to_dict(x))`` re-expands — characterized, not endorsed.

    This is **not** a specification. ``to_dict()`` emits already-substituted
    values and ``from_dict()`` substitutes by default, so composing them
    double-expands any value containing ``${``. Unlike the resolution
    layers, this composition is one the *caller* performs and can spell
    correctly today, which is why it is documented rather than fixed
    (D-199-5). The correct spelling is asserted alongside the wrong one.

    Revisit under **199-FU3** if callers round-trip configs often enough
    that documenting this stops being sufficient.
    """

    @pytest.fixture(autouse=True)
    def _guard_vars(self, monkeypatch):
        monkeypatch.setenv("GUARD_PW", SECRET_WITH_VAR_SYNTAX)
        monkeypatch.setenv("GUARD_INNER", INNER_VALUE)

    def test_round_trip_through_to_dict_re_expands_and_needs_substitute_vars_false(
        self,
    ):
        original = _environment()
        assert original.get_resource("databases", "main")["password"] == (SECRET_WITH_VAR_SYNTAX)

        # The naive round-trip: WRONG. Pinned so it cannot silently worsen.
        naive = EnvironmentConfig.from_dict(original.to_dict())
        assert naive.get_resource("databases", "main")["password"] == (f"p{INNER_VALUE}ss")

        # The correct spelling, available today.
        correct = EnvironmentConfig.from_dict(original.to_dict(), substitute_vars=False)
        assert correct.get_resource("databases", "main")["password"] == (SECRET_WITH_VAR_SYNTAX)

"""Tests for the string reference system."""

import pytest

from dataknobs_config import Config, ConfigNotFoundError, InvalidReferenceError
from dataknobs_config.references import ReferenceResolver


class TestReferenceParser:
    """Test reference parsing."""

    def test_parse_named_reference(self):
        """Test parsing named references."""
        config = Config()
        resolver = ReferenceResolver(config)

        type_name, selector = resolver.parse_reference("xref:database[primary]")
        assert type_name == "database"
        assert selector == "primary"

    def test_parse_index_reference(self):
        """Test parsing index references."""
        config = Config()
        resolver = ReferenceResolver(config)

        type_name, selector = resolver.parse_reference("xref:cache[0]")
        assert type_name == "cache"
        assert selector == 0

        type_name, selector = resolver.parse_reference("xref:cache[42]")
        assert type_name == "cache"
        assert selector == 42

    def test_parse_negative_index(self):
        """Test parsing negative index references."""
        config = Config()
        resolver = ReferenceResolver(config)

        type_name, selector = resolver.parse_reference("xref:database[-1]")
        assert type_name == "database"
        assert selector == -1

    def test_parse_no_selector(self):
        """Test parsing references without selector."""
        config = Config()
        resolver = ReferenceResolver(config)

        type_name, selector = resolver.parse_reference("xref:database")
        assert type_name == "database"
        assert selector == 0  # Default to first item

    def test_invalid_reference_format(self):
        """Test invalid reference formats."""
        config = Config()
        resolver = ReferenceResolver(config)

        with pytest.raises(InvalidReferenceError):
            resolver.parse_reference("invalid:database[0]")

        with pytest.raises(InvalidReferenceError):
            resolver.parse_reference("database[0]")

        with pytest.raises(InvalidReferenceError):
            resolver.parse_reference("xref:")


class TestReferenceResolution:
    """Test reference resolution."""

    def test_resolve_by_name(self):
        """Test resolving references by name."""
        config = Config(
            {
                "database": [
                    {"name": "primary", "host": "localhost"},
                    {"name": "secondary", "host": "backup"},
                ]
            }
        )

        resolved = config.resolve_reference("xref:database[primary]")
        assert resolved["host"] == "localhost"

        resolved = config.resolve_reference("xref:database[secondary]")
        assert resolved["host"] == "backup"

    def test_resolve_by_index(self):
        """Test resolving references by index."""
        config = Config(
            {"cache": [{"name": "redis1", "port": 6379}, {"name": "redis2", "port": 6380}]}
        )

        resolved = config.resolve_reference("xref:cache[0]")
        assert resolved["port"] == 6379

        resolved = config.resolve_reference("xref:cache[1]")
        assert resolved["port"] == 6380

    def test_resolve_negative_index(self):
        """Test resolving with negative index."""
        config = Config({"server": [{"name": "web1"}, {"name": "web2"}, {"name": "web3"}]})

        resolved = config.resolve_reference("xref:server[-1]")
        assert resolved["name"] == "web3"

    def test_resolve_no_selector(self):
        """Test resolving without selector."""
        config = Config({"database": [{"name": "only", "host": "localhost"}]})

        resolved = config.resolve_reference("xref:database")
        assert resolved["name"] == "only"

    def test_resolve_nonexistent(self):
        """Test resolving nonexistent references."""
        config = Config({"database": [{"name": "db1"}]})

        with pytest.raises(ConfigNotFoundError):
            config.resolve_reference("xref:nonexistent[0]")

        with pytest.raises(ConfigNotFoundError):
            config.resolve_reference("xref:database[nonexistent]")


class TestReferenceBuilder:
    """Test reference building."""

    def test_build_named_reference(self):
        """Test building named references."""
        config = Config()

        ref = config.build_reference("database", "primary")
        assert ref == "xref:database[primary]"

    def test_build_index_reference(self):
        """Test building index references."""
        config = Config()

        ref = config.build_reference("cache", 1)
        assert ref == "xref:cache[1]"

        # Index 0 can omit selector
        ref = config.build_reference("database", 0)
        assert ref == "xref:database"


class TestNestedReferences:
    """Test nested reference resolution."""

    def test_nested_reference_in_config(self):
        """Test resolving nested references within configurations."""
        config = Config(
            {
                "database": [{"name": "primary", "host": "localhost", "port": 5432}],
                "api": [{"name": "main", "database": "xref:database[primary]"}],
            }
        )

        api = config.resolve_reference("xref:api[main]")

        # The nested reference should be resolved
        assert isinstance(api["database"], dict)
        assert api["database"]["host"] == "localhost"
        assert api["database"]["port"] == 5432

    def test_reference_in_list(self):
        """Test references within lists."""
        config = Config(
            {
                "database": [{"name": "db1", "host": "host1"}, {"name": "db2", "host": "host2"}],
                "cluster": [
                    {"name": "main", "databases": ["xref:database[db1]", "xref:database[db2]"]}
                ],
            }
        )

        cluster = config.resolve_reference("xref:cluster[main]")

        assert len(cluster["databases"]) == 2
        assert cluster["databases"][0]["host"] == "host1"
        assert cluster["databases"][1]["host"] == "host2"

    def test_circular_reference_detection(self):
        """Test that circular references are detected."""
        # This would require setting up a circular reference scenario
        # The implementation should detect and raise an error
        pass  # Implementation depends on how circular refs are created


class TestReferenceIntegration:
    """Test reference system integration with Config class."""

    def test_cross_type_references(self):
        """Test references across different types."""
        config = Config(
            {
                "database": [{"name": "db", "host": "dbhost"}],
                "cache": [{"name": "redis", "host": "cachehost"}],
                "app": [
                    {
                        "name": "webapp",
                        "database": "xref:database[db]",
                        "cache": "xref:cache[redis]",
                    }
                ],
            }
        )

        app = config.get("app", "webapp")

        # Raw get should return the reference strings
        assert app["database"] == "xref:database[db]"
        assert app["cache"] == "xref:cache[redis]"

        # Resolve should follow the references
        app_resolved = config.resolve_reference("xref:app[webapp]")
        assert app_resolved["database"]["host"] == "dbhost"
        assert app_resolved["cache"]["host"] == "cachehost"

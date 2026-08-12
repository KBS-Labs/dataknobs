"""Tests for ``dataknobs_common.config_loading`` helpers."""

from __future__ import annotations

import builtins
import logging
from pathlib import Path

import pytest

from dataknobs_common.config_loading import (
    DEFAULT_CONFIG_EXTENSIONS,
    ConfigLoadError,
    ConfigParseError,
    ConfigPathEscapeError,
    ConfigShapeError,
    ConfigUnsupportedFormatError,
    ConfigYAMLNotInstalledError,
    find_config_file,
    load_yaml_or_json,
    parse_yaml_or_json,
)


class TestFindConfigFile:
    def test_yaml_priority_over_json(self, tmp_path: Path) -> None:
        (tmp_path / "x.yaml").write_text("a: 1\n")
        (tmp_path / "x.json").write_text('{"a": 1}')
        match = find_config_file(tmp_path, "x")
        assert match is not None
        assert match.suffix == ".yaml"

    def test_yml_picked_when_yaml_absent(self, tmp_path: Path) -> None:
        (tmp_path / "x.yml").write_text("a: 1\n")
        match = find_config_file(tmp_path, "x")
        assert match is not None
        assert match.suffix == ".yml"

    def test_json_picked_when_no_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "x.json").write_text('{"a": 1}')
        match = find_config_file(tmp_path, "x")
        assert match is not None
        assert match.suffix == ".json"

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        assert find_config_file(tmp_path, "x") is None

    def test_accepts_str_dir(self, tmp_path: Path) -> None:
        (tmp_path / "x.yaml").write_text("a: 1\n")
        match = find_config_file(str(tmp_path), "x")
        assert match is not None
        assert match.name == "x.yaml"

    def test_custom_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "x.toml").write_text("")
        match = find_config_file(tmp_path, "x", extensions=(".toml",))
        assert match is not None
        assert match.suffix == ".toml"
        # Default extensions don't include .toml.
        assert find_config_file(tmp_path, "x") is None

    def test_default_extension_order(self) -> None:
        assert DEFAULT_CONFIG_EXTENSIONS == (".yaml", ".yml", ".json")

    def test_extensions_without_leading_dot_are_normalized(self, tmp_path: Path) -> None:
        """Callers passing ``"yaml"`` (no dot) should match ``.yaml`` files."""
        (tmp_path / "x.yaml").write_text("a: 1\n")
        match = find_config_file(tmp_path, "x", extensions=("yaml",))
        assert match is not None
        assert match.suffix == ".yaml"


class TestFindConfigFileContainment:
    """A name addresses a file *inside* ``config_dir``, and nowhere else.

    The name reaching this helper is not always the caller's own literal:
    it can be an ``extends:`` value read out of a config file, an
    environment name read from ``DATAKNOBS_ENVIRONMENT``, or the output of
    a consumer-supplied resolver. A name that composes a path outside the
    search directory is rejected before any file is probed.
    """

    @staticmethod
    def _tree(tmp_path: Path) -> tuple[Path, Path]:
        """Build ``<tmp>/configs`` beside a readable ``<tmp>/outside``."""
        configs = tmp_path / "configs"
        configs.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.yaml").write_text("leaked: true\n")
        return configs, outside

    def test_parent_segment_raises(self, tmp_path: Path) -> None:
        configs, _ = self._tree(tmp_path)
        with pytest.raises(ConfigPathEscapeError) as excinfo:
            find_config_file(configs, "../outside/secret")
        assert "../outside/secret" in str(excinfo.value)

    def test_absolute_name_raises(self, tmp_path: Path) -> None:
        """``Path.__truediv__`` discards the directory for an absolute name."""
        configs, outside = self._tree(tmp_path)
        with pytest.raises(ConfigPathEscapeError):
            find_config_file(configs, str(outside / "secret"))

    def test_parent_segment_after_a_subdirectory_raises(self, tmp_path: Path) -> None:
        configs, _ = self._tree(tmp_path)
        with pytest.raises(ConfigPathEscapeError):
            find_config_file(configs, "sub/../../outside/secret")

    def test_it_fails_closed_before_probing(self, tmp_path: Path) -> None:
        """No file exists at the escaping name -- it still raises, not ``None``.

        A guard that only fired when the target happened to exist would
        report "not found" for a probe and "escaped" for a hit, which is a
        disclosure oracle rather than a guard.
        """
        configs, _ = self._tree(tmp_path)
        with pytest.raises(ConfigPathEscapeError):
            find_config_file(configs, "../outside/does-not-exist")

    def test_the_error_is_a_config_load_error(self, tmp_path: Path) -> None:
        """Consumers already catch the base type and re-raise as their own."""
        configs, _ = self._tree(tmp_path)
        with pytest.raises(ConfigLoadError):
            find_config_file(configs, "../outside/secret")

    def test_a_subdirectory_name_still_resolves(self, tmp_path: Path) -> None:
        """The layout-convention case: a name may address a subdirectory."""
        configs, _ = self._tree(tmp_path)
        (configs / "domains").mkdir()
        (configs / "domains" / "child.yaml").write_text("a: 1\n")
        match = find_config_file(configs, "domains/child")
        assert match == configs / "domains" / "child.yaml"

    def test_an_interior_parent_segment_that_stays_inside_resolves(self, tmp_path: Path) -> None:
        """``sub/../x`` never leaves the directory, so the guard allows it.

        The probe still composes the name as written, so ``sub`` has to
        exist for the candidate to resolve — the guard reads the name, it
        does not rewrite what gets probed.
        """
        configs, _ = self._tree(tmp_path)
        (configs / "sub").mkdir()
        (configs / "x.yaml").write_text("a: 1\n")
        match = find_config_file(configs, "sub/../x")
        assert match is not None
        assert match.exists()

    def test_allow_outside_opts_back_out(self, tmp_path: Path) -> None:
        """The escape hatch for a caller that deliberately addresses a sibling tree."""
        configs, outside = self._tree(tmp_path)
        match = find_config_file(configs, "../outside/secret", allow_outside=True)
        assert match is not None
        assert match.resolve() == (outside / "secret.yaml").resolve()

    def test_allow_outside_warns_when_it_is_load_bearing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silent bypass would leave the guard unauditable in a deployment."""
        configs, _ = self._tree(tmp_path)
        with caplog.at_level(logging.WARNING, logger="dataknobs_common.config_loading"):
            find_config_file(configs, "../outside/secret", allow_outside=True)
        assert any("../outside/secret" in r.getMessage() for r in caplog.records)

    def test_allow_outside_is_silent_for_a_contained_name(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning marks a name that *did* escape, not the flag being set."""
        configs, _ = self._tree(tmp_path)
        (configs / "x.yaml").write_text("a: 1\n")
        with caplog.at_level(logging.WARNING, logger="dataknobs_common.config_loading"):
            assert find_config_file(configs, "x", allow_outside=True) is not None
        assert caplog.records == []


class TestLoadYamlOrJson:
    def test_loads_yaml_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("a: 1\nb: hello\n")
        assert load_yaml_or_json(p) == {"a": 1, "b": "hello"}

    def test_loads_json_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "c.json"
        p.write_text('{"a": 1, "b": "hello"}')
        assert load_yaml_or_json(p) == {"a": 1, "b": "hello"}

    def test_yml_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yml"
        p.write_text("a: 1\n")
        assert load_yaml_or_json(p) == {"a": 1}

    def test_extension_case_insensitive(self, tmp_path: Path) -> None:
        p = tmp_path / "c.YAML"
        p.write_text("a: 1\n")
        assert load_yaml_or_json(p) == {"a": 1}

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        p = tmp_path / "c.json"
        p.write_text('{"a": 1}')
        assert load_yaml_or_json(str(p)) == {"a": 1}

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "c.toml"
        p.write_text("")
        with pytest.raises(ConfigUnsupportedFormatError):
            load_yaml_or_json(p)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml_or_json(tmp_path / "missing.yaml")

    def test_require_dict_rejects_list_root(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("- a\n- b\n")
        with pytest.raises(ConfigShapeError):
            load_yaml_or_json(p)

    def test_require_dict_false_allows_list_root(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("- a\n- b\n")
        assert load_yaml_or_json(p, require_dict=False) == ["a", "b"]

    def test_require_dict_false_allows_none_root(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("")
        assert load_yaml_or_json(p, require_dict=False) is None

    def test_yaml_parse_error_wrapped(self, tmp_path: Path) -> None:
        p = tmp_path / "c.yaml"
        p.write_text("a: [unclosed\n")
        with pytest.raises(ConfigParseError):
            load_yaml_or_json(p)

    def test_json_parse_error_wrapped(self, tmp_path: Path) -> None:
        p = tmp_path / "c.json"
        p.write_text("{not json")
        with pytest.raises(ConfigParseError):
            load_yaml_or_json(p)

    def test_source_name_in_load_error(self, tmp_path: Path) -> None:
        p = tmp_path / "named.yaml"
        p.write_text("- a\n- b\n")
        with pytest.raises(ConfigShapeError, match=str(p)):
            load_yaml_or_json(p)

    def test_empty_json_raises_parse_error(self, tmp_path: Path) -> None:
        """Empty JSON files raise ``ConfigParseError`` even with
        ``require_dict=False`` — stdlib ``json.loads("")`` is a parse
        error. (Empty YAML, by contrast, parses to ``None`` and passes
        through with ``require_dict=False``.)
        """
        p = tmp_path / "empty.json"
        p.write_text("")
        with pytest.raises(ConfigParseError):
            load_yaml_or_json(p, require_dict=False)

    def test_oserror_propagates_uncaught(self, tmp_path: Path) -> None:
        """``OSError`` (here: ``IsADirectoryError``) propagates
        unwrapped — it is not a ``ConfigLoadError`` subclass and the
        helper documents this as a caller-wraps responsibility.
        """
        # A directory with a .yaml suffix passes the extension check
        # but fails when ``open()`` is called against it.
        p = tmp_path / "dir.yaml"
        p.mkdir()
        with pytest.raises(OSError) as excinfo:
            load_yaml_or_json(p)
        # Must NOT be a ConfigLoadError subclass — caller is expected
        # to wrap OSError separately if it wants a uniform error type.
        assert not isinstance(excinfo.value, ConfigLoadError)


class TestParseYamlOrJson:
    def test_yaml_from_bytes(self) -> None:
        assert parse_yaml_or_json(b"a: 1\n", format="yaml") == {"a": 1}

    def test_json_from_bytes(self) -> None:
        assert parse_yaml_or_json(b'{"a": 1}', format="json") == {"a": 1}

    def test_yaml_from_str(self) -> None:
        assert parse_yaml_or_json("a: 1\n", format="yaml") == {"a": 1}

    def test_json_from_str(self) -> None:
        assert parse_yaml_or_json('{"a": 1}', format="json") == {"a": 1}

    def test_invalid_format(self) -> None:
        with pytest.raises(ConfigUnsupportedFormatError):
            parse_yaml_or_json(
                "",
                format="toml",  # type: ignore[arg-type]
            )

    def test_source_name_in_shape_error(self) -> None:
        with pytest.raises(ConfigShapeError, match="my-source"):
            parse_yaml_or_json("- a", format="yaml", source_name="my-source")

    def test_source_name_in_parse_error(self) -> None:
        with pytest.raises(ConfigParseError, match="my-source"):
            parse_yaml_or_json("{not json", format="json", source_name="my-source")

    def test_default_source_name_used_when_omitted(self) -> None:
        with pytest.raises(ConfigShapeError, match=r"<yaml>"):
            parse_yaml_or_json("- a", format="yaml")

    def test_require_dict_false_allows_list(self) -> None:
        assert parse_yaml_or_json("- a\n- b\n", format="yaml", require_dict=False) == ["a", "b"]

    def test_non_utf8_bytes_raise_parse_error(self) -> None:
        """Non-UTF-8 bytes are wrapped as ``ConfigParseError`` so the
        helper has a single error type — the stdlib
        ``UnicodeDecodeError`` does not leak past the helper boundary.
        """
        # Latin-1 encoded "é" — invalid as UTF-8.
        bad_bytes = b"name: \xe9"
        with pytest.raises(ConfigParseError, match="UTF-8"):
            parse_yaml_or_json(bad_bytes, format="yaml")

    def test_non_utf8_bytes_includes_source_name(self) -> None:
        bad_bytes = b"\xe9"
        with pytest.raises(ConfigParseError, match="my-source"):
            parse_yaml_or_json(bad_bytes, format="yaml", source_name="my-source")


class TestExceptionHierarchy:
    """Every helper-raised error subclasses ``ConfigLoadError``."""

    @pytest.mark.parametrize(
        "subclass",
        [
            ConfigParseError,
            ConfigShapeError,
            ConfigUnsupportedFormatError,
            ConfigYAMLNotInstalledError,
        ],
    )
    def test_subclass_of_base(self, subclass: type[Exception]) -> None:
        assert issubclass(subclass, ConfigLoadError)


class TestYAMLNotInstalled:
    def test_parse_yaml_raises_when_pyyaml_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``parse_yaml_or_json(format='yaml')`` raises a clear error
        when PyYAML cannot be imported.
        """
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ConfigYAMLNotInstalledError, match=r"<yaml>"):
            parse_yaml_or_json("a: 1", format="yaml")

    def test_yaml_not_installed_uses_explicit_source_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ConfigYAMLNotInstalledError, match="my-source"):
            parse_yaml_or_json("a: 1", format="yaml", source_name="my-source")

    def test_load_yaml_raises_when_pyyaml_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``load_yaml_or_json`` on a ``.yaml`` file surfaces the same
        error path through the file-loader.
        """
        p = tmp_path / "c.yaml"
        p.write_text("a: 1\n")
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ConfigYAMLNotInstalledError):
            load_yaml_or_json(p)

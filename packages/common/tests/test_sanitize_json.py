"""Tests for sanitize_for_json enhancements and validate_json_safe.

Validates:
- on_drop="silent" (backward-compatible default)
- on_drop="warn" logs at WARNING with key path
- on_drop="error" raises SerializationError listing dropped paths
- _path tracking for nested dicts, lists, and dataclasses
- validate_json_safe returns paths without modifying input
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from dataknobs_common.exceptions import SerializationError
from dataknobs_common.serialization import sanitize_for_json, validate_json_safe


class TestSanitizeForJsonSilent:
    """on_drop='silent' preserves backward-compatible behavior."""

    def test_primitives_pass_through(self) -> None:
        assert sanitize_for_json(None) is None
        assert sanitize_for_json(True) is True
        assert sanitize_for_json(42) == 42
        assert sanitize_for_json(3.14) == 3.14
        assert sanitize_for_json("hello") == "hello"

    def test_dict_drops_non_serializable_silently(self, caplog: Any) -> None:
        data = {"a": 1, "fn": lambda: None, "b": "ok"}
        with caplog.at_level(logging.DEBUG):
            result = sanitize_for_json(data)
        assert result == {"a": 1, "b": "ok"}

    def test_list_drops_non_serializable_silently(self) -> None:
        data = [1, lambda: None, "ok"]
        result = sanitize_for_json(data)
        assert result == [1, "ok"]

    def test_nested_dict(self) -> None:
        data = {"outer": {"inner": lambda: None, "keep": 1}}
        result = sanitize_for_json(data)
        assert result == {"outer": {"keep": 1}}


class TestSanitizeForJsonWarn:
    """on_drop='warn' logs at WARNING with key path."""

    def test_warns_with_path(self, caplog: Any) -> None:
        data = {"a": {"b": lambda: None}}
        with caplog.at_level(logging.WARNING):
            result = sanitize_for_json(data, on_drop="warn")
        assert result == {"a": {}}
        assert "a.b" in caplog.text
        assert "function" in caplog.text

    def test_list_index_in_path(self, caplog: Any) -> None:
        data = {"items": [1, lambda: None, 3]}
        with caplog.at_level(logging.WARNING):
            result = sanitize_for_json(data, on_drop="warn")
        assert result == {"items": [1, 3]}
        assert "items[1]" in caplog.text

    def test_dataclass_field_in_path(self, caplog: Any) -> None:
        @dataclass
        class Cfg:
            name: str = "test"
            callback: Any = None

        data = {"config": Cfg(callback=lambda: None)}
        with caplog.at_level(logging.WARNING):
            result = sanitize_for_json(data, on_drop="warn")
        assert result == {"config": {"name": "test"}}
        assert "config.callback" in caplog.text


class TestSanitizeForJsonError:
    """on_drop='error' raises SerializationError listing all dropped paths."""

    def test_raises_with_paths(self) -> None:
        data = {"a": lambda: None, "b": {"c": object()}}
        with pytest.raises(SerializationError, match="a.*type=function"):
            sanitize_for_json(data, on_drop="error")

    def test_collects_all_paths(self) -> None:
        data = {"fn1": lambda: None, "fn2": lambda: None}
        with pytest.raises(SerializationError) as exc_info:
            sanitize_for_json(data, on_drop="error")
        dropped = exc_info.value.context["dropped_paths"]
        assert len(dropped) == 2
        paths_text = " ".join(dropped)
        assert "fn1" in paths_text
        assert "fn2" in paths_text

    def test_no_error_when_clean(self) -> None:
        data = {"a": 1, "b": [2, 3], "c": {"d": "ok"}}
        result = sanitize_for_json(data, on_drop="error")
        assert result == data


class TestPathTracking:
    """_path builds correct diagnostic paths for nested structures."""

    def test_deeply_nested_path(self) -> None:
        data = {"level1": {"level2": {"level3": lambda: None}}}
        paths = validate_json_safe(data)
        assert len(paths) == 1
        assert "level1.level2.level3" in paths[0]

    def test_mixed_dict_list_path(self) -> None:
        data = {"items": [{"fn": lambda: None}]}
        paths = validate_json_safe(data)
        assert len(paths) == 1
        assert "items[0].fn" in paths[0]


class TestValidateJsonSafe:
    """validate_json_safe returns paths without modifying input."""

    def test_fully_safe_returns_empty(self) -> None:
        data = {"a": 1, "b": [2, "three"], "c": {"d": None}}
        assert validate_json_safe(data) == []

    def test_returns_paths_for_non_serializable(self) -> None:
        fn = lambda: None  # noqa: E731
        data = {"ok": 1, "bad": fn, "nested": {"also_bad": object()}}
        paths = validate_json_safe(data)
        assert len(paths) == 2
        path_text = " ".join(paths)
        assert "bad" in path_text
        assert "nested.also_bad" in path_text

    def test_does_not_modify_input(self) -> None:
        fn = lambda: None  # noqa: E731
        data = {"a": 1, "fn": fn}
        original_keys = set(data.keys())
        validate_json_safe(data)
        assert set(data.keys()) == original_keys
        assert data["fn"] is fn

    def test_dataclass_paths(self) -> None:
        @dataclass
        class Inner:
            value: int = 1
            callback: Any = None

        paths = validate_json_safe(Inner(callback=lambda: None))
        assert len(paths) == 1
        assert "callback" in paths[0]

    def test_root_non_serializable(self) -> None:
        paths = validate_json_safe(lambda: None)
        assert len(paths) == 1
        assert "<root>" in paths[0]

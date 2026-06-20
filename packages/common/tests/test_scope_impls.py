"""Behavior tests for ScopeProjector reference implementations."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.scope import (
    CachedProjector,
    CallableProjector,
    ChainedProjector,
    IdentityProjector,
    ReadOnlyProjector,
    WhitelistProjector,
)


# --- IdentityProjector ------------------------------------------------ #

def test_identity_returns_mapping_as_is() -> None:
    proj = IdentityProjector()
    source = {"a": 1, "b": 2}
    assert proj.project(source) == {"a": 1, "b": 2}


def test_identity_returns_empty_for_non_mapping() -> None:
    proj = IdentityProjector()
    assert proj.project(42) == {}
    assert proj.project("not a mapping") == {}
    assert proj.project(None) == {}


def test_identity_preserves_source_identity() -> None:
    """IdentityProjector returns the source mapping unchanged —
    mutations through the returned mapping propagate.
    """
    proj = IdentityProjector()
    source = {"a": 1}
    returned = proj.project(source)
    assert returned is source  # Same object identity


# --- ReadOnlyProjector ------------------------------------------------ #

def test_readonly_returns_proxy() -> None:
    source = {"a": 1, "b": 2}
    proj = ReadOnlyProjector(source)
    result = proj.project(None)  # source arg ignored
    assert result == {"a": 1, "b": 2}


def test_readonly_view_raises_on_write() -> None:
    source = {"a": 1}
    proj = ReadOnlyProjector(source)
    result = proj.project(None)
    with pytest.raises(TypeError):
        result["new_key"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        del result["a"]  # type: ignore[attr-defined]


def test_readonly_view_reflects_source_mutation() -> None:
    """The view is live — mutations to the source are visible via the
    proxy. The projection is read-only at the boundary, not a
    point-in-time snapshot.
    """
    source = {"a": 1}
    proj = ReadOnlyProjector(source)
    result = proj.project(None)
    source["b"] = 2
    assert result["b"] == 2


def test_readonly_constructor_captures_source() -> None:
    """Subsequent `source` arguments to project() are ignored."""
    captured = {"capture": True}
    proj = ReadOnlyProjector(captured)
    result = proj.project({"different": "source"})
    assert "capture" in result
    assert "different" not in result


# --- WhitelistProjector ----------------------------------------------- #

def test_whitelist_returns_only_declared_keys() -> None:
    source = {"a": 1, "b": 2, "c": 3}
    proj = WhitelistProjector(source, frozenset({"a", "b"}))
    assert proj.project(None) == {"a": 1, "b": 2}


def test_whitelist_omits_absent_declared_keys() -> None:
    """Absent declared keys are not injected with a default."""
    source = {"a": 1}
    proj = WhitelistProjector(source, frozenset({"a", "missing"}))
    assert proj.project(None) == {"a": 1}


def test_whitelist_returns_fresh_dict() -> None:
    """Mutations through the returned dict do not affect the source."""
    source = {"a": 1, "b": 2}
    proj = WhitelistProjector(source, frozenset({"a"}))
    result = proj.project(None)
    result["a"] = 999
    assert source["a"] == 1


def test_whitelist_empty_allowed_keys_returns_empty() -> None:
    proj = WhitelistProjector({"a": 1}, frozenset())
    assert proj.project(None) == {}


# --- ChainedProjector ------------------------------------------------- #

def test_chained_requires_at_least_one_inner() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ChainedProjector()


def test_chained_merges_in_order() -> None:
    proj = ChainedProjector(
        CallableProjector(lambda _: {"a": 1, "b": 2}),
        CallableProjector(lambda _: {"c": 3}),
    )
    assert proj.project(None) == {"a": 1, "b": 2, "c": 3}


def test_chained_later_wins_on_key_collision() -> None:
    proj = ChainedProjector(
        CallableProjector(lambda _: {"a": "first"}),
        CallableProjector(lambda _: {"a": "second"}),
    )
    assert proj.project(None)["a"] == "second"


def test_chained_passes_source_to_each_inner() -> None:
    captured: list[Any] = []
    proj = ChainedProjector(
        CallableProjector(
            lambda s: (captured.append(s), {"a": s})[1],
        ),
        CallableProjector(
            lambda s: (captured.append(s), {"b": s})[1],
        ),
    )
    proj.project("the_source")
    assert captured == ["the_source", "the_source"]


def test_chained_returns_fresh_dict() -> None:
    inner = IdentityProjector()
    proj = ChainedProjector(inner)
    source = {"a": 1}
    result = proj.project(source)
    result["b"] = 2  # type: ignore[index]
    assert "b" not in source


# --- CallableProjector ------------------------------------------------ #

def test_callable_invokes_wrapped_fn() -> None:
    proj = CallableProjector(lambda s: {"derived": s * 2})
    assert proj.project(3) == {"derived": 6}


def test_callable_passes_source_to_fn() -> None:
    proj = CallableProjector(lambda s: {"src_type": type(s).__name__})
    assert proj.project([1, 2, 3]) == {"src_type": "list"}


def test_callable_raises_on_non_mapping_return() -> None:
    proj = CallableProjector(lambda _: [1, 2, 3])  # list, not Mapping
    with pytest.raises(TypeError, match="requires a Mapping"):
        proj.project(None)


# --- CachedProjector -------------------------------------------------- #

def test_cached_returns_inner_result() -> None:
    proj = CachedProjector(CallableProjector(lambda s: {"doubled": s * 2}))
    assert proj.project(4) == {"doubled": 8}


def test_cached_memoizes_by_source() -> None:
    """The inner projector is invoked once per distinct source."""
    calls: list[Any] = []

    def fn(s: Any) -> dict[str, Any]:
        calls.append(s)
        return {"v": s}

    proj = CachedProjector(CallableProjector(fn))
    proj.project("a")
    proj.project("a")
    proj.project("b")
    assert calls == ["a", "b"]  # "a" computed once, "b" once


def test_cached_source_capturing_inner_uses_sentinel_key() -> None:
    """A source-capturing inner (ReadOnly) caches against a constant
    source key — the common cached-projector pattern.
    """
    source = {"x": 1}
    proj = CachedProjector(ReadOnlyProjector(source))
    result = proj.project(None)
    assert result == {"x": 1}
    assert proj.project(None) == {"x": 1}


def test_cached_unhashable_source_raises() -> None:
    proj = CachedProjector(IdentityProjector())
    with pytest.raises(TypeError):
        proj.project({"unhashable": "dict"})

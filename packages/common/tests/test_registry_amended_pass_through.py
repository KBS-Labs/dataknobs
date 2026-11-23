"""A registry passes its own error through only while the words are still its own.

A ``PluginRegistry`` bounds what a factory raises, because a factory builds a
backend from deployment config and its text can carry the connection URL it
was handed. An error a registry raised itself -- a nested lookup that missed,
a nested factory already bounded -- is text the registry wrote, and passes
through an outer factory as itself. A factory that catches such an error,
appends what it tried and re-raises the same instance has made it carry the
factory's text, and the outer registry must bound it like any other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from dataknobs_common import (
    NotFoundError,
    OperationError,
    PluginConfigError,
    PluginRegistry,
    naming_refusals,
)
from dataknobs_common.structured_config import StructuredConfig

DSN = "postgres://admin:hunter2@db.internal:5432/prod"

LANES = ["create", "create_async", "get", "get_async"]


def _assert_no_dsn(error: BaseException) -> None:
    assert DSN not in str(error)
    assert DSN not in repr(getattr(error, "context", {}) or {})


async def _build(lane: str, registry: PluginRegistry[Any], key: str) -> Any:
    """Every entry point that shares the registry's failure rule."""
    if lane == "create":
        return registry.create(key=key, config={})
    if lane == "create_async":
        return await registry.create_async(key=key, config={})
    if lane == "get":
        return registry.get(key, use_cache=False)
    return await registry.get_async(key, use_cache=False)


def _outer(lane: str, factory: Callable[[], Any]) -> PluginRegistry[Any]:
    """An outer registry whose one factory calls ``factory``, in the lane's arity."""
    outer: PluginRegistry[Any] = PluginRegistry("widgets")
    if lane.startswith("get"):
        outer.register("composite", lambda key, config: factory())
    else:
        outer.register("composite", lambda config, **_: factory())
    return outer


def _amending(call: Callable[[], Any]) -> Callable[[], Any]:
    """A factory body that re-raises a registry's error with what it tried appended."""

    def body() -> Any:
        try:
            return call()
        except (NotFoundError, OperationError) as exc:
            exc.args = (f"{exc} tried {DSN}",)
            raise

    return body


def _raising(error: BaseException) -> Callable[..., Any]:
    def factory(config: Any, **_: Any) -> Any:
        raise error

    return factory


@pytest.mark.parametrize("lane", LANES)
class TestAnAmendedRegistryErrorIsBounded:
    async def test_an_amended_inner_miss_is_a_bounded_not_found_error(self, lane: str) -> None:
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        outer = _outer(lane, _amending(lambda: inner.create(key="nope")))

        with pytest.raises(NotFoundError) as caught:
            await _build(lane, outer, "composite")

        _assert_no_dsn(caught.value)
        assert type(caught.value) is NotFoundError
        assert str(caught.value) == "Failed to create plugin 'composite' (NotFoundError)"
        assert caught.value.context == {"key": "composite", "registry": "widgets"}
        assert DSN in str(caught.value.__cause__)

    async def test_an_amended_inner_bounded_failure_is_a_bounded_operation_error(
        self, lane: str
    ) -> None:
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        inner.register("boom", _raising(RuntimeError("no")))
        outer = _outer(lane, _amending(lambda: inner.create(key="boom")))

        with pytest.raises(OperationError) as caught:
            await _build(lane, outer, "composite")

        _assert_no_dsn(caught.value)
        assert type(caught.value) is OperationError
        assert str(caught.value) == "Failed to create plugin 'composite' (OperationError)"
        assert caught.value.context == {"key": "composite", "registry": "widgets"}


@pytest.mark.parametrize("lane", LANES)
class TestAnUnamendedRegistryErrorPassesThrough:
    """Positive controls: the fix must not bound what is still the registry's."""

    async def test_an_inner_miss_passes_through_word_for_word(self, lane: str) -> None:
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        outer = _outer(lane, lambda: inner.create(key="nope"))

        with pytest.raises(NotFoundError) as caught:
            await _build(lane, outer, "composite")

        assert str(caught.value) == "Plugin 'nope' not registered"
        assert caught.value.context["registry"] == "gadgets"

    async def test_an_inner_bounded_failure_passes_through_word_for_word(self, lane: str) -> None:
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        inner.register("boom", _raising(RuntimeError("no")))
        outer = _outer(lane, lambda: inner.create(key="boom"))

        with pytest.raises(OperationError) as caught:
            await _build(lane, outer, "composite")

        assert str(caught.value) == "Failed to create plugin 'boom' (RuntimeError)"
        assert caught.value.context == {"key": "boom", "registry": "gadgets"}

    async def test_a_config_refusal_prefixed_by_naming_refusals_passes_through(
        self, lane: str
    ) -> None:
        """``naming_refusals`` edits in place and re-marks, so the words stay dataknobs'."""

        class Section(StructuredConfig):
            _UNKNOWN_KEYS = "raise"
            path: str = ""

        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        inner.register("closed", lambda config, **_: Section.from_dict({"pth": "x"}))

        def body() -> Any:
            with naming_refusals("source 'cases'"):
                return inner.create(key="closed")

        outer = _outer(lane, body)
        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, outer, "composite")

        assert str(caught.value).startswith("source 'cases': Failed to create plugin 'closed'")
        assert caught.value.context["registry"] == "gadgets"


@pytest.mark.parametrize("lane", LANES)
class TestANoteIsAnAmendment:
    async def test_an_inner_miss_given_a_note_is_bounded(self, lane: str) -> None:
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")

        def body() -> Any:
            try:
                return inner.create(key="nope")
            except NotFoundError as exc:
                exc.add_note(f"tried {DSN}")
                raise

        outer = _outer(lane, body)
        with pytest.raises(NotFoundError) as caught:
            await _build(lane, outer, "composite")

        assert str(caught.value) == "Failed to create plugin 'composite' (NotFoundError)"
        assert not getattr(caught.value, "__notes__", None)

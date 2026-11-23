"""What a helper in front of a registry may show of a factory's failure.

A ``PluginRegistry`` bounds a factory's error to ``"Failed to create plugin
'<key>' (<type>)"`` because a factory builds a backend from deployment config,
and its text can carry the connection URL it was handed. Helpers in front of
registries used to read that error back off ``__cause__`` by type, which handed
the URL to their callers. ``surfaced_cause`` decides by who wrote the text
instead, and ``create_surfacing`` is the shared body every such helper uses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from dataknobs_common import (
    ConcurrencyError,
    ConfigurationError,
    FactoryValueError,
    NotFoundError,
    OperationError,
    PluginConfigError,
    PluginRegistry,
    PluginUnavailableError,
    RateLimitError,
    ResourceError,
    SerializationError,
    TimeoutError,
    ValidationError,
    create_surfacing,
    create_surfacing_async,
    naming_refusals,
    surfaced_cause,
)
from dataknobs_common.exceptions import _is_written_by_dataknobs, _written_by_dataknobs
from dataknobs_common.structured_config import StructuredConfig

DSN = "postgres://admin:hunter2@db.internal:5432/prod"


def _raising(error: BaseException) -> Callable[..., Any]:
    def factory(config: Any, **_: Any) -> Any:
        raise error

    return factory


def _registry(**factories: Callable[..., Any]) -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry("widgets", config_key="type")
    for key, factory in factories.items():
        registry.register(key, factory)
    return registry


def _unknown(registry: PluginRegistry[Any]) -> Callable[[NotFoundError], ValueError]:
    def build(error: NotFoundError) -> ValueError:
        return ValueError(
            f"Unknown widget type: {error.context['key']}. "
            f"Available types: {', '.join(sorted(registry.list_keys()))}"
        )

    return build


async def _create(
    lane: str, registry: PluginRegistry[Any], config: dict[str, Any], **kw: Any
) -> Any:
    """Both lanes, one body: the two must raise the same things."""
    kw.setdefault("subject", "widget")
    if lane == "sync":
        return create_surfacing(registry, config=config, **kw)
    return await create_surfacing_async(registry, config=config, **kw)


LANES = ["sync", "async"]


class _Ambiguous:
    """A value whose comparison raises, as a numpy array's truth value does."""

    def __eq__(self, other: object) -> bool:
        raise ValueError("the truth value is ambiguous")

    def __hash__(self) -> int:
        return id(self)


def _assert_no_dsn(error: BaseException) -> None:
    assert DSN not in str(error)
    context = getattr(error, "context", {}) or {}
    assert DSN not in repr(context)


# ---------------------------------------------------------------------------
# surfaced_cause
# ---------------------------------------------------------------------------


class TestSurfacedCause:
    def test_no_cause_is_nothing_to_surface(self) -> None:
        assert surfaced_cause(OperationError("bounded")) is None

    def test_a_factorys_own_error_is_not_surfaced(self) -> None:
        bounded = OperationError("bounded")
        bounded.__cause__ = ValueError(DSN)
        assert surfaced_cause(bounded) is None

    def test_an_error_dataknobs_wrote_is_surfaced(self) -> None:
        bounded = OperationError("bounded")
        bounded.__cause__ = _written_by_dataknobs(ValueError("Summary memory requires an LLM"))
        assert surfaced_cause(bounded) is bounded.__cause__

    def test_an_error_a_registry_raised_is_surfaced(self) -> None:
        inner = _registry()
        with pytest.raises(NotFoundError) as missed:
            inner.create(config={"type": "nope"})
        bounded = OperationError("bounded")
        bounded.__cause__ = missed.value
        assert surfaced_cause(bounded) is missed.value

    def test_an_error_a_registry_raised_then_amended_is_not_surfaced(self) -> None:
        """The registry's mark vouches for its words, like any other."""
        inner = _registry()
        with pytest.raises(NotFoundError) as missed:
            inner.create(config={"type": "nope"})
        missed.value.args = (f"{missed.value}; tried {DSN}",)
        bounded = OperationError("bounded")
        bounded.__cause__ = missed.value
        assert surfaced_cause(bounded) is None

    def test_it_reads_one_link_only(self) -> None:
        """What the cause was raised ``from`` is the cause's business."""
        middle = ValueError("a factory's own wrapper")
        middle.__cause__ = _written_by_dataknobs(ValueError("ours, one link further down"))
        bounded = OperationError("bounded")
        bounded.__cause__ = middle
        assert surfaced_cause(bounded) is None


# ---------------------------------------------------------------------------
# The mark is a snapshot
# ---------------------------------------------------------------------------


class TestTheMarkVouchesForWordsNotForAnInstance:
    def test_an_error_amended_after_marking_is_no_longer_ours(self) -> None:
        error = _written_by_dataknobs(ValueError("ours"))
        error.args = (f"ours, plus {DSN}",)
        assert not _is_written_by_dataknobs(error)

    def test_an_error_whose_context_changed_after_marking_is_no_longer_ours(self) -> None:
        error = _written_by_dataknobs(ValidationError("ours", context={"field": "x"}))
        error.context["dsn"] = DSN
        assert not _is_written_by_dataknobs(error)

    def test_an_unchanged_marked_error_is_ours(self) -> None:
        error = _written_by_dataknobs(ValidationError("ours", context={"field": "x"}))
        assert _is_written_by_dataknobs(error)

    def test_a_context_that_cannot_be_copied_is_never_vouched_for(self) -> None:
        class Uncopyable:
            def __deepcopy__(self, memo: Any) -> Any:
                raise RuntimeError("no")

        error = _written_by_dataknobs(ValidationError("ours", context={"x": Uncopyable()}))
        assert not _is_written_by_dataknobs(error)

    def test_a_context_whose_comparison_raises_is_not_vouched_for(self) -> None:
        """Asked while handling another error, the question must not raise."""
        error = _written_by_dataknobs(ValidationError("ours", context={"x": _Ambiguous()}))
        assert not _is_written_by_dataknobs(error)

    def test_naming_refusals_survives_a_context_whose_comparison_raises(self) -> None:
        with pytest.raises(ValidationError) as caught:
            with naming_refusals("source 'cases'"):
                raise _written_by_dataknobs(ValidationError("ours", context={"x": _Ambiguous()}))
        assert str(caught.value) == "source 'cases': ours"
        assert not _is_written_by_dataknobs(caught.value)

    def test_naming_refusals_keeps_a_marked_refusal_ours_with_its_origin(self) -> None:
        with pytest.raises(ValidationError) as caught:
            with naming_refusals("source 'cases'", context={"source": "cases"}):
                raise _written_by_dataknobs(ValidationError("unknown field type 'money'"))
        assert str(caught.value) == "source 'cases': unknown field type 'money'"
        assert _is_written_by_dataknobs(caught.value)

    def test_naming_refusals_leaves_an_unmarked_refusal_unmarked(self) -> None:
        with pytest.raises(ValidationError) as caught:
            with naming_refusals("source 'cases'"):
                raise ValidationError(f"cannot reach {DSN}")
        assert not _is_written_by_dataknobs(caught.value)


# ---------------------------------------------------------------------------
# create_surfacing / create_surfacing_async
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", LANES)
class TestAFactorysOwnWordsStayBounded:
    async def test_a_value_error_becomes_a_factory_value_error(self, lane: str) -> None:
        registry = _registry(pg=_raising(ValueError(f"cannot connect to {DSN}")))

        with pytest.raises(FactoryValueError) as caught:
            await _create(lane, registry, {"type": "pg"})

        _assert_no_dsn(caught.value)
        assert str(caught.value) == "Failed to create widget 'pg' (ValueError)"
        assert caught.value.context == {
            "key": "pg",
            "registry": "widgets",
            "cause_type": "ValueError",
        }
        assert DSN in str(caught.value.__cause__)

    async def test_a_factory_value_error_is_still_caught_as_both_contracts(self, lane: str) -> None:
        registry = _registry(pg=_raising(ValueError(DSN)))
        with pytest.raises(ValueError):
            await _create(lane, registry, {"type": "pg"})
        with pytest.raises(ValidationError):
            await _create(lane, registry, {"type": "pg"})

    async def test_a_factorys_own_validation_error_becomes_a_factory_value_error(
        self, lane: str
    ) -> None:
        """The dataknobs type for a refused config is reported as a ValueError is."""
        registry = _registry(pg=_raising(ValidationError(f"refused {DSN}", context={"u": DSN})))

        with pytest.raises(FactoryValueError) as caught:
            await _create(lane, registry, {"type": "pg"})

        _assert_no_dsn(caught.value)
        assert str(caught.value) == "Failed to create widget 'pg' (ValidationError)"

    @pytest.mark.parametrize(
        "family",
        [ResourceError, TimeoutError, ConcurrencyError, SerializationError, ConfigurationError],
    )
    async def test_a_common_family_survives_bounded(self, lane: str, family: type) -> None:
        registry = _registry(pg=_raising(family(f"cannot reach {DSN}", context={"url": DSN})))

        with pytest.raises(family) as caught:
            await _create(lane, registry, {"type": "pg"})

        _assert_no_dsn(caught.value)
        assert type(caught.value) is family
        assert str(caught.value) == f"Failed to create widget 'pg' ({family.__name__})"

    async def test_a_subclass_of_a_family_survives_as_the_family(self, lane: str) -> None:
        class DriverDownError(ResourceError):
            pass

        registry = _registry(pg=_raising(DriverDownError(DSN)))
        with pytest.raises(ResourceError) as caught:
            await _create(lane, registry, {"type": "pg"})
        assert type(caught.value) is ResourceError
        assert caught.value.context["cause_type"] == "DriverDownError"

    @pytest.mark.parametrize(
        "error", [RuntimeError(DSN), RateLimitError(DSN, retry_after=1.0), KeyError(DSN)]
    )
    async def test_anything_else_is_the_registrys_bounded_operation_error(
        self, lane: str, error: BaseException
    ) -> None:
        registry = _registry(pg=_raising(error))

        with pytest.raises(OperationError) as caught:
            await _create(lane, registry, {"type": "pg"})

        _assert_no_dsn(caught.value)
        assert type(caught.value) is OperationError
        assert str(caught.value) == f"Failed to create plugin 'pg' ({type(error).__name__})"


@pytest.mark.parametrize("lane", LANES)
class TestWhatDataknobsWroteIsShown:
    async def test_a_marked_error_arrives_as_itself(self, lane: str) -> None:
        refusal = _written_by_dataknobs(ValueError("Summary memory requires an LLM provider"))
        registry = _registry(summary=_raising(refusal))

        with pytest.raises(ValueError) as caught:
            await _create(lane, registry, {"type": "summary"})

        assert caught.value is refusal

    async def test_a_marked_error_amended_by_the_factory_is_bounded(self, lane: str) -> None:
        def factory(config: Any, **_: Any) -> Any:
            refusal = _written_by_dataknobs(ValueError("ours"))
            refusal.args = (f"ours; while connecting to {DSN}",)
            raise refusal

        registry = _registry(pg=factory)
        with pytest.raises(FactoryValueError) as caught:
            await _create(lane, registry, {"type": "pg"})
        _assert_no_dsn(caught.value)

    async def test_a_consumer_factory_building_its_own_error_is_bounded(self, lane: str) -> None:
        """Even worded exactly like a dataknobs refusal: the words are not the test."""
        registry = _registry(
            summary=_raising(ValueError("Summary memory requires an LLM provider"))
        )
        with pytest.raises(FactoryValueError):
            await _create(lane, registry, {"type": "summary"})

    async def test_a_nested_helpers_bounded_error_surfaces_through_an_outer_one(
        self, lane: str
    ) -> None:
        inner = _registry(pg=_raising(ValueError(DSN)))

        def composite(config: Any, **_: Any) -> Any:
            return create_surfacing(inner, subject="child widget", config={"type": "pg"})

        outer = _registry(composite=composite)
        with pytest.raises(FactoryValueError) as caught:
            await _create(lane, outer, {"type": "composite"})

        assert str(caught.value) == "Failed to create child widget 'pg' (ValueError)"
        _assert_no_dsn(caught.value)

    async def test_a_nested_helpers_bounded_error_amended_by_a_factory_is_bounded(
        self, lane: str
    ) -> None:
        """The bounded error's mark records its words; adding a DSN voids it."""
        inner = _registry(pg=_raising(ValueError(DSN)))

        def composite(config: Any, **_: Any) -> Any:
            try:
                return create_surfacing(inner, subject="child widget", config={"type": "pg"})
            except FactoryValueError as exc:
                exc.args = (f"{exc}; tried {DSN}",)
                raise

        outer = _registry(composite=composite)
        with pytest.raises(FactoryValueError) as caught:
            await _create(lane, outer, {"type": "composite"})

        _assert_no_dsn(caught.value)
        assert str(caught.value) == "Failed to create widget 'composite' (FactoryValueError)"

    async def test_a_nested_registrys_error_amended_by_a_factory_is_bounded(
        self, lane: str
    ) -> None:
        """The outer registry bounds it, naming its own key; the helper shows that."""
        inner = _registry(pg=_raising(ValueError(DSN)))

        def composite(config: Any, **_: Any) -> Any:
            try:
                return inner.create(config={"type": "pg"})
            except OperationError as exc:
                exc.args = (f"{exc}; tried {DSN}",)
                raise

        outer = _registry(composite=composite)
        with pytest.raises(OperationError) as caught:
            await _create(lane, outer, {"type": "composite"})

        _assert_no_dsn(caught.value)
        assert str(caught.value) == "Failed to create plugin 'composite' (OperationError)"
        assert caught.value.context == {"key": "composite", "registry": "widgets"}

    @pytest.mark.parametrize("family", [NotFoundError, OperationError])
    async def test_an_error_a_registrys_populator_raised_is_bounded(
        self, lane: str, family: type
    ) -> None:
        """A first-access populator's error leaves ``create`` unwrapped, so the helper bounds it."""

        def populate(registry: PluginRegistry[Any]) -> None:
            raise family(f"could not list backends at {DSN}")

        registry: PluginRegistry[Any] = PluginRegistry(
            "widgets", config_key="type", on_first_access=populate
        )
        with pytest.raises(family) as caught:
            await _create(lane, registry, {"type": "pg"})

        _assert_no_dsn(caught.value)
        assert str(caught.value) == f"Failed to create widget ({family.__name__})"

    async def test_a_nested_registrys_bounded_error_keeps_its_own_key(self, lane: str) -> None:
        """Its words name the inner key, so the outer subject is not put in front of them."""
        inner: PluginRegistry[Any] = PluginRegistry("gadgets", config_key="type")
        inner.register("pg", _raising(ValueError(DSN)))

        def composite(config: Any, **_: Any) -> Any:
            return inner.create(config={"type": "pg"})

        outer = _registry(composite=composite)
        with pytest.raises(OperationError) as caught:
            await _create(lane, outer, {"type": "composite"})

        assert type(caught.value) is OperationError
        assert str(caught.value) == "Failed to create plugin 'pg' (ValueError)"

    async def test_a_config_refusal_is_carried_as_the_registry_built_it(self, lane: str) -> None:
        class Section(StructuredConfig):
            _UNKNOWN_KEYS = "raise"
            path: str = ""

        def factory(config: Any, **_: Any) -> Any:
            return Section.from_dict({"pth": "x"})

        registry = _registry(closed=factory)
        with pytest.raises(PluginConfigError) as caught:
            await _create(lane, registry, {"type": "closed"})
        assert caught.value.context["unknown"] == ["pth"]

    async def test_a_registrys_own_refusal_with_nothing_behind_it_is_raised_as_is(
        self, lane: str
    ) -> None:
        registry: PluginRegistry[Any] = PluginRegistry(
            "widgets", config_key="type", validate_type=int
        )
        registry.register("text", lambda config, **_: "not an int")

        with pytest.raises(OperationError) as caught:
            await _create(lane, registry, {"type": "text"})

        assert caught.value.__cause__ is None
        assert "text" in str(caught.value)


@pytest.mark.parametrize("lane", LANES)
class TestNotFound:
    async def test_the_registrys_miss_becomes_the_helpers_own_text(self, lane: str) -> None:
        registry = _registry(pg=_raising(ValueError(DSN)))

        with pytest.raises(ValueError) as caught:
            await _create(lane, registry, {"type": "nope"}, unknown=_unknown(registry))

        assert str(caught.value) == "Unknown widget type: nope. Available types: pg"
        assert _is_written_by_dataknobs(caught.value)
        assert isinstance(caught.value.__cause__, NotFoundError)

    async def test_without_unknown_the_registrys_miss_propagates(self, lane: str) -> None:
        registry = _registry()
        with pytest.raises(NotFoundError):
            await _create(lane, registry, {"type": "nope"})

    async def test_a_factorys_own_not_found_error_stays_a_bounded_not_found_error(
        self, lane: str
    ) -> None:
        """``unknown`` reads only the registry's miss, never a factory's NotFoundError."""
        calls: list[NotFoundError] = []

        def unknown(error: NotFoundError) -> ValueError:
            calls.append(error)
            return ValueError("should not be built")

        registry = _registry(pg=_raising(NotFoundError(f"no table at {DSN}")))
        with pytest.raises(NotFoundError) as caught:
            await _create(lane, registry, {"type": "pg"}, unknown=unknown)

        assert calls == []
        _assert_no_dsn(caught.value)
        assert str(caught.value) == "Failed to create plugin 'pg' (NotFoundError)"

    async def test_another_registrys_miss_is_not_this_helpers_unknown(self, lane: str) -> None:
        elsewhere: PluginRegistry[Any] = PluginRegistry("gadgets", config_key="type")

        def factory(config: Any, **_: Any) -> Any:
            return elsewhere.create(config={"type": "missing-gadget"})

        calls: list[NotFoundError] = []

        def unknown(error: NotFoundError) -> ValueError:
            calls.append(error)
            return ValueError("wrong registry")

        registry = _registry(pg=factory)
        with pytest.raises(NotFoundError) as caught:
            await _create(lane, registry, {"type": "pg"}, unknown=unknown)

        assert calls == []
        assert "missing-gadget" in str(caught.value)


@pytest.mark.parametrize("lane", LANES)
class TestAKnownButUnavailableKey:
    """The registry's "not available here: <reason>" reaches the caller.

    A helper's ``unknown`` hook words a typo: "Unknown widget type ...
    Available types: ...". A key declared unavailable is spelled correctly,
    and its reason says what to install, so the hook is not called for it.
    """

    def _declared(self) -> PluginRegistry[Any]:
        registry = _registry(pg=_raising(ValueError(DSN)))
        registry.declare_unavailable("vec", reason="needs the 'faiss' extra")
        return registry

    async def test_the_registrys_reason_reaches_the_caller(self, lane: str) -> None:
        calls: list[NotFoundError] = []

        def unknown(error: NotFoundError) -> ValueError:
            calls.append(error)
            return ValueError("Unknown widget type")

        with pytest.raises(PluginUnavailableError) as caught:
            await _create(lane, self._declared(), {"type": "vec"}, unknown=unknown)

        assert calls == []
        assert str(caught.value) == "Plugin 'vec' is not available here: needs the 'faiss' extra"
        assert caught.value.context["unavailable"] == "needs the 'faiss' extra"

    async def test_it_is_caught_by_both_contracts(self, lane: str) -> None:
        """A helper's ``except ValueError`` and a registry's ``except NotFoundError`` both hold."""
        registry = self._declared()
        for unknown in (_unknown(registry), None):
            with pytest.raises(ValueError) as as_value:
                await _create(lane, registry, {"type": "vec"}, unknown=unknown)
            assert isinstance(as_value.value, NotFoundError)
            assert "not available here" in str(as_value.value)

    async def test_a_truly_unknown_key_still_gets_the_helpers_text(self, lane: str) -> None:
        registry = self._declared()
        with pytest.raises(ValueError) as caught:
            await _create(lane, registry, {"type": "nope"}, unknown=_unknown(registry))

        assert not isinstance(caught.value, NotFoundError)
        assert str(caught.value) == "Unknown widget type: nope. Available types: pg"


class TestTheRegistryRaisesItsOwnUnavailableType:
    def test_a_default_registry_raises_plugin_unavailable_error(self) -> None:
        registry: PluginRegistry[Any] = PluginRegistry("widgets")
        registry.declare_unavailable("vec", reason="needs faiss", aliases=("vector",))

        for key in ("vec", "vector"):
            with pytest.raises(PluginUnavailableError) as caught:
                registry.create(key=key)
            assert caught.value.context == {
                "key": key,
                "registry": "widgets",
                "available": [],
                "unavailable": "needs faiss",
            }
            assert _is_written_by_dataknobs(caught.value)

    def test_a_registry_with_its_own_not_found_type_keeps_it(self) -> None:
        registry: PluginRegistry[Any] = PluginRegistry(
            "widgets", not_found_kind="widget", not_found_exception=ValueError
        )
        registry.declare_unavailable("vec", reason="needs faiss")

        with pytest.raises(ValueError) as caught:
            registry.create(key="vec")

        assert type(caught.value) is ValueError
        assert str(caught.value) == "widget 'vec' is not available here: needs faiss"

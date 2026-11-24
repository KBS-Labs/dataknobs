"""What a registry carries from above a refusal, and what its get lane says of a key it cannot build.

Two rules about a :class:`PluginRegistry`:

* **Text dataknobs vouches for above a config refusal is carried; text nobody
  vouched for is not.** A factory that names the refusal it re-raises -- "Source
  'cases': <the refusal's words>" -- in an error dataknobs marked as its own
  wants that name to reach the caller, and it may. A consumer's
  ``naming_refusals`` origin is the consumer's text, which can hold the
  connection URL they tried, and must not: the public function never vouches.
* **``get`` knows a key can be unavailable rather than unknown,** as ``create``
  does, and does not build the default plugin in place of one it was named.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from dataknobs_common import (
    NotFoundError,
    OperationError,
    PluginConfigError,
    PluginRegistry,
    PluginUnavailableError,
    ValidationError,
    create_surfacing,
    create_surfacing_async,
    naming_refusals,
)
from dataknobs_common.exceptions import (
    ConfigFieldError,
    _is_written_by_dataknobs,
    _naming_refusals_as_dataknobs,
    _value_free,
    _value_free_words,
    _written_by_dataknobs,
)
from dataknobs_common.structured_config import StructuredConfig

DSN = "postgres://admin:hunter2@db.internal:5432/prod"

LANES = ["create", "create_async", "get", "get_async"]


class _Section(StructuredConfig):
    _UNKNOWN_KEYS = "raise"
    path: str = ""


def _refuse() -> Any:
    return _Section.from_dict({"pth": "x"})


def _assert_no_dsn(error: BaseException) -> None:
    assert DSN not in str(error)
    assert DSN not in repr(getattr(error, "context", {}) or {})


def _registry(name: str, key: str, body: Any, *, lane: str) -> PluginRegistry[Any]:
    """A registry whose one factory runs ``body``, in the lane's arity."""
    registry: PluginRegistry[Any] = PluginRegistry(name)
    if lane.startswith("get"):
        registry.register(key, lambda _key, _config: body())
    else:
        registry.register(key, lambda config, **_: body())
    return registry


async def _build(lane: str, registry: PluginRegistry[Any], key: str) -> Any:
    if lane == "create":
        return registry.create(key=key, config={})
    if lane == "create_async":
        return await registry.create_async(key=key, config={})
    if lane == "get":
        return registry.get(key, use_cache=False)
    return await registry.get_async(key, use_cache=False)


def _naming_the_source() -> Any:
    """What the bots source factory does: names the refusal in an error marked as ours."""
    try:
        return _refuse()
    except ConfigFieldError as exc:
        raise _written_by_dataknobs(ValueError(f"Source 'cases': {exc}")) from exc


# ---------------------------------------------------------------------------
# A registry carries the outermost vouched words above a refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", LANES)
class TestVouchedWordsAboveARefusalAreCarried:
    async def test_a_marked_error_naming_the_refusal_is_carried(self, lane: str) -> None:
        registry = _registry("sources", "database", _naming_the_source, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        assert str(caught.value).startswith(
            "Failed to create plugin 'database': Source 'cases': _Section does not accept 'pth'"
        )
        assert caught.value.context["unknown"] == ["pth"]
        assert caught.value.context["key"] == "database"
        assert caught.value.context["registry"] == "sources"

    async def test_a_vouched_links_context_is_merged_over_the_refusals(self, lane: str) -> None:
        def body() -> Any:
            try:
                return _refuse()
            except ConfigFieldError as exc:
                raise _written_by_dataknobs(
                    ValidationError(
                        f"Source 'cases': {exc}",
                        context={"source": "cases", "unknown": ["overridden"]},
                    )
                ) from exc

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        assert caught.value.context["source"] == "cases"
        assert caught.value.context["unknown"] == ["overridden"]
        assert caught.value.context["config_class"] == "_Section"

    async def test_a_marked_link_amended_afterwards_is_not_carried(self, lane: str) -> None:
        """An amendment breaks the mark's snapshot, so the walk goes on to the refusal."""

        def body() -> Any:
            try:
                _naming_the_source()
            except ValueError as exc:
                exc.args = (f"{exc.args[0]} (tried {DSN})",)
                raise

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        _assert_no_dsn(caught.value)
        assert str(caught.value).startswith(
            "Failed to create plugin 'database': _Section does not accept 'pth'"
        )

    async def test_a_not_found_error_raised_from_a_refusal_stays_bounded(self, lane: str) -> None:
        def body() -> Any:
            try:
                return _refuse()
            except ConfigFieldError as exc:
                raise _written_by_dataknobs(NotFoundError("no table 'cases'")) from exc

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(NotFoundError) as caught:
            await _build(lane, registry, "database")

        assert not isinstance(caught.value, PluginConfigError)
        assert str(caught.value) == "Failed to create plugin 'database' (NotFoundError)"

    async def test_a_marked_error_not_naming_the_refusal_is_not_carried(self, lane: str) -> None:
        """A mark vouches for text, not for its relation to the refusal under it.

        Carrying it would replace the refusal's words with a message about
        something else, beside a ``context`` still describing the refusal.
        """

        def body() -> Any:
            try:
                return _refuse()
            except ConfigFieldError as exc:
                raise _written_by_dataknobs(ValueError("summary memory needs an llm")) from exc

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        assert str(caught.value).startswith(
            "Failed to create plugin 'database': _Section does not accept 'pth'"
        )
        assert "llm" not in str(caught.value)

    async def test_a_registrys_miss_raised_from_a_refusal_is_not_carried(self, lane: str) -> None:
        """A registry's miss names a key the factory chose, which may be a URL."""
        lookups: PluginRegistry[Any] = PluginRegistry("vr", not_found_exception=ValueError)

        def body() -> Any:
            try:
                return _refuse()
            except ConfigFieldError as exc:
                try:
                    lookups.create(key=DSN)
                except ValueError as miss:
                    raise miss from exc

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        _assert_no_dsn(caught.value)
        assert str(caught.value).startswith(
            "Failed to create plugin 'database': _Section does not accept 'pth'"
        )

    async def test_a_not_found_error_below_the_top_is_not_carried(self, lane: str) -> None:
        """ "A NotFoundError never carries a refusal", wherever on the chain it is."""
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")

        def inner_factory(config: Any, **_: Any) -> Any:
            try:
                return _refuse()
            except ConfigFieldError as exc:
                raise NotFoundError("no table") from exc

        inner.register("i", inner_factory)

        def body() -> Any:
            try:
                return inner.create(key="i")
            except NotFoundError as nf:
                raise RuntimeError("wrapped") from nf

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        assert "NotFoundError" not in str(caught.value)
        assert str(caught.value).startswith(
            "Failed to create plugin 'database': _Section does not accept 'pth'"
        )

    async def test_a_nested_registrys_carried_refusal_is_not_carried_in_its_place(
        self, lane: str
    ) -> None:
        """The outer key names the failure; the inner registry's wording is not stacked on it."""
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        inner.register("closed", lambda config, **_: _refuse())

        def body() -> Any:
            try:
                return inner.create(key="closed")
            except PluginConfigError as exc:
                raise RuntimeError("wrapped") from exc

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        assert str(caught.value).startswith(
            "Failed to create plugin 'database': _Section does not accept 'pth'"
        )

    async def test_an_empty_refusal_message_vouches_for_no_link(self, lane: str) -> None:
        """Every string contains the empty one, so an empty recorded message names nothing."""

        def body() -> Any:
            refusal = _value_free(ConfigFieldError("", context={"config_class": "_Section"}))
            raise _written_by_dataknobs(ValueError("summary memory needs an llm")) from refusal

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        assert "llm" not in str(caught.value)
        assert str(caught.value) == "Failed to create plugin 'database': "

    async def test_the_source_factorys_shape_over_an_amended_refusal_is_carried(
        self, lane: str
    ) -> None:
        """The bots source factory's shape: its own words over a refusal a caller amended.

        The refusal was amended with an origin nobody vouches for; the factory
        repeats the words the refusal's mark recorded, not the amended text.
        """

        def body() -> Any:
            try:
                with naming_refusals(f"source at {DSN}", context={"dsn": DSN}):
                    _refuse()
            except ConfigFieldError as exc:
                words = _value_free_words(exc)
                assert words is not None
                raise _written_by_dataknobs(ValueError(f"Source 'cases': {words[0]}")) from exc

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        _assert_no_dsn(caught.value)
        assert str(caught.value).startswith(
            "Failed to create plugin 'database': Source 'cases': _Section does not accept 'pth'"
        )

    async def test_a_re_marked_refusal_is_carried_with_its_origin(self, lane: str) -> None:
        """The refusal itself, named by dataknobs' own ``naming_refusals``, is its own link."""

        def body() -> Any:
            refusal: ConfigFieldError | None = None
            try:
                _refuse()
            except ConfigFieldError as exc:
                refusal = exc
            assert refusal is not None
            _written_by_dataknobs(refusal)
            with _naming_refusals_as_dataknobs("source 'cases'", context={"source": "cases"}):
                raise refusal

        registry = _registry("sources", "database", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, registry, "database")

        assert str(caught.value).startswith(
            "Failed to create plugin 'database': source 'cases': _Section does not accept"
        )
        assert caught.value.context["source"] == "cases"


# ---------------------------------------------------------------------------
# The public naming_refusals never vouches for its caller's origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", LANES)
class TestAConsumersOriginIsNotVouchedFor:
    async def test_a_consumers_origin_around_an_inner_registry_does_not_leak(
        self, lane: str
    ) -> None:
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        inner.register("closed", lambda config, **_: _refuse())

        def body() -> Any:
            with naming_refusals(f"source at {DSN}", context={"dsn": DSN}):
                return inner.create(key="closed")

        outer = _registry("widgets", "composite", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, outer, "composite")

        _assert_no_dsn(caught.value)
        assert str(caught.value).startswith(
            "Failed to create plugin 'composite': _Section does not accept 'pth'"
        )

    async def test_dataknobs_own_origin_around_an_inner_registry_passes_through(
        self, lane: str
    ) -> None:
        inner: PluginRegistry[Any] = PluginRegistry("gadgets")
        inner.register("closed", lambda config, **_: _refuse())

        def body() -> Any:
            with _naming_refusals_as_dataknobs("source 'cases'"):
                return inner.create(key="closed")

        outer = _registry("widgets", "composite", body, lane=lane)

        with pytest.raises(PluginConfigError) as caught:
            await _build(lane, outer, "composite")

        assert str(caught.value).startswith("source 'cases': Failed to create plugin 'closed'")
        assert caught.value.context["registry"] == "gadgets"


@pytest.mark.parametrize("lane", LANES)
async def test_a_consumers_origin_on_another_refusal_leaves_only_its_type(lane: str) -> None:
    """Only a ``StructuredConfig`` refusal's words are carried; another is named by type."""

    def body() -> Any:
        with naming_refusals(f"source at {DSN}", context={"dsn": DSN}):
            raise _written_by_dataknobs(ValidationError("unknown field type 'money'"))

    registry = _registry("widgets", "composite", body, lane=lane)

    with pytest.raises(OperationError) as caught:
        await _build(lane, registry, "composite")

    _assert_no_dsn(caught.value)
    assert str(caught.value) == "Failed to create plugin 'composite' (ValidationError)"


@pytest.mark.parametrize("lane", ["create", "create_async"])
async def test_a_consumers_origin_does_not_leak_through_create_surfacing(lane: str) -> None:
    inner: PluginRegistry[Any] = PluginRegistry("gadgets")
    inner.register("closed", lambda config, **_: _refuse())

    def factory(config: Any, **_: Any) -> Any:
        with naming_refusals(f"source at {DSN}", context={"dsn": DSN}):
            return inner.create(key="closed")

    outer: PluginRegistry[Any] = PluginRegistry("widgets", config_key="type")
    outer.register("composite", factory)

    with pytest.raises(PluginConfigError) as caught:
        if lane == "create":
            create_surfacing(outer, subject="widget", config={"type": "composite"})
        else:
            await create_surfacing_async(outer, subject="widget", config={"type": "composite"})

    _assert_no_dsn(caught.value)
    assert str(caught.value).startswith(
        "Failed to create plugin 'composite': _Section does not accept 'pth'"
    )


class TestThePublicNamingRefusals:
    def test_the_public_one_leaves_a_marked_refusal_unvouched(self) -> None:
        with pytest.raises(ValidationError) as caught:
            with naming_refusals("source 'cases'", context={"source": "cases"}):
                raise _written_by_dataknobs(ValidationError("unknown field type 'money'"))
        assert str(caught.value) == "source 'cases': unknown field type 'money'"
        assert caught.value.context == {"source": "cases"}
        assert not _is_written_by_dataknobs(caught.value)


# ---------------------------------------------------------------------------
# The get lane knows a key can be unavailable
# ---------------------------------------------------------------------------


async def _get(lane: str, registry: PluginRegistry[Any], key: str) -> Any:
    result = getattr(registry, lane)(key)
    return await result if inspect.isawaitable(result) else result


def _declared(**kwargs: Any) -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry("stores", **kwargs)
    registry.register("memory", lambda key, config: "memory store")
    registry.declare_unavailable("vec", reason="needs faiss", aliases=("v",))
    return registry


@pytest.mark.parametrize("lane", ["get", "get_async"])
class TestGetRefusesAnUnavailableKey:
    @pytest.mark.parametrize("key", ["vec", "v"])
    async def test_an_unavailable_key_says_why(self, lane: str, key: str) -> None:
        with pytest.raises(PluginUnavailableError) as caught:
            await _get(lane, _declared(), key)

        assert str(caught.value) == f"Plugin '{key}' is not available here: needs faiss"
        assert caught.value.context["unavailable"] == "needs faiss"
        assert caught.value.context["key"] == key
        assert caught.value.context["registry"] == "stores"
        assert caught.value.context["available"] == ["memory"]

    async def test_the_default_does_not_stand_in_for_an_unavailable_key(self, lane: str) -> None:
        registry = _declared(default_factory=lambda key, config: "default store")

        with pytest.raises(PluginUnavailableError):
            await _get(lane, registry, "vec")

    async def test_the_default_still_stands_in_for_an_unknown_key(self, lane: str) -> None:
        registry = _declared(default_factory=lambda key, config: "default store")

        assert await _get(lane, registry, "nope") == "default store"

    async def test_an_unknown_keys_message_is_unchanged(self, lane: str) -> None:
        with pytest.raises(NotFoundError) as caught:
            await _get(lane, _declared(), "nope")

        assert not isinstance(caught.value, PluginUnavailableError)
        assert str(caught.value) == "Plugin 'nope' not registered and no default available"
        assert "unavailable" not in caught.value.context

    async def test_a_value_error_registry_still_raises_a_not_found_error(self, lane: str) -> None:
        """The get lane's class is ``NotFoundError``; the subclass is also a ``ValueError``."""
        registry = _declared(not_found_exception=ValueError, not_found_kind="vector store")

        with pytest.raises(PluginUnavailableError) as caught:
            await _get(lane, registry, "vec")

        assert isinstance(caught.value, ValueError)
        assert str(caught.value) == "vector store 'vec' is not available here: needs faiss"


def test_create_and_get_word_an_unavailable_key_alike() -> None:
    registry = _declared()
    registry.register("annoy", lambda key, config: "annoy store")

    with pytest.raises(PluginUnavailableError) as by_create:
        registry.create(key="vec")
    with pytest.raises(PluginUnavailableError) as by_get:
        registry.get("vec")

    assert str(by_create.value) == str(by_get.value)
    assert by_create.value.context == by_get.value.context


# ---------------------------------------------------------------------------
# Declared metadata is the plugin's, and outlives the declaration
# ---------------------------------------------------------------------------


def test_registering_with_no_metadata_keeps_what_was_declared() -> None:
    registry: PluginRegistry[Any] = PluginRegistry("stores")
    declared = {"requires_install": "pip install dataknobs-data[postgres]"}
    registry.declare_unavailable("pg", metadata=declared, reason="psycopg2 is missing")

    registry.register("pg", lambda key, config: "pg store")

    assert registry.unavailable_reason("pg") is None
    assert registry.get_metadata("pg") == declared


def test_registering_with_metadata_replaces_what_was_declared() -> None:
    registry: PluginRegistry[Any] = PluginRegistry("stores")
    registry.declare_unavailable(
        "pg", metadata={"requires_install": "pip install x"}, reason="psycopg2 is missing"
    )

    registry.register("pg", lambda key, config: "pg store", metadata={"description": "PG"})

    assert registry.get_metadata("pg") == {"description": "PG"}


def test_operation_error_is_unrelated_to_the_get_lanes_miss() -> None:
    """A miss is never an ``OperationError``, so an existing handler for one is not widened."""
    with pytest.raises(NotFoundError) as caught:
        _declared().get("vec")
    assert not isinstance(caught.value, OperationError)

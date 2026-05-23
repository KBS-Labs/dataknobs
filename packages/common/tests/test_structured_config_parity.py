"""Tests for ``assert_structured_config_consumer``.

The helper combines these structural checks for adopters of
``StructuredConfigConsumer[ConfigT]``:

1. ``CONFIG_CLS`` declared.
2. ``CONFIG_CLS`` is a ``StructuredConfig`` subclass.
3. Dataclass field set matches consumer ctor surface.
4. MRO ordering — the mixin's ``__init__`` is the resolved entry point.
5. Entry-point symmetry — an overridden ``from_config_async`` routes
   through ``_coerce_config``; an overridden ``from_config`` routes
   through ``_coerce_config`` when sync, or delegates to
   ``from_config_async`` when async.
6. (Optional) Registry factory delegates to ``from_config``.

These tests exercise each failure mode against synthetic
``Consumer``/``Cfg`` pairs so the helper's diagnostic output stays
deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)
from dataknobs_common.testing import assert_structured_config_consumer

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(frozen=True)
class _GoodCfg(StructuredConfig):
    x: int = 0


class _GoodConsumer(StructuredConfigConsumer[_GoodCfg]):
    CONFIG_CLS: ClassVar[type[_GoodCfg]] = _GoodCfg


class _NoConfigCls(StructuredConfigConsumer[_GoodCfg]):
    """Adopter that forgets to declare ``CONFIG_CLS``."""


class _PlainBase:
    """A plain class — not a ``StructuredConfig`` subclass."""


class _WrongConfigCls(StructuredConfigConsumer[_GoodCfg]):
    CONFIG_CLS = _PlainBase  # type: ignore[assignment]


@dataclass(frozen=True)
class _CfgWithExtra(StructuredConfig):
    x: int = 0
    extra: str = "default"


class _ConsumerMissingExtra(StructuredConfigConsumer[_CfgWithExtra]):
    """``CONFIG_CLS`` has ``extra`` field but ctor doesn't accept it."""

    CONFIG_CLS: ClassVar[type[_CfgWithExtra]] = _CfgWithExtra

    def __init__(self, *, x: int = 0) -> None:
        self._config = _CfgWithExtra(x=x)


def _good_factory(config: dict) -> _GoodConsumer:
    """Factory that delegates to ``from_config`` — the structured path."""
    return _GoodConsumer.from_config(config)


def _allowlist_factory(config: dict) -> _GoodConsumer:
    """Factory that hand-rolls kwargs from a config dict (drift mode)."""
    return _GoodConsumer(x=config.get("x", 0))


class TestAssertStructuredConfigConsumer:
    def test_well_formed_consumer_passes(self) -> None:
        assert_structured_config_consumer(_GoodConsumer)

    def test_missing_config_cls_raises(self) -> None:
        with pytest.raises(AssertionError, match="CONFIG_CLS"):
            assert_structured_config_consumer(_NoConfigCls)

    def test_wrong_config_cls_type_raises(self) -> None:
        with pytest.raises(
            AssertionError, match="not a StructuredConfig subclass"
        ):
            assert_structured_config_consumer(_WrongConfigCls)

    def test_field_drift_raises(self) -> None:
        """Dataclass has a field the ctor doesn't accept."""
        with pytest.raises(AssertionError):
            assert_structured_config_consumer(_ConsumerMissingExtra)

    def test_factory_delegation_passes(self) -> None:
        """``expected_factory`` arg accepts a from_config-delegating factory."""
        assert_structured_config_consumer(
            _GoodConsumer, expected_factory=_good_factory
        )

    def test_factory_allowlist_drift_passes_when_kwargs_valid(self) -> None:
        """Allowlist factory only flagged when it passes invalid kwargs.

        ``_allowlist_factory`` passes ``x=...`` (a valid ctor kwarg)
        and therefore does not trigger the helper. The drift-detection
        contract is "factories that pass kwargs unknown to the ctor"
        — not "factories that don't use ``from_config``."
        """
        assert_structured_config_consumer(
            _GoodConsumer, expected_factory=_allowlist_factory
        )


class _PlainMixinBase:
    """Non-config base that continues the cooperative chain."""

    def __init__(self) -> None:
        super().__init__()


class _GoodMRO(StructuredConfigConsumer[_GoodCfg], _PlainMixinBase):
    """Mixin listed first — ``__init__`` resolves to the mixin's."""

    CONFIG_CLS: ClassVar[type[_GoodCfg]] = _GoodCfg


class _BaseWithInit:
    """A base whose own ``__init__`` would shadow the mixin if listed first."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class _BadMRO(_BaseWithInit, StructuredConfigConsumer[_GoodCfg]):
    """Mixin listed AFTER a base with its own ``__init__`` (misuse)."""

    CONFIG_CLS: ClassVar[type[_GoodCfg]] = _GoodCfg


class TestMROOrdering:
    def test_mixin_first_passes(self) -> None:
        assert_structured_config_consumer(_GoodMRO)

    def test_mixin_after_init_base_raises(self) -> None:
        with pytest.raises(AssertionError, match="must precede"):
            assert_structured_config_consumer(_BadMRO)


class _GoodAsync(StructuredConfigConsumer[_GoodCfg]):
    """Overrides ``from_config_async`` but routes through the guard."""

    CONFIG_CLS: ClassVar[type[_GoodCfg]] = _GoodCfg

    @classmethod
    async def from_config_async(
        cls, config: Mapping[str, object] | StructuredConfig
    ) -> Self:
        obj = cls(cls._coerce_config(config))
        await obj._ainit()
        return obj


class _BadAsync(StructuredConfigConsumer[_GoodCfg]):
    """Overrides ``from_config_async`` and bypasses ``_coerce_config``."""

    CONFIG_CLS: ClassVar[type[_GoodCfg]] = _GoodCfg

    @classmethod
    async def from_config_async(
        cls, config: Mapping[str, object] | StructuredConfig
    ) -> Self:
        return cls(config)  # type: ignore[arg-type]


class TestAsyncEntrySymmetry:
    def test_default_async_entry_passes(self) -> None:
        """A consumer that does not override ``from_config_async`` is fine."""
        assert_structured_config_consumer(_GoodConsumer)

    def test_override_through_guard_passes(self) -> None:
        assert_structured_config_consumer(_GoodAsync)

    def test_override_bypassing_guard_raises(self) -> None:
        with pytest.raises(AssertionError, match="_coerce_config"):
            assert_structured_config_consumer(_BadAsync)


@dataclass(frozen=True)
class _AsyncCfg(StructuredConfig):
    name: str = "x"


class _GoodAsyncFromConfig(StructuredConfigConsumer[_AsyncCfg]):
    """Async ``from_config`` override delegating to ``from_config_async``.

    The blessed back-compat shim for an object whose canonical
    construction is async but that keeps a public ``await
    X.from_config(...)`` API. Also records ``_ainit`` so the behavioral
    test can prove the lifecycle ran.
    """

    CONFIG_CLS: ClassVar[type[_AsyncCfg]] = _AsyncCfg

    def _setup(self) -> None:
        self.ainit_calls = 0
        self.ready = False

    async def _ainit(self, **components: object) -> None:
        self.ainit_calls += 1
        self.ready = True

    @classmethod
    async def from_config(
        cls,
        config: Mapping[str, object] | StructuredConfig,
        **components: object,
    ) -> Self:
        return await cls.from_config_async(config, **components)


class _BadAsyncFromConfig(StructuredConfigConsumer[_AsyncCfg]):
    """Async ``from_config`` that builds directly, skipping ``_ainit``.

    It calls ``_coerce_config`` (which alone would satisfy a *sync*
    override) but never delegates to ``from_config_async``, so ``_ainit``
    never runs and the returned object is half-built. The guard rejects
    it specifically on the missing ``from_config_async`` delegation that
    an async override requires — not on the ``_coerce_config`` call.
    """

    CONFIG_CLS: ClassVar[type[_AsyncCfg]] = _AsyncCfg

    @classmethod
    async def from_config(
        cls,
        config: Mapping[str, object] | StructuredConfig,
        **components: object,
    ) -> Self:
        return cls(cls._coerce_config(config))  # type: ignore[arg-type]


class _GoodSyncFromConfig(StructuredConfigConsumer[_AsyncCfg]):
    """Sync ``from_config`` override that routes through the guard."""

    CONFIG_CLS: ClassVar[type[_AsyncCfg]] = _AsyncCfg

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | StructuredConfig,
        **components: object,
    ) -> Self:
        return cls(cls._coerce_config(config), _components=components or None)


class _BadSyncFromConfig(StructuredConfigConsumer[_AsyncCfg]):
    """Sync ``from_config`` override that bypasses ``_coerce_config``."""

    CONFIG_CLS: ClassVar[type[_AsyncCfg]] = _AsyncCfg

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | StructuredConfig,
        **components: object,
    ) -> Self:
        return cls(config)  # type: ignore[arg-type]


class _PlainConsumer(StructuredConfigConsumer[_AsyncCfg]):
    """Minimal consumer with no ``from_config`` override.

    Dedicated to the "no override" symmetry case so this test stays a
    true no-override fixture even if a shared fixture elsewhere gains a
    ``from_config`` override later.
    """

    CONFIG_CLS: ClassVar[type[_AsyncCfg]] = _AsyncCfg


class TestFromConfigOverrideSymmetry:
    def test_no_from_config_override_passes(self) -> None:
        """A plain consumer (no ``from_config`` override) is unaffected."""
        assert_structured_config_consumer(_PlainConsumer)

    def test_async_delegator_passes(self) -> None:
        assert_structured_config_consumer(_GoodAsyncFromConfig)

    def test_async_non_delegator_raises(self) -> None:
        with pytest.raises(AssertionError, match="from_config_async"):
            assert_structured_config_consumer(_BadAsyncFromConfig)

    def test_sync_override_through_guard_passes(self) -> None:
        assert_structured_config_consumer(_GoodSyncFromConfig)

    def test_sync_override_bypassing_guard_raises(self) -> None:
        with pytest.raises(AssertionError, match="_coerce_config"):
            assert_structured_config_consumer(_BadSyncFromConfig)

    async def test_async_canonical_from_config_is_lifecycle_faithful(
        self,
    ) -> None:
        """The blessed shim builds via ``await X.from_config`` and runs
        ``_ainit`` exactly once, yielding a fully-initialized instance."""
        obj = await _GoodAsyncFromConfig.from_config({"name": "alice"})
        assert obj.ready is True
        assert obj.ainit_calls == 1
        assert obj.config.name == "alice"

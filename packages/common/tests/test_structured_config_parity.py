"""Tests for ``assert_structured_config_consumer``.

The helper combines these structural checks for adopters of
``StructuredConfigConsumer[ConfigT]``:

1. ``CONFIG_CLS`` declared.
2. ``CONFIG_CLS`` is a ``StructuredConfig`` subclass.
3. Dataclass field set matches consumer ctor surface.
4. MRO ordering — the mixin's ``__init__`` is the resolved entry point.
5. Async-entry symmetry — an overridden ``from_config_async`` routes
   through ``_coerce_config``.
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

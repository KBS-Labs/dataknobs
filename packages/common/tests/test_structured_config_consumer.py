"""Dispatch + lifecycle tests for ``StructuredConfigConsumer``.

Three construction shapes must all reach the same internal state:

- ``Consumer(typed_cfg)``
- ``Consumer({...})`` / ``Consumer(config={...})``
- ``Consumer(**kwargs)``

Mixing typed ``config=`` with loose kwargs raises ``TypeError``.
Non-Mapping non-``ConfigT`` ``config`` argument raises ``TypeError``.
``_setup()`` runs exactly once after ``self._config`` is established.
``from_config`` dispatches dict or typed config identically to
direct construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)


@dataclass(frozen=True)
class _TestCfg(StructuredConfig):
    x: int = 0
    y: str = "default"


@dataclass(frozen=True)
class _OtherCfg(StructuredConfig):
    """A different ``StructuredConfig`` subclass — not ``_Consumer``'s
    ``CONFIG_CLS``. Used to assert that passing the wrong typed config
    raises a clear ``TypeError`` rather than an opaque crash."""

    z: int = 0


class _Consumer(StructuredConfigConsumer[_TestCfg]):
    CONFIG_CLS: ClassVar[type[_TestCfg]] = _TestCfg

    def _setup(self) -> None:
        # Track invocation count so tests can assert single-shot setup.
        self._setup_calls = getattr(self, "_setup_calls", 0) + 1


class _NoCfgClsConsumer(StructuredConfigConsumer[_TestCfg]):
    """Consumer that *forgets* to declare ``CONFIG_CLS``.

    Used to assert the parity-guard fires; also ensures a clear
    runtime error at construction time.
    """


class TestConstructionShapes:
    """All three construction shapes reach the same state."""

    def test_typed_config_construction(self) -> None:
        cfg = _TestCfg(x=1, y="hello")
        c = _Consumer(cfg)
        assert c.config is cfg

    def test_dict_only_construction(self) -> None:
        c = _Consumer({"x": 1, "y": "hello"})
        assert c.config == _TestCfg(x=1, y="hello")

    def test_kwargs_only_construction(self) -> None:
        c = _Consumer(x=1, y="hello")
        assert c.config == _TestCfg(x=1, y="hello")

    def test_none_and_kwargs_construction(self) -> None:
        c = _Consumer(None, x=1)
        assert c.config == _TestCfg(x=1, y="default")

    def test_no_arguments_uses_defaults(self) -> None:
        c = _Consumer()
        assert c.config == _TestCfg()

    def test_partial_dict_uses_field_defaults(self) -> None:
        c = _Consumer({"x": 7})
        assert c.config == _TestCfg(x=7, y="default")

    def test_dict_via_keyword_arg(self) -> None:
        c = _Consumer(config={"x": 2})
        assert c.config.x == 2

    def test_typed_via_keyword_arg(self) -> None:
        cfg = _TestCfg(x=3)
        c = _Consumer(config=cfg)
        assert c.config is cfg


class TestDispatchTypeErrors:
    """Ambiguous call shapes raise ``TypeError`` rather than guess."""

    def test_mixed_typed_and_kwargs_raises(self) -> None:
        cfg = _TestCfg(x=1)
        with pytest.raises(TypeError, match="cannot mix"):
            _Consumer(cfg, x=2)

    def test_non_mapping_non_config_raises(self) -> None:
        with pytest.raises(TypeError, match="must be"):
            _Consumer(42)  # type: ignore[arg-type]

    def test_string_config_raises(self) -> None:
        with pytest.raises(TypeError, match="must be"):
            _Consumer("not a config")  # type: ignore[arg-type]


class TestSetupLifecycle:
    """``_setup`` runs exactly once and after ``self._config``."""

    def test_setup_called_once_typed(self) -> None:
        c = _Consumer(_TestCfg())
        assert c._setup_calls == 1

    def test_setup_called_once_dict(self) -> None:
        c = _Consumer({"x": 1})
        assert c._setup_calls == 1

    def test_setup_called_once_kwargs(self) -> None:
        c = _Consumer(x=1)
        assert c._setup_calls == 1

    def test_setup_can_read_self_config(self) -> None:
        """``_setup`` runs after ``self._config`` is assigned."""

        class _ReadsConfig(StructuredConfigConsumer[_TestCfg]):
            CONFIG_CLS: ClassVar[type[_TestCfg]] = _TestCfg

            def _setup(self) -> None:
                self.derived = self._config.x * 2

        c = _ReadsConfig(x=5)
        assert c.derived == 10


class TestConfigProperty:
    """``config`` property returns the typed ``ConfigT`` instance."""

    def test_property_returns_typed_config(self) -> None:
        c = _Consumer(x=1, y="hello")
        assert isinstance(c.config, _TestCfg)
        assert c.config.x == 1
        assert c.config.y == "hello"


class TestFromConfig:
    """``from_config`` classmethod accepts dicts and typed configs."""

    def test_from_config_with_dict(self) -> None:
        c = _Consumer.from_config({"x": 1, "y": "z"})
        assert c.config == _TestCfg(x=1, y="z")

    def test_from_config_with_typed(self) -> None:
        cfg = _TestCfg(x=2)
        c = _Consumer.from_config(cfg)
        assert c.config is cfg

    def test_from_config_with_empty_dict(self) -> None:
        c = _Consumer.from_config({})
        assert c.config == _TestCfg()

    def test_from_config_wrong_typed_config_raises(self) -> None:
        """A ``StructuredConfig`` of the wrong subclass raises a clear
        ``TypeError`` — not an opaque ``dict()``-on-dataclass crash.

        ``from_config`` must reject a typed config that is not
        ``CONFIG_CLS`` the same way ``__init__`` rejects a non-Mapping
        argument, naming both the expected and received type.
        """
        with pytest.raises(TypeError, match="must be"):
            _Consumer.from_config(_OtherCfg(z=1))


class TestMissingConfigCls:
    """Forgetting ``CONFIG_CLS`` surfaces immediately."""

    def test_no_config_cls_raises_at_construction(self) -> None:
        with pytest.raises(AttributeError):
            _NoCfgClsConsumer({})


class _AsyncConsumer(StructuredConfigConsumer[_TestCfg]):
    """Records the order of ``_setup`` / ``_ainit`` and their call counts."""

    CONFIG_CLS: ClassVar[type[_TestCfg]] = _TestCfg

    def _setup(self) -> None:
        self.events: list[str] = getattr(self, "events", [])
        self.events.append("setup")

    async def _ainit(self) -> None:
        self.events.append("ainit")


class TestAsyncLifecycle:
    """``_ainit`` runs once, after ``_setup``, only on the async path."""

    async def test_from_config_async_runs_setup_then_ainit(self) -> None:
        c = await _AsyncConsumer.from_config_async({"x": 1})
        assert c.events == ["setup", "ainit"]
        assert c.config == _TestCfg(x=1)

    async def test_from_config_async_accepts_typed(self) -> None:
        cfg = _TestCfg(x=2)
        c = await _AsyncConsumer.from_config_async(cfg)
        assert c.config is cfg
        assert c.events == ["setup", "ainit"]

    async def test_from_config_async_wrong_typed_raises(self) -> None:
        with pytest.raises(TypeError, match="must be"):
            await _AsyncConsumer.from_config_async(_OtherCfg(z=1))

    def test_sync_path_does_not_run_ainit(self) -> None:
        c = _AsyncConsumer({"x": 1})
        assert c.events == ["setup"]

    def test_from_config_sync_does_not_run_ainit(self) -> None:
        c = _AsyncConsumer.from_config({"x": 1})
        assert c.events == ["setup"]

    async def test_default_ainit_is_noop(self) -> None:
        """A consumer that doesn't override ``_ainit`` still builds."""
        c = await _Consumer.from_config_async({"x": 5})
        assert c.config == _TestCfg(x=5)


class _MarkerBase:
    """A non-config base that initializes its own state via ``__init__``.

    Models the unified-construction contract: non-config bases take no
    construction args and continue the cooperative chain via ``super()``.
    """

    def __init__(self) -> None:
        self.marker = "marker-set"
        super().__init__()


class _MIConsumer(StructuredConfigConsumer[_TestCfg], _MarkerBase):
    """Multiple-inheritance consumer: mixin first, then a non-config base."""

    CONFIG_CLS: ClassVar[type[_TestCfg]] = _TestCfg


class TestCooperativeMultipleInheritance:
    """``super().__init__()`` runs the remaining MI bases (146b Piece A)."""

    def test_other_base_init_runs(self) -> None:
        c = _MIConsumer({"x": 4})
        # Config dispatch happened ...
        assert c.config == _TestCfg(x=4)
        # ... and the cooperative chain reached ``_MarkerBase.__init__``.
        assert c.marker == "marker-set"

    async def test_mi_consumer_async_path(self) -> None:
        c = await _MIConsumer.from_config_async({"x": 6})
        assert c.config == _TestCfg(x=6)
        assert c.marker == "marker-set"


class TestMappingSubtypes:
    """``__init__`` accepts any ``Mapping``, not just ``dict``."""

    def test_custom_mapping_accepted(self) -> None:
        class _CustomMap(Mapping):  # type: ignore[type-arg]
            def __init__(self, data: dict[str, Any]) -> None:
                self._data = data

            def __getitem__(self, key: str) -> Any:
                return self._data[key]

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter(self._data)

            def __len__(self) -> int:
                return len(self._data)

        c = _Consumer(_CustomMap({"x": 9}))
        assert c.config.x == 9

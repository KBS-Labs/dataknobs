"""Collaborator-injection and async-dispatch tests.

These cover the construction contract for an *interconnected* object
graph — where the orchestrating parent supplies pre-built collaborators
(a shared knowledge base, a bot's main LLM, a pre-built store) that are
NOT part of a child's own config. The collaborators travel through a
keyword channel distinct from config, land on ``self.components``, and
are delivered to the async ``_ainit`` hook; the dual-input
``from_components`` path assembles directly from pre-built collaborators.
``PluginRegistry.create_async`` dispatches the same way, awaiting an
asynchronous factory before the ``validate_type`` guard.

Real test consumers throughout (no mocks), per the testing mandate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from dataknobs_common.exceptions import NotFoundError, OperationError
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)
from dataknobs_common.testing import (
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
)


@dataclass(frozen=True)
class _Cfg(StructuredConfig):
    label: str = "x"


class _SyncDepConsumer(StructuredConfigConsumer[_Cfg]):
    """Reads an injected collaborator from ``self.components`` in ``_setup``."""

    CONFIG_CLS: ClassVar[type[_Cfg]] = _Cfg

    def _setup(self) -> None:
        # The injected collaborator (if any) is available synchronously.
        self.dep = self.components.get("dep")


class _AsyncDepConsumer(StructuredConfigConsumer[_Cfg]):
    """Consumes an injected collaborator via the async ``_ainit`` hook."""

    CONFIG_CLS: ClassVar[type[_Cfg]] = _Cfg

    def _setup(self) -> None:
        self.ainit_dep: Any = "unset"

    async def _ainit(self, *, dep: Any = None, **_: Any) -> None:
        self.ainit_dep = dep


class _DualConsumer(StructuredConfigConsumer[_Cfg]):
    """Supports both config-driven build and pre-built-collaborator assembly."""

    CONFIG_CLS: ClassVar[type[_Cfg]] = _Cfg

    def _setup(self) -> None:
        self.store: Any = None
        self.built_from_config = False

    def _adopt_components(self, *, store: Any = None, **_: Any) -> None:
        # Pre-built path: bind the collaborator the parent supplied.
        self.store = store

    async def _ainit(self, *, store: Any = None, **_: Any) -> None:
        # Already wired from pre-built collaborators — do not rebuild.
        if self._prebuilt:
            return
        # Config-driven path: build the collaborator from config.
        self.store = f"built:{self.config.label}"
        self.built_from_config = True


# ── Piece 1: collaborator pass-through ────────────────────────────────
class TestSyncPassThrough:
    def test_from_config_stores_injected_collaborator(self) -> None:
        dep = object()
        c = _SyncDepConsumer.from_config({"label": "a"}, dep=dep)
        assert c.dep is dep
        assert c.components["dep"] is dep

    def test_from_config_without_components_is_empty(self) -> None:
        c = _SyncDepConsumer.from_config({"label": "a"})
        assert c.dep is None
        assert dict(c.components) == {}

    def test_components_view_is_read_only(self) -> None:
        c = _SyncDepConsumer.from_config({"label": "a"}, dep=object())
        with pytest.raises(TypeError):
            c.components["dep"] = "mutated"  # type: ignore[index]


class TestAsyncPassThrough:
    async def test_from_config_async_delivers_collaborator_to_ainit(
        self,
    ) -> None:
        dep = object()
        c = await _AsyncDepConsumer.from_config_async({"label": "a"}, dep=dep)
        assert c.ainit_dep is dep
        assert c.components["dep"] is dep

    async def test_from_config_async_without_components_does_not_crash(
        self,
    ) -> None:
        c = await _AsyncDepConsumer.from_config_async({"label": "a"})
        assert c.ainit_dep is None
        assert dict(c.components) == {}


class TestComponentsNeverLeakIntoConfig:
    def test_collaborator_absent_from_config(self) -> None:
        dep = object()
        with_dep = _SyncDepConsumer.from_config({"label": "a"}, dep=dep)
        without = _SyncDepConsumer.from_config({"label": "a"})
        # The injected object does not appear as a config field.
        assert with_dep.config == without.config == _Cfg(label="a")
        assert_structured_config_roundtrip(with_dep.config)


# ── Piece 3: dual-input (config vs pre-built collaborators) ───────────
class TestFromComponents:
    def test_assembles_from_prebuilt_collaborator(self) -> None:
        store = object()
        c = _DualConsumer.from_components(store=store)
        assert c.store is store
        assert c._prebuilt is True
        assert c.built_from_config is False
        assert c.components["store"] is store

    def test_accepts_config_snapshot(self) -> None:
        store = object()
        c = _DualConsumer.from_components({"label": "z"}, store=store)
        assert c.config == _Cfg(label="z")
        assert c.store is store

    async def test_prebuilt_short_circuits_ainit(self) -> None:
        store = object()
        c = _DualConsumer.from_components(store=store)
        # A subsequent async-init must not rebuild over the injected store.
        await c._ainit(**c.components)
        assert c.store is store
        assert c.built_from_config is False

    async def test_config_driven_build_path(self) -> None:
        c = await _DualConsumer.from_config_async({"label": "y"})
        assert c._prebuilt is False
        assert c.store == "built:y"
        assert c.built_from_config is True


# ── Piece 2: PluginRegistry.create_async ──────────────────────────────
class TestCreateAsync:
    def _registry(self) -> PluginRegistry[_AsyncDepConsumer]:
        reg: PluginRegistry[_AsyncDepConsumer] = PluginRegistry(
            "async-consumers", validate_type=_AsyncDepConsumer
        )
        reg.register("dep", _AsyncDepConsumer)
        return reg

    async def test_create_async_runs_async_init(self) -> None:
        reg = self._registry()
        c = await reg.create_async("dep", {"label": "a"})
        assert isinstance(c, _AsyncDepConsumer)
        assert c.config == _Cfg(label="a")

    async def test_create_async_forwards_components(self) -> None:
        reg = self._registry()
        dep = object()
        c = await reg.create_async("dep", {"label": "a"}, dep=dep)
        assert c.ainit_dep is dep

    async def test_create_async_unknown_key_raises_not_found(self) -> None:
        reg = self._registry()
        with pytest.raises(NotFoundError):
            await reg.create_async("missing", {})

    async def test_create_async_wrong_return_type_wrapped(self) -> None:
        reg: PluginRegistry[_AsyncDepConsumer] = PluginRegistry(
            "async-consumers", validate_type=_AsyncDepConsumer
        )
        reg.register("bad", lambda cfg, **kw: "not a consumer")  # type: ignore[arg-type,return-value]
        with pytest.raises(OperationError, match="_AsyncDepConsumer"):
            await reg.create_async("bad", {})

    async def test_async_factory_guard_runs_on_resolved_instance(self) -> None:
        """A factory whose result is a coroutine.

        The sync ``create`` runs the type guard on the un-awaited
        coroutine and rejects it; ``create_async`` awaits first, so the
        guard sees the resolved instance and passes.
        """
        coros: list[Any] = []

        async def _abuild(config: dict[str, Any]) -> _AsyncDepConsumer:
            return _AsyncDepConsumer(config)

        def _coro_factory(
            config: dict[str, Any], **_: Any
        ) -> _AsyncDepConsumer:
            coro = _abuild(config)
            coros.append(coro)
            return coro  # type: ignore[return-value]

        reg: PluginRegistry[_AsyncDepConsumer] = PluginRegistry(
            "coro", validate_type=_AsyncDepConsumer
        )
        reg.register("a", _coro_factory)

        # Sync create cannot await — the guard sees a coroutine, rejects it.
        with pytest.raises(OperationError):
            reg.create("a", {"label": "a"})
        # create_async awaits the result before the guard runs.
        c = await reg.create_async("a", {"label": "a"})
        assert isinstance(c, _AsyncDepConsumer)
        # Tidy up the coroutine the sync-create attempt left un-awaited.
        for coro in coros:
            coro.close()


# ── Parity guard: collaborator-hook safety (Check 7) ──────────────────
class TestParityGuardCollaboratorHooks:
    def test_keyword_only_with_default_passes(self) -> None:
        assert_structured_config_consumer(_AsyncDepConsumer)
        assert_structured_config_consumer(_DualConsumer)

    def test_required_positional_ainit_param_fails(self) -> None:
        class _BadPositional(StructuredConfigConsumer[_Cfg]):
            CONFIG_CLS: ClassVar[type[_Cfg]] = _Cfg

            async def _ainit(self, dep: Any) -> None:  # not keyword-only
                self.dep = dep

        with pytest.raises(AssertionError, match="keyword-only"):
            assert_structured_config_consumer(_BadPositional)

    def test_keyword_only_without_default_fails(self) -> None:
        class _BadNoDefault(StructuredConfigConsumer[_Cfg]):
            CONFIG_CLS: ClassVar[type[_Cfg]] = _Cfg

            async def _ainit(self, *, dep: Any) -> None:  # no default
                self.dep = dep

        with pytest.raises(AssertionError, match="keyword-only"):
            assert_structured_config_consumer(_BadNoDefault)

    def test_bad_adopt_components_signature_fails(self) -> None:
        class _BadAdopt(StructuredConfigConsumer[_Cfg]):
            CONFIG_CLS: ClassVar[type[_Cfg]] = _Cfg

            def _adopt_components(self, store: Any) -> None:  # positional
                self.store = store

        with pytest.raises(AssertionError, match="keyword-only"):
            assert_structured_config_consumer(_BadAdopt)


# ── Backward compatibility: config-only construction unaffected ───────
class TestBackwardCompatibility:
    def test_config_only_consumer_has_empty_components(self) -> None:
        @dataclass(frozen=True)
        class _PlainCfg(StructuredConfig):
            n: int = 0

        class _Plain(StructuredConfigConsumer[_PlainCfg]):
            CONFIG_CLS: ClassVar[type[_PlainCfg]] = _PlainCfg

        c = _Plain.from_config({"n": 3})
        assert c.config == _PlainCfg(n=3)
        assert dict(c.components) == {}
        assert c._prebuilt is False

    async def test_config_only_async_consumer_unaffected(self) -> None:
        @dataclass(frozen=True)
        class _PlainCfg(StructuredConfig):
            n: int = 0

        class _PlainAsync(StructuredConfigConsumer[_PlainCfg]):
            CONFIG_CLS: ClassVar[type[_PlainCfg]] = _PlainCfg

            async def _ainit(self) -> None:  # legacy no-arg override
                self.ready = True

        c = await _PlainAsync.from_config_async({"n": 4})
        assert c.ready is True
        assert dict(c.components) == {}

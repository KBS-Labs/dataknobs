"""Shared fixtures for ``dataknobs-bots`` tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_fsm import WizardFSM
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.testing import assert_no_leaked_bridge_threads


@pytest.fixture(scope="session", autouse=True)
def _no_leaked_daemon_threads() -> Iterator[None]:
    """Fail the session if this package leaks a dataknobs daemon thread.

    A ``WizardFSM`` driven through its synchronous ``step()`` allocates a
    daemon event-loop thread. Until the wrapper exposed a ``close()``, a
    full run of this suite left 32 of them alive — invisible here, and
    surfacing only as *another package's* teardown assertions failing,
    which named the wrong culprit entirely.

    Keeping the detector in the package that creates the threads means the
    next such leak fails where it is caused.
    """
    with assert_no_leaked_bridge_threads():
        yield


@pytest.fixture
def wizard_loader() -> Iterator[WizardConfigLoader]:
    """A real ``WizardConfigLoader`` that closes the FSMs it builds.

    Every wizard FSM this loader produces is closed at test teardown,
    releasing the daemon event-loop thread a synchronous ``step()``
    allocates. ``close()`` is idempotent and leaves the FSM usable, so the
    teardown is unconditional — a test that closes its own FSM, or never
    steps one at all, is unaffected.

    It is a real loader (a subclass overriding only the two build methods
    to record their results), so tests exercise the production loading path
    and can call any other loader method unchanged.

    Use this in preference to constructing ``WizardConfigLoader()``
    directly. Adopt it at *every* construction site, not only the ones
    that step the FSM today: converting selectively encodes the current
    call graph into the tests, and the next test to add a ``step()``
    leaks again silently.
    """
    built: list[WizardFSM] = []

    class _ClosingLoader(WizardConfigLoader):
        def load_from_dict(self, *args: Any, **kwargs: Any) -> WizardFSM:
            fsm = super().load_from_dict(*args, **kwargs)
            built.append(fsm)
            return fsm

        def load(self, *args: Any, **kwargs: Any) -> WizardFSM:
            fsm = super().load(*args, **kwargs)
            built.append(fsm)
            return fsm

    yield _ClosingLoader()

    for fsm in built:
        try:
            fsm.close()
        except Exception:  # noqa: BLE001 - teardown must not mask a failure
            pass


@pytest.fixture
def wizard_fsm_factory(
    wizard_loader: WizardConfigLoader,
) -> Callable[..., WizardFSM]:
    """Build a wizard FSM from an inline config; closed on teardown.

    The one-call form of :func:`wizard_loader` for the common case. Both
    share the same teardown.
    """

    def _build(
        config: dict[str, Any],
        custom_functions: dict[str, Any] | None = None,
        *,
        config_base_path: Path | None = None,
    ) -> WizardFSM:
        return wizard_loader.load_from_dict(
            config, custom_functions, config_base_path=config_base_path
        )

    return _build


@pytest.fixture
def register_untyped_backend() -> Iterator[Callable[..., str]]:
    """Register a bare-callable backend (no ``CONFIG_CLS``) into a construction
    registry for the duration of one test.

    Exercises the ``SKIP_VALIDATION`` resolver path: a backend that is
    *registered* (so its discriminator is recognized) but exposes no typed
    ``StructuredConfig`` to validate against. Returns a callable that registers
    the throwaway backend into the given registry under ``name`` (default
    ``"untyped_test_backend"``); every registration is unregistered on teardown.

    Teardown via the fixture finalizer — not an inline ``try``/``finally`` — so
    the shared module-global registry is restored even if the test body raises
    or the run is interrupted (e.g. ``KeyboardInterrupt`` between register and
    cleanup), which a ``try``/``finally`` does not reliably cover.
    """
    registered: list[tuple[PluginRegistry[Any], str]] = []

    def _register(
        registry: PluginRegistry[Any], name: str = "untyped_test_backend"
    ) -> str:
        def _factory(config: object = None, **_: object) -> object:
            raise NotImplementedError  # never built — the resolver only reads the type

        registry.register(name, _factory, override=True)
        registered.append((registry, name))
        return name

    yield _register

    for registry, name in registered:
        if registry.is_registered(name):
            registry.unregister(name)

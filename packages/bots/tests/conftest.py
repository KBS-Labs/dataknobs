"""Shared fixtures for ``dataknobs-bots`` tests."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_fsm import WizardFSM
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_common.registry import PluginRegistry
from dataknobs_common.testing import (
    assert_no_leaked_bridge_threads,
    declare_import_root,
)

# Shared scaffolding here is imported by a name relative to this directory
# (``fixtures.tools``, ``unit.conftest``). Declared rather than inherited: this
# directory carried an ``__init__.py`` instead, which made every module below
# it resolve under the top-level name ``tests`` — a name ``packages/fsm/tests``
# and the repo's own ``tests/`` both also claimed, so whichever was imported
# first won and the other two vanished.
declare_import_root(__file__)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _no_leaked_daemon_threads() -> Iterator[None]:
    """Fail any test that leaks a dataknobs daemon thread.

    A ``WizardFSM`` driven through its synchronous ``step()`` allocates a
    daemon event-loop thread. Until the wrapper exposed a ``close()``, a
    full run of this suite left 32 of them alive — invisible here, and
    surfacing only as *another package's* teardown assertions failing,
    which named the wrong culprit entirely.

    Scoped per **test**, not per session. A session-scoped guard fires once
    at the very end with a thread count and no test identity — "something
    in this suite leaked", which is a better error than the one it replaced
    but still sends the reader hunting. Per-test, the failure names the
    culprit outright. The guard measures a delta, so an earlier leak cannot
    cascade into every later test; the cost is two
    ``threading.enumerate()`` calls per test, and the one-second grace only
    applies on the failure path.

    This is also what lets the ``wizard_loader`` fixture below be adopted
    incrementally: a construction site that has not been converted is
    harmless until it actually steps an FSM, and on the day it does, this
    reports *which* test did it.
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
    would leak.

    Conversion of the existing direct-construction sites is incremental
    rather than complete, which is safe only because the per-test guard
    above names any test that does leak. Without that guard the two would
    have to land together.
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
        except Exception:
            # Teardown must not mask the test's own failure, so this is
            # caught rather than raised — but swallowing it silently turns a
            # broken close() into a leaked thread reported with no cause,
            # which is the diagnosis problem this whole fixture exists to
            # solve. Log it and carry on to the remaining FSMs.
            logger.exception("Error closing wizard FSM during test teardown")


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

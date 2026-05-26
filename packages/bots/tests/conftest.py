"""Shared fixtures for ``dataknobs-bots`` tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from dataknobs_common.registry import PluginRegistry


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

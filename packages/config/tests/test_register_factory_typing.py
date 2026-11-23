"""``register_factory`` type-checks for every shape the resolver builds through.

The resolver calls ``create``, ``create_async``, ``build``, ``from_config``,
``from_config_async`` and plain callables, but its parameter was typed
``ResourceFactory | Callable[..., Any]``: registering an object reachable
only through one of the other four was a type error for the consumer, though
it worked. ``packages/*/tests`` is outside the type checker's target set, so
this asks mypy directly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

mypy_api = pytest.importorskip("mypy.api")

_SHAPES = textwrap.dedent(
    """
    from typing import Any

    from dataknobs_config.binding_resolver import ConfigBindingResolver
    from dataknobs_config.environment_config import EnvironmentConfig


    class Creates:
        def create(self, **config: Any) -> int:
            return 1


    class CreatesAsync:
        async def create_async(self, **config: Any) -> int:
            return 1


    class Builds:
        def build(self, **config: Any) -> int:
            return 1


    class Consumer:
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "Consumer":
            return cls()


    class Singleton:
        def from_config(self, config: dict[str, Any]) -> int:
            return 1


    class AsyncSingleton:
        async def from_config_async(self, config: dict[str, Any]) -> int:
            return 1


    def function(**config: Any) -> int:
        return 1


    resolver = ConfigBindingResolver(EnvironmentConfig(name="t"))
    resolver.register_factory("creates", Creates())
    resolver.register_factory("creates_async", CreatesAsync())
    resolver.register_factory("builds", Builds())
    resolver.register_factory("consumer_class", Consumer)
    resolver.register_factory("singleton", Singleton())
    resolver.register_factory("async_singleton", AsyncSingleton())
    resolver.register_factory("function", function)
    resolver.register_factory("not_a_factory", {"backend": "memory"})  # LINE-OF-THE-CONTROL
    """
)


def test_every_accepted_shape_type_checks_and_a_non_factory_does_not(tmp_path: Path) -> None:
    source = tmp_path / "shapes.py"
    source.write_text(_SHAPES)
    control_line = next(
        number
        for number, line in enumerate(_SHAPES.splitlines(), start=1)
        if "LINE-OF-THE-CONTROL" in line
    )

    stdout, stderr, _ = mypy_api.run(
        [
            "--strict",
            "--no-incremental",
            "--config-file",
            "/dev/null",
            "--follow-imports",
            "silent",
            str(source),
        ]
    )

    errors = [line for line in stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, stdout + stderr
    assert f"shapes.py:{control_line}: error:" in errors[0]
    assert "register_factory" in errors[0]

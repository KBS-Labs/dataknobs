"""A consumer class named by ``factory:`` is built from the config, not probed.

Resolving ``factory:`` used to call the class with no arguments to find out
whether it was a factory class or something to call as is, and read a
``TypeError`` as the second answer. A ``StructuredConfigConsumer`` whose
config has required fields answers that probe with
:class:`~dataknobs_common.exceptions.ConfigFieldError`, so the probe's
refusal escaped as the build's -- naming a field the config does supply.
A consumer with no required fields was built once for nothing, from an
empty config, before the real build.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from dataknobs_config import Config

_MODULE = textwrap.dedent(
    """
    from dataclasses import dataclass

    from dataknobs_common.structured_config import (
        StructuredConfig,
        StructuredConfigConsumer,
    )

    CONSTRUCTED: list[dict] = []


    @dataclass(frozen=True)
    class GreeterConfig(StructuredConfig):
        greeting: str


    class Greeter(StructuredConfigConsumer[GreeterConfig]):
        CONFIG_CLS = GreeterConfig

        def __init__(self, config=None, **kwargs):
            super().__init__(config, **kwargs)
            CONSTRUCTED.append({"greeting": self._config.greeting})
            self.warmed = False

        async def _ainit(self) -> None:
            self.warmed = True
    """
)


@pytest.fixture
def greeter_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    package = tmp_path / "greeterpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "greeter.py").write_text(_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "greeterpkg", raising=False)
    monkeypatch.delitem(sys.modules, "greeterpkg.greeter", raising=False)
    import greeterpkg.greeter as module  # type: ignore[import-not-found]

    return module


def _config() -> Config:
    return Config(
        {"widget": [{"name": "w", "factory": "greeterpkg.greeter.Greeter", "greeting": "hi"}]}
    )


def test_the_sync_path_builds_the_consumer_from_its_config(greeter_module: object) -> None:
    obj = _config().build_object("xref:widget[w]")
    assert type(obj).__name__ == "Greeter"
    assert obj._config.greeting == "hi"
    assert greeter_module.CONSTRUCTED == [{"greeting": "hi"}], "built once, from the config"


@pytest.mark.asyncio
async def test_the_async_path_builds_the_consumer_and_runs_its_ainit(
    greeter_module: object,
) -> None:
    obj = await _config().build_object_async("xref:widget[w]")
    assert obj._config.greeting == "hi"
    assert obj.warmed is True
    assert greeter_module.CONSTRUCTED == [{"greeting": "hi"}]


def test_get_factory_hands_back_the_class_without_constructing_it(
    greeter_module: object,
) -> None:
    factory = _config().get_factory("widget", "w")
    assert factory is greeter_module.Greeter
    assert greeter_module.CONSTRUCTED == []

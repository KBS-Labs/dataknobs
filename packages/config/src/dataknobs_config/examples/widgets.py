"""Example StructuredConfig consumers for async object construction.

These demonstrate the async build path: ``Config.build_object_async`` /
``ObjectBuilder.build_async`` prefer a target's ``from_config_async``
(the ``StructuredConfigConsumer`` async entry point) or a factory's
``create_async``, falling back to synchronous construction otherwise.
Used by the builder tests and importable by a stable dotted path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)


@dataclass(frozen=True)
class WidgetConfig(StructuredConfig):
    """Config for the example async widget."""

    name: str = "widget"
    size: int = 1


class AsyncWidget(StructuredConfigConsumer[WidgetConfig]):
    """Consumer whose async init runs only via ``from_config_async``."""

    CONFIG_CLS: ClassVar[type[WidgetConfig]] = WidgetConfig

    def _setup(self) -> None:
        self.warmed = False

    async def _ainit(self) -> None:
        self.warmed = True


class SyncWidget:
    """Target with ``from_config`` but no ``from_config_async``."""

    def __init__(self, name: str = "sync") -> None:
        self.name = name

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SyncWidget:
        return cls(name=config.get("name", "sync"))


class PlainWidget:
    """Target with neither ``from_config`` nor ``from_config_async``."""

    def __init__(self, name: str = "plain") -> None:
        self.name = name


class AsyncWidgetFactory:
    """Factory exposing ``create_async`` for the factory build path."""

    async def create_async(self, **config: Any) -> AsyncWidget:
        return await AsyncWidget.from_config_async(config)

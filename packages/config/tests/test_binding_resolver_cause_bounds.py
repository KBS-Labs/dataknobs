"""What ``ConfigBindingResolver`` says when a resource factory raises.

It interpolated the factory's exception into its own message, so a database
factory refusing with its connection URL put that URL in front of whoever
caught ``BindingResolverError``. The message now names the resource and the
error's type; the factory's error is on ``__cause__``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dataknobs_config.binding_resolver import (
    BindingResolverError,
    ConfigBindingResolver,
    FactoryNotFoundError,
)
from dataknobs_config.environment_config import EnvironmentConfig

DSN = "postgres://admin:hunter2@db.internal:5432/prod"


def _resolver(factory: Any) -> ConfigBindingResolver:
    resolver = ConfigBindingResolver(
        EnvironmentConfig(
            name="test", resources={"databases": {"primary": {"backend": "postgres"}}}
        )
    )
    resolver.register_factory("databases", factory)
    return resolver


class _Leaky:
    def create(self, **config: Any) -> Any:
        raise ValueError(f"cannot connect to {DSN}")

    async def create_async(self, **config: Any) -> Any:
        raise ValueError(f"cannot connect to {DSN}")


def _resolve(lane: str, resolver: ConfigBindingResolver) -> Any:
    if lane == "sync":
        return resolver.resolve("databases", "primary")
    return asyncio.run(resolver.resolve_async("databases", "primary"))


@pytest.mark.parametrize("lane", ["sync", "async"])
class TestTheResolversMessageIsBounded:
    def test_a_factorys_words_stay_on_the_cause(self, lane: str) -> None:
        with pytest.raises(BindingResolverError) as caught:
            _resolve(lane, _resolver(_Leaky()))

        assert DSN not in str(caught.value)
        assert str(caught.value) == "Failed to create databases[primary] (ValueError)"
        assert DSN in str(caught.value.__cause__)

    def test_a_factory_with_no_callable_surface_is_refused_in_the_resolvers_words(
        self, lane: str
    ) -> None:
        """Refused before the call, so it is the resolver's own text, unwrapped."""

        class Inert:
            def __repr__(self) -> str:
                return "Inert()"

        with pytest.raises(BindingResolverError) as caught:
            _resolve(lane, _resolver(Inert()))

        assert str(caught.value) == "Factory Inert() has no create(), build(), or __call__ method"
        assert caught.value.__cause__ is None

    def test_an_unregistered_type_is_still_a_factory_not_found_error(self, lane: str) -> None:
        resolver = ConfigBindingResolver(
            EnvironmentConfig(
                name="test", resources={"databases": {"primary": {"backend": "postgres"}}}
            )
        )
        with pytest.raises(FactoryNotFoundError):
            _resolve(lane, resolver)

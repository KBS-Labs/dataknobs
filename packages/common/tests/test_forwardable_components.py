"""Tests for ``StructuredConfigConsumer.forwardable_components``.

Pins the documented opaque pass-through contract for composing
consumers that build children from a registry:

- A consumer with no declared ``INTERNAL_COMPONENTS`` forwards every
  collaborator in ``self.components``.
- A consumer that declares ``INTERNAL_COMPONENTS`` returns
  ``self.components`` MINUS those names — its own consumed
  collaborator(s) never leak to children.
- The returned dict is a fresh dict; caller mutation does not bleed
  back into ``self.components``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)


@dataclass(frozen=True)
class _NoConfig(StructuredConfig):
    pass


class _PlainConsumer(StructuredConfigConsumer[_NoConfig]):
    CONFIG_CLS: ClassVar[type[_NoConfig]] = _NoConfig


class _ComposingConsumer(StructuredConfigConsumer[_NoConfig]):
    CONFIG_CLS: ClassVar[type[_NoConfig]] = _NoConfig
    INTERNAL_COMPONENTS: ClassVar[frozenset[str]] = frozenset({"internal"})


def test_default_returns_all_components() -> None:
    a, b = object(), object()
    c = _PlainConsumer(config=_NoConfig(), _components={"a": a, "b": b})
    assert c.forwardable_components() == {"a": a, "b": b}


def test_excludes_declared_internal_components() -> None:
    internal, external = object(), object()
    c = _ComposingConsumer(
        config=_NoConfig(),
        _components={"internal": internal, "external": external},
    )
    forwardable = c.forwardable_components()
    assert "internal" not in forwardable
    assert forwardable["external"] is external


def test_empty_components_returns_empty_dict() -> None:
    c = _PlainConsumer(config=_NoConfig(), _components={})
    assert c.forwardable_components() == {}


def test_returns_fresh_dict_not_aliased_to_components() -> None:
    """Caller mutation of the returned dict must not affect self.components."""
    c = _PlainConsumer(config=_NoConfig(), _components={"a": object()})
    out = c.forwardable_components()
    out["mutated"] = object()
    assert "mutated" not in c.components


def test_default_internal_components_is_empty_frozenset() -> None:
    """The mixin default must be empty so existing adopters are no-ops."""
    assert _PlainConsumer.INTERNAL_COMPONENTS == frozenset()
    assert isinstance(_PlainConsumer.INTERNAL_COMPONENTS, frozenset)


def test_internal_component_with_no_external_components_returns_empty() -> None:
    """Subclass declaring INTERNAL_COMPONENTS but receiving nothing else
    returns ``{}`` — the declared name is the only entry, and it's excluded.
    """
    internal = object()
    c = _ComposingConsumer(
        config=_NoConfig(),
        _components={"internal": internal},
    )
    assert c.forwardable_components() == {}

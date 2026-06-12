"""Behavior tests for composing Discriminator reference implementations."""

from __future__ import annotations

from enum import Enum

import pytest

from dataknobs_common.discriminator import (
    AsyncCallableDiscriminator,
    AsyncChainedDiscriminator,
    AsyncDiscriminator,
    CallableDiscriminator,
    ChainedDiscriminator,
    Discriminator,
    MappingDiscriminator,
    MultiFieldDiscriminator,
)


class _Kind(Enum):
    A = "a"
    B = "b"
    UNKNOWN = "unknown"


# ---- CallableDiscriminator ----


def test_callable_discriminator_dispatches() -> None:
    d = CallableDiscriminator(lambda v: _Kind.A if v.startswith("a") else _Kind.B)
    assert d.classify("apple") == _Kind.A
    assert d.classify("banana") == _Kind.B


def test_callable_discriminator_conforms_to_protocol() -> None:
    d = CallableDiscriminator(lambda v: _Kind.A)
    assert isinstance(d, Discriminator)


def test_callable_discriminator_is_frozen() -> None:
    d = CallableDiscriminator(lambda v: _Kind.A)
    with pytest.raises(Exception):
        d.fn = lambda v: _Kind.B  # type: ignore[misc]


# ---- MappingDiscriminator ----


def test_mapping_discriminator_returns_mapped_value() -> None:
    d = MappingDiscriminator(
        mapping={"yes": _Kind.A, "no": _Kind.B},
        default=_Kind.UNKNOWN,
    )
    assert d.classify("yes") == _Kind.A
    assert d.classify("no") == _Kind.B


def test_mapping_discriminator_returns_default_for_unknown() -> None:
    d = MappingDiscriminator(
        mapping={"yes": _Kind.A},
        default=_Kind.UNKNOWN,
    )
    assert d.classify("maybe") == _Kind.UNKNOWN


def test_mapping_discriminator_conforms_to_protocol() -> None:
    d = MappingDiscriminator(mapping={}, default=_Kind.UNKNOWN)
    assert isinstance(d, Discriminator)


# ---- MultiFieldDiscriminator ----


def test_multi_field_discriminator_classifies_each_field() -> None:
    backend_classifier = MappingDiscriminator(
        mapping={"content": _Kind.A, "state": _Kind.B},
        default=_Kind.UNKNOWN,
    )
    intent_classifier = MappingDiscriminator(
        mapping={"yes": _Kind.A, "no": _Kind.B},
        default=_Kind.UNKNOWN,
    )
    d = MultiFieldDiscriminator({
        "key_kind": backend_classifier,
        "intent": intent_classifier,
    })
    result = d.classify({"key_kind": "content", "intent": "yes"})
    assert result == {"key_kind": _Kind.A, "intent": _Kind.A}


def test_multi_field_discriminator_missing_field_returns_none() -> None:
    """Missing fields are classified as None, not omitted from result."""
    classifier = MappingDiscriminator(mapping={"x": _Kind.A}, default=_Kind.UNKNOWN)
    d = MultiFieldDiscriminator({
        "present": classifier,
        "absent": classifier,
    })
    result = d.classify({"present": "x"})
    assert result == {"present": _Kind.A, "absent": None}


def test_multi_field_discriminator_empty_field_set() -> None:
    d = MultiFieldDiscriminator({})
    assert d.classify({"any": "thing"}) == {}


def test_multi_field_discriminator_preserves_field_order() -> None:
    """Result dict iterates in field-declaration order (Python dict ordered)."""
    classifier = MappingDiscriminator(
        mapping={"x": _Kind.A},
        default=_Kind.UNKNOWN,
    )
    d = MultiFieldDiscriminator({
        "first": classifier,
        "second": classifier,
        "third": classifier,
    })
    result = d.classify({"first": "x", "second": "x", "third": "x"})
    assert list(result.keys()) == ["first", "second", "third"]


# ---- ChainedDiscriminator ----


def test_chained_discriminator_first_non_default_wins() -> None:
    keyword = MappingDiscriminator(
        mapping={"yes": _Kind.A},
        default=_Kind.UNKNOWN,
    )
    llm = MappingDiscriminator(
        mapping={"ok": _Kind.A, "no": _Kind.B},
        default=_Kind.UNKNOWN,
    )
    d = ChainedDiscriminator(
        inner=[keyword, llm],
        default=_Kind.UNKNOWN,
    )
    assert d.classify("yes") == _Kind.A  # caught by keyword
    assert d.classify("ok") == _Kind.A   # caught by llm
    assert d.classify("no") == _Kind.B   # caught by llm
    assert d.classify("maybe") == _Kind.UNKNOWN  # neither catches


def test_chained_discriminator_empty_chain_returns_default() -> None:
    d: ChainedDiscriminator[str, _Kind] = ChainedDiscriminator(
        inner=[], default=_Kind.UNKNOWN
    )
    assert d.classify("anything") == _Kind.UNKNOWN


def test_chained_discriminator_conforms_to_protocol() -> None:
    d: ChainedDiscriminator[str, _Kind] = ChainedDiscriminator(
        inner=[], default=_Kind.UNKNOWN
    )
    assert isinstance(d, Discriminator)


# ---- AsyncCallableDiscriminator ----


@pytest.mark.asyncio
async def test_async_callable_discriminator_dispatches() -> None:
    async def fn(v: str) -> _Kind:
        return _Kind.A if v.startswith("a") else _Kind.B

    d = AsyncCallableDiscriminator(fn)
    assert (await d.classify("apple")) == _Kind.A
    assert (await d.classify("banana")) == _Kind.B


def test_async_callable_discriminator_conforms_to_protocol() -> None:
    async def fn(v: str) -> _Kind:
        return _Kind.A

    d = AsyncCallableDiscriminator(fn)
    assert isinstance(d, AsyncDiscriminator)


# ---- AsyncChainedDiscriminator ----


@pytest.mark.asyncio
async def test_async_chained_handles_sync_inner() -> None:
    """Sync inner discriminators work in async chain."""
    sync_keyword = MappingDiscriminator(
        mapping={"yes": _Kind.A},
        default=_Kind.UNKNOWN,
    )

    async def llm(v: str) -> _Kind:
        return _Kind.B if v == "no" else _Kind.UNKNOWN

    async_llm = AsyncCallableDiscriminator(llm)

    d: AsyncChainedDiscriminator[str, _Kind] = AsyncChainedDiscriminator(
        inner=[sync_keyword, async_llm],
        default=_Kind.UNKNOWN,
    )
    assert (await d.classify("yes")) == _Kind.A   # sync caught
    assert (await d.classify("no")) == _Kind.B    # async caught
    assert (await d.classify("maybe")) == _Kind.UNKNOWN


@pytest.mark.asyncio
async def test_async_chained_empty_returns_default() -> None:
    d: AsyncChainedDiscriminator[str, _Kind] = AsyncChainedDiscriminator(
        inner=[], default=_Kind.UNKNOWN
    )
    assert (await d.classify("anything")) == _Kind.UNKNOWN


def test_async_chained_discriminator_conforms_to_protocol() -> None:
    d: AsyncChainedDiscriminator[str, _Kind] = AsyncChainedDiscriminator(
        inner=[], default=_Kind.UNKNOWN
    )
    assert isinstance(d, AsyncDiscriminator)

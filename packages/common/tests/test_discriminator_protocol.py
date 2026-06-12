"""Tests for the :class:`Discriminator` / :class:`AsyncDiscriminator` Protocols."""

from __future__ import annotations

from enum import Enum

import pytest

from dataknobs_common.discriminator import AsyncDiscriminator, Discriminator


class _Kind(Enum):
    A = "a"
    B = "b"


class _MinimalSync:
    def classify(self, value: str) -> _Kind:
        return _Kind.A if value.startswith("a") else _Kind.B


class _MinimalAsync:
    async def classify(self, value: str) -> _Kind:
        return _Kind.A if value.startswith("a") else _Kind.B


class _MissingMethod:
    pass


def test_sync_protocol_runtime_checkable_accepts_conforming_class() -> None:
    assert isinstance(_MinimalSync(), Discriminator)


def test_sync_protocol_runtime_checkable_rejects_missing_method() -> None:
    assert not isinstance(_MissingMethod(), Discriminator)


def test_async_protocol_runtime_checkable_accepts_conforming_class() -> None:
    assert isinstance(_MinimalAsync(), AsyncDiscriminator)


def test_async_protocol_runtime_checkable_rejects_missing_method() -> None:
    assert not isinstance(_MissingMethod(), AsyncDiscriminator)


def test_minimal_sync_impl_returns_expected_kind() -> None:
    d: Discriminator[str, _Kind] = _MinimalSync()
    assert d.classify("apple") == _Kind.A
    assert d.classify("banana") == _Kind.B


@pytest.mark.asyncio
async def test_minimal_async_impl_returns_expected_kind() -> None:
    d: AsyncDiscriminator[str, _Kind] = _MinimalAsync()
    assert (await d.classify("apple")) == _Kind.A
    assert (await d.classify("banana")) == _Kind.B

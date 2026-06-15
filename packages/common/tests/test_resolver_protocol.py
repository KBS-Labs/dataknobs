"""Protocol-conformance tests for ResourceResolver / AsyncResourceResolver."""

from __future__ import annotations

import pytest

from dataknobs_common.resolver import (
    AsyncResourceResolver,
    ResourceResolver,
)


class _MinimalSync:
    def resolve(self, key: str) -> int | None:
        return len(key)


class _MinimalAsync:
    async def resolve(self, key: str) -> int | None:
        return len(key)


class _MissingMethod:
    pass


def test_sync_protocol_accepts_conforming_class() -> None:
    assert isinstance(_MinimalSync(), ResourceResolver)


def test_sync_protocol_rejects_missing_method() -> None:
    assert not isinstance(_MissingMethod(), ResourceResolver)


def test_async_protocol_accepts_conforming_class() -> None:
    assert isinstance(_MinimalAsync(), AsyncResourceResolver)


def test_minimal_sync_impl_returns_expected() -> None:
    r: ResourceResolver[str, int] = _MinimalSync()
    assert r.resolve("abc") == 3


@pytest.mark.asyncio
async def test_minimal_async_impl_returns_expected() -> None:
    r: AsyncResourceResolver[str, int] = _MinimalAsync()
    assert (await r.resolve("abc")) == 3

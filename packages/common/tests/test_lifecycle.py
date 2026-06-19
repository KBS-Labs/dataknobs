"""Tests for the owned-vs-injected close helpers."""

from __future__ import annotations

import asyncio

import pytest

from dataknobs_common import close_if_owned, close_if_owned_sync


class AsyncClosable:
    """Real test construct with an async close() and a usable flag."""

    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    async def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class SyncClosable:
    """Real test construct with a synchronous close()."""

    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class NoCloseAttr:
    """A collaborator with no close() method."""


class RaisingAsyncClosable:
    """A collaborator whose async close() raises."""

    async def close(self) -> None:
        raise RuntimeError("boom")


class RaisingSyncClosable:
    """A collaborator whose sync close() raises."""

    def close(self) -> None:
        raise RuntimeError("boom")


# --------------------------------------------------------------------------
# Async helper
# --------------------------------------------------------------------------


async def test_async_closes_when_owned() -> None:
    resource = AsyncClosable()
    await close_if_owned(resource, True)
    assert resource.closed is True
    assert resource.close_calls == 1


async def test_async_skips_when_not_owned() -> None:
    resource = AsyncClosable()
    await close_if_owned(resource, False)
    assert resource.closed is False
    assert resource.close_calls == 0


async def test_async_handles_none_resource() -> None:
    # Owned but None — must not raise.
    await close_if_owned(None, True)


async def test_async_skips_resource_without_close() -> None:
    # No close() attribute — must not raise even when owned.
    await close_if_owned(NoCloseAttr(), True)


async def test_async_error_propagates_without_on_error() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        await close_if_owned(RaisingAsyncClosable(), True)


async def test_async_error_isolated_with_on_error() -> None:
    captured: list[Exception] = []
    await close_if_owned(
        RaisingAsyncClosable(), True, on_error=captured.append
    )
    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)


async def test_async_on_error_not_invoked_on_success() -> None:
    captured: list[Exception] = []
    resource = AsyncClosable()
    await close_if_owned(resource, True, on_error=captured.append)
    assert resource.closed is True
    assert captured == []


@pytest.mark.parametrize(
    "base_exc",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
)
async def test_async_base_exception_always_propagates(
    base_exc: type[BaseException],
) -> None:
    class RaisingBaseClosable:
        async def close(self) -> None:
            raise base_exc()

    captured: list[Exception] = []
    # BaseException subclasses (cancellation, interpreter shutdown) are
    # never swallowed, even with on_error supplied.
    with pytest.raises(base_exc):
        await close_if_owned(
            RaisingBaseClosable(), True, on_error=captured.append
        )
    assert captured == []


# --------------------------------------------------------------------------
# Sync helper
# --------------------------------------------------------------------------


def test_sync_closes_when_owned() -> None:
    resource = SyncClosable()
    close_if_owned_sync(resource, True)
    assert resource.closed is True
    assert resource.close_calls == 1


def test_sync_skips_when_not_owned() -> None:
    resource = SyncClosable()
    close_if_owned_sync(resource, False)
    assert resource.closed is False


def test_sync_handles_none_resource() -> None:
    close_if_owned_sync(None, True)


def test_sync_skips_resource_without_close() -> None:
    close_if_owned_sync(NoCloseAttr(), True)


def test_sync_error_propagates_without_on_error() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        close_if_owned_sync(RaisingSyncClosable(), True)


def test_sync_error_isolated_with_on_error() -> None:
    captured: list[Exception] = []
    close_if_owned_sync(RaisingSyncClosable(), True, on_error=captured.append)
    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)


@pytest.mark.parametrize("base_exc", [KeyboardInterrupt, SystemExit])
def test_sync_base_exception_always_propagates(
    base_exc: type[BaseException],
) -> None:
    class RaisingBaseSyncClosable:
        def close(self) -> None:
            raise base_exc()

    captured: list[Exception] = []
    # BaseException subclasses are never swallowed, even with on_error.
    with pytest.raises(base_exc):
        close_if_owned_sync(
            RaisingBaseSyncClosable(), True, on_error=captured.append
        )
    assert captured == []

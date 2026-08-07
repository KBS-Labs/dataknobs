"""Tests for the owned-vs-injected close helpers."""

from __future__ import annotations

import asyncio
import logging

import pytest

from dataknobs_common import (
    aclose_if_owned,
    close_if_owned,
    close_if_owned_sync,
)


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


class DualClosable:
    """The shape ``aclose_if_owned`` exists for.

    A *synchronous* ``close()`` alongside an ``aclose()`` that awaits
    cleanup the sync form skips — mirroring ``AdvancedFSM`` /
    ``WizardFSM``. Neither sibling helper serves it: ``close_if_owned``
    would await ``close()``'s ``None`` return, and ``close_if_owned_sync``
    would take the lossy half.
    """

    def __init__(self) -> None:
        self.close_calls = 0
        self.aclose_calls = 0
        self.async_cleanup_ran = False

    def close(self) -> None:
        self.close_calls += 1

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.async_cleanup_ran = True


class NoAcloseAttr:
    """A collaborator with a close() but no aclose()."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RaisingAclosable:
    """A collaborator whose aclose() raises."""

    async def aclose(self) -> None:
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


# --------------------------------------------------------------------------
# Async-aclose helper
#
# Mirrors the close_if_owned block above case-for-case: divergence between
# siblings is how a guard family stops being learnable, so a case that
# exists for one must exist for all three.
# --------------------------------------------------------------------------


async def test_aclose_closes_when_owned() -> None:
    resource = DualClosable()
    await aclose_if_owned(resource, True)
    assert resource.aclose_calls == 1
    assert resource.async_cleanup_ran is True


async def test_aclose_skips_when_not_owned() -> None:
    resource = DualClosable()
    await aclose_if_owned(resource, False)
    assert resource.aclose_calls == 0
    assert resource.close_calls == 0


async def test_aclose_handles_none_resource() -> None:
    # Owned but None — must not raise.
    await aclose_if_owned(None, True)


async def test_aclose_skips_resource_without_aclose() -> None:
    # The hasattr probe is on ``aclose``, so a close()-only collaborator is
    # left untouched rather than closed through the wrong method.
    resource = NoAcloseAttr()
    await aclose_if_owned(resource, True)
    assert resource.close_calls == 0


async def test_aclose_error_propagates_without_on_error() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        await aclose_if_owned(RaisingAclosable(), True)


async def test_aclose_error_isolated_with_on_error() -> None:
    captured: list[Exception] = []
    await aclose_if_owned(RaisingAclosable(), True, on_error=captured.append)
    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)


async def test_aclose_on_error_not_invoked_on_success() -> None:
    captured: list[Exception] = []
    resource = DualClosable()
    await aclose_if_owned(resource, True, on_error=captured.append)
    assert resource.aclose_calls == 1
    assert captured == []


@pytest.mark.parametrize(
    "base_exc",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
)
async def test_aclose_base_exception_always_propagates(
    base_exc: type[BaseException],
) -> None:
    class RaisingBaseAclosable:
        async def aclose(self) -> None:
            raise base_exc()

    captured: list[Exception] = []
    # BaseException subclasses (cancellation, interpreter shutdown) are
    # never swallowed, even with on_error supplied. Cancellation reaching
    # here under shutdown is the case most likely to be dropped when
    # mirroring a sibling by hand, and the one that matters most.
    with pytest.raises(base_exc):
        await aclose_if_owned(
            RaisingBaseAclosable(), True, on_error=captured.append
        )
    assert captured == []


# --------------------------------------------------------------------------
# Why the third helper exists: neither sibling serves a collaborator with a
# synchronous close() alongside an aclose(). These pin that gap.
# --------------------------------------------------------------------------


async def test_close_if_owned_cannot_serve_a_sync_close_plus_aclose() -> None:
    # close_if_owned awaits close()'s return value. For a synchronous
    # close() that is None, which is not awaitable.
    with pytest.raises(TypeError):
        await close_if_owned(DualClosable(), True)


def test_close_if_owned_sync_takes_the_lossy_half() -> None:
    # The sync helper does close the resource, but through close() — so
    # the coroutine cleanup aclose() performs never runs. Not an error,
    # which is precisely what makes it the dangerous choice.
    resource = DualClosable()
    close_if_owned_sync(resource, True)
    assert resource.close_calls == 1
    assert resource.async_cleanup_ran is False


async def test_aclose_if_owned_is_the_one_that_works() -> None:
    resource = DualClosable()
    await aclose_if_owned(resource, True)
    assert resource.async_cleanup_ran is True
    assert resource.close_calls == 0


class TestUnclosableOwnedResourceIsAudible:
    """Owning something with no closer is a wiring bug, not a no-op.

    All three helpers guard on ``hasattr``, so a collaborator exposing
    neither the probed method nor anything else is skipped in silence. For
    ``close_if_owned`` / ``close_if_owned_sync`` that is mostly benign — the
    probed name is the ordinary one. For ``aclose_if_owned`` it is the
    likeliest misuse: reach for the newest helper on a plain ``close()``-only
    collaborator and *nothing at all* is closed, with no exception and no
    log — a worse outcome than either sibling's, since one raises loudly and
    the other at least closes something.

    Declining to close is still the right behavior (raising would make an
    optional-teardown collaborator un-holdable). Doing it inaudibly is not.
    """

    class _NoCloser:
        """A collaborator with no teardown method of any kind."""

    class _SyncOnly:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    async def test_aclose_logs_when_the_owned_resource_has_no_aclose(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        resource = self._SyncOnly()

        with caplog.at_level(logging.DEBUG, logger="dataknobs_common.lifecycle"):
            await aclose_if_owned(resource, True)

        assert not resource.closed, "aclose_if_owned must not fall back to close()"
        assert any(
            "aclose" in record.getMessage() for record in caplog.records
        ), "the skipped close left no diagnostic at any level"

    async def test_close_logs_when_the_owned_resource_has_no_close(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="dataknobs_common.lifecycle"):
            await close_if_owned(self._NoCloser(), True)

        assert caplog.records, "the skipped close left no diagnostic at any level"

    def test_sync_close_logs_when_the_owned_resource_has_no_close(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="dataknobs_common.lifecycle"):
            close_if_owned_sync(self._NoCloser(), True)

        assert caplog.records, "the skipped close left no diagnostic at any level"

    async def test_an_unowned_resource_is_not_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Not owning it is the normal case, not a wiring bug.

        The diagnostic must fire on "I own this and cannot close it", never
        on the injected-collaborator path every consumer takes.
        """
        with caplog.at_level(logging.DEBUG, logger="dataknobs_common.lifecycle"):
            await close_if_owned(self._NoCloser(), False)
            await aclose_if_owned(self._NoCloser(), False)
            close_if_owned_sync(self._NoCloser(), False)

        assert caplog.records == []

    async def test_none_is_not_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A collaborator that was never built is absence, not misuse."""
        with caplog.at_level(logging.DEBUG, logger="dataknobs_common.lifecycle"):
            await close_if_owned(None, True)
            await aclose_if_owned(None, True)
            close_if_owned_sync(None, True)

        assert caplog.records == []

# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The sync provider adapter is callable from inside a running event loop.

Every one of ``SyncProviderAdapter``'s six async-reaching methods carried the
same eight-line preamble --- ``asyncio.get_event_loop()``, fall back to
``new_event_loop()``, then ``run_until_complete`` --- and ``run_until_complete``
raises ``RuntimeError: This event loop is already running`` when the caller is
already on a loop. So all six failed in the one situation that matters most:
synchronous code reached from async code, which is what
``create_llm_provider(config, is_async=False)`` hands a consumer and what
``LLMResource`` holds one of per model.

Six methods, one defect, so six tests. The class had one instance of this
pattern per method rather than one shared helper, which is why fixing the
method a consumer happened to report would have left five.

The two tests that are *not* red before the fix are the over-correction
guards. ``the_sync_path_off_a_loop_still_works`` is the behaviour that works
today and must survive; ``close_is_still_idempotent`` is a property the
preamble had for free --- a second ``close()`` simply ran a second
already-closed provider close --- and which a bridge does not, because a
``run`` after ``close`` raises.

The fix is :class:`~dataknobs_common.sync_bridge.SyncLoopBridge`: a private
loop on a daemon thread, so the coroutine never runs on the caller's loop.
The thread is the cost, it is named, and ``close()`` ends it --- which is what
``close_leaves_no_bridge_thread_alive`` pins.
"""

from __future__ import annotations

import asyncio

import pytest
from dataknobs_common.testing import assert_no_leaked_bridge_threads

from dataknobs_llm import EchoProvider
from dataknobs_llm.fsm_integration.resources import LLMResource
from dataknobs_llm.llm.providers import SyncProviderAdapter, create_llm_provider

ECHO_CONFIG = {"provider": "echo", "model": "test"}


@pytest.fixture()
def adapter():
    """The adapter as a consumer gets it --- through the documented factory.

    ``create_llm_provider(..., is_async=False)`` is the only in-tree source of
    a ``SyncProviderAdapter``; constructing one directly would test the class
    but not the path that reaches it.
    """
    provider = create_llm_provider(ECHO_CONFIG, is_async=False)
    assert isinstance(provider, SyncProviderAdapter)
    try:
        yield provider
    finally:
        provider.close()


async def test_complete_works_from_inside_a_running_loop(adapter) -> None:
    response = adapter.complete("hello")

    assert response.content == "Echo: hello"


async def test_stream_works_from_inside_a_running_loop(adapter) -> None:
    chunks = list(adapter.stream("hi"))

    assert "".join(chunk.delta for chunk in chunks) == "Echo: hi"
    assert chunks[-1].is_final is True


async def test_embed_works_from_inside_a_running_loop(adapter) -> None:
    embedding = adapter.embed("hello")

    assert len(embedding) == 768
    assert all(isinstance(value, float) for value in embedding)


async def test_validate_model_works_from_inside_a_running_loop(adapter) -> None:
    assert adapter.validate_model() is True


async def test_initialize_works_from_inside_a_running_loop(adapter) -> None:
    adapter.initialize()

    assert adapter.is_initialized


async def test_close_works_from_inside_a_running_loop(adapter) -> None:
    adapter.initialize()

    adapter.close()

    assert not adapter.is_initialized


async def test_a_partially_consumed_stream_closes_cleanly(adapter) -> None:
    """Abandoning a stream runs its ``finally`` --- on the same loop.

    An async generator binds to whichever loop first iterates it, so the
    ``aclose`` in ``stream``'s ``finally`` has to reach that same loop. Taking
    one chunk and dropping the generator is what exercises it: consuming the
    stream to exhaustion never runs the interesting branch.
    """
    stream = adapter.stream("hi")

    first = next(stream)
    stream.close()

    assert first.delta


async def test_llm_resource_completes_from_inside_a_running_loop() -> None:
    """The in-tree consumer, not just the class.

    ``LLMResource`` holds one adapter per model key and translates a provider
    failure into ``ResourceError``, so the loop defect reached its callers
    wearing a different exception type and naming no loop at all.
    """
    resource = LLMResource(name="r", provider="echo", model="test")
    try:
        result = resource.complete("hello")

        assert "Echo: hello" in result["choices"][0]["text"]
    finally:
        resource.close()


def test_close_leaves_no_bridge_thread_alive() -> None:
    """The daemon thread is the cost of the fix, and ``close()`` pays it back.

    The guard watches every name a bridge has been *constructed* under, not
    just the default, so the adapter's own thread name is covered by it
    rather than exempted from it.
    """
    with assert_no_leaked_bridge_threads():
        provider = create_llm_provider(ECHO_CONFIG, is_async=False)
        provider.initialize()
        provider.complete("hello")
        provider.close()


def test_the_sync_path_off_a_loop_still_works() -> None:
    """The over-correction guard: no loop running, everything still works."""
    assert asyncio.get_event_loop_policy()  # a policy, but no *running* loop
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    provider = create_llm_provider(ECHO_CONFIG, is_async=False)
    try:
        provider.initialize()

        assert provider.complete("hello").content == "Echo: hello"
        assert list(provider.stream("hi"))[-1].is_final is True
        assert provider.validate_model() is True
        assert len(provider.embed("hello")) == 768
    finally:
        provider.close()


def test_close_is_still_idempotent() -> None:
    """A second ``close()`` returns, as it did before the bridge.

    The old preamble got this for free: a second call ran a second
    already-closed provider close on a fresh loop. A bridge does not --- a
    ``run`` issued after ``close`` raises ``RuntimeError`` --- so the adapter
    has to stop rather than reach for a loop it has already torn down.
    """
    provider = SyncProviderAdapter(EchoProvider(ECHO_CONFIG))
    provider.initialize()

    provider.close()
    provider.close()

    assert not provider.is_initialized

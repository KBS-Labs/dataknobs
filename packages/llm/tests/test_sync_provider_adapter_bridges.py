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

The second half of this file is the bridge's *edges*, where it behaves
unlike the preamble it replaced: a ``run`` after ``close`` raises where the
preamble simply built another loop, construction now allocates a thread where
it used to allocate nothing, and a teardown that fails part way can leave the
thread unreachable. Those are properties of owning a loop, not of reaching a
provider, so they get their own tests rather than being asserted incidentally
by the six above.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from dataknobs_common import SyncLoopBridge
from dataknobs_common.testing import assert_no_leaked_bridge_threads, live_dk_daemon_threads

from dataknobs_llm import EchoProvider, ErrorResponse
from dataknobs_llm.fsm_integration.resources import LLMResource
from dataknobs_llm.llm.providers import SyncProviderAdapter, create_llm_provider

ECHO_CONFIG = {"provider": "echo", "model": "test"}


@pytest.fixture()
def adapter():
    """The adapter as a consumer gets it --- through the documented factory.

    ``create_llm_provider(..., is_async=False)`` is the path a consumer
    actually travels; ``LLMProviderFactory(is_async=False).create(...)`` is
    the same construction one layer down, and constructing the class directly
    reaches neither. The tests below that do construct it directly are the
    ones whose subject is the class rather than the path to it.
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
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()  # the assertion: nothing is running here

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
    assert live_dk_daemon_threads([SyncProviderAdapter.BRIDGE_THREAD_NAME])

    provider.close()
    provider.close()

    assert not provider.is_initialized
    assert not live_dk_daemon_threads([SyncProviderAdapter.BRIDGE_THREAD_NAME])


# ---------------------------------------------------------------------------
# The bridge's edges: teardown, re-entrancy, and the cost of construction.
#
# The tests above prove the adapter reaches its provider from inside a running
# loop. These prove it survives the moments where a bridge behaves unlike the
# `run_until_complete` preamble it replaced --- a `run` after `close` raises
# where the preamble merely built another loop, and construction now costs a
# thread where it used to cost nothing.
# ---------------------------------------------------------------------------


BRIDGE_THREAD = SyncProviderAdapter.BRIDGE_THREAD_NAME


class _CleanupRecordingProvider(EchoProvider):
    """Records whether its stream generator's ``finally`` actually ran.

    ``test_a_partially_consumed_stream_closes_cleanly`` asserts that the first
    chunk arrived, which is true whether or not the abandoned generator was
    ever closed. The property that test names --- the ``aclose`` in ``stream``'s
    ``finally`` reaching the loop the generator is bound to --- is only
    observable from inside the generator, so this subclass observes it.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.stream_cleanup_ran = False

    async def stream_complete(self, messages: Any, **kwargs: Any) -> Any:
        try:
            async for chunk in super().stream_complete(messages, **kwargs):
                yield chunk
        finally:
            self.stream_cleanup_ran = True


class _ReentrantProvider(EchoProvider):
    """A provider that re-enters its own adapter from the bridge loop.

    The case :meth:`SyncLoopBridge.run` documents by name --- "a transform
    re-entering the same synchronous wrapper". Both of the adapter's calls
    into the bridge refuse from the bridge's own thread, so this is the
    reachable way to make teardown raise *after* the adapter has committed to
    it.
    """

    adapter: SyncProviderAdapter | None = None

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        if self.adapter is not None:
            self.adapter.close()
        return await super().complete(messages, **kwargs)


def test_an_abandoned_stream_runs_the_providers_cleanup() -> None:
    """The abandoned generator is really closed, not merely dropped.

    Taking one chunk and closing the stream has to run the provider
    generator's ``finally`` --- on the bridge loop, which is the only loop it
    is bound to.
    """
    provider = _CleanupRecordingProvider(ECHO_CONFIG)
    adapter = SyncProviderAdapter(provider)
    try:
        adapter.initialize()
        stream = adapter.stream("hi")

        assert next(stream).delta
        assert not provider.stream_cleanup_ran, "cleanup cannot have run yet"

        stream.close()

        assert provider.stream_cleanup_ran, "the abandoned stream's finally never ran"
    finally:
        adapter.close()


def test_closing_the_adapter_leaves_an_abandoned_stream_safe_to_close() -> None:
    """A stream outliving its adapter must not raise from its own finalizer.

    ``stream``'s ``finally`` reaches the bridge unconditionally, and a ``run``
    after ``close`` raises ``RuntimeError: SyncLoopBridge is closed``. So
    closing the adapter while a partially-consumed stream is still alive turns
    that stream's ``close()`` --- or its collection, at a moment nobody chose
    --- into an exception. The bridge's own teardown has already drained the
    generator by then, so there is nothing left for the ``finally`` to do.
    """
    adapter = create_llm_provider(ECHO_CONFIG, is_async=False)
    adapter.initialize()
    stream = adapter.stream("hi")
    assert next(stream).delta

    adapter.close()

    stream.close()


def test_a_reentrant_close_does_not_strand_the_bridge_thread() -> None:
    """A teardown that raises must leave the bridge closable, not stranded.

    The adapter marks itself closed before it does any teardown work, so a
    teardown that raises leaves ``_closed`` true with the loop thread still
    running --- and every later ``close()`` returns at that flag without ever
    reaching the bridge. The thread is then unreachable for the life of the
    process. The bridge is idempotent and concurrency-safe on its own, so it,
    not a flag beside it, is what teardown should be asking.
    """
    before = set(live_dk_daemon_threads([BRIDGE_THREAD]))
    provider = _ReentrantProvider(ECHO_CONFIG)
    adapter = SyncProviderAdapter(provider)
    provider.adapter = adapter
    try:
        adapter.initialize()

        # The re-entrant `close()` runs on the bridge loop: its `run` refuses,
        # and so does the `_bridge.close()` in its `finally`.
        with pytest.raises(RuntimeError, match="bridge loop"):
            adapter.complete("hello")

        # From a thread that *can* join the loop thread, closing must work.
        provider.adapter = None
        adapter.close()

        assert set(live_dk_daemon_threads([BRIDGE_THREAD])) == before
    finally:
        provider.adapter = None
        adapter.close()


def test_construction_costs_no_thread_until_the_provider_is_reached() -> None:
    """Building a sync provider to read its metadata must stay free.

    ``create_llm_provider(..., is_async=False)`` is the documented entry
    point, and discovery callers use it for exactly this: build one, read
    ``provider_name`` or ``get_capabilities()``, drop it. None of those
    reach the wrapped provider, so none of them need a loop --- and a daemon
    thread per construction is a cost the factory did not have before.
    """
    before = set(live_dk_daemon_threads([BRIDGE_THREAD]))
    adapter = create_llm_provider(ECHO_CONFIG, is_async=False)
    try:
        assert adapter.provider_name == "echo"
        assert adapter.get_capabilities()
        assert not adapter.is_initialized

        assert set(live_dk_daemon_threads([BRIDGE_THREAD])) == before
    finally:
        adapter.close()

    assert set(live_dk_daemon_threads([BRIDGE_THREAD])) == before


def test_the_adapter_is_a_context_manager() -> None:
    """The form the leaked-thread guard's own failure message recommends.

    Without it every consumer and every test writes the same ``try``/
    ``finally``, which is what the seven leaking tests in this suite were.
    """
    with assert_no_leaked_bridge_threads():
        with create_llm_provider(ECHO_CONFIG, is_async=False) as adapter:
            adapter.initialize()

            assert adapter.complete("hello").content == "Echo: hello"


def test_a_timeout_bounds_the_blocking_wait() -> None:
    """A synchronous caller gets an upper bound on a wait it cannot cancel.

    ``SyncTextEmbedder`` --- the in-tree adopter of this same bridge --- takes
    a ``timeout`` for this reason, and an LLM completion is a longer stall
    than an embedding. Without one, a hung provider socket blocks the calling
    thread forever, and an interrupt reaches the caller while the coroutine
    goes on running.
    """
    provider = EchoProvider(ECHO_CONFIG)
    provider.set_response_delay(30.0)
    adapter = SyncProviderAdapter(provider, timeout=0.05)
    try:
        adapter.initialize()

        with pytest.raises(TimeoutError):
            adapter.complete("hello")
    finally:
        adapter.close()


def test_an_injected_bridge_is_shared_rather_than_owned() -> None:
    """One bridge can serve many adapters, and the borrower does not close it.

    ``LLMResource`` builds one adapter per model key, so a resource serving
    three models holds three threads and a multi-tenant service multiplies
    that. ``sync_bridge``'s own module docstring says to share a long-lived
    bridge rather than spawn one per caller; without an injection point there
    is no way to.
    """
    with SyncLoopBridge(thread_name=BRIDGE_THREAD) as bridge:
        first = SyncProviderAdapter(EchoProvider(ECHO_CONFIG), bridge=bridge)
        first.initialize()
        first.close()

        # The bridge belongs to this block, so the borrower's close did not
        # end it and a second adapter can still use it.
        second = SyncProviderAdapter(EchoProvider(ECHO_CONFIG), bridge=bridge)
        try:
            second.initialize()

            assert second.complete("hello").content == "Echo: hello"
        finally:
            second.close()


def test_concurrent_callers_share_the_one_bridge() -> None:
    """Many threads, one adapter, one loop thread --- and no interleaving loss.

    The bridge advertises thread-safety and the adapter now inherits it;
    nothing exercised it.
    """
    adapter = create_llm_provider(ECHO_CONFIG, is_async=False)
    try:
        adapter.initialize()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: adapter.complete(f"m{i}").content, range(32)))

        assert results == [f"Echo: m{i}" for i in range(32)]
        assert len(live_dk_daemon_threads([BRIDGE_THREAD])) == 1
    finally:
        adapter.close()


def test_a_provider_error_reaches_the_caller_unchanged() -> None:
    """The exception crosses the bridge with its identity and its traceback.

    ``run_coroutine_threadsafe`` re-raises in the calling thread; a wrapper
    that reshaped the exception on the way would make every ``except`` clause
    a consumer already wrote wrong.
    """
    boom = RuntimeError("provider unavailable")
    provider = EchoProvider(ECHO_CONFIG)
    provider.set_responses([ErrorResponse(boom)])
    adapter = SyncProviderAdapter(provider)
    try:
        adapter.initialize()

        with pytest.raises(RuntimeError) as excinfo:
            adapter.complete("hello")

        assert excinfo.value is boom
        frames = [frame.name for frame in traceback.extract_tb(excinfo.value.__traceback__)]
        assert "complete" in frames, f"the raising frame is missing from {frames}"
    finally:
        adapter.close()


def test_an_error_inside_the_consumer_still_closes_the_stream() -> None:
    """The consumer raising is the other way ``stream``'s ``finally`` is reached."""
    provider = _CleanupRecordingProvider(ECHO_CONFIG)
    adapter = SyncProviderAdapter(provider)
    try:
        adapter.initialize()

        with pytest.raises(ZeroDivisionError):
            for _ in adapter.stream("hi"):
                raise ZeroDivisionError("consumer failed")

        assert provider.stream_cleanup_ran, "the consumer's error skipped the cleanup"
    finally:
        adapter.close()


class _CloseThreadRecordingProvider(EchoProvider):
    """Records which thread its ``close()`` coroutine actually ran on."""

    closed_on: str | None = None

    async def close(self) -> None:  # type: ignore[override]
        self.closed_on = threading.current_thread().name
        await super().close()


async def test_aclose_awaits_the_provider_instead_of_bridging_to_it() -> None:
    """An async holder must not stall its own loop to tear one of these down.

    ``close()`` reaches the provider through the bridge, which is right for a
    synchronous caller and wrong for an async one: the calling thread --- the
    one running the event loop --- blocks for the provider's whole teardown,
    an HTTP round trip for every provider that is not ``echo``. The thread the
    coroutine runs on is what tells the two apart, so that is what is asserted
    rather than a duration.
    """
    provider = _CloseThreadRecordingProvider(ECHO_CONFIG)
    adapter = SyncProviderAdapter(provider)
    adapter.initialize()
    assert live_dk_daemon_threads([BRIDGE_THREAD]), "the bridge exists to be avoided"

    await adapter.aclose()

    assert provider.closed_on == threading.current_thread().name
    assert provider.closed_on != BRIDGE_THREAD
    assert not live_dk_daemon_threads([BRIDGE_THREAD]), "aclose still ends the thread"


async def test_aclose_is_idempotent_and_agrees_with_close() -> None:
    """The twins share one teardown decision, so either order is safe."""
    adapter = SyncProviderAdapter(EchoProvider(ECHO_CONFIG))
    adapter.initialize()

    await adapter.aclose()
    await adapter.aclose()
    adapter.close()

    assert not adapter.is_initialized
    assert not live_dk_daemon_threads([BRIDGE_THREAD])


def test_a_closed_adapter_refuses_rather_than_building_a_second_bridge() -> None:
    """Reuse after ``close()`` must fail, not quietly allocate a new thread.

    The bridge is built on first use, so an adapter closed before it was ever
    used has no closed bridge to refuse the call --- and would build a fresh
    one that ``close()`` has already run past, leaking it. The adapter has to
    answer for that itself.
    """
    before = set(live_dk_daemon_threads([BRIDGE_THREAD]))
    adapter = create_llm_provider(ECHO_CONFIG, is_async=False)
    adapter.close()

    with pytest.raises(RuntimeError, match="closed"):
        adapter.complete("hello")

    assert set(live_dk_daemon_threads([BRIDGE_THREAD])) == before

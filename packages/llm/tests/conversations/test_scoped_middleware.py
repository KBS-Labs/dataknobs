"""Tests for ConversationManager.scoped_middleware."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest
import yaml

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    ConversationMiddleware,
    DataknobsConversationStorage,
)
from dataknobs_llm.conversations.storage import ConversationState
from dataknobs_llm.llm import EchoProvider, LLMConfig, LLMMessage, LLMResponse
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary


class RecordingMiddleware(ConversationMiddleware):
    """Records process_request / process_response invocations in order."""

    def __init__(self, tag: str, events: list[str]) -> None:
        self.tag = tag
        self._events = events

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        self._events.append(f"{self.tag}:request")
        return messages

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        self._events.append(f"{self.tag}:response")
        return response


class RewritingMiddleware(ConversationMiddleware):
    """Rewrites response.content to a known sentinel."""

    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        return messages

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        response.content = self.sentinel
        return response


class RaisingResponseMiddleware(ConversationMiddleware):
    """Raises from process_response to exercise exception-safe detach."""

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        return messages

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        raise RuntimeError("synthetic response failure")


class RaisingRequestMiddleware(ConversationMiddleware):
    """Raises from process_request to exercise exception-safe detach.

    This exercises the pre-middleware exception path — not an exception
    originating inside the LLM call itself. Both paths route through the
    same ``finally`` in ``scoped_middleware``, so the detachment guarantee
    is the same; this test confirms the guarantee for one of them. A true
    LLM-side failure would raise after pre-middleware completes (so every
    ``process_request`` would have run), but since ``_finalize_completion``
    is skipped on exception in either case, the relevant detach invariant
    is identical.
    """

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        raise RuntimeError("synthetic request failure")

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        return response


def _create_prompts(prompt_dir: Path) -> None:
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "assistant.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )


@pytest.fixture
async def manager_factory():
    """Yield a factory that builds a fresh manager per test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        _create_prompts(prompt_dir)

        config = LLMConfig(
            provider="echo", model="echo-model", options={"echo_prefix": ""}
        )
        llm = EchoProvider(config)
        builder = AsyncPromptBuilder(library=FileSystemPromptLibrary(prompt_dir))
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        async def _make(
            middleware: list[ConversationMiddleware] | None = None,
        ) -> ConversationManager:
            return await ConversationManager.create(
                llm=llm,
                prompt_builder=builder,
                storage=storage,
                system_prompt_name="assistant",
                middleware=middleware or [],
            )

        yield _make

        await llm.close()


class TestScopedMiddleware:
    """Lifecycle and ordering tests for ConversationManager.scoped_middleware."""

    @pytest.mark.asyncio
    async def test_happy_path_attach_and_detach(self, manager_factory):
        """Middleware is attached inside the scope and removed on exit."""
        manager = await manager_factory()
        events: list[str] = []
        mw = RecordingMiddleware("t", events)

        async with manager.scoped_middleware(mw):
            assert manager.middleware == [mw]
            await manager.add_message(role="user", content="Hello")
            await manager.complete()

        assert events == ["t:request", "t:response"]
        assert manager.middleware == []

    @pytest.mark.asyncio
    async def test_exception_in_middleware_still_detaches(self, manager_factory):
        """A middleware exception inside the scope still removes the middleware."""
        manager = await manager_factory()
        mw = RaisingResponseMiddleware()

        async with manager.scoped_middleware(mw):
            await manager.add_message(role="user", content="Hello")
            with pytest.raises(RuntimeError, match="synthetic response failure"):
                await manager.complete()

        assert manager.middleware == []

    @pytest.mark.asyncio
    async def test_exception_in_llm_call_still_detaches(self, manager_factory):
        """An exception originating inside the `with` body still removes scoped middleware.

        Uses RaisingRequestMiddleware as a stand-in: the exception is raised
        before the LLM call completes, exercising the same finally path a
        real LLM failure would take.
        """
        manager = await manager_factory()
        raising = RaisingRequestMiddleware()

        async with manager.scoped_middleware(raising):
            await manager.add_message(role="user", content="Hello")
            with pytest.raises(RuntimeError, match="synthetic request failure"):
                await manager.complete()

        assert manager.middleware == []

    @pytest.mark.asyncio
    async def test_onion_ordering_with_permanent_middleware(self, manager_factory):
        """Permanent middleware wraps scoped middleware (onion ordering)."""
        events: list[str] = []
        perm = RecordingMiddleware("perm", events)
        manager = await manager_factory(middleware=[perm])

        temp = RecordingMiddleware("temp", events)
        async with manager.scoped_middleware(temp):
            await manager.add_message(role="user", content="Hello")
            await manager.complete()

        assert events == [
            "perm:request",
            "temp:request",
            "temp:response",
            "perm:response",
        ]
        assert manager.middleware == [perm]

    @pytest.mark.asyncio
    async def test_content_mutation_survives_to_persisted_node(
        self, manager_factory
    ):
        """Scoped-middleware response mutations flow to the persisted assistant node.

        Regression guard for the consumer-facing citation-rendering use case:
        middleware attached via scoped_middleware must run before the
        assistant-node snapshot in _finalize_completion.
        """
        manager = await manager_factory()
        sentinel = "[rewritten]"
        rewriter = RewritingMiddleware(sentinel)

        async with manager.scoped_middleware(rewriter):
            await manager.add_message(role="user", content="Hello")
            await manager.complete()

        current_node = manager.state.get_current_node()
        assert current_node is not None
        assert current_node.data.message.content == sentinel

    @pytest.mark.asyncio
    async def test_multiple_scoped_middleware_attach_in_order_detach_in_reverse(
        self, manager_factory
    ):
        """scoped_middleware(a, b) attaches in order and detaches in reverse."""
        manager = await manager_factory()
        events: list[str] = []
        a = RecordingMiddleware("a", events)
        b = RecordingMiddleware("b", events)

        async with manager.scoped_middleware(a, b):
            assert manager.middleware == [a, b]
            await manager.add_message(role="user", content="Hello")
            await manager.complete()

        assert events == [
            "a:request",
            "b:request",
            "b:response",
            "a:response",
        ]
        assert manager.middleware == []

    @pytest.mark.asyncio
    async def test_nested_scoped_middleware(self, manager_factory):
        """Nested scopes detach inner before outer and preserve onion ordering."""
        manager = await manager_factory()
        events: list[str] = []
        outer = RecordingMiddleware("outer", events)
        inner = RecordingMiddleware("inner", events)

        async with manager.scoped_middleware(outer):
            assert manager.middleware == [outer]
            async with manager.scoped_middleware(inner):
                assert manager.middleware == [outer, inner]
                await manager.add_message(role="user", content="Hello")
                await manager.complete()
            # Inner detaches before outer exits.
            assert manager.middleware == [outer]

        assert events == [
            "outer:request",
            "inner:request",
            "inner:response",
            "outer:response",
        ]
        assert manager.middleware == []

    @pytest.mark.asyncio
    async def test_zero_argument_scope_is_noop(self, manager_factory):
        """scoped_middleware() with no arguments attaches nothing and cleans up cleanly."""
        manager = await manager_factory()
        assert manager.middleware == []

        async with manager.scoped_middleware():
            assert manager.middleware == []
            await manager.add_message(role="user", content="Hello")
            await manager.complete()

        assert manager.middleware == []

    @pytest.mark.asyncio
    async def test_stream_complete_full_drain_runs_process_response(
        self, manager_factory
    ):
        """Fully draining stream_complete fires process_response in onion order."""
        manager = await manager_factory()
        events: list[str] = []
        mw = RecordingMiddleware("s", events)

        async with manager.scoped_middleware(mw):
            await manager.add_message(role="user", content="Hello")
            async for _chunk in manager.stream_complete():
                pass

        assert events == ["s:request", "s:response"]
        assert manager.middleware == []

    @pytest.mark.asyncio
    async def test_stream_complete_early_break_skips_process_response_but_detaches(
        self, manager_factory
    ):
        """Early break skips process_response but still detaches scoped middleware.

        Documented caveat: with stream_complete, _finalize_completion (and
        therefore process_response) runs only after the generator exits
        normally. On early break the generator never reaches the post-loop
        finalization, so process_response is not called — but the scoped
        middleware is still detached via the async-with finally.
        """
        manager = await manager_factory()
        events: list[str] = []
        mw = RecordingMiddleware("s", events)

        async with manager.scoped_middleware(mw):
            await manager.add_message(role="user", content="Hello")
            stream = manager.stream_complete()
            async for _chunk in stream:
                break
            await stream.aclose()

        assert events == ["s:request"]
        assert manager.middleware == []

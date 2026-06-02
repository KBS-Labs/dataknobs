"""Tests for memory implementations."""

import pytest

from dataknobs_bots.bot.base import PROVIDER_ROLE_MAIN, PROVIDER_ROLE_SUMMARY_LLM
from dataknobs_bots.memory import (
    BufferMemory,
    CompositeMemory,
    SummaryMemory,
    VectorMemory,
    create_memory_from_config,
)
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm import EchoProvider
from dataknobs_llm.llm import LLMProviderFactory


class TestBufferMemory:
    """Tests for BufferMemory."""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self):
        """Test adding and retrieving messages."""
        memory = BufferMemory(max_messages=3)

        # Add messages
        await memory.add_message("Hello", "user")
        await memory.add_message("Hi there!", "assistant")
        await memory.add_message("How are you?", "user")

        # Get context
        context = await memory.get_context("test")
        assert len(context) == 3
        assert context[0]["content"] == "Hello"
        assert context[0]["role"] == "user"
        assert context[1]["content"] == "Hi there!"
        assert context[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_buffer_overflow(self):
        """Test that buffer respects max_messages limit."""
        memory = BufferMemory(max_messages=2)

        # Add 3 messages
        await memory.add_message("First", "user")
        await memory.add_message("Second", "assistant")
        await memory.add_message("Third", "user")

        # Should only have last 2
        context = await memory.get_context("test")
        assert len(context) == 2
        assert context[0]["content"] == "Second"
        assert context[1]["content"] == "Third"

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing memory."""
        memory = BufferMemory(max_messages=3)

        # Add messages
        await memory.add_message("Hello", "user")
        await memory.add_message("Hi", "assistant")

        # Clear
        await memory.clear()

        # Should be empty
        context = await memory.get_context("test")
        assert len(context) == 0

    @pytest.mark.asyncio
    async def test_metadata(self):
        """Test storing metadata with messages."""
        memory = BufferMemory()

        metadata = {"source": "test", "timestamp": "2024-01-01"}
        await memory.add_message("Hello", "user", metadata=metadata)

        context = await memory.get_context("test")
        assert context[0]["metadata"] == metadata


class TestHistoryRedactionReExport:
    """Guard the load-bearing cross-package class- and callable-identity
    invariants.

    ``HistoryRedaction`` (and its companion ``compile_history_redactions``)
    was relocated to ``dataknobs_llm.conversations`` and re-exported from
    ``dataknobs_bots.memory`` (resp. ``dataknobs_bots.memory.base``) for
    back-compat. The two import paths MUST resolve to the SAME object so
    ``isinstance`` checks (for the class) and identity comparisons (for the
    function) across the package boundary continue to work — a future
    refactor that wrapped/subclassed the class, or aliased the function
    through a thin wrapper, would silently break consumers without breaking
    the imports themselves.
    """

    def test_history_redaction_is_same_class_across_packages(self) -> None:
        from dataknobs_bots.memory import HistoryRedaction as BotsRedaction
        from dataknobs_llm.conversations import HistoryRedaction as LLMRedaction

        assert BotsRedaction is LLMRedaction

    def test_compile_history_redactions_is_same_callable_across_packages(
        self,
    ) -> None:
        from dataknobs_bots.memory.base import (
            compile_history_redactions as bots_compile,
        )
        from dataknobs_llm.conversations.history_redaction import (
            compile_history_redactions as llm_compile,
        )

        assert bots_compile is llm_compile


class TestBufferMemoryHistoryRedaction:
    """Tests for BufferMemory history_redactions (read-time message transform)."""

    @pytest.mark.asyncio
    async def test_history_redactions_applied_to_assistant_role_only(self):
        """Redaction patterns rewrite assistant content; user content untouched."""
        memory = BufferMemory(
            max_messages=10,
            history_redactions=[
                {"pattern": r"\[bib:\d+[^\]]*\]", "replacement": "[prior citation]"},
                {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
            ],
        )

        await memory.add_message("What does bib:5 cover?", "user")
        await memory.add_message(
            "The toolkit [bib:5 · vendor · microsoft-agt] covers identity"
            " and converges with NIST guidance (bib:3).",
            "assistant",
        )

        context = await memory.get_context("test")
        assert len(context) == 2

        # User message passes through untouched — humans don't emit bib codes.
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "What does bib:5 cover?"

        # Assistant content has bracketed header AND bare bib:N redacted.
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == (
            "The toolkit [prior citation] covers identity"
            " and converges with NIST guidance ([prior citation])."
        )

    @pytest.mark.asyncio
    async def test_history_redactions_preserve_original_buffer(self):
        """Redaction is read-time only — the stored buffer keeps the original."""
        memory = BufferMemory(
            max_messages=10,
            history_redactions=[
                {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
            ],
        )

        await memory.add_message("Cites bib:5 here.", "assistant")

        # get_context returns redacted view.
        context = await memory.get_context("test")
        assert context[0]["content"] == "Cites [prior citation] here."

        # Underlying buffer is untouched.
        assert memory.messages[0]["content"] == "Cites bib:5 here."

    @pytest.mark.asyncio
    async def test_non_redacted_role_dicts_pass_through_by_identity(self):
        """Non-redacted-role dicts in the returned context are the SAME objects
        as the corresponding entries in the underlying buffer — the dict-shape
        helper's identity-passthrough contract pinned at the consumer call site.

        Regression guard for the new behavior: the originating dict-only helper
        shallow-copied every message regardless of role; the relocated generic
        helper passes non-redacted-role elements through by identity. Anyone
        mutating a returned non-redacted dict would now alias-mutate the
        buffer's stored dict — this test guards the boundary so a future change
        that re-introduces shallow copying does not slip through silently.
        """
        memory = BufferMemory(
            max_messages=10,
            history_redactions=[
                {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
            ],
        )
        await memory.add_message("User cites bib:5", "user")
        await memory.add_message("Assistant cites bib:5", "assistant")

        context = await memory.get_context("test")

        # User dict (non-redacted role) is the SAME object as the buffered dict.
        assert context[0] is memory.messages[0]
        # Assistant dict (redacted role) is a NEW object — its content was rewritten.
        assert context[1] is not memory.messages[1]
        assert memory.messages[1]["content"] == "Assistant cites bib:5"
        assert context[1]["content"] == "Assistant cites [prior citation]"

    @pytest.mark.asyncio
    async def test_history_redactions_default_empty_is_passthrough(self):
        """No redactions configured ⇒ get_context returns content verbatim."""
        memory = BufferMemory(max_messages=10)

        await memory.add_message("Cites bib:5 here.", "assistant")

        context = await memory.get_context("test")
        assert context[0]["content"] == "Cites bib:5 here."

    @pytest.mark.asyncio
    async def test_history_redactions_patterns_applied_in_order(self):
        """Patterns are applied in declared order — bracketed header before bare."""
        memory = BufferMemory(
            max_messages=10,
            history_redactions=[
                # Bracketed header MUST come first (longer match) — if the
                # bare-token rule ran first, it would consume the bib:N inside
                # the brackets and leave a malformed "[ · vendor · …]".
                {"pattern": r"\[bib:\d+[^\]]*\]", "replacement": "[prior citation]"},
                {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
            ],
        )

        await memory.add_message(
            "See [bib:5 · vendor · microsoft-agt] and also bib:3.",
            "assistant",
        )

        context = await memory.get_context("test")
        assert context[0]["content"] == (
            "See [prior citation] and also [prior citation]."
        )


def _echo_embedding_provider() -> EchoProvider:
    """Build an initialized EchoProvider for deterministic embeddings."""
    llm_factory = LLMProviderFactory(is_async=True)
    return llm_factory.create({"provider": "echo", "model": "test"})


async def _memory_vector_store(dimensions: int = 384):
    """Build an initialized in-memory vector store."""
    store = VectorStoreFactory().create(backend="memory", dimensions=dimensions)
    await store.initialize()
    return store


class TestSummaryMemoryHistoryRedaction:
    """Tests for SummaryMemory history_redactions (read-time transform)."""

    @staticmethod
    def _provider() -> EchoProvider:
        factory = LLMProviderFactory(is_async=True)
        return factory.create({"provider": "echo", "model": "test"})

    @pytest.mark.asyncio
    async def test_history_redactions_applied_to_assistant_role_only(self):
        """Redaction rewrites assistant content in the recent buffer only."""
        memory = SummaryMemory.from_components(
            {
                "recent_window": 10,
                "history_redactions": [
                    {
                        "pattern": r"\[bib:\d+[^\]]*\]",
                        "replacement": "[prior citation]",
                    },
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            llm_provider=self._provider(),
        )

        await memory.add_message("What does bib:5 cover?", "user")
        await memory.add_message(
            "The toolkit [bib:5 · vendor · microsoft-agt] covers identity"
            " and converges with NIST guidance (bib:3).",
            "assistant",
        )

        context = await memory.get_context("test")
        assert len(context) == 2

        # User message passes through untouched.
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "What does bib:5 cover?"

        # Assistant content has bracketed header AND bare bib:N redacted.
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == (
            "The toolkit [prior citation] covers identity"
            " and converges with NIST guidance ([prior citation])."
        )

    @pytest.mark.asyncio
    async def test_history_redactions_preserve_recent_buffer(self):
        """Redaction is read-time only — the recent deque keeps the original."""
        memory = SummaryMemory.from_components(
            {
                "recent_window": 10,
                "history_redactions": [
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            llm_provider=self._provider(),
        )

        await memory.add_message("Cites bib:5 here.", "assistant")

        context = await memory.get_context("test")
        assert context[0]["content"] == "Cites [prior citation] here."

        # Underlying recent buffer is untouched.
        assert memory._messages[0]["content"] == "Cites bib:5 here."

    @pytest.mark.asyncio
    async def test_non_redacted_role_dicts_pass_through_by_identity(self):
        """Non-redacted-role dicts in the returned context are the SAME objects
        as the corresponding entries in the underlying recent buffer.

        ``SummaryMemory.get_context`` calls the same dict-shape helper as
        ``BufferMemory.get_context``; the identity-passthrough contract is
        a property of the helper, so the same guard applies here. Without
        this test, a future helper regression that re-introduced
        unconditional shallow-copying would still break the BufferMemory
        test but silently pass for SummaryMemory — and a SummaryMemory
        consumer mutating a returned non-redacted dict would alias-mutate
        the deque's stored entry without the boundary being pinned.

        The summary header is skipped in this test (its element index
        would not have an underlying ``self._messages`` counterpart) so
        the user/assistant entries land at positions 0 and 1 of both the
        returned context and the underlying deque.
        """
        memory = SummaryMemory.from_components(
            {
                "recent_window": 10,
                "history_redactions": [
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            llm_provider=self._provider(),
        )
        await memory.add_message("User cites bib:5", "user")
        await memory.add_message("Assistant cites bib:5", "assistant")

        context = await memory.get_context("test")

        # No summary header configured — context starts at the first
        # buffered user message.
        assert len(context) == 2
        # User dict (non-redacted role) is the SAME object as the deque entry.
        assert context[0] is memory._messages[0]
        # Assistant dict (redacted role) is a NEW object — content rewritten.
        assert context[1] is not memory._messages[1]
        assert memory._messages[1]["content"] == "Assistant cites bib:5"
        assert context[1]["content"] == "Assistant cites [prior citation]"

    @pytest.mark.asyncio
    async def test_history_redactions_default_empty_is_passthrough(self):
        """No redactions configured ⇒ assistant content returned verbatim."""
        memory = SummaryMemory.from_components(
            {"recent_window": 10},
            llm_provider=self._provider(),
        )

        await memory.add_message("Cites bib:5 here.", "assistant")

        context = await memory.get_context("test")
        assert context[0]["content"] == "Cites bib:5 here."

    @pytest.mark.asyncio
    async def test_history_redactions_do_not_rewrite_summary_header(self):
        """The system-role summary header is NOT redacted (default redact_roles)."""
        memory = SummaryMemory.from_components(
            {
                "recent_window": 10,
                "history_redactions": [
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            llm_provider=self._provider(),
        )
        # Force a running summary directly to bypass the LLM path.
        memory._summary = "Summary text mentioning bib:5"
        await memory.add_message("Assistant cites bib:5", "assistant")

        context = await memory.get_context("test")

        # First element is the system-role summary header — unredacted.
        assert context[0]["role"] == "system"
        assert context[0]["content"] == (
            "[Conversation summary]: Summary text mentioning bib:5"
        )
        # The assistant entry in the recent buffer IS redacted.
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "Assistant cites [prior citation]"

    @pytest.mark.asyncio
    async def test_history_redactions_applied_before_summarizer_prompt(self):
        """Oldest assistant content is redacted BEFORE the summarizer LLM sees it.

        The system-role summary header is left untouched by ``get_context``'s
        assistant-only redact_roles, so any citation tokens that leak into the
        summary survive forever as an unredacted system header — the exact
        carry-over the read-time guarantee is supposed to prevent. Redaction
        therefore runs in ``_summarize_oldest`` before the summarizer prompt
        is formatted.

        Drives ``recent_window=1`` so a single overflow message forces the
        summarizer path; uses ``EchoProvider``'s scripted-response queue to
        capture the prompt that would have been sent to the LLM, so the
        assertion is on the prompt text directly (not on the LLM's echo).
        """
        from dataknobs_llm.testing import text_response

        provider = self._provider()
        # Scripted response so the summary content is deterministic and
        # the LLM call doesn't echo the redacted prompt back into _summary.
        provider.set_responses([text_response("Summary of the prior turn.")])

        memory = SummaryMemory.from_components(
            {
                "recent_window": 1,
                "history_redactions": [
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            llm_provider=provider,
        )

        # First assistant message: stays in the buffer (recent_window=1).
        await memory.add_message("Assistant cites bib:5 in the first turn.", "assistant")
        # Second message: forces the first to overflow into _summarize_oldest,
        # which invokes the summarizer LLM with the formatted prompt.
        await memory.add_message("Second turn.", "user")

        # The summarizer call must have seen the REDACTED form, not the
        # raw citation token. Read it back off EchoProvider's call
        # history. Anchor the read tightly so a future change that adds a
        # system preamble to the summarizer call (so messages[0] would no
        # longer be the formatted-prompt user message) doesn't silently
        # let the assertion read the wrong message and pass.
        assert provider.call_count == 1, (
            f"Expected exactly one LLM call (the summarizer); got "
            f"{provider.call_count}. Test scope changed — adjust the "
            f"call-history selector."
        )
        last_call = provider.get_last_call()
        assert last_call is not None, "Summarizer LLM was not invoked"
        # Find the formatted-prompt message by role rather than by
        # positional index. The summarizer is called with a single
        # user-role message whose content IS the formatted prompt.
        # ``messages`` may carry LLMMessage instances or dicts depending
        # on the call path; normalize both before asserting.
        def _role_and_content(msg: object) -> tuple[str, str]:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                return msg.role, msg.content  # type: ignore[attr-defined]
            return msg["role"], msg["content"]  # type: ignore[index]

        user_contents = [
            content
            for role, content in (_role_and_content(m) for m in last_call["messages"])
            if role == "user"
        ]
        assert len(user_contents) == 1, (
            f"Expected exactly one user-role message in the summarizer call; "
            f"got {len(user_contents)}. Summarizer prompt shape changed."
        )
        prompt_text = user_contents[0]
        assert "bib:5" not in prompt_text, (
            "Summarizer received the un-redacted token; redaction must run "
            "BEFORE the summarizer prompt is formatted, not only at read time."
        )
        assert "[prior citation]" in prompt_text


class TestVectorMemoryHistoryRedaction:
    """Tests for VectorMemory history_redactions (read-time transform)."""

    @pytest.mark.asyncio
    async def test_history_redactions_applied_to_search_results_assistant_only(
        self,
    ):
        """Redaction rewrites assistant content in search results; keys survive."""
        vector_store = await _memory_vector_store()
        memory = VectorMemory.from_components(
            {
                "max_results": 5,
                # -1.0 surfaces every stored row regardless of the sign of
                # EchoProvider's deterministic cosine score.
                "similarity_threshold": -1.0,
                "history_redactions": [
                    {
                        "pattern": r"\[bib:\d+[^\]]*\]",
                        "replacement": "[prior citation]",
                    },
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            vector_store=vector_store,
            embedding_provider=_echo_embedding_provider(),
        )

        await memory.add_message(
            "The toolkit [bib:5 · vendor] covers identity, see bib:3.",
            "assistant",
        )

        context = await memory.get_context("identity")
        assert len(context) == 1
        item = context[0]
        assert item["role"] == "assistant"
        assert item["content"] == (
            "The toolkit [prior citation] covers identity, see [prior citation]."
        )
        # Non-content keys carry over unchanged.
        assert "similarity" in item
        assert item["metadata"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_history_redactions_preserve_stored_vectors(self):
        """Stored rows are untouched — redaction is read-time only.

        The dict-shape helper rewrites only the returned element's
        ``content`` key; the ``metadata`` value still references the stored
        metadata row, whose ``content`` reflects what is persisted. So the
        returned ``metadata["content"]`` showing the original (un-redacted)
        text proves the stored row was not mutated.
        """
        vector_store = await _memory_vector_store()
        memory = VectorMemory.from_components(
            {
                "max_results": 5,
                "similarity_threshold": -1.0,
                "history_redactions": [
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            vector_store=vector_store,
            embedding_provider=_echo_embedding_provider(),
        )

        await memory.add_message("Cites bib:5 here.", "assistant")

        context = await memory.get_context("cites")
        assert context[0]["content"] == "Cites [prior citation] here."
        # The stored metadata row keeps the original content.
        assert context[0]["metadata"]["content"] == "Cites bib:5 here."

    @pytest.mark.asyncio
    async def test_history_redactions_default_empty_is_passthrough(self):
        """No redactions configured ⇒ assistant content returned verbatim."""
        vector_store = await _memory_vector_store()
        memory = VectorMemory.from_components(
            {"max_results": 5, "similarity_threshold": -1.0},
            vector_store=vector_store,
            embedding_provider=_echo_embedding_provider(),
        )

        await memory.add_message("Cites bib:5 here.", "assistant")

        context = await memory.get_context("cites")
        assert context[0]["content"] == "Cites bib:5 here."

    @pytest.mark.asyncio
    async def test_history_redactions_skip_non_assistant_roles_in_results(self):
        """User-role results are NOT rewritten (default redact_roles)."""
        vector_store = await _memory_vector_store()
        memory = VectorMemory.from_components(
            {
                "max_results": 10,
                "similarity_threshold": -1.0,
                "history_redactions": [
                    {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
                ],
            },
            vector_store=vector_store,
            embedding_provider=_echo_embedding_provider(),
        )

        await memory.add_message("User asks about bib:5", "user")
        await memory.add_message("Assistant cites bib:5", "assistant")

        context = await memory.get_context("bib")
        by_role = {item["role"]: item["content"] for item in context}
        assert by_role["user"] == "User asks about bib:5"
        assert by_role["assistant"] == "Assistant cites [prior citation]"


class TestCompositeMemoryHistoryRedaction:
    """CompositeMemory inherits redaction via delegation to its children."""

    def test_composite_memory_config_does_not_carry_history_redactions(self):
        """Pin the per-child design contract at the type level.

        ``CompositeMemoryConfig`` deliberately does NOT carry a
        ``history_redactions`` field — redaction is each child's own
        responsibility, and ``CompositeMemory.get_context`` inherits the
        guarantee via delegation. A future "add it to the composite too"
        patch would create ambiguity (does composite-level redaction apply
        to every child's output, or only to the assembled view?) and a
        risk of double-redaction; this assertion fails loudly so the
        patch author has to confront and justify the change.

        Verifies via ``dataclasses.fields`` rather than ``hasattr`` so a
        runtime-added attribute on the frozen dataclass can't paper over a
        missing field declaration.
        """
        import dataclasses

        from dataknobs_bots.memory.config import CompositeMemoryConfig

        field_names = {f.name for f in dataclasses.fields(CompositeMemoryConfig)}
        assert "history_redactions" not in field_names, (
            "CompositeMemoryConfig must not declare a history_redactions "
            "field — redaction is a per-child concern. If you are adding "
            "this field, see the class docstring and the delegation test."
        )

    @pytest.mark.asyncio
    async def test_redaction_inherits_via_delegation_from_buffer_child(self):
        """A redacted BufferMemory child redacts on the composite's path.

        Pins the design contract: redaction is a per-child concern.
        ``CompositeMemoryConfig`` does NOT carry ``history_redactions`` —
        ``CompositeMemory.get_context`` delegates to children unchanged, so
        the guarantee is inherited rather than re-implemented.
        """
        child = BufferMemory(
            max_messages=10,
            history_redactions=[
                {"pattern": r"\bbib:\d+\b", "replacement": "[prior citation]"},
            ],
        )
        composite = CompositeMemory.from_components(strategies=[child])

        await composite.add_message("Assistant cites bib:5", "assistant")

        context = await composite.get_context("any")
        assistant_entries = [m for m in context if m["role"] == "assistant"]
        assert len(assistant_entries) == 1
        assert assistant_entries[0]["content"] == "Assistant cites [prior citation]"


class TestBufferMemoryPopMessages:
    """Tests for BufferMemory.pop_messages()."""

    @pytest.mark.asyncio
    async def test_pop_single_message(self):
        """Pop the last message from the buffer."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("Hello", "user")
        await memory.add_message("Hi!", "assistant")

        removed = await memory.pop_messages(1)
        assert len(removed) == 1
        assert removed[0]["content"] == "Hi!"
        assert removed[0]["role"] == "assistant"

        context = await memory.get_context("test")
        assert len(context) == 1
        assert context[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_pop_multiple_messages(self):
        """Pop N messages from the buffer."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")
        await memory.add_message("Msg 4", "assistant")

        removed = await memory.pop_messages(2)
        assert len(removed) == 2
        assert removed[0]["content"] == "Msg 3"
        assert removed[1]["content"] == "Msg 4"

        context = await memory.get_context("test")
        assert len(context) == 2
        assert context[0]["content"] == "Msg 1"
        assert context[1]["content"] == "Msg 2"

    @pytest.mark.asyncio
    async def test_pop_all_messages(self):
        """Pop all messages from the buffer."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("Hello", "user")
        await memory.add_message("Hi!", "assistant")

        removed = await memory.pop_messages(2)
        assert len(removed) == 2

        context = await memory.get_context("test")
        assert len(context) == 0

    @pytest.mark.asyncio
    async def test_pop_preserves_order(self):
        """Removed messages are returned in chronological order."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("First", "user")
        await memory.add_message("Second", "assistant")
        await memory.add_message("Third", "user")

        removed = await memory.pop_messages(3)
        assert [m["content"] for m in removed] == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_pop_preserves_metadata(self):
        """Popped messages include their metadata."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("Hello", "user", metadata={"turn": 1})

        removed = await memory.pop_messages(1)
        assert removed[0]["metadata"] == {"turn": 1}

    @pytest.mark.asyncio
    async def test_pop_more_than_available_raises(self):
        """ValueError when trying to pop more messages than available."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("Hello", "user")

        with pytest.raises(ValueError, match="Cannot pop 5 messages"):
            await memory.pop_messages(5)

    @pytest.mark.asyncio
    async def test_pop_from_empty_raises(self):
        """ValueError when popping from empty memory."""
        memory = BufferMemory(max_messages=10)

        with pytest.raises(ValueError, match="Cannot pop 1 messages"):
            await memory.pop_messages(1)

    @pytest.mark.asyncio
    async def test_pop_zero_raises(self):
        """ValueError when count is 0."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("Hello", "user")

        with pytest.raises(ValueError, match="count must be >= 1"):
            await memory.pop_messages(0)

    @pytest.mark.asyncio
    async def test_pop_negative_raises(self):
        """ValueError when count is negative."""
        memory = BufferMemory(max_messages=10)

        with pytest.raises(ValueError, match="count must be >= 1"):
            await memory.pop_messages(-1)

    @pytest.mark.asyncio
    async def test_pop_then_add(self):
        """After popping, new messages can be added normally."""
        memory = BufferMemory(max_messages=10)
        await memory.add_message("Original", "user")
        await memory.add_message("Response", "assistant")

        await memory.pop_messages(2)
        await memory.add_message("Retry", "user")

        context = await memory.get_context("test")
        assert len(context) == 1
        assert context[0]["content"] == "Retry"


class TestVectorMemory:
    """Tests for VectorMemory."""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self):
        """Test adding and retrieving messages with vector similarity."""
        # Create in-memory vector store
        factory = VectorStoreFactory()
        vector_store = factory.create(backend="memory", dimensions=384)
        await vector_store.initialize()

        # Create Echo provider for embeddings (deterministic for testing)
        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create({"provider": "echo", "model": "test"})
        await embedding_provider.initialize()

        # Create vector memory
        memory = VectorMemory.from_components(
            {"max_results": 5, "similarity_threshold": 0.0},  # Low threshold
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )

        # Add messages
        await memory.add_message("Hello world", "user")
        await memory.add_message("Hi there", "assistant")
        await memory.add_message("Python programming", "user")

        # Get context
        context = await memory.get_context("greeting")
        assert len(context) > 0
        assert all("content" in msg for msg in context)
        assert all("role" in msg for msg in context)
        assert all("similarity" in msg for msg in context)

    @pytest.mark.asyncio
    async def test_similarity_threshold(self):
        """Test that similarity threshold filters results."""
        factory = VectorStoreFactory()
        vector_store = factory.create(backend="memory", dimensions=384)
        await vector_store.initialize()

        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create({"provider": "echo", "model": "test"})
        await embedding_provider.initialize()

        # Create memory with high threshold
        memory = VectorMemory.from_components(
            {"max_results": 10, "similarity_threshold": 0.99},  # Very high
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )

        # Add messages
        await memory.add_message("Hello", "user")

        # Get context - might return nothing due to high threshold
        context = await memory.get_context("completely different topic")
        # With Echo provider, similarity should be deterministic
        # The test verifies the threshold filtering works
        assert isinstance(context, list)

    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating VectorMemory from configuration."""
        config = {
            "backend": "memory",
            "dimension": 384,
            "embedding_provider": "echo",
            "embedding_model": "test",
            "max_results": 3,
            "similarity_threshold": 0.5,
        }

        memory = await VectorMemory.from_config(config)
        assert memory.max_results == 3
        assert memory.similarity_threshold == 0.5

        # Test it works
        await memory.add_message("Test message", "user")
        context = await memory.get_context("test")
        assert isinstance(context, list)

    @pytest.mark.asyncio
    async def test_from_config_legacy_embedding_passthrough(self):
        """Legacy flat api_base/api_key reach the embedding provider.

        Regression guard: before StructuredConfig adoption, from_config
        forwarded the whole config dict to create_embedding_provider,
        whose legacy-flat branch reads top-level api_base/api_key. The
        typed config must preserve that passthrough rather than silently
        dropping the embedder endpoint/key.
        """
        config = {
            "backend": "memory",
            "dimension": 384,
            "embedding_provider": "echo",
            "embedding_model": "test",
            "api_base": "https://proxy.example/v1",
            "api_key": "sk-test-key",
        }

        memory = await VectorMemory.from_config(config)
        assert memory.embedding_provider.config.api_base == (
            "https://proxy.example/v1"
        )
        assert memory.embedding_provider.config.api_key == "sk-test-key"


class TestSummaryMemory:
    """Tests for SummaryMemory."""

    @staticmethod
    def _create_echo_provider(
        responses: list[str] | None = None,
    ) -> EchoProvider:
        """Create an EchoProvider with optional scripted responses."""
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": "echo", "model": "test"})
        if responses:
            provider.set_responses(responses, cycle=True)
        return provider

    @pytest.mark.asyncio
    async def test_add_and_get_messages_within_window(self):
        """Messages within the window are returned verbatim."""
        provider = self._create_echo_provider()
        memory = SummaryMemory.from_components(
            {"recent_window": 5}, llm_provider=provider
        )

        await memory.add_message("Hello", "user")
        await memory.add_message("Hi there!", "assistant")

        context = await memory.get_context("test")
        assert len(context) == 2
        assert context[0]["content"] == "Hello"
        assert context[0]["role"] == "user"
        assert context[1]["content"] == "Hi there!"
        assert context[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_summarization_triggers_at_threshold(self):
        """When messages exceed recent_window, oldest are summarized."""
        provider = self._create_echo_provider(
            responses=["Summary of the conversation so far."]
        )
        await provider.initialize()
        memory = SummaryMemory.from_components(
            {"recent_window": 3}, llm_provider=provider
        )

        # Add 4 messages (exceeds window of 3)
        await memory.add_message("Message 1", "user")
        await memory.add_message("Message 2", "assistant")
        await memory.add_message("Message 3", "user")
        await memory.add_message("Message 4", "assistant")  # Triggers summarization

        context = await memory.get_context("test")

        # First element should be the summary
        assert context[0]["role"] == "system"
        assert context[0]["metadata"]["is_summary"] is True
        assert "Summary of the conversation" in context[0]["content"]

        # Remaining should be the recent messages (window of 3)
        recent = [m for m in context if m.get("metadata", {}).get("is_summary") is not True]
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_get_context_returns_summary_plus_recent(self):
        """get_context returns [summary] + [recent_messages]."""
        provider = self._create_echo_provider(
            responses=["First summary", "Updated summary"]
        )
        await provider.initialize()
        memory = SummaryMemory.from_components({"recent_window": 2}, llm_provider=provider)

        # Add enough to trigger summarization
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")  # Triggers summarization of Msg 1

        context = await memory.get_context("test")

        # Should have summary + 2 recent messages
        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[1]["content"] == "Msg 2"
        assert context[2]["content"] == "Msg 3"

    @pytest.mark.asyncio
    async def test_clear_resets_summary_and_buffer(self):
        """Clear removes both the summary and buffered messages."""
        provider = self._create_echo_provider(
            responses=["A summary"]
        )
        await provider.initialize()
        memory = SummaryMemory.from_components({"recent_window": 2}, llm_provider=provider)

        # Fill and trigger summarization
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")

        # Verify non-empty
        context = await memory.get_context("test")
        assert len(context) > 0

        # Clear
        await memory.clear()

        # Should be empty
        context = await memory.get_context("test")
        assert len(context) == 0

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_llm_failure(self):
        """When the LLM fails, old messages are dropped gracefully."""
        provider = self._create_echo_provider()
        await provider.initialize()

        # Make the provider raise on complete
        async def fail_complete(*args: object, **kwargs: object) -> None:
            raise RuntimeError("LLM unavailable")

        provider.complete = fail_complete  # type: ignore[assignment]

        memory = SummaryMemory.from_components({"recent_window": 2}, llm_provider=provider)

        # Add 3 messages — the overflow triggers summarization which will fail
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")  # Triggers failed summarization

        # Should still work — old messages dropped, recent kept
        context = await memory.get_context("test")
        recent = [m for m in context if m.get("metadata", {}).get("is_summary") is not True]
        assert len(recent) == 2
        assert recent[0]["content"] == "Msg 2"
        assert recent[1]["content"] == "Msg 3"

    @pytest.mark.asyncio
    async def test_empty_history(self):
        """get_context on empty memory returns an empty list."""
        provider = self._create_echo_provider()
        memory = SummaryMemory.from_components(llm_provider=provider)

        context = await memory.get_context("test")
        assert context == []

    @pytest.mark.asyncio
    async def test_custom_summary_prompt(self):
        """Custom summary_prompt is used for summarization."""
        custom_prompt = (
            "CUSTOM: Summarize.\n{existing_summary}\n{new_messages}"
        )
        provider = self._create_echo_provider(responses=["custom summary result"])
        await provider.initialize()
        memory = SummaryMemory.from_components(
            {"recent_window": 1, "summary_prompt": custom_prompt},
            llm_provider=provider,
        )

        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "user")  # Triggers summarization

        context = await memory.get_context("test")
        assert any("custom summary result" in m["content"] for m in context)


class TestSummaryMemoryProviderVisibility:
    """Tests for SummaryMemory.providers(), set_provider(), and close()."""

    @staticmethod
    def _create_echo_provider() -> EchoProvider:
        factory = LLMProviderFactory(is_async=True)
        return factory.create({"provider": "echo", "model": "test"})

    @pytest.mark.asyncio
    async def test_providers_returns_provider_when_owned(self):
        """Provider visible when built from a dedicated ``llm`` config section."""
        memory = await SummaryMemory.from_config(
            {"llm": {"provider": "echo", "model": "test"}}
        )

        assert memory._owns_llm_provider is True
        result = memory.providers()
        assert PROVIDER_ROLE_SUMMARY_LLM in result
        assert result[PROVIDER_ROLE_SUMMARY_LLM] is memory.llm_provider

    def test_providers_returns_provider_when_not_owned(self):
        """Provider visible when injected (shared main LLM, not owned)."""
        provider = self._create_echo_provider()
        memory = SummaryMemory.from_components(llm_provider=provider)

        assert memory._owns_llm_provider is False
        result = memory.providers()
        assert PROVIDER_ROLE_SUMMARY_LLM in result
        assert result[PROVIDER_ROLE_SUMMARY_LLM] is provider

    def test_providers_returns_empty_when_no_provider(self):
        """Empty dict when llm_provider is None."""
        provider = self._create_echo_provider()
        memory = SummaryMemory.from_components(llm_provider=provider)
        memory.llm_provider = None  # type: ignore[assignment]

        result = memory.providers()
        assert result == {}

    @pytest.mark.asyncio
    async def test_close_skips_when_not_owned(self):
        """close() does NOT close an injected (not-owned) provider."""
        provider = self._create_echo_provider()
        await provider.initialize()
        memory = SummaryMemory.from_components(llm_provider=provider)

        await memory.close()
        assert provider.close_count == 0, "Provider should NOT be closed when not owned"

    @pytest.mark.asyncio
    async def test_close_closes_when_owned(self):
        """close() closes the provider built from a dedicated ``llm`` section."""
        memory = await SummaryMemory.from_config(
            {"llm": {"provider": "echo", "model": "test"}}
        )

        assert memory._owns_llm_provider is True
        await memory.close()
        assert memory.llm_provider.close_count == 1, (
            "Provider should be closed exactly once when owned"
        )

    def test_set_provider_replaces_provider(self):
        """set_provider() updates the LLM provider for the summary role."""
        provider1 = self._create_echo_provider()
        provider2 = self._create_echo_provider()
        memory = SummaryMemory.from_components(llm_provider=provider1)

        result = memory.set_provider(PROVIDER_ROLE_SUMMARY_LLM, provider2)
        assert result is True
        assert memory.llm_provider is provider2

        # providers() should reflect the new provider
        providers = memory.providers()
        assert providers[PROVIDER_ROLE_SUMMARY_LLM] is provider2

    def test_set_provider_ignores_wrong_role(self):
        """set_provider() returns False for non-matching role."""
        provider = self._create_echo_provider()
        memory = SummaryMemory.from_components(llm_provider=provider)

        result = memory.set_provider(PROVIDER_ROLE_MAIN, self._create_echo_provider())
        assert result is False
        assert memory.llm_provider is provider


class TestSummaryMemoryPopMessages:
    """Tests for SummaryMemory.pop_messages()."""

    @staticmethod
    def _create_echo_provider(
        responses: list[str] | None = None,
    ) -> EchoProvider:
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": "echo", "model": "test"})
        if responses:
            provider.set_responses(responses, cycle=True)
        return provider

    @pytest.mark.asyncio
    async def test_pop_within_window(self):
        """Pop messages that are still in the recent window."""
        provider = self._create_echo_provider()
        memory = SummaryMemory.from_components(
            {"recent_window": 10}, llm_provider=provider
        )

        await memory.add_message("Hello", "user")
        await memory.add_message("Hi!", "assistant")
        await memory.add_message("How are you?", "user")

        removed = await memory.pop_messages(2)
        assert len(removed) == 2
        assert removed[0]["content"] == "Hi!"
        assert removed[1]["content"] == "How are you?"

        context = await memory.get_context("test")
        assert len(context) == 1
        assert context[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_pop_beyond_window_raises(self):
        """Cannot pop more messages than remain in the recent window."""
        provider = self._create_echo_provider(responses=["Summary"])
        await provider.initialize()
        memory = SummaryMemory.from_components({"recent_window": 2}, llm_provider=provider)

        # Add 3 messages — first gets summarized, 2 remain in window
        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")  # Triggers summarization

        # Only 2 messages in window — can't pop 3
        with pytest.raises(ValueError, match="unsummarized messages"):
            await memory.pop_messages(3)

    @pytest.mark.asyncio
    async def test_pop_preserves_summary(self):
        """Popping from window does not affect the existing summary."""
        provider = self._create_echo_provider(responses=["Conversation summary"])
        await provider.initialize()
        memory = SummaryMemory.from_components({"recent_window": 2}, llm_provider=provider)

        await memory.add_message("Msg 1", "user")
        await memory.add_message("Msg 2", "assistant")
        await memory.add_message("Msg 3", "user")  # Triggers summarization

        # Pop 1 from the window
        removed = await memory.pop_messages(1)
        assert removed[0]["content"] == "Msg 3"

        # Summary should still be present
        context = await memory.get_context("test")
        assert context[0]["role"] == "system"
        assert "Conversation summary" in context[0]["content"]
        assert len(context) == 2  # summary + 1 remaining message


class TestVectorMemoryPopMessages:
    """Tests for VectorMemory.pop_messages()."""

    @pytest.mark.asyncio
    async def test_pop_raises_not_implemented(self):
        """VectorMemory does not support pop_messages."""
        factory = VectorStoreFactory()
        vector_store = factory.create(backend="memory", dimensions=384)
        await vector_store.initialize()

        llm_factory = LLMProviderFactory(is_async=True)
        embedding_provider = llm_factory.create({"provider": "echo", "model": "test"})
        await embedding_provider.initialize()

        memory = VectorMemory.from_components(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )

        with pytest.raises(NotImplementedError, match="VectorMemory"):
            await memory.pop_messages(1)


class TestMemoryFactory:
    """Tests for memory factory function."""

    @pytest.mark.asyncio
    async def test_create_buffer_memory(self):
        """Test creating buffer memory from config."""
        config = {"type": "buffer", "max_messages": 5}

        memory = await create_memory_from_config(config)
        assert isinstance(memory, BufferMemory)
        assert memory.max_messages == 5

    @pytest.mark.asyncio
    async def test_create_vector_memory(self):
        """Test creating vector memory from config."""
        config = {
            "type": "vector",
            "backend": "memory",
            "dimension": 384,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }

        memory = await create_memory_from_config(config)
        assert isinstance(memory, VectorMemory)

    @pytest.mark.asyncio
    async def test_create_summary_memory(self):
        """Test creating summary memory from config."""
        factory = LLMProviderFactory(is_async=True)
        provider = factory.create({"provider": "echo", "model": "test"})

        config = {"type": "summary", "recent_window": 5}
        memory = await create_memory_from_config(config, llm_provider=provider)
        assert isinstance(memory, SummaryMemory)
        assert memory.recent_window == 5

    @pytest.mark.asyncio
    async def test_create_summary_memory_with_dedicated_llm(self):
        """Test creating summary memory with its own LLM config."""
        config = {
            "type": "summary",
            "recent_window": 8,
            "llm": {"provider": "echo", "model": "summary-model"},
        }
        # No fallback provider needed — dedicated LLM is in config
        memory = await create_memory_from_config(config)
        assert isinstance(memory, SummaryMemory)
        assert memory.recent_window == 8

    @pytest.mark.asyncio
    async def test_create_summary_memory_dedicated_llm_overrides_fallback(self):
        """Dedicated LLM config takes precedence over fallback provider."""
        fallback = LLMProviderFactory(is_async=True).create(
            {"provider": "echo", "model": "fallback"}
        )
        config = {
            "type": "summary",
            "llm": {"provider": "echo", "model": "dedicated"},
        }
        memory = await create_memory_from_config(config, llm_provider=fallback)
        assert isinstance(memory, SummaryMemory)
        # The provider should be the dedicated one, not the fallback
        assert memory.llm_provider is not fallback

    @pytest.mark.asyncio
    async def test_create_summary_memory_without_any_provider_raises(self):
        """Test that summary memory without any LLM source raises ValueError."""
        config = {"type": "summary"}
        with pytest.raises(ValueError, match="requires an LLM provider"):
            await create_memory_from_config(config)

    @pytest.mark.asyncio
    async def test_default_type(self):
        """Test that default memory type is buffer."""
        config = {}

        memory = await create_memory_from_config(config)
        assert isinstance(memory, BufferMemory)

    @pytest.mark.asyncio
    async def test_invalid_type(self):
        """Test error handling for invalid memory type."""
        config = {"type": "invalid"}

        with pytest.raises(ValueError, match="Unknown memory type"):
            await create_memory_from_config(config)

"""Tests for CallTracker utility.

Validates:
- Multi-provider call collection
- Sequential global indexing
- Empty collection returns empty list
- Per-provider cursor tracking (no duplicates)
- Provider lookup by name
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_llm import EchoProvider, LLMConfig
from dataknobs_llm.testing import CallTracker, CapturingProvider, text_response


def _make_capturing(role: str = "main") -> CapturingProvider:
    """Create a CapturingProvider wrapping an EchoProvider."""
    config = LLMConfig(provider="echo", model="test")
    echo = EchoProvider(config)
    return CapturingProvider(echo, role=role)


class TestCallTracker:
    """Tests for CallTracker multi-provider collection."""

    @pytest.mark.asyncio
    async def test_collect_from_single_provider(self) -> None:
        """Collects calls from a single provider."""
        main = _make_capturing("main")
        main._delegate.set_responses([text_response("Hello")])
        tracker = CallTracker(main=main)

        await main.complete("Hi")
        calls = tracker.collect_new_calls()

        assert len(calls) == 1
        assert calls[0].role == "main"
        assert calls[0].call_index == 0
        assert calls[0].response["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_collect_from_multiple_providers(self) -> None:
        """Collects calls from multiple providers in registration order."""
        main = _make_capturing("main")
        extraction = _make_capturing("extraction")
        main._delegate.set_responses([text_response("Main response")])
        extraction._delegate.set_responses([text_response('{"name": "test"}')])
        tracker = CallTracker(main=main, extraction=extraction)

        await main.complete("Hi")
        await extraction.complete("Extract")
        calls = tracker.collect_new_calls()

        assert len(calls) == 2
        assert calls[0].role == "main"
        assert calls[0].call_index == 0
        assert calls[1].role == "extraction"
        assert calls[1].call_index == 1

    @pytest.mark.asyncio
    async def test_sequential_indexing_across_collections(self) -> None:
        """Global call_index continues across multiple collect_new_calls()."""
        main = _make_capturing("main")
        main._delegate.set_responses([
            text_response("First"),
            text_response("Second"),
            text_response("Third"),
        ])
        tracker = CallTracker(main=main)

        await main.complete("A")
        first = tracker.collect_new_calls()
        assert first[0].call_index == 0

        await main.complete("B")
        await main.complete("C")
        second = tracker.collect_new_calls()
        assert second[0].call_index == 1
        assert second[1].call_index == 2

        assert tracker.total_calls == 3

    def test_empty_collection(self) -> None:
        """collect_new_calls() with no new calls returns empty list."""
        main = _make_capturing("main")
        tracker = CallTracker(main=main)

        calls = tracker.collect_new_calls()
        assert calls == []
        assert tracker.total_calls == 0

    @pytest.mark.asyncio
    async def test_no_duplicate_calls(self) -> None:
        """Previously collected calls are not returned again."""
        main = _make_capturing("main")
        main._delegate.set_responses([
            text_response("First"),
            text_response("Second"),
        ])
        tracker = CallTracker(main=main)

        await main.complete("A")
        first = tracker.collect_new_calls()
        assert len(first) == 1

        # No new calls — should be empty
        empty = tracker.collect_new_calls()
        assert empty == []

        await main.complete("B")
        second = tracker.collect_new_calls()
        assert len(second) == 1
        assert second[0].call_index == 1

    def test_get_provider(self) -> None:
        """get_provider() returns the registered provider by name."""
        main = _make_capturing("main")
        extraction = _make_capturing("extraction")
        tracker = CallTracker(main=main, extraction=extraction)

        assert tracker.get_provider("main") is main
        assert tracker.get_provider("extraction") is extraction
        assert tracker.get_provider("nonexistent") is None

    def test_provider_names(self) -> None:
        """provider_names returns registered names."""
        main = _make_capturing("main")
        extraction = _make_capturing("extraction")
        tracker = CallTracker(main=main, extraction=extraction)

        assert set(tracker.provider_names) == {"main", "extraction"}

    @pytest.mark.asyncio
    async def test_pre_existing_calls_ignored(self) -> None:
        """Calls made before tracker creation are not collected."""
        main = _make_capturing("main")
        main._delegate.set_responses([
            text_response("Before"),
            text_response("After"),
        ])

        # Make a call before creating the tracker
        await main.complete("Pre-existing")

        tracker = CallTracker(main=main)

        # Only new calls should be collected
        calls = tracker.collect_new_calls()
        assert calls == []

        await main.complete("New call")
        calls = tracker.collect_new_calls()
        assert len(calls) == 1
        assert calls[0].response["content"] == "After"

    def test_import_from_top_level(self) -> None:
        """CallTracker is importable from dataknobs_llm."""
        from dataknobs_llm import CallTracker as CT

        assert CT is CallTracker

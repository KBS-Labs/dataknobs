# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The ingestion package's sync entry points work from inside a running loop.

``DirectoryProcessor.process`` collected ``process_async()`` through
``asyncio.run``, which raises ``RuntimeError: asyncio.run() cannot be called
from a running event loop`` for any caller already on one. Unlike its
``dataknobs-data`` sibling the limitation was *written down* --- the method's
docstring and the module's both said the wrapper "cannot be called from inside
a running event loop" --- which made it a documented limitation rather than a
silent failure, and removable rather than merely reportable.

Two entry points, not one: ``process_directory`` is the module-level
convenience function and is what a first-time caller reaches for, and it
delegates to ``process()``, so it carried the same limitation without
restating it.

The fix is :func:`~dataknobs_common.sync_bridge.run_coro_sync` --- one
coroutine, driven on a throwaway bridge loop, so nothing runs on the caller's.
One call, so a throwaway bridge rather than a held one: the criterion is how
many coroutines the entry point drives, and ``process()`` drives exactly one
(``_collect`` is a single coroutine however many files it reads).

``the_sync_path_off_a_loop_still_works`` is the over-correction guard: it is
green before the fix as well as after, because the behaviour a caller *not* on
a loop already had is the thing that must survive.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from dataknobs_common.testing import assert_no_leaked_bridge_threads

from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    KnowledgeBaseConfig,
    ProcessedDocument,
)
from dataknobs_xization.ingestion.processor import process_directory


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two files of different types, so the walk has something to dispatch on."""
    (tmp_path / "intro.md").write_text("# Intro\n\nWelcome.\n")
    (tmp_path / "data.json").write_text('[{"title": "A"}, {"title": "B"}]')
    return tmp_path


def test_process_works_from_inside_a_running_loop(corpus: Path) -> None:
    """Red before the fix: ``asyncio.run()`` refuses on a running loop."""

    async def main() -> list[ProcessedDocument]:
        processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), corpus)
        return list(processor.process())

    docs = asyncio.run(main())

    assert [d.document_type for d in docs]


def test_process_directory_works_from_inside_a_running_loop(corpus: Path) -> None:
    """The convenience function inherits the entry point's fix, and its defect."""

    async def main() -> list[ProcessedDocument]:
        return list(process_directory(corpus, KnowledgeBaseConfig(name="t")))

    docs = asyncio.run(main())

    assert [d.document_type for d in docs]


def test_the_sync_path_off_a_loop_still_works(corpus: Path) -> None:
    """The over-correction guard --- green before the fix and after.

    Same documents, same order, still eager: ``process()`` returns an iterator
    over an already-collected list, so a caller that reads ``files_skipped``
    straight after the call still sees the final count.
    """
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), corpus)

    docs = list(processor.process())

    assert [d.document_type for d in docs] == ["markdown", "json"]
    assert processor.files_skipped == 0


def test_the_walk_does_not_run_on_the_callers_thread(corpus: Path) -> None:
    """The mechanism: a bridge loop on its own thread, not the caller's."""
    seen: list[str] = []

    class Witness(DirectoryProcessor):
        async def _collect_files_async(self):  # type: ignore[no-untyped-def]
            seen.append(threading.current_thread().name)
            async for ref in super()._collect_files_async():
                yield ref

    list(Witness(KnowledgeBaseConfig(name="t"), corpus).process())

    assert seen and threading.current_thread().name not in seen


def test_the_bridge_thread_does_not_outlive_the_call(corpus: Path) -> None:
    """One call, one throwaway loop --- and no teardown obligation acquired."""
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), corpus)

    with assert_no_leaked_bridge_threads():
        list(processor.process())
        list(processor.process())

    assert not hasattr(processor, "close")

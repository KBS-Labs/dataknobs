"""The async rungs do not run a caller's tokenizer on the event loop.

A tokenizer is a caller's callable, and nothing bounds what it does: a
segmenter may load its dictionary on first use or reach a service. The rungs
that take one -- :class:`AsyncScanningSignal` and :class:`AsyncLexicalSignal`
-- call it once per query, so a blocking tokenizer called on the loop freezes
every other task there for as long as it runs.

The defect is **non-functional**: the candidates are the same either way, so
no assertion about the answer can see it. The check is the runtime detector,
bracketing the ``await`` alone, plus a structural proof that the tokenizer ran
on a thread other than the loop's -- the detector sees only the syscalls it
patches, and a tokenizer need not make one to be slow.

The default tokenizer, :func:`token_spans`, stays on the loop: it is pure and
linear in the query, and a thread hop would cost more than it saves.
"""

from __future__ import annotations

import threading
import time

import pytest

from dataknobs_common.entity_resolution import (
    AsyncLexicalSignal,
    AsyncScanningSignal,
    token_spans,
)
from dataknobs_common.ontology import AsyncMappingEntitySource, Entity
from dataknobs_common.testing import assert_no_blocking, is_blockbuster_available

pytestmark = pytest.mark.skipif(
    not is_blockbuster_available(),
    reason="blockbuster is required to detect a blocking call on the loop",
)

_QUERY = "a beagle b"


def _source() -> AsyncMappingEntitySource:
    return AsyncMappingEntitySource({"beagle": Entity(id="beagle", type="Breed", name="Beagle")})


def _rungs(tokenizer: object) -> list[AsyncScanningSignal | AsyncLexicalSignal]:
    return [
        AsyncScanningSignal(_source(), tokenizer=tokenizer, max_window=3, max_edge=0),
        AsyncLexicalSignal(_source(), tokenizer=tokenizer),
    ]


def _blocking_tokenizer(text: str) -> tuple[tuple[int, int], ...]:
    """The default's tokens, after a call the detector reports on a live loop."""
    time.sleep(0)
    return token_spans(text)


@pytest.mark.parametrize("index", [0, 1], ids=["scanning", "lexical"])
async def test_a_blocking_tokenizer_does_not_run_on_the_event_loop(index: int) -> None:
    rung = _rungs(_blocking_tokenizer)[index]

    with assert_no_blocking():
        found = await rung.candidates(_QUERY, k=5)

    assert [c.entity_id for c in found] == ["beagle"]


@pytest.mark.parametrize("index", [0, 1], ids=["scanning", "lexical"])
async def test_a_caller_s_tokenizer_runs_on_a_worker_thread(index: int) -> None:
    ran_on: list[int] = []

    def recording(text: str) -> tuple[tuple[int, int], ...]:
        ran_on.append(threading.get_ident())
        return token_spans(text)

    await _rungs(recording)[index].candidates(_QUERY, k=5)

    assert ran_on, "the rung never called its tokenizer"
    assert threading.get_ident() not in ran_on, "the tokenizer ran on the event loop's thread"

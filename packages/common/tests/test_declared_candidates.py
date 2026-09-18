"""The assembly a rung over declared forms owes, published for rungs written elsewhere.

``declared_candidates`` was private while :class:`DeclaredSignal` was the only
way to reach it. That class hard-requires an ``EntitySource`` and its own
docstring names the rung it cannot serve -- one whose backing is not a
dictionary lookup -- so a rung written against the bare protocol had the whole
assembly left to reimplement, and the shape of a declared hit's evidence
existed in as many copies as there were such rungs.

These are the claims a second caller now depends on, asserted here rather than
through whichever rung happens to call it: a claim tested only through
``DeclaredSignal`` is a claim about ``DeclaredSignal``.
"""

from __future__ import annotations

import pytest

from dataknobs_common.entity_resolution import (
    EvidenceKind,
    FormHit,
    Scoring,
    declared_candidates,
)
from dataknobs_common.exceptions import ValidationError

QUERY = "my golden retriever met a beagle"

#: The containing form first, which is the order a rung that cares publishes.
FOUND = [
    FormHit(entity_id="golden_retriever", span=(3, 19)),
    FormHit(entity_id="retriever", span=(10, 19)),
    FormHit(entity_id="beagle", span=(25, 31)),
]


def _ids(candidates) -> list[str]:
    return [candidate.entity_id for candidate in candidates]


def test_every_hit_becomes_a_candidate_when_nothing_narrows():
    """The default admits every id found.

    Which is what a rung answering ``narrows() is False`` needs: it is never
    offered a filter, so a required ``admitted`` argument would have made
    every such rung pass a set it had just built from its own hits.
    """
    assert _ids(declared_candidates(FOUND, k=5, signal="test", query=QUERY)) == [
        "golden_retriever",
        "retriever",
        "beagle",
    ]


def test_the_order_found_is_the_order_returned():
    """A declared score is 1.0 by fiat and carries no order, so the rung's stands."""
    reversed_hits = list(reversed(FOUND))

    assert _ids(declared_candidates(reversed_hits, k=5, signal="test", query=QUERY)) == [
        "beagle",
        "retriever",
        "golden_retriever",
    ]


def test_k_counts_entities_rather_than_hits():
    """One entity found twice is one candidate carrying two pieces of evidence.

    The same shape the cascade produces when two *rungs* agree on an id, so a
    rung that found it twice itself does not spend two of the caller's ``k``
    on one answer.
    """
    twice = [
        FormHit(entity_id="retriever", span=(3, 12)),
        FormHit(entity_id="retriever", span=(20, 29)),
    ]

    (candidate,) = declared_candidates(twice, k=1, signal="test", query="a retriever, a retriever")

    assert candidate.entity_id == "retriever"
    assert [evidence.span for evidence in candidate.evidence] == [(3, 12), (20, 29)]


def test_k_cuts_entities_in_the_order_they_were_proposed():
    assert _ids(declared_candidates(FOUND, k=2, signal="test", query=QUERY)) == [
        "golden_retriever",
        "retriever",
    ]


def test_matched_text_is_sliced_from_the_query():
    """So the text and the span agree by construction rather than by a caller's trust.

    The caller hands over spans and a query and never hands over text, which
    is what makes this unfalsifiable by a rung that computes its own.
    """
    candidates = declared_candidates(FOUND, k=5, signal="test", query=QUERY)

    for candidate in candidates:
        for evidence in candidate.evidence:
            assert evidence.matched_text == QUERY[evidence.span[0] : evidence.span[1]]

    assert candidates[0].evidence[0].matched_text == "golden retriever"


def test_the_evidence_is_declared_by_fiat_and_carries_the_rungs_name():
    """1.0 with :attr:`Scoring.DECLARED`, and the ``signal`` the caller named.

    The ``signal`` is what lets a consumer reading ``evidence.signal``
    correlate a hit back to the ``kind:`` they configured, so it is the
    caller's string and not a constant of this function.
    """
    (candidate,) = declared_candidates(FOUND[:1], k=1, signal="authority", query=QUERY)
    (evidence,) = candidate.evidence

    assert candidate.score == 1.0
    assert evidence.score == 1.0
    assert evidence.kind is EvidenceKind.DECLARED
    assert evidence.scoring is Scoring.DECLARED
    assert evidence.signal == "authority"


def test_admitted_drops_an_id_a_filter_ruled_out():
    """The rung-side narrowing, for a rung that answers ``narrows() is True``."""
    admitted = frozenset({"golden_retriever", "beagle"})

    assert _ids(declared_candidates(FOUND, k=5, signal="test", query=QUERY, admitted=admitted)) == [
        "golden_retriever",
        "beagle",
    ]


def test_an_empty_admitted_set_is_not_the_same_as_no_filter():
    """Empty admits nothing; ``None`` admits everything.

    The distinction the default exists for, and the one a falsy test would
    collapse -- a rung whose filter ruled out every id must return nothing,
    not everything.
    """
    assert declared_candidates(FOUND, k=5, signal="test", query=QUERY, admitted=frozenset()) == []


def test_no_hits_is_no_candidates():
    assert declared_candidates([], k=5, signal="test", query=QUERY) == []


def test_a_caller_asking_for_nothing_gets_nothing():
    """Zero is a real request, and the slice already answers it."""
    assert declared_candidates(FOUND, k=0, signal="test", query=QUERY) == []


def test_a_negative_k_is_refused_rather_than_read_as_counting_back():
    """The cut is a list slice, and a slice reads ``-1`` as *all but the last*.

    So ``k=-1`` returned two of the three entities: not an error, not the
    empty list, and indistinguishable from a rung that genuinely found two.
    Nothing upstream validates ``k`` -- a resolver takes it as a keyword and
    hands it down untouched -- so this is where it first becomes visible, and
    refusing it here refuses it for every rung that assembles through this
    function.
    """
    with pytest.raises(ValidationError, match="must not be negative"):
        declared_candidates(FOUND, k=-1, signal="test", query=QUERY)

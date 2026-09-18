"""What the shipped rungs put in their evidence, asserted before the base could score.

Every rung in this package assembles through
:func:`~dataknobs_common.entity_resolution.declared_candidates`, which wrote
three constants into every piece of evidence it produced: ``1.0``,
:attr:`EvidenceKind.DECLARED` and :attr:`Scoring.DECLARED`. A scored rung needs
two of those to come from the rung and the third from the hit, so the base
grows them as **defaults** -- and a default that reproduces a constant and a
default that quietly redefines it are indistinguishable from the far side of
the change.

So this file is written first, run green against the unwidened code, and run
green again after. That is the inverse of the usual reproduce-first shape and
deliberately so: there is no defect here and nothing to make fail. What the
order buys is that the claim *these rungs are unchanged* is asserted by a test
that could not have been written to match whatever the widening happened to
produce.

It goes through the **rungs** rather than through the function, which
``test_declared_candidates.py`` already covers at its own layer. The widening
moves values from the function's body onto the classes, so a rung is the only
place the two can be seen to agree.
"""

from __future__ import annotations

import pytest

from dataknobs_common.entity_resolution import (
    AliasSignal,
    FormHit,
    AsyncAliasSignal,
    AsyncExactNormalizedSignal,
    AsyncScanningSignal,
    EvidenceKind,
    ExactNormalizedSignal,
    ScanningSignal,
    Scoring,
)
from dataknobs_common.ontology import AsyncMappingEntitySource, Entity, MappingEntitySource

#: One query reaching all three rungs: ``beagle`` is a name, ``beagles`` an
#: alias, and both sit inside a sentence so the scan has somewhere to scan.
QUERY = "my beagles met a golden retriever"

VOCABULARY = {
    "beagle": Entity(id="beagle", type="Breed", name="Beagle", aliases=("Beagles",)),
    "golden_retriever": Entity(id="golden_retriever", type="Breed", name="Golden Retriever"),
}


@pytest.fixture
def entities() -> MappingEntitySource:
    return MappingEntitySource(VOCABULARY)


@pytest.fixture
def async_entities() -> AsyncMappingEntitySource:
    return AsyncMappingEntitySource(VOCABULARY)


def _declared(candidates) -> None:
    """Every candidate and every piece of its evidence, as a declared rung means it."""
    assert candidates, "the rung found nothing, so this asserts nothing"
    for candidate in candidates:
        assert candidate.score == 1.0
        for evidence in candidate.evidence:
            assert evidence.score == 1.0
            assert evidence.kind is EvidenceKind.DECLARED
            assert evidence.scoring is Scoring.DECLARED


def test_an_unmeasured_hit_carries_no_score_rather_than_a_declared_one() -> None:
    """The half of the widening the rung-level assertions below cannot see.

    Every assertion in this file reads ``evidence.score``, which
    :func:`~dataknobs_common.entity_resolution.declared_candidates` fills
    with ``1.0`` when the hit carries nothing -- so a default that quietly
    became ``1.0`` on the **field** would satisfy all of them while claiming
    a measurement nobody took. ``None`` is the whole distinction
    :attr:`Scoring.DECLARED` exists to mark, and this is where it is asserted.
    """
    assert FormHit(entity_id="beagle", span=(0, 6)).score is None


@pytest.mark.parametrize(
    ("rung", "query"),
    [(ExactNormalizedSignal, "beagles"), (AliasSignal, "beagles"), (ScanningSignal, QUERY)],
)
def test_a_shipped_rung_declares_by_fiat(entities, rung, query) -> None:
    """The whole-string pair on their own form, the scan on a sentence.

    Each rung is asked the query it can answer rather than one shared string,
    because a rung that found nothing would satisfy a loop over its evidence
    without asserting anything -- which is what the guard inside
    :func:`_declared` refuses.
    """
    _declared(rung(entities).candidates(query, k=5))


@pytest.mark.parametrize(
    ("rung", "query"),
    [
        (AsyncExactNormalizedSignal, "beagles"),
        (AsyncAliasSignal, "beagles"),
        (AsyncScanningSignal, QUERY),
    ],
)
@pytest.mark.asyncio
async def test_an_async_shipped_rung_declares_by_fiat(async_entities, rung, query) -> None:
    """The twin, because the widening lands on two bases and could land on one."""
    _declared(await rung(async_entities).candidates(query, k=5))

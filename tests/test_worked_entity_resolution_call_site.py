"""The published scanning call site runs, from outside the package that ships it.

A consumer hands over a *sentence*, not a phrase: they do not know where the
declared forms are, how many there are, or what the vocabulary missed. That
claim is easy to make in a design document and cheap to check exactly once -- a
call site fails. It cannot be written around a class that does not exist, or
around a name a door does not export.

So this runs the call site
``packages/common/docs/guides/entity-resolution.md`` publishes, as written,
against the vocabulary the guide publishes beside it -- and it runs it from the
workspace root rather than from ``packages/common/tests``. That is not a filing
preference. A test living inside the package can reach anything the package
defines whether or not the door exports it, which makes it structurally unable
to fail for the one reason this test exists: a name that is implemented and not
published. :mod:`tests.test_worked_ontology_call_site` is the sibling this
copies, and states the same reason for the same location.

**Why the rung is named rather than built by the door.** The block constructs
``ScanningSignal`` itself. A page showing a reader how to find declared forms
inside a sentence should name the rung that does it -- and a block that reached
``build_resolver`` instead would pass this file while the published spelling of
the class stayed untested, which is half of what the criterion behind this
guard is for. ``test_the_call_site_names_the_shipped_rung`` is where that stops
being incidental.

**Where the input comes from.** The ``worked-input`` fence is the same
vocabulary as ``MAMMALS_V11_DOCUMENT`` in ``packages/common/tests/conftest.py``,
and ``packages/common/tests/test_worked_input_fences.py`` holds the two
identical. Until that file arrived nothing did: each copy was guarded by its own suite -- a fence that
drifts takes this file red, a constant that drifts takes
``test_resolution_spans.py`` red -- which left *that they are one document*
guarded by nobody, and the two had already parted by a line. This is the pair
where that costs something, because the offsets asserted below are offsets into
a query resolved against the forms the fence declares.

The assertions come after the block rather than inside it, because the block's
lines are mostly bare expressions whose values are discarded -- the form that
shows a reader what each call answers. Re-calling them to assert is only sound
because every accessor here is pure over the vocabulary's own fields: nothing
opens anything and nothing is lazy.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._workspace import ROOT, executed_source, published_fence

if TYPE_CHECKING:
    from collections.abc import Iterator

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "entity-resolution.md"
EXECUTED = ROOT / "tests" / "worked_entity_resolution_call_site.py"

INPUT_MARKER = "worked-input"
CALL_SITE_MARKER = "worked-call-site"

#: The sentence the block resolves, and the two forms it carries.
QUERY = "my golden retriever has been limping"


@pytest.fixture
def vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``mammals.yaml`` on disk, written from the guide, and made the cwd.

    The call site names the file by a bare relative path, which is what a
    reader would type. Honouring that means giving it a directory rather than
    rewriting the line to suit the harness.
    """
    (tmp_path / "mammals.yaml").write_text(
        published_fence(GUIDE, INPUT_MARKER) + "\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def ran(vocabulary: Path) -> dict[str, Any]:
    """What the call site leaves bound, having run it exactly as published."""
    return runpy.run_path(str(EXECUTED), run_name="__worked_call_site__")


def test_the_executed_copy_is_the_published_one() -> None:
    """Character for character, in both directions.

    This is what makes *verbatim* a property of the tree rather than an
    intention. Without it the guide could drift from the code, or the code from
    the guide, and every other assertion in this file would keep passing.
    """
    published = published_fence(GUIDE, CALL_SITE_MARKER)
    executed = executed_source(EXECUTED)

    assert executed == published, (
        "the executed call site and the one the guide publishes have diverged. "
        "They are one text with two homes: edit the fence in "
        f"{GUIDE.relative_to(ROOT)} and copy it to {EXECUTED.relative_to(ROOT)}, "
        "or the reverse -- but never one alone."
    )


def test_the_call_site_imports_only_through_the_doors() -> None:
    """Every import in the block is a package door, not a module path.

    The point of the leg this test arrives with is that these names are
    *exported*. A call site reaching into
    ``dataknobs_common.entity_resolution.signals`` would run identically and
    would prove nothing about that, so the spelling is the subject and not an
    incidental.
    """
    doors = {"dataknobs_common", "dataknobs_common.entity_resolution", "dataknobs_common.ontology"}
    reached = {
        line.split()[1]
        for line in published_fence(GUIDE, CALL_SITE_MARKER).splitlines()
        if line.startswith("from dataknobs_common")
    }

    assert reached, "the call site imports nothing from this package"
    assert reached <= doors, (
        f"{sorted(reached - doors)} is a module path rather than a door. "
        f"The call site is the increment's own acceptance that the names are "
        f"published, so reaching past a door defeats it"
    )


def test_the_call_site_names_the_shipped_rung() -> None:
    """``ScanningSignal``, spelled out, rather than a cascade the door composed.

    The three placement assertions below are true of *any* rung that locates
    forms, which is correct -- they are about what a cascade returns. This is
    the one that is about the shipped class: a page teaching a reader to find
    forms inside a sentence, while quietly building its own rung or reaching
    for a default that does not scan, would satisfy every other guard here.
    """
    published = published_fence(GUIDE, CALL_SITE_MARKER)

    assert "ScanningSignal(" in published, (
        "the published call site no longer constructs the shipped scanning "
        "rung. Every other assertion in this file passes against a rung the "
        "page defined for itself, which is the failure this one exists for"
    )
    assert "class " not in published, (
        "the page defines a rung instead of constructing the one that ships"
    )


def test_the_declared_forms_the_sentence_carries_come_back_longest_first(
    ran: dict[str, Any],
) -> None:
    """Step 1: two forms at overlapping spans, and both are candidates.

    Choosing the longer one and dropping the other is a verdict, and this
    family refuses verdicts. The order is the half a consumer cannot recover
    for themselves: a declared score is ``1.0`` by fiat, so if the rung
    published a different order nothing downstream would notice.
    """
    result = ran["result"]

    assert [c.entity_id for c in result.candidates] == ["golden_retriever", "retriever"]
    assert result.query == QUERY


def test_each_form_reports_where_it_sat(ran: dict[str, Any]) -> None:
    """Step 2: half-open offsets into ``result.query``, and the slice agrees.

    ``matched_text`` is asserted against the query sliced by the span rather
    than against the literal on the page, so the claim is *the two fields
    agree* rather than two copies of one expectation that can be edited apart.
    """
    result = ran["result"]

    assert result.explain("golden_retriever")[0].span == (3, 19)
    assert result.explain("retriever")[0].span == (10, 19)
    assert result.explain("golden_retriever")[0].matched_text == "golden retriever"
    assert result.query[3:19] == result.explain("golden_retriever")[0].matched_text


def test_the_coverage_reports_what_the_vocabulary_did_not_account_for(
    ran: dict[str, Any],
) -> None:
    """Step 3: the union of the evidence spans, and the residue.

    ``(10, 19)`` is inside ``(3, 19)``, so ``matched`` is one interval rather
    than the two the rungs reported -- coverage is derived from the evidence
    rather than computed a second time beside it. The residue is trimmed of
    the whitespace that bounded it, which is why ``"my"`` is ``(0, 2)`` and not
    ``(0, 3)``.
    """
    result = ran["result"]

    assert result.coverage.matched == ((3, 19),)
    assert result.coverage.unmatched == ((0, 2), (20, 36))
    assert result.unmatched_text() == ("my", "has been limping")
    assert result.matched_text() == ("golden retriever",)


def test_nothing_in_the_call_site_needs_a_loop_or_a_backend(ran: dict[str, Any]) -> None:
    """The claim the block's own comment makes, asserted rather than repeated.

    ``ran`` having been produced at all is most of the proof -- the module ran
    to completion outside any event loop, so nothing in it awaited. What is
    left is the other half: that the vocabulary describes itself as authored,
    rather than as something with a backend behind it.
    """
    description = ran["onto"].entities.describe()

    assert description.backend == "authored"
    assert description.table is None

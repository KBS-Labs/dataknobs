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
vocabulary as ``MAMMALS_V11_DOCUMENT`` in
``packages/common/tests/_vocabularies.py``, and
``packages/common/tests/test_worked_input_fences.py`` holds the two identical. Until that file arrived nothing did: each copy was guarded by its own suite -- a fence that
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

import ast
import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.entity_resolution import BridgedEntityResolver

from tests._workspace import ROOT, door_imports, executed_source, published_fence

if TYPE_CHECKING:
    from collections.abc import Iterator

GUIDE = ROOT / "packages" / "common" / "docs" / "guides" / "entity-resolution.md"
EXECUTED = ROOT / "tests" / "worked_entity_resolution_call_site.py"

INPUT_MARKER = "worked-input"
CALL_SITE_MARKER = "worked-call-site"
BRIDGE_MARKER = "worked-bridge"

BRIDGE_EXECUTED = ROOT / "tests" / "worked_entity_resolution_bridge.py"

#: The daemon thread the published block allocates, named off the wrapper that
#: allocates it rather than off the bridge's default. Read from the class so a
#: rename there moves the watch set with it, and shared by the leak assertion
#: and its positive control so the two cannot come to watch different things.
BRIDGE_THREADS = (BridgedEntityResolver.BRIDGE_THREAD_NAME,)

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
    reached = door_imports(published_fence(GUIDE, CALL_SITE_MARKER))

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
        "page defined for itself, which is the failure this one exists for. "
        "The trailing parenthesis is the assertion: without it the import "
        "line alone satisfies this, which it briefly did"
    )

    # Parsed rather than searched. `"class " not in published` is the same
    # claim spelled so that a comment or a string containing the word answers
    # it -- and the thing being ruled out is a *definition*, which the grammar
    # knows about and a substring does not.
    defined = [
        node.name for node in ast.walk(ast.parse(published)) if isinstance(node, ast.ClassDef)
    ]
    assert not defined, (
        f"the published call site defines {defined} instead of constructing "
        f"the rung that ships. A page teaching a reader to find declared forms "
        f"inside a sentence, while writing its own rung to do it, satisfies "
        f"every other assertion in this file"
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


RUNG_MARKER = "worked-rung"
RUNG = ROOT / "tests" / "worked_punctuated_rung.py"


@pytest.fixture
def punctuated_rung() -> type:
    """The class the guide's *Writing your own rung* fence defines, as published.

    The ``isinstance`` is what makes the annotation a checked claim rather than
    one the reader has to take on trust: :func:`runpy.run_path` hands back a
    namespace of ``Any``, so nothing else here would notice the fence binding
    that name to something other than a class.
    """
    published = runpy.run_path(str(RUNG), run_name="__worked_rung__")["PunctuatedFormRung"]
    assert isinstance(published, type), (
        f"{RUNG.relative_to(ROOT)} no longer binds PunctuatedFormRung to a class, so the "
        "fence has stopped defining the rung the guide tells a reader to write"
    )
    return published


def test_the_executed_rung_is_the_published_one() -> None:
    """The second fence on this page is held to the first one's standard.

    The page describes itself as executed, and until this existed one block on
    it was not: the sample a reader is most likely to copy, since it is the one
    the page tells them to write themselves. Its claims are also the sharper
    ones -- three named forms said to be reachable this way and no other -- and
    a claim a reader cannot check without first writing the class is exactly
    the kind that should not rest on prose.
    """
    published = published_fence(GUIDE, RUNG_MARKER)
    executed = executed_source(RUNG)

    assert executed == published, (
        "the executed rung and the one the guide publishes have diverged. They "
        "are one text with two homes: edit the fence in "
        f"{GUIDE.relative_to(ROOT)} and copy it to {RUNG.relative_to(ROOT)}, or "
        "the reverse -- but never one alone."
    )


def test_the_published_rung_reaches_the_three_forms_the_page_names(
    punctuated_rung: type,
) -> None:
    """`(beagle)`, `C.D.C.` and `K-9`, and the shipped scan reaching two fewer.

    Both halves are the page's claim and both have to hold, because the section
    exists to answer *what is worth writing yourself*. A rung that found the
    three forms would still not justify the section if the shipped scan found
    them too, so the contrast is asserted rather than described.
    """
    from dataknobs_common.entity_resolution import ScanningSignal
    from dataknobs_common.ontology import Entity, MappingEntitySource

    vocabulary = MappingEntitySource(
        {"k9": Entity(id="k9", type="Thing", name="K-9", aliases=["(beagle)", "C.D.C."])}
    )
    written = punctuated_rung(vocabulary)
    shipped = ScanningSignal(vocabulary)

    for form in ("(beagle)", "C.D.C.", "K-9"):
        found = written.candidates(form, k=5)
        assert [c.entity_id for c in found] == ["k9"], f"the published rung misses {form!r}"
        assert found[0].evidence[0].span == (0, len(form))

    assert [c.entity_id for c in shipped.candidates("K-9", k=5)] == ["k9"]
    assert shipped.candidates("(beagle)", k=5) == []
    assert shipped.candidates("C.D.C.", k=5) == []


def test_the_published_rung_reaches_no_multi_word_form(punctuated_rung: type) -> None:
    """The other half of the trade, measured where the page states it.

    A chunk is whitespace-delimited, so the rung the page teaches cannot join
    two of them -- it reaches ``(beagle)`` and reaches ``golden retriever``
    never, for any query. That is the cost of the boundary it picks, and it is
    exactly invisible in the test above, whose vocabulary is single-chunk
    forms only: a reader with a vocabulary carrying both kinds copies the
    block and silently loses half of it.

    So the loss is asserted rather than described, and asserted beside the
    shipped rung finding the same form -- because *compose rather than choose*
    is the page's conclusion, and a reader has to be able to see that the two
    rungs fail in opposite directions.
    """
    from dataknobs_common.entity_resolution import ScanningSignal
    from dataknobs_common.ontology import Entity, MappingEntitySource

    vocabulary = MappingEntitySource(
        {"golden_retriever": Entity(id="golden_retriever", type="Breed", name="Golden Retriever")}
    )
    query = "my golden retriever has been limping"

    assert punctuated_rung(vocabulary).candidates(query, k=5) == [], (
        "the published rung reached a multi-word form. It probes one "
        "whitespace-delimited chunk at a time and never joins two, so this "
        "passing would mean the fence no longer says what the page explains"
    )
    assert [c.entity_id for c in ScanningSignal(vocabulary).candidates(query, k=5)] == [
        "golden_retriever"
    ], "the shipped scan is the half of the composition that reaches it"


# --------------------------------------------------------------------------
# The second fence: the bridge, which is the same page's other published block
# --------------------------------------------------------------------------


@pytest.fixture
def bridged(vocabulary: Path) -> dict[str, Any]:
    """The bridge block, run the way a reader would run it.

    Its own fixture rather than a second call inside :func:`ran`, because
    ``published_fence`` admits exactly one fence per marker and two markers are
    two *programs*: ``runpy.run_path`` gives each its own namespace, so nothing
    here is inherited from the call site above. That is the property the block
    is written against --- it builds its own cascade instead of reaching for
    one the page bound earlier.
    """
    return runpy.run_path(str(BRIDGE_EXECUTED), run_name="__worked_bridge__")


def test_the_executed_bridge_is_the_published_one() -> None:
    """Character for character, as the call site's copy is held."""
    published = published_fence(GUIDE, BRIDGE_MARKER)
    executed = executed_source(BRIDGE_EXECUTED)

    assert executed == published, (
        "the executed bridge block and the one the guide publishes have "
        f"diverged. Edit the fence in {GUIDE.relative_to(ROOT)} and copy it to "
        f"{BRIDGE_EXECUTED.relative_to(ROOT)}, or the reverse -- never one alone."
    )


def test_the_bridge_block_imports_only_through_the_doors() -> None:
    """The same three doors, and the same reason.

    This page is ``dataknobs_common``'s and teaches a rung from another
    distribution through the *mark* -- an install sentence and a link -- rather
    than by importing it. So the door set is unchanged by the arrival of a
    second block, and a block that widened it would be teaching the wrong thing
    on the wrong page.
    """
    doors = {"dataknobs_common", "dataknobs_common.entity_resolution", "dataknobs_common.ontology"}
    reached = door_imports(published_fence(GUIDE, BRIDGE_MARKER))

    assert reached, "the bridge block imports nothing from this package"
    assert reached <= doors, f"{sorted(reached - doors)} is a module path rather than a door"


def test_the_bridge_block_asserts_what_it_teaches(bridged: dict[str, Any]) -> None:
    """Running it *is* the check, so a block with no assertions checks nothing.

    The published shape puts the cascade's construction inside an
    ``asyncio.run`` and the resolve outside it, which leaves ``result`` at
    module scope -- so unlike the sibling page's async block this one *could*
    be read from outside. It asserts inside itself anyway: the two published
    blocks on the two pages are read together, and a reader who copies one
    should not find that only the other says what it expects.
    """
    published = published_fence(GUIDE, BRIDGE_MARKER)

    asserted = [node for node in ast.walk(ast.parse(published)) if isinstance(node, ast.Assert)]
    assert len(asserted) >= 2, "the bridge block no longer asserts what it teaches"

    result = bridged["result"]
    assert [candidate.entity_id for candidate in result.candidates] == ["beagle"]

    # The `with` is the half a reader most easily drops, and dropping it leaks
    # a daemon thread per resolver rather than failing. Asserted through the
    # published reader rather than the object's private flag, because what a
    # reader copying this block can go and check for themselves is the thread.
    from dataknobs_common.testing import live_dk_daemon_threads

    assert live_dk_daemon_threads(BRIDGE_THREADS) == [], (
        "the block leaves the bridge's daemon thread running, which is what the `with` is for"
    )


def test_the_watch_set_sees_the_thread_this_block_allocates(bridged: dict[str, Any]) -> None:
    """The positive control, without which the assertion above cannot fail correctly.

    ``live_dk_daemon_threads`` is scoped by the names handed to it, so an
    assertion naming a thread the block never allocates is empty for the wrong
    reason --- it passes over a leak and can only ever fail spuriously, when
    something unrelated leaves a differently-named bridge alive.

    That is not hypothetical. This watch set was
    :data:`~dataknobs_common.testing.DK_SYNC_BRIDGE_THREAD`, which is
    :class:`~dataknobs_common.sync_bridge.SyncLoopBridge`'s **default** name,
    while every :class:`~dataknobs_common.sync_bridge.SyncBridgeAdapter`
    subclass is required to name its own --- ``BridgedEntityResolver`` names
    ``dk-sync-resolver``. ``DK_DAEMON_THREAD_NAMES``'s own docstring states the
    hazard in those words: *a bridge that is watched only under its default
    name is one that is not watched at all*.

    So this constructs the leak the sibling assertion exists to catch and
    requires the watch set to see it, over the same constant the sibling uses.
    A watch set that stops covering this wrapper fails here rather than going
    quiet there.
    """
    from dataknobs_common.entity_resolution import BridgedEntityResolver
    from dataknobs_common.testing import live_dk_daemon_threads

    leaked = BridgedEntityResolver(bridged["async_resolver"])
    try:
        leaked.resolve("beagles", k=5)
        assert live_dk_daemon_threads(BRIDGE_THREADS), (
            "the watch set does not cover the thread this wrapper allocates, so the "
            "assertion it scopes is empty whether or not the `with` is there"
        )
    finally:
        leaked.close()

    assert live_dk_daemon_threads(BRIDGE_THREADS) == [], (
        "and the same watch set reports the thread gone once it is closed, which is "
        "the other half of a control: a set that always answers non-empty would pass "
        "the assertion above and fail the block's"
    )


def test_the_bridge_answers_what_an_await_would(bridged: dict[str, Any]) -> None:
    """The bridge's whole contract, asserted against the resolver it wraps.

    A forwarder that answered *something* would satisfy every assertion above.
    What makes it a bridge rather than a second implementation is that the
    answer is the one the wrapped cascade gives, so that is compared directly
    -- through a loop of this test's own, which is the arrangement a
    synchronous caller does not have and the bridge exists to spare them.
    """
    import asyncio

    direct = asyncio.run(bridged["async_resolver"].resolve("beagles", k=5))
    result = bridged["result"]

    assert [candidate.entity_id for candidate in direct.candidates] == [
        candidate.entity_id for candidate in result.candidates
    ]
    assert [evidence.signal for evidence in direct.explain("beagle")] == [
        evidence.signal for evidence in result.explain("beagle")
    ]

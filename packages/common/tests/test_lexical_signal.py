"""The near-spelling rung: an entity the query did not spell, and how near it came.

One of these is the increment's acceptance criterion and is marked as such.
It asserts **both** halves -- that the three declared rungs return nothing for
a query carrying a typo, and that this one returns the entity -- because a
criterion asserting only the second would pass against a rung that matched
everything, and one asserting only the first would pass against no rung at all.

The negative half is also the half that can rot. The declared rungs returning
nothing for ``"goldne retriver"`` is a property of *today's* folds; a
normalizer that stripped vowels would make it false without touching this rung.
Asserting it directly rather than assuming it means such a change fails here
and names the rung that moved.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import pytest

from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncAliasSignal,
    CascadingResolver,
    AsyncDeclaredSignal,
    AsyncExactNormalizedSignal,
    AsyncLexicalSignal,
    AsyncScanningSignal,
    DeclaredSignal,
    EvidenceKind,
    ExactNormalizedSignal,
    LexicalSignal,
    ScanningSignal,
    Scoring,
)
from dataknobs_common.entity_resolution.registry import (
    async_signal_backends,
    signal_backends,
)
from dataknobs_common.entity_resolution.signals import (
    _RatioScorer,
    _scored_probe_spans,
    _scorer_for,
    _window_chars,
)
from dataknobs_common.exceptions import OperationError, ValidationError
from dataknobs_common.ontology import AsyncMappingEntitySource, Entity, MappingEntitySource
from dataknobs_common.testing import assert_twin_types_agree, assert_twins_agree

#: The query the criterion is written around: two typos, one of which sits
#: inside the other's form. Neither word is in the vocabulary.
TYPO = "my goldne retriver has been limping"

#: The same sentence spelled correctly, so the rung can be shown scoring 1.0
#: where it scores 0.9 above -- a rung that always scored below 1.0 would pass
#: the criterion and be wrong.
CLEAN = "my golden retriever has been limping"

#: One form contains the other, which is what lets the ordering and the
#: overlap assertions below say anything. The **id** and the **name** are
#: spelled differently on purpose: an assertion is meaningless if both sides
#: agree only by echoing the text they matched.
VOCABULARY = {
    "golden_retriever": Entity(id="golden_retriever", type="Breed", name="Golden Retriever"),
    "retriever": Entity(id="retriever", type="Breed", name="Retriever"),
}


@pytest.fixture
def entities() -> MappingEntitySource:
    return MappingEntitySource(VOCABULARY)


@pytest.fixture
def async_entities() -> AsyncMappingEntitySource:
    return AsyncMappingEntitySource(VOCABULARY)


def _found(candidates) -> list[tuple[str, str, float]]:
    """Every candidate's id, the text it matched, and what it scored.

    Flattened over **evidence**, so a candidate found at two windows appears
    twice. That is the shape the assertions here want: this rung reports every
    window that scored rather than choosing one, so a test reading only the
    candidates would not see the spans at all.
    """
    return [
        (str(candidate.entity_id), evidence.matched_text, round(evidence.score, 3))
        for candidate in candidates
        for evidence in candidate.evidence
    ]


def _proposed(candidates) -> list[tuple[str, float]]:
    """Each candidate's id and its own score, ignoring where it was found."""
    return [(str(candidate.entity_id), round(candidate.score, 3)) for candidate in candidates]


# ===== The increment's acceptance criterion =====


def test_a_near_spelling_rung_proposes_what_the_declared_rungs_cannot(entities):
    """**Criterion: a rung that proposes rather than looks up.**

    A query the vocabulary does not carry, and an entity it does. The three
    declared rungs each answer *this form is not declared*, correctly and
    uselessly; this one proposes the entity the query was reaching for, says
    how near it came, and points at the words in the **query** rather than at
    a form the query never contained.
    """
    for rung in (ExactNormalizedSignal, AliasSignal, ScanningSignal):
        assert rung(entities).candidates(TYPO, k=5) == [], (
            f"{rung.__name__} matched a query spelling none of its forms, so "
            f"the half of this criterion that makes the other half mean "
            f"something is no longer true"
        )

    candidates = LexicalSignal(entities).candidates(TYPO, k=5)
    proposed = {str(candidate.entity_id): candidate for candidate in candidates}
    assert "golden_retriever" in proposed

    candidate = proposed["golden_retriever"]
    (evidence,) = candidate.evidence
    assert evidence.kind is EvidenceKind.INFERRED
    assert evidence.scoring is Scoring.NATIVE
    assert 0.0 < evidence.score < 1.0
    assert evidence.matched_text == "goldne retriver"
    assert TYPO[evidence.span[0] : evidence.span[1]] == evidence.matched_text


def test_the_same_query_spelled_correctly_scores_one(entities):
    """The other end of the measurement, so ``below 1.0`` means something.

    A rung that returned ``0.9`` for everything would satisfy the criterion
    above. This pins that the number tracks the spelling.
    """
    assert _proposed(LexicalSignal(entities).candidates(CLEAN, k=5)) == [
        ("golden_retriever", 1.0),
        ("retriever", 1.0),
    ]


# ===== What the threshold is, measured =====


def test_the_default_threshold_returns_the_two_intended_entities_and_nothing_else(entities):
    assert _found(LexicalSignal(entities).candidates(TYPO, k=9)) == [
        ("retriever", "retriver", 0.941),
        ("golden_retriever", "goldne retriver", 0.903),
    ]


def test_a_lower_threshold_buys_windows_that_overreach(entities):
    """0.75, and why the default is not it.

    Every extra hit is a **real entity at a span that runs past it** --
    ``retriver has`` reaching into the next token. The entity is right and the
    offsets are wrong, which is the worse of the two failures: a caller
    trusting the span highlights a word the vocabulary never matched.

    The threshold is also what sets how far the probe reaches, so this is one
    effect rather than two: a lower number widens the window bound and the
    windows it buys are these.
    """
    assert _found(LexicalSignal(entities, threshold=0.75).candidates(TYPO, k=9)) == [
        ("retriever", "retriver", 0.941),
        ("retriever", "retriver has", 0.762),
        ("golden_retriever", "goldne retriver", 0.903),
        ("golden_retriever", "my goldne retriver", 0.824),
        ("golden_retriever", "goldne retriver has", 0.8),
    ]


def test_the_cutoff_that_is_right_for_configuration_keys_is_wrong_for_a_vocabulary():
    """0.60 -- ``structured_config.py``'s cutoff -- admits ``log`` as ``dog``.

    It is right *there*, where the candidates are long configuration keys and
    a caller has already misspelled one. A vocabulary carries three-letter
    forms, and any three-letter word is within one edit of many others -- so
    the same number turns an ordinary sentence naming no entity into a match.
    """
    animals = MappingEntitySource({"dog": Entity(id="dog", type="Species", name="Dog")})

    assert _found(LexicalSignal(animals, threshold=0.60).candidates("the log fell over", k=5)) == [
        ("dog", "log", 0.667)
    ]
    assert LexicalSignal(animals).candidates("the log fell over", k=5) == []


# ===== The probe's bound =====


def test_the_bound_is_characters_because_a_token_bound_loses_a_whole_entity():
    """The typo class *a space where the vocabulary has none*.

    A form declared as one token, spelled by the caller as two. Bounding the
    probe at :meth:`EntitySource.longest_form_tokens` -- which is exactly what
    the scanning rung does, correctly, for an exact lookup -- would cap the
    window at one token here, and the best single token scores 0.69. The
    entity is not found at a worse span; it is not found.
    """
    source = MappingEntitySource(
        {
            "labradorretriever": Entity(
                id="labradorretriever", type="Breed", name="LabradorRetriever"
            )
        }
    )
    assert source.longest_form_tokens() == 1, (
        "this fixture only says anything while every form it declares is one "
        "token -- the id and the name fold together here, which is what keeps "
        "it so"
    )

    found = _found(LexicalSignal(source).candidates("my labrador retriever is limping", k=5))

    assert found[0] == ("labradorretriever", "labrador retriever", 0.971)
    assert all(len(text.split()) > 1 for _id, text, _score in found), (
        "every window that matched spans more than one token, which is the "
        "whole finding: a probe capped at the vocabulary's own token count "
        "would have produced none of them"
    )


def test_no_window_past_the_bound_could_have_cleared_its_threshold():
    """The claim that makes the bound arithmetic rather than a budget.

    :meth:`difflib.SequenceMatcher.ratio` is ``2M/T``, and ``M`` cannot exceed
    the shorter string -- so a window longer than ``|F|(2-t)/t`` scores below
    ``t`` against a form of length ``|F|``, whatever the two strings are. Every
    probe the bound removes would have answered *no*.

    Pairs are drawn to include the shape that would break it if anything did:
    a form with padding on one side or the other, which is what an overreaching
    window actually is. Random rather than hand-picked, because the cases that
    would falsify this are exactly the ones nobody thinks to write down.
    """
    rng = random.Random(20260917)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    for threshold in (0.90, 0.85, 0.75, 0.60):
        for _ in range(2000):
            form = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 20)))
            pad = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20)))
            window = rng.choice([form + pad, pad + form, pad])
            bound = _window_chars([form], threshold)
            if len(window) <= bound:
                continue
            score = SequenceMatcher(None, window, form).ratio()
            assert score < threshold, (
                f"{window!r} is longer than the bound {bound} for {form!r} at "
                f"threshold {threshold}, and scores {score:.4f} -- so the probe "
                f"stops before a window that would have matched"
            )


def test_the_bound_rounds_up_because_the_equality_case_is_a_real_match():
    """A 17-character form at 0.85 admits a 23-character window, scoring exactly 0.85.

    ``17 * 1.15 / 0.85`` is ``22.999999999999996`` in binary floating point, so
    a bound that truncated would stop at 22 and lose the equality case. One
    character of generosity costs a handful of probes; one character short
    costs an answer.
    """
    assert _window_chars(["a" * 17], 0.85) == 23
    assert SequenceMatcher(None, "b" * 6 + "a" * 17, "a" * 17).ratio() == pytest.approx(0.85)


def test_the_probe_stops_widening_rather_than_filtering():
    """Spans grow from each start and the loop breaks at the first one too wide.

    Which is what keeps the enumeration linear in the query: filtering would
    still have built every one of the *n(n+1)/2* spans before discarding them.
    """
    query = "alpha beta gamma delta"
    assert [span for span, _window in _scored_probe_spans(query, 10, str)] == [
        (0, 5),
        (0, 10),
        (6, 10),
        (6, 16),
        (11, 16),
        (17, 22),
    ]


# ===== The default scorer =====


def test_the_default_scorer_agrees_with_a_plain_ratio_wherever_it_matters():
    """It prefilters and it caches, and neither may move a score at the threshold.

    The default refuses a pair the cheap upper bounds already rule out, and
    answers ``0.0`` there rather than the true ratio. That is invisible to the
    rung, which keeps only what clears the threshold -- so what has to hold is
    that **at or above** the threshold it is exactly
    :meth:`difflib.SequenceMatcher.ratio`, and below it never claims to clear.

    It also reuses one matcher across calls, so this runs many pairs through
    **one** scorer object: a cache that answered the previous form's question
    would show up here and nowhere else.
    """
    rng = random.Random(4242)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    threshold = 0.85
    scorer = _RatioScorer(threshold)
    for _ in range(4000):
        form = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 20)))
        window = (
            form
            if rng.random() < 0.3
            else form[: rng.randint(0, len(form))]
            + "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
        )
        plain = SequenceMatcher(None, window, form).ratio()
        answered = scorer(window, form)
        if plain >= threshold:
            assert answered == plain, (
                f"the default scorer answered {answered} where a plain ratio "
                f"answers {plain} for {window!r} against {form!r}"
            )
        else:
            assert answered < threshold


def test_the_cheap_bounds_the_default_scorer_skips_on_are_upper_bounds():
    """The property the prefilter rests on, asserted rather than cited.

    :meth:`~difflib.SequenceMatcher.quick_ratio` and
    :meth:`~difflib.SequenceMatcher.real_quick_ratio` are documented as upper
    bounds on ``ratio()``. The default scorer skips any pair either of them
    puts below the threshold, so if that documentation were ever wrong the
    rung would silently stop finding matches -- which is the failure mode this
    file exists to make loud.
    """
    rng = random.Random(909)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    for _ in range(4000):
        left = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 26)))
        right = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 26)))
        if rng.random() < 0.4:
            right = left[: rng.randint(0, len(left))] + right
        matcher = SequenceMatcher(None, left, right)
        exact = matcher.ratio()
        assert matcher.quick_ratio() >= exact
        assert matcher.real_quick_ratio() >= exact


# ===== The scorer seam =====


def test_a_scorer_can_be_injected(entities):
    """The seam that is this rung's answer to a vocabulary too large for difflib.

    Asserted with a scorer that answers differently rather than faster, since
    *it was called* is the property and speed is not testable here.
    """
    calls: list[tuple[str, str]] = []

    def always(window: str, form: str) -> float:
        calls.append((window, form))
        return 1.0

    candidates = LexicalSignal(entities, scorer=always).candidates("x", k=9)

    assert calls, "the injected scorer was never called"
    assert {str(candidate.entity_id) for candidate in candidates} == {
        "golden_retriever",
        "retriever",
    }


def test_a_substring_scorer_picks_the_wrong_entity(entities):
    """Why ``partial_ratio`` is named in the docstring as the one not to pass.

    It scores any substring 1.0, so over a vocabulary whose forms contain one
    another every query containing the shorter form matches the longer one
    just as well -- and the discrimination this rung exists to have is exactly
    what buys its "it can locate" property.

    Written inline rather than imported: the claim is about the *shape* of
    such a scorer, and asserting it needs no dependency.
    """

    def substring(window: str, form: str) -> float:
        return 1.0 if window in form or form in window else 0.0

    found = _found(LexicalSignal(entities, scorer=substring).candidates("gold retriever", k=9))

    assert ("golden_retriever", "retriever", 1.0) in found, (
        "a substring scorer proposes the containing entity for a query that "
        "names the contained one, at the same score as the right answer"
    )


# ===== The fold =====


def test_both_sides_of_the_comparison_are_folded(entities):
    """A capital letter is not a misspelling.

    This rung compares strings itself instead of handing them to the index, so
    it performs the fold the index would have performed -- which is why its
    ``normalizer`` defaults to one where every other rung here defaults to
    none. Without it, ``Golden Retriever`` against the folded form would lose
    a character's worth of score for each capital.
    """
    assert _proposed(LexicalSignal(entities).candidates("My GOLDEN RETRIEVER", k=5)) == [
        ("golden_retriever", 1.0),
        ("retriever", 1.0),
    ]


def test_a_supplied_normalizer_folds_both_sides(entities):
    """A caller whose source folds differently passes theirs, and it reaches both.

    The identity fold makes the case difference visible again, which is the
    only way to show that the parameter is doing anything.
    """
    unfolded = LexicalSignal(entities, normalizer=lambda form: form)

    assert _found(unfolded.candidates("GOLDEN RETRIEVER", k=5)) == []
    assert _found(unfolded.candidates("golden retriever", k=5)) == [
        ("golden_retriever", "golden retriever", 1.0),
        ("retriever", "retriever", 1.0),
    ]


# ===== What the rung reports =====


def test_one_entity_at_one_span_is_one_piece_of_evidence(entities):
    """An id, a name and an alias all fold into the catalogue separately.

    So a window routinely clears the threshold against several forms of the
    same entity -- here ``golden_retriever`` the id and ``golden retriever``
    the name. Reporting both would say the vocabulary was found twice where it
    was found once, so the best score wins and the rest are the same finding
    measured against a worse spelling.
    """
    (candidate,) = [
        candidate
        for candidate in LexicalSignal(entities).candidates("golden retriever", k=5)
        if candidate.entity_id == "golden_retriever"
    ]

    assert len(candidate.evidence) == 1
    assert candidate.evidence[0].score == 1.0


def test_a_window_wider_than_the_form_is_reported_rather_than_resolved(entities):
    """Overlap is visible in the offsets; nothing here chooses between spans.

    A window one token wider than a form it contains can still clear the
    threshold -- ``my golden retriever`` against ``golden retriever`` is
    ``0.914`` -- so the exact window and the overreaching one are both
    reported, best first. Choosing one would be a verdict, and this family
    does not make verdicts: it is the same answer
    :class:`~dataknobs_common.entity_resolution.FormHit` gives for two
    declared forms at overlapping spans, and the reason an overlap policy is
    nobody's yet.

    The candidate's own score is the exact window's, so a caller who wants
    one number is not shown the overreach at all.
    """
    at_spans = [
        (text, score)
        for entity_id, text, score in _found(LexicalSignal(entities).candidates(CLEAN, k=9))
        if entity_id == "golden_retriever"
    ]

    assert at_spans == [
        ("golden retriever", 1.0),
        ("my golden retriever", 0.914),
        ("golden retriever has", 0.889),
    ]


def test_the_best_score_is_proposed_first(entities):
    """The order :meth:`DeclaredSignal._order` says the rung has to publish.

    A declared score is 1.0 by fiat and carries no order; a measured one does,
    and the cascade positions by arrival -- so this is what decides which
    near-spelling a caller sees first.
    """
    scores = [score for _id, _text, score in _found(LexicalSignal(entities).candidates(TYPO, k=9))]

    assert scores == sorted(scores, reverse=True)


def test_the_candidates_own_score_is_its_best_hit(entities):
    """An entity found twice is not penalised for having been found twice."""
    (candidate,) = [
        candidate
        for candidate in LexicalSignal(entities, threshold=0.75).candidates(TYPO, k=9)
        if candidate.entity_id == "retriever"
    ]

    assert len(candidate.evidence) > 1
    assert candidate.score == max(evidence.score for evidence in candidate.evidence)


def test_the_rung_narrows(entities):
    """It reads an entity source, so a filter can be honoured against it."""
    assert LexicalSignal(entities).narrows() is True
    assert AsyncLexicalSignal(AsyncMappingEntitySource(VOCABULARY)).narrows() is True


def test_a_filter_narrows_the_rungs_own_answer(entities):
    """Rung-side narrowing comes from the base, and reaches this rung unchanged."""
    assert LexicalSignal(entities).candidates(TYPO, k=5, filter="Breed")
    assert LexicalSignal(entities).candidates(TYPO, k=5, filter="Species") == []


# ===== Refusals =====


def test_a_source_that_does_not_publish_its_forms_is_refused_at_construction():
    """Not an empty rung, which is what :class:`AliasSignal` correctly answers.

    A vocabulary may declare no aliases; none has no forms. So an empty answer
    from this rung means *nothing was near enough*, and a source that cannot
    be asked has to fail where the mistake was made rather than look like a
    query that matched nothing.
    """

    class Bare:
        def by_surface_form(self, form: str) -> frozenset[str]:
            return frozenset()

    with pytest.raises(ValidationError, match="does not publish its surface forms"):
        LexicalSignal(Bare())
    with pytest.raises(ValidationError, match="surface_forms"):
        AsyncLexicalSignal(Bare())


@pytest.mark.parametrize("threshold", [0.0, -0.5, 1.5])
def test_a_threshold_outside_the_unit_interval_is_refused(entities, threshold):
    """Zero admits the whole vocabulary for any query; above one admits nothing."""
    with pytest.raises(ValidationError, match="threshold must be"):
        LexicalSignal(entities, threshold=threshold)


def test_a_threshold_of_exactly_one_is_a_real_request(entities):
    """*Only an exact fold* -- which is a rung a caller may genuinely want."""
    rung = LexicalSignal(entities, threshold=1.0)

    assert _found(rung.candidates(CLEAN, k=5)) == [
        ("golden_retriever", "golden retriever", 1.0),
        ("retriever", "retriever", 1.0),
    ]
    assert rung.candidates(TYPO, k=5) == []


def test_a_vocabulary_declaring_nothing_answers_nothing_rather_than_dividing():
    """The empty catalogue reaches :func:`_window_chars`, whose ``max`` has a default."""
    assert LexicalSignal(MappingEntitySource({})).candidates(TYPO, k=5) == []


# ===== Both flavours =====


@pytest.mark.asyncio
async def test_the_twin_answers_identically(async_entities, entities):
    """Same vocabulary, same query, same candidates -- ids, spans and scores."""
    assert _found(await AsyncLexicalSignal(async_entities).candidates(TYPO, k=9)) == _found(
        LexicalSignal(entities).candidates(TYPO, k=9)
    )


def test_the_twins_agree_on_their_surface():
    """``name`` is excluded: a property has no signature to compare, and what
    it answers is asserted directly below.

    ``__init__`` **is** included, and was the gap: it is where every one of
    this rung's four knobs is named and defaulted, and a check over the query
    members alone would pass a twin whose ``threshold`` defaulted to a
    different number or whose ``max_query_tokens`` it never grew. It is
    unflavoured -- a constructor is a plain ``def`` on both sides -- and its
    ``entities`` parameter is the one that is flavour-typed by design.
    """
    assert_twin_types_agree(
        LexicalSignal,
        AsyncLexicalSignal,
        members=["candidates", "candidates_many", "narrows"],
        unflavoured_members=["narrows"],
    )
    assert_twins_agree(
        LexicalSignal.__init__,
        AsyncLexicalSignal.__init__,
        unflavoured=True,
        flavour_typed=["entities"],
        label="LexicalSignal/AsyncLexicalSignal.__init__",
    )


def test_the_two_bases_carry_the_same_scoring_declarations():
    """The parity obligation the widening **creates**, and the one a twin check
    over the rungs could not see.

    :attr:`kind` and :attr:`scoring` are declared on each base separately --
    the two share no superclass by design -- so a widening applied to one
    flavour and not the other is invisible from the rungs: both twins would
    simply be wrong in the same way.
    """
    assert (DeclaredSignal.kind, DeclaredSignal.scoring) == (
        AsyncDeclaredSignal.kind,
        AsyncDeclaredSignal.scoring,
    )
    for sync_rung, async_rung in [
        (ExactNormalizedSignal, AsyncExactNormalizedSignal),
        (AliasSignal, AsyncAliasSignal),
        (ScanningSignal, AsyncScanningSignal),
        (LexicalSignal, AsyncLexicalSignal),
    ]:
        assert (sync_rung.kind, sync_rung.scoring) == (async_rung.kind, async_rung.scoring), (
            f"{sync_rung.__name__} and {async_rung.__name__} declare different "
            f"evidence, so one flavour of the same rung reports a kind the "
            f"other does not"
        )


def test_the_rung_is_named_for_the_key_it_registers_under(entities, async_entities):
    assert LexicalSignal(entities).name == "lexical"
    assert AsyncLexicalSignal(async_entities).name == "lexical"


@pytest.mark.asyncio
async def test_both_registries_build_it(entities, async_entities):
    """And the factory forwards the two arguments a document is the only way to set.

    **Each flavour gets a source of its own flavour**, and each built rung is
    then *asked a query*. Building one and asserting its type is what this
    test used to do, and it passed while holding a rung whose every query
    would raise: ``isinstance`` is the one thing a wrong-flavour source does
    not fail. The ``await`` is the assertion.
    """
    built = signal_backends.create("lexical", {"entities": entities, "threshold": 1.0})
    built_async = async_signal_backends.create("lexical", {"entities": async_entities})

    assert isinstance(built, LexicalSignal)
    assert isinstance(built_async, AsyncLexicalSignal)
    assert built.candidates(TYPO, k=5) == [], "the threshold the config named was dropped"
    assert _found(await built_async.candidates(TYPO, k=9)) == _found(
        LexicalSignal(entities).candidates(TYPO, k=9)
    ), "the rung the asynchronous factory built cannot answer a query"


# ===== What a review found: the probe's units, the loop, and the flavours =====


def test_a_fold_that_deletes_characters_does_not_shorten_the_probe():
    """The bound is derived in folded characters and must be spent in them.

    :func:`_window_chars` measures the **folded** forms, and the comparison
    scores a folded window against a folded form -- so a probe that decides
    how far to widen by counting the *raw* slice is mixing two units. With
    the default fold that is invisible, because ``strip`` is a no-op inside a
    token span and ``casefold`` never shortens, so the raw length is a lower
    bound on the folded one and the enumeration is merely conservative.

    A ``normalizer`` that **deletes** characters inverts that, and one is an
    ordinary thing for a consumer to write: the fold here removes spaces, so
    the query spells the declared form exactly and scores ``1.0`` -- at a raw
    width of 19 against a bound of 14, which a raw-counting probe never
    reaches. The entity is not found at a worse span. It is not found.
    """
    source = MappingEntitySource(
        {"abcdefghij": Entity(id="abcdefghij", type="Breed", name="abcdefghij")}
    )
    spaces_are_nothing = LexicalSignal(
        source, normalizer=lambda form: form.replace(" ", "").casefold()
    )

    assert _window_chars(["abcdefghij"], 0.85) == 14, (
        "the fixture only says anything while the bound is narrower than the "
        "raw query below -- otherwise the probe reaches it for the wrong reason"
    )
    found = _found(spaces_are_nothing.candidates("a b c d e f g h i j", k=5))

    assert ("abcdefghij", "a b c d e f g h i j", 1.0) in found, (
        "the query folds to exactly the declared form, so the widest window "
        "scores 1.0 -- and it is 19 raw characters against a bound of 14, "
        "which is the window a probe counting raw characters never reaches"
    )
    assert found[0] == ("abcdefghij", "a b c d e f g h i j", 1.0), (
        "and it is the best of them, so it is proposed first"
    )


def test_a_probe_span_is_admitted_on_what_the_fold_leaves():
    """The enumerator's own account of the above, one layer down.

    Four tokens of one character each, separated by spaces a fold deletes.
    Measured after the fold, three characters of budget reach three tokens;
    measured before it, the same budget reaches two -- and the difference is
    every window the rung would have been asked about and was not.
    """

    def deletes_spaces(form: str) -> str:
        return form.replace(" ", "")

    folded = list(_scored_probe_spans("a b c d", 3, deletes_spaces))

    assert folded == [
        ((0, 1), "a"),
        ((0, 3), "ab"),
        ((0, 5), "abc"),
        ((2, 3), "b"),
        ((2, 5), "bc"),
        ((2, 7), "bcd"),
        ((4, 5), "c"),
        ((4, 7), "cd"),
        ((6, 7), "d"),
    ]
    raw = [span for span, _window in _scored_probe_spans("a b c d", 3, str)]
    assert (0, 5) not in raw and (0, 5) in [span for span, _ in folded], (
        "the widest window from the first token is five raw characters and "
        "three folded ones, so which unit the probe counts in decides whether "
        "it is ever scored"
    )


@pytest.mark.asyncio
async def test_the_scoring_loop_runs_off_the_event_loop(async_entities):
    """A per-query scan of the vocabulary is CPU, and CPU on the loop is a stall.

    The guide measures this rung at ~315 ms per query over 10,000 entities.
    Held on the event loop that freezes every other task on it for as long --
    the harm ``async-transport.md`` names, arriving through arithmetic rather
    than through a syscall, so neither the ``ASYNC2xx`` lint nor
    ``assert_no_blocking`` can see it. The proof has to be structural: the
    scorer records which thread called it, and it must not be the loop's.
    """
    ran_on: set[int] = set()

    def scorer(window: str, form: str) -> float:
        ran_on.add(threading.get_ident())
        return SequenceMatcher(None, window, form).ratio()

    rung = AsyncLexicalSignal(async_entities, scorer=scorer)
    await rung.candidates(TYPO, k=5)

    assert ran_on, "the scorer was never called, so this asserts nothing"
    assert threading.get_ident() not in ran_on, (
        "the scoring loop ran on the event loop's own thread, so every other "
        "task on that loop was stalled for the length of the scan"
    )


def test_a_source_of_the_wrong_flavour_is_refused_at_construction(entities, async_entities):
    """The refusal this rung advertises, against the mistake config actually makes.

    ``isinstance`` against a runtime-checkable protocol compares member
    *names*, and both catalogues spell it ``surface_forms`` -- so a check that
    asks only *does this publish its forms* accepts either flavour, and the
    rung constructs cleanly around a source it can never await correctly. A
    missing member is the mistake a reader makes; the wrong flavour is the one
    a document makes, because ``entities:`` is resolved before either registry
    sees it.
    """
    with pytest.raises(ValidationError, match="synchronous"):
        LexicalSignal(async_entities)
    with pytest.raises(ValidationError, match="asynchronous"):
        AsyncLexicalSignal(entities)


def test_the_async_registry_refuses_a_synchronous_source(entities):
    """The same mistake by the route that actually makes it.

    A document names ``kind: lexical`` and the flavour of ``entities:`` is
    resolved elsewhere, so nothing between the two registries stops a
    synchronous source reaching the asynchronous factory. The registry wraps
    what a factory raises, so the refusal arrives as the cause -- which is
    what a caller reading a traceback gets, and it is the sentence that says
    what to change.
    """
    with pytest.raises(OperationError) as raised:
        async_signal_backends.create("lexical", {"entities": entities})

    assert isinstance(raised.value.__cause__, ValidationError)
    assert "asynchronous" in str(raised.value.__cause__)


@pytest.mark.parametrize("parameter", ["scorer", "normalizer"])
def test_a_value_that_is_not_callable_is_refused_at_construction(entities, parameter):
    """A document cannot write a callable, and a string is truthy.

    ``scorer`` and ``normalizer`` are both forwarded straight from the config
    dict, so ``scorer: "rapidfuzz.fuzz.ratio"`` -- the obvious thing to write,
    and the one the registry's own comment invites by naming the parameter as
    the reason for forwarding -- passes ``scorer or _RatioScorer(...)`` and
    fails at the first query, once per ``(window, form)`` pair, from inside
    the scan.
    """
    with pytest.raises(OperationError) as raised:
        signal_backends.create("lexical", {"entities": entities, parameter: "fuzz.ratio"})

    assert isinstance(raised.value.__cause__, ValidationError)
    assert parameter in str(raised.value.__cause__)


@pytest.mark.parametrize("kind", ["exact", "alias", "scan"])
def test_a_normalizer_that_is_not_callable_is_refused_on_every_rung(entities, kind):
    """The same defect one layer up, where it was already reachable.

    ``normalizer`` is forwarded from the document by every factory here, so
    the refusal belongs on the shared base rather than beside the rung that
    happened to surface it.
    """
    with pytest.raises(OperationError) as raised:
        signal_backends.create(kind, {"entities": entities, "normalizer": "casefold"})

    assert isinstance(raised.value.__cause__, ValidationError)
    assert "normalizer" in str(raised.value.__cause__)


def test_the_default_scorer_is_built_per_query_and_an_injected_one_is_not():
    """The seam that keeps :class:`_RatioScorer`'s cache safe to have.

    A fresh default per query is the whole fix for sharing: the object is
    stateful, and one owner for the length of one scan is what its cache
    needs. A scorer the caller supplied is passed through untouched, because
    nothing here could copy a module-level function and whether theirs is
    shareable is theirs to know.
    """

    def supplied(window: str, form: str) -> float:
        return 1.0

    assert _scorer_for(None, 0.85) is not _scorer_for(None, 0.85)
    assert isinstance(_scorer_for(None, 0.85), _RatioScorer)
    assert _scorer_for(supplied, 0.85) is supplied


def test_a_matcher_whose_form_is_cached_is_why_it_cannot_be_shared():
    """The hazard the seam above exists for, characterized rather than fixed.

    :class:`_RatioScorer` skips ``set_seq2`` when the form it is handed equals
    the one it last saw, which is its second optimisation and is correct for a
    single caller. What it makes is an object whose bookkeeping and whose
    matcher must stay in step, with nothing but sole ownership keeping them
    there: a second caller moving ``set_seq2`` between this one's cache check
    and its ``ratio()`` leaves the first scoring against a form it never
    asked about -- a wrong number, not a crash.

    Reproduced by moving the matcher directly, because a thread cannot be
    scheduled into that window on demand. **This asserts the hazard, not a
    defect**: it is what makes the per-query construction above load-bearing,
    so a future scorer that stopped caching should delete this test rather
    than satisfy it.
    """
    scorer = _RatioScorer(0.0)
    assert scorer("golden retriver", "golden retriever") == pytest.approx(0.968, abs=0.001)

    scorer._matcher.set_seq2("dog")  # what another caller's turn does

    assert scorer("golden retriver", "golden retriever") != pytest.approx(0.968, abs=0.001), (
        "the cached form no longer decides what the matcher holds, so sharing "
        "one of these between callers would be safe and the per-query "
        "construction it forces is dead weight"
    )


def test_two_threads_asking_one_rung_get_the_same_answers_as_one():
    """The rung is an ordinary thing to build once and serve from a pool."""
    vocabulary = {
        f"breed_{index:03d}": Entity(
            id=f"breed_{index:03d}", type="Breed", name=f"Retriever Number {index:03d}"
        )
        for index in range(60)
    }
    rung = LexicalSignal(MappingEntitySource(vocabulary), threshold=0.75)
    queries = ["retreiver number 017 is limping", "my retriver numbr 042 has been limping"]
    expected = [_found(rung.candidates(query, k=9)) for query in queries]

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda q: _found(rung.candidates(q, k=9)), queries * 24))

    assert answers == expected * 24


# ===== In a cascade: the composition the docstrings recommend =====


@pytest.fixture
def declared_rungs(entities):
    """The three rungs this one is documented as going *after*."""
    return [ExactNormalizedSignal(entities), AliasSignal(entities), ScanningSignal(entities)]


def test_a_declared_hit_is_positioned_ahead_of_a_near_spelling_one(entities, declared_rungs):
    """The claim :class:`LexicalSignal`'s docstring makes about its own placement.

    That docstring says a near-spelling hit "still sits behind a declared hit,
    and not because 0.94 is smaller: a cascade positions by the first rung
    that produced an id" -- and until this test nothing composed the rung into
    a cascade at all, so the sentence was prose with no assertion under it.

    The clean query is the one that can distinguish the two readings: both
    rungs score the same entities at ``1.0`` there, so an order by *number*
    would be a coin toss and an order by *arrival* is the scan's.
    """
    cascade = CascadingResolver([*declared_rungs, LexicalSignal(entities)], entities)

    result = cascade.resolve(CLEAN, k=5)

    assert [str(candidate.entity_id) for candidate in result.ranked()] == [
        "golden_retriever",
        "retriever",
    ]
    assert next(evidence.signal for evidence in result.explain("golden_retriever")) == "scan", (
        "the declared rung produced the id first, so its evidence leads -- "
        "which is what 'the cascade positions by arrival' means"
    )
    assert result.ranked()[0].declared, "a declared hit is still declared in this composition"


def test_the_typo_query_resolves_only_because_the_near_spelling_rung_is_there(
    entities, declared_rungs
):
    """The acceptance criterion's claim, this time through a real cascade."""
    declared_only = CascadingResolver(declared_rungs, entities)
    with_lexical = CascadingResolver([*declared_rungs, LexicalSignal(entities)], entities)

    assert declared_only.resolve(TYPO, k=5).ranked() == ()

    candidates = with_lexical.resolve(TYPO, k=5).ranked()
    assert [str(candidate.entity_id) for candidate in candidates] == [
        "retriever",
        "golden_retriever",
    ]
    assert not any(candidate.declared for candidate in candidates), (
        "nothing here was declared -- the vocabulary carries neither word the "
        "query spelled, which is what INFERRED says"
    )


def test_a_native_score_keeps_the_result_out_of_distribution_arithmetic(entities, declared_rungs):
    """``Scoring.NATIVE`` is the half of the rung's identity that stops a caller.

    A scorer-defined number means whatever that scorer means, so two rungs
    with different scorers produce incomparable ``0.8``s. ``as_distribution``
    admits ``NORMALIZED`` alone, and this is the first rung in the package
    that can put anything else in a result.
    """
    cascade = CascadingResolver([*declared_rungs, LexicalSignal(entities)], entities)

    assert cascade.resolve(TYPO, k=5).as_distribution() is None


def test_an_inferred_hit_does_not_cover_the_words_it_read(entities, declared_rungs):
    """**Coverage counts declared spans, and this rung is why it says so.**

    ``Coverage`` is the account of *what the vocabulary accounted for*, and
    the phrase it is read for is the residue: the phrases a corpus's users
    ask about and the vocabulary does not carry are the next entries somebody
    should add. A near-spelling proposal is the opposite of an entry that
    exists -- it is the rung saying the vocabulary carries **nothing** the
    query spelled -- so counting the words it read as covered would delete
    exactly the line the field is maintained for.

    So adding this rung to a cascade leaves coverage alone. The typo sentence
    reads the same with it and without it, and the entities it proposes are
    in ``ranked()`` where a caller who wants them looks.
    """
    declared_only = CascadingResolver(declared_rungs, entities)
    with_lexical = CascadingResolver([*declared_rungs, LexicalSignal(entities)], entities)

    assert declared_only.resolve(TYPO, k=5).unmatched_text() == (TYPO,)

    result = with_lexical.resolve(TYPO, k=5)
    assert result.matched_text() == ()
    assert result.unmatched_text() == (TYPO,), (
        "the vocabulary still carries nothing this query spelled, which is "
        "the line a consumer maintaining one acts on"
    )
    assert [str(candidate.entity_id) for candidate in result.ranked()] == [
        "retriever",
        "golden_retriever",
    ], "and the proposals are still there, which is what ranked() is for"


def test_an_overreaching_window_does_not_widen_coverage(entities, declared_rungs):
    """The reading that makes the rule worth having, on a query with no typo.

    The rung reports every window that cleared the threshold rather than
    choosing one -- its documented policy, and right for evidence, since
    containment stays visible in the offsets. At the default threshold a
    window padded by a neighbouring word still clears it, because the padding
    is small against the form: ``my golden retriever`` at ``0.914`` and
    ``golden retriever has`` at ``0.889``.

    Merged positionally those would carry ``my`` and ``has`` into ``matched``,
    and neither is a word the vocabulary matched. Counting declared spans
    alone means the scan decides the extent and the measured rung cannot
    widen it -- so adding this rung changes what a caller is *told*, and
    never what is *covered*.
    """
    declared_only = CascadingResolver(declared_rungs, entities)
    with_lexical = CascadingResolver([*declared_rungs, LexicalSignal(entities)], entities)

    result = with_lexical.resolve(CLEAN, k=5)
    spans = {
        (evidence.signal, evidence.matched_text)
        for candidate in result.ranked()
        for evidence in result.explain(candidate.entity_id)
    }
    assert ("lexical", "my golden retriever") in spans, (
        "the overreaching evidence is still reported -- it is the coverage "
        "reading that changed, not what the rung found"
    )
    assert ("lexical", "golden retriever has") in spans

    assert result.matched_text() == ("golden retriever",)
    assert result.unmatched_text() == ("my", "has been limping")
    assert result.coverage == declared_only.resolve(CLEAN, k=5).coverage, (
        "identical to the cascade without this rung, which is the property: "
        "a rung that measures adds candidates and never coverage"
    )


# ===== The cost a caller's own text can impose =====


def test_a_query_longer_than_the_cap_is_refused_rather_than_truncated(entities):
    """``max_query_tokens``: the knob this rung was missing.

    The character bound caps how *wide* a window may be. It says nothing
    about how *many* there are, and that count is the query's token count --
    the caller's input, not the vocabulary's. Each window costs one scorer
    call per declared form, where a scan's costs one dictionary lookup, so
    the same paste that :func:`_probe_spans` bounds for a scan is a far
    larger bill here. Measured: linear in the token count, and a
    nine-hundred-token paste over a five-hundred-entity vocabulary is two
    seconds of one CPU.

    **Refused rather than truncated**, which is the choice this family makes
    everywhere it has one: probing the first *n* tokens of a paste and
    answering from them is a plausible-looking answer to a question nobody
    asked, and a caller who set this number wants to hear about the input
    rather than to be quietly served less of it.
    """
    rung = LexicalSignal(entities, max_query_tokens=6)

    assert _found(rung.candidates(TYPO, k=9)) == _found(
        LexicalSignal(entities).candidates(TYPO, k=9)
    ), "six tokens is exactly this query, so the cap changes nothing about it"

    with pytest.raises(ValidationError, match="max_query_tokens"):
        rung.candidates(TYPO + " and also quite sad", k=9)


def test_the_cap_is_off_by_default(entities):
    """It can cost an answer, so nothing gets it without asking."""
    assert LexicalSignal(entities)._max_query_tokens is None
    assert _found(LexicalSignal(entities).candidates(TYPO + " " * 0 + " and more words here", k=9))


@pytest.mark.parametrize("max_query_tokens", [0, -1])
def test_a_cap_below_one_is_refused(entities, max_query_tokens):
    """A rung that probes nothing, which is the silence this family refuses."""
    with pytest.raises(ValidationError, match="max_query_tokens"):
        LexicalSignal(entities, max_query_tokens=max_query_tokens)


@pytest.mark.asyncio
async def test_the_twin_caps_identically(async_entities, entities):
    """The refusal is shared, so the flavours cannot disagree about the limit."""
    with pytest.raises(ValidationError, match="max_query_tokens"):
        await AsyncLexicalSignal(async_entities, max_query_tokens=2).candidates(TYPO, k=9)
    with pytest.raises(ValidationError, match="max_query_tokens"):
        LexicalSignal(entities, max_query_tokens=2).candidates(TYPO, k=9)


def test_the_cap_reaches_the_rung_through_the_registry(entities):
    """A document is where a caller who cannot reach the constructor sets it."""
    rung = signal_backends.create("lexical", {"entities": entities, "max_query_tokens": 2})

    with pytest.raises(ValidationError, match="max_query_tokens"):
        rung.candidates(TYPO, k=9)

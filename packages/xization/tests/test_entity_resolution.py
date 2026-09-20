"""The authority stack as a resolution rung, and how it differs from the default.

Two of the tests here are the increment's acceptance criteria and are marked
as such. They assert the relationship in **both** directions, because it runs
both ways: the stack reaches declared forms the ``dataknobs_common`` default
cannot, and suppresses overlap the default keeps. A criterion naming only the
first would call this an upgrade and be half right, which is worse than being
wrong about it -- a documented difference nothing executes is how the
documentation goes quietly out of date.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

import dataknobs_common.exceptions as dk_exc
import dataknobs_xization.authorities as dk_auth
import dataknobs_xization.lexicon as dk_lex
from dataknobs_common.entity_resolution import EvidenceKind, ScanningSignal, Scoring
from dataknobs_common.entity_resolution.registry import (
    async_signal_backends,
    signal_backends,
)
from dataknobs_common.ontology import Entity, MappingEntitySource
from dataknobs_common.testing import assert_twin_types_agree
import dataknobs_xization.entity_resolution as dk_er
from dataknobs_xization.entity_resolution import AsyncAuthoritySignal, AuthoritySignal

QUERY = "my golden retriever has been limping"

#: One vocabulary, two entities, and one of their forms contains the other --
#: which is what makes the query able to say anything about overlap at all.
#: The **id** and the **surface form** are deliberately spelled differently,
#: because the whole comparison below is meaningless if the two sides only
#: agree by both echoing the text they matched.
VOCABULARY = {"golden_retriever": "Golden Retriever", "retriever": "Retriever"}


@pytest.fixture
def entities() -> MappingEntitySource:
    """The ``dataknobs_common`` side: a vocabulary the default rungs can read."""
    return MappingEntitySource(
        {
            entity_id: Entity(id=entity_id, type="Breed", name=name)
            for entity_id, name in VOCABULARY.items()
        }
    )


@pytest.fixture
def one_axis() -> dk_lex.DataframeAuthority:
    """The stack side, loaded the way a vocabulary normally is: one axis, one authority.

    **The frame is indexed by the entity id**, which is not decoration: an
    authority's value id *is* its frame's index -- ``AuthorityData.lookup_values``
    reads ``df.index`` for one -- and that id is what this rung answers with.
    A frame left on its default ``RangeIndex`` resolves every query to a row
    number, which is a plausible-looking id and never the right one.
    """
    return dk_lex.DataframeAuthority(
        "animal",
        # `str.lower` as the normalizer, because the frame carries display
        # spellings and the query does not: an identity fold would match only
        # a consumer who typed their vocabulary's capitalization back at it.
        dk_lex.LexicalExpander(None, str.lower),
        dk_auth.AuthorityData(
            pd.DataFrame({"animal": list(VOCABULARY.values())}, index=list(VOCABULARY)),
            "animal",
        ),
    )


@pytest.fixture
def one_per_form() -> dk_auth.AuthoritiesBundle:
    """The stack side, loaded one authority per form rather than per axis.

    A regex authority's ``canonical_fn`` computes the value id from the
    matched text, which is the arm's equivalent of the frame index above.
    """
    return dk_auth.AuthoritiesBundle(
        "animals",
        auths=[
            dk_auth.RegexAuthority(
                entity_id,
                re.compile(re.escape(name), re.IGNORECASE),
                lambda _text, _group, entity_id=entity_id: entity_id,
            )
            for entity_id, name in VOCABULARY.items()
        ],
    )


def _spans(candidates) -> list[tuple[str, tuple[int, int]]]:
    """Every candidate's id paired with each span it was found at."""
    return [
        (str(candidate.entity_id), evidence.span)
        for candidate in candidates
        for evidence in candidate.evidence
    ]


# ===== The increment's acceptance criteria =====


def test_a_declared_pattern_resolves_where_no_enumeration_could(entities):
    """**Criterion: the upgrade direction.**

    A vocabulary *describes* a chip number rather than listing every one, so
    no entity source can carry a surface form for it and the default rung
    matches nothing. This is the capability the injected stack exists for,
    and the one that makes it an upgrade rather than an alternative.
    """
    query = "the chip reads K-901 on the collar"
    pattern = dk_auth.RegexAuthority("chip", re.compile(r"\bK-\d{3}\b"), lambda text, _group: text)

    assert ScanningSignal(entities).candidates(query, k=5) == []
    assert _spans(AuthoritySignal(pattern).candidates(query, k=5)) == [("K-901", (15, 20))]


def test_one_authority_per_axis_returns_the_containing_form_alone(entities, one_axis):
    """**Criterion: the downgrade direction**, and it is the same query.

    ``golden retriever`` contains ``retriever`` and both are declared. The
    default rung probes every token span independently and reports both, at
    the two places they sit. An authority suppresses a form contained by one
    it already matched -- ``TokenAligner`` marks a match's tokens consumed --
    so a vocabulary loaded as one authority per axis returns the containing
    form and nothing else.

    Neither answer is wrong. They are different questions, and a consumer
    choosing a rung is choosing which one they are asking.
    """
    assert _spans(ScanningSignal(entities).candidates(QUERY, k=5)) == [
        ("golden_retriever", (3, 19)),
        ("retriever", (10, 19)),
    ]
    assert _spans(AuthoritySignal(one_axis).candidates(QUERY, k=5)) == [
        ("golden_retriever", (3, 19)),
    ]


def test_one_authority_per_form_keeps_the_overlap(one_per_form):
    """The suppression is *within* an authority, not across a bundle.

    Which is why the criterion above says *per axis* rather than *the
    authority stack*: the same vocabulary, spelled one authority per form,
    answers as the default does. A consumer who needs both spans has a way to
    get them, and it is a loading decision rather than a rung parameter.
    """
    assert _spans(AuthoritySignal(one_per_form).candidates(QUERY, k=5)) == [
        ("golden_retriever", (3, 19)),
        ("retriever", (10, 19)),
    ]


# ===== The rung's own surface =====


def test_the_rung_is_named_for_the_key_it_registers_under(one_axis):
    assert AuthoritySignal(one_axis).name == "authority"
    assert AsyncAuthoritySignal(one_axis).name == "authority"


def test_the_rung_does_not_narrow(one_axis):
    """An authority stack holds no declared types, so it must say it cannot filter."""
    assert AuthoritySignal(one_axis).narrows() is False
    assert AsyncAuthoritySignal(one_axis).narrows() is False


def test_the_evidence_is_declared_by_fiat(one_axis):
    """A form the vocabulary carries, matched -- not a guess with a number on it."""
    (candidate,) = AuthoritySignal(one_axis).candidates(QUERY, k=5)
    (evidence,) = candidate.evidence

    assert candidate.score == 1.0
    assert evidence.kind is EvidenceKind.DECLARED
    assert evidence.scoring is Scoring.DECLARED
    assert evidence.score == 1.0
    assert evidence.signal == "authority"


def test_matched_text_is_what_the_span_points_at(one_axis):
    """The two fields agree by construction rather than by a caller's trust."""
    (candidate,) = AuthoritySignal(one_axis).candidates(QUERY, k=5)
    (evidence,) = candidate.evidence

    assert evidence.matched_text == QUERY[evidence.span[0] : evidence.span[1]]
    assert evidence.matched_text == "golden retriever"


def test_k_counts_entities_rather_than_hits(one_per_form):
    """A query naming one entity twice spends one of the caller's ``k``, not two."""
    twice = "a retriever met another retriever"
    (candidate,) = AuthoritySignal(one_per_form).candidates(twice, k=1)

    assert str(candidate.entity_id) == "retriever"
    assert [evidence.span for evidence in candidate.evidence] == [(2, 11), (24, 33)]


def test_k_cuts_entities_in_the_order_the_stack_proposed_them(one_per_form):
    """Longest first at a shared start, which the stack's own sort already gives."""
    found = AuthoritySignal(one_per_form).candidates(QUERY, k=1)

    assert [str(candidate.entity_id) for candidate in found] == ["golden_retriever"]


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_input_carrying_no_text_proposes_nothing(one_axis, empty):
    """A rung is handed whatever a consumer typed, including nothing at all."""
    assert AuthoritySignal(one_axis).candidates(empty, k=5) == []


def test_the_batch_form_answers_one_list_per_query_in_the_order_asked(one_axis):
    answers = AuthoritySignal(one_axis).candidates_many([QUERY, "nothing here"], k=5)

    assert [[str(c.entity_id) for c in answer] for answer in answers] == [
        ["golden_retriever"],
        [],
    ]


# ===== The twin =====


def test_the_twins_agree_on_their_surface():
    """Named rather than discovered, so a member added to one half is caught here."""
    assert_twin_types_agree(
        AuthoritySignal,
        AsyncAuthoritySignal,
        members=["candidates", "candidates_many", "narrows"],
        # `narrows` stays synchronous on both halves: it reports a property of
        # the rung and reaches for nothing. `name` is absent because it is a
        # property rather than a member with a signature to compare -- what it
        # answers is asserted directly above.
        unflavoured_members=["narrows"],
        compare_return=True,
    )


@pytest.mark.asyncio
async def test_the_async_twin_answers_exactly_as_its_sync_twin_does(one_axis):
    """Not merely *an* answer: the same one, because it runs the same core."""
    assert _spans(await AsyncAuthoritySignal(one_axis).candidates(QUERY, k=5)) == _spans(
        AuthoritySignal(one_axis).candidates(QUERY, k=5)
    )


@pytest.mark.asyncio
async def test_the_async_batch_form_answers_in_the_order_asked(one_axis):
    answers = await AsyncAuthoritySignal(one_axis).candidates_many([QUERY, "nothing here"], k=5)

    assert [[str(c.entity_id) for c in answer] for answer in answers] == [
        ["golden_retriever"],
        [],
    ]


# ===== The registry =====


def test_importing_this_package_registers_the_kind_in_both_flavours():
    """The mark ``dataknobs_common`` leaves is cleared by the registration.

    `dataknobs_xization`'s own test suite has imported the module, so this
    asserts the post-import state; what the *pre*-import state says is
    `dataknobs_common`'s to assert, and it does.
    """
    assert "authority" in signal_backends.list_keys()
    assert "authority" in async_signal_backends.list_keys()


def test_the_factory_builds_the_rung_a_consumer_configured(one_axis):
    """``kind: authority`` reaches a rung, and the config key is the stack."""
    built = signal_backends.create("authority", {"authorities": one_axis})
    built_async = async_signal_backends.create("authority", {"authorities": one_axis})

    assert isinstance(built, AuthoritySignal)
    assert isinstance(built_async, AsyncAuthoritySignal)
    assert _spans(built.candidates(QUERY, k=5)) == [("golden_retriever", (3, 19))]


# ===== A stack whose members name their own columns =====


@pytest.fixture
def mixed_columns() -> dk_auth.AuthoritiesBundle:
    """The same vocabulary as ``one_per_form``, with one member renaming its columns.

    A supported configuration rather than an exotic one: ``find_matches``
    documents that a caller reading rows back "wants the other one, because
    the rows are in their finder's columns and not in this bundle's", and
    ``find_matches_with_finders`` exists to carry that pairing. So a member
    built with its own metadata is a case the stack already answers -- and a
    rung reading every row through one vocabulary is a rung that cannot.
    """
    return dk_auth.AuthoritiesBundle(
        "animals",
        auths=[
            dk_auth.RegexAuthority(
                "golden_retriever",
                re.compile(re.escape("Golden Retriever"), re.IGNORECASE),
                lambda _text, _group: "golden_retriever",
            ),
            dk_auth.RegexAuthority(
                "retriever",
                re.compile(re.escape("Retriever"), re.IGNORECASE),
                lambda _text, _group: "retriever",
                auth_anns_builder=dk_auth.AuthorityAnnotationsBuilder(
                    metadata=dk_auth.AuthorityAnnotationsMetaData(
                        start_pos_col="begin", end_pos_col="finish"
                    )
                ),
            ),
        ],
    )


def test_a_member_that_named_its_own_columns_is_read_through_its_own_names(mixed_columns):
    """Each row is read through the authority that wrote it, not through the stack.

    The rung used to take the column names from the *stack* it was handed and
    read every row with them. For a bundle that is the wrong vocabulary by
    construction: its members write in theirs, and a member built with its
    own metadata produced rows whose position columns the bundle's names do
    not reach -- read back as ``NaN`` and turned into a span by ``int()``,
    which raises ``ValueError`` several frames below anything naming an
    authority.
    """
    assert _spans(AuthoritySignal(mixed_columns).candidates(QUERY, k=5)) == [
        ("golden_retriever", (3, 19)),
        ("retriever", (10, 19)),
    ]


def test_the_order_survives_a_member_naming_its_own_columns(mixed_columns):
    """Start ascending and end descending, which is the order the rung publishes.

    The order used to be the shared frame's, and a frame can only sort by one
    vocabulary -- so a member whose position column the sort could not read
    was placed by where its ``NaN`` landed rather than by where its match
    sat. The containing form comes first here because it starts earlier, and
    :func:`test_one_authority_per_form_keeps_the_overlap` is the same claim
    over a stack that shares one vocabulary.
    """
    spans = [span for _, span in _spans(AuthoritySignal(mixed_columns).candidates(QUERY, k=5))]

    assert spans == sorted(spans, key=lambda span: (span[0], -span[1]))


def test_a_row_no_finder_can_read_is_named_rather_than_turned_into_a_span():
    """The branch that says an authority's ``finders()`` is wrong about it.

    Reachable only by an ``Authority`` that writes rows in a vocabulary it
    does not report -- which the two shipped arms cannot do, so the subclass
    below is what such an authority looks like. Worth a named refusal rather
    than a skip: dropping the row would lose a match silently, and reading it
    through the wrong names is what produced ``int(NaN)``.
    """

    class WritesElsewhere(dk_auth.Authority):
        """Reports its own metadata and writes in somebody else's columns."""

        def has_value(self, value):
            return True

        def add_annotations(self, text_obj):
            text_obj.annotations.add_dict(
                {"who": "retriever", "from": 10, "to": 19, "text": "retriever"}
            )
            return text_obj.annotations

    with pytest.raises(dk_exc.ValidationError, match="names the columns of a row it produced"):
        AuthoritySignal(WritesElsewhere("animal")).candidates(QUERY, k=5)


# ===== One assembly, not two =====


def test_the_declared_evidence_agrees_with_a_dataknobs_common_rung(entities, one_axis):
    """The same shape from both, because both assemble through the same function.

    Asserted as an agreement rather than as two literals, because the claim is
    that what a declared hit's evidence *is* belongs to the cascade's
    vocabulary and not to whichever rung found the hit. This rung used to
    carry its own copy of that assembly -- the ``1.0`` by fiat, the
    :class:`Scoring` member, the ``matched_text`` slice -- in a second
    distribution, with nothing comparing the two. A change to declared
    evidence in ``dataknobs_common`` would have reached one and not the other,
    and no test would have failed.

    The **spans** are deliberately left out of the comparison: that the two
    rungs locate forms differently is the whole subject of the criteria at the
    top of this file. What must agree is everything else.
    """
    (from_common,) = [
        candidate
        for candidate in ScanningSignal(entities).candidates(QUERY, k=5)
        if candidate.entity_id == "golden_retriever"
    ]
    (from_stack,) = AuthoritySignal(one_axis).candidates(QUERY, k=5)

    assert str(from_stack.entity_id) == from_common.entity_id
    assert from_stack.score == from_common.score

    (stack_evidence,) = from_stack.evidence
    (common_evidence,) = from_common.evidence

    assert stack_evidence.kind is common_evidence.kind
    assert stack_evidence.scoring is common_evidence.scoring
    assert stack_evidence.score == common_evidence.score
    assert stack_evidence.matched_text == common_evidence.matched_text


# ===== A consumer who registered their own rung under this key =====


def _consumer_rung(config):
    """Stand-in for a rung a consumer wrote and registered themselves."""
    return AuthoritySignal(config["authorities"])


@pytest.fixture
def consumer_owns_the_key():
    """This key already registered when this module's registration runs.

    Which is the order a consumer following ``dataknobs_common``'s own
    registry prose produces: *"A consumer who writes either kind and
    registers their own clears the mark the same way -- which is the
    extension point, not a leak."*
    """
    signal_backends.register("authority", _consumer_rung, override=True)
    async_signal_backends.register("authority", _consumer_rung, override=True)
    yield
    dk_er._register_rungs(override=True)


def test_this_packages_registration_does_not_crash_on_a_consumers_own(consumer_owns_the_key):
    """``register`` refuses a key it already holds, and this one runs at import.

    So a consumer who registered their own ``authority`` rung first made
    **any** later ``import dataknobs_xization`` raise ``OperationError`` --
    out of the import statement, naming a registry they may never have heard
    of, and taking down every other thing this package does. Reaching the
    import indirectly, for the markdown chunker or the normalizer, was enough.
    """
    dk_er._register_rungs()


def test_a_consumers_own_rung_survives_this_packages_registration(consumer_owns_the_key):
    """And it is theirs that stands, not ours.

    The half a bare ``override=True`` would have got wrong: not crashing is
    not the same as not clobbering, and the registry's prose calls a
    consumer's own registration the extension point.
    """
    dk_er._register_rungs()

    assert signal_backends.get_factory("authority") is _consumer_rung
    assert async_signal_backends.get_factory("authority") is _consumer_rung

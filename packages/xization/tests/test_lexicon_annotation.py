"""The dictionary arm of the authority stack, exercised end to end.

``RegexAuthority`` and ``DataframeAuthority`` are the two arms of
:class:`~dataknobs_xization.authorities.Authority`, and only the regex one had
ever been run. Three independently fatal breaks stood between a caller and a
single annotation from the dictionary arm, all three dating from the module's
first commit and none of them reachable by reading one file:

* ``add_annotations`` took two parameters where its own base class declares
  one and passes one,
* the call into ``LexicalExpander.build_first_token`` supplied an ``input_id``
  keyword that method has never accepted, and
* ``TokenMatch.matched_text`` read ``Token.input_text``, a member no ``Token``
  has ever had.

Every test here goes through :meth:`Authority.annotate_input` -- the entry
point a consumer calls -- rather than through the repaired members directly,
because reaching those members by their new spellings is what a reviewer can
already do by reading. What a reviewer cannot do is run the path.
"""

import re
import sys

import pandas as pd
import pytest

import dataknobs_xization.annotations as dk_anns
import dataknobs_xization.authorities as dk_auth
import dataknobs_xization.lexicon as dk_lex
import dataknobs_xization.masking_tokenizer as dk_tok

QUERY = "my golden retriever has been limping"


@pytest.fixture
def animals() -> dk_auth.AuthorityData:
    """Three declared forms, two of which overlap in the query above.

    ``"golden retriever"`` contains ``"retriever"``, which is what makes this
    data able to say anything about containment; ``"beagle"`` is declared and
    absent, so a match on it would be a false positive rather than a miss.
    """
    return dk_auth.AuthorityData(
        pd.DataFrame({"animal": ["golden retriever", "retriever", "beagle"]}),
        "animal",
    )


@pytest.fixture
def authority(animals: dk_auth.AuthorityData) -> dk_lex.DataframeAuthority:
    """The authority under test, with the identity expander the factory defaults to.

    ``LexicalExpander(None, None)`` is what
    :meth:`MultiAuthorityFactory.get_lexical_expander` builds when a consumer
    supplies none, so the default path is the one exercised here.
    """
    return dk_lex.DataframeAuthority("animal", dk_lex.LexicalExpander(None, None), animals)


# --- the path runs at all --------------------------------------------------


def test_a_declared_form_is_annotated_where_it_sat(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """One call, one annotation, and the offsets point back into the query.

    This is the whole of what the arm was supposed to do and had never done.
    Each of the three breaks aborts this test on its own, which is why they
    are one defect rather than three: no one of the three fixes gets a row out
    of this call.
    """
    anns = authority.annotate_input(QUERY)

    assert anns.df is not None, "the dictionary arm produced no annotations at all"
    assert len(anns.df) == 1
    row = anns.df.iloc[0]
    assert row["text"] == "golden retriever"
    assert (row["start_pos"], row["end_pos"]) == (3, 19)
    assert QUERY[row["start_pos"] : row["end_pos"]] == row["text"]
    assert row["ann_type"] == "animal"
    assert row["auth_id"] == 0, "the value id is the authority row the form came from"


def test_the_matched_text_is_sliced_from_the_original_not_rebuilt(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """``matched_text`` reads the whole text off the token and slices it.

    The break here was a member name, so the assertion that matters is not
    that *a* string comes back but that it is the one the source text
    contains, separator and all -- a join of the matched tokens would produce
    the same characters here for the wrong reason, so the query is spelled
    with two spaces to tell the two apart.
    """
    query = "my golden  retriever has been limping"
    anns = authority.annotate_input(query)

    row = anns.df.iloc[0]
    assert row["text"] == "golden  retriever"
    assert query[row["start_pos"] : row["end_pos"]] == row["text"]


def test_the_text_id_survives_into_the_token_stream(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """The id travels on the text object, which is why the keyword was redundant.

    The removed ``input_id=`` argument was passing ``doctext.text_id``
    alongside ``doctext.text``; the object carries both, and
    ``TextFeatures`` keeps it. Asserted through the aligner's own token stream
    rather than through a member of the repair, so the claim is that the
    identified text reached the tokenizer.
    """
    text_obj = dk_anns.AnnotatedText(
        QUERY,
        metadata=dk_tok.dk_doc.TextMetaData("case-42", "text"),
        annots_metadata=authority.metadata,
    )

    authority.annotate_input(text_obj)

    first_token = authority.prev_aligner.first_token
    assert first_token.text_id == "case-42"
    assert first_token.full_text == QUERY


# --- what the aligner decides ---------------------------------------------


def test_a_contained_form_is_suppressed_by_the_form_containing_it(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """``"retriever"`` is declared, present, and deliberately not returned.

    ``TokenAligner`` marks every token of a match as processed, so a form
    inside a longer match is never offered -- the behaviour is a property of
    the traversal rather than a setting, which is why nothing here is
    configurable. Pinned because it is the one behavioural claim a reader
    would otherwise have to take from the traversal's shape.
    """
    anns = authority.annotate_input(QUERY)

    assert anns.df["text"].tolist() == ["golden retriever"]
    assert "retriever" not in anns.df["text"].tolist()

    # The contained form is genuinely declared and genuinely present: its
    # absence above is the aligner's decision, not a gap in the vocabulary.
    assert authority.has_value("retriever")
    assert not authority.find_variations("retriever", starts_with=True).empty
    assert QUERY[10:19] == "retriever"


def test_each_occurrence_of_a_form_is_annotated(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """Two mentions of one value are two rows carrying one authority id."""
    anns = authority.annotate_input("a beagle met a beagle")

    assert anns.df["text"].tolist() == ["beagle", "beagle"]
    assert anns.df["auth_id"].tolist() == [2, 2]
    assert list(zip(anns.df["start_pos"], anns.df["end_pos"], strict=True)) == [(2, 8), (15, 21)]


def test_a_text_with_no_declared_form_annotates_nothing(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """The empty answer is empty rather than raising.

    ``validate_ann_dicts`` refuses an empty list, so nothing is added --
    which has to be the quiet path, since most texts carry no declared form.
    """
    anns = authority.annotate_input("my cat has been limping")

    assert anns.df is None


# --- the contract the base class calls through -----------------------------


def test_add_annotations_takes_what_annotate_input_passes(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """The deviant signature, pinned against the contract it deviated from.

    ``Authority.annotate_input`` calls ``self.add_annotations(text_obj)`` with
    one argument, and ``RegexAuthority`` and ``AuthoritiesBundle`` both accept
    exactly that. This asserts the third implementation agrees -- structurally,
    so that a future signature change fails here rather than at whichever
    consumer happens to run first.
    """
    text_obj = dk_anns.AnnotatedText(QUERY, annots_metadata=authority.metadata)

    annotations = authority.add_annotations(text_obj)

    assert annotations is text_obj.annotations, "the annotations belong to the text object"
    assert len(annotations.df) == 1


def test_the_two_arms_annotate_one_text_through_one_bundle(
    animals: dk_auth.AuthorityData,
) -> None:
    """A bundle over both arms, which is the composite door a consumer opens.

    The regex arm is the positive control: it worked before this change and
    works after, so a failure confined to the dictionary rows is a failure of
    the arm rather than of the fixture. ``AuthoritiesBundle`` calls
    ``annotate_input`` per member, so this also exercises the repaired
    signature through a second caller.
    """
    lexical = dk_lex.DataframeAuthority("animal", dk_lex.LexicalExpander(None, None), animals)
    regex = dk_auth.RegexAuthority("date", re.compile(r"\d{2}/\d{2}/\d{4}"))
    bundle = dk_auth.AuthoritiesBundle("intake", auths=[lexical, regex])

    anns = bundle.annotate_input("my golden retriever limped in on 07/04/1776")

    assert sorted(anns.df["ann_type"].tolist()) == ["animal", "date"]
    assert sorted(anns.df["text"].tolist()) == ["07/04/1776", "golden retriever"]


def test_the_factory_builds_an_authority_that_annotates(animals: dk_auth.AuthorityData) -> None:
    """The constructed-by-factory door reaches the same repaired path.

    ``MultiAuthorityFactory`` is the only code in the tree that builds a
    ``DataframeAuthority``, so a fix the factory's product does not inherit
    would leave the arm as inert as it was.
    """

    class _Animals(dk_lex.SimpleMultiAuthorityData):
        pass

    multi = _Animals(animals.df, "animals")
    factory = dk_lex.MultiAuthorityFactory("animal")

    authority = factory.build_authority("animal", dk_auth.AuthorityAnnotationsBuilder(), multi)
    anns = authority.annotate_input(QUERY)

    assert isinstance(authority, dk_lex.DataframeAuthority)
    assert anns.df["text"].tolist() == ["golden retriever"]


# --- the lookups the aligner is built on -----------------------------------


def test_the_variation_index_answers_in_both_directions(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """Value to id, variation to id, and id back to value.

    These are the members ``TokenAligner`` consults on every token, and none
    of them had a test. Exercised together because the round trip is the
    property worth holding: an id that does not lead back to its value makes
    every annotation row above unresolvable.
    """
    assert authority.get_value_ids("golden retriever") == {0}
    assert authority.get_id_by_variation("beagle") == {2}
    assert authority.get_values_by_id(1) == {"retriever"}
    assert authority.get_variations("beagle") == {"beagle"}
    assert not authority.has_value("cat")
    assert authority.get_value_ids("cat") == set()


def test_a_variation_lookup_works_before_anything_has_expanded_the_values(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """A second defect, found by auditing the class rather than by the arm.

    ``get_id_by_variation`` reads the expander's variation-to-term index,
    which is populated as a *side effect* of expanding this authority's
    values. Nothing in the method did that, so on a freshly built authority it
    returned an empty set for every variation -- the same answer it gives for
    a variation that genuinely is not declared, which is why no caller could
    have noticed.

    The first call is the whole test: a second one passes either way, because
    by then some other member has usually warmed the index.
    """
    assert authority.get_id_by_variation("golden retriever") == {0}


def test_find_variations_scopes_its_match(authority: dk_lex.DataframeAuthority) -> None:
    """``starts_with`` is the aligner's mode; the others are the consumer's."""
    assert authority.find_variations("golden", starts_with=True).tolist() == ["golden retriever"]
    assert authority.find_variations("retriever", ends_with=True).tolist() == [
        "golden retriever",
        "retriever",
    ]
    assert authority.find_variations("beagle").tolist() == ["beagle"]
    assert authority.find_variations("retriever", scope="contains").tolist() == [
        "golden retriever",
        "retriever",
    ]
    assert authority.find_variations("cat", starts_with=True).empty


def test_the_aligner_records_one_group_per_match_and_flattens_on_request(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """``matches`` is the record; ``annotations`` is the view derived from it.

    The aligner keeps its rows grouped by match because that is the unit an
    authority's validator judges. ``annotations`` stayed, as the flat list it
    always was -- it can only have been read rather than appended to, since
    building an aligner over a text with any match raised before this change.
    """
    authority.annotate_input("a beagle met a beagle")
    aligner = authority.prev_aligner

    assert [[row["text"] for row in match] for match in aligner.matches] == [
        ["beagle"],
        ["beagle"],
    ]
    assert [row["text"] for row in aligner.annotations] == ["beagle", "beagle"]


# --- how far the walk can go ----------------------------------------------


@pytest.fixture
def prefixed_animals() -> dk_auth.AuthorityData:
    """A vocabulary where one declared form is a prefix of another.

    ``"golden"`` and ``"golden retriever"`` both match starting at the same
    token, which is the only arrangement that makes the aligner's recording
    order observable: with no shared start token the matches are recorded in
    start-position order whatever the traversal does.
    """
    return dk_auth.AuthorityData(
        pd.DataFrame({"animal": ["golden", "golden retriever", "beagle"]}),
        "animal",
    )


def test_a_document_longer_than_the_recursion_limit_is_annotated(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """Walking N tokens must not cost N stack frames.

    The token stream is a linked list and the aligner followed it by
    recursing on ``next_token``, so the longest document it could annotate
    was fixed by ``sys.getrecursionlimit()`` rather than by anything about
    the document. At the default limit of 1000 the last length that survived
    was 980 whitespace tokens -- this repository's ``README.md`` (690 words)
    annotated and ``packages/xization/CHANGELOG.md`` (2849 words) raised
    ``RecursionError``, so the package's own changelog was past the edge.

    The length is taken from the live limit rather than written in, so
    raising the limit cannot quietly leave this test reproducing nothing.
    """
    sentence = "a beagle wandered past the quiet house "
    repeats = (sys.getrecursionlimit() + 200) // len(sentence.split()) + 1
    document = (sentence * repeats).strip()
    assert len(document.split()) > sys.getrecursionlimit(), "the document must outrun the stack"

    anns = authority.annotate_input(document)

    assert anns.df is not None, "the long document produced no annotations at all"
    assert anns.df["text"].tolist() == ["beagle"] * repeats


def test_every_token_is_looked_up_once(prefixed_animals: dk_auth.AuthorityData) -> None:
    """The walk visits each token once, so its cost is linear in the document.

    Depth was the failure; this is the cost that outlived it. ``_processed_idx``
    records only tokens that were *part of a match*, so a token matching
    nothing was never marked and the traversal re-queried the authority for it
    once per enclosing level of the walk -- quadratic in the token count, on
    the unmatched tokens that are the overwhelming majority of any real
    document.

    Counted through a real subclass rather than a mock: the authority under
    test is the production one, and the override forwards every call.
    """

    class _Counting(dk_lex.DataframeAuthority):
        """A ``DataframeAuthority`` that records what the aligner asked it."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.queried: list[str] = []

        def find_variations(self, text: str, **kwargs: object) -> pd.Series:
            self.queried.append(text)
            return super().find_variations(text, **kwargs)

    authority = _Counting("animal", dk_lex.LexicalExpander(None, None), prefixed_animals)

    authority.annotate_input("my golden retriever met a beagle")

    assert authority.queried == ["my", "golden", "met", "a", "beagle"], (
        "each token is offered to the authority once, and a token consumed by "
        "a match is not offered at all"
    )


def test_matches_are_recorded_in_document_order(
    prefixed_animals: dk_auth.AuthorityData,
) -> None:
    """The recording order is the document's, including at a shared start token.

    Two declared forms starting at the same token is the only arrangement
    that can separate document order from the traversal's, and it used to:
    the walk recorded a match, walked everything reachable past its end, and
    only then recorded the next match starting at the same token, so
    ``"beagle"`` -- six tokens later -- landed between the two forms that
    both start at ``"golden"``.

    Asserted against both views the order reaches. ``Annotations`` sorts by
    span on every add, so the dataframe showed document order throughout and
    is the control here rather than the claim: it reads the same before and
    after, which is what makes the change one of consultation order alone.
    """
    authority = dk_lex.DataframeAuthority(
        "animal", dk_lex.LexicalExpander(None, None), prefixed_animals
    )

    anns = authority.annotate_input("my golden retriever met a beagle")

    assert [[row["text"] for row in match] for match in authority.prev_aligner.matches] == [
        ["golden"],
        ["golden retriever"],
        ["beagle"],
    ]
    assert [match[0]["start_pos"] for match in authority.prev_aligner.matches] == [3, 3, 26]
    assert anns.df["text"].tolist() == ["golden retriever", "golden", "beagle"]


def test_the_walk_visits_each_token_once_so_it_needs_no_visited_set(
    prefixed_animals: dk_auth.AuthorityData,
) -> None:
    """Document order is what makes the walk a chain rather than a search.

    The traversal order was held by an explicit stack, and a match's
    continuation could reach a token the plain ``next_token`` step would
    reach later, so a second set of token numbers was needed to keep the walk
    from re-walking one. Following ``next_token`` and nothing else visits
    every token exactly once by construction: ``token_num`` increases by one
    each step and the chain ends at ``None``.

    Pinned through the authority's own query count, because that is where a
    re-walk would show up -- a token walked twice is a token offered to the
    authority twice.
    """

    class _Counting(dk_lex.DataframeAuthority):
        """A ``DataframeAuthority`` that records what the aligner asked it."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.queried: list[str] = []

        def find_variations(self, text: str, **kwargs: object) -> pd.Series:
            self.queried.append(text)
            return super().find_variations(text, **kwargs)

    authority = _Counting("animal", dk_lex.LexicalExpander(None, None), prefixed_animals)

    authority.annotate_input("my golden retriever met a beagle")

    assert authority.queried == ["my", "golden", "met", "a", "beagle"], (
        "a token consumed by a match is not offered, and no token is offered twice"
    )
    assert not hasattr(authority.prev_aligner, "_walked_idx"), (
        "the walked set is what a chain walk does not need"
    )


# --- what the answers are made of ---


def test_membership_is_answered_with_a_python_bool(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """Bug: ``has_value`` answered with ``numpy.bool``, not ``bool``.

    It returned ``np.any(...)`` unconverted, so the value a caller got back
    was a numpy scalar wearing the annotation ``bool``. Everything that reads
    it for truth is unaffected, which is why nothing noticed; everything that
    reads it for *identity* -- ``is True``, ``is False``, a ``bool`` key in a
    dict, an equality check against a sentinel -- silently disagreed with the
    signature.
    """
    assert authority.has_value("beagle") is True
    assert authority.has_value("cat") is False


class _FalsyVariations:
    """A callable that is falsy: what ``if variations_fn`` cannot tell from ``None``."""

    def __len__(self) -> int:
        return 0

    def __call__(self, term: str) -> set[str]:
        return {term, term.upper()}


def test_an_expansion_function_that_is_falsy_is_still_used() -> None:
    """Bug: a falsy callable was silently replaced by the identity expansion.

    ``LexicalExpander`` accepts ``None`` for either function and substitutes a
    default, and it asked which it had been given with ``if variations_fn``.
    A function object is never falsy, so the question read correctly for every
    argument anyone had passed -- but it is the wrong question, and any other
    callable that defines ``__len__`` or ``__bool__`` answers it wrongly. The
    caller's expansion is then dropped without a word, and every term expands
    to itself.
    """
    expander = dk_lex.LexicalExpander(_FalsyVariations(), None)

    assert expander("dog") == {"dog", "DOG"}


def test_the_latest_aligner_is_none_before_anything_is_annotated(
    authority: dk_lex.DataframeAuthority,
) -> None:
    """``prev_aligner`` reports the last call to ``add_annotations``; there has been none."""
    assert authority.prev_aligner is None

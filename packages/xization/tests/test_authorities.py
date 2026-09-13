import re

import pandas as pd
import pytest

import dataknobs_xization.authorities as dk_auth
import dataknobs_xization.lexicon as dk_lex


def text_authority_annotations_metadata():
    md = dk_auth.AuthorityAnnotationsMetaData()
    assert md.auth_id_col == dk_auth.KEY_AUTH_ID_COL


def test_regex_authority_no_groups():
    r = re.compile(r"\d{2}/\d{2}/\d{4}")
    rauth = dk_auth.RegexAuthority("date", r, lambda x, y: f"{y}:{x}")
    anns = rauth.annotate_input("abc 07/04/1776 xyz")
    assert len(anns.df) == 1
    assert anns.df["text"].to_list() == ["07/04/1776"]


def test_regex_authority_no_name_groups():
    r = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
    rauth = dk_auth.RegexAuthority("date", r, lambda x, y: f"{y}:{x}")
    anns = rauth.annotate_input("abc 07/04/1776 xyz")
    assert len(anns.df) == 3
    assert anns.df["text"].to_list() == ["07", "04", "1776"]
    assert anns.df["date_field"].to_list() == [1, 2, 3]


def test_regex_authority_named_groups():
    r = re.compile(r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4}|\d{2})")

    class DateValidator(dk_auth.AnnotationsValidator):
        def validate_annotation_rows(self, auth_annotations):
            atts = auth_annotations.attributes
            month = int(atts["month"])
            day = int(atts["day"])
            int(atts["year"])  # must parse; range is not validated here
            if month < 1 or month > 12:
                return False
            if day < 1 or day > 31:
                return False
            # Just simple validation here; ignoring days per month
            return True

    rauth = dk_auth.RegexAuthority(
        "date",
        r,
        lambda x, y: f"{y}:{x}",
        anns_validator=DateValidator(),
    )
    anns = rauth.annotate_input("abc 07/04/1776 xyz")
    assert len(anns.df) == 3
    assert anns.df["text"].to_list() == ["07", "04", "1776"]
    assert anns.df["date_field"].to_list() == ["day", "month", "year"]

    anns = rauth.annotate_input("abc 13/32/1776 xyz")
    assert anns.df is None


# --- what the validator is shown: one match at a time ----------------------
#
# `anns_validator` is documented identically on `Authority`, `LexicalAuthority`
# and `AuthoritiesBundle`: "fn(auth, anns_dict_list) that returns True if the
# list of annotation row dicts are valid to be added as annotations for a
# single match or 'entity'". The unit is the contract, so the two arms of
# `Authority` have to agree on it, and the tests below assert that agreement
# rather than either arm's behaviour on its own.


VOCABULARY = ["golden retriever", "beagle"]
QUERY = "my golden retriever met a beagle"


class _Recorder(dk_auth.AnnotationsValidator):
    """A real validator that records what it is shown and rejects beagles.

    Written to the documented interface -- it reads the rows of one entity
    through ``AuthAnnotations`` -- because the harm the contract prevents is
    precisely that a validator written this way is handed a batch spanning
    unrelated entities and cannot judge it.
    """

    def __init__(self) -> None:
        self.shown: list[list[str]] = []

    def validate_annotation_rows(self, auth_annotations) -> bool:
        text_col = auth_annotations.auth.metadata.text_col
        texts = [row[text_col] for row in auth_annotations.ann_row_dicts]
        self.shown.append(texts)
        return "beagle" not in texts


def _dictionary_arm(anns_validator=None) -> dk_lex.DataframeAuthority:
    return dk_lex.DataframeAuthority(
        "animal",
        dk_lex.LexicalExpander(None, None),
        dk_auth.AuthorityData(pd.DataFrame({"animal": VOCABULARY}), "animal"),
        anns_validator=anns_validator,
    )


def _regex_arm(anns_validator=None) -> dk_auth.RegexAuthority:
    return dk_auth.RegexAuthority(
        "animal",
        re.compile(r"golden retriever|beagle"),
        anns_validator=anns_validator,
    )


def test_both_arms_show_the_validator_one_match_at_a_time():
    """The same validator, the same query, and the same two calls.

    The dictionary arm used to make one call carrying every match in the
    document, so a validator that rejected any one of them destroyed all of
    them. Asserted as an agreement between the arms rather than as two
    literals, because the claim is that the unit is the contract's and not
    each implementation's.
    """
    lexical_seen, regex_seen = _Recorder(), _Recorder()

    lexical_anns = _dictionary_arm(lexical_seen).annotate_input(QUERY)
    regex_anns = _regex_arm(regex_seen).annotate_input(QUERY)

    assert lexical_seen.shown == regex_seen.shown
    assert lexical_seen.shown == [["golden retriever"], ["beagle"]]
    assert lexical_anns.df["text"].to_list() == regex_anns.df["text"].to_list()
    assert lexical_anns.df["text"].to_list() == ["golden retriever"]


def test_a_rejected_match_does_not_take_the_valid_ones_with_it():
    """The consumer-visible half: one bad match costs one match.

    Stated separately from the call-count claim above because this is the
    harm -- the previous behaviour returned an empty `Annotations`, which is
    indistinguishable from a text carrying no declared form at all.
    """
    anns = _dictionary_arm(_Recorder()).annotate_input(QUERY)

    assert anns.df is not None, "rejecting one match emptied the whole document"
    assert anns.df["text"].to_list() == ["golden retriever"]


def test_with_no_validator_both_arms_return_both_matches():
    """The control: the difference above is the validation, not the matching."""
    lexical_anns = _dictionary_arm().annotate_input(QUERY)
    regex_anns = _regex_arm().annotate_input(QUERY)

    assert lexical_anns.df["text"].to_list() == ["golden retriever", "beagle"]
    assert regex_anns.df["text"].to_list() == ["golden retriever", "beagle"]


def test_a_multi_row_match_is_judged_and_added_as_one_unit():
    """One match's rows stand or fall together, and its neighbours are untouched.

    A pattern with named groups produces one row per group, so "a single
    match" is three rows here rather than one. Both halves of the unit are
    asserted at once: the invalid date loses all three of its rows, and the
    valid date keeps all three of its own.
    """
    r = re.compile(r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4})")

    class MonthValidator(dk_auth.AnnotationsValidator):
        def validate_annotation_rows(self, auth_annotations):
            return 1 <= int(auth_annotations.attributes["month"]) <= 12

    rauth = dk_auth.RegexAuthority("date", r, anns_validator=MonthValidator())
    anns = rauth.annotate_input("abc 07/04/1776 and 13/32/1776 xyz")

    assert anns.df["text"].to_list() == ["07", "04", "1776"]
    assert anns.df["start_pos"].to_list() == [4, 7, 10]


# --- what order the validator is shown them in -----------------------------
#
# The unit is the match; this is the sequence those matches arrive in.
# `Authority.add_valid_annotations` iterates the matches as the arm found
# them, so the order an arm emits *is* the consultation order, and the two
# arms have to agree on it for the same reason they have to agree on the unit.


PREFIXED_VOCABULARY = ["golden", "golden retriever", "beagle"]


class _Positions(dk_auth.AnnotationsValidator):
    """A validator that records the start position of each match it is shown."""

    def __init__(self) -> None:
        self.starts: list[int] = []

    def validate_annotation_rows(self, auth_annotations) -> bool:
        metadata = auth_annotations.auth.metadata
        self.starts.append(auth_annotations.ann_row_dicts[0][metadata.start_pos_col])
        return True


def test_both_arms_consult_the_validator_in_document_order():
    """The order is the document's on both arms, asserted as the same property.

    The dictionary arm used to emit the traversal's order instead: a match,
    then everything reachable past its end, and only then the next match
    starting at the same token. So a validator was shown ``"golden"``, then
    ``"beagle"`` twenty-three characters later, then ``"golden retriever"``
    back at the first -- while the regex arm, iterating ``re.finditer``, has
    always been in document order.

    Asserted as ascending starts rather than as two literals because the
    claim is about the contract both arms owe, and the arms cannot be given
    the same matches here: ``re.finditer`` does not produce two matches at
    one position, which is exactly the case that separated them.
    """
    lexical_seen, regex_seen = _Positions(), _Positions()

    dk_lex.DataframeAuthority(
        "animal",
        dk_lex.LexicalExpander(None, None),
        dk_auth.AuthorityData(pd.DataFrame({"animal": PREFIXED_VOCABULARY}), "animal"),
        anns_validator=lexical_seen,
    ).annotate_input(QUERY)

    dk_auth.RegexAuthority(
        "animal",
        re.compile(r"golden retriever|beagle"),
        anns_validator=regex_seen,
    ).annotate_input(QUERY)

    assert lexical_seen.starts == sorted(lexical_seen.starts)
    assert regex_seen.starts == sorted(regex_seen.starts)
    assert lexical_seen.starts == [3, 3, 26], "both forms at 'golden', then 'beagle'"


# --- the data a factory is handed answers for the name it holds -------------


def test_flat_authority_data_answers_for_its_own_name():
    """Bug: only a container could answer "give me the data for name N".

    ``get_authority_data`` was declared on ``MultiAuthorityData`` alone, so
    the one shipped ``AuthorityFactory`` -- whose body opens with that call --
    raised ``AttributeError`` from inside itself for every plain
    ``AuthorityData`` it was handed. A leaf holds one authority's values, and
    the name it answers for is its own.
    """
    authdata = dk_auth.AuthorityData(pd.DataFrame({"animal": ["dog", "cat"]}), "animal")

    assert authdata.get_authority_data("animal") is authdata


def test_flat_authority_data_refuses_a_name_it_does_not_hold():
    """The refusal names both the data and the name asked of it."""
    authdata = dk_auth.AuthorityData(pd.DataFrame({"animal": ["dog", "cat"]}), "animal")

    with pytest.raises(KeyError) as excinfo:
        authdata.get_authority_data("colour")

    assert "colour" in str(excinfo.value)
    assert "animal" in str(excinfo.value)


# --- a bundle judges what its members found --------------------------------
#
# `AuthoritiesBundle` accepted an `anns_validator`, stored it, documented it
# with the same "single match or entity" sentence as the base class -- and
# never called it. The leaf arms reach the validator through
# `Authority.add_valid_annotations`; a composite could not, because its
# members added their rows straight to the shared text object and by the time
# the bundle could look there were no match boundaries left to judge. So the
# bundle needs its members to hand matches back rather than only add them,
# which is what `find_matches` is for.


DATE_PATTERN = re.compile(r"\d{2}/\d{2}/\d{4}")
BUNDLE_QUERY = "my golden retriever met a beagle on 07/04/1776"

# The same three matches, with the date first -- so chaining the members in
# the order they were added would hand the bundle [17, 40, 3] rather than the
# document's [3, 17, 40]. The dictionary arm is added first in both.
UNSORTED_QUERY = "on 07/04/1776 my golden retriever met a beagle"


class _Rejects(dk_auth.AnnotationsValidator):
    """A recorder that rejects the one match text it was built to reject."""

    def __init__(self, unwanted: str) -> None:
        self.unwanted = unwanted
        self.shown: list[list[str]] = []

    def validate_annotation_rows(self, auth_annotations) -> bool:
        text_col = auth_annotations.auth.metadata.text_col
        texts = [row[text_col] for row in auth_annotations.ann_row_dicts]
        self.shown.append(texts)
        return self.unwanted not in texts


def _date_arm(anns_validator=None) -> dk_auth.RegexAuthority:
    return dk_auth.RegexAuthority("date", DATE_PATTERN, anns_validator=anns_validator)


def _bundle(anns_validator=None, auths=None) -> dk_auth.AuthoritiesBundle:
    bundle = dk_auth.AuthoritiesBundle("intake", anns_validator=anns_validator)
    for auth in auths if auths is not None else (_dictionary_arm(), _date_arm()):
        bundle.add(auth)
    return bundle


def test_a_bundle_shows_its_validator_one_match_at_a_time():
    """The defect: the validator was stored and never consulted at all.

    Not a granularity mismatch like the dictionary arm's -- a total absence.
    The bundle was handed a validator, kept it, and returned a result
    indistinguishable from the one it returns with no validator at all.
    """
    seen = _Recorder()

    _bundle(seen).annotate_input(BUNDLE_QUERY)

    assert seen.shown == [["golden retriever"], ["beagle"], ["07/04/1776"]]


def test_a_match_the_bundle_rejects_is_the_only_one_missing():
    """The consumer-visible half, and the same unit claim as the leaf arms."""
    anns = _bundle(_Recorder()).annotate_input(BUNDLE_QUERY)

    assert anns.df["text"].to_list() == ["golden retriever", "07/04/1776"]


def test_a_member_validator_still_runs_and_runs_first():
    """Both validators are consulted, and a member's rejection is final.

    The member judges its own matches before offering them, so a match it
    rejects is never shown to the bundle -- the bundle judges what survived
    its member, not what the member found.
    """
    # The member rejects the date; the bundle rejects the beagle.
    bundle_seen, member_seen = _Recorder(), _Rejects("07/04/1776")

    anns = _bundle(bundle_seen, auths=[_dictionary_arm(), _date_arm(member_seen)]).annotate_input(
        BUNDLE_QUERY
    )

    assert member_seen.shown == [["07/04/1776"]]
    assert bundle_seen.shown == [["golden retriever"], ["beagle"]]
    assert anns.df["text"].to_list() == ["golden retriever"]


def test_a_bundle_consults_its_validator_in_document_order():
    """The order is the document's, not the order the members were added.

    `Authority.add_valid_annotations` states that every arm gives it matches
    in document order. A bundle routes through that same seam, so chaining
    its members -- all of member one's matches, then all of member two's --
    would make the composite the one arm that contradicts it. The members are
    merged rather than chained, each read through its own `start_pos_col`.

    Invisible in the result either way: `Annotations.add_dicts` sorts on
    every add, so this is the order a *validator* is consulted in.
    """
    seen = _Positions()

    _bundle(seen).annotate_input(UNSORTED_QUERY)

    assert seen.starts == sorted(seen.starts)
    assert seen.starts == [3, 17, 40], "the date is first in the text"


def test_a_bundle_with_no_validator_returns_what_it_always_did():
    """The compatibility net: the delegating path is untouched."""
    anns = _bundle().annotate_input(BUNDLE_QUERY)

    assert anns.df["text"].to_list() == ["golden retriever", "beagle", "07/04/1776"]


def test_a_nested_bundle_judges_its_members_members():
    """Bundles are authorities, so they nest, and the outer one still judges."""
    seen = _Recorder()
    inner = _bundle()

    anns = _bundle(seen, auths=[inner]).annotate_input(BUNDLE_QUERY)

    assert seen.shown == [["golden retriever"], ["beagle"], ["07/04/1776"]]
    assert anns.df["text"].to_list() == ["golden retriever", "07/04/1776"]


class _NoHook(dk_auth.Authority):
    """A consumer subclass written against the ABC as it was published.

    It implements both abstract methods and nothing else, which is what a
    correct subclass looked like before `find_matches` existed. Adding the
    hook must not break it at construction -- which is the whole reason the
    hook is not abstract.
    """

    def add_annotations(self, text_obj):
        return text_obj.annotations

    def has_value(self, value):
        return False


def test_a_member_without_the_hook_works_in_an_unvalidated_bundle():
    """Nothing that works today stops working."""
    bundle = _bundle(auths=[_dictionary_arm(), _NoHook("legacy")])

    anns = bundle.annotate_input(BUNDLE_QUERY)

    assert anns.df["text"].to_list() == ["golden retriever", "beagle"]


def test_a_member_without_the_hook_in_a_validated_bundle_says_so():
    """The one new failure: loud, named, and only where the answer was wrong.

    A bundle carrying a validator cannot judge matches a member will not hand
    back, and answering as though it had is the defect this change fixes. So
    it raises instead, naming the class that cannot take part.
    """
    bundle = _bundle(_Recorder(), auths=[_dictionary_arm(), _NoHook("legacy")])

    with pytest.raises(NotImplementedError) as excinfo:
        bundle.annotate_input(BUNDLE_QUERY)

    assert "_NoHook" in str(excinfo.value)
    assert "find_matches" in str(excinfo.value)


def test_a_multi_row_match_survives_the_bundle_as_one_unit():
    """The grouping the arms pin individually has to survive the chain.

    Each arm judges a named-group match as one unit of three rows; nothing
    yet pinned that the unit is still one unit once a bundle has carried it.
    """
    r = re.compile(r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4})")
    seen = _Recorder()

    anns = _bundle(seen, auths=[dk_auth.RegexAuthority("date", r)]).annotate_input(
        "abc 07/04/1776 xyz"
    )

    assert seen.shown == [["07", "04", "1776"]], "one call carrying three rows"
    assert anns.df["text"].to_list() == ["07", "04", "1776"]

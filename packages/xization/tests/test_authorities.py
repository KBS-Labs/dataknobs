import re

import pandas as pd
import pytest

import dataknobs_xization.annotations as dk_annots
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
    unrelated entities and cannot judge it. The column name comes off the
    finder, which is the authority the rows were written by, and so the
    authority that named their columns.
    """

    def __init__(self) -> None:
        self.shown: list[list[str]] = []

    def validate_annotation_rows(self, auth_annotations) -> bool:
        text_col = auth_annotations.finder.metadata.text_col
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
        metadata = auth_annotations.finder.metadata
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
        text_col = auth_annotations.finder.metadata.text_col
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


# --- a validator reads rows through the authority that found them ----------
#
# Every accessor on `AuthAnnotations` used to read the rows through the
# authority the validator was called for, which assumes the authority that
# JUDGES a match is the one that FOUND it. That held for every caller until a
# bundle began judging what its members found: the rows are then a member's,
# written in the member's column vocabulary, while the authority is the
# bundle. The bundle's `field_type` column name does not exist in the
# member's rows, so `attributes` collapsed three fields onto one `None` key
# and said nothing; its `text_col` does not either, so `get_text` raised.
#
# `auth` still names the authority proposing the annotations, which is what
# it has always been documented as. The authority whose columns the rows are
# in is `finder`, and the accessors read that.


NAMED_DATE_PATTERN = re.compile(r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4})")
NAMED_DATE_QUERY = "abc 07/04/1776 xyz"
NAMED_DATE_FIELDS = {"day": "07", "month": "04", "year": "1776"}


class _Vocabulary(dk_auth.AnnotationsValidator):
    """A validator that records the `attributes` accessor's answer per match."""

    def __init__(self) -> None:
        self.attributes: list[dict] = []

    def validate_annotation_rows(self, auth_annotations) -> bool:
        self.attributes.append(dict(auth_annotations.attributes))
        return True


class _Parties(dk_auth.AnnotationsValidator):
    """A validator that records both authorities it is told about per match."""

    def __init__(self) -> None:
        self.proposers: list[dk_auth.Authority] = []
        self.finders: list[dk_auth.Authority] = []

    def validate_annotation_rows(self, auth_annotations) -> bool:
        self.proposers.append(auth_annotations.auth)
        self.finders.append(auth_annotations.finder)
        return True


def _part_date_arm(anns_validator=None) -> dk_auth.RegexAuthority:
    """A member with its own field groups: its rows carry `date_part` columns.

    Reachable through `MultiAuthorityFactory.get_field_groups(name)`, which
    takes the authority's name so that a subclass can vary the field groups
    per authority -- so a bundle over members whose field groups differ is
    one factory subclass away, not a construction invented for this test.
    """
    return dk_auth.RegexAuthority(
        "date",
        NAMED_DATE_PATTERN,
        field_groups=dk_auth.DerivedFieldGroups(field_type_suffix="_part"),
        anns_validator=anns_validator,
    )


def _surface_date_arm(anns_validator=None) -> dk_auth.RegexAuthority:
    """A member with its own metadata: its rows carry the text in `surface_form`."""
    return dk_auth.RegexAuthority(
        "date",
        NAMED_DATE_PATTERN,
        auth_anns_builder=dk_auth.AuthorityAnnotationsBuilder(
            metadata=dk_auth.AuthorityAnnotationsMetaData(text_col="surface_form")
        ),
        anns_validator=anns_validator,
    )


def _auth_annotations(arm, query) -> dk_auth.AnnotationsValidator.AuthAnnotations:
    """The first match `arm` finds in `query`, wrapped as a validator sees it."""
    text_obj = dk_annots.AnnotatedText(query, annots_metadata=arm.metadata)
    return dk_auth.AnnotationsValidator.AuthAnnotations(arm, next(iter(arm.find_matches(text_obj))))


def test_a_bundle_reads_a_members_rows_through_the_members_field_groups():
    """The silent half: three fields collapsed onto one `None` key.

    Asserted as an agreement between the two validators -- the member's own
    and the bundle's -- because the claim is that being judged by a composite
    does not change what the rows say. The literal is spelled out as well, so
    that the agreement cannot be satisfied by both of them being wrong.
    """
    by_itself, by_the_bundle = _Vocabulary(), _Vocabulary()

    _part_date_arm(by_itself).annotate_input(NAMED_DATE_QUERY)
    _bundle(by_the_bundle, auths=[_part_date_arm()]).annotate_input(NAMED_DATE_QUERY)

    assert by_itself.attributes == [NAMED_DATE_FIELDS]
    assert by_the_bundle.attributes == by_itself.attributes


def test_a_bundle_reads_a_members_rows_through_the_members_metadata():
    """The loud half: a `KeyError` raised from inside pandas."""
    by_itself, by_the_bundle = _Vocabulary(), _Vocabulary()

    _surface_date_arm(by_itself).annotate_input(NAMED_DATE_QUERY)
    _bundle(by_the_bundle, auths=[_surface_date_arm()]).annotate_input(NAMED_DATE_QUERY)

    assert by_itself.attributes == [NAMED_DATE_FIELDS]
    assert by_the_bundle.attributes == by_itself.attributes


def test_the_finder_of_a_nested_bundles_match_is_the_member_that_found_it():
    """The finder is the authority whose columns the rows are in, not the member.

    An inner bundle hands the outer one rows its own member wrote, so naming
    the member of the outer bundle would be the same defect one level down.
    """
    seen = _Vocabulary()
    inner = _bundle(auths=[_part_date_arm()])

    _bundle(seen, auths=[inner]).annotate_input(NAMED_DATE_QUERY)

    assert seen.attributes == [NAMED_DATE_FIELDS]


def test_the_validator_is_still_told_which_authority_proposed_the_match():
    """`auth` keeps its documented meaning; the finder is carried beside it.

    `AnnotationsValidator.__call__` documents its first argument as "the
    authority proposing annotations". For a bundle that is the bundle -- it
    is the one whose validator is being consulted and whose decision stands
    -- and a validator that reads it is reading what it was promised.
    """
    seen = _Parties()
    member = _date_arm()
    bundle = _bundle(seen, auths=[member])

    bundle.annotate_input(BUNDLE_QUERY)

    assert seen.proposers == [bundle]
    assert seen.finders == [member]


def test_an_authority_that_finds_its_own_matches_is_its_own_finder():
    """The control: where the two were never distinct, nothing moves."""
    seen = _Parties()
    arm = _date_arm(seen)

    arm.annotate_input(BUNDLE_QUERY)

    assert seen.proposers == [arm]
    assert seen.finders == [arm]


def test_a_plain_callable_validator_is_offered_the_finder_too():
    """The seam is not locked to `AnnotationsValidator`.

    A plain callable is handed the row dicts themselves, so it reads their
    columns off an authority just as the accessors do, and is wrong in the
    same way if that authority is the wrong one. It is offered the finder as
    a third argument wherever the finder is not the authority already given.
    """
    seen = []

    def validator(auth, ann_dicts, finder):
        seen.append((auth.name, finder.name, finder.metadata.text_col))
        return True

    _bundle(validator, auths=[_surface_date_arm()]).annotate_input(NAMED_DATE_QUERY)

    assert seen == [("intake", "date", "surface_form")]


def test_a_three_argument_callable_works_on_an_authority_that_finds_its_own():
    """The finder is offered because the validator asks for it, not because it differs.

    A leaf is always its own finder, so conditioning the call on the two
    authorities differing means a validator written to the three-argument
    form can never be called by one -- it is handed two arguments and raises
    `TypeError` before it runs. What decides how many arguments a call may
    carry is the callable, so a validator that asks for the finder is given
    it, and on a leaf that is the authority itself.
    """
    seen = []

    def validator(auth, ann_dicts, finder):
        seen.append((auth.name, finder.name, finder is auth))
        return True

    anns = _part_date_arm(validator).annotate_input(NAMED_DATE_QUERY)

    assert seen == [("date", "date", True)]
    assert len(anns.df) == 3


def test_a_two_argument_callable_still_works_as_a_bundles_own_validator():
    """The two-argument contract survives the one place the authorities never coincide.

    A bundle never finds its own matches, so `finder` differs from `auth` for
    every match it judges. Offering the third argument on that basis alone
    hands three arguments to every validator a bundle carries, and a callable
    written to the documented two-argument form -- which is every validator
    written before a bundle could judge one -- raises `TypeError`. A
    validator that does not ask for the finder is one that reads nothing
    through it, so it keeps being called the way it was written.
    """
    seen = []

    def validator(auth, ann_dicts):
        seen.append(auth.name)
        return True

    anns = _bundle(validator, auths=[_surface_date_arm()]).annotate_input(NAMED_DATE_QUERY)

    assert seen == ["intake"]
    assert len(anns.df) == 3


def test_a_validator_that_takes_star_args_is_offered_the_finder():
    """A wrapped validator declares `*args`, and that accepts the finder.

    The question is whether the callable can carry a third positional
    argument, which `*args` can -- a validator behind a decorator is the
    ordinary way one arrives in this shape, and reading its arity as two
    would withhold the finder from something able to use it.
    """
    seen = []

    def validator(*args):
        seen.append(len(args))
        return True

    _bundle(validator, auths=[_surface_date_arm()]).annotate_input(NAMED_DATE_QUERY)

    assert seen == [3]


def test_the_text_accessor_reads_the_column_its_metadata_names():
    """A neighbouring defect in the same class, on a leaf, with no bundle.

    `get_text` passed `metadata.text_col` -- a column *name* -- where
    `AnnotationsRowAccessor.get_col_value` takes a column *type*, and the two
    coincide only for the default metadata. An authority built with any other
    `text_col` could not read its own rows back.
    """
    auth_anns = _auth_annotations(_surface_date_arm(), NAMED_DATE_QUERY)

    texts = [auth_anns.get_text(row) for _, row in auth_anns.df.iterrows()]

    assert texts == ["07", "04", "1776"]


def test_an_unrecognized_column_reads_as_missing_rather_than_raising():
    """`DerivedFieldGroups.get_col_value` documents a missing value; it raised.

    Any column type that is neither a key column nor one of the three derived
    ones reached the derived lookup, which left its column name unbound and
    raised `UnboundLocalError` instead of returning what the caller asked to
    be told for an unknown column.
    """
    auth_anns = _auth_annotations(_date_arm(), BUNDLE_QUERY)

    row = auth_anns.df.iloc[0]

    assert auth_anns.colval("no_such_column", row) is None


def test_a_bundle_orders_a_match_by_the_column_its_finder_names():
    """The merge reads each match's position through its finder too.

    The same assumption in the other place it was made: the merge read the
    start position through the member it asked, which is not the authority
    that wrote the row once that member is itself a bundle.
    """
    leaf = dk_auth.RegexAuthority(
        "date",
        DATE_PATTERN,
        auth_anns_builder=dk_auth.AuthorityAnnotationsBuilder(
            metadata=dk_auth.AuthorityAnnotationsMetaData(start_pos_col="begin")
        ),
    )
    outer = _bundle(_Vocabulary(), auths=[_bundle(auths=[leaf])])

    found = list(outer.find_matches_with_finders(dk_annots.AnnotatedText(BUNDLE_QUERY)))

    assert [finder for finder, _ in found] == [leaf]
    assert [rows[0]["begin"] for _, rows in found] == [36], "where the date starts"

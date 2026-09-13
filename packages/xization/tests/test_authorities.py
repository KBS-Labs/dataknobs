import re

import pandas as pd

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

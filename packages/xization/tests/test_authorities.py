import re

import dataknobs_xization.authorities as dk_auth


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
            year = int(atts["year"])
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

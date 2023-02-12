import re
import dataknobs.xization.authorities as dk_auth


def text_authority_annotations_metadata():
    md = dk_auth.AuthorityAnnotationsMetaData()
    assert md.auth_id_col == dk_auth.KEY_AUTH_ID_COL


def test_regex_authority_no_groups():
    r = re.compile(r'\d{2}/\d{2}/\d{4}')
    rauth = dk_auth.RegexAuthority(
        'date', r, lambda x,y: f'{y}:{x}'
    )
    anns = rauth.annotate_text('abc 07/04/1776 xyz')
    assert len(anns.df) == 1
    assert anns.df['text'].to_list() == ['07/04/1776']


def test_regex_authority_no_name_groups():
    r = re.compile(r'(\d{2})/(\d{2})/(\d{4})')
    rauth = dk_auth.RegexAuthority(
        'date', r, lambda x,y: f'{y}:{x}'
    )
    anns = rauth.annotate_text('abc 07/04/1776 xyz')
    assert len(anns.df) == 3
    assert anns.df['text'].to_list() == ['07', '04', '1776']
    assert anns.df['date_field'].to_list() == [1, 2, 3]


def test_regex_authority_named_groups():
    r = re.compile(r'(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4}|\d{2})')
    rauth = dk_auth.RegexAuthority(
        'date', r, lambda x,y: f'{y}:{x}'
    )
    anns = rauth.annotate_text('abc 07/04/1776 xyz')
    assert len(anns.df) == 3
    assert anns.df['text'].to_list() == ['07', '04', '1776']
    assert anns.df['date_field'].to_list() == ['day', 'month', 'year']

"""``get_lexical_variations`` keeps its released output now that it is re-expressed.

The function was released in ``dataknobs-xization`` 2.2.1, so its output is a
contract. It is now written over :mod:`dataknobs_common.text`'s rules plus its
own, and these tests compare it, call for call, with the implementation it
replaced -- frozen below as a reference, verbatim, patterns and helpers
included. The reference is the specification; do not edit it to make a
comparison pass.

The grid covers every pair of the twelve boolean flags in all four
combinations (the defaults, each flag flipped alone, every pair flipped
together, and everything flipped), crossed with the two sequence parameters at
their default, empty and a custom value and the two length thresholds at their
defaults and changed.
"""

import itertools
import re
from typing import Any, Sequence, Set

import pytest

import dataknobs_xization.normalize as dk_norm

# ---- reference implementation: dataknobs-xization's normalize.py as released ----
# fmt: off
# squash whitespace: to collapse consecutive whitespace to a single space by
#    x.sub(' ', text)
SQUASH_WS_RE = re.compile(r"\s+")


# to identify strings with any symbols by
#    x.search(text)
ALL_SYMBOLS_RE = re.compile(r"[^\w\s]+")


# camelcase LU: to split between consecutive lower and upper chars by
#    x.sub(r'\1 \2', text)
CAMELCASE_LU_RE = re.compile(r"([a-z]+)([A-Z])")


# camelcase UL: to split between consecutive upper and uppler lower chars by
#    x.sub(r'\1 \2', text)
CAMELCASE_UL_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")


# non-embedded symbols: those without a word char on both sides by
#    x.sub('', text)
NON_EMBEDDED_WORD_SYMS_RE = re.compile(r"((?<!\w)[^\w\s]+)|([^\w\s]+(?!\w))")


# embedded symbols: to drop embedded symbols by
#    x.sub('', text)
EMBEDDED_SYMS_RE = re.compile(r"(?<=\w)[^\w\s]+(?=\w)")


# hyphen-slash: to split between an embedded hyphen and/or slash by
#    x.split(text)
HYPHEN_SLASH_RE = re.compile(r"(?<=\w)[\-\/ ](?=\w)")


# hyphen-only: to split between an embedded hyphen by
#    x.split(text)
HYPHEN_ONLY_RE = re.compile(r"(?<=\w)[\- ](?=\w)")


# slash-only: to split between an embedded slash by
#    x.split(text)
SLASH_ONLY_RE = re.compile(r"(?<=\w)\/(?=\w)")


# parenthetical expressions: to drop parenthetical expressions by
#    x.sub('', text)
# The class is negated rather than the quantifier made lazy: both stop the
# match at the first ')', but `[^)]*` also cannot span one, so two
# parentheticals in a string are two matches instead of one match over
# everything between them.
PARENTHETICAL_RE = re.compile(r"\([^)]*\)")


# ampersand: to replace an ampersand with " and " by
#    x.sub(' and ', text)
AMPERSAND_RE = re.compile(r"\s*\&\s*")


def expand_camelcase_fn(text: str) -> str:
    """Expand both "lU" and "UUl" camelcasing to "l U" and "U Ul" """
    text = CAMELCASE_LU_RE.sub(r"\1 \2", text)
    return CAMELCASE_UL_RE.sub(r"\1 \2", text)


def drop_non_embedded_symbols_fn(text: str, repl: str = "") -> str:
    """Drop symbols not embedded within word characters"""
    return NON_EMBEDDED_WORD_SYMS_RE.sub(repl, text)


def drop_embedded_symbols_fn(text: str, repl: str = "") -> str:
    """Drop symbols embedded within word characters"""
    return EMBEDDED_SYMS_RE.sub(repl, text)


def get_hyphen_slash_expansions_fn(
    text: str,
    subs: Sequence[str] = ("-", " ", ""),
    add_self: bool = True,
    do_split: bool = True,
    min_split_token_len: int = 2,
    hyphen_slash_re: re.Pattern[str] = HYPHEN_SLASH_RE,
) -> Set[str]:
    """Given text with words that may or may not appear as hyphenated or with a
    slash, return the set potential variations:
        - the text as-is (add_self)
        - with a hyphen between all words (if '-' in subs)
        - with a space between all words (if ' ' in subs)
        - with all words squashed together (empty string between if '' in subs)
        - with each word separately (do_split as long as min_split_token_len is
              met for all tokens)

    Note:
        * To add a variation with a slash, add '/' to subs.
        * To not add any variations with symbols, leave them out of subs
          and don't add self.

    Args:
        text: The hyphen-worthy snippet of text, either already
            hyphenated or with a slash or space delimited.
        subs: A string of characters or list of strings to insert between
            tokens.
        add_self: True to include the text itself in the result.
        do_split: True to add split tokens separately.
        min_split_token_len: If any of the split tokens fail
            to meet the min token length, don't add any of the splits.
        hyphen_slash_re: The regex to identify hyphen/slash to expand.

    Returns:
        The set of text variations.
    """
    variations = {text} if add_self else set()
    if subs is not None and len(subs) > 0:
        # create variant with all <s>'s
        for s in subs:
            variations.add(HYPHEN_SLASH_RE.sub(s, text))
    if do_split:
        # add each word separately
        tokens = set(hyphen_slash_re.split(text))
        if not max(len(t) < min_split_token_len for t in tokens):
            variations.update(tokens)
    return variations


def drop_parentheticals_fn(text: str) -> str:
    """Drop parenthetical expressions from the text."""
    return PARENTHETICAL_RE.sub("", text)


def expand_ampersand_fn(text: str) -> str:
    """Replace '&' with ' and '."""
    return AMPERSAND_RE.sub(" and ", text)


def get_lexical_variations(
    text: str,
    include_self: bool = True,
    expand_camelcase: bool = True,
    drop_non_embedded_symbols: bool = True,
    drop_embedded_symbols: bool = True,
    spacify_embedded_symbols: bool = False,
    do_hyphen_expansion: bool = True,
    hyphen_subs: Sequence[str] = (" ", ""),
    do_hyphen_split: bool = True,
    min_hyphen_split_token_len: int = 2,
    do_slash_expansion: bool = True,
    slash_subs: Sequence[str] = (" ", " or "),
    do_slash_split: bool = True,
    min_slash_split_token_len: int = 1,
    drop_parentheticals: bool = True,
    expand_ampersands: bool = True,
    add_eng_plurals: bool = True,
) -> Set[str]:
    """Get all variations for the text (including the text itself).

    Args:
        text: The text to generate variations for.
        include_self: True to include the original text in the result.
        expand_camelcase: True to expand camelCase text.
        drop_non_embedded_symbols: True to drop symbols not embedded in words.
        drop_embedded_symbols: True to drop symbols embedded in words.
        spacify_embedded_symbols: True to replace embedded symbols with spaces.
        do_hyphen_expansion: True to expand hyphenated text.
        hyphen_subs: List of strings to substitute for hyphens.
        do_hyphen_split: True to split on hyphens.
        min_hyphen_split_token_len: Minimum token length for hyphen splits.
        do_slash_expansion: True to expand slashes.
        slash_subs: List of strings to substitute for slashes.
        do_slash_split: True to split on slashes.
        min_slash_split_token_len: Minimum token length for slash splits.
        drop_parentheticals: True to drop parenthetical expressions.
        expand_ampersands: True to expand ampersands to ' and '.
        add_eng_plurals: True to add English plural forms.

    Returns:
        The set of all text variations.
    """
    variations = {text} if include_self else set()
    if expand_camelcase:
        variations.add(expand_camelcase_fn(text))
    if drop_non_embedded_symbols:
        variations.add(drop_non_embedded_symbols_fn(text))
    if drop_embedded_symbols:
        variations.add(drop_embedded_symbols_fn(text))
    if spacify_embedded_symbols:
        variations.add(drop_embedded_symbols_fn(text, " "))
    if (
        do_hyphen_expansion and hyphen_subs is not None and len(hyphen_subs) > 0
    ) or do_hyphen_split:
        variations.update(
            get_hyphen_slash_expansions_fn(
                text,
                subs=hyphen_subs,
                add_self=False,
                do_split=do_hyphen_split,
                min_split_token_len=min_hyphen_split_token_len,
            )
        )
    if (do_slash_expansion and slash_subs is not None and len(slash_subs) > 0) or do_slash_split:
        variations.update(
            get_hyphen_slash_expansions_fn(
                text,
                subs=slash_subs,
                add_self=False,
                do_split=do_slash_split,
                min_split_token_len=min_slash_split_token_len,
            )
        )
    if drop_parentheticals:
        variations.add(drop_parentheticals_fn(text))
    if expand_ampersands:
        variations.add(expand_ampersand_fn(text))
    if add_eng_plurals:
        # (released: a comment here asked for a better pluralizer)
        plurals = {f"{v}s" for v in variations}
        variations.update(plurals)
    return variations


# fmt: on
# ---- end of reference implementation ----

_FLAGS = (
    "include_self",
    "expand_camelcase",
    "drop_non_embedded_symbols",
    "drop_embedded_symbols",
    "spacify_embedded_symbols",
    "do_hyphen_expansion",
    "do_hyphen_split",
    "do_slash_expansion",
    "do_slash_split",
    "drop_parentheticals",
    "expand_ampersands",
    "add_eng_plurals",
)
_DEFAULTS = {
    "include_self": True,
    "expand_camelcase": True,
    "drop_non_embedded_symbols": True,
    "drop_embedded_symbols": True,
    "spacify_embedded_symbols": False,
    "do_hyphen_expansion": True,
    "do_hyphen_split": True,
    "do_slash_expansion": True,
    "do_slash_split": True,
    "drop_parentheticals": True,
    "expand_ampersands": True,
    "add_eng_plurals": True,
}

_CORPUS = (
    "",
    "x",
    "ThisIsATest",
    "TheNLPToolkit",
    "BBQSauce",
    "   .te-st. .this. .   .#$now..   ",
    "a#b",
    "gluten-free",
    "jam/jelly",
    "jam / jelly",
    "a-b/c d",
    "Fruit & Veg (fresh)",
    "AI (Artificial Intelligence) and ML (Machine Learning)",
    "glasses",
    "Glasses (x) y",
    "golden retriever",
    "(only)",
    "US-1/2",
    "e-mail/web-site",
)


def _flag_rows() -> list[dict[str, bool]]:
    """Defaults, each flag flipped, each pair flipped, and all flipped."""
    rows = [dict(_DEFAULTS)]
    for size in (1, 2):
        for flipped in itertools.combinations(_FLAGS, size):
            row = dict(_DEFAULTS)
            for name in flipped:
                row[name] = not row[name]
            rows.append(row)
    rows.append({name: not value for name, value in _DEFAULTS.items()})
    return rows


_SEQUENCES: list[dict[str, Any]] = [
    {"hyphen_subs": hyphen, "slash_subs": slash}
    for hyphen in ((" ", ""), (), ("/",))
    for slash in ((" ", " or "), (), ("-",))
]
_LENGTHS: list[dict[str, Any]] = [
    {},
    {"min_hyphen_split_token_len": 1, "min_slash_split_token_len": 3},
]


def test_the_grid_covers_every_pair_of_flags() -> None:
    """Every pair of flags is seen in all four combinations of values."""
    rows = _flag_rows()
    for first, second in itertools.combinations(_FLAGS, 2):
        seen = {(row[first], row[second]) for row in rows}
        assert len(seen) == 4, (first, second)


@pytest.mark.parametrize("text", _CORPUS)
def test_the_output_is_the_released_output(text: str) -> None:
    for flags in _flag_rows():
        for sequences in _SEQUENCES:
            for lengths in _LENGTHS:
                kwargs = {**flags, **sequences, **lengths}
                assert dk_norm.get_lexical_variations(text, **kwargs) == (
                    get_lexical_variations(text, **kwargs)
                ), kwargs


@pytest.mark.parametrize("text", _CORPUS)
def test_the_released_helpers_are_unchanged(text: str) -> None:
    """The public helpers the function used, called the way it called them."""
    assert dk_norm.expand_camelcase_fn(text) == expand_camelcase_fn(text)
    assert dk_norm.drop_parentheticals_fn(text) == drop_parentheticals_fn(text)
    assert dk_norm.expand_ampersand_fn(text) == expand_ampersand_fn(text)
    for subs in ((" ", ""), (" ", " or "), ("-", " ", ""), ()):
        for add_self, do_split in itertools.product((True, False), repeat=2):
            assert dk_norm.get_hyphen_slash_expansions_fn(
                text, subs=subs, add_self=add_self, do_split=do_split
            ) == get_hyphen_slash_expansions_fn(
                text, subs=subs, add_self=add_self, do_split=do_split
            )

"""Text normalization utilities and regular expressions.

Provides functions and regex patterns for normalizing text including
whitespace handling, camelCase splitting, and symbol processing.
"""

import math
import re
from itertools import product
from typing import List, Set

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
PARENTHETICAL_RE = re.compile(r"\(.*\)")


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
    subs: List[str] = ("-", " ", ""),
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
    hyphen_subs: List[str] = (" ", ""),
    do_hyphen_split: bool = True,
    min_hyphen_split_token_len: int = 2,
    do_slash_expansion: bool = True,
    slash_subs: List[str] = (" ", " or "),
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
        # TODO: Use a better pluralizer
        plurals = {f"{v}s" for v in variations}
        variations.update(plurals)
    return variations


def int_to_en(num: int) -> str:
    d = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }
    k = 1000
    m = k * 1000
    b = m * 1000
    t = b * 1000

    if not isinstance(num, int):
        return num

    if num < 0:
        return "negative " + int_to_en(abs(num))

    if num < 20:
        return d[num]

    if num < 100:
        if num % 10 == 0:
            return d[num]
        else:
            return d[num // 10 * 10] + " " + d[num % 10]

    if num < k:
        if num % 100 == 0:
            return d[num // 100] + " hundred"
        else:
            return d[num // 100] + " hundred and " + int_to_en(num % 100)

    if num < m:
        if num % k == 0:
            return int_to_en(num // k) + " thousand"
        else:
            return int_to_en(num // k) + " thousand " + int_to_en(num % k)

    if num < b:
        if (num % m) == 0:
            return int_to_en(num // m) + " million"
        else:
            return int_to_en(num // m) + " million " + int_to_en(num % m)

    if num < t:
        if (num % b) == 0:
            return int_to_en(num // b) + " billion"
        else:
            return int_to_en(num // b) + " billion " + int_to_en(num % b)

    if num % t == 0:
        return int_to_en(num // t) + " trillion"
    else:
        return int_to_en(num // t) + " trillion " + int_to_en(num % t)

    # num is too large
    return str(num)


def zero_pad_variations(
    val: int,
    min_zpad_len: int,
    max_zpad_len: int,
) -> Set[str]:
    """Get (only) zero-padded variations of the given value from min (inclusive)
    to max (exclusive) zero-pad lengths.

    Examples:
        >>> from dataknobs_xization.normalize import zero_pad_variations
        >>> zero_pad_variations(9, 2, 4)
        {'09', '009'}
        >>> zero_pad_variations(90, 2, 4)
        {'090'}
        >>> zero_pad_variations(90, 2, 3)
        set()
        >>> zero_pad_variations(3, 0, 5)
        {'03', '003', '0003'}

    Args:
        val: The integer value to zero-pad.
        min_zpad_len: The minimum zero-padded string length (inclusive).
        max_zpad_len: The maximum zero-padded string length (exclusive).

    Returns:
        The set of all requested zero-padded number strings.
    """
    return {
        f"{val:0{zpad}d}"
        for zpad in range(
            max(min_zpad_len, math.ceil(math.log10(val)) + 1 if val > 0 else 1), max_zpad_len
        )
    }


def month_day_variations_fn(
    month_or_day: int,
    do_int_to_en: bool = False,
) -> Set[str]:
    """Get the variations for a month or day number, including the number
    itself as a string, a 2-digit zero-padded form of the number, and
    (optionally) english word for the number.

    Args:
        month_or_day: The month or day for which to get variations.
        do_int_to_en: Optionally include the english word for the number.

    Returns:
        The set of variations for the value.
    """
    result = zero_pad_variations(month_or_day, 2, 3)
    result.add(str(month_or_day))
    if do_int_to_en:
        result.add(int_to_en(month_or_day))
    return result


def year_variations_fn(
    year: int,
    min_year: int = 0,
    max_year: int = 9999,
    do_int_to_en_below_100: bool = False,
    numeric_only: bool = False,
) -> Set[str]:
    """Convert a year to various text representations.

    Generates variations including:
        * "1999" (numeric)
        * Long text: "one thousand, nine hundred and ninety nine"
        * Short text: "nineteen [hundred and] ninety nine"

    Args:
        year: The year value to convert.
        min_year: Minimum year to process (inclusive).
        max_year: Maximum year to process (inclusive).
        do_int_to_en_below_100: True to convert years below 100 to English text.
        numeric_only: True to return only numeric variations.

    Returns:
        The set of year variations.
    """
    variations = {str(year)}

    if year < min_year or year > max_year:
        return variations

    # one thousand, nine hundred and ninety nine
    if not numeric_only and (do_int_to_en_below_100 or year >= 100):
        variations.add(int_to_en(year))

    # nineteen ninety five
    century = year // 100
    remainder = year % 100
    remainder_text = int_to_en(remainder)

    variations.update(zero_pad_variations(remainder, 2, 3))

    if century > 0:
        remainder_texts = []
        if remainder > 0:
            if remainder < 10:
                if not numeric_only:
                    remainder_texts.append(f" oh {remainder_text}")
                remainder_texts.append(f" 0{remainder}")
            else:
                if not numeric_only:
                    remainder_texts.append(f" {remainder_text}")
                remainder_texts.append(f" {remainder}")
            if not numeric_only:
                remainder_texts.append(f" and {remainder_text}")

        century_text = int_to_en(century)
        scales = ["", century_text]
        if century % 10 == 0:
            mil_text = int_to_en(century // 10)
            scales.append(f"{mil_text} thousand")
        else:
            scales.append(f"{century_text} hundred")

        def clean_up(s):
            s = s.strip()
            if s.startswith("and "):
                s = s[4:]
            return s

        variations.update({clean_up("".join(v)) for v in product(scales, remainder_texts)})

    return variations


def replace_smart_quotes_fn(text: str) -> str:
    """Replace "smart" quotes with their ascii version."""
    return (
        text.replace(
            "\u201c",
            '"',  # left double quote U+201C
        )
        .replace(
            "\u201d",
            '"',  # right double quote U+201D
        )
        .replace(
            "\u2018",
            "'",  # left single quote U+2018
        )
        .replace(
            "\u2019",
            "'",  # right single quote U+2019
        )
    )


def basic_normalization_fn(
    text: str,
    lowercase: bool = True,
    expand_camelcase: bool = True,
    simplify_quote_chars: bool = True,
    drop_non_embedded_symbols: bool = False,
    spacify_embedded_symbols: bool = False,
    drop_embedded_symbols: bool = False,
    squash_whitespace: bool = False,
    do_all: bool = False,
) -> str:
    """Basic normalization functions include:
        * lowercasing [default]
        * expanding camelcase [default]
        * replacing "smart" quotes and apostrophes with ascii versions [default]
        * dropping non_embedded symbols [optional]
        * replacing embedded symbols with a space [takes precedence over dropping unless do_all]
        * or dropping embedded symbols [optional]
        * collapsing multiple spaces and stripping spaces from ends [optional]

    Args:
        text: The text to normalize.
        lowercase: True to convert to lowercase.
        expand_camelcase: True to expand camelCase text.
        simplify_quote_chars: True to replace smart quotes with ASCII quotes.
        drop_non_embedded_symbols: True to drop symbols not embedded in words.
        spacify_embedded_symbols: True to replace embedded symbols with spaces.
        drop_embedded_symbols: True to drop embedded symbols.
        squash_whitespace: True to collapse whitespace and strip ends.
        do_all: True to apply all normalization steps.

    Returns:
        The normalized text.
    """
    # NOTE: do this before changing case
    if expand_camelcase or do_all:
        text = expand_camelcase_fn(text)

    if lowercase or do_all:
        text = text.lower()
    if (drop_non_embedded_symbols and drop_embedded_symbols) or do_all:
        text = re.sub(r"[^\w\s]+", "", text)
    elif drop_non_embedded_symbols:
        text = drop_non_embedded_symbols_fn(text)
    elif spacify_embedded_symbols:
        text = drop_embedded_symbols_fn(text, " ")
    elif drop_embedded_symbols:
        text = drop_embedded_symbols_fn(text)

    # NOTE: do this after dropping (only some) symbols
    if simplify_quote_chars and (not drop_non_embedded_symbols or not drop_embedded_symbols):
        # NOTE: It only makes sense to do this if we're keeping symbols
        text = replace_smart_quotes_fn(text)

    # NOTE: do this last
    if squash_whitespace or do_all:
        text = re.sub(r"\s+", " ", text).strip()
    return text

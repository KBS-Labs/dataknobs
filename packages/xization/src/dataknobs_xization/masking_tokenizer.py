"""Character-level text feature extraction and tokenization.

Provides abstract classes for extracting character-level features from text,
building DataFrames with character features for masking and tokenization.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

import dataknobs_structures.document as dk_doc
from dataknobs_utils import emoji_utils


class CharacterFeatures(ABC):
    """Class representing features of text as a dataframe with each character
    as a row and columns representing character features.
    """

    def __init__(self, doctext: Union[dk_doc.Text, str], roll_padding: int = 0):
        """Initialize with the text to tokenize.

        Args:
            doctext: The text to tokenize (or dk_doc.Text with its metadata).
            roll_padding: The number of pad characters added to each end of
                the text.
        """
        self._doctext = doctext
        self._roll_padding = roll_padding
        self._padded_text = None

    @property
    def cdf(self) -> pd.DataFrame:
        """The character dataframe with each padded text character as a row."""
        raise NotImplementedError

    @property
    def doctext(self) -> dk_doc.Text:
        if isinstance(self._doctext, str):
            self._doctext = dk_doc.Text(self._doctext, None)
        return self._doctext

    @property
    def text_col(self) -> str:
        """The name of the cdf column holding the text characters."""
        return self.doctext.text_label

    @property
    def text(self) -> str:
        """The text string."""
        return self.doctext.text

    @property
    def text_id(self) -> Any:
        """The ID of the text."""
        return self.doctext.text_id

    @abstractmethod
    def build_first_token(
        self,
        normalize_fn: Callable[[str], str],
    ) -> "Token":
        """Build the first token as the start of tokenization.

        Args:
            normalize_fn: A function to normalize a raw text term or any
                of its variations. If None, then the identity function is used.

        Returns:
            The first text token.
        """
        raise NotImplementedError

    @property
    def roll_padding(self) -> int:
        """The number of pad characters added to each end of the text."""
        return self._roll_padding

    @property
    def padded_text(self) -> str:
        """The text with padding included."""
        if self._padded_text is None:
            padding = " " * self.roll_padding
            self._padded_text = padding + self.text + padding
        return self._padded_text

    def get_tokens(
        self,
        normalize_fn: Callable[[str], str] = lambda x: x,
    ) -> List["Token"]:
        """Get all token instances using the given normalize function.

        Args:
            normalize_fn: The normalization function (default=identity fn).

        Returns:
            A list of token instances.
        """
        token = self.build_first_token(normalize_fn)
        tokens = []
        while token is not None:
            tokens.append(token)
            token = token.next_token
        return tokens


class TextFeatures(CharacterFeatures):
    """Extracts text-specific character features for tokenization.

    Extends CharacterFeatures to provide text tokenization with support for
    camelCase splitting, character type features (alpha, digit, upper, lower),
    and emoji handling. Builds a character DataFrame with features for
    token boundary detection.
    """

    def __init__(
        self,
        doctext: Union[dk_doc.Text, str],
        split_camelcase: bool = True,
        mark_alpha: bool = False,
        mark_digit: bool = False,
        mark_upper: bool = False,
        mark_lower: bool = False,
        emoji_data: emoji_utils.EmojiData = None,
    ):
        """Initialize with text tokenization parameters.

        Note:
            If emoji_data is non-null:
                * Then emojis will be treated as text (instead of as non-text)
                * If split_camelcase is True,
                    * then each emoji will be in its own token
                    * otherwise, each sequence of (adjacent) emojis will be treated
                      as a single token.

        Args:
            doctext: The text to tokenize with its metadata.
            split_camelcase: True to mark camel-case features.
            mark_alpha: True to mark alpha features (separate from alnum).
            mark_digit: True to mark digit features (separate from alnum).
            mark_upper: True to mark upper features (auto-included for
                camel-case).
            mark_lower: True to mark lower features (auto-included for
                camel-case).
            emoji_data: An EmojiData instance to mark emoji BIO features.
        """
        # NOTE: roll_padding is determined by "roll" feature needs. Currently 1.
        super().__init__(doctext, roll_padding=1)
        self.split_camelcase = split_camelcase
        self._cdf = self._build_character_dataframe(
            split_camelcase,
            mark_alpha,
            mark_digit,
            mark_upper,
            mark_lower,
            emoji_data,
        )

    @property
    def cdf(self) -> pd.DataFrame:
        """The character dataframe with each padded text character as a row."""
        return self._cdf

    def build_first_token(
        self,
        normalize_fn: Callable[[str], str],
    ) -> "Token":
        """Build the first token as the start of tokenization.

        Args:
            normalize_fn: A function to normalize a raw text term or any
                of its variations. If None, then the identity function is used.

        Returns:
            The first text token.
        """
        token_mask = (
            DualTokenMask(
                self,
                self.cdf["tok_start"],
                self.cdf["tok_end"],
            )
            if self.split_camelcase
            else SimpleTokenMask(self, self.cdf["alnum"])
        )
        token = Token(token_mask, normalize_fn=normalize_fn)
        return token

    def _build_character_dataframe(
        self,
        split_camelcase,
        mark_alpha,
        mark_digit,
        mark_upper,
        mark_lower,
        emoji_data,
    ):
        if split_camelcase:
            mark_upper = True
            mark_lower = True
        cdf = pd.DataFrame({self.text_col: list(self.padded_text)})
        if mark_alpha:
            cdf["alpha"] = cdf[self.text_col].str.isalpha()
        if mark_digit:
            cdf["digit"] = cdf[self.text_col].str.isdigit()
        cdf["alnum"] = cdf[self.text_col].str.isalnum()
        cdf["space"] = cdf[self.text_col].str.isspace()
        if mark_upper:
            cdf["upper"] = cdf[self.text_col].str.isupper()
        if mark_lower:
            cdf["lower"] = cdf[self.text_col].str.islower()
        cdf["sym"] = ~(cdf["alnum"] | cdf["space"])
        if split_camelcase:
            cdf["cc1"] = np.roll(cdf["lower"], 1) & cdf["upper"]
            cdf["cc2"] = (  # Mark 2nd U of UUl
                np.roll(cdf["upper"], 1) & cdf["upper"] & np.roll(cdf["lower"], -1)
            )
        # NOTE: tok_start and tok_end are both INCLUSIVE
        cdf["tok_start"] = (  # mark a char following a non-char
            cdf["alnum"] & ~np.roll(cdf["alnum"], 1)
        )
        cdf["tok_end"] = (  # mark a char followed by a non-char
            cdf["alnum"] & ~np.roll(cdf["alnum"], -1)
        )
        if split_camelcase:
            cdf["tok_start"] = cdf["tok_start"] | cdf["cc1"] | cdf["cc2"]
            cdf["tok_end"] = cdf["tok_end"] | np.roll(cdf["cc1"] | cdf["cc2"], -1)
        if emoji_data is not None:
            cdf["emoji"] = pd.Series(list(emoji_data.emoji_bio(self.padded_text)))
            if split_camelcase:
                # Splitting camelcase includes splitting distinct emojis
                cdf["tok_start"] |= cdf["emoji"] == "B"
                cdf["tok_end"] |= (  # mark an 'I' followed by not 'I'
                    (cdf["emoji"] == "I") & np.roll(cdf["emoji"] != "I", -1)
                )
                cdf["tok_end"] |= (  # mark an 'B' followed by not 'I'
                    (cdf["emoji"] == "B") & np.roll(cdf["emoji"] != "I", -1)
                )
            else:
                # Not splitting camelcase keeps consecutive emojis together
                cdf["alnum"] |= cdf["emoji"] != "O"
        return cdf


class CharacterInputFeatures(CharacterFeatures):
    """A wrapper that starts with a pre-built character features dataframe."""

    def __init__(
        self,
        cdf: pd.DataFrame,
        token_mask: "TokenMask",
        doctext: Union[dk_doc.Text, str],
        roll_padding: int = 0,
    ):
        super().__init__(doctext, roll_padding=roll_padding)
        self._cdf = cdf
        self._token_mask = token_mask

    @property
    def cdf(self) -> pd.DataFrame:
        """The character dataframe with each padded text character as a row."""
        return self._cdf

    def build_first_token(
        self,
        normalize_fn: Callable[[str], str] = None,
    ) -> "Token":
        """Build the first token as the start of tokenization.

        Args:
            normalize_fn: A function to normalize a raw text term or any
                of its variations. If None, then the identity function is used.

        Returns:
            The first text token.
        """
        token = Token(self._token_mask, normalize_fn=normalize_fn)
        return token


class TokenLoc:
    """Simple structure holding information about a token's location."""

    def __init__(
        self,
        start_loc: int,
        end_loc: int,
        token_num: int = None,
        start_incl: bool = True,
        end_incl: bool = False,
    ):
        """Initialize with the available information.

        Args:
            start_loc: The starting location of the token.
            end_loc: The ending location of the token.
            token_num: The position of the token within its text string.
            start_incl: True if start_loc is part of the token; otherwise
                start_loc+1 is part of the token.
            end_incl: True if end_loc is part of the token; otherwise
                end_loc-1 is part of the token.
        """
        self._start_loc = start_loc
        self._end_loc = end_loc
        self._token_num = token_num
        self._start_incl = int(start_incl)
        self._end_incl = int(end_incl)

    def __repr__(self) -> str:
        token_num = f"#{self._token_num}" if self._token_num >= 0 else ""

        def inclc(incl, left):
            if incl:
                return "[" if left else "]"
            else:
                return "(" if left else ")"

        return f"{token_num}{inclc(self._start_incl, True)}{self._start_loc}:{self._end_loc}{inclc(self._end_incl, False)}"

    def _incl_offset(self, wanted_incl: bool, current_incl: int) -> int:
        """Get the inclusivity offset based on what is wanted versus what is."""
        return int(wanted_incl) - current_incl

    @property
    def len(self) -> int:
        """Get the length of the token at this location."""
        return self.end_loc_excl - self.start_loc_incl

    @property
    def start_loc_incl(self) -> int:
        """Get the inclusive start location."""
        return self._start_loc + self._incl_offset(True, self._start_incl)

    @property
    def start_loc_excl(self) -> int:
        """Get the exclusive start location."""
        return self._start_loc + self._incl_offset(False, self._start_incl)

    @property
    def end_loc_incl(self) -> int:
        """Get the inclusive end location."""
        return self._end_loc - self._incl_offset(True, self._end_incl)

    @property
    def end_loc_excl(self) -> int:
        """Get the exclusive end location."""
        return self._end_loc - self._incl_offset(False, self._end_incl)

    @property
    def token_num(self) -> int:
        """Get the token's position within its text string, or -1 if unknown."""
        return self._token_num if self._token_num is not None else -1


class TokenMask(ABC):
    """A class for accessing text characters through feature masks."""

    def __init__(self, text_features: CharacterFeatures):
        self.text_features = text_features
        self.pad = self.text_features.roll_padding
        self.max_ploc = max(self.text_features.cdf.index)

    def _get_next_start(self, ref_ploc: int, token_mask: pd.Series) -> int:
        """Given the end of a prior token or possible start of the next, get
        the "next" start token's starting ploc. If there is no subsequent
        token, then return None.

        Args:
            ref_ploc: The end ploc of the prior token or start of string.
            token_mask: The token mask to use.

        Returns:
            The ploc of the start of the next token or None.
        """
        # if not at end of string or already at the start of a token, increment
        if ref_ploc > self.max_ploc:
            ref_ploc = None  # At end of string
        elif not token_mask.loc[ref_ploc]:
            next_ploc = increment(ref_ploc, token_mask)
            ref_ploc = next_ploc if next_ploc > ref_ploc else None
        return ref_ploc

    def get_padded_text(self, start_loc_incl: int, end_loc_excl: int) -> str:
        return self.text_features.padded_text[start_loc_incl:end_loc_excl]

    def get_text(self, token_loc: TokenLoc) -> str:
        """Get the text at the (padded) token location.

        Args:
            token_loc: The token location.

        Returns:
            The token text.
        """
        return self.get_padded_text(token_loc.start_loc_incl, token_loc.end_loc_excl)

    @abstractmethod
    def get_next_token_loc(self, ref_ploc: int, token_num: int = -1) -> TokenLoc:
        """Given the end of a prior token or possible start of the next, get
        the "next" token's location.
        If there is no subsequent token, then return None.

        Args:
            ref_ploc: The end ploc of the prior token or start of string.
            token_num: The token position within its text string.

        Returns:
            The TokenLoc of the next token or None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prev_token_loc(self, from_token_loc: TokenLoc) -> TokenLoc:
        """Get the previous token bounds before the given token start ploc.
        If there is no prior token, then return None.

        Args:
            from_token_loc: The token location after the result.

        Returns:
            The TokenLoc of the prior token or None.
        """
        raise NotImplementedError


def increment(start_loc: int, mask: pd.Series) -> Tuple[int, bool]:
    """Increment to the opposite True or False index location in the given mask
    from the given start index location.

    If the mask value at index (loc) start_idx is False, then find the
    index (loc) value where the mask is True. Then the mask values from
    start_idx (inclusive) to end_idx (exclusive) are all False.
    And vice-versa for if the mask value at start_idx is True.

    Args:
        start_loc: The start index location.
        mask: The boolean feature mask.

    Returns:
        end_loc Where the mask value is opposite that at start_loc.
        If unable to increment (e.g., at the end of the mask or no flips),
        then end_idx will equal start_idx.
    """
    end_loc = start_loc
    if start_loc in mask.index:
        m = mask.loc[start_loc:]
        end_iloc = m.argmin() if m.iloc[0] else m.argmax()
        if end_iloc > 0:
            end_loc = m.index[end_iloc]
    return end_loc


class SimpleTokenMask(TokenMask):
    """A mask where "in" tokens are ones and "out" are zeros."""

    def __init__(self, text_features: CharacterFeatures, token_mask: pd.Series):
        """Initialize with the text_features and token mask.

        Args:
            text_features: The text features to tokenize.
            token_mask: The token mask identifying token characters as True
                and characters between tokens as False.
        """
        super().__init__(text_features)
        self.token_mask = token_mask
        self.revmask = token_mask[::-1]

    def get_next_token_loc(self, ref_ploc: int, token_num: int = -1) -> TokenLoc:
        """Given the end of a prior token or possible start of the next, get
        the "next" token's location.
        If there is no subsequent token, then return None.

        Args:
            ref_ploc: The end ploc of the prior token or start of string.
            token_num: The token position within its text string.

        Returns:
            The TokenLoc of the next token or None.
        """
        result = None
        start_ploc = self._get_next_start(ref_ploc, self.token_mask)
        if start_ploc is not None:
            end_ploc = increment(start_ploc, self.token_mask)
            result = TokenLoc(start_ploc, end_ploc, token_num=token_num)
        return result

    def get_prev_token_loc(self, from_token_loc: TokenLoc) -> TokenLoc:
        """Get the previous token bounds before the given token start ploc.
        If there is no prior token, then return None.

        Args:
            from_token_loc: The token location after the result.

        Returns:
            The TokenLoc of the prior token or None.
        """
        result = None

        from_loc = from_token_loc.start_loc_excl
        start_loc = increment(increment(from_loc, self.revmask), self.revmask)
        if start_loc != from_loc:
            start_loc += 1
            end_loc = increment(start_loc, self.token_mask)
            result = TokenLoc(start_loc, end_loc, token_num=from_token_loc.token_num - 1)
        return result


class DualTokenMask(TokenMask):
    """A mask comprised of a mask for token starts and a mask for token ends."""

    def __init__(
        self,
        text_features: CharacterFeatures,
        start_mask: pd.Series,
        end_mask: pd.Series,
    ):
        super().__init__(text_features)
        self.start_mask = start_mask
        self.end_mask = end_mask
        # self.tok_starts = start_mask.index[start_mask]
        # self.tok_ends = end_mask.index[end_mask]
        self.tok_starts = start_mask
        self.tok_ends = end_mask
        self.rev_starts = self.tok_starts[::-1]
        self.rev_ends = self.tok_starts[::-1]

    def _get_token_end(self, start_ploc: int) -> int:
        return self._get_next_start(start_ploc, self.tok_ends) + 1

    def get_next_token_loc(self, ref_ploc: int, token_num: int = -1) -> TokenLoc:
        """Given the end of a prior token or possible start of the next, get
        the "next" token's location.
        If there is no subsequent token, then return None.

        Args:
            ref_ploc: The end ploc of the prior token or start of string.
            token_num: The token position within its text string.

        Returns:
            The TokenLoc of the next token or None.
        """
        result = None
        start_ploc = self._get_next_start(ref_ploc, self.tok_starts)
        if start_ploc is not None:
            end_ploc = self._get_token_end(start_ploc)
            result = TokenLoc(start_ploc, end_ploc, token_num=token_num)
        return result

    def get_prev_token_loc(self, from_token_loc: TokenLoc) -> TokenLoc:
        """Get the previous token bounds before the given token start ploc.
        If there is no prior token, then return None.

        Args:
            from_token_loc: The token location after the result.

        Returns:
            The TokenLoc of the prior token or None.
        """
        result = None
        from_loc = from_token_loc.start_loc_excl
        if from_loc > self.pad:
            start_loc = increment(from_loc, self.rev_starts)
            result = TokenLoc(
                start_loc, self._get_token_end(start_loc), token_num=from_token_loc.token_num + 1
            )
        return result


class Token:
    """A structure identifying the token start (inclusive) and end (exclusive)
    index positions according to text features mask.

    NOTE: The masks in CharacterFeatures include padding, which displaces indices
    relative to positions in the original text. In this class, padded indices
    are referred to with a "p".
    """

    def __init__(
        self,
        token_mask: TokenMask,
        token_loc: TokenLoc = None,
        start_ploc: int = 0,
        prev_token: "Token" = None,
        next_token: "Token" = None,
        normalize_fn: Callable[[str], str] = None,
    ):
        """Initialize the token pointer with text features and the token_mask.

        Args:
            token_mask: The token mask to use.
            token_loc: The (padded) token location, if known or None.
                If token_loc is None and start_ploc is 0, then this will be the
                first token of the text.
            start_ploc: The padded character index for the start of this
                token as an alternate to specifying token_loc. If start_ploc is not
                at a token character according to the token mask, then it will be
                auto-incremented to the next token.
            prev_token: The token prior to this token.
            next_token: The token following this token.
            normalize_fn: A function to normalize token text.
        """
        self.token_mask = token_mask
        self._next = next_token
        self._prev = prev_token
        self.normalize_fn = normalize_fn
        self._text = None
        self._norm_text = None
        self._pre_delims = None
        self._post_delims = None
        if token_loc is not None:
            self.token_loc = token_loc
        else:
            self.token_loc = self.token_mask.get_next_token_loc(
                max(start_ploc, token_mask.pad),
                token_num=0,
            )
        # If token_loc is None, the text is empty
        if self.token_loc is None:
            self._text = ""
            self.token_loc = TokenLoc(
                self.token_mask.max_ploc + 1,
                self.token_mask.max_ploc + 1,
                token_num=0,
            )
            self._pre_delims = ""
            self._post_delims = ""

    def __repr__(self) -> str:
        return f"Token({self.token_text}){self.token_loc}"

    @property
    def doctext(self) -> dk_doc.Text:
        """Get the text object with metadata."""
        return self.token_mask.text_features.doctext

    @property
    def full_text(self) -> str:
        """Get the full original text of which this token is a part."""
        return self.token_mask.text_features.text

    @property
    def text_id(self) -> Any:
        """Get the full text ID."""
        return self.token_mask.text_features.text_id

    @property
    def token_num(self) -> int:
        """Get the position of this token within its text string."""
        return self.token_loc.token_num

    @property
    def len(self) -> int:
        """Get the length of this token."""
        return self.token_loc.len

    @property
    def token_text(self) -> str:
        """Get this token's original text."""
        if self._text is None:
            self._text = self.token_mask.get_text(self.token_loc)
        return self._text

    @property
    def norm_text(self) -> str:
        """Get this token's normalized text."""
        if self._norm_text is None:
            self._norm_text = (
                self.normalize_fn(self.token_text)
                if self.normalize_fn is not None
                else self.token_text
            )
        return self._norm_text

    @property
    def start_pos(self) -> int:
        """Get this token's start (incl) position in the original text."""
        return self.token_loc.start_loc_incl - self.token_mask.pad

    @property
    def end_pos(self) -> int:
        """Get this token's end (excl) position in the original text."""
        return self.token_loc.end_loc_excl - self.token_mask.pad

    @property
    def token_pos(self) -> Tuple[int, int]:
        """Get the token start (incl) and end (excl) indexes in the original text."""
        return (self.start_pos, self.end_pos)

    @property
    def pre_delims(self) -> str:
        if self._pre_delims is None:
            delims = ""
            prev_loc = self.token_mask.get_prev_token_loc(self.token_loc)
            if prev_loc is not None:
                delims = self.token_mask.get_padded_text(
                    prev_loc.end_loc_excl, self.token_loc.start_loc_incl
                )
            self._pre_delims = delims
        return self._pre_delims

    @property
    def post_delims(self) -> str:
        if self._post_delims is None:
            delims = ""
            next_loc = self.token_mask.get_next_token_loc(
                self.token_loc.end_loc_excl,
            )
            if next_loc is not None:
                delims = self.token_mask.get_padded_text(
                    self.token_loc.end_loc_excl, next_loc.start_loc_incl
                )
            else:
                # There isn't a next token. Get remainder of text after tok.
                delims = self.token_mask.get_padded_text(
                    self.token_loc.end_loc_excl,
                    self.token_mask.max_ploc,
                )

            self._post_delims = delims
        return self._post_delims

    @property
    def next_token(self) -> "Token":
        if self._next is None:
            next_token_loc = self.token_mask.get_next_token_loc(
                self.token_loc.end_loc_excl,
                token_num=self.token_loc.token_num + 1,
            )
            if next_token_loc is not None:
                self._next = Token(
                    self.token_mask,
                    token_loc=next_token_loc,
                    prev_token=self,
                    normalize_fn=self.normalize_fn,
                )
        return self._next

    @property
    def prev_token(self) -> "Token":
        if self._prev is None:
            prev_token_loc = self.token_mask.get_prev_token_loc(self.token_loc)
            if prev_token_loc is not None:
                self._prev = Token(
                    self.token_mask,
                    token_loc=prev_token_loc,
                    next_token=self,
                    normalize_fn=self.normalize_fn,
                )
        return self._prev

    @property
    def first_token(self) -> "Token":
        """Get the first token for this token's input."""
        first = self
        while first.prev_token is not None:
            first = first.prev_token
        return first

    @property
    def last_token(self) -> "Token":
        """Get the last token for this token's input."""
        last = self
        while last.next_token is not None:
            last = last.next_token
        return last

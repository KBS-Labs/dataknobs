"""Lexical matching and token alignment for text processing.

Provides classes for lexical expansion, normalization, token alignment,
and pattern matching in text with support for variations and fuzzy matching.
"""

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import Any, Dict, List, Set, Union

import numpy as np
import pandas as pd

import dataknobs_structures.document as dk_doc
import dataknobs_xization.annotations as dk_anns
import dataknobs_xization.authorities as dk_auth
import dataknobs_xization.masking_tokenizer as dk_tok
from dataknobs_utils import emoji_utils


class LexicalExpander:
    """A class to expand and/or normalize original lexical input terms, to
    keep back-references from generated data to corresponding original input,
    and to build consistent tokens for lexical matching.
    """

    def __init__(
        self,
        variations_fn: Callable[[str], Set[str]] | None,
        normalize_fn: Callable[[str], str] | None,
        split_input_camelcase: bool = True,
        detect_emojis: bool = False,
    ):
        """Initialize with the given functions.

        Args:
            variations_fn: A function, f(t), to expand a raw input term to
                all of its variations (including itself if desired). If None, the
                default is to expand each term to itself.
            normalize_fn: A function to normalize a raw input term or any
                of its variations. If None, then the identity function is used.
            split_input_camelcase: True to split input camelcase tokens.
            detect_emojis: True to detect emojis. If split_input_camelcase,
                then adjacent emojis will also be split; otherwise, adjacent
                emojis will appear as a single token.
        """
        # ``is not None`` rather than truthiness: a callable that defines
        # ``__len__`` or ``__bool__`` can be falsy, and asking for truth here
        # discards it in favour of the default without a word.
        self.variations_fn = variations_fn if variations_fn is not None else lambda x: {x}
        self.normalize_fn = normalize_fn if normalize_fn is not None else lambda x: x
        self.split_input_camelcase = split_input_camelcase
        self.emoji_data = emoji_utils.load_emoji_data() if detect_emojis else None
        self.v2t: defaultdict[str, Set[Any]] = defaultdict(set)

    def __call__(self, term: Any, normalize: bool = True) -> Set[str]:
        """Get all variations of the original term.

        Args:
            term: The term whose variations to compute.
            normalize: True to normalize the resulting variations.

        Returns:
            All variations.
        """
        variations = self.variations_fn(term)
        if normalize:
            variations = {self.normalize_fn(v) for v in variations}
        # Add a mapping from each variation to its original term
        if variations is not None and len(variations) > 0:
            for variation in variations:
                self.v2t[variation].add(term)
        return variations

    def normalize(self, input_term: str) -> str:
        """Normalize the given input term or variation.

        Args:
            input_term: An input term to normalize.

        Returns:
            The normalized string of the input_term.
        """
        return self.normalize_fn(input_term)

    def get_terms(self, variation: str) -> Set[Any]:
        """Get the term ids for which the given variation was generated.

        Args:
            variation: A variation whose reference term(s) to retrieve.

        Returns:
            The set term ids for the variation or the missing_value.
        """
        return self.v2t.get(variation, set())

    def build_first_token(
        self,
        doctext: Union[dk_doc.Text, str],
    ) -> dk_tok.Token:
        inputf = dk_tok.TextFeatures(
            doctext, split_camelcase=self.split_input_camelcase, emoji_data=self.emoji_data
        )
        return inputf.build_first_token(normalize_fn=self.normalize_fn)


class TokenMatch:
    """Represents a match between tokens and a lexical authority variation.

    Matches a sequence of tokens against a lexical authority variation,
    tracking whether the match is complete and providing access to
    matched text and annotation generation.
    """

    def __init__(
        self,
        auth: dk_auth.LexicalAuthority,
        val_idx: Hashable,
        var: str,
        token: dk_tok.Token,
    ):
        self.auth = auth
        self.val_idx = val_idx
        self.var = var
        self.token = token

        self.varparts = var.split()
        self.matches = True
        self.tokens = []
        t = token
        for v in self.varparts:
            if t is not None and v == t.norm_text:
                self.tokens.append(t)
                t = t.next_token
            else:
                self.matches = False
                break

    def __repr__(self):
        ttext = " ".join(t.token_text for t in self.tokens)
        return (
            f"Match_{self.tokens[0].token_num}-{self.tokens[-1].token_num}({ttext})[{self.val_idx}]"
        )

    @property
    def next_token(self):
        next_token = None
        if self.matches:
            next_token = self.tokens[-1].next_token
        return next_token

    @property
    def matched_text(self):
        """Get the matched original text."""
        return self.token.full_text[self.tokens[0].start_pos : self.tokens[-1].end_pos]

    def build_annotation(self):
        return self.auth.build_annotation(
            start_pos=self.tokens[0].start_pos,
            end_pos=self.tokens[-1].end_pos,
            entity_text=self.matched_text,
            auth_value_id=self.val_idx,
        )


class TokenAligner:
    """Aligns tokens with a lexical authority to generate annotations.

    Processes a token stream, matching tokens against lexical authority
    variations and generating annotations for matches. Handles overlapping
    matches and tracks processed tokens.
    """

    def __init__(self, first_token: dk_tok.Token, authority: dk_auth.LexicalAuthority):
        self.first_token = first_token
        self.auth = authority
        self.matches: list[list[dict[str, Any]]] = []  # one list per match
        self._processed_idx: set[int] = set()
        self._walked_idx: set[int] = set()
        self._process(self.first_token)

    @property
    def annotations(self) -> List[Dict[str, Any]]:
        """Every matched annotation row, flattened out of the matches.

        The per-match grouping is the aligner's record because that is the
        unit an authority's annotations validator judges; this is the view
        for a caller that wants the rows and not the grouping.

        A fresh list is built on every access, so appending to what this
        returns discards the row silently. Add to ``matches`` instead, as one
        list per match.
        """
        return [ann_dict for match in self.matches for ann_dict in match]

    def _process(self, first_token: dk_tok.Token) -> None:
        """Walk the stream from ``first_token``, recording every match found.

        The walk is depth-first and carries its own stack. It used to use the
        interpreter's, following ``next_token`` by recursing once per token,
        which made the longest annotatable document a property of
        ``sys.getrecursionlimit()`` rather than of the document: at the
        default limit of 1000 the last length that survived was 980
        whitespace tokens, short enough that this package's own changelog
        raised ``RecursionError``.

        Each entry pairs a match to record with the token to walk on from,
        and entries are pushed so that they pop in the order the recursion
        made its calls -- a match, then everything reachable past its end,
        and only then the next match starting at the same token. That order
        is what ``matches`` has always held, and the order an authority's
        ``anns_validator`` is consulted in, so it is preserved here rather
        than tidied: making it start-position order is a visible change to
        what a consumer is shown, and belongs to a change that says so.

        Two sets of token numbers, because a token can be in either state
        without the other. ``_processed_idx`` holds tokens some match has
        consumed, which may not begin another. ``_walked_idx`` holds tokens
        already walked from; walking one a second time can record nothing --
        it finds the token either consumed, and skipped, or unmatched, and
        barren on a re-query -- so skipping it drops no match, and it is what
        keeps the walk linear in the token count rather than quadratic on the
        unmatched tokens that make up most of a document.
        """
        pending: list[tuple[TokenMatch | None, dk_tok.Token | None]] = [(None, first_token)]
        while pending:
            token_match, token = pending.pop()
            if token_match is not None:
                # A TokenMatch spans one variation and so builds exactly one
                # row; a match is still a list, because that is what a match
                # is elsewhere -- a regex match with named groups carries one
                # row per group.
                self.matches.append([token_match.build_annotation()])
            if token is None or token.token_num in self._walked_idx:
                continue
            self._walked_idx.add(token.token_num)
            # Pushed before the matches so that it pops after them: the walk
            # past this token resumes only once every match starting here,
            # and everything those matches lead on to, has been recorded.
            pending.append((None, token.next_token))
            if token.token_num not in self._processed_idx:
                for next_match in reversed(self._get_token_matches(token)):
                    pending.append((next_match, next_match.next_token))

    def _get_token_matches(self, token):
        """Find every declared variation beginning at ``token``.

        The result must stay a function of ``token`` and ``self.auth`` alone.
        ``_process`` skips a token it has already walked from on the grounds
        that a re-query could only return what the first one did; reading
        ``_processed_idx`` here -- or anything else the caller has changed
        since -- would make that false, and the walk would start dropping
        matches rather than deduplicating arrivals at them.
        """
        token_matches = []
        vs = self.auth.find_variations(token.norm_text, starts_with=True)
        if len(vs) > 0:
            for val_idx, var in vs.items():
                token_match = TokenMatch(self.auth, val_idx, var, token)
                if token_match.matches:
                    # mark token position(s) as matched
                    self._processed_idx.update({t.token_num for t in token_match.tokens})
                    token_matches.append(token_match)
        return token_matches


class DataframeAuthority(dk_auth.LexicalAuthority):
    """A pandas dataframe-based lexical authority."""

    #: Narrowed from the base, where an authority may hold no data at all: a
    #: dataframe authority is built from its dataframe and its constructor
    #: requires one, so every read below has something to read.
    authdata: dk_auth.AuthorityData

    def __init__(
        self,
        name: str,
        lexical_expander: LexicalExpander,
        authdata: dk_auth.AuthorityData,
        auth_anns_builder: dk_auth.AuthorityAnnotationsBuilder | None = None,
        field_groups: dk_auth.DerivedFieldGroups | None = None,
        anns_validator: Callable[[dk_auth.Authority, Dict[str, Any]], bool] | None = None,
        parent_auth: dk_auth.Authority | None = None,
    ):
        """Initialize with the name, values, and associated ids of the authority;
        and with the lexical expander for authoritative values.

        Args:
            name: The authority name, if different from df.columns[0].
            lexical_expander: The lexical expander for the values.
            authdata: The data for this authority.
            auth_anns_builder: The authority annotations row builder to use
                for building annotation rows.
            field_groups: The derived field groups to use.
            anns_validator: fn(auth, anns_dict_list) that returns True if
                the list of annotation row dicts are valid to be added as
                annotations for a single match or "entity".
            parent_auth: This authority's parent authority (if any).
        """
        super().__init__(
            name if name else authdata.df.columns[0],
            auth_anns_builder=auth_anns_builder,
            authdata=authdata,
            field_groups=field_groups,
            anns_validator=anns_validator,
            parent_auth=parent_auth,
        )
        self.lexical_expander = lexical_expander
        self._variations: pd.Series | None = None
        self._prev_aligner: TokenAligner | None = None

    @property
    def prev_aligner(self) -> TokenAligner | None:
        """Get the token aligner created in the latest call to annotate_text.

        Returns:
            The latest aligner, or None if nothing has been annotated yet.
        """
        return self._prev_aligner

    @property
    def variations(self) -> pd.Series:
        """Get all lexical variations in a series whose index has associated
        value IDs.

        Returns:
            A pandas series with index-identified variations.
        """
        return self._materialize_variations()

    def _materialize_variations(self) -> pd.Series:
        """Expand this authority's values, once, caching the result.

        Expanding is also what populates the lexical expander's
        variation-to-term back-index, so anything reading that index goes
        through here first.

        Returns:
            A pandas series with index-identified variations.
        """
        if self._variations is None:
            self._variations = (
                self.authdata.df[self.name].apply(self.lexical_expander).explode().dropna()
            )
        return self._variations

    def get_id_by_variation(self, variation: str) -> Set[str]:
        """Get the IDs of the value(s) associated with the given variation.

        Args:
            variation: Variation text.

        Returns:
            The possibly empty set of associated value IDS.
        """
        # The expander's variation-to-term back-index is populated as a side
        # effect of expanding this authority's values, so materialize those
        # first: a lookup against a cold index answers "no such variation" for
        # every variation, including the ones this authority declares.
        self._materialize_variations()
        ids = set()
        for value in self.lexical_expander.get_terms(variation):
            ids.update(self.get_value_ids(value))
        return ids

    def get_variations(self, value: Any, normalize: bool = True) -> Set[Any]:
        """Convenience method to compute variations for the value.

        Args:
            value: The authority value, or term, whose variations to compute.
            normalize: True to normalize the variations.

        Returns:
            The set of variations for the value.
        """
        return self.lexical_expander(value, normalize=normalize)

    def has_value(self, value: Any) -> bool:
        """Determine whether the given value is in this authority.

        Args:
            value: A possible authority value.

        Returns:
            True if the value is a valid entity value.
        """
        # ``np.any`` answers with ``numpy.bool``, which is not a ``bool``:
        # true enough for a truth test, but not for an identity one.
        return bool(np.any(self.authdata.df[self.name] == value))

    def get_value_ids(self, value: Any) -> Set[Any]:
        """Get all IDs associated with the given value. Note that typically
        there is a single ID for any value, but this allows for inherent
        ambiguities in the authority.

        Args:
            value: An authority value.

        Returns:
            The associated IDs or an empty set if the value is not valid.
        """
        return set(self.authdata.lookup_values(value).index.tolist())

    def get_values_by_id(self, value_id: Any) -> Set[Any]:
        """Get all values for the associated value ID. Note that typically
        there is a single value for an ID, but this allows for inherent
        ambiguities in the authority.

        Args:
            value_id: An authority value ID.

        Returns:
            The associated values or an empty set if the value ID is not valid.
        """
        return set(self.authdata.lookup_values(value_id, is_id=True)[self.name].tolist())

    def find_variations(
        self,
        variation: str,
        starts_with: bool = False,
        ends_with: bool = False,
        scope: str = "fullmatch",
    ) -> pd.Series:
        """Find all matches to the given variation.

        Note:
            Only the first true of starts_with, ends_with, and scope will
            be applied. If none of these are true, a full match on the pattern
            is performed.

        Args:
            variation: The text to find; treated as a regular expression
                unless either starts_with or ends_with is True.
            starts_with: When True, find all terms that start with the
                variation text.
            ends_with: When True, find all terms that end with the variation
                text.
            scope: 'fullmatch' (default), 'match', or 'contains' for
                strict, less strict, and least strict matching.

        Returns:
            The matching variations as a pd.Series.
        """
        vs = self.variations
        if starts_with:
            vs = vs[vs.str.startswith(variation)]
        elif ends_with:
            vs = vs[vs.str.endswith(variation)]
        else:
            if scope == "fullmatch":
                hits = vs.str.fullmatch(variation)
            elif scope == "match":
                hits = vs.str.match(variation)
            else:
                hits = vs.str.contains(variation)
            vs = vs[hits]
        vs = vs.drop_duplicates()
        return vs

    def get_variations_df(
        self,
        variations: pd.Series,
        variations_colname: str = "variation",
        ids_colname: str | None = None,
        lookup_values: bool = False,
    ) -> pd.DataFrame:
        """Create a DataFrame including associated ids for each variation.

        Args:
            variations: The variations to include in the dataframe.
            variations_colname: The name of the variations column.
            ids_colname: The column name for value ids.
            lookup_values: When True, include a self.name column
                with associated values.
        """
        if ids_colname is None:
            ids_colname = f"{self.name}_id"
        df = pd.DataFrame(
            {
                variations_colname: variations,
                ids_colname: variations.apply(self.get_id_by_variation),
            }
        ).explode(ids_colname)
        if lookup_values:
            df[self.name] = df[ids_colname].apply(self.get_values_by_id)
            df = df.explode(self.name)
        return df

    def add_annotations(
        self,
        text_obj: dk_anns.AnnotatedText,
    ) -> dk_anns.Annotations:
        """Method to do the work of finding, validating, and adding annotations.

        The text object carries both halves this needs: it is a
        :class:`~dataknobs_structures.document.Text`, so the tokenizer reads
        its id and label straight off it, and it owns the annotations the
        matches are added to.

        The aligner's matches are offered one at a time, which is the unit
        `anns_validator` is documented to judge and the unit the regex arm
        already used.

        Args:
            text_obj: The annotated text object to process and add annotations.

        Returns:
            The added Annotations.
        """
        first_token = self.lexical_expander.build_first_token(text_obj)
        token_aligner = TokenAligner(first_token, self)
        self._prev_aligner = token_aligner
        return self.add_valid_annotations(text_obj, token_aligner.matches)


class CorrelatedAuthorityData(dk_auth.AuthorityData):
    """Container for authoritative data containing correlated data for multiple
    "sub" authorities.
    """

    def __init__(self, df: pd.DataFrame, name: str):
        super().__init__(df, name)
        self._authority_data: Dict[str, dk_auth.AuthorityData] = {}

    def sub_authority_names(self) -> List[str]:
        """Get the "sub" authority names.

        Returns:
            The names of the "sub" authorities this data correlates.
        """
        raise NotImplementedError

    @abstractmethod
    def auth_values_mask(self, name: str, value_id: int) -> pd.Series:
        """Identify full-authority data corresponding to this sub-value.

        Args:
            name: The sub-authority name.
            value_id: The sub-authority value_id.

        Returns:
            A series representing relevant full-authority data.
        """
        raise NotImplementedError

    @abstractmethod
    def auth_records_mask(
        self,
        record_value_ids: Dict[str, int],
        filter_mask: pd.Series | None = None,
    ) -> pd.Series | None:
        """Get a series identifying records in the full authority matching
        the given records of the form {<sub-name>: <sub-value-id>}.

        Args:
            record_value_ids: The dict of field names to value_ids.
            filter_mask: A pre-filter limiting records to consider and/or
                building records incrementally.

        Returns:
            A series identifying where all fields exist.
        """
        raise NotImplementedError

    @abstractmethod
    def get_auth_records(self, records_mask: pd.Series) -> pd.DataFrame:
        """Get the authority records identified by the mask.

        Args:
            records_mask: A series identifying records in the full data.

        Returns:
            The records for which the mask is True.
        """
        raise NotImplementedError

    @abstractmethod
    def combine_masks(self, mask1: pd.Series | None, mask2: pd.Series | None) -> pd.Series | None:
        """Combine the masks if possible, returning the valid combination or None.

        Args:
            mask1: An auth_records_mask consistent with this data.
            mask2: Another data auth_records_mask.

        Returns:
            The combined consistent records_mask or None.
        """
        raise NotImplementedError


class MultiAuthorityData(CorrelatedAuthorityData):
    """Container for authoritative data containing correlated data for multiple
    "sub" authorities composed of explicit data for each component.
    """

    @abstractmethod
    def build_authority_data(self, name: str) -> dk_auth.AuthorityData:
        """Build an authority for the named sub-authority.

        Args:
            name: The "sub" authority name.

        Returns:
            The "sub" authority data.
        """
        raise NotImplementedError

    def peek_authority_data(self, name: str) -> dk_auth.AuthorityData | None:
        """Retrieve the named "sub" authority data if already built, without building it.

        The non-building counterpart to :meth:`get_authority_data`.

        Args:
            name: The "sub" authority name.

        Returns:
            The "sub" authority data, or None if it has not been built.
        """
        return self._authority_data.get(name)

    def get_authority_data(self, name: str) -> dk_auth.AuthorityData:
        """Get AuthorityData for the named "sub" authority, building if needed.

        Overrides the flat answer in :meth:`AuthorityData.get_authority_data`:
        the names this data holds are its "sub" authorities rather than its
        own, and each is built on first request and kept.

        Args:
            name: The "sub" authority name.

        Returns:
            The "sub" authority data.
        """
        if name not in self._authority_data:
            self._authority_data[name] = self.build_authority_data(name)
        return self._authority_data[name]

    @staticmethod
    def get_unique_vals_df(col: pd.Series, name: str) -> pd.DataFrame:
        """Get a dataframe with the unique values from the column and the given
        column name.
        """
        # ``pd.unique`` is typed ``np_1darray | ExtensionArray``; ``np.asarray``
        # narrows that union for the type checker. Runtime behaviour is
        # unchanged — the call already returned an ndarray for every dtype.
        data = np.sort(np.asarray(pd.unique(col.dropna())))
        # ``pd.api.types.is_integer_dtype`` rather than ``np.issubdtype``: the
        # latter raises ``TypeError`` on every pandas ExtensionDtype, and
        # reports ``timedelta64`` as integer (it subclasses ``np.signedinteger``).
        if pd.api.types.is_integer_dtype(col.dtype):
            # IDs for an integer column are the integers themselves
            col_df = pd.DataFrame({name: data}, index=data)
        else:
            # IDs for other columns are auto-generated from 0 to n-1
            col_df = pd.DataFrame({name: data})
        return col_df

    def lookup_subauth_values(
        self, name: str, value: int, is_id: bool = False
    ) -> pd.DataFrame | None:
        """Lookup "sub" authority data for the named "sub" authority value.

        Peeks rather than builds, so None is an ordinary answer rather than an
        edge case: until something has built the named "sub" authority, there
        is nothing to look the value up in.

        Args:
            name: The sub-authority name.
            value: The value for the sub-authority to lookup.
            is_id: True if value is an ID.

        Returns:
            The applicable authority dataframe rows, or None if the "sub"
            authority has not been built.
        """
        values_df = None
        authdata = self.peek_authority_data(name)
        if authdata is not None:
            values_df = authdata.lookup_values(value, is_id=is_id)
        return values_df

    def lookup_auth_values(
        self,
        name: str,
        value: str,
    ) -> pd.DataFrame:
        """Lookup original authority data for the named "sub" authority value.

        Args:
            name: The sub-authority name.
            value: The sub-authority value(s) (or dataframe row(s)).

        Returns:
            The original authority dataframe rows.
        """
        return self.df[self.df[name] == value]

    def auth_values_mask(self, name: str, value_id: int) -> pd.Series:
        """Identify the rows in the full authority corresponding to this sub-value.

        Args:
            name: The sub-authority name.
            value_id: The sub-authority value_id.

        Returns:
            A boolean series where the field exists. All False if the "sub"
            authority has not been built, since no record can carry a value
            that does not exist yet.
        """
        field_values = self.lookup_subauth_values(name, value_id, is_id=True)
        if field_values is None:
            return pd.Series(False, index=self.df.index)
        return self.df[name].isin(field_values[name].tolist())

    def auth_records_mask(
        self,
        record_value_ids: Dict[str, int],
        filter_mask: pd.Series | None = None,
    ) -> pd.Series | None:
        """Get a boolean series identifying records in the full authority matching
        the given records of the form {<sub-name>: <sub-value-id>}.

        Args:
            record_value_ids: The dict of field names to value_ids.
            filter_mask: A pre-filter limiting records to consider and/or
                building records incrementally.

        Returns:
            A boolean series where all fields exist or None.
        """
        has_fields = filter_mask
        for name, value_id in record_value_ids.items():
            has_field = self.auth_values_mask(name, value_id)
            if has_fields is None:
                has_fields = has_field
            else:
                has_fields &= has_field
        return has_fields

    def get_auth_records(self, records_mask: pd.Series) -> pd.DataFrame:
        """Get the authority records identified by the mask.

        Args:
            records_mask: A boolean series identifying records in the full df.

        Returns:
            The records/rows for which the mask is True.
        """
        return self.df[records_mask]

    def combine_masks(self, mask1: pd.Series | None, mask2: pd.Series | None) -> pd.Series | None:
        """Combine the masks if possible, returning the valid combination or None.

        Args:
            mask1: An auth_records_mask consistent with this data.
            mask2: Another data auth_records_mask.

        Returns:
            The combined consistent records_mask or None.
        """
        result = None
        if mask1 is not None and mask2 is not None:
            result = mask1 & mask2
        elif mask1 is not None:
            result = mask1
        elif mask2 is not None:
            result = mask2
        return result if np.any(result) else None


class SimpleMultiAuthorityData(MultiAuthorityData):
    """Data class for pulling a single column from the multi-authority data
    as a "sub" authority.
    """

    def build_authority_data(self, name: str) -> dk_auth.AuthorityData:
        """Build an authority for the named column holding authority data.

        Note:
            Only unique values are kept and the full dataframe's index
            will not be preserved.

        Args:
            name: The "sub" authority (and column) name.

        Returns:
            The "sub" authority data.
        """
        col = self.df[name]
        col_df = self.get_unique_vals_df(col, name)
        return dk_auth.AuthorityData(col_df, name)


class MultiAuthorityFactory(dk_auth.AuthorityFactory[dk_auth.AuthorityData]):
    """A factory for building a "sub" authority directly or indirectly
    from authority data.

    Indirectly from :class:`MultiAuthorityData`, whose named "sub" authority
    it pulls; directly from any other :class:`~dataknobs_xization.authorities.AuthorityData`,
    which supplies the authority of its own name. The factory asks the data
    the same question either way, so it is substitutable for the abstract it
    implements rather than usable only through its own concrete type.

    The three things the built authority is configured with -- its lexical
    expander, its derived field groups and its annotations validator -- are
    given to the factory and read back per name, so a subclass can vary any
    of them by authority without reimplementing the build.
    """

    def __init__(
        self,
        auth_name: str,
        lexical_expander: LexicalExpander | None = None,
        field_groups: dk_auth.DerivedFieldGroups | None = None,
        anns_validator: Callable[[dk_auth.Authority, Dict[str, Any]], bool] | None = None,
    ):
        """Initialize the MultiAuthorityFactory.

        Args:
            auth_name: The name of the dataframe authority to build.
            lexical_expander: The lexical expander to use (default=identity).
            field_groups: The derived field groups the built authorities use
                (default=the authority's own default groups).
            anns_validator: fn(auth, anns_dict_list) the built authorities
                validate each match with (default=accept every match).
        """
        self.auth_name = auth_name
        self._lexical_expander = lexical_expander
        self._field_groups = field_groups
        self._anns_validator = anns_validator

    def get_lexical_expander(self, name: str) -> LexicalExpander:
        """Get the lexical expander for the named (column) data.

        Args:
            name: The name of the column to expand.

        Returns:
            The appropriate lexical_expander.
        """
        if self._lexical_expander is None:
            self._lexical_expander = LexicalExpander(None, None)
        return self._lexical_expander

    def get_field_groups(self, name: str) -> dk_auth.DerivedFieldGroups | None:
        """Get the derived field groups for the named authority.

        Args:
            name: The name of the authority being built.

        Returns:
            The field groups to build with, or None to leave the authority
            its own default.
        """
        return self._field_groups

    def get_anns_validator(
        self, name: str
    ) -> Callable[[dk_auth.Authority, Dict[str, Any]], bool] | None:
        """Get the annotations validator for the named authority.

        Args:
            name: The name of the authority being built.

        Returns:
            fn(auth, anns_dict_list) judging each match, or None to accept
            every match the authority finds.
        """
        return self._anns_validator

    def build_authority(
        self,
        name: str,
        auth_anns_builder: dk_auth.AuthorityAnnotationsBuilder,
        authdata: dk_auth.AuthorityData,
        parent_auth: dk_auth.Authority | None = None,
    ) -> DataframeAuthority:
        """Build a DataframeAuthority.

        Args:
            name: The name of the authority to build.
            auth_anns_builder: The authority annotations row builder the built
                authority builds annotation rows with.
            authdata: The source data supplying the named authority: a
                :class:`MultiAuthorityData` holding it as a "sub" authority,
                or the authority's own data.
            parent_auth: The parent authority.

        Returns:
            The DataframeAuthority instance.
        """
        return DataframeAuthority(
            name,
            self.get_lexical_expander(name),
            authdata.get_authority_data(name),
            auth_anns_builder=auth_anns_builder,
            field_groups=self.get_field_groups(name),
            anns_validator=self.get_anns_validator(name),
            parent_auth=parent_auth,
        )

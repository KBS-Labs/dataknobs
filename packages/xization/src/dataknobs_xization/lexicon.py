"""Lexical matching and token alignment for text processing.

Provides classes for lexical expansion, normalization, token alignment,
and pattern matching in text with support for variations and fuzzy matching.
"""

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List, Set, Union

import more_itertools
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
        variations_fn: Callable[[str], Set[str]],
        normalize_fn: Callable[[str], str],
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
        self.variations_fn = variations_fn if variations_fn else lambda x: {x}
        self.normalize_fn = normalize_fn if normalize_fn else lambda x: x
        self.split_input_camelcase = split_input_camelcase
        self.emoji_data = emoji_utils.load_emoji_data() if detect_emojis else None
        self.v2t = defaultdict(set)

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
            more_itertools.consume(self.v2t[v].add(term) for v in variations)
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

    def __init__(self, auth: dk_auth.LexicalAuthority, val_idx: int, var: str, token: dk_tok.Token):
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
        return self.token.input_text[self.tokens[0].start_pos : self.tokens[-1].end_pos]

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
        self.annotations = []  # List[Dict[str, Any]]
        self._processed_idx = set()
        self._process(self.first_token)

    def _process(self, token):
        if token is not None:
            if token.token_num not in self._processed_idx:
                token_matches = self._get_token_matches(token)
                for token_match in token_matches:
                    self.annotations.append(token_match.build_annotation())
                    self._process(token_match.next_token)
            self._process(token.next_token)

    def _get_token_matches(self, token):
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

    def __init__(
        self,
        name: str,
        lexical_expander: LexicalExpander,
        authdata: dk_auth.AuthorityData,
        auth_anns_builder: dk_auth.AuthorityAnnotationsBuilder = None,
        field_groups: dk_auth.DerivedFieldGroups = None,
        anns_validator: Callable[[dk_auth.Authority, Dict[str, Any]], bool] = None,
        parent_auth: dk_auth.Authority = None,
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
        self._variations = None
        self._prev_aligner = None

    @property
    def prev_aligner(self) -> TokenAligner:
        """Get the token aligner created in the latest call to annotate_text."""
        return self._prev_aligner

    @property
    def variations(self) -> pd.Series:
        """Get all lexical variations in a series whose index has associated
        value IDs.

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
        return np.any(self.authdata.df[self.name] == value)

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
        ids_colname: str = None,
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
        doctext: dk_doc.Text,
        annotations: dk_anns.Annotations,
    ) -> dk_anns.Annotations:
        """Method to do the work of finding, validating, and adding annotations.

        Args:
            doctext: The text to process.
            annotations: The annotations object to add annotations to.

        Returns:
            The given or a new Annotations instance.
        """
        first_token = self.lexical_expander.build_first_token(
            doctext.text, input_id=doctext.text_id
        )
        token_aligner = TokenAligner(first_token, self)
        self._prev_aligner = token_aligner
        if self.validate_ann_dicts(token_aligner.annotations):
            annotations.add_dicts(token_aligner.annotations)
        return annotations


class CorrelatedAuthorityData(dk_auth.AuthorityData):
    """Container for authoritative data containing correlated data for multiple
    "sub" authorities.
    """

    def __init__(self, df: pd.DataFrame, name: str):
        super().__init__(df, name)
        self._authority_data = {}

    def sub_authority_names(self) -> List[str]:
        """Get the "sub" authority names."""
        return None

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
        filter_mask: pd.Series = None,
    ) -> pd.Series:
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
    def combine_masks(self, mask1: pd.Series, mask2: pd.Series) -> pd.Series:
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

    def __init__(self, df: pd.DataFrame, name: str):
        super().__init__(df, name)
        self._authority_data = {}

    @abstractmethod
    def build_authority_data(self, name: str) -> dk_auth.AuthorityData:
        """Build an authority for the named sub-authority.

        Args:
            name: The "sub" authority name.

        Returns:
            The "sub" authority data.
        """
        raise NotImplementedError

    @property
    def authority_data(self, name: str) -> dk_auth.AuthorityData:
        """Retrieve without building the named authority data, or None"""
        return self._authority_data.get(name, None)

    def get_authority_data(self, name: str) -> dk_auth.AuthorityData:
        """Get AuthorityData for the named "sub" authority, building if needed.

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
        data = np.sort(pd.unique(col.dropna()))
        if np.issubdtype(col.dtype, np.integer):
            # IDs for an integer column are the integers themselves
            col_df = pd.DataFrame({name: data}, index=data)
        else:
            # IDs for other columns are auto-generated from 0 to n-1
            col_df = pd.DataFrame({name: data})
        return col_df

    def lookup_subauth_values(self, name: str, value: int, is_id: bool = False) -> pd.DataFrame:
        """Lookup "sub" authority data for the named "sub" authority value.

        Args:
            name: The sub-authority name.
            value: The value for the sub-authority to lookup.
            is_id: True if value is an ID.

        Returns:
            The applicable authority dataframe rows.
        """
        values_df = None
        authdata = self._authority_data.get(name, None)
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
            A boolean series where the field exists.
        """
        field_values = self.lookup_subauth_values(name, value_id, is_id=True)
        return self.df[name].isin(field_values[name].tolist())

    def auth_records_mask(
        self,
        record_value_ids: Dict[str, int],
        filter_mask: pd.Series = None,
    ) -> pd.Series:
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

    def combine_masks(self, mask1: pd.Series, mask2: pd.Series) -> pd.Series:
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


class MultiAuthorityFactory(dk_auth.AuthorityFactory):
    """An factory for building a "sub" authority directly or indirectly
    from MultiAuthorityData.
    """

    def __init__(
        self,
        auth_name: str,
        lexical_expander: LexicalExpander = None,
    ):
        """Initialize the MultiAuthorityFactory.

        Args:
            auth_name: The name of the dataframe authority to build.
            lexical_expander: The lexical expander to use (default=identity).
        """
        self.auth_name = auth_name
        self._lexical_expander = lexical_expander

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

    def build_authority(
        self,
        name: str,
        auth_anns_builder: dk_auth.AuthorityAnnotationsBuilder,
        multiauthdata: MultiAuthorityData,
        parent_auth: dk_auth.Authority = None,
    ) -> DataframeAuthority:
        """Build a DataframeAuthority.

        Args:
            name: The name of the authority to build.
            auth_anns_builder: The authority annotations row builder to use
                for building annotation rows.
            multiauthdata: The multi-authority source data.
            parent_auth: The parent authority.

        Returns:
            The DataframeAuthority instance.
        """
        authdata = multiauthdata.get_authority_data(name)
        field_groups = None  # TODO: get from instance var set on construction?
        anns_validator = None  # TODO: get from multiauthdata?
        return DataframeAuthority(
            name,
            self.get_lexical_expander(name),
            authdata,
            field_groups=field_groups,
            anns_validator=anns_validator,
            parent_auth=parent_auth,
        )

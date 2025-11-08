"""Authority-based annotation processing and field grouping.

Provides classes for managing authority-based annotations, field groups,
and derived annotation columns for structured text extraction.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Set, Union

import pandas as pd

import dataknobs_xization.annotations as dk_annots

# Key annotation column name constants
KEY_AUTH_ID_COL = "auth_id"


class DerivedFieldGroups(dk_annots.DerivedAnnotationColumns):
    """Defines derived column types:
    * "field_type" -- The column holding they type of field of an annotation row
    * "field_group" -- The column holding the group number(s) of the field
    * "field_record" -- The column holding record number(s) of the field
    """

    def __init__(
        self,
        field_type_suffix: str = "_field",
        field_group_suffix: str = "_num",
        field_record_suffix: str = "_recsnum",
    ):
        """Add derived column types/names: Given an annnotation row,
          * field_type(row) == f'{row[ann_type_col]}_field'
          * field_group(row) == f'{row[ann_type_col]}_num'
          * field_record(row) == f'{row[ann_type_col])_recsnum'

        Where:
          * A field_type column holds annotation "sub"- type values, or fields
          * A field_group column identifies groups of annotation fields
          * A field_record column identifies groups of annotation field groups

        Args:
            field_type_suffix: The field_type col name suffix (if not _field).
            field_group_suffix: The field_group col name suffix (if not _num).
            field_record_suffix: field_record colname sfx (if not _recsnum).
        """
        self.field_type_suffix = field_type_suffix
        self.field_group_suffix = field_group_suffix
        self.field_record_suffix = field_record_suffix

    def get_col_value(
        self,
        metadata: dk_annots.AnnotationsMetaData,
        col_type: str,
        row: pd.Series,
        missing: str = None,
    ) -> str:
        """Get the value of the column in the given row derived from col_type,
        where col_type is one of:
          * "field_type" == f"{field}_field"
          * "field_group" == f"{field}_num"
          * "field_record" == f"{field}_recsnum"

        And "field" is the row_accessor's metadata's "ann_type" col's value.

        Args:
            metadata: The AnnotationsMetaData.
            col_type: The type of column value to derive.
            row: A row from which to get the value.
            missing: The value to return for unknown or missing column.

        Returns:
            The row value or the missing value.
        """
        value = missing
        if metadata.ann_type_col in row.index:
            field = row[metadata.ann_type_col]
            if field is not None:
                if col_type == "field_type":
                    col_name = self.get_field_type_col(field)
                elif col_type == "field_group":
                    col_name = self.get_field_group_col(field)
                elif col_type == "field_record":
                    col_name = self.get_field_record_col(field)
                if col_name is not None and col_name in row.index:
                    value = row[col_name]
        return value

    def unpack_field(self, field_value: str) -> str:
        """Given a field in any of its derivatives (like field type, field group
        or field record,) unpack and return the basic field value itself.
        """
        field = field_value
        if field.endswith(self.field_record_suffix):
            field = field.replace(self.field_record_suffix, "")
        elif field.endswith(self.field_group_suffix):
            field = field.replace(self.field_group_suffix, "")
        elif field.endswith(self.field_type_suffix):
            field = field.replace(self.field_type_suffix, "")
        return field

    def get_field_name(self, field_value: str) -> str:
        """Given a field name or field col name, e.g., an annotation type col's
        value (the field name); or a field type, group, or record column name,
        get the field name.
        """
        return self.unpack_field(field_value)

    def get_field_type_col(self, field_value: str) -> str:
        """Given a field name or field col name, e.g., an annotation type col's
        value; or a field type, group, or record column name, get the field
        name.
        """
        field = self.unpack_field(field_value)
        return f"{field}{self.field_type_suffix}"

    def get_field_group_col(self, field_value: str) -> str:
        """Given a field name or field col name, e.g., an annotation type col's
        value; or a field type, group, or record, get the name of the derived
        field group column.
        """
        field = self.unpack_field(field_value)
        return f"{field}{self.field_group_suffix}"

    def get_field_record_col(self, field_value: str) -> str:
        """Given a field name or field col name, e.g., an annotation type col's
        value; or a field type, group, or record, get the name of the derived
        field record column.
        """
        field = self.unpack_field(field_value)
        return f"{field}{self.field_record_suffix}"


class AuthorityAnnotationsMetaData(dk_annots.AnnotationsMetaData):
    """An extension of AnnotationsMetaData that adds an 'auth_id_col' to the
    standard (key) annotation columns (attributes).
    """

    def __init__(
        self,
        start_pos_col: str = dk_annots.KEY_START_POS_COL,
        end_pos_col: str = dk_annots.KEY_END_POS_COL,
        text_col: str = dk_annots.KEY_TEXT_COL,
        ann_type_col: str = dk_annots.KEY_ANN_TYPE_COL,
        auth_id_col: str = KEY_AUTH_ID_COL,
        sort_fields: List[str] = (dk_annots.KEY_START_POS_COL, dk_annots.KEY_END_POS_COL),
        sort_fields_ascending: List[bool] = (True, False),
        **kwargs: Any,
    ):
        """Initialize with key (and more) column names and info.

        Key column types:
          * start_pos
          * end_pos
          * text
          * ann_type
          * auth_id

        Note:
            Actual table columns can be named arbitrarily, BUT interactions
            through annotations classes and interfaces relating to the "key"
            columns must use the key column constants.

        Args:
            start_pos_col: Col name for the token starting position.
            end_pos_col: Col name for the token ending position.
            text_col: Col name for the token text.
            ann_type_col: Col name for the annotation types.
            auth_id_col: Col name for the authority value ID.
            sort_fields: The col types relevant for sorting annotation rows.
            sort_fields_ascending: To specify sort order of sort_fields.
            **kwargs: More column types mapped to column names.
        """
        super().__init__(
            start_pos_col=start_pos_col,
            end_pos_col=end_pos_col,
            text_col=text_col,
            ann_type_col=ann_type_col,
            sort_fields=sort_fields,
            sort_fields_ascending=sort_fields_ascending,
            auth_id=auth_id_col,
            **kwargs,
        )

    @property
    def auth_id_col(self) -> str:
        """Get the column name for the auth_id"""
        return self.data[KEY_AUTH_ID_COL]


class AuthorityAnnotationsBuilder(dk_annots.AnnotationsBuilder):
    """An extension of an AnnotationsBuilder that adds the 'auth_id' column."""

    def __init__(
        self,
        metadata: AuthorityAnnotationsMetaData = None,
        data_defaults: Dict[str, Any] = None,
    ):
        """Initialize AuthorityAnnotationsBuilder.

        Args:
            metadata: The authority annotations metadata.
            data_defaults: Dict[ann_colname, default_value] with default
                values for annotation columns.
        """
        super().__init__(
            metadata if metadata is not None else AuthorityAnnotationsMetaData(), data_defaults
        )

    def build_annotation_row(
        self, start_pos: int, end_pos: int, text: str, ann_type: str, auth_id: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Build an annotation row with the mandatory key values and those from
        the remaining keyword arguments.

        For those kwargs whose names match metadata column names, override the
        data_defaults and add remaining data_default attributes.

        Args:
            start_pos: The token start position.
            end_pos: The token end position.
            text: The token text.
            ann_type: The annotation type.
            auth_id: The authority ID for the row.
            **kwargs: Additional keyword arguments.

        Returns:
            The result row dictionary.
        """
        return self.do_build_row(
            {
                self.metadata.start_pos_col: start_pos,
                self.metadata.end_pos_col: end_pos,
                self.metadata.text_col: text,
                self.metadata.ann_type_col: ann_type,
                self.metadata.auth_id_col: auth_id,
            },
            **kwargs,
        )


class AuthorityData:
    """A wrapper for authority data."""

    def __init__(self, df: pd.DataFrame, name: str):
        self._df = df
        self.name = name

    @property
    def df(self) -> pd.DataFrame:
        """Get the authority data in a dataframe"""
        return self._df

    def lookup_values(self, value: Any, is_id: bool = False) -> pd.DataFrame:
        """Lookup authority value(s) for the given value or value id.

        Args:
            value: A value or value_id for this authority.
            is_id: True if value is an ID.

        Returns:
            The applicable authority dataframe rows.
        """
        col = self.df.index if is_id else self.df[self.name]
        return self.df[col == value]


class Authority(dk_annots.Annotator):
    """A class for managing and defining tabular authoritative data for e.g.,
    taxonomies, etc., and using them to annotate instances within text.
    """

    def __init__(
        self,
        name: str,
        auth_anns_builder: AuthorityAnnotationsBuilder = None,
        authdata: AuthorityData = None,
        field_groups: DerivedFieldGroups = None,
        anns_validator: Callable[["Authority", Dict[str, Any]], bool] = None,
        parent_auth: "Authority" = None,
    ):
        """Initialize with this authority's metadata.

        Args:
            name: This authority's entity name.
            auth_anns_builder: The authority annotations row builder to use
                for building annotation rows.
            authdata: The authority data.
            field_groups: The derived field groups to use.
            anns_validator: fn(auth, anns_dict_list) that returns True if
                the list of annotation row dicts are valid to be added as
                annotations for a single match or "entity".
            parent_auth: This authority's parent authority (if any).
        """
        super().__init__(name)
        self.anns_builder = (
            auth_anns_builder if auth_anns_builder is not None else AuthorityAnnotationsBuilder()
        )
        self.authdata = authdata
        self.field_groups = field_groups if field_groups is not None else DerivedFieldGroups()
        self.anns_validator = anns_validator
        self._parent = parent_auth

    @property
    def metadata(self) -> AuthorityAnnotationsMetaData:
        """Get the meta-data"""
        return self.anns_builder.metadata

    @property
    def parent(self) -> "Authority":
        """Get this authority's parent, or None."""
        return self._parent

    @abstractmethod
    def has_value(self, value: Any) -> bool:
        """Determine whether the given value is in this authority.

        Args:
            value: A possible authority value.

        Returns:
            True if the value is a valid entity value.
        """
        raise NotImplementedError

    def annotate_input(
        self,
        text_obj: Union[dk_annots.AnnotatedText, str],
        **kwargs: Any,
    ) -> dk_annots.Annotations:
        """Find and annotate this authority's entities in the document text
        as dictionaries like:
        [
            {
                'input_id': <id>,
                'start_pos': <start_char_pos>,
                'end_pos': <end_char_pos>,
                'entity_text': <entity_text>,
                'ann_type': <authority_name>,
                '<auth_id>': <auth_value_id_or_canonical_form>,
                'confidence': <confidence_if_available>,
            },
        ]

        Args:
            text_obj: The text object or string to process.
            **kwargs: Additional keyword arguments.

        Returns:
            An Annotations instance.
        """
        if text_obj is not None:
            if isinstance(text_obj, str) and len(text_obj.strip()) > 0:
                text_obj = dk_annots.AnnotatedText(
                    text_obj,
                    annots_metadata=self.metadata,
                )
        if text_obj is not None:
            annotations = self.add_annotations(text_obj)
        return annotations

    @abstractmethod
    def add_annotations(
        self,
        text_obj: dk_annots.AnnotatedText,
    ) -> dk_annots.Annotations:
        """Method to do the work of finding, validating, and adding annotations.

        Args:
            text_obj: The annotated text object to process and add annotations.

        Returns:
            The added Annotations.
        """
        raise NotImplementedError

    def validate_ann_dicts(self, ann_dicts: List[Dict[str, Any]]) -> bool:
        """The annotation row dictionaries are valid if:
          * They are non-empty
          * and
             * either there is no annotations validator
             * or they are valid according to the validator

        Args:
            ann_dicts: Annotation dictionaries.

        Returns:
            True if valid.
        """
        return len(ann_dicts) > 0 and (
            self.anns_validator is None or self.anns_validator(self, ann_dicts)
        )

    def compose(
        self,
        annotations: dk_annots.Annotations,
    ) -> dk_annots.Annotations:
        """Compose annotations into groups.

        Args:
            annotations: The annotations.

        Returns:
            Composed annotations.
        """
        return annotations

    def build_annotation(
        self,
        start_pos: int = None,
        end_pos: int = None,
        entity_text: str = None,
        auth_value_id: Any = None,
        conf: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build annotations with the given components."""
        return self.anns_builder.build_annotation_row(
            start_pos, end_pos, entity_text, self.name, auth_value_id, auth_valconf=conf, **kwargs
        )


class AnnotationsValidator(ABC):
    """A base class with helper functions for performing validations on annotation
    rows.
    """

    def __call__(
        self,
        auth: Authority,
        ann_row_dicts: List[Dict[str, Any]],
    ) -> bool:
        """Call function to enable instances of this type of class to be passed in
        as a anns_validator function to an Authority.

        Args:
            auth: The authority proposing annotations.
            ann_row_dicts: The proposed annotations.

        Returns:
            True if the annotations are valid; otherwise, False.
        """
        return self.validate_annotation_rows(
            AnnotationsValidator.AuthAnnotations(auth, ann_row_dicts)
        )

    @abstractmethod
    def validate_annotation_rows(
        self,
        auth_annotations: "AnnotationsValidator.AuthAnnotations",
    ) -> bool:
        """Determine whether the proposed authority annotation rows are valid.

        Args:
            auth_annotations: The AuthAnnotations instance with the
                proposed data.

        Returns:
            True if valid; False if not.
        """
        raise NotImplementedError

    class AuthAnnotations:
        """A wrapper class for convenient access to the entity annotations."""

        def __init__(self, auth: Authority, ann_row_dicts: List[Dict[str, Any]]):
            self.auth = auth
            self.ann_row_dicts = ann_row_dicts
            self._row_accessor = None  # AnnotationsRowAccessor
            self._anns = None  # Annotations
            self._atts = None  # Dict[str, str]

        @property
        def row_accessor(self) -> dk_annots.AnnotationsRowAccessor:
            """Get the row accessor for this instance's annotations."""
            if self._row_accessor is None:
                self._row_accessor = dk_annots.AnnotationsRowAccessor(
                    self.auth.metadata, derived_cols=self.auth.field_groups
                )
            return self._row_accessor

        @property
        def anns(self) -> dk_annots.Annotations:
            """Get this instance's annotation rows as an annotations object"""
            if self._anns is None:
                self._anns = dk_annots.Annotations(self.auth.metadata)
                for row_dict in self.ann_row_dicts:
                    self._anns.add_dict(row_dict)
            return self._anns

        @property
        def df(self) -> pd.DataFrame:
            """Get the annotation's dataframe"""
            return self.anns.df

        def get_field_type(self, row: pd.Series) -> str:
            """Get the entity field type value"""
            return self.row_accessor.get_col_value("field_type", row, None)

        def get_text(self, row: pd.Series) -> str:
            """Get the entity text from the row"""
            return self.row_accessor.get_col_value(self.auth.metadata.text_col, row, None)

        @property
        def attributes(self) -> Dict[str, str]:
            """Get this instance's annotation entity attributes"""
            if self._atts is None:
                self._atts = {
                    self.get_field_type(row): self.get_text(row) for _, row in self.df.iterrows()
                }
            return self._atts

        def colval(self, col_name, row) -> Any:
            """Get the column's value from the given row"""
            return self.row_accessor.get_col_value(col_name, row)


class AuthorityFactory(ABC):
    """A factory class for building an authority."""

    @abstractmethod
    def build_authority(
        self,
        name: str,
        auth_anns_builder: AuthorityAnnotationsBuilder,
        authdata: AuthorityData,
        parent_auth: Authority = None,
    ) -> Authority:
        """Build an authority with the given name and data.

        Args:
            name: The authority name.
            auth_anns_builder: The authority annotations row builder to use
                for building annotation rows.
            authdata: The authority data.
            parent_auth: The parent authority.

        Returns:
            The authority.
        """
        raise NotImplementedError


class LexicalAuthority(Authority):
    """A class for managing named entities by ID with associated values and
    variations.
    """

    def __init__(
        self,
        name: str,
        auth_anns_builder: AuthorityAnnotationsBuilder = None,
        authdata: AuthorityData = None,
        field_groups: DerivedFieldGroups = None,
        anns_validator: Callable[["Authority", Dict[str, Any]], bool] = None,
        parent_auth: "Authority" = None,
    ):
        """Initialize with this authority's metadata.

        Args:
            name: This authority's entity name.
            auth_anns_builder: The authority annotations row builder to use
                for building annotation rows.
            authdata: The authority data.
            field_groups: The derived field groups to use.
            anns_validator: fn(auth, anns_dict_list) that returns True if
                the list of annotation row dicts are valid to be added as
                annotations for a single match or "entity".
            parent_auth: This authority's parent authority (if any).
        """
        super().__init__(
            name,
            auth_anns_builder=auth_anns_builder,
            authdata=authdata,
            field_groups=field_groups,
            anns_validator=anns_validator,
            parent_auth=parent_auth,
        )

    @abstractmethod
    def get_value_ids(self, value: Any) -> Set[Any]:
        """Get all IDs associated with the given value. Note that typically
        there is a single ID for any value, but this allows for inherent
        ambiguities in the authority.

        Args:
            value: An authority value.

        Returns:
            The associated IDs or an empty set if the value is not valid.
        """
        raise NotImplementedError

    @abstractmethod
    def get_values_by_id(self, value_id: Any) -> Set[Any]:
        """Get all values for the associated value ID. Note that typically
        there is a single value for an ID, but this allows for inherent
        ambiguities in the authority.

        Args:
            value_id: An authority value ID.

        Returns:
            The associated values or an empty set if the value is not valid.
        """
        raise NotImplementedError

    @abstractmethod
    def get_id_by_variation(self, variation: str) -> Set[str]:
        """Get the IDs of the value(s) associated with the given variation.

        Args:
            variation: Variation text.

        Returns:
            The possibly empty set of associated value IDS.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class RegexAuthority(Authority):
    """A class for managing named entities by ID with associated values and
    variations.
    """

    def __init__(
        self,
        name: str,
        regex: re.Pattern,
        canonical_fn: Callable[[str, str], Any] = None,
        auth_anns_builder: AuthorityAnnotationsBuilder = None,
        authdata: AuthorityData = None,
        field_groups: DerivedFieldGroups = None,
        anns_validator: Callable[[Authority, Dict[str, Any]], bool] = None,
        parent_auth: "Authority" = None,
    ):
        """Initialize with this authority's entity name.

        Note:
            If the regular expression has capturing groups, each group
            will result in a separate entity, with the group name if provided
            in the regular expression as ...(?P<group_name>group_regex)...

        Args:
            name: The authority name.
            regex: The regular expression to apply.
            canonical_fn: A function, fn(match_text, group_name), to
                transform input matches to a canonical form as a value_id.
                Where group_name will be None and the full match text will be
                passed in if there are no group names. Note that the canonical form
                is computed before the match_validator is applied and its value
                will be found as the value to the <auth_id> key.
            auth_anns_builder: The authority annotations row builder to use
                for building annotation rows.
            authdata: The authority data.
            field_groups: The derived field groups to use.
            anns_validator: A validation function for each regex match
                formed as a list of annotation row dictionaries, one row dictionary
                for each matching regex group. If the validator returns False,
                then the annotation rows will be rejected. The entity_text key
                will hold matched text and the <auth_name>_field key will hold
                the group name or number (if there are groups with or without names)
                or the <auth_name> if there are no groups in the regular expression.
                Note that the validator function takes the regex authority instance
                as its first parameter to provide access to the field_groups, etc.
                The validation_fn signature is: fn(regexAuthority, ann_row_dicts)
                and returns a boolean.
            parent_auth: This authority's parent authority (if any).
        """
        super().__init__(
            name,
            auth_anns_builder=auth_anns_builder,
            authdata=authdata,
            field_groups=field_groups,
            anns_validator=anns_validator,
            parent_auth=parent_auth,
        )
        self.regex = regex
        self.canonical_fn = canonical_fn

    def has_value(self, value: Any) -> re.Match:
        """Determine whether the given value is in this authority.

        Args:
            value: A possible authority value.

        Returns:
            None if the value is not a valid entity value; otherwise,
            return the re.Match object.
        """
        return self.regex.match(str(value))

    def add_annotations(
        self,
        text_obj: dk_annots.AnnotatedText,
    ) -> dk_annots.Annotations:
        """Method to do the work of finding, validating, and adding annotations.

        Args:
            text_obj: The annotated text object to process and add annotations.

        Returns:
            The added Annotations.
        """
        for match in re.finditer(self.regex, text_obj.text):
            ann_dicts = []
            if match.lastindex is not None:
                if len(self.regex.groupindex) > 0:  # we have named groups
                    for group_name, group_num in self.regex.groupindex.items():
                        group_text = match.group(group_num)
                        kwargs = {self.field_groups.get_field_type_col(self.name): group_name}
                        ann_dicts.append(
                            self.build_annotation(
                                start_pos=match.start(group_name),
                                end_pos=match.end(group_name),
                                entity_text=group_text,
                                auth_value_id=self.get_canonical_form(group_text, group_name),
                                **kwargs,
                            )
                        )
                else:  # we have only numbers for groups
                    for group_num, group_text in enumerate(match.groups()):
                        group_num += 1
                        kwargs = {self.field_groups.get_field_type_col(self.name): group_num}
                        ann_dicts.append(
                            self.build_annotation(
                                start_pos=match.start(group_num),
                                end_pos=match.end(group_num),
                                entity_text=group_text,
                                auth_value_id=self.get_canonical_form(group_text, group_num),
                                **kwargs,
                            )
                        )
            else:  # we have no groups
                ann_dicts.append(
                    self.build_annotation(
                        start_pos=match.start(),
                        end_pos=match.end(),
                        entity_text=match.group(),
                        auth_value_id=self.get_canonical_form(match.group(), self.name),
                    )
                )
            if self.validate_ann_dicts(ann_dicts):
                # Add non-empty, valid annotation dicts to the result
                text_obj.annotations.add_dicts(ann_dicts)
        return text_obj.annotations

    def get_canonical_form(self, entity_text: str, entity_type: str) -> Any:
        if self.canonical_fn is not None:
            entity_text = self.canonical_fn(entity_text, entity_type)
        return entity_text


class AuthoritiesBundle(Authority):
    """An authority for expressing values through multiple bundled "authorities"
    like dictionary-based and/or multiple regular expression patterns.
    """

    def __init__(
        self,
        name: str,
        auth_anns_builder: AuthorityAnnotationsBuilder = None,
        authdata: AuthorityData = None,
        field_groups: DerivedFieldGroups = None,
        parent_auth: "Authority" = None,
        anns_validator: Callable[["Authority", Dict[str, Any]], bool] = None,
        auths: List[Authority] = None,
    ):
        """Initialize the AuthoritiesBundle.

        Args:
            name: This authority's entity name.
            auth_anns_builder: The authority annotations row builder to use
                for building annotation rows.
            authdata: The authority data.
            field_groups: The derived field groups to use.
            anns_validator: fn(auth, anns_dict_list) that returns True if
                the list of annotation row dicts are valid to be added as
                annotations for a single match or "entity".
            parent_auth: This authority's parent authority (if any).
            auths: The authorities to bundle together.
        """
        super().__init__(
            name,
            auth_anns_builder=auth_anns_builder,
            authdata=authdata,
            field_groups=field_groups,
            anns_validator=anns_validator,
            parent_auth=parent_auth,
        )
        self.auths = auths.copy() if auths is not None else []

    def add(self, auth: Authority):
        """Add the authority to this bundle.

        Args:
            auth: The authority to add.
        """
        self.auths.append(auth)

    def has_value(self, value: Any) -> bool:
        """Determine whether the given value is in this authority.

        Args:
            value: A possible authority value.

        Returns:
            True if the value is a valid entity value.
        """
        for auth in self.auths:
            if auth.has_value(value):
                return True
        return False

    def add_annotations(
        self,
        text_obj: dk_annots.AnnotatedText,
    ) -> dk_annots.Annotations:
        """Method to do the work of finding, validating, and adding annotations.

        Args:
            text_obj: The annotated text object to process and add annotations.

        Returns:
            The added Annotations.
        """
        for auth in self.auths:
            auth.annotate_input(text_obj)
        return text_obj.annotations

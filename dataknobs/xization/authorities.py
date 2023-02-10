import pandas as pd
import re
import dataknobs.xization.annotations as annots
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Set


class DerivedFieldGroups(annots.DerivedAnnotationColumns):
    
    def __init__(
            self,
            field_type_suffix: str = '_field',
            field_group_suffix: str = '_num',
            field_record_suffix: str = '_recsnum',
    ):
        '''
        Add derived column types/names: Given an annnotation row,
          * field_type(row) == f'{row[ann_type_col]}_field'
          * field_group(row) == f'{row[ann_type_col]}_num'
          * field_record(row) == f'{row[ann_type_col])_recsnum'

        Where:
          * A field_type column holds annotation "sub"- type values, or fields
          * A field_group column identifies groups of annotation fields
          * A field_record column identifies groups of annotation field groups
        
        :param field_type_suffix: The field_type col name suffix (if not _field)
        :param field_group_suffix: The field_group col name suffix (if not _num)
        :param field_record_suffix: field_record colname sfx (if not _recsnum)
        '''
        self.field_type_suffix = field_type_suffix
        self.field_group_suffix = field_group_suffix
        self.field_record_suffix = field_record_suffix

    def get_col_value(
            self,
            row_accessor: annots.AnnotationsRowAccessor,
            col_type: str,
            row: pd.Series,
            missing: str = None,
    ) -> str:
        '''
        Get the value of the column in the given row derived from col_type,
        where col_type is one of:
          * "field_type" == f"{field}_field"
          * "field_group" == f"{field}_num"
          * "field_record" == f"{field}_recsnum"

        And "field" is the row_accessor's metadata's "ann_type" col's value.

        :param row_accessor: The AnnotationsRowAccessor
        :param col_type: The type of column value to derive
        :param row: A row from which to get the value.
        :param missing: The value to return for unknown or missing column
        :return: The row value or the missing value
        '''
        value = missing
        field = row_accessor.get_col_value('ann_type', row, None)
        if field is not None:
            if col_type == 'field_type':
                value = row_accessor.get_col_value(
                    self.get_field_type_col(field),
                    row, missing=missing
                )
            elif col_type == 'field_group':
                value = row_accessor.get_col_value(
                    self.get_field_group_col(field),
                    row, missing=missing
                )
            elif col_type == 'field_record':
                value = row_accessor.get_col_value(
                    self.get_field_record_col(field),
                    row, missing=missing
                )
        return value

    def get_field_type(
            self,
            row_accessor: annots.AnnotationsRowAccessor,
            row: pd.Series,
    ) -> str:
        '''
        Get a label for the field type, or attribute, of the row.

        :param row_accessor: The AnnotationsRowAccessor
        :param row: A row whose field_type to get.
        '''
        result = row.name
        group_type = row_accessor.get_col_value('ann_type', row, None)
        field_type = row_accessor.get_col_value('field_type', row, None)
        if group_type is not None:
            if field_type is None:
                result = group_type
            else:
                result = f'{group_type}.{field_type}'
        elif field_type is not None:
            result = field_type
        return result

    def unpack_field(self, field_value: str) -> str:
        '''
        Given a field in any of its derivatives (like field type, field group
        or field record,) unpack and return the basic field value itself.
        '''
        field = field_value
        if field.endswith(self.field_record_suffix):
            field = field.replace(self.field_record_suffix, '')
        elif field.endswith(self.field_group_suffix):
            field = field.replace(self.field_group_suffix, '')
        elif field.endswith(self.field_type_suffix):
            field = field.replace(self.field_type_suffix, '')
        return field

    def get_field_name(self, field_value: str) -> str:
        '''
        Given a field name or field col name, e.g., an annotation type col's
        value (the field name); or a field type, group, or record column name,
        get the field name.
        '''
        return self.unpack_field(field_value)

    def get_field_type_col(self, field_value: str) -> str:
        '''
        Given a field name or field col name, e.g., an annotation type col's
        value; or a field type, group, or record column name, get the field
        name.
        '''
        field = self.unpack_field(field_value)
        return f'{field}{self.field_type_suffix}'

    def get_field_group_col(self, field_value: str) -> str:
        '''
        Given a field name or field col name, e.g., an annotation type col's
        value; or a field type, group, or record, get the name of the derived
        field group column.
        '''
        field = self.unpack_field(field_value)
        return f'{field}{self.field_group_suffix}'

    def get_field_record_col(self, field_value: str) -> str:
        '''
        Given a field name or field col name, e.g., an annotation type col's
        value; or a field type, group, or record, get the name of the derived
        field record column.
        '''
        field = self.unpack_field(field_value)
        return f'{field}{self.field_record_suffix}'


class Authority(ABC):
    '''
    A class for managing and defining tabular authoritative data for e.g.,
    taxonomies, etc., and using them to annotate instances within text.
    '''
    def __init__(
            self,
            name:str,
            auth_name_colname: str = 'auth_name',
            auth_value_id_colname: str = 'auth_value_id',
            parent_auth:'Authority' = None,
    ):
        '''
        Initialize with this authority's metadata.
        :param name: This authority's entity name
        :param auth_name_colname: The name of the annotation column for this
            authority's name. If None, then auth_name's will not automatically
            be added to annotations.
        :param auth_value_id_colname: The name of the annotation column for
            this authority's value_id. If None, then auth_value_id's will not
            automatically be added to annotations.
        :param parent_auth: This authority's parent authority (if any)
        '''
        self._name = name
        self.auth_name_colname = auth_name_colname  # MIGRATION: Add to data_defaults (ann_type=name)
        self.auth_value_id_colname = auth_value_id_colname
        self._parent = parent_auth

    @property
    def name(self) -> str:
        '''
        Get the name of this authority, which is usually the name or type
        of entities defined herein.
        '''
        return self._name

    @property
    def parent(self) -> 'Authority':
        '''
        Get this authority's parent, or None.
        '''
        return self._parent

    @abstractmethod
    def has_value(self, value: Any) -> bool:
        '''
        Determine whether the given value is in this authority.
        :param value: A possible authority value.
        :return: True if the value is a valid entity value.
        '''
        raise NotImplementedError

    @abstractmethod
    def annotate_text(
            self,
            input_text: str,
            input_id: Any = None,
            annotations: annots.Annotations = None,
    ) -> annots.Annotations:
        '''
        Find and annotate this authority's entities in the input text
        as dictionaries like:
        [
            {
                'input_id': <id>,
                'start_pos': <start_char_pos>,
                'end_pos': <end_char_pos>,
                'entity_text': <entity_text>,
                'confidence': <confidence_if_available>,
                '<auth_name_col>': <authority_name>,
                '<auth_value_id>': <value_id_or_canonical_form>,
            },
        ]
        :param input_text: The text to process.
        :param input_id: The id of the input text
        :param annotations: The annotations object to add annotations to
        :return: The given or a new Annotations instance
        '''
        raise NotImplementedError

    def compose(
            self,
            annotations: annots.Annotations,
    ) -> annots.Annotations:
        '''
        Compose annotations into groups.
        :param annotations: The annotations
        :return: composed annotations
        '''
        return annotations

    def build_annotation(  # MIGRATION: Moving to AnnotationsBuilder
            self,
            input_id: Any = None,  # MIGRATION: Add to kwargs
            start_pos: int = None,
            end_pos: int = None,
            entity_text: str = None,
            auth_value_id: Any = None,  # MIGRATION: Add to AnnotationsBuilder.build_annotation_row kwargs
            conf: float = 1.0,  # MIGRATION: Add to data_defaults AND kwargs
            **kwargs,
    ) -> Dict[str, Any]:
        '''
        Build annotations with the given components.
        '''
        result = {
            'input_id': input_id,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'entity_text': entity_text,
            'confidence': conf,
        }
        self.add_metadata_to_annotation(result, auth_value_id=auth_value_id)
        if kwargs is not None:
            result.update(kwargs)
        return result

    def add_metadata_to_annotation(
            self,
            adict:Dict[str, Any],
            auth_value_id: Any = None,
    ):
        if self.parent:
            self.parent.add_metadata_to_annotation(adict)
            # MIGRATION: Instead, make sure AnnotationsBuilder.data_defaults has correct (e.g., parent or self) auth_name_colname
        if self.auth_name_colname and self.name:  # MIGRATION: Add to data_defaults
            adict[self.auth_name_colname] = self.name
        if self.auth_value_id_colname and auth_value_id:  # MIGRATION: Add to data_defaults
            adict[self.auth_value_id_colname] = auth_value_id


class AuthorityData:
    '''
    A wrapper for authority data.
    '''
    def __init__(self, df: pd.DataFrame, name: str):
        self._df = df
        self.name = name

    @property
    def df(self) -> pd.DataFrame:
        '''
        Get the authority data in a dataframe
        '''
        return self._df

    def lookup_values(self, value: Any, is_id=False) -> pd.DataFrame:
        '''
        Lookup authority value(s) for the given value or value id.
        :param value: A value or value_id for this authority
        :param is_id: True if value is an ID
        :return: The applicable authority dataframe rows.
        '''
        col = self.df.index if is_id else self.df[self.name]
        return self.df[col == value]


class AuthorityFactory(ABC):
    '''
    A factory class for building an authority.
    '''
    @abstractmethod
    def build_authority(
            self,
            name: str,
            authdata: AuthorityData,
            parent_auth: Authority = None,
    ) -> Authority:
        '''
        Build an authority with the given name and data.
        :param name: The authority name
        :param authdata: The authority data
        :param parent_auth: The parent authority.
        :return: The authority
        '''
        raise NotImplementedError


class LexicalAuthority(Authority):
    '''
    A class for managing named entities by ID with associated values and
    variations.
    '''
    def __init__(
            self,
            name:str,
            auth_name_colname: str = 'auth_name',
            auth_value_id_colname: str = 'auth_value_id',
            parent_auth:'Authority' = None,
    ):
        '''
        Initialize with this authority's metadata.
        :param name: This authority's entity name
        :param auth_name_colname: The name of the annotation column for this
            authority's name. If None, then auth_name's will not automatically
            be added to annotations.
        :param auth_value_id_colname: The name of the annotation column for
            this authority's value_id. If None, then auth_value_id's will not
            automatically be added to annotations.
        :param parent_auth: This authority's parent authority (if any)
        '''
        super().__init__(
            name,
            auth_name_colname=auth_name_colname,
            auth_value_id_colname=auth_value_id_colname,
            parent_auth=parent_auth,
        )

    @abstractmethod
    def get_value_ids(self, value:Any) -> Set[Any]:
        '''
        Get all IDs associated with the given value. Note that typically
        there is a single ID for any value, but this allows for inherent
        ambiguities in the authority.
        :param value: An authority value
        :return: The associated IDs or an empty set if the value is not valid.
        '''
        raise NotImplementedError

    @abstractmethod
    def get_values_by_id(self, value_id:Any) -> Set[Any]:
        '''
        Get all values for the associated value ID. Note that typically
        there is a single value for an ID, but this allows for inherent
        ambiguities in the authority.
        
        :param value: An authority value
        :return: The associated IDs or an empty set if the value is not valid.
        '''
        raise NotImplementedError

    @abstractmethod
    def get_id_by_variation(self, variation:str) -> Set[str]:
        '''
        Get the IDs of the value(s) associated with the given variation.
        :param variation: Variation text
        :return: The possibly empty set of associated value IDS.
        '''
        raise NotImplementedError

    @abstractmethod
    def find_variations(
            self,
            variation: str,
            starts_with: bool = False,
            ends_with: bool =False,
            scope: str = 'fullmatch',
    ) -> pd.Series:
        '''
        Find all matches to the given variation.
        :param variation: The text to find; treated as a regular expression
            unless either starts_with or ends_with is True.
        :param starts_with: When True, find all terms that start with the
            variation text.
        :param ends_with: When True, find all terms that end with the variation
            text.
        :param scope: 'fullmatch' (default), 'match', or 'contains' for
            strict, less strict, and least strict matching
        :param category_constraints: When present, limit results to terms with
            the given constraints.
        :return: The matching variations as a pd.Series

        Note only the first true of starts_with, ends_with, and scope will
        be applied. If none of these are true, an full match on the pattern
        is performed.
        '''
        raise NotImplementedError


class RegexAuthority(Authority):
    '''
    A class for managing named entities by ID with associated values and
    variations.
    '''
    def __init__(
            self,
            name:str,
            regex:re.Pattern,
            group_names: List[str] = None,
            canonical_fn:Callable[[str, str], Any] = None,
            auth_name_colname: str = 'auth_name',
            auth_value_id_colname: str = 'auth_value_id',
            parent_auth:'Authority' = None,
            group_name_colname: str = 'regex_group',  # MIGRATION: use DerivedAnnotationColumns
    ):
        '''
        Initialize with this authority's entity name.
        :param name: The authority name
        :param regex: The regular expression to apply
        :param canonical_fn: A function, fn(match_text, group_name), to
            transform input matches to a canonical form as a value_id.
            Where group_name will be None and the full match text will be
            passed in if there are no group names.
        :param auth_name_colname: The name of the annotation column for this
            authority's name. If None, then auth_name's will not automatically
            be added to annotations.
        :param auth_value_id_colname: The name of the annotation column for
            this authority's value_id. If None, then auth_value_id's will not
            automatically be added to annotations.
        :param parent_auth: This authority's parent authority (if any)
        :param group_name_colname: The name of the annotations column for
            the regex group names, or None to ignore group_names.

        NOTE: If the regular expression has capturing groups, each group
        will result in a separate entity, with the group name if provided
        in the regular expression as ...(?P<group_name>group_regex)...
        '''
        super().__init__(
            name,
            auth_name_colname=auth_name_colname,
            auth_value_id_colname=auth_value_id_colname,
            parent_auth=parent_auth,
        )
        self.regex = regex
        self.canonical_fn = canonical_fn
        self.group_name_colname = group_name_colname

    def has_value(self, value: Any) -> re.Match:
        '''
        Determine whether the given value is in this authority.
        :param value: A possible authority value.
        :return: True if the value is a valid entity value.

        In the case of this regex authority, this will the match object
        if the regex matches the value; otherwise None.
        '''
        return self.regex.match(str(value))

    def annotate_text(
            self,
            input_text: str,
            input_id: Any = None,
            annotations: annots.Annotations = None,
    ) -> annots.Annotations:
        '''
        Find and annotate this authority's entities in the input text
        as dictionaries like:
        [
            {
                'input_id': <id>,
                'start_pos': <start_char_pos>,
                'end_pos': <end_char_pos>,
                'entity_text': <entity_text>,
                'confidence': <confidence_if_available>,
                '<auth_name_col>': <authority_name>,
                '<auth_value_id>': <value_id_or_canonical_form>,
            },
        ]
        :param input_text: The text to process.
        :param input_id: The id of the input text
        :param annotations: The annotations object to add annotations to
        :return: The given or a new Annotations instance
        '''
        result = (
            annotations if annotations is not None
            else annots.Annotations(annots.AnnotationsMetaData())
        )
        for match in re.finditer(self.regex, input_text):
            if match.lastindex is not None:
                for group_name, group_text in self.regex.groupindex.items():
                    result.add_dict(self.build_annotation(
                        input_id=input_id,
                        start_pos=match.start(group_name),
                        end_pos=match.end(group_name),
                        entity_text=group_text,
                        auth_value_id=self.get_canonical_form(group_text, group_name),
                        entity_type=group_name,
                    ))
            else:
                result.add_dict(self.build_annotation(
                    input_id=input_id,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    entity_text=match.group(),
                    auth_value_id=self.get_canonical_form(match.group(), None),
                ))
        return result

    def get_canonical_form(self, entity_text:str, entity_type:str) -> Any:
        if self.canonical_fn is not None:
            entity_text = self.canonical_fn(entity_text, entity_type)
        return entity_text

    def build_annotation(
            self,
            input_id: Any = None,
            start_pos: int = None,
            end_pos: int = None,
            entity_text: str = None,
            auth_value_id: Any = None,
            conf: float = 1.0,
            entity_type: str = None,
            **kwargs,
    ):
        '''
        Override build annotations to add matched group names as an entity type
        '''
        result = super().build_annotation(
            input_id=input_id,
            start_pos=start_pos,
            end_pos=end_pos,
            entity_text=entity_text,
            auth_value_id=auth_value_id,
            conf=conf,
            **kwargs,
        )
        if self.group_name_colname and entity_type:
            result[self.group_name_colname] = entity_type


class AuthoritiesBundle(Authority):
    '''
    An authority for expressing values through multiple bundled "authorities"
    like dictionary-based and/or multiple regular expression patterns.
    '''

    def __init__(
            self,
            name:str,
            auth_name_colname: str = 'auth_name',
            auth_value_id_colname: str = 'auth_value_id',
            parent_auth:'Authority' = None,
            auths: List[Authority] = None,
    ):
        '''
        :param name: This authority's entity name
        :param auth_name_colname: The name of the annotation column for this
            authority's name. If None, then auth_name's will not automatically
            be added to annotations.
        :param auth_value_id_colname: The name of the annotation column for
            this authority's value_id. If None, then auth_value_id's will not
            automatically be added to annotations.
        :param parent_auth: This authority's parent authority (if any)
        :param auths: The authorities to bundle together.
        '''
        super().__init__(
            name, name, auth_value_id_colname, parent_auth
        )
        self.auths = auths.copy() if auths is not None else list()

    def add(self, auth: Authority):
        '''
        Add the authority to this bundle
        :param auth: The authority to add.
        '''
        self.auths.append(auth)

    def has_value(self, value: Any) -> bool:
        '''
        Determine whether the given value is in this authority.
        :param value: A possible authority value.
        :return: True if the value is a valid entity value.
        '''
        for auth in self.auths:
            if auth.has_value(value):
                return True
        return False

    def annotate_text(
            self,
            input_text: str,
            input_id: Any = None,
            annotations: annots.Annotations = None,
    ) -> annots.Annotations:
        '''
        Find and annotate this authority's entities in the input text
        as dictionaries like:
        [
            {
                'input_id': <id>,
                'start_pos': <start_char_pos>,
                'end_pos': <end_char_pos>,
                'entity_text': <entity_text>,
                'confidence': <confidence_if_available>,
                '<auth_name_col>': <authority_name>,
                '<auth_value_id>': <value_id_or_canonical_form>,
            },
        ]
        :param input_text: The text to process.
        :param input_id: The id of the input text
        :param annotations: The annotations object to add annotations to
        :return: The given or a new Annotations instance
        '''
        result = (
            annotations if annotations is not None
            else annots.Annotations(annots.AnnotationsMetaData())
        )
        for auth in self.auths:
            auth.annotate_text(input_text, input_id, result)
        return result

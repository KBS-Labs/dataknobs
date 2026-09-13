"""Tests for dataknobs_xization.lexicon.

Scoped to the authority *data* containers -- ``MultiAuthorityData`` and the
masks built over it. The annotation path that reads them is covered by
``test_lexicon_annotation.py``; extending either file is welcome.
"""

import pandas as pd
import pytest

import dataknobs_xization.authorities as dk_auth
import dataknobs_xization.lexicon as dk_lex


def _unique_vals(values, dtype):
    return dk_lex.MultiAuthorityData.get_unique_vals_df(pd.Series(values, dtype=dtype), "col")


# --- integer columns: IDs are the integers themselves ----------------------


@pytest.mark.parametrize("dtype", ["int64", "int32", "uint8", "uint64", "Int64"])
def test_integer_column_indexes_by_value(dtype):
    """Integer columns — including unsigned and the nullable extension dtype — index by value."""
    df = _unique_vals([3, 1, 2, 1], dtype)

    assert df["col"].tolist() == [1, 2, 3]
    assert df.index.tolist() == [1, 2, 3]


def test_integer_column_drops_missing_values():
    df = _unique_vals([3, None, 1, None], "Int64")

    assert df["col"].tolist() == [1, 3]
    assert df.index.tolist() == [1, 3]


# --- non-integer columns: IDs are auto-generated 0..n-1 --------------------


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([2.5, 1.5, 2.5], "float64", [1.5, 2.5]),
        ([2.5, 1.5, 2.5], "Float64", [1.5, 2.5]),
        ([True, False, True], "bool", [False, True]),
        ([True, False, True], "boolean", [False, True]),
        (["b", "a", "b"], "category", ["a", "b"]),
        (["b", "a", "b"], "string", ["a", "b"]),
        (["b", "a", "b"], "object", ["a", "b"]),
    ],
)
def test_non_integer_column_uses_positional_ids(values, dtype, expected):
    """Non-integer columns — including extension dtypes — get a 0..n-1 index."""
    df = _unique_vals(values, dtype)

    assert df["col"].tolist() == expected
    assert df.index.tolist() == list(range(len(expected)))


# --- temporal columns ------------------------------------------------------


def test_datetime_column_uses_positional_ids():
    """``datetime64`` is not an integer dtype under either predicate."""
    df = _unique_vals(pd.to_datetime(["2020-01-02", "2020-01-01"]), "datetime64[ns]")

    assert df["col"].tolist() == pd.to_datetime(["2020-01-01", "2020-01-02"]).tolist()
    assert df.index.tolist() == [0, 1]


def test_timedelta_column_uses_positional_ids():
    """``timedelta64`` takes the positional branch — the one silent branch change.

    ``np.timedelta64`` subclasses ``np.signedinteger``, so the previous
    ``np.issubdtype(col.dtype, np.integer)`` test returned ``True`` for a
    timedelta column and used the raw timedelta values as row IDs.
    ``pd.api.types.is_integer_dtype`` returns ``False``, so such a column now
    gets a 0..n-1 index like every other non-integer column.

    This is the *only* dtype whose branch changed without previously raising —
    every other behavioural difference replaced a ``TypeError`` with a result.
    It is pinned here so the switch stays deliberate.
    """
    df = _unique_vals(pd.to_timedelta([2, 1], unit="D"), "timedelta64[ns]")

    assert df["col"].tolist() == pd.to_timedelta([1, 2], unit="D").tolist()
    assert df.index.tolist() == [0, 1]


def test_non_integer_column_drops_missing_values():
    df = _unique_vals(["b", None, "a"], "string")

    assert df["col"].tolist() == ["a", "b"]
    assert df.index.tolist() == [0, 1]


def test_column_name_is_applied():
    df = dk_lex.MultiAuthorityData.get_unique_vals_df(
        pd.Series([1, 2], dtype="int64"), "authority_id"
    )

    assert list(df.columns) == ["authority_id"]


# --- peeking at built sub-authority data -----------------------------------


class _StubAuthority(dk_auth.AuthorityData):
    """Minimal real AuthorityData, so the container is exercised with the type it holds."""

    def __init__(self, name):
        super().__init__(pd.DataFrame({"value": [1]}), name)


class _CountingMultiAuthority(dk_lex.MultiAuthorityData):
    """MultiAuthorityData with the one abstract method filled in, counting builds."""

    def __init__(self, df, name):
        super().__init__(df, name)
        self.builds = []

    def build_authority_data(self, name):
        self.builds.append(name)
        return _StubAuthority(name)


def _multi():
    return _CountingMultiAuthority(pd.DataFrame({"a": [1, 2]}), "top")


def test_peek_returns_none_before_the_sub_authority_is_built():
    """Bug: this accessor was a @property with a required parameter.

    A property getter is invoked by attribute access with no arguments, so
    ``authority_data`` raised TypeError on every access and the "retrieve
    without building" capability did not exist at all -- while its docstring
    said it returned None when absent and its annotation said it could not.
    """
    multi = _multi()

    assert multi.peek_authority_data("missing") is None
    assert multi.builds == [], "peeking must not build"


def test_peek_returns_the_built_object_without_rebuilding():
    multi = _multi()
    built = multi.get_authority_data("colour")

    assert multi.peek_authority_data("colour") is built
    assert multi.builds == ["colour"], "peeking after a build must not build again"


# --- masking against a sub-authority that has not been built ---------------


def test_lookup_subauth_values_is_none_when_the_sub_authority_is_unbuilt():
    """``lookup_subauth_values`` peeks rather than builds, so None is its normal answer.

    Its annotation said ``-> pd.DataFrame`` while its body initialized the
    result to None and returned it untouched whenever the peek missed --
    which, because the peek deliberately does not build, is the default state
    rather than an edge case.
    """
    multi = _multi()

    assert multi.lookup_subauth_values("a", 1, is_id=True) is None
    assert multi.builds == [], "looking up sub-values must not build"


def test_auth_values_mask_is_all_false_when_the_sub_authority_is_unbuilt():
    """Bug: this raised TypeError on the default path.

    ``auth_values_mask`` subscripted ``lookup_subauth_values``'s result
    without checking it, so an unbuilt sub-authority produced
    ``TypeError: 'NoneType' object is not subscriptable``. No sub-authority
    values means no record carries one, which is an all-False mask -- and
    ``auth_records_mask`` conjoins these, where all-False correctly excludes
    every record rather than exploding.
    """
    multi = _multi()

    mask = multi.auth_values_mask("a", 1)

    assert not mask.any(), "no sub-authority values means no record can match"
    assert mask.index.equals(multi.df.index), "the mask must align with the authority rows"


def test_auth_records_mask_conjoins_an_unbuilt_field_without_raising():
    """The consumer-visible half: the only caller of auth_values_mask."""
    multi = _multi()

    mask = multi.auth_records_mask({"a": 1})

    assert not mask.any()


# --- the masks that answer None ---------------------------------------------


def test_auth_records_mask_is_none_when_no_fields_are_named():
    """No fields named and no pre-filter means nothing to conjoin, which is None.

    The docstring said so and the annotation said ``-> pd.Series``; pinned here
    because the annotation now agrees with both.
    """
    multi = _multi()

    assert multi.auth_records_mask({}) is None


def test_auth_records_mask_starts_from_the_filter_mask():
    """With no fields named, the pre-filter is the whole answer."""
    multi = _multi()
    prefilter = pd.Series([True, False], index=multi.df.index)

    assert multi.auth_records_mask({}, filter_mask=prefilter) is prefilter


def test_combine_masks_is_none_when_the_masks_select_nothing_together():
    """An empty conjunction is reported as None rather than as an all-False mask."""
    multi = _multi()
    first = pd.Series([True, False], index=multi.df.index)
    second = pd.Series([False, True], index=multi.df.index)

    assert multi.combine_masks(first, second) is None
    assert multi.combine_masks(first, None).tolist() == [True, False]
    assert multi.combine_masks(None, second).tolist() == [False, True]
    assert multi.combine_masks(None, None) is None


def test_asking_for_the_sub_authority_names_says_they_are_not_supplied():
    """Bug: the base class answered None to a question annotated ``List[str]``.

    ``sub_authority_names`` was the one member of ``CorrelatedAuthorityData``
    that did not say it was unimplemented -- its four siblings all raise --
    and it returned None instead. A caller iterating the result got
    ``TypeError: 'NoneType' object is not iterable`` from its own frame, and a
    caller testing it for truth got a silent "this data correlates no
    sub-authorities", which is the one thing a ``CorrelatedAuthorityData``
    cannot be. Nothing in the tree calls it and nothing overrides it, so the
    lie had never been collected.
    """
    multi = _multi()

    with pytest.raises(NotImplementedError):
        multi.sub_authority_names()


# --- what the factory is given, and what it passes on -----------------------


def _flat(name="animal"):
    return dk_auth.AuthorityData(pd.DataFrame({name: ["dog", "cat"]}), name)


def test_the_factory_builds_from_the_data_its_base_declares():
    """Bug: the only ``AuthorityFactory`` refused its base's declared type.

    ``AuthorityFactory`` publishes ``build_authority(name, builder, authdata)``
    over an ``AuthorityData``. ``MultiAuthorityFactory`` opened its body with
    a lookup declared only on ``MultiAuthorityData``, so a consumer holding
    the abstract -- which is what publishing an abstract factory invites --
    got ``AttributeError: 'AuthorityData' object has no attribute
    'get_authority_data'`` from inside the factory.
    """
    factory = dk_lex.MultiAuthorityFactory("animal")
    authdata = _flat()

    authority = factory.build_authority("animal", dk_auth.AuthorityAnnotationsBuilder(), authdata)

    assert isinstance(authority, dk_lex.DataframeAuthority)
    assert authority.authdata is authdata
    assert authority.has_value("dog")


def test_a_container_still_resolves_the_sub_authority_rather_than_itself():
    """The control: the leaf answer must not displace the container's.

    The same call against ``MultiAuthorityData`` still builds the named
    "sub" authority, so the fix above widens what the factory accepts without
    changing what it does with the data it already accepted.
    """
    factory = dk_lex.MultiAuthorityFactory("animal")
    multi = dk_lex.SimpleMultiAuthorityData(
        pd.DataFrame({"animal": ["dog", "cat"], "colour": ["red", "blue"]}), "animals"
    )

    authority = factory.build_authority("animal", dk_auth.AuthorityAnnotationsBuilder(), multi)

    assert authority.authdata is multi.peek_authority_data("animal")
    assert authority.authdata is not multi


def test_the_factory_passes_on_the_annotations_builder_it_was_handed():
    """Bug: the builder the caller passed was accepted and discarded.

    ``auth_anns_builder`` is the abstract factory's own second parameter, and
    the only implementation named it, documented it, and then built the
    authority without it -- so every authority built through the factory used
    a fresh default builder, and a consumer's annotation metadata (column
    names, id column, anything the builder carries) silently did not apply.
    """

    class _Marked(dk_auth.AuthorityAnnotationsBuilder):
        pass

    builder = _Marked()
    factory = dk_lex.MultiAuthorityFactory("animal")

    authority = factory.build_authority("animal", builder, _flat())

    assert authority.anns_builder is builder


def test_the_factory_passes_on_the_field_groups_it_was_configured_with():
    """Gap: ``field_groups`` was hardcoded ``None``, asking where it came from.

    ``DataframeAuthority`` accepts derived field groups and the factory could
    not supply them, so every authority the only shipped factory built used
    the default suffixes -- with no way for a consumer holding the factory to
    say otherwise.
    """
    field_groups = dk_auth.DerivedFieldGroups(field_type_suffix="_kind")
    factory = dk_lex.MultiAuthorityFactory("animal", field_groups=field_groups)

    authority = factory.build_authority("animal", dk_auth.AuthorityAnnotationsBuilder(), _flat())

    assert authority.field_groups is field_groups


def test_the_factory_passes_on_the_anns_validator_it_was_configured_with():
    """Gap: ``anns_validator`` was hardcoded ``None``, asking where it came from.

    An authority with no validator accepts every match it finds. The factory
    could not give one to the authority it built, so the validation the
    annotation path is written around was unreachable through the factory
    door.
    """

    def reject_everything(auth, ann_dicts):
        return False

    factory = dk_lex.MultiAuthorityFactory("animal", anns_validator=reject_everything)

    authority = factory.build_authority("animal", dk_auth.AuthorityAnnotationsBuilder(), _flat())

    assert authority.anns_validator is reject_everything
    assert authority.annotate_input("a dog barked").is_empty()


def test_the_factory_defaults_leave_the_authority_as_it_was():
    """The control for both gaps: unconfigured, the factory builds what it did.

    ``Authority`` substitutes a default ``DerivedFieldGroups`` for None, and
    no validator means every match is kept.
    """
    factory = dk_lex.MultiAuthorityFactory("animal")

    authority = factory.build_authority("animal", dk_auth.AuthorityAnnotationsBuilder(), _flat())

    assert isinstance(authority.field_groups, dk_auth.DerivedFieldGroups)
    assert authority.anns_validator is None
    assert authority.annotate_input("a dog barked").df["text"].tolist() == ["dog"]

import io
import json
import pytest
import dataknobs.utils.json_utils as jutils


def test_count_uniques(test_json_001):
    jdata = test_json_001
    builder = jutils.JsonSchemaBuilder(jdata, keep_unique_values=True)
    schema = builder.schema
    df = schema.df
    assert df['unique_count'].to_list() == [1, 1, 4, 4, 4, 4, 3, 2, 2]


def test_count_limited_uniques(test_json_001):
    jdata = test_json_001
    builder = jutils.JsonSchemaBuilder(jdata, keep_unique_values=1)
    schema = builder.schema
    df = schema.df
    assert df['unique_count'].to_list() == [1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_invert_values(test_json_001):
    jdata = test_json_001
    builder = jutils.JsonSchemaBuilder(
        jdata, keep_unique_values=True, invert_uniques=True
    )
    schema = builder.schema
    df = schema.df
    assert df['unique_count'].to_list() == [1, 1, 4, 4, 4, 4, 3, 2, 2]
    assert schema.values.path_values['.c[].A[].i.k'][111]._indices == '(2 (0 0) (1 1))'
    assert schema.values.path_values['.c[].A[].i.k'][112]._indices == '(1 (0 1))'


def test_squash_and_explode_data1(test_json_001):
    jdata = test_json_001
    squashed = jutils.collect_squashed(
        jdata,
        prune_at=['i', "B"],
    )
    assert max(squashed.values()) < 100
    exploded = jutils.explode(squashed)
    j = json.dumps(exploded)
    assert j == '{"a": 1, "b": 2, "c": [{"A": [{"e": [{"f": 3, "g": 4, "h": 5}]}, {"e": [{"f": 8, "g": 9, "h": 10}]}]}, {"A": [{"e": [{"f": 15, "g": 16, "h": 17}]}, {"e": [{"f": 20, "g": 21, "h": 22}]}]}]}'


def test_squash_and_explode_data2(test_json_001):
    jdata = test_json_001
    squashed = jutils.collect_squashed(
        jdata,
        prune_at=['i'],
    )
    for val in squashed.values():
        assert val < 100 or val > 200
    exploded = jutils.explode(squashed)
    j = json.dumps(exploded)
    assert j == '{"a": 1, "b": 2, "c": [{"A": [{"e": [{"f": 3, "g": 4, "h": 5}]}, {"e": [{"f": 8, "g": 9, "h": 10}]}], "B": {"l": 13, "m": 222}}, {"A": [{"e": [{"f": 15, "g": 16, "h": 17}]}, {"e": [{"f": 20, "g": 21, "h": 22}]}], "B": {"l": 25, "m": 223}}]}'


def test_stream_record_paths1(test_json_001):
    df = jutils.get_records_df(test_json_001)
    assert len(df['rec_id'].unique()) == 2
    assert all(df[df['rec_id'] == 0]['line_num'] == range(14))
    assert all(df[df['rec_id'] == 1]['line_num'] == [0, 1] + list(range(14, 26)))


def test_stream_record_paths2(test_json_002):
    df = jutils.get_records_df(test_json_002)
    assert len(df['rec_id'].unique()) == 2
    assert all(df[df['rec_id'] == 0]['line_num'] == range(22))
    assert all(df[df['rec_id'] == 1]['line_num'] == [0, 1] + list(range(22, 42)))


def test_stream_record_paths2(test_json_003):
    df = jutils.get_records_df(test_json_003)
    assert len(df['rec_id'].unique()) == 3
    assert all(df[df['rec_id'] == 0]['line_num'] == range(13))
    assert all(df[df['rec_id'] == 1]['line_num'] == [0, 1] + list(range(13, 24)))
    assert all(df[df['rec_id'] == 2]['line_num'] == [0, 1] + list(range(24, 68)))

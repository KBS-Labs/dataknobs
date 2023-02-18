import json
import pytest
import dataknobs.utils.json_utils as jutils


def test_count_uniques(test_json_001):
    jdata = test_json_001
    builder = jutils.JsonSchemaBuilder(jdata, keep_unique_values=True)
    schema = builder.schema
    df = schema.df
    assert df['unique_count'].to_list() == [1, 1, 4, 4, 4, 4, 3, 2, 2]


def test_squash_and_explode_data1(test_json_001):
    jdata = test_json_001
    squashed = jutils.squash_data(
        jdata,
        ['i', "B"]
    )
    assert max(squashed.values()) < 100
    exploded = jutils.explode(squashed)
    j = json.dumps(exploded)
    assert j == '{"a": 1, "b": 2, "c": [{"A": [{"e": [{"f": 3, "g": 4, "h": 5}]}, {"e": [{"f": 8, "g": 9, "h": 10}]}]}, {"A": [{"e": [{"f": 15, "g": 16, "h": 17}]}, {"e": [{"f": 20, "g": 21, "h": 22}]}]}]}'


def test_squash_and_explode_data2(test_json_001):
    jdata = test_json_001
    squashed = jutils.squash_data(
        jdata,
        ['i']
    )
    for val in squashed.values():
        assert val < 100 or val > 200
    exploded = jutils.explode(squashed)
    j = json.dumps(exploded)
    assert j == '{"a": 1, "b": 2, "c": [{"A": [{"e": [{"f": 3, "g": 4, "h": 5}]}, {"e": [{"f": 8, "g": 9, "h": 10}]}], "B": {"l": 13, "m": 222}}, {"A": [{"e": [{"f": 15, "g": 16, "h": 17}]}, {"e": [{"f": 20, "g": 21, "h": 22}]}], "B": {"l": 25, "m": 223}}]}'


def test_block_collector(test_json_001):
    jdata = test_json_001
    builder = jutils.JsonSchemaBuilder(
        jdata,
        value_typer=lambda value: 'tag' if value > 100 else None
    )
    schema = builder.schema
    tagged = {
        jq_path: schema.extract_values(jq_path, jdata, unique=True)
        for jq_path in schema.df[schema.df['value_type'] == 'tag']['jq_path']
    }
    
    expected = {
        0: {
            0: [{'.c[0].A[1].e[0].g': 9, '.c[0].A[1].e[0].h': 10, '.c[0].A[1].i.j': 11, '.c[0].A[1].i.k': 112}],
            1: [{'.c[1].A[0].e[0].f': 15, '.c[1].A[0].e[0].g': 16, '.c[1].A[0].e[0].h': 17, '.c[1].A[0].i.j': 18, '.c[1].A[0].i.k': 113}],
            2: [{'.c[0].A[0].e[0].f': 3, '.c[0].A[0].e[0].g': 4, '.c[0].A[0].e[0].h': 5, '.c[0].A[0].i.j': 6, '.c[0].A[0].i.k': 111}, {'.c[1].A[1].e[0].g': 21, '.c[1].A[1].e[0].h': 22, '.c[1].A[1].i.j': 23, '.c[1].A[1].i.k': 111}],
        },
        1: {
            0: [{'.c[0].A[0].e[0].f': 3, '.c[0].A[0].e[0].g': 4, '.c[0].A[0].e[0].h': 5, '.c[0].A[0].i.j': 6, '.c[0].A[0].i.k': 111, '.c[0].A[1].e[0].f': 8, '.c[0].A[1].e[0].g': 9, '.c[0].A[1].e[0].h': 10, '.c[0].A[1].i.j': 11, '.c[0].A[1].i.k': 112, '.c[0].B.l': 13, '.c[0].B.m': 222}],
            1: [{'.c[1].A[0].e[0].g': 16, '.c[1].A[0].e[0].h': 17, '.c[1].A[0].i.j': 18, '.c[1].A[0].i.k': 113, '.c[1].A[1].e[0].f': 20, '.c[1].A[1].e[0].g': 21, '.c[1].A[1].e[0].h': 22, '.c[1].A[1].i.j': 23, '.c[1].A[1].i.k': 111, '.c[1].B.l': 25, '.c[1].B.m': 223}],
        }
    }
            
    for idx, (jq_path, values) in enumerate(tagged.items()):
        for j, val in enumerate(values):
            blocks = schema.collect_value_blocks(
                jq_path, val, jdata, max_count=0,
            )
            assert expected[idx][j] == blocks

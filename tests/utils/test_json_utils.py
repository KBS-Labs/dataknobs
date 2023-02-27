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


def test_flat_record_generator1(test_json_001):
    jdata = test_json_001
    str_io = io.StringIO()
    jutils.write_squashed(
        str_io, jdata, format_fn=jutils.indexing_format_fn
    )
    str_io.seek(0)
    recs = list()
    for rec in jutils.flat_record_generator(
            str_io, jutils.indexing_format_splitter
    ):
        recs.append(rec)
    assert len(recs) == 10
    assert json.dumps(recs) == '[{"a": "1", "b": "2", "f": "3", "g": "4", "h": "5"}, {"a": "1", "b": "2", "j": "6", "k": "111"}, {"a": "1", "b": "2", "f": "8", "g": "9", "h": "10"}, {"a": "1", "b": "2", "j": "11", "k": "112"}, {"a": "1", "b": "2", "l": "13", "m": "222"}, {"a": "1", "b": "2", "f": "15", "g": "16", "h": "17"}, {"a": "1", "b": "2", "j": "18", "k": "113"}, {"a": "1", "b": "2", "f": "20", "g": "21", "h": "22"}, {"a": "1", "b": "2", "j": "23", "k": "111"}, {"a": "1", "b": "2", "l": "25", "m": "223"}]'


def test_flat_records_builder(test_json_002):
    jdata = test_json_002
    flat_io = io.StringIO()
    jutils.write_squashed(
        flat_io, jdata, format_fn=jutils.indexing_format_fn
    )
    flat_io.seek(0)

    recs_io = io.StringIO()
    rmi = jutils.RecordMetaInfo(
        None, pivot_pfx = '.', ignore_pfxs = None,
        jq_clean = jutils.clean_jq_style_records, full_path_attrs = True,
        file_obj = recs_io
    )
    frb = jutils.FlatRecordsBuilder(
        jutils.indexing_format_splitter,
        [rmi],
    )

    frb.process_flatfile(flat_io)

    recs_io.seek(0)
    recs = list()
    for line in recs_io:
        recs.append(json.loads(line))
    assert len(recs) == 2
    assert len(recs[0]) == len(recs[1])
    assert recs[0].keys() == recs[1].keys()


def check_record(
        rec, #: Dict[str, str],
        kv_vals #: Dict[str, Set[int]],
):
    '''
    Check that the rec has keys with values in the specified range
    and no other keys.
    '''
    seen_keys = set()
    for k, v in rec.items():
        if k in kv_vals:
            for val in v.split(','):
                value = int(val.strip())
                assert value in kv_vals[k]
                seen_keys.add(k)
        else:
            assert False  # record has an unexpected key
    if len(seen_keys - kv_vals.keys()) > 0:
        assert False  # record is missing expected key(s)


def check_records(
        recs,  #: List[Dict[str, str]],
        kv_vals, #: Dict[str, Set[int]],
):
    '''
    Check that the each rec has keys with values in the specified range
    and no other keys.
    '''
    for rec in recs:
        check_record(rec, kv_vals)


def test_count_uniques1(test_json_003):
    ''' using explicit ignore_pfxx '''
    jdata = test_json_003
    builder = jutils.JsonSchemaBuilder(jdata, keep_unique_values=True)
    schema = builder.schema
    df = schema.df

    flat_io = io.StringIO()
    jutils.write_squashed(
        flat_io, jdata, format_fn=jutils.indexing_format_fn
    )
    flat_io.seek(0)

    recs_io1 = io.StringIO()
    recs_io2 = io.StringIO()
    recs_io3 = io.StringIO()
    rmi1 = jutils.RecordMetaInfo(
        None, pivot_pfx = '.A', ignore_pfxs = ['.B'],
        jq_clean = jutils.clean_jq_style_records, full_path_attrs = True,
        add_idx_attrs = True,
        file_obj = recs_io1
    )
    rmi2 = jutils.RecordMetaInfo(
        None, pivot_pfx = '.B.j', ignore_pfxs = ['.A', '.B[].l'],
        jq_clean = jutils.clean_jq_style_records, full_path_attrs = True,
        add_idx_attrs = True,
        file_obj = recs_io2
    )
    rmi3 = jutils.RecordMetaInfo(
        None, pivot_pfx = '.B.l', ignore_pfxs = ['.A', '.B[].j'],
        jq_clean = jutils.clean_jq_style_records, full_path_attrs = True,
        add_idx_attrs = True,
        file_obj = recs_io3
    )
    frb = jutils.FlatRecordsBuilder(
        jutils.indexing_format_splitter,
        [rmi1, rmi2, rmi3],
    )

    frb.process_flatfile(flat_io)

    recs_io1.seek(0)
    recs1 = list()
    for line in recs_io1:
        recs1.append(json.loads(line))
    assert len(recs1) == 4
    # all recs with keys a,b,c,e,f,g
    # and values in range [1,24] (after splitting on commas)
    # idx keys: A,d 0,0, 0,1, 1,0, 1,1
    check_records(
        recs1,
        {
            'a': {1},
            'b': {2},
            'c': {3, 14},
            'e': {4, 5, 6, 9, 10, 11, 15, 16, 17, 20, 21, 22},
            'f': {7, 12, 18, 23},
            'g': {8, 13, 19, 24},
            'A': {0, 1},
            'd': {0, 1},
        }
    )

    recs_io2.seek(0)
    recs2 = list()
    for line in recs_io2:
        recs2.append(json.loads(line))
    assert len(recs2) == 2
    # all recs with keys a,b,h,i,k
    # with values 1, 2, [25-32]
    # idx keys: B,j 0,0, 0,1
    check_records(
        recs2,
        {
            'a': {1},
            'b': {2},
            'h': {25},
            'i': {26},
            'k': {27, 28, 29, 30, 31, 32},
            'B': {0},
            'j': {0, 1},
        }
    )

    recs_io3.seek(0)
    recs3 = list()
    for line in recs_io3:
        recs3.append(json.loads(line))
    assert len(recs3) == 4
    # all recs with keys a,b,h,i,m,n,o,p.q
    # with values 1,2,25,26,[33-68]
    # idx keys: B,l,m0 0,0,0, 0,0,1 0,1,0 0,1,1
    check_records(
        recs3,
        {
            'a': {1},
            'b': {2},
            'h': {25},
            'i': {26},
            'm': {33, 42, 51, 60},
            'n': {34, 43, 52, 61},
            'o': {35, 36, 37, 44, 45, 46, 53, 54, 55, 62, 63, 64},
            'p': {38, 47, 56, 65},
            'q': {39, 40, 41, 48, 49, 50, 57, 58, 59, 66, 67, 68},
            'B': {0},
            'l': {0, 1},
            'm0': {0, 1},
        }
    )

    x1 = json.dumps(recs1, indent=2)
    x2 = json.dumps(recs2, indent=2)
    x3 = json.dumps(recs3, indent=3)
    #import pdb; pdb.set_trace()
    #stop_here1 = True

    
def test_count_uniques2(test_json_003):
    ''' w/ignore_pfxs='' to  exclude all other paths '''
    jdata = test_json_003
    builder = jutils.JsonSchemaBuilder(jdata, keep_unique_values=True)
    schema = builder.schema
    df = schema.df

    flat_io = io.StringIO()
    jutils.write_squashed(
        flat_io, jdata, format_fn=jutils.indexing_format_fn
    )
    flat_io.seek(0)

    recs_io1 = io.StringIO()
    recs_io2 = io.StringIO()
    recs_io3 = io.StringIO()
    rmi1 = jutils.RecordMetaInfo(
        None, pivot_pfx = '.A', ignore_pfxs = '',
        jq_clean = jutils.clean_jq_style_records, full_path_attrs = True,
        add_idx_attrs = True,
        file_obj = recs_io1
    )
    rmi2 = jutils.RecordMetaInfo(
        None, pivot_pfx = '.B[].j', ignore_pfxs = '',
        jq_clean = jutils.clean_jq_style_records, full_path_attrs = True,
        add_idx_attrs = True,
        file_obj = recs_io2
    )
    rmi3 = jutils.RecordMetaInfo(
        None, pivot_pfx = '.B[].l', ignore_pfxs = '',
        jq_clean = jutils.clean_jq_style_records, full_path_attrs = True,
        add_idx_attrs = True,
        file_obj = recs_io3
    )
    frb = jutils.FlatRecordsBuilder(
        jutils.indexing_format_splitter,
        [rmi1, rmi2, rmi3],
    )

    frb.process_flatfile(flat_io)

    recs_io1.seek(0)
    recs1 = list()
    for line in recs_io1:
        recs1.append(json.loads(line))
    assert len(recs1) == 4
    # keys: c,e,f,g, values: [3-24]
    # idx keys: A,d 0,0, 0,1, 1,0, 1,1
    check_records(
        recs1,
        {
            'c': {3, 14},
            'e': {4, 5, 6, 9, 10, 11, 15, 16, 17, 20, 21, 22},
            'f': {7, 12, 18, 23},
            'g': {8, 13, 19, 24},
            'A': {0, 1},
            'd': {0, 1},
        }
    )


    recs_io2.seek(0)
    recs2 = list()
    for line in recs_io2:
        recs2.append(json.loads(line))
    assert len(recs2) == 2
    # keys: "k", values: [27-32]
    # idx keys: B,j 0,0, 0,1
    check_records(
        recs2,
        {
            'k': {27, 28, 29, 30, 31, 32},
            'B': {0},
            'j': {0, 1},
        }
    )

    recs_io3.seek(0)
    recs3 = list()
    for line in recs_io3:
        recs3.append(json.loads(line))
    assert len(recs3) == 4
    # keys: m,n,o,p,q, values:[33-68]
    # idx keys: B,l,m0 0,0,0, 0,0,1 0,1,0 0,1,1
    check_records(
        recs3,
        {
            'm': {33, 42, 51, 60},
            'n': {34, 43, 52, 61},
            'o': {35, 36, 37, 44, 45, 46, 53, 54, 55, 62, 63, 64},
            'p': {38, 47, 56, 65},
            'q': {39, 40, 41, 48, 49, 50, 57, 58, 59, 66, 67, 68},
            'B': {0},
            'l': {0, 1},
            'm0': {0, 1},
        }
    )

    x1 = json.dumps(recs1, indent=2)
    x2 = json.dumps(recs2, indent=2)
    x3 = json.dumps(recs3, indent=3)
    #import pdb; pdb.set_trace()
    #stop_here2 = True

    

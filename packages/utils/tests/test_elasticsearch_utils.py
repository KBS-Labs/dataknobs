import json
import os
import tempfile

import pandas as pd

import dataknobs_utils.elasticsearch_utils as es_utils
from dataknobs_utils import requests_utils


def test_build_field_query_dict_basic_noop():
    fqd = es_utils.build_field_query_dict("data", "this is a test")
    expected = {"query": {"match": {"data": {"query": "this is a test"}}}}
    assert fqd == expected


def test_build_field_query_dict_basic_withop():
    fqd = es_utils.build_field_query_dict("data", "this is a test", "AND")
    expected = {"query": {"match": {"data": {"query": "this is a test", "operator": "AND"}}}}
    assert fqd == expected


def test_build_field_query_dict_single_multifield():
    fqd = es_utils.build_field_query_dict(["data"], "this is a test", "OR")
    expected = {"query": {"match": {"data": {"query": "this is a test", "operator": "OR"}}}}
    assert fqd == expected


def test_build_field_query_dict_multifield():
    expected = {"query": {"multi_match": {"query": "this is a test", "fields": ["data1", "data2"]}}}
    fqd = es_utils.build_field_query_dict(["data1", "data2"], "this is a test")
    assert fqd == expected
    fqd = es_utils.build_field_query_dict(["data1", "data2"], "this is a test", "OR")
    assert fqd == expected  # operator has no effect with multifield


def test_build_phrase_query_dict():
    pqd = es_utils.build_phrase_query_dict("data", "this is a test", 2)
    expected = {"query": {"match_phrase": {"data": {"query": "this is a test", "slop": 2}}}}
    assert pqd == expected


def test_batchfile_functions():
    batchdir = tempfile.TemporaryDirectory(suffix=".batchfiles", prefix="test-es.")
    batchfile = os.path.join(batchdir.name, "bf-001.jsonl")

    def batchgen(start_idx, end_idx):
        for n in range(start_idx, end_idx):
            yield {"data": n}

    with open(batchfile, "w", encoding="utf-8") as bf:
        es_utils.add_batch_data(bf, batchgen(0, 10), "test-data", cur_id=0)

    assert es_utils.collect_batchfile_values(batchfile, "data") == list(range(0, 10))
    recs = es_utils.collect_batchfile_records(batchfile)
    assert recs.equals(pd.DataFrame([{"data": n, "id": n} for n in range(0, 10)]))


SAMPLE_ES_SEARCH = {
    "query": {"query": {"match": {"sgloss": {"query": "xylophone_NN", "operator": "OR"}}}},
    "result": {
        "took": 3,
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "max_score": 13.805645,
            "hits": [
                {
                    "_index": "data",
                    "_id": "80435",
                    "_score": 13.805645,
                    "_source": {
                        "id": 80435,
                        "word": "xylophonist",
                        "pos": "n",
                        "sense_num": "01",
                        "gloss": "xylophonist: someone who plays a xylophone",
                        "synset_name": "xylophonist.n.01",
                        "raw_gloss": "someone who plays a xylophone",
                        "egloss": "xylophonist: someone who plays a xylophone",
                        "synsets": "xylophonist",
                        "sgloss": "someone_NN who_WP play_VBZ xylophone_NN",
                        "score": 13.805645,
                    },
                },
                {
                    "_index": "data",
                    "_id": "47073",
                    "_score": 8.001685,
                    "_source": {
                        "id": 47073,
                        "word": "vibraphone",
                        "pos": "n",
                        "sense_num": "01",
                        "gloss": "vibraphone: a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
                        "synset_name": "vibraphone.n.01",
                        "raw_gloss": "a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
                        "egloss": "vibraphone: a percussion instrument similar to a xylophone but having metal bars and rotating disks in the resonators that produce a vibrato sound",
                        "synsets": "vibraphone vibraharp vibes",
                        "sgloss": "percussion_NN instrument_NN similar_JJ to_IN xylophone_NN have_VBG metal_NN bar_NNS rotate_VBG disk_NNS in_IN resonator_NNS that_WDT produce_VBP vibrato_NN sound_NN",
                        "score": 8.001685,
                    },
                },
            ],
        },
    },
}


# Columnar with cursor -- first request
SAMPLE_ES_SQL_1A = {
    "api_query": "select pos, count(pos) as count from data group by pos order by count desc",
    "query": {
        "query": "select pos, count(pos) as count from data group by pos order by count desc",
        "fetch_size": 3,
        "columnar": True,
    },
    "result": {
        "columns": [{"name": "pos", "type": "text"}, {"name": "count", "type": "long"}],
        "values": [["n", "v", "s"], [82115, 13767, 10693]],
        "cursor": "s5CTBERGTABijGJgzGFnYmdiYExkYgADWXUQrwjK41NlYgYAAAD//wMA",
    },
}


# Columnar with cursor -- cursor request
SAMPLE_ES_SQL_1B = {
    "query": {
        "cursor": "s5CTBERGTABijGJgzGFnYmdiYExkYgADWXUQrwjK41NlYgYAAAD//wMA",
        "columnar": True,
    },
    "result": {"values": [["a", "r"], [7463, 3621]]},
}


# Non-columnar with cursor -- first request
SAMPLE_ES_SQL_2A = {
    "api_query": "select pos, count(pos) as count from data group by pos order by count desc",
    "query": {
        "query": "select pos, count(pos) as count from data group by pos order by count desc",
        "fetch_size": 3,
        "columnar": False,
    },
    "result": {
        "columns": [{"name": "pos", "type": "text"}, {"name": "count", "type": "long"}],
        "rows": [["n", 82115], ["v", 13767], ["s", 10693]],
        "cursor": "s5CTBERGTABijGJgzGFnYmdiYExkYgADWXUQrwjK41NlYgYAAAD//wMA",
    },
}


# Non-columnar with cursor -- cursor request
SAMPLE_ES_SQL_2B = {
    "query": {
        "cursor": "s5CTBERGTABijGJgzGFnYmdiYExkYgADWXUQrwjK41NlYgYAAAD//wMA",
        "columnar": False,
    },
    "result": {"rows": [["a", 7463], ["r", 3621]]},
}


SAMPLE_ES_ANALYZE = {
    "payload": {"analyzer": "standard", "text": "just testing"},
    "result": {
        "tokens": [
            {
                "token": "just",
                "start_offset": 0,
                "end_offset": 4,
                "type": "<ALPHANUM>",
                "position": 0,
            },
            {
                "token": "testing",
                "start_offset": 5,
                "end_offset": 12,
                "type": "<ALPHANUM>",
                "position": 1,
            },
        ]
    },
}


def build_mock_requests():
    mock_requests = requests_utils.MockRequests()

    # For ElasticSearchIndex._init_tables
    mock_requests.add(
        requests_utils.MockResponse(200, {"foo": "bar"}),
        "put",
        "http://localhost:9200/",
        data="test-data",
        headers=requests_utils.HEADERS,
    )
    mock_requests.add(
        requests_utils.MockResponse(200, {"foo": "bar"}),
        "post",
        "http://localhost:9200/_close",
    )
    mock_requests.add(
        requests_utils.MockResponse(200, {"foo": "bar"}),
        "put",
        "http://localhost:9200/_settings",
        data=json.dumps({"foo": "bar"}),
        headers=requests_utils.HEADERS,
    )
    mock_requests.add(
        requests_utils.MockResponse(200, {"foo": "bar"}),
        "post",
        "http://localhost:9200/_open",
    )
    mock_requests.add(
        requests_utils.MockResponse(200, {"foo": "bar"}),
        "put",
        "http://localhost:9200/_mapping",
        data=json.dumps({"properties": None}),
        headers=requests_utils.HEADERS,
    )

    # For ElasticSearchIndex.get_cluster_health() && .is_up()
    mock_requests.add(
        requests_utils.MockResponse(200, {"foo": "bar"}),
        "get",
        "http://localhost:9200/_cluster/health",
        headers=requests_utils.HEADERS,
    )

    # For ElasticSearchIndex.inspect_indices()
    mock_requests.add(
        requests_utils.MockResponse(400, "plain text, non-json response"),
        "get",
        "http://localhost:9200/_cat/indices?v&pretty",
        headers=requests_utils.HEADERS,
    )

    # For ElasticSearchIndex.purge()
    mock_requests.add(
        requests_utils.MockResponse(200, {"foo": "bar"}),
        "delete",
        "http://localhost:9200/test-data",
        headers=requests_utils.HEADERS,
    )

    # For ElasticSearchIndex.analyze()
    mock_requests.add(
        requests_utils.MockResponse(200, SAMPLE_ES_ANALYZE["result"]),
        "post",
        "http://localhost:9200/_analyze",
        data=json.dumps(SAMPLE_ES_ANALYZE["payload"]),
        headers=requests_utils.HEADERS,
    )

    # For ElasticSearchIndex.search()
    mock_requests.add(
        requests_utils.MockResponse(200, SAMPLE_ES_SEARCH["result"]),
        "post",
        "http://localhost:9200/test-data/_search",
        data=json.dumps(SAMPLE_ES_SEARCH["query"]),
        headers=requests_utils.HEADERS,
    )

    # For ElasticSearchIndex.sql()
    mock_requests.add(
        requests_utils.MockResponse(200, SAMPLE_ES_SQL_1A["result"]),
        "post",
        "http://localhost:9200/_sql?format=json",
        data=json.dumps(SAMPLE_ES_SQL_1A["query"]),
        headers=requests_utils.HEADERS,
    )
    mock_requests.add(
        requests_utils.MockResponse(200, SAMPLE_ES_SQL_1B["result"]),
        "post",
        "http://localhost:9200/_sql?format=json",
        data=json.dumps(SAMPLE_ES_SQL_1B["query"]),
        headers=requests_utils.HEADERS,
    )
    mock_requests.add(
        requests_utils.MockResponse(200, SAMPLE_ES_SQL_2A["result"]),
        "post",
        "http://localhost:9200/_sql?format=json",
        data=json.dumps(SAMPLE_ES_SQL_2A["query"]),
        headers=requests_utils.HEADERS,
    )
    mock_requests.add(
        requests_utils.MockResponse(200, SAMPLE_ES_SQL_2B["result"]),
        "post",
        "http://localhost:9200/_sql?format=json",
        data=json.dumps(SAMPLE_ES_SQL_2B["query"]),
        headers=requests_utils.HEADERS,
    )

    return mock_requests


def test_elasticsearch_index():
    requests = build_mock_requests()
    eidx = es_utils.ElasticsearchIndex(
        None,
        [es_utils.TableSettings("test-data", {"settings": None}, {"properties": None})],
        elasticsearch_ip="localhost",
        elasticsearch_port=9200,
        mock_requests=requests,
    )
    assert eidx.is_up()

    resp = eidx.get_cluster_health()
    assert resp.status == 200
    assert resp.result == {"foo": "bar"}

    resp = eidx.inspect_indices()
    assert resp.status == 400
    assert resp.result == "plain text, non-json response"

    resp = eidx.purge()
    assert resp.status == 200
    assert resp.result == {"foo": "bar"}

    resp = eidx.analyze("just testing", "standard")
    assert resp.status == 200
    assert resp.result == SAMPLE_ES_ANALYZE["result"]

    resp = eidx.search(SAMPLE_ES_SEARCH["query"])
    assert resp.status == 200
    assert resp.has_extra()
    assert "hits_df" in resp.extra
    assert resp.extra["hits_df"].shape == (2, 11)

    resp = eidx.sql(SAMPLE_ES_SQL_1A["api_query"], fetch_size=3)
    assert resp.status == 200
    assert resp.has_extra()
    assert "df" in resp.extra
    assert resp.extra["df"].shape == (5, 2)

    resp = eidx.sql(SAMPLE_ES_SQL_2A["api_query"], fetch_size=3, columnar=False)
    assert resp.status == 200
    assert resp.has_extra()
    assert "df" in resp.extra
    assert resp.extra["df"].shape == (5, 2)

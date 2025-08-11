import json

from dataknobs_utils import requests_utils


def test_get_current_ip():
    cur_ip = requests_utils.get_current_ip()
    assert len(cur_ip.split(".")) == 4


def test_server_response_repr1():
    result = {"foo": "bar"}
    mock_response = requests_utils.MockResponse(200, result)
    sr = mock_response.to_server_response()
    assert str(sr) == f"(200):\n{json.dumps(result, indent=2)}"


def test_server_response_repr2():
    mock_response = requests_utils.MockResponse(400, None)
    sr = mock_response.to_server_response()
    assert str(sr) == "400"

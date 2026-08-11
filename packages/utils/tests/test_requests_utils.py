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


def _helper_with(response, *, timeout, registered_timeout):
    """A RequestHelper whose transport answers only at ``registered_timeout``.

    ``MockRequests`` keys its responses on every request parameter, timeout
    included, so it doubles as a probe for which timeout actually reached the
    transport: a mismatch comes back as the default 404 rather than the
    registered response.
    """
    mock_requests = requests_utils.MockRequests()
    mock_requests.add(
        response,
        "get",
        "http://h:1/p",
        headers=requests_utils.HEADERS,
        timeout=registered_timeout,
    )
    return requests_utils.RequestHelper("h", 1, timeout=timeout, mock_requests=mock_requests)


def test_convenience_method_uses_the_instance_timeout():
    """``get()`` without a timeout must fall back to the instance default.

    ``request()`` spells "unset" as ``0`` while the convenience wrappers spell
    it as ``None``, and the wrappers passed theirs straight through. ``None``
    is not ``0``, so the fallback never fired and ``timeout=None`` reached the
    transport — which for ``requests`` means *wait forever*, on a call the
    caller believed carried the helper's default.
    """
    helper = _helper_with(
        requests_utils.MockResponse(200, '"ok"'),
        timeout=17,
        registered_timeout=17,
    )

    assert helper.get("p").result == "ok"


def test_convenience_method_honours_an_explicit_timeout():
    """An explicitly passed timeout still overrides the instance default."""
    helper = _helper_with(
        requests_utils.MockResponse(200, '"ok"'),
        timeout=17,
        registered_timeout=5,
    )

    assert helper.get("p", timeout=5).result == "ok"

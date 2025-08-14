import json
import socket
import sys
from collections.abc import Callable
from typing import Any, Dict, Tuple, Optional, Union

import requests

DEFAULT_TIMEOUT = 5
HEADERS = {"Content-Type": "application/json"}
DBG_HEADERS = {"Content-Type": "application/json", "error_trace": "true"}


def get_current_ip() -> str:
    """Get the running machine's IPv4 address.

    This function attempts to get the machine's IP address by connecting to
    an external service (Google DNS) to determine the local IP used for
    outbound connections. If that fails, it falls back to trying hostname
    resolution, and finally returns localhost as a last resort.

    :return: The IP address
    """
    try:
        # Create a socket and connect to an external service to get the local IP
        # We use Google's DNS server (8.8.8.8) as it's widely available
        # Note: This doesn't actually send any DNS queries, just establishes a connection
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        # Fallback to hostname resolution
        try:
            return socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            # If hostname doesn't resolve (common on macOS), return localhost
            return "127.0.0.1"


def json_api_response_handler(resp: requests.models.Response) -> Tuple[requests.models.Response, Optional[Dict[str, Any]]]:
    result = None
    if resp.text:
        result = json.loads(resp.text)
    return (resp, result)


def plain_api_response_handler(resp: requests.models.Response) -> Tuple[requests.models.Response, str]:
    return (resp, resp.text)


default_api_response_handler = json_api_response_handler


def get_request(
    api_request: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Submit the api get request and collect the response as a Dict.
    :param api_request: The api request
    :param params: The request parameters
    :param headers: The request headers (defaults to HEADERS)
    :param timeout: The request timeout (in seconds)
    :param api_response_handler: A handler function for api responses
    :param requests: Arg for alternate or "mock" requests package override
    :return: The response code and the json result as a dict (or None)
    """
    if headers is None:
        headers = HEADERS
    return api_response_handler(
        requests.get(api_request, headers=headers, params=params, timeout=timeout)
    )


def post_request(
    api_request: str,
    payload: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Submit the api post request and collect the response as a Dict.
    :param api_request: The api request
    :param params: The request parameters
    :param headers: The request headers (defaults to HEADERS)
    :param timeout: The request timeout (in seconds)
    :param api_response_handler: A handler function for api responses
    :param requests: Arg for alternate or "mock" requests package override
    :return: The response code and the json result as a dict (or None)
    """
    if headers is None:
        headers = HEADERS
    return api_response_handler(
        requests.post(
            api_request,
            data=payload,
            headers=headers,
            params=params,
            timeout=timeout,
        )
    )


def post_files_request(
    api_request: str,
    files: Dict[str, Any],
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Post data from one or more files.
    :param api_request: The api request
    :param files: A dict of {<file_id>: <file_data>}} entries for each file,
        where files can be of the form of the following examples:
            1. Just file data:
               - {'myfile': open('report.xls', 'rb')}  # NOTE: open in binary mode!
            2. Including filename, content_type, and headers
               - {'myfile': (
                                'report.xls',
                                open('report.xls', 'rb'),
                                'application/vnd.ms-excel',
                                {'Expires': '0'}
                            )}
    :param headers: The request headers
    :param timeout: The request timeout (in seconds)
    :param api_response_handler: A handler function for api responses
    :param requests: Arg for alternate or "mock" requests package override
    :return: The response code and the json result as a dict (or None)
    """
    return api_response_handler(
        requests.post(api_request, files=files, headers=headers, timeout=timeout)
    )


def put_request(
    api_request: str,
    payload: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Submit the api put request and collect the response as a Dict.
    :param api_request: The api request
    :param params: The request parameters
    :param headers: The request headers (defaults to HEADERS)
    :param timeout: The request timeout (in seconds)
    :param api_response_handler: A handler function for api responses
    :param requests: Arg for alternate or "mock" requests package override
    :return: The response code and the json result as a dict (or None)
    """
    if headers is None:
        headers = HEADERS
    return api_response_handler(
        requests.put(
            api_request,
            data=payload,
            headers=headers,
            params=params,
            timeout=timeout,
        )
    )


def delete_request(
    api_request: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Submit the api delete request and collect the response as a Dict.
    :param api_request: The api request
    :param params: The request parameters
    :param headers: The request headers (defaults to HEADERS)
    :param timeout: The request timeout (in seconds)
    :param api_response_handler: A handler function for api responses
    :param requests: Arg for alternate or "mock" requests package override
    :return: The response code and the json result as a dict (or None)
    """
    if headers is None:
        headers = HEADERS
    return api_response_handler(
        requests.delete(api_request, headers=headers, params=params, timeout=timeout)
    )


class ServerResponse:
    """Class to encapsulate request response data from the elasticsearch server."""

    def __init__(self, resp: Optional[requests.models.Response], result: Any) -> None:
        self.resp = resp
        self.result = result
        self._extra: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        rv = ""
        if self.result:
            rv = f"({self.status}):\n{json.dumps(self.result, indent=2)}"
        else:
            rv = str(self.status)
        return rv

    @property
    def succeeded(self) -> bool:
        return self.resp.status_code in {200, 201} if self.resp is not None else False

    @property
    def status(self) -> Optional[int]:
        return self.resp.status_code if self.resp is not None else None

    @property
    def extra(self) -> Dict[str, Any]:
        if self._extra is None:
            self._extra = dict()
        return self._extra

    def has_extra(self) -> bool:
        return self._extra is not None and len(self._extra) > 0

    def add_extra(self, key: str, value: Any) -> None:
        self.extra[key] = value


class RequestHelper:
    """Class to simplify sending api request commands to a server."""

    def __init__(
        self,
        server_ip: str,
        server_port: Union[str, int],
        api_response_handler: Callable = json_api_response_handler,
        headers: Optional[Dict[str, Any]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        mock_requests: Optional[Any] = None,
    ) -> None:
        self.ip = server_ip
        self.port = server_port
        self.response_handler = api_response_handler
        self.headers = headers
        self.timeout = timeout
        self.requests = mock_requests if mock_requests else requests

    def build_url(self, path: str) -> str:
        return f"http://{self.ip}:{self.port}/{path}"

    def request(
        self,
        rtype: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        response_handler: Optional[Callable] = None,
        headers: Optional[Dict[str, Any]] = None,
        timeout: int = 0,
        verbose: Union[bool, Any] = True,
    ) -> ServerResponse:
        """:param rtype: The request type, or command. One of:
            ['get', 'post', 'post-files', 'put', 'delete']
        :param path: The api path portion of the request
        :param payload: The request payload
        :param params: The request params
        :param files: The request files
        :param response_handler: Response handler to override instance value
        :param headers: Headers to override instance value
        :param timeout: The request timeout override (in seconds) if not 0
        :param verbose: True (or an output stream) to print server response info
        :return: A ServerResponse instance with the results
        """
        rtype = rtype.lower()
        if timeout == 0:
            timeout = self.timeout
        if headers is None:
            headers = self.headers
        if response_handler is None:
            response_handler = self.response_handler
        url = self.build_url(path)
        resp, result = None, None
        if rtype == "get":
            resp, result = get_request(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                api_response_handler=response_handler
                if response_handler is not None
                else self.response_handler,
                requests=self.requests,
            )
        elif rtype == "post":
            resp, result = post_request(
                url,
                payload if payload is not None else {},
                params=params,
                headers=headers,
                timeout=timeout,
                api_response_handler=response_handler,
                requests=self.requests,
            )
        elif rtype == "post-files":
            resp, result = post_files_request(
                url,
                files=files if files is not None else {},
                headers=headers,
                timeout=timeout,
                api_response_handler=response_handler,
                requests=self.requests,
            )
        elif rtype == "put":
            resp, result = put_request(
                url,
                payload if payload is not None else {},
                params=params,
                headers=headers,
                timeout=timeout,
                api_response_handler=response_handler,
                requests=self.requests,
            )
        elif rtype == "delete":
            resp, result = delete_request(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                api_response_handler=response_handler,
                requests=self.requests,
            )

        rv = ServerResponse(resp, result)

        if verbose is not None:
            if isinstance(verbose, bool) and verbose:
                verbose = sys.stderr
            else:
                verbose = None
            if verbose is not None:
                print(rv, file=verbose)

        return rv


class MockResponse:
    """A mock response object"""

    def __init__(self, status_code: int, result: Any) -> None:
        self.status_code = status_code
        self.result = result
        self.text = result
        if not isinstance(result, str):
            self.text = json.dumps(result)

    def to_server_response(self) -> ServerResponse:
        """Convenience method for creating a ServerResponse"""
        return ServerResponse(self, self.result)  # type: ignore[arg-type]


class MockRequests:
    def __init__(self) -> None:
        self.responses: Dict[str, MockResponse] = dict()
        self.r404 = MockResponse(404, '"Not found"')

    def add(
        self,
        response: MockResponse,
        api: str,
        api_request: str,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        key = self._make_key(
            api,
            api_request,
            data,
            files,
            headers,
            params,
            timeout,
        )
        self.responses[key] = response

    def _make_key(
        self,
        api: str,
        api_request: str,
        data: Optional[Any],
        files: Optional[Any],
        headers: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        timeout: Optional[int],
    ) -> str:
        return json.dumps(
            {
                "api": api,
                "req": api_request,
                "data": data,
                "files": files,
                "headers": headers,
                "params": params,
                "timeout": timeout,
            }
        )

    def get(self, api_request: str, headers: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> MockResponse:
        key = self._make_key("get", api_request, None, None, headers, params, timeout)
        return self.responses.get(key, self.r404)

    def post(self, api_request: str, data: Optional[Any] = None, files: Optional[Any] = None, headers: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> MockResponse:
        key = self._make_key("post", api_request, data, files, headers, params, timeout)
        return self.responses.get(key, self.r404)

    def put(self, api_request: str, data: Optional[Any] = None, headers: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> MockResponse:
        key = self._make_key("put", api_request, data, None, headers, params, timeout)
        return self.responses.get(key, self.r404)

    def delete(self, api_request: str, headers: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> MockResponse:
        key = self._make_key("delete", api_request, None, None, headers, params, timeout)
        return self.responses.get(key, self.r404)

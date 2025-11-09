"""HTTP request utilities for making API calls and handling responses.

Provides convenience functions for making HTTP requests with error handling,
timeout management, and response parsing.
"""

import json
import socket
import sys
from collections.abc import Callable
from typing import Any, Dict, Tuple, Union

import requests

DEFAULT_TIMEOUT = 5
HEADERS = {"Content-Type": "application/json"}
DBG_HEADERS = {"Content-Type": "application/json", "error_trace": "true"}


def get_current_ip() -> str:
    """Get the running machine's IPv4 address.

    Attempts to determine the local IP address by:
    1. Connecting to an external service (Google DNS) to find the outbound IP
    2. Falling back to hostname resolution
    3. Finally returning localhost as last resort

    Returns:
        str: The machine's IPv4 address (or "127.0.0.1" if detection fails).
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


def json_api_response_handler(
    resp: requests.models.Response,
) -> Tuple[requests.models.Response, Dict[str, Any] | None]:
    result = None
    if resp.text:
        result = json.loads(resp.text)
    return (resp, result)


def plain_api_response_handler(
    resp: requests.models.Response,
) -> Tuple[requests.models.Response, str]:
    return (resp, resp.text)


default_api_response_handler = json_api_response_handler


def get_request(
    api_request: str,
    params: Dict[str, Any] | None = None,
    headers: Dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Execute an HTTP GET request and return the response.

    Args:
        api_request: Full URL for the API request.
        params: Query parameters for the request. Defaults to None.
        headers: HTTP headers. If None, uses HEADERS constant. Defaults to None.
        timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
        api_response_handler: Function to process the response. Defaults to
            json_api_response_handler.
        requests: Requests library or mock for testing. Defaults to requests.

    Returns:
        Tuple[requests.models.Response, Any]: Tuple of (response object, parsed result).
    """
    if headers is None:
        headers = HEADERS
    return api_response_handler(
        requests.get(api_request, headers=headers, params=params, timeout=timeout)
    )


def post_request(
    api_request: str,
    payload: Dict[str, Any],
    params: Dict[str, Any] | None = None,
    headers: Dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Execute an HTTP POST request and return the response.

    Args:
        api_request: Full URL for the API request.
        payload: Request body data to send.
        params: Query parameters for the request. Defaults to None.
        headers: HTTP headers. If None, uses HEADERS constant. Defaults to None.
        timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
        api_response_handler: Function to process the response. Defaults to
            json_api_response_handler.
        requests: Requests library or mock for testing. Defaults to requests.

    Returns:
        Tuple[requests.models.Response, Any]: Tuple of (response object, parsed result).
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
    headers: Dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Execute an HTTP POST request with file uploads.

    Args:
        api_request: Full URL for the API request.
        files: Dictionary of {file_id: file_data} entries, where file_data can be:
            - Simple: open('report.xls', 'rb') (must open in binary mode)
            - Detailed: ('filename', file_object, 'content_type', {'headers': 'values'})
        headers: HTTP headers. Defaults to None.
        timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
        api_response_handler: Function to process the response. Defaults to
            json_api_response_handler.
        requests: Requests library or mock for testing. Defaults to requests.

    Returns:
        Tuple[requests.models.Response, Any]: Tuple of (response object, parsed result).

    Examples:
        ```python
        from dataknobs_utils.requests_utils import post_files_request
        url = "http://example.com/upload"

        # Simple file upload
        with open('report.xls', 'rb') as f:
            post_files_request(url, {'myfile': f})

        # With metadata
        with open('report.xls', 'rb') as f:
            post_files_request(url, {
                'myfile': ('report.xls', f,
                           'application/vnd.ms-excel', {'Expires': '0'})
            })
        ```
    """
    return api_response_handler(
        requests.post(api_request, files=files, headers=headers, timeout=timeout)
    )


def put_request(
    api_request: str,
    payload: Dict[str, Any],
    params: Dict[str, Any] | None = None,
    headers: Dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Execute an HTTP PUT request and return the response.

    Args:
        api_request: Full URL for the API request.
        payload: Request body data to send.
        params: Query parameters for the request. Defaults to None.
        headers: HTTP headers. If None, uses HEADERS constant. Defaults to None.
        timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
        api_response_handler: Function to process the response. Defaults to
            json_api_response_handler.
        requests: Requests library or mock for testing. Defaults to requests.

    Returns:
        Tuple[requests.models.Response, Any]: Tuple of (response object, parsed result).
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
    params: Dict[str, Any] | None = None,
    headers: Dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_response_handler: Callable[
        [requests.models.Response], Tuple[requests.models.Response, Any]
    ] = default_api_response_handler,
    requests: Any = requests,  # pylint: disable-msg=W0621
) -> Tuple[requests.models.Response, Any]:
    """Execute an HTTP DELETE request and return the response.

    Args:
        api_request: Full URL for the API request.
        params: Query parameters for the request. Defaults to None.
        headers: HTTP headers. If None, uses HEADERS constant. Defaults to None.
        timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT.
        api_response_handler: Function to process the response. Defaults to
            json_api_response_handler.
        requests: Arg for alternate or "mock" requests package override

    Returns:
        Tuple[requests.models.Response, Any]: Tuple of (response object, parsed result).
    """
    if headers is None:
        headers = HEADERS
    return api_response_handler(
        requests.delete(api_request, headers=headers, params=params, timeout=timeout)
    )


class ServerResponse:
    """Wrapper for HTTP response data with convenience properties.

    Encapsulates response data from HTTP requests, providing easy access to
    status codes, JSON data, and response text with consistent interface.

    Attributes:
        resp: The underlying requests Response object.
        result: Parsed response data (typically JSON).
    """

    def __init__(self, resp: requests.models.Response | None, result: Any) -> None:
        self.resp = resp
        self.result = result
        self._extra: Dict[str, Any] | None = None

    def __repr__(self) -> str:
        rv = ""
        if self.result:
            rv = f"({self.status}):\n{json.dumps(self.result, indent=2)}"
        else:
            rv = str(self.status)
        return rv

    @property
    def succeeded(self) -> bool:
        """Check if the request succeeded (status 200 or 201).

        Returns:
            bool: True if status code is 200 or 201, False otherwise.
        """
        return self.resp.status_code in {200, 201} if self.resp is not None else False

    @property
    def status(self) -> int | None:
        """Get the HTTP status code.

        Returns:
            int | None: Status code, or None if no response.
        """
        return self.resp.status_code if self.resp is not None else None

    @property
    def status_code(self) -> int | None:
        """Get the HTTP status code (alias for status).

        Provided for consistency with requests.Response interface.

        Returns:
            int | None: Status code, or None if no response.
        """
        return self.status

    @property
    def json(self) -> Any:
        """Get the parsed JSON response data.

        Alias for the result attribute.

        Returns:
            Any: Parsed response data.
        """
        return self.result

    @property
    def text(self) -> str:
        """Get the response as text.

        Returns:
            str: Response text (JSON-serialized if result is not a string).
        """
        if self.result:
            if isinstance(self.result, str):
                return self.result
            else:
                return json.dumps(self.result)
        return ""

    @property
    def extra(self) -> Dict[str, Any]:
        """Get the extra data dictionary.

        Lazily initializes an empty dictionary for storing additional metadata.

        Returns:
            Dict[str, Any]: Extra data dictionary.
        """
        if self._extra is None:
            self._extra = {}
        return self._extra

    def has_extra(self) -> bool:
        """Check if extra data has been added.

        Returns:
            bool: True if extra data exists and is non-empty.
        """
        return self._extra is not None and len(self._extra) > 0

    def add_extra(self, key: str, value: Any) -> None:
        """Add additional metadata to the response.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.extra[key] = value


class RequestHelper:
    """Helper class for making HTTP requests to a server.

    Simplifies sending API requests by managing server connection details,
    headers, timeouts, and response handling in a reusable instance.

    Attributes:
        ip: Server IP address.
        port: Server port number.
        response_handler: Function for processing responses.
        headers: Default HTTP headers.
        timeout: Default request timeout in seconds.
        requests: Requests library or mock for testing.
    """

    def __init__(
        self,
        server_ip: str,
        server_port: Union[str, int],
        api_response_handler: Callable = json_api_response_handler,
        headers: Dict[str, Any] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        mock_requests: Any | None = None,
    ) -> None:
        """Initialize request helper with server details.

        Args:
            server_ip: Server IP address or hostname.
            server_port: Server port number.
            api_response_handler: Default response handler function.
                Defaults to json_api_response_handler.
            headers: Default HTTP headers. Defaults to None.
            timeout: Default timeout in seconds. Defaults to DEFAULT_TIMEOUT.
            mock_requests: Mock requests object for testing. Defaults to None.
        """
        self.ip = server_ip
        self.port = server_port
        self.response_handler = api_response_handler
        self.headers = headers
        self.timeout = timeout
        self.requests = mock_requests if mock_requests else requests

    def build_url(self, path: str) -> str:
        """Construct full URL from path.

        Args:
            path: API path (without leading slash recommended).

        Returns:
            str: Complete URL (http://ip:port/path).
        """
        return f"http://{self.ip}:{self.port}/{path}"

    def request(
        self,
        rtype: str,
        path: str,
        payload: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        files: Dict[str, Any] | None = None,
        response_handler: Callable | None = None,
        headers: Dict[str, Any] | None = None,
        timeout: int = 0,
        verbose: Union[bool, Any] = True,
    ) -> ServerResponse:
        """Execute an HTTP request of any type.

        Args:
            rtype: Request type - one of: 'get', 'post', 'post-files', 'put',
                'delete', 'head'.
            path: API path portion (will be appended to server URL).
            payload: Request body data. Defaults to None.
            params: Query parameters. Defaults to None.
            files: Files for upload (for post-files requests). Defaults to None.
            response_handler: Response handler to override instance default.
                Defaults to None.
            headers: Headers to override instance default. Defaults to None.
            timeout: Timeout override in seconds (0 uses instance default).
                Defaults to 0.
            verbose: If True, prints response to stderr. If a file object,
                prints to that stream. Defaults to True.

        Returns:
            ServerResponse: Response object with status and parsed data.
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
        elif rtype == "head":
            # HEAD requests are like GET but without body
            try:
                resp = self.requests.head(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                # HEAD requests don't have a body, but we still need to process the response
                result = {}  # Empty dict for HEAD response
                if response_handler:
                    resp, result = response_handler(resp)
            except Exception:
                resp = None
                result = None

        rv = ServerResponse(resp, result)

        if verbose is not None:
            if isinstance(verbose, bool) and verbose:
                verbose = sys.stderr
            else:
                verbose = None
            if verbose is not None:
                print(rv, file=verbose)

        return rv

    def get(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, Any] | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Any:
        """Convenience method for GET requests.
        
        Args:
            path: API path
            params: Optional query parameters
            headers: Optional headers
            timeout: Optional timeout
            verbose: Whether to print debug info
            
        Returns:
            ServerResponse object
        """
        return self.request(
            "get",
            path,
            params=params,
            headers=headers,
            timeout=timeout,
            verbose=verbose,
        )

    def post(
        self,
        path: str,
        payload: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        files: Dict[str, Any] | None = None,
        headers: Dict[str, Any] | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Any:
        """Convenience method for POST requests.
        
        Args:
            path: API path
            payload: Optional request body
            params: Optional query parameters
            files: Optional files to upload
            headers: Optional headers
            timeout: Optional timeout
            verbose: Whether to print debug info
            
        Returns:
            ServerResponse object
        """
        return self.request(
            "post",
            path,
            payload=payload,
            params=params,
            files=files,
            headers=headers,
            timeout=timeout,
            verbose=verbose,
        )

    def put(
        self,
        path: str,
        payload: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, Any] | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Any:
        """Convenience method for PUT requests.
        
        Args:
            path: API path
            payload: Optional request body
            params: Optional query parameters
            headers: Optional headers
            timeout: Optional timeout
            verbose: Whether to print debug info
            
        Returns:
            ServerResponse object
        """
        return self.request(
            "put",
            path,
            payload=payload,
            params=params,
            headers=headers,
            timeout=timeout,
            verbose=verbose,
        )

    def delete(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, Any] | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Any:
        """Convenience method for DELETE requests.
        
        Args:
            path: API path
            params: Optional query parameters
            headers: Optional headers
            timeout: Optional timeout
            verbose: Whether to print debug info
            
        Returns:
            ServerResponse object
        """
        return self.request(
            "delete",
            path,
            params=params,
            headers=headers,
            timeout=timeout,
            verbose=verbose,
        )

    def head(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, Any] | None = None,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Any:
        """Convenience method for HEAD requests.
        
        Args:
            path: API path
            params: Optional query parameters
            headers: Optional headers
            timeout: Optional timeout
            verbose: Whether to print debug info
            
        Returns:
            ServerResponse object
        """
        return self.request(
            "head",
            path,
            params=params,
            headers=headers,
            timeout=timeout,
            verbose=verbose,
        )


class MockResponse:
    """Mock HTTP response for testing.

    Simulates a requests.Response object with status code and result data.

    Attributes:
        status_code: HTTP status code.
        result: Response data.
        text: Response as text (JSON-serialized if result is not a string).
    """

    def __init__(self, status_code: int, result: Any) -> None:
        self.status_code = status_code
        self.result = result
        self.text = result
        if not isinstance(result, str):
            self.text = json.dumps(result)

    def to_server_response(self) -> ServerResponse:
        """Convert to a ServerResponse object.

        Returns:
            ServerResponse: Wrapped response object.
        """
        return ServerResponse(self, self.result)  # type: ignore[arg-type]


class MockRequests:
    """Mock requests library for testing.

    Simulates the requests library by storing expected responses keyed by
    request parameters, allowing deterministic testing without network calls.

    Attributes:
        responses: Dictionary of registered mock responses.
        r404: Default 404 response for unregistered requests.
    """

    def __init__(self) -> None:
        self.responses: Dict[str, MockResponse] = {}
        self.r404 = MockResponse(404, '"Not found"')

    def add(
        self,
        response: MockResponse,
        api: str,
        api_request: str,
        data: Any | None = None,
        files: Any | None = None,
        headers: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Register a mock response for specific request parameters.

        Args:
            response: Mock response to return.
            api: Request method ('get', 'post', 'put', 'delete').
            api_request: Request URL.
            data: Request body data. Defaults to None.
            files: Request files. Defaults to None.
            headers: Request headers. Defaults to None.
            params: Query parameters. Defaults to None.
            timeout: Request timeout. Defaults to DEFAULT_TIMEOUT.
        """
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
        data: Any | None,
        files: Any | None,
        headers: Dict[str, Any] | None,
        params: Dict[str, Any] | None,
        timeout: int | None,
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

    def get(
        self,
        api_request: str,
        headers: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> MockResponse:
        key = self._make_key("get", api_request, None, None, headers, params, timeout)
        return self.responses.get(key, self.r404)

    def post(
        self,
        api_request: str,
        data: Any | None = None,
        files: Any | None = None,
        headers: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> MockResponse:
        key = self._make_key("post", api_request, data, files, headers, params, timeout)
        return self.responses.get(key, self.r404)

    def put(
        self,
        api_request: str,
        data: Any | None = None,
        headers: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> MockResponse:
        key = self._make_key("put", api_request, data, None, headers, params, timeout)
        return self.responses.get(key, self.r404)

    def delete(
        self,
        api_request: str,
        headers: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> MockResponse:
        key = self._make_key("delete", api_request, None, None, headers, params, timeout)
        return self.responses.get(key, self.r404)

"""HTTP service resource provider."""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, Union
from urllib.parse import urljoin
import urllib.request
import urllib.error
import urllib.parse

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceStatus,
)


@dataclass
class HTTPSession:
    """HTTP session with configuration and state."""
    
    base_url: str
    headers: Dict[str, str] = dataclass_field(default_factory=dict)
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Circuit breaker state
    failure_count: int = 0
    failure_threshold: int = 5
    last_failure_time: float | None = None
    circuit_open: bool = False
    circuit_half_open_after: float = 60.0  # seconds
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open.
        
        Returns:
            True if circuit is open and requests should be blocked.
        """
        if not self.circuit_open:
            return False
        
        # Check if we should transition to half-open
        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.circuit_half_open_after:
                # Try half-open state
                self.circuit_open = False
                return False
        
        return True
    
    def record_success(self) -> None:
        """Record a successful request."""
        self.failure_count = 0
        self.circuit_open = False
    
    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.circuit_open = True


class HTTPServiceResource(BaseResourceProvider):
    """HTTP service resource provider with session management."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        headers: Dict[str, str] | None = None,
        auth: Dict[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        **config
    ):
        """Initialize HTTP service resource.
        
        Args:
            name: Resource name.
            base_url: Base URL for the service.
            headers: Default headers for requests.
            auth: Authentication configuration.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
            **config: Additional configuration.
        """
        super().__init__(name, config)
        
        self.base_url = base_url.rstrip("/")
        self.default_headers = headers or {}
        self.auth = auth
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Set up authentication headers if provided
        if self.auth:
            self._setup_auth()
        
        self._sessions = {}
        self.status = ResourceStatus.IDLE
    
    def _setup_auth(self) -> None:
        """Set up authentication headers."""
        if self.auth.get("type") == "bearer":
            token = self.auth.get("token")
            if token:
                self.default_headers["Authorization"] = f"Bearer {token}"
        elif self.auth.get("type") == "basic":
            username = self.auth.get("username")
            password = self.auth.get("password")
            if username and password:
                import base64
                credentials = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                self.default_headers["Authorization"] = f"Basic {credentials}"
        elif self.auth.get("type") == "api_key":
            key_name = self.auth.get("key_name", "X-API-Key")
            key_value = self.auth.get("key_value")
            if key_value:
                self.default_headers[key_name] = key_value
    
    def acquire(self, **kwargs) -> HTTPSession:
        """Acquire an HTTP session.
        
        Args:
            **kwargs: Session configuration overrides.
            
        Returns:
            HTTPSession instance.
            
        Raises:
            ResourceError: If acquisition fails.
        """
        try:
            session = HTTPSession(
                base_url=kwargs.get("base_url", self.base_url),
                headers={**self.default_headers, **kwargs.get("headers", {})},
                timeout=kwargs.get("timeout", self.timeout),
                max_retries=kwargs.get("max_retries", self.max_retries),
                retry_delay=kwargs.get("retry_delay", 1.0)
            )
            
            session_id = id(session)
            self._sessions[session_id] = session
            self._resources.append(session)
            
            self.status = ResourceStatus.ACTIVE
            return session
            
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to acquire HTTP session: {e}",
                resource_name=self.name
            ) from e
    
    def release(self, resource: Any) -> None:
        """Release an HTTP session.
        
        Args:
            resource: The HTTPSession to release.
        """
        if isinstance(resource, HTTPSession):
            session_id = id(resource)
            if session_id in self._sessions:
                del self._sessions[session_id]
            
            if resource in self._resources:
                self._resources.remove(resource)
        
        if not self._resources:
            self.status = ResourceStatus.IDLE
    
    def validate(self, resource: Any) -> bool:
        """Validate an HTTP session.
        
        Args:
            resource: The HTTPSession to validate.
            
        Returns:
            True if the session is valid.
        """
        if not isinstance(resource, HTTPSession):
            return False
        
        # Check if circuit breaker is open
        if resource.is_circuit_open():
            return False
        
        return True
    
    def health_check(self) -> ResourceHealth:
        """Check HTTP service health.
        
        Returns:
            Health status.
        """
        session = None
        try:
            session = self.acquire()
            
            # Try a simple HEAD or GET request to base URL
            response = self._request(session, "HEAD", "/")
            
            if response.get("status", 0) < 500:
                self.metrics.record_health_check(True)
                return ResourceHealth.HEALTHY
            else:
                self.metrics.record_health_check(False)
                return ResourceHealth.DEGRADED
                
        except Exception:
            self.metrics.record_health_check(False)
            return ResourceHealth.UNHEALTHY
        finally:
            if session:
                self.release(session)
    
    def _request(
        self,
        session: HTTPSession,
        method: str,
        path: str,
        data: Union[Dict, bytes] | None = None,
        headers: Dict[str, str] | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute an HTTP request with retry logic.
        
        Args:
            session: HTTP session.
            method: HTTP method.
            path: Request path.
            data: Request body data.
            headers: Additional headers.
            **kwargs: Additional parameters.
            
        Returns:
            Response dictionary with status, headers, and body.
            
        Raises:
            ResourceError: If request fails after retries.
        """
        if session.is_circuit_open():
            raise ResourceError(
                "Circuit breaker is open - service unavailable",
                resource_name=self.name
            )
        
        url = urljoin(session.base_url, path)
        request_headers = {**session.headers}
        if headers:
            request_headers.update(headers)
        
        # Prepare request data
        request_data = None
        if data is not None:
            if isinstance(data, dict):
                request_data = json.dumps(data).encode("utf-8")
                if "Content-Type" not in request_headers:
                    request_headers["Content-Type"] = "application/json"
            else:
                request_data = data
        
        last_error = None
        for attempt in range(session.max_retries):
            try:
                # Create request
                req = urllib.request.Request(
                    url,
                    data=request_data,
                    headers=request_headers,
                    method=method
                )
                
                # Execute request
                with urllib.request.urlopen(req, timeout=session.timeout) as response:
                    # Read response
                    body = response.read()
                    
                    # Try to decode as JSON
                    try:
                        body_data = json.loads(body.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        body_data = body
                    
                    result = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "body": body_data
                    }
                    
                    session.record_success()
                    return result
                    
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code < 500:
                    # Client error, don't retry
                    session.record_failure()
                    raise ResourceError(
                        f"HTTP {e.code}: {e.reason}",
                        resource_name=self.name,
                        operation=f"{method} {path}"
                    ) from e
                # Server error, retry
                if attempt < session.max_retries - 1:
                    time.sleep(session.retry_delay * (attempt + 1))
                    
            except Exception as e:
                last_error = e
                if attempt < session.max_retries - 1:
                    time.sleep(session.retry_delay * (attempt + 1))
        
        # All retries failed
        session.record_failure()
        raise ResourceError(
            f"Request failed after {session.max_retries} retries: {last_error}",
            resource_name=self.name,
            operation=f"{method} {path}"
        )
    
    @contextmanager
    def session_context(self, **kwargs):
        """Context manager for HTTP session.
        
        Args:
            **kwargs: Session configuration.
            
        Yields:
            HTTPSession instance.
        """
        session = self.acquire(**kwargs)
        try:
            yield session
        finally:
            self.release(session)
    
    def get(
        self,
        path: str,
        session: HTTPSession | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute GET request.
        
        Args:
            path: Request path.
            session: Optional session to use.
            **kwargs: Additional parameters.
            
        Returns:
            Response dictionary.
        """
        if session is None:
            with self.session_context() as ctx_session:
                return self._request(ctx_session, "GET", path, **kwargs)
        return self._request(session, "GET", path, **kwargs)
    
    def post(
        self,
        path: str,
        data: Union[Dict, bytes] | None = None,
        session: HTTPSession | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute POST request.
        
        Args:
            path: Request path.
            data: Request body.
            session: Optional session to use.
            **kwargs: Additional parameters.
            
        Returns:
            Response dictionary.
        """
        if session is None:
            with self.session_context() as ctx_session:
                return self._request(ctx_session, "POST", path, data, **kwargs)
        return self._request(session, "POST", path, data, **kwargs)
    
    def put(
        self,
        path: str,
        data: Union[Dict, bytes] | None = None,
        session: HTTPSession | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute PUT request.
        
        Args:
            path: Request path.
            data: Request body.
            session: Optional session to use.
            **kwargs: Additional parameters.
            
        Returns:
            Response dictionary.
        """
        if session is None:
            with self.session_context() as ctx_session:
                return self._request(ctx_session, "PUT", path, data, **kwargs)
        return self._request(session, "PUT", path, data, **kwargs)
    
    def delete(
        self,
        path: str,
        session: HTTPSession | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute DELETE request.
        
        Args:
            path: Request path.
            session: Optional session to use.
            **kwargs: Additional parameters.
            
        Returns:
            Response dictionary.
        """
        if session is None:
            with self.session_context() as ctx_session:
                return self._request(ctx_session, "DELETE", path, **kwargs)
        return self._request(session, "DELETE", path, **kwargs)

"""API orchestration pattern implementation.

This module provides pre-configured FSM patterns for orchestrating API calls,
including parallel requests, sequential workflows, rate limiting, and retries.
"""

from typing import Any, Dict, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta

from ..api.simple import SimpleFSM
from ..core.data_modes import DataHandlingMode
from ..io.base import IOConfig, IOMode, IOFormat
from ..io.utils import create_io_provider, retry_io_operation, IOMetrics


class OrchestrationMode(Enum):
    """API orchestration modes."""
    SEQUENTIAL = "sequential"  # Execute APIs one after another
    PARALLEL = "parallel"  # Execute APIs concurrently
    FANOUT = "fanout"  # One request triggers multiple APIs
    PIPELINE = "pipeline"  # Output of one API feeds into next
    CONDITIONAL = "conditional"  # Execute based on conditions
    HYBRID = "hybrid"  # Mix of above patterns


@dataclass
class APIEndpoint:
    """Configuration for a single API endpoint."""
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] | None = None
    params: Dict[str, Any] | None = None
    body: Union[Dict[str, Any], str] | None = None
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    rate_limit: int | None = None  # Requests per minute
    burst_limit: int | None = None  # Max burst size
    
    # Response handling
    response_parser: Callable[[Any], Any] | None = None
    error_handler: Callable[[Exception], Any] | None = None
    
    # Dependencies
    depends_on: List[str] | None = None
    transform_input: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None


@dataclass
class APIOrchestrationConfig:
    """Configuration for API orchestration."""
    endpoints: List[APIEndpoint]
    mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL
    
    # Global settings
    max_concurrent: int = 10
    total_timeout: float = 300.0
    fail_fast: bool = False  # Stop on first error
    
    # Rate limiting (global)
    global_rate_limit: int | None = None
    rate_limit_window: int = 60  # seconds
    
    # Result handling
    result_merger: Callable[[List[Dict[str, Any]]], Any] | None = None
    result_transformer: Callable[[Any], Any] | None = None
    
    # Error handling
    error_threshold: float = 0.1  # Max 10% errors
    circuit_breaker_threshold: int = 5  # Consecutive failures
    circuit_breaker_timeout: float = 60.0  # seconds
    
    # Caching
    cache_ttl: int | None = None  # seconds
    cache_key_generator: Callable[[APIEndpoint], str] | None = None
    
    # Monitoring
    metrics_enabled: bool = True
    log_requests: bool = False
    log_responses: bool = False


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, rate_limit: int, window: int = 60):
        """Initialize rate limiter.
        
        Args:
            rate_limit: Maximum requests per window
            window: Time window in seconds
        """
        self.rate_limit = rate_limit
        self.window = window
        self.requests = []
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        while True:
            async with self._lock:
                now = datetime.now()
                cutoff = now - timedelta(seconds=self.window)
                
                # Remove old requests
                self.requests = [t for t in self.requests if t > cutoff]
                
                # Check if we can make a request
                if len(self.requests) < self.rate_limit:
                    # Record this request and return
                    self.requests.append(now)
                    return
                    
                # Calculate wait time
                oldest = self.requests[0]
                wait_time = (oldest + timedelta(seconds=self.window) - now).total_seconds()
                
            # Wait outside the lock
            if wait_time > 0:
                await asyncio.sleep(min(wait_time, 0.1))  # Sleep in small increments


class CircuitBreaker:
    """Circuit breaker for API calls."""
    
    def __init__(self, threshold: int, timeout: float):
        """Initialize circuit breaker.
        
        Args:
            threshold: Number of consecutive failures to trip
            timeout: Time to wait before attempting reset
        """
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure = None
        self.is_open = False
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Any, **kwargs: Any: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            # Check if circuit is open
            if self.is_open:
                if self.last_failure:
                    elapsed = (datetime.now() - self.last_failure).total_seconds()  # type: ignore[unreachable]
                    if elapsed < self.timeout:
                        from ..core.exceptions import CircuitBreakerError
                        raise CircuitBreakerError(wait_time=self.timeout - elapsed)
                # Try to reset
                self.is_open = False
                self.failure_count = 0
                
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Success - reset failure count
            async with self._lock:
                self.failure_count = 0
                
            return result
            
        except Exception:
            # Record failure
            async with self._lock:
                self.failure_count += 1
                self.last_failure = datetime.now()
                
                if self.failure_count >= self.threshold:
                    self.is_open = True
                    
            raise


class APIOrchestrator:
    """API orchestrator using FSM pattern."""
    
    def __init__(self, config: APIOrchestrationConfig):
        """Initialize API orchestrator.
        
        Args:
            config: Orchestration configuration
        """
        self.config = config
        self._fsm = self._build_fsm()
        self._providers = {}
        self._rate_limiters = {}
        self._circuit_breakers = {}
        self._cache = {}
        self._metrics = IOMetrics() if config.metrics_enabled else None
        
        # Initialize rate limiters
        if config.global_rate_limit:
            self._global_rate_limiter = RateLimiter(
                config.global_rate_limit,
                config.rate_limit_window
            )
        else:
            self._global_rate_limiter = None
            
        for endpoint in config.endpoints:
            if endpoint.rate_limit:
                self._rate_limiters[endpoint.name] = RateLimiter(
                    endpoint.rate_limit,
                    config.rate_limit_window
                )
                
        # Initialize circuit breakers
        for endpoint in config.endpoints:
            self._circuit_breakers[endpoint.name] = CircuitBreaker(
                config.circuit_breaker_threshold,
                config.circuit_breaker_timeout
            )
            
    def _build_fsm(self) -> SimpleFSM:
        """Build FSM for API orchestration."""
        # Create FSM configuration based on orchestration mode
        # Add start state
        states = [{'name': 'start', 'type': 'initial', 'is_start': True}]
        arcs = []
        
        if self.config.mode == OrchestrationMode.SEQUENTIAL:
            # Create sequential states
            for i, endpoint in enumerate(self.config.endpoints):
                state_name = f"call_{endpoint.name}"
                states.append({
                    'name': state_name,
                    'type': 'task'
                })
                
                if i == 0:
                    arcs.append({
                        'from': 'start',
                        'to': state_name,
                        'name': f'init_{endpoint.name}'
                    })
                    
                if i < len(self.config.endpoints) - 1:
                    next_state = f"call_{self.config.endpoints[i + 1].name}"
                    arcs.append({
                        'from': state_name,
                        'to': next_state,
                        'name': f'{endpoint.name}_to_{self.config.endpoints[i + 1].name}'
                    })
                else:
                    arcs.append({
                        'from': state_name,
                        'to': 'end',
                        'name': f'{endpoint.name}_complete'
                    })
                    
        elif self.config.mode == OrchestrationMode.PARALLEL:
            # Create parallel states with fork/join
            states.append({'name': 'fork', 'type': 'fork'})
            states.append({'name': 'join', 'type': 'join'})
            
            arcs.append({'from': 'start', 'to': 'fork', 'name': 'init_parallel'})
            
            for endpoint in self.config.endpoints:
                state_name = f"call_{endpoint.name}"
                states.append({
                    'name': state_name,
                    'type': 'task'
                })
                arcs.append({
                    'from': 'fork',
                    'to': state_name,
                    'name': f'fork_to_{endpoint.name}'
                })
                arcs.append({
                    'from': state_name,
                    'to': 'join',
                    'name': f'{endpoint.name}_to_join'
                })
                
            arcs.append({'from': 'join', 'to': 'end', 'name': 'parallel_complete'})
            
        elif self.config.mode == OrchestrationMode.PIPELINE:
            # Create pipeline with data transformation
            for i, endpoint in enumerate(self.config.endpoints):
                state_name = f"call_{endpoint.name}"
                transform_name = f"transform_{endpoint.name}"
                
                states.append({
                    'name': state_name,
                    'type': 'task'
                })
                
                if endpoint.transform_input:
                    states.append({
                        'name': transform_name,
                        'type': 'task'
                    })
                    
                if i == 0:
                    if endpoint.transform_input:
                        arcs.append({
                            'from': 'start',
                            'to': transform_name,
                            'name': f'init_transform_{endpoint.name}'
                        })
                        arcs.append({
                            'from': transform_name,
                            'to': state_name,
                            'name': f'transform_to_{endpoint.name}'
                        })
                    else:
                        arcs.append({
                            'from': 'start',
                            'to': state_name,
                            'name': f'init_{endpoint.name}'
                        })
                        
                if i < len(self.config.endpoints) - 1:
                    next_endpoint = self.config.endpoints[i + 1]
                    if next_endpoint.transform_input:
                        next_transform = f"transform_{next_endpoint.name}"
                        arcs.append({
                            'from': state_name,
                            'to': next_transform,
                            'name': f'{endpoint.name}_to_transform'
                        })
                    else:
                        next_state = f"call_{next_endpoint.name}"
                        arcs.append({
                            'from': state_name,
                            'to': next_state,
                            'name': f'{endpoint.name}_to_{next_endpoint.name}'
                        })
                else:
                    arcs.append({
                        'from': state_name,
                        'to': 'end',
                        'name': f'{endpoint.name}_complete'
                    })
        
        # Add end state
        states.append({
            'name': 'end',
            'type': 'terminal'
        })
                    
        # Build FSM configuration
        fsm_config = {
            'name': 'API_Orchestration',
            'data_mode': DataHandlingMode.REFERENCE.value,
            'states': states,
            'arcs': arcs,
            'resources': []  # HTTP providers created dynamically
        }
        
        return SimpleFSM(fsm_config)
        
    def _create_provider(self, endpoint: APIEndpoint):
        """Create I/O provider for endpoint.
        
        Args:
            endpoint: API endpoint configuration
            
        Returns:
            I/O provider instance
        """
        io_config = IOConfig(
            mode=IOMode.READ if endpoint.method == "GET" else IOMode.WRITE,
            format=IOFormat.API,
            source=endpoint.url,
            headers=endpoint.headers,
            timeout=endpoint.timeout,
            retry_count=endpoint.retry_count,
            retry_delay=endpoint.retry_delay
        )
        
        return create_io_provider(io_config, is_async=True)
        
    async def _call_endpoint(
        self,
        endpoint: APIEndpoint,
        input_data: Dict[str, Any] | None = None
    ) -> Any:
        """Call a single API endpoint.
        
        Args:
            endpoint: Endpoint configuration
            input_data: Input data for the endpoint
            
        Returns:
            API response
        """
        # Apply rate limiting
        if self._global_rate_limiter:
            await self._global_rate_limiter.acquire()
            
        if endpoint.name in self._rate_limiters:
            await self._rate_limiters[endpoint.name].acquire()
            
        # Check cache
        if self.config.cache_ttl and self.config.cache_key_generator:
            cache_key = self.config.cache_key_generator(endpoint)
            if cache_key in self._cache:
                cached_data, cached_time = self._cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self.config.cache_ttl:
                    return cached_data
                    
        # Transform input if needed
        if endpoint.transform_input and input_data:
            request_data = endpoint.transform_input(input_data)
        else:
            request_data = endpoint.body or {}
            
        # Create provider if not exists
        if endpoint.name not in self._providers:
            self._providers[endpoint.name] = self._create_provider(endpoint)
            
        provider = self._providers[endpoint.name]
        
        # Make API call with circuit breaker
        circuit_breaker = self._circuit_breakers[endpoint.name]
        
        async def make_request():
            if not provider.is_open:
                await provider.open()
                
            if endpoint.method == "GET":
                response = await provider.read(params=endpoint.params)
            elif endpoint.method == "POST":
                response = await provider.write(request_data, params=endpoint.params)
            else:
                # Handle other methods
                response = await provider.read(params=endpoint.params)
                
            return response
            
        try:
            # Execute with retry
            response = await retry_io_operation(
                lambda: circuit_breaker.call(make_request),
                max_retries=endpoint.retry_count,
                delay=endpoint.retry_delay
            )
            
            # Parse response if parser provided
            if endpoint.response_parser:
                response = endpoint.response_parser(response)
                
            # Cache response
            if self.config.cache_ttl and self.config.cache_key_generator:
                cache_key = self.config.cache_key_generator(endpoint)
                self._cache[cache_key] = (response, datetime.now())
                
            # Record metrics
            if self._metrics:
                self._metrics.record_read()
                
            return response
            
        except Exception as e:
            # Handle error
            if endpoint.error_handler:
                return endpoint.error_handler(e)
                
            if self._metrics:
                self._metrics.record_error()
                
            if self.config.fail_fast:
                raise
                
            return None
            
    async def orchestrate(
        self,
        input_data: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Execute API orchestration.
        
        Args:
            input_data: Initial input data
            
        Returns:
            Orchestration results
        """
        results = {}
        
        if self.config.mode == OrchestrationMode.SEQUENTIAL:
            # Execute sequentially
            current_data = input_data
            for endpoint in self.config.endpoints:
                result = await self._call_endpoint(endpoint, current_data)
                results[endpoint.name] = result
                current_data = result  # Pass result to next
                
        elif self.config.mode == OrchestrationMode.PARALLEL:
            # Execute in parallel
            tasks = []
            for endpoint in self.config.endpoints:
                task = self._call_endpoint(endpoint, input_data)
                tasks.append((endpoint.name, task))
                
            # Wait for all tasks
            for name, task in tasks:
                results[name] = await task
                
        elif self.config.mode == OrchestrationMode.PIPELINE:
            # Execute as pipeline
            current_data = input_data
            for endpoint in self.config.endpoints:
                result = await self._call_endpoint(endpoint, current_data)
                results[endpoint.name] = result
                current_data = result  # Pass result to next
                
        elif self.config.mode == OrchestrationMode.FANOUT:
            # Fan out to multiple endpoints
            tasks = []
            for endpoint in self.config.endpoints:
                task = self._call_endpoint(endpoint, input_data)
                tasks.append((endpoint.name, task))
                
            # Gather results
            for name, task in tasks:
                results[name] = await task
                
        # Merge results if merger provided
        if self.config.result_merger:
            merged = self.config.result_merger(list(results.values()))
            results['merged'] = merged
            
        # Transform final result if transformer provided
        if self.config.result_transformer:
            results = self.config.result_transformer(results)
            
        # Get metrics
        if self._metrics:
            results['_metrics'] = self._metrics.get_metrics()
            
        return results
        
    async def close(self) -> None:
        """Close all providers."""
        for provider in self._providers.values():
            if provider.is_open:
                await provider.close()


def create_rest_api_orchestrator(
    base_url: str,
    endpoints: List[Dict[str, Any]],
    auth_token: str | None = None,
    rate_limit: int = 60,
    mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL
) -> APIOrchestrator:
    """Create REST API orchestrator.
    
    Args:
        base_url: Base URL for all endpoints
        endpoints: List of endpoint configurations
        auth_token: Optional authentication token
        rate_limit: Requests per minute
        mode: Orchestration mode
        
    Returns:
        Configured API orchestrator
    """
    headers = {}
    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'
        
    api_endpoints = []
    for ep in endpoints:
        endpoint = APIEndpoint(
            name=ep['name'],
            url=f"{base_url}{ep['path']}",
            method=ep.get('method', 'GET'),
            headers={**headers, **ep.get('headers', {})},
            params=ep.get('params'),
            body=ep.get('body'),
            depends_on=ep.get('depends_on'),
            transform_input=ep.get('transform_input')
        )
        api_endpoints.append(endpoint)
        
    config = APIOrchestrationConfig(
        endpoints=api_endpoints,
        mode=mode,
        global_rate_limit=rate_limit,
        metrics_enabled=True
    )
    
    return APIOrchestrator(config)


def create_graphql_orchestrator(
    endpoint: str,
    queries: List[Dict[str, Any]],
    auth_token: str | None = None,
    batch_queries: bool = True
) -> APIOrchestrator:
    """Create GraphQL API orchestrator.
    
    Args:
        endpoint: GraphQL endpoint URL
        queries: List of GraphQL queries
        auth_token: Optional authentication token
        batch_queries: Whether to batch queries
        
    Returns:
        Configured API orchestrator
    """
    headers = {'Content-Type': 'application/json'}
    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'
        
    if batch_queries:
        # Create single batched endpoint
        batched_query = {
            'query': '\\n'.join(q['query'] for q in queries),
            'variables': {}
        }
        for q in queries:
            if 'variables' in q:
                batched_query['variables'].update(q['variables'])
                
        endpoint_config = APIEndpoint(
            name='graphql_batch',
            url=endpoint,
            method='POST',
            headers=headers,
            body=batched_query
        )
        
        config = APIOrchestrationConfig(
            endpoints=[endpoint_config],
            mode=OrchestrationMode.SEQUENTIAL
        )
    else:
        # Create separate endpoints for each query
        api_endpoints = []
        for q in queries:
            endpoint_config = APIEndpoint(
                name=q.get('name', f"query_{len(api_endpoints)}"),
                url=endpoint,
                method='POST',
                headers=headers,
                body={
                    'query': q['query'],
                    'variables': q.get('variables', {})
                }
            )
            api_endpoints.append(endpoint_config)
            
        config = APIOrchestrationConfig(
            endpoints=api_endpoints,
            mode=OrchestrationMode.PARALLEL
        )
        
    return APIOrchestrator(config)

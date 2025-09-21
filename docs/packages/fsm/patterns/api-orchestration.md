# API Orchestration Pattern

## Overview

The API Orchestration pattern provides pre-configured FSM patterns for orchestrating multiple API calls with support for sequential, parallel, pipeline, and conditional execution modes. It includes built-in rate limiting, circuit breakers, caching, and comprehensive error handling.

## Core Components

### APIEndpoint

Configuration for individual API endpoints:

```python
from dataknobs_fsm.patterns.api_orchestration import APIEndpoint

endpoint = APIEndpoint(
    name="user_service",
    url="https://api.example.com/users",
    method="GET",
    headers={"Authorization": "Bearer token"},
    params={"limit": 100},
    timeout=30.0,
    retry_count=3,
    retry_delay=1.0,
    rate_limit=60,  # 60 requests per minute
    response_parser=lambda r: r.json(),
    depends_on=["auth_service"]  # Dependencies
)
```

### OrchestrationMode

Execution modes for API orchestration:

```python
from dataknobs_fsm.patterns.api_orchestration import OrchestrationMode

OrchestrationMode.SEQUENTIAL   # Execute APIs one after another
OrchestrationMode.PARALLEL     # Execute APIs concurrently
OrchestrationMode.FANOUT       # One request triggers multiple APIs
OrchestrationMode.PIPELINE     # Output of one API feeds into next
OrchestrationMode.CONDITIONAL  # Execute based on conditions
OrchestrationMode.HYBRID       # Mix of above patterns
```

### APIOrchestrationConfig

Complete orchestration configuration:

```python
from dataknobs_fsm.patterns.api_orchestration import APIOrchestrationConfig

config = APIOrchestrationConfig(
    endpoints=[endpoint1, endpoint2, endpoint3],
    mode=OrchestrationMode.SEQUENTIAL,
    max_concurrent=10,
    total_timeout=300.0,
    fail_fast=False,
    global_rate_limit=100,  # 100 requests/minute globally
    error_threshold=0.1,     # Max 10% errors
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0,
    cache_ttl=300,  # Cache for 5 minutes
    metrics_enabled=True
)
```

## Basic Usage

### APIOrchestrator

```python
from dataknobs_fsm.patterns.api_orchestration import (
    APIOrchestrator,
    APIOrchestrationConfig,
    APIEndpoint
)
import asyncio

# Define endpoints
user_api = APIEndpoint(
    name="users",
    url="https://api.example.com/users",
    method="GET"
)

posts_api = APIEndpoint(
    name="posts",
    url="https://api.example.com/posts",
    method="GET",
    depends_on=["users"]
)

# Create configuration
config = APIOrchestrationConfig(
    endpoints=[user_api, posts_api],
    mode=OrchestrationMode.SEQUENTIAL
)

# Create orchestrator
orchestrator = APIOrchestrator(config)

# Execute orchestration
result = asyncio.run(orchestrator.execute({"user_id": 123}))
```

## Orchestration Modes

### Sequential Execution

APIs are called one after another:

```python
config = APIOrchestrationConfig(
    endpoints=[api1, api2, api3],
    mode=OrchestrationMode.SEQUENTIAL
)

# Execution order: api1 -> api2 -> api3
```

### Parallel Execution

APIs are called concurrently:

```python
config = APIOrchestrationConfig(
    endpoints=[api1, api2, api3],
    mode=OrchestrationMode.PARALLEL,
    max_concurrent=3
)

# All APIs called simultaneously
```

### Pipeline Execution

Output of one API feeds into the next:

```python
# Define transformation between APIs
api2 = APIEndpoint(
    name="enrichment",
    url="https://api.example.com/enrich",
    transform_input=lambda data: {
        "id": data["user"]["id"],
        "name": data["user"]["name"]
    }
)

config = APIOrchestrationConfig(
    endpoints=[api1, api2],
    mode=OrchestrationMode.PIPELINE
)
```

### Fanout Execution

One request triggers multiple APIs:

```python
config = APIOrchestrationConfig(
    endpoints=[notification_api, analytics_api, audit_api],
    mode=OrchestrationMode.FANOUT
)

# Single input sent to all APIs
```

### Conditional Execution

Execute APIs based on conditions:

```python
api2 = APIEndpoint(
    name="premium_service",
    url="https://api.example.com/premium",
    condition=lambda data: data.get("user_type") == "premium"
)

config = APIOrchestrationConfig(
    endpoints=[api1, api2],
    mode=OrchestrationMode.CONDITIONAL
)
```

## Rate Limiting

### Global Rate Limiting

Apply rate limits across all APIs:

```python
config = APIOrchestrationConfig(
    endpoints=endpoints,
    global_rate_limit=100,  # 100 requests per minute
    rate_limit_window=60    # 60 second window
)
```

### Per-Endpoint Rate Limiting

Apply specific limits to each endpoint:

```python
endpoint = APIEndpoint(
    name="limited_api",
    url="https://api.example.com/limited",
    rate_limit=10,  # 10 requests per minute
    burst_limit=5    # Allow burst of 5 requests
)
```

### Custom Rate Limiter

```python
from dataknobs_fsm.patterns.api_orchestration import RateLimiter

# Create custom rate limiter
limiter = RateLimiter(rate_limit=100, window=60)

# Use in async context
async def make_request():
    await limiter.acquire()  # Wait for permission
    # Make API call
    response = await http_client.get(url)
    return response
```

## Circuit Breaker

Prevent cascading failures with circuit breakers:

### Configuration

```python
config = APIOrchestrationConfig(
    endpoints=endpoints,
    circuit_breaker_threshold=5,    # Trip after 5 consecutive failures
    circuit_breaker_timeout=60.0    # Reset after 60 seconds
)
```

### Custom Circuit Breaker

```python
from dataknobs_fsm.patterns.api_orchestration import CircuitBreaker

breaker = CircuitBreaker(threshold=3, timeout=30.0)

async def protected_call():
    return await breaker.call(api_function, arg1, arg2)
```

## Error Handling

### Error Threshold

Stop orchestration if error rate exceeds threshold:

```python
config = APIOrchestrationConfig(
    endpoints=endpoints,
    error_threshold=0.1,  # Stop if >10% errors
    fail_fast=True        # Stop on first error
)
```

### Per-Endpoint Error Handling

```python
endpoint = APIEndpoint(
    name="api",
    url="https://api.example.com",
    error_handler=lambda e: {
        "error": str(e),
        "fallback": "default_value"
    }
)
```

### Retry Configuration

```python
endpoint = APIEndpoint(
    name="api",
    url="https://api.example.com",
    retry_count=3,
    retry_delay=1.0,  # Exponential backoff
    retry_on=[500, 502, 503, 504]  # Retry on these status codes
)
```

## Caching

### Response Caching

Cache API responses to reduce load:

```python
config = APIOrchestrationConfig(
    endpoints=endpoints,
    cache_ttl=300,  # Cache for 5 minutes
    cache_key_generator=lambda ep: f"{ep.name}:{ep.params}"
)
```

### Cache Invalidation

```python
orchestrator = APIOrchestrator(config)

# Clear cache for specific endpoint
orchestrator.clear_cache("user_service")

# Clear all cache
orchestrator.clear_all_cache()
```

## Response Handling

### Response Parsing

Parse and transform responses:

```python
endpoint = APIEndpoint(
    name="api",
    url="https://api.example.com",
    response_parser=lambda r: r.json()["data"],
    transform_output=lambda data: {
        "id": data["user_id"],
        "name": data["full_name"]
    }
)
```

### Result Merging

Merge results from multiple APIs:

```python
def merge_results(results):
    """Merge results from parallel APIs."""
    merged = {}
    for result in results:
        merged.update(result)
    return merged

config = APIOrchestrationConfig(
    endpoints=endpoints,
    mode=OrchestrationMode.PARALLEL,
    result_merger=merge_results
)
```

## Monitoring and Metrics

### Enable Metrics

```python
config = APIOrchestrationConfig(
    endpoints=endpoints,
    metrics_enabled=True,
    log_requests=True,
    log_responses=True
)

orchestrator = APIOrchestrator(config)
result = await orchestrator.execute(data)

# Access metrics
metrics = orchestrator.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Success rate: {metrics.success_rate}%")
print(f"Average latency: {metrics.avg_latency}ms")
```

### Custom Metrics

```python
from dataknobs_fsm.io.utils import IOMetrics

metrics = IOMetrics()

# Track custom metrics
metrics.record_request("custom_api", latency=150)
metrics.record_error("custom_api", error_type="timeout")
```

## Factory Functions

### Sequential API Chain

```python
from dataknobs_fsm.patterns.api_orchestration import create_sequential_api_chain

orchestrator = create_sequential_api_chain(
    endpoints=[
        ("auth", "https://api.example.com/auth"),
        ("user", "https://api.example.com/user"),
        ("profile", "https://api.example.com/profile")
    ],
    headers={"API-Key": "secret"}
)

result = await orchestrator.execute({"username": "user@example.com"})
```

### Parallel API Aggregator

```python
from dataknobs_fsm.patterns.api_orchestration import create_parallel_aggregator

orchestrator = create_parallel_aggregator(
    endpoints=[
        ("service1", "https://api1.example.com/data"),
        ("service2", "https://api2.example.com/data"),
        ("service3", "https://api3.example.com/data")
    ],
    max_concurrent=3,
    timeout=10.0
)

aggregated = await orchestrator.execute()
```

## Complete Examples

### Example 1: User Data Enrichment

```python
import asyncio
from dataknobs_fsm.patterns.api_orchestration import (
    APIOrchestrator,
    APIOrchestrationConfig,
    APIEndpoint,
    OrchestrationMode
)

async def enrich_user_data(user_id):
    # Define API endpoints
    user_api = APIEndpoint(
        name="user",
        url=f"https://api.example.com/users/{user_id}",
        method="GET",
        response_parser=lambda r: r.json()
    )

    posts_api = APIEndpoint(
        name="posts",
        url=f"https://api.example.com/users/{user_id}/posts",
        method="GET",
        response_parser=lambda r: r.json()["posts"]
    )

    analytics_api = APIEndpoint(
        name="analytics",
        url=f"https://analytics.example.com/users/{user_id}",
        method="GET",
        rate_limit=10  # Limited API
    )

    # Configure orchestration
    config = APIOrchestrationConfig(
        endpoints=[user_api, posts_api, analytics_api],
        mode=OrchestrationMode.PARALLEL,
        max_concurrent=3,
        fail_fast=False,  # Continue even if one fails
        cache_ttl=600,     # Cache for 10 minutes
        result_merger=lambda results: {
            "user": results[0],
            "posts": results[1],
            "analytics": results[2] if len(results) > 2 else None
        }
    )

    # Execute orchestration
    orchestrator = APIOrchestrator(config)
    enriched_data = await orchestrator.execute({"user_id": user_id})

    return enriched_data

# Run enrichment
result = asyncio.run(enrich_user_data(123))
```

### Example 2: Payment Processing Pipeline

```python
async def process_payment(payment_data):
    # Validation API
    validate_api = APIEndpoint(
        name="validate",
        url="https://payment.example.com/validate",
        method="POST",
        body=payment_data,
        timeout=5.0
    )

    # Fraud check API
    fraud_api = APIEndpoint(
        name="fraud_check",
        url="https://fraud.example.com/check",
        method="POST",
        transform_input=lambda data: {
            "amount": data["amount"],
            "card_hash": data["card_token"]
        },
        condition=lambda data: data["amount"] > 100  # Only for large amounts
    )

    # Payment processor
    payment_api = APIEndpoint(
        name="process",
        url="https://processor.example.com/charge",
        method="POST",
        retry_count=1,  # Limited retries for payments
        depends_on=["validate", "fraud_check"]
    )

    # Notification service
    notify_api = APIEndpoint(
        name="notify",
        url="https://notify.example.com/payment",
        method="POST",
        depends_on=["process"]
    )

    # Configure pipeline
    config = APIOrchestrationConfig(
        endpoints=[validate_api, fraud_api, payment_api, notify_api],
        mode=OrchestrationMode.PIPELINE,
        fail_fast=True,  # Stop on any failure
        circuit_breaker_threshold=3,
        total_timeout=30.0
    )

    orchestrator = APIOrchestrator(config)
    result = await orchestrator.execute(payment_data)

    return result
```

### Example 3: Microservices Health Check

```python
async def health_check_all_services():
    services = [
        "auth", "user", "payment", "inventory",
        "shipping", "notification", "analytics"
    ]

    # Create health check endpoints
    endpoints = [
        APIEndpoint(
            name=service,
            url=f"https://{service}.example.com/health",
            method="GET",
            timeout=2.0,
            retry_count=0,  # No retries for health checks
            error_handler=lambda e: {"status": "down", "error": str(e)}
        )
        for service in services
    ]

    # Configure parallel health checks
    config = APIOrchestrationConfig(
        endpoints=endpoints,
        mode=OrchestrationMode.PARALLEL,
        max_concurrent=10,
        total_timeout=5.0,
        fail_fast=False,  # Check all services
        result_merger=lambda results: {
            endpoints[i].name: results[i]
            for i in range(len(results))
        }
    )

    orchestrator = APIOrchestrator(config)
    health_status = await orchestrator.execute({})

    # Analyze results
    healthy = sum(1 for s in health_status.values() if s.get("status") == "ok")
    total = len(health_status)

    return {
        "healthy": healthy,
        "total": total,
        "percentage": (healthy / total) * 100,
        "services": health_status
    }
```

## Best Practices

### 1. Use Appropriate Mode

- **Sequential**: When order matters or dependencies exist
- **Parallel**: For independent APIs to reduce latency
- **Pipeline**: When data flows through transformations
- **Fanout**: For event broadcasting

### 2. Configure Timeouts

```python
# Set reasonable timeouts
endpoint = APIEndpoint(
    name="api",
    url="https://api.example.com",
    timeout=10.0  # Endpoint timeout
)

config = APIOrchestrationConfig(
    endpoints=[endpoint],
    total_timeout=30.0  # Overall timeout
)
```

### 3. Handle Failures Gracefully

```python
# Provide fallbacks
endpoint = APIEndpoint(
    name="api",
    url="https://api.example.com",
    error_handler=lambda e: {"fallback": "default_value"}
)

# Use circuit breakers
config = APIOrchestrationConfig(
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0
)
```

### 4. Monitor Performance

```python
# Enable metrics
config = APIOrchestrationConfig(
    metrics_enabled=True,
    log_requests=True
)

# Review metrics regularly
metrics = orchestrator.get_metrics()
if metrics.error_rate > 0.05:  # >5% errors
    alert_ops_team()
```

## Troubleshooting

### Common Issues

1. **Rate Limit Exceeded**
   - Reduce request rate
   - Implement backoff
   - Use caching

2. **Circuit Breaker Open**
   - Check endpoint health
   - Review error logs
   - Adjust thresholds

3. **Timeout Errors**
   - Increase timeouts
   - Optimize API calls
   - Use parallel execution

4. **High Error Rate**
   - Check API availability
   - Review request format
   - Implement retries

## Next Steps

- [Error Recovery Pattern](error-recovery.md) - Advanced error handling
- [LLM Workflow Pattern](llm-workflow.md) - LLM API orchestration
- [Streaming Guide](../guides/streaming.md) - Stream processing
- [Examples](../examples/api-workflow.md) - More API examples
# Error Recovery Pattern

## Overview

The Error Recovery pattern provides pre-configured FSM patterns for building resilient workflows with comprehensive error handling strategies including retry mechanisms, circuit breakers, fallback options, compensation logic, and bulkhead isolation.

## Core Components

### RecoveryStrategy

Available error recovery strategies:

```python
from dataknobs_fsm.patterns.error_recovery import RecoveryStrategy

RecoveryStrategy.RETRY           # Simple retry with backoff
RecoveryStrategy.CIRCUIT_BREAKER # Circuit breaker pattern
RecoveryStrategy.FALLBACK        # Use fallback value/service
RecoveryStrategy.COMPENSATE      # Compensation/rollback
RecoveryStrategy.DEADLINE        # Deadline-based timeout
RecoveryStrategy.BULKHEAD        # Isolate failures
RecoveryStrategy.CACHE            # Use cached results
```

### BackoffStrategy

Retry backoff strategies:

```python
from dataknobs_fsm.patterns.error_recovery import BackoffStrategy

BackoffStrategy.FIXED         # Fixed delay between retries
BackoffStrategy.LINEAR        # Linear increase in delay
BackoffStrategy.EXPONENTIAL   # Exponential increase
BackoffStrategy.JITTER        # Random jitter added
BackoffStrategy.DECORRELATED  # Decorrelated jitter
```

## Configuration

### RetryConfig

Configure retry behavior:

```python
from dataknobs_fsm.patterns.error_recovery import RetryConfig, BackoffStrategy

retry_config = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,  # seconds
    max_delay=60.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    backoff_multiplier=2.0,
    jitter_range=0.1,  # 10% jitter
    retry_on_exceptions=[ConnectionError, TimeoutError],
    retry_on_result=lambda r: r.status_code >= 500,
    on_retry=lambda n, e: print(f"Retry {n}: {e}"),
    on_failure=lambda e: print(f"Final failure: {e}")
)
```

### CircuitBreakerConfig

Configure circuit breaker:

```python
from dataknobs_fsm.patterns.error_recovery import CircuitBreakerConfig

circuit_config = CircuitBreakerConfig(
    failure_threshold=5,      # Failures to open circuit
    success_threshold=2,      # Successes to close circuit
    timeout=60.0,            # Time before half-open state
    window_size=10,          # Rolling window size
    window_duration=60.0,    # Window duration in seconds
    on_open=lambda: print("Circuit opened"),
    on_close=lambda: print("Circuit closed"),
    on_half_open=lambda: print("Circuit half-open")
)
```

### FallbackConfig

Configure fallback behavior:

```python
from dataknobs_fsm.patterns.error_recovery import FallbackConfig

fallback_config = FallbackConfig(
    fallback_value="default_response",
    fallback_function=lambda e: f"Error occurred: {e}",
    fallback_service="https://backup.api.com",
    use_cache=True,
    cache_ttl=300.0,  # 5 minutes
    fallback_on_exceptions=[ServiceUnavailable],
    fallback_on_timeout=True
)
```

### CompensationConfig

Configure compensation/rollback:

```python
from dataknobs_fsm.patterns.error_recovery import CompensationConfig

compensation_config = CompensationConfig(
    compensation_actions=[
        lambda state: rollback_database(state),
        lambda state: cancel_payment(state),
        lambda state: notify_failure(state)
    ],
    save_state=True,
    use_sagas=True,
    saga_timeout=300.0,
    on_compensation_start=lambda: print("Starting compensation"),
    on_compensation_complete=lambda: print("Compensation complete")
)
```

### BulkheadConfig

Configure bulkhead isolation:

```python
from dataknobs_fsm.patterns.error_recovery import BulkheadConfig

bulkhead_config = BulkheadConfig(
    max_concurrent=10,
    max_queue_size=100,
    queue_timeout=30.0,
    use_thread_pool=False,
    thread_pool_size=5,
    track_metrics=True
)
```

## Basic Usage

### Error Recovery Workflow

```python
from dataknobs_fsm.patterns.error_recovery import (
    ErrorRecoveryWorkflow,
    ErrorRecoveryConfig,
    RecoveryStrategy
)
import asyncio

# Configure error recovery
config = ErrorRecoveryConfig(
    primary_strategy=RecoveryStrategy.RETRY,
    retry_config=retry_config,
    circuit_breaker_config=circuit_config,
    max_total_attempts=10,
    global_timeout=300.0,
    transient_errors=[ConnectionError, TimeoutError],
    permanent_errors=[ValueError, TypeError],
    log_errors=True,
    metrics_enabled=True
)

# Create workflow
workflow = ErrorRecoveryWorkflow(config)

# Execute with recovery
result = await workflow.execute(
    func=unreliable_function,
    args=(arg1, arg2),
    kwargs={"timeout": 10}
)
```

## Recovery Strategies

### Retry Pattern

Automatic retry with configurable backoff:

```python
from dataknobs_fsm.patterns.error_recovery import RetryExecutor

executor = RetryExecutor(retry_config)

async def flaky_operation():
    # Operation that might fail
    if random.random() < 0.3:
        raise ConnectionError("Network issue")
    return "success"

# Execute with retries
result = await executor.execute(flaky_operation)
```

### Circuit Breaker Pattern

Prevent cascading failures:

```python
from dataknobs_fsm.patterns.error_recovery import CircuitBreaker

breaker = CircuitBreaker(circuit_config)

async def external_service_call():
    response = await http_client.get("https://api.example.com")
    return response.json()

# Execute with circuit breaker
try:
    result = await breaker.call(external_service_call)
except CircuitBreakerError as e:
    print(f"Circuit is open, wait {e.wait_time}s")
    result = use_fallback()
```

### Fallback Pattern

Provide alternative responses on failure:

```python
config = ErrorRecoveryConfig(
    primary_strategy=RecoveryStrategy.FALLBACK,
    fallback_config=FallbackConfig(
        fallback_function=lambda e: {
            "status": "degraded",
            "data": get_cached_data(),
            "error": str(e)
        }
    )
)

workflow = ErrorRecoveryWorkflow(config)
result = await workflow.execute(primary_function)
```

### Compensation Pattern

Rollback on failure with sagas:

```python
# Define compensation actions
def compensate_payment(state):
    # Reverse the payment
    payment_id = state["payment_id"]
    cancel_payment(payment_id)

def compensate_inventory(state):
    # Restore inventory
    items = state["reserved_items"]
    release_inventory(items)

config = ErrorRecoveryConfig(
    primary_strategy=RecoveryStrategy.COMPENSATE,
    compensation_config=CompensationConfig(
        compensation_actions=[
            compensate_payment,
            compensate_inventory
        ],
        save_state=True
    )
)
```

### Bulkhead Isolation

Isolate failures to prevent system-wide impact:

```python
from dataknobs_fsm.patterns.error_recovery import Bulkhead

bulkhead = Bulkhead(bulkhead_config)

async def resource_intensive_operation():
    # Operation that uses shared resources
    await process_data()

# Execute with bulkhead isolation
try:
    result = await bulkhead.execute(resource_intensive_operation)
except BulkheadTimeoutError:
    print("Too many concurrent requests")
```

## Advanced Patterns

### Layered Recovery

Combine multiple strategies:

```python
config = ErrorRecoveryConfig(
    primary_strategy=RecoveryStrategy.RETRY,
    secondary_strategies=[
        RecoveryStrategy.CIRCUIT_BREAKER,
        RecoveryStrategy.FALLBACK
    ],
    retry_config=retry_config,
    circuit_breaker_config=circuit_config,
    fallback_config=fallback_config
)

# Execution flow:
# 1. Retry on transient errors
# 2. Check circuit breaker
# 3. Use fallback if all else fails
```

### Deadline-Based Execution

Set execution deadlines:

```python
async def deadline_execution():
    config = ErrorRecoveryConfig(
        primary_strategy=RecoveryStrategy.DEADLINE,
        global_timeout=30.0  # 30 second deadline
    )

    workflow = ErrorRecoveryWorkflow(config)

    try:
        result = await asyncio.wait_for(
            workflow.execute(long_running_task),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        result = handle_timeout()
```

### Cache-Based Recovery

Use cached results on failure:

```python
config = ErrorRecoveryConfig(
    primary_strategy=RecoveryStrategy.CACHE,
    fallback_config=FallbackConfig(
        use_cache=True,
        cache_ttl=600.0  # 10 minutes
    )
)

workflow = ErrorRecoveryWorkflow(config)

# First call populates cache
result1 = await workflow.execute(fetch_data)

# Subsequent failures use cache
result2 = await workflow.execute(fetch_data)  # Uses cache if fetch fails
```

## Error Classification

### Transient vs Permanent Errors

```python
config = ErrorRecoveryConfig(
    primary_strategy=RecoveryStrategy.RETRY,
    transient_errors=[
        ConnectionError,
        TimeoutError,
        ServiceUnavailable,
        RateLimitError
    ],
    permanent_errors=[
        ValueError,
        TypeError,
        AuthenticationError,
        NotFoundError
    ],
    retry_config=RetryConfig(
        max_attempts=3,
        retry_on_exceptions=transient_errors  # Only retry transient
    )
)
```

## Monitoring and Metrics

### Track Recovery Metrics

```python
workflow = ErrorRecoveryWorkflow(config)
result = await workflow.execute(operation)

# Get metrics
metrics = workflow.get_metrics()
print(f"Total attempts: {metrics['attempts']}")
print(f"Success rate: {metrics['successes'] / metrics['attempts']}")
print(f"Fallback usage: {metrics['fallbacks']}")
print(f"Compensations: {metrics['compensations']}")
```

### Alert on Threshold

```python
config = ErrorRecoveryConfig(
    primary_strategy=RecoveryStrategy.RETRY,
    alert_threshold=10,  # Alert after 10 errors
    log_errors=True,
    metrics_enabled=True
)

workflow = ErrorRecoveryWorkflow(config)

# Set up alerting
workflow.on_alert = lambda count: send_alert(f"Error count: {count}")
```

## Factory Functions

### Create Retry Workflow

```python
from dataknobs_fsm.patterns.error_recovery import create_retry_workflow

workflow = create_retry_workflow(
    max_attempts=3,
    backoff_strategy="exponential",
    initial_delay=1.0
)

result = await workflow.execute(operation)
```

### Create Circuit Breaker Workflow

```python
from dataknobs_fsm.patterns.error_recovery import create_circuit_breaker_workflow

workflow = create_circuit_breaker_workflow(
    failure_threshold=5,
    timeout=60.0,
    fallback_value="service_unavailable"
)
```

## Complete Examples

### Example 1: Resilient API Client

```python
import asyncio
from dataknobs_fsm.patterns.error_recovery import (
    ErrorRecoveryWorkflow,
    ErrorRecoveryConfig,
    RecoveryStrategy,
    RetryConfig,
    CircuitBreakerConfig,
    BackoffStrategy
)

class ResilientAPIClient:
    def __init__(self):
        # Configure retry with exponential backoff
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=10.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter_range=0.2,
            retry_on_exceptions=[ConnectionError, TimeoutError],
            on_retry=self.log_retry
        )

        # Configure circuit breaker
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=30.0,
            on_open=lambda: print("API circuit opened - using fallback")
        )

        # Configure error recovery
        self.workflow = ErrorRecoveryWorkflow(
            ErrorRecoveryConfig(
                primary_strategy=RecoveryStrategy.RETRY,
                secondary_strategies=[RecoveryStrategy.CIRCUIT_BREAKER],
                retry_config=retry_config,
                circuit_breaker_config=circuit_config,
                fallback_config=FallbackConfig(
                    fallback_function=self.get_cached_response,
                    use_cache=True,
                    cache_ttl=300.0
                )
            )
        )

    def log_retry(self, attempt, error):
        print(f"Retry attempt {attempt} after error: {error}")

    def get_cached_response(self, error):
        return {"cached": True, "data": "fallback_data"}

    async def call_api(self, endpoint, **kwargs):
        async def api_request():
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()

        return await self.workflow.execute(api_request)

# Use resilient client
client = ResilientAPIClient()
result = await client.call_api("https://api.example.com/data")
```

### Example 2: Database Transaction with Compensation

```python
async def process_order_with_compensation(order_data):
    # Define compensation actions
    compensation_actions = []

    async def reserve_inventory():
        # Reserve items from inventory
        reservation_id = await inventory_service.reserve(order_data["items"])
        compensation_actions.append(
            lambda: inventory_service.release(reservation_id)
        )
        return reservation_id

    async def charge_payment():
        # Charge customer
        payment_id = await payment_service.charge(
            order_data["customer_id"],
            order_data["amount"]
        )
        compensation_actions.append(
            lambda: payment_service.refund(payment_id)
        )
        return payment_id

    async def create_shipment():
        # Create shipment
        shipment_id = await shipping_service.create(
            order_data["address"],
            order_data["items"]
        )
        compensation_actions.append(
            lambda: shipping_service.cancel(shipment_id)
        )
        return shipment_id

    # Configure compensation
    config = ErrorRecoveryConfig(
        primary_strategy=RecoveryStrategy.COMPENSATE,
        compensation_config=CompensationConfig(
            compensation_actions=compensation_actions,
            save_state=True,
            use_sagas=True
        )
    )

    workflow = ErrorRecoveryWorkflow(config)

    try:
        # Execute order processing
        reservation = await workflow.execute(reserve_inventory)
        payment = await workflow.execute(charge_payment)
        shipment = await workflow.execute(create_shipment)

        return {
            "status": "success",
            "reservation_id": reservation,
            "payment_id": payment,
            "shipment_id": shipment
        }
    except Exception as e:
        # Compensation will be triggered automatically
        return {
            "status": "failed",
            "error": str(e),
            "compensated": True
        }
```

### Example 3: Microservice with Bulkhead Isolation

```python
class MicroserviceEndpoint:
    def __init__(self):
        # Configure bulkhead for each endpoint
        self.bulkheads = {
            "compute": Bulkhead(BulkheadConfig(
                max_concurrent=5,
                max_queue_size=20,
                queue_timeout=10.0
            )),
            "io": Bulkhead(BulkheadConfig(
                max_concurrent=20,
                max_queue_size=100,
                queue_timeout=30.0
            )),
            "database": Bulkhead(BulkheadConfig(
                max_concurrent=10,
                max_queue_size=50,
                queue_timeout=15.0
            ))
        }

    async def handle_compute_request(self, data):
        async def compute():
            # CPU-intensive operation
            result = await perform_computation(data)
            return result

        return await self.bulkheads["compute"].execute(compute)

    async def handle_io_request(self, data):
        async def io_operation():
            # I/O operation
            result = await fetch_external_data(data)
            return result

        return await self.bulkheads["io"].execute(io_operation)

    async def handle_database_request(self, query):
        async def db_operation():
            # Database operation
            result = await execute_query(query)
            return result

        return await self.bulkheads["database"].execute(db_operation)

    def get_metrics(self):
        return {
            name: bulkhead.metrics
            for name, bulkhead in self.bulkheads.items()
        }
```

## Best Practices

### 1. Choose Appropriate Strategy

- **Retry**: For transient network errors
- **Circuit Breaker**: For external service failures
- **Fallback**: When degraded service is acceptable
- **Compensation**: For transactional operations
- **Bulkhead**: For resource isolation

### 2. Configure Sensible Timeouts

```python
config = ErrorRecoveryConfig(
    global_timeout=30.0,  # Overall timeout
    retry_config=RetryConfig(
        max_delay=5.0  # Max retry delay
    ),
    circuit_breaker_config=CircuitBreakerConfig(
        timeout=60.0  # Circuit reset timeout
    )
)
```

### 3. Monitor and Alert

```python
workflow = ErrorRecoveryWorkflow(config)
workflow.on_error = lambda e: log_error(e)
workflow.on_alert = lambda count: send_alert(count)

# Regular health checks
health = workflow.get_health_status()
if health["error_rate"] > 0.1:
    trigger_investigation()
```

### 4. Test Recovery Scenarios

```python
import pytest

@pytest.mark.asyncio
async def test_retry_on_transient_error():
    config = ErrorRecoveryConfig(
        primary_strategy=RecoveryStrategy.RETRY,
        retry_config=RetryConfig(max_attempts=3)
    )

    workflow = ErrorRecoveryWorkflow(config)

    call_count = 0

    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Transient error")
        return "success"

    result = await workflow.execute(flaky_function)
    assert result == "success"
    assert call_count == 3
```

## Troubleshooting

### Common Issues

1. **Too Many Retries**
   - Reduce max_attempts
   - Increase backoff delay
   - Check for permanent errors

2. **Circuit Breaker Always Open**
   - Check failure threshold
   - Increase timeout
   - Verify service health

3. **Bulkhead Rejection**
   - Increase max_concurrent
   - Increase queue_size
   - Optimize operation time

4. **Compensation Failures**
   - Ensure idempotent operations
   - Add compensation retry logic
   - Log compensation state

## Next Steps

- [ETL Pattern](etl.md) - ETL with error recovery
- [API Orchestration Pattern](api-orchestration.md) - API resilience
- [CLI Guide](../guides/cli.md) - Command-line interface guide


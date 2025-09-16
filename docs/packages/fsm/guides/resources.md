# Resource Management Guide

## Overview

The DataKnobs FSM package provides comprehensive resource management for external services and dependencies. Resources are managed through a central ResourceManager that handles acquisition, release, pooling, and health monitoring.

## Understanding Resources

Resources represent external dependencies that your FSM needs to interact with:
- Database connections
- File systems
- HTTP services
- LLM providers
- Custom services

### Key Features

- **Automatic Lifecycle Management**: Resources are acquired when needed and released when done
- **Connection Pooling**: Efficient reuse of connections
- **Health Monitoring**: Health checks and metrics tracking
- **Owner Tracking**: Track which states own which resources
- **Cleanup Guarantees**: Automatic cleanup on errors or completion

## Resource Interface

All resources implement the `IResourceProvider` protocol:

```python
from dataknobs_fsm.resources.base import IResourceProvider, ResourceHealth, ResourceMetrics
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class IResourceProvider(Protocol):
    """Interface for resource providers."""

    def acquire(self, **kwargs) -> Any:
        """Acquire a resource."""
        ...

    def release(self, resource: Any) -> None:
        """Release a resource."""
        ...

    def validate(self, resource: Any) -> bool:
        """Validate that a resource is still valid."""
        ...

    def health_check(self) -> ResourceHealth:
        """Check the health of the resource provider."""
        ...

    def get_metrics(self) -> ResourceMetrics:
        """Get resource metrics."""
        ...
```

## Resource Manager

The ResourceManager handles all resource lifecycle operations:

```python
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.resources.database import DatabaseResourceAdapter

# Create resource manager
manager = ResourceManager()

# Register a database resource
db_resource = DatabaseResourceAdapter(
    name="main_db",
    backend="postgresql",
    connection_string="postgresql://user:pass@localhost/db"
)
manager.register_provider("database", db_resource)

# Acquire resource with owner tracking
resource = manager.acquire("database", owner_id="state_123")

# Use resource...

# Release when done
manager.release("database", owner_id="state_123")

# Or use context manager
with manager.resource_context("database", owner_id="state_456") as db:
    # Use database
    pass  # Automatically released
```

### Resource Status and Metrics

```python
# Get resource status
status = manager.get_resource_status("database")
print(f"Active connections: {status['active_count']}")
print(f"Current owners: {status['owners']}")

# Get metrics
metrics = manager.get_metrics("database")
print(f"Total acquisitions: {metrics['database'].total_acquisitions}")
print(f"Average hold time: {metrics['database'].average_hold_time}s")

# Health check
health = manager.health_check("database")
print(f"Database health: {health['database']}")  # ResourceHealth.HEALTHY
```

## Built-in Resource Adapters

### Database Resource

Connect to any database supported by dataknobs_data:

```python
from dataknobs_fsm.resources.database import DatabaseResourceAdapter

# Create database resource
db_resource = DatabaseResourceAdapter(
    name="postgres_db",
    backend="postgresql",
    host="localhost",
    port=5432,
    database="mydb",
    user="user",
    password="pass"
)

# Register with manager
manager.register_provider("database", db_resource)

# Use in FSM
from dataknobs_fsm.api.simple import SimpleFSM

fsm = SimpleFSM(config)
# Resources can be passed to SimpleFSM constructor
fsm = SimpleFSM(config, resources={"database": db_config})
```

#### Supported Backends

Via dataknobs_data integration:
- PostgreSQL
- MySQL
- SQLite
- Memory (for testing)
- File-based databases

### File System Resource

Manage file system operations:

```python
from dataknobs_fsm.resources.filesystem import FileSystemResourceAdapter

# Create filesystem resource
fs_resource = FileSystemResourceAdapter(
    name="data_fs",
    base_path="/data/processing",
    temp_dir="/tmp/fsm",
    auto_cleanup=True
)

manager.register_provider("filesystem", fs_resource)

# Acquire and use
with manager.resource_context("filesystem", "state_123") as fs:
    # Read file
    content = fs.read("input.csv")

    # Write file
    fs.write("output.json", processed_data)

    # Create temp file
    temp_path = fs.create_temp()
```

### HTTP Service Resource

Manage HTTP connections:

```python
from dataknobs_fsm.resources.http import HTTPResourceAdapter

# Create HTTP resource
http_resource = HTTPResourceAdapter(
    name="api_service",
    base_url="https://api.example.com",
    timeout=30,
    headers={"Authorization": "Bearer token"},
    retry_count=3
)

manager.register_provider("api", http_resource)

# Use in state
with manager.resource_context("api", "state_123") as api:
    response = api.get("/users/123")
    data = response.json()
```

### LLM Resource

Integrate with Large Language Model providers:

```python
from dataknobs_fsm.resources.llm import LLMResourceAdapter

# Create LLM resource
llm_resource = LLMResourceAdapter(
    name="gpt4",
    provider="openai",
    model="gpt-4",
    api_key="${OPENAI_API_KEY}",
    temperature=0.7,
    max_tokens=1000
)

manager.register_provider("llm", llm_resource)

# Use in state
with manager.resource_context("llm", "state_123") as llm:
    response = llm.complete(
        prompt="Analyze this text",
        system="You are a helpful assistant"
    )
```

## Resource Pooling

Configure connection pooling for better performance:

```python
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig

# Configure pool
pool_config = PoolConfig(
    min_size=2,
    max_size=10,
    max_idle_time=300,  # 5 minutes
    acquire_timeout=30
)

# Register pooled resource
db_resource = DatabaseResourceAdapter(name="db", backend="postgresql")
manager.register_provider("database", db_resource, pool_config=pool_config)

# Pool metrics
pool_metrics = manager.get_metrics("database_pool")
print(f"Pool size: {pool_metrics['database_pool'].size}")
print(f"Available: {pool_metrics['database_pool'].available}")
```

## Resource Requirements in FSM

Define resource requirements in FSM configuration:

### Configuration-Based Requirements

```yaml
networks:
  - name: main
    resources:
      - database
      - filesystem
    states:
      - name: process_data
        resources:
          - database  # This state needs database
        functions:
          transform:
            type: custom
            name: process_with_db
```

### Code-Based Requirements

```python
from dataknobs_fsm.functions.base import ResourceConfig

# Define resource requirements
requirements = [
    ResourceConfig(name="database", required=True, timeout=30),
    ResourceConfig(name="cache", required=False)
]

# Configure resources from requirements
resources = manager.configure_from_requirements(requirements, owner_id="state_123")
```

## Resource Lifecycle

### Acquisition and Release Flow

```python
# Resource lifecycle in state execution
class StateExecution:
    def execute(self, state, context):
        owner_id = f"state_{state.name}_{context.id}"

        # 1. Acquire required resources
        for resource_name in state.resources:
            resource = manager.acquire(resource_name, owner_id)
            context.resources[resource_name] = resource

        try:
            # 2. Execute state with resources
            result = state.transform(context.data, context.resources)

            # 3. Release resources on success
            for resource_name in state.resources:
                manager.release(resource_name, owner_id)

            return result

        except Exception as e:
            # 4. Cleanup on error
            manager.release_all(owner_id)
            raise
```

### Owner Tracking

Resources are tracked by owner ID:

```python
# Check if owner has resource
if manager.has_resource("database", "state_123"):
    db = manager.get_resource("database", "state_123")

# Get all owners of a resource
owners = manager.get_resource_owners("database")
print(f"Database is used by: {owners}")

# Release all resources for an owner
manager.release_all("state_123")
```

## Custom Resources

Create custom resources by implementing the IResourceProvider interface:

```python
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceStatus,
    ResourceMetrics
)

class RedisResourceProvider(BaseResourceProvider):
    def __init__(self, name: str, host: str, port: int = 6379):
        super().__init__(name, {"host": host, "port": port})
        self.host = host
        self.port = port
        self.client = None

    def acquire(self, **kwargs):
        import redis
        if not self.client:
            self.client = redis.Redis(host=self.host, port=self.port)
        self.status = ResourceStatus.ACTIVE
        return self.client

    def release(self, resource):
        self.status = ResourceStatus.IDLE
        # Keep connection for reuse

    def validate(self, resource):
        try:
            return resource.ping()
        except:
            return False

    def health_check(self):
        if self.validate(self.client):
            return ResourceHealth.HEALTHY
        return ResourceHealth.UNHEALTHY

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

# Register custom resource
redis = RedisResourceProvider("cache", "localhost")
manager.register_provider("redis", redis)
```

## Simple Resource Provider

For testing or simple use cases, create resources from dictionaries:

```python
# Create simple provider from dict
manager.register_from_dict("config", {
    "data": {
        "api_key": "secret",
        "endpoint": "https://api.example.com"
    }
})

# Or create simple provider with static data
simple_provider = manager.create_simple_provider("settings", {"timeout": 30})
manager.register_provider("settings", simple_provider)
```

## Resource Health Monitoring

### Health Checks

```python
# Check specific resource health
health = manager.health_check("database")
if health["database"] == ResourceHealth.UNHEALTHY:
    # Handle unhealthy resource
    pass

# Check all resources
all_health = manager.health_check()
for name, status in all_health.items():
    print(f"{name}: {status}")
```

### Resource Metrics

Track resource usage:

```python
from dataknobs_fsm.resources.base import ResourceMetrics

# Get metrics for specific resource
metrics = manager.get_metrics("database")
db_metrics = metrics["database"]

print(f"Total acquisitions: {db_metrics.total_acquisitions}")
print(f"Active connections: {db_metrics.active_connections}")
print(f"Failed acquisitions: {db_metrics.failed_acquisitions}")
print(f"Average hold time: {db_metrics.average_hold_time}s")
print(f"Average acquisition time: {db_metrics.average_acquisition_time}s")

# Get all metrics
all_metrics = manager.get_metrics()
```

## Async Cleanup

The ResourceManager supports async cleanup for resources:

```python
import asyncio

async def cleanup_resources():
    # Async cleanup of all resources
    await manager.cleanup()

# Run cleanup
asyncio.run(cleanup_resources())
```

## Resource Patterns

### Resource Manager as Context Manager

```python
# Auto-cleanup with context manager
with ResourceManager() as manager:
    manager.register_provider("db", db_resource)

    # Use resources
    with manager.resource_context("db", "owner1") as db:
        result = db.query("SELECT * FROM users")

# Manager and all resources cleaned up automatically
```

### Shared Resource Manager

```python
# Singleton pattern for shared resources
class SharedResources:
    _instance = None
    _manager = None

    @classmethod
    def get_manager(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._manager = ResourceManager()
            cls._setup_resources()
        return cls._manager

    @classmethod
    def _setup_resources(cls):
        # Register shared resources
        cls._manager.register_provider(
            "database",
            DatabaseResourceAdapter("db", backend="postgresql")
        )
```

### Resource with FSM

```python
from dataknobs_fsm.api.simple import SimpleFSM

# Pass resources to FSM
fsm = SimpleFSM(
    config,
    resources={
        "database": {"backend": "postgresql", "host": "localhost"},
        "cache": {"type": "redis", "host": "localhost"}
    }
)

# Resources are automatically registered and available in states
```

## Best Practices

### 1. Always Release Resources

Use context managers or try/finally:

```python
# Good - automatic cleanup
with manager.resource_context("database", owner_id) as db:
    result = db.query(sql)

# Also good - explicit cleanup
resource = manager.acquire("database", owner_id)
try:
    result = use_resource(resource)
finally:
    manager.release("database", owner_id)
```

### 2. Use Owner IDs

Always provide meaningful owner IDs:

```python
# Good - traceable owner
owner_id = f"state_{state_name}_{execution_id}"
resource = manager.acquire("database", owner_id)

# Bad - generic owner
resource = manager.acquire("database", "state")
```

### 3. Monitor Resource Health

Regularly check resource health:

```python
# Periodic health check
import threading
import time

def monitor_resources():
    while True:
        health = manager.health_check()
        for name, status in health.items():
            if status != ResourceHealth.HEALTHY:
                logger.warning(f"Resource {name} is {status}")
        time.sleep(60)

monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()
```

### 4. Configure Appropriate Pools

Size pools based on workload:

```python
# High-throughput configuration
pool_config = PoolConfig(
    min_size=5,
    max_size=20,
    acquire_timeout=10
)

# Low-throughput configuration
pool_config = PoolConfig(
    min_size=1,
    max_size=5,
    max_idle_time=600  # Longer idle time
)
```

### 5. Handle Resource Errors

```python
from dataknobs_fsm.functions.base import ResourceError

try:
    resource = manager.acquire("database", owner_id, timeout=30)
except ResourceError as e:
    logger.error(f"Failed to acquire {e.resource_name}: {e}")
    # Use fallback or retry
```

## Common Pitfalls

### 1. Resource Leaks

```python
# ❌ Bad - resource not released
resource = manager.acquire("database", owner_id)
result = process(resource)
return result  # Resource never released!

# ✅ Good - proper cleanup
with manager.resource_context("database", owner_id) as resource:
    result = process(resource)
    return result
```

### 2. Not Tracking Owners

```python
# ❌ Bad - no owner tracking
db = db_resource.acquire()  # Direct acquisition

# ✅ Good - use manager with owner
db = manager.acquire("database", "state_123")
```

### 3. Ignoring Health Checks

```python
# ❌ Bad - no health validation
resource = manager.acquire("api", owner_id)
result = resource.call()  # Might be unhealthy!

# ✅ Good - validate health
if manager.validate_resource("api"):
    resource = manager.acquire("api", owner_id)
    result = resource.call()
else:
    # Handle unhealthy resource
    pass
```

## Troubleshooting

### Debug Resource Status

```python
# Get detailed resource information
all_resources = manager.get_all_resources()
for name, status in all_resources.items():
    print(f"\nResource: {name}")
    print(f"  Has provider: {status['provider_exists']}")
    print(f"  Has pool: {status['has_pool']}")
    print(f"  Active count: {status['active_count']}")
    print(f"  Owners: {status['owners']}")
```

### Common Issues

1. **Resource Timeout**: Increase acquire_timeout in pool config
2. **Pool Exhausted**: Increase max_size or reduce hold time
3. **Health Check Failures**: Check resource connectivity and credentials
4. **Memory Leaks**: Ensure all resources are released properly

## Conclusion

The DataKnobs FSM resource management system provides:

- **Centralized Management**: Single point for all resource operations
- **Owner Tracking**: Know who owns what resources
- **Health Monitoring**: Track resource health and metrics
- **Pooling Support**: Efficient resource reuse
- **Cleanup Guarantees**: Automatic cleanup on errors

Use the ResourceManager for all external dependencies to ensure proper lifecycle management.

## Next Steps

- [Streaming Guide](streaming.md) - Process large datasets with resources
- [Data Modes Guide](data-modes.md) - Understand data handling modes
- [API Reference](../api/index.md) - Resource API documentation
- [Pattern Catalog](../patterns/index.md) - Resource usage patterns
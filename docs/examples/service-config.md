# Service Configuration Example

This example demonstrates how to configure microservices, APIs, and background workers using the DataKnobs Config package with dependency injection and cross-references.

## Basic Service Configuration

### Simple API Service

```yaml
# config/services.yaml
services:
  - name: api
    host: 0.0.0.0
    port: 8000
    workers: 4
    debug: false
    log_level: INFO
```

```python
from dataknobs_config import Config

# Load configuration
config = Config.from_file("config/services.yaml")

# Get service configuration
api_config = config.get("services", "api")
print(f"Starting API on {api_config['host']}:{api_config['port']}")
```

## Complete Microservices Architecture

### Full Service Configuration

```yaml
# config/services.yaml
databases:
  - name: primary
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    database: ${DB_NAME:myapp}
    username: ${DB_USER:postgres}
    password: ${DB_PASSWORD}
    
caches:
  - name: redis
    host: ${REDIS_HOST:localhost}
    port: ${REDIS_PORT:6379}
    
queues:
  - name: celery
    broker: redis://${REDIS_HOST:localhost}:${REDIS_PORT:6379}/0
    backend: redis://${REDIS_HOST:localhost}:${REDIS_PORT:6379}/1

services:
  - name: api
    type: fastapi
    host: ${API_HOST:0.0.0.0}
    port: ${API_PORT:8000}
    workers: ${API_WORKERS:4}
    database: "xref:databases[primary]"
    cache: "xref:caches[redis]"
    settings:
      debug: ${DEBUG:false}
      log_level: ${LOG_LEVEL:INFO}
      cors:
        enabled: true
        origins:
          - http://localhost:3000
          - https://app.example.com
      rate_limiting:
        enabled: true
        requests_per_minute: 60
      authentication:
        jwt_secret: ${JWT_SECRET}
        jwt_algorithm: HS256
        jwt_expiry: 3600
    
  - name: worker
    type: celery
    queue: "xref:queues[celery]"
    database: "xref:databases[primary]"
    cache: "xref:caches[redis]"
    concurrency: ${WORKER_CONCURRENCY:10}
    tasks:
      - email.send
      - reports.generate
      - data.process
    settings:
      task_time_limit: 3600
      task_soft_time_limit: 3000
      result_expires: 86400
      
  - name: websocket
    type: socketio
    host: ${WS_HOST:0.0.0.0}
    port: ${WS_PORT:8001}
    cache: "xref:caches[redis]"
    cors_allowed_origins: "*"
    async_mode: aiohttp
    
  - name: scheduler
    type: apscheduler
    database: "xref:databases[primary]"
    timezone: UTC
    jobs:
      - name: cleanup
        function: tasks.cleanup.run
        trigger: cron
        hour: 2
        minute: 0
      - name: backup
        function: tasks.backup.run
        trigger: interval
        hours: 6
      - name: metrics
        function: tasks.metrics.collect
        trigger: interval
        minutes: 5
```

## Service Factory Implementation

### Base Service Factory

```python
# factories/service.py
from dataknobs_config import FactoryBase, Config
from abc import ABC, abstractmethod
from typing import Dict, Any

class ServiceBase(ABC):
    """Base class for all services."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.dependencies = {}
    
    @abstractmethod
    async def start(self):
        """Start the service."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the service."""
        pass
    
    def inject_dependency(self, name: str, dependency: Any):
        """Inject a dependency into the service."""
        self.dependencies[name] = dependency

class ServiceFactory(FactoryBase):
    """Factory for creating services with dependency injection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.service_registry = {}
    
    def create(self, **service_config) -> ServiceBase:
        """Create a service based on type."""
        service_type = service_config.get("type", "generic")
        name = service_config.get("name", "unnamed")
        
        # Create service based on type
        if service_type == "fastapi":
            service = self._create_fastapi_service(name, service_config)
        elif service_type == "celery":
            service = self._create_celery_worker(name, service_config)
        elif service_type == "socketio":
            service = self._create_websocket_service(name, service_config)
        elif service_type == "apscheduler":
            service = self._create_scheduler_service(name, service_config)
        else:
            raise ValueError(f"Unknown service type: {service_type}")
        
        # Inject dependencies
        self._inject_dependencies(service, service_config)
        
        # Register service
        self.service_registry[name] = service
        
        return service
    
    def _inject_dependencies(self, service: ServiceBase, config: Dict[str, Any]):
        """Inject dependencies into service."""
        # Inject database if specified
        if "database" in config:
            db_ref = config["database"]
            if isinstance(db_ref, str) and db_ref.startswith("xref:"):
                db = self.config.construct_from_ref(db_ref)
                service.inject_dependency("database", db)
        
        # Inject cache if specified
        if "cache" in config:
            cache_ref = config["cache"]
            if isinstance(cache_ref, str) and cache_ref.startswith("xref:"):
                cache = self.config.construct_from_ref(cache_ref)
                service.inject_dependency("cache", cache)
        
        # Inject queue if specified
        if "queue" in config:
            queue_ref = config["queue"]
            if isinstance(queue_ref, str) and queue_ref.startswith("xref:"):
                queue = self.config.construct_from_ref(queue_ref)
                service.inject_dependency("queue", queue)
```

### FastAPI Service Implementation

```python
# services/fastapi_service.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from factories.service import ServiceBase

class FastAPIService(ServiceBase):
    """FastAPI service implementation."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.app = FastAPI(title=name)
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware."""
        settings = self.config.get("settings", {})
        
        # CORS
        if settings.get("cors", {}).get("enabled", False):
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=settings["cors"].get("origins", ["*"]),
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Rate limiting
        if settings.get("rate_limiting", {}).get("enabled", False):
            limiter = Limiter(
                key_func=get_remote_address,
                default_limits=[
                    f"{settings['rate_limiting']['requests_per_minute']}/minute"
                ]
            )
            self.app.state.limiter = limiter
            self.app.add_exception_handler(
                RateLimitExceeded,
                _rate_limit_exceeded_handler
            )
    
    def _setup_routes(self):
        """Configure API routes."""
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": self.name}
        
        @self.app.get("/config")
        async def get_config():
            # Return non-sensitive configuration
            return {
                "service": self.name,
                "host": self.config["host"],
                "port": self.config["port"],
                "workers": self.config.get("workers", 1)
            }
    
    async def start(self):
        """Start the FastAPI service."""
        config = uvicorn.Config(
            app=self.app,
            host=self.config["host"],
            port=self.config["port"],
            workers=self.config.get("workers", 1),
            log_level=self.config.get("settings", {}).get("log_level", "info").lower()
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop the FastAPI service."""
        # Cleanup logic
        pass
```

### Celery Worker Implementation

```python
# services/celery_worker.py
from celery import Celery
from factories.service import ServiceBase
import logging

logger = logging.getLogger(__name__)

class CeleryWorkerService(ServiceBase):
    """Celery worker service implementation."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.celery_app = None
        self._setup_celery()
    
    def _setup_celery(self):
        """Configure Celery application."""
        queue = self.dependencies.get("queue")
        if not queue:
            raise ValueError("Queue dependency required for Celery worker")
        
        self.celery_app = Celery(
            self.name,
            broker=queue.get("broker"),
            backend=queue.get("backend")
        )
        
        # Configure Celery settings
        settings = self.config.get("settings", {})
        self.celery_app.conf.update(
            task_time_limit=settings.get("task_time_limit", 3600),
            task_soft_time_limit=settings.get("task_soft_time_limit", 3000),
            result_expires=settings.get("result_expires", 86400),
            worker_concurrency=self.config.get("concurrency", 10),
            worker_prefetch_multiplier=1,
            task_acks_late=True
        )
        
        # Register tasks
        self._register_tasks()
    
    def _register_tasks(self):
        """Register Celery tasks."""
        tasks = self.config.get("tasks", [])
        
        for task_path in tasks:
            # Import and register task
            module_path, task_name = task_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[task_name])
            task = getattr(module, task_name)
            self.celery_app.task(task)
            logger.info(f"Registered task: {task_path}")
    
    async def start(self):
        """Start the Celery worker."""
        logger.info(f"Starting Celery worker: {self.name}")
        
        # Start worker
        worker = self.celery_app.Worker(
            loglevel=self.config.get("settings", {}).get("log_level", "INFO"),
            concurrency=self.config.get("concurrency", 10)
        )
        worker.start()
    
    async def stop(self):
        """Stop the Celery worker."""
        logger.info(f"Stopping Celery worker: {self.name}")
        if self.celery_app:
            self.celery_app.control.shutdown()
```

## Service Manager Pattern

### Service Orchestrator

```python
# managers/service_manager.py
from dataknobs_config import Config
from typing import Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class ServiceManager:
    """Manages the lifecycle of multiple services."""
    
    def __init__(self, config: Config):
        self.config = config
        self.services: Dict[str, ServiceBase] = {}
        self._running = False
        self._register_factories()
    
    def _register_factories(self):
        """Register service factories."""
        from factories.service import ServiceFactory
        from factories.database import PostgreSQLFactory
        from factories.cache import RedisFactory
        
        # Register dependency factories
        self.config.register_factory("postgresql", PostgreSQLFactory())
        self.config.register_factory("redis", RedisFactory())
        
        # Register service factory
        self.config.register_factory("service", ServiceFactory(self.config))
    
    async def start_service(self, name: str):
        """Start a specific service."""
        if name in self.services:
            logger.warning(f"Service {name} is already running")
            return
        
        # Get service configuration
        service_config = self.config.get("services", name)
        if not service_config:
            raise ValueError(f"Service configuration not found: {name}")
        
        # Set factory
        service_config["factory"] = "service"
        
        # Create and start service
        service = self.config.construct("services", name)
        self.services[name] = service
        
        logger.info(f"Starting service: {name}")
        await service.start()
    
    async def stop_service(self, name: str):
        """Stop a specific service."""
        if name not in self.services:
            logger.warning(f"Service {name} is not running")
            return
        
        service = self.services[name]
        logger.info(f"Stopping service: {name}")
        await service.stop()
        
        del self.services[name]
    
    async def start_all(self, service_names: Optional[List[str]] = None):
        """Start all configured services."""
        self._running = True
        
        # Get services to start
        if service_names:
            services = [
                self.config.get("services", name) 
                for name in service_names
            ]
        else:
            services = self.config.get("services", default=[])
        
        # Start services concurrently
        tasks = []
        for service_config in services:
            name = service_config["name"]
            tasks.append(self.start_service(name))
        
        await asyncio.gather(*tasks)
        logger.info(f"Started {len(tasks)} services")
    
    async def stop_all(self):
        """Stop all running services."""
        self._running = False
        
        # Stop services concurrently
        tasks = []
        for name in list(self.services.keys()):
            tasks.append(self.stop_service(name))
        
        await asyncio.gather(*tasks)
        logger.info(f"Stopped {len(tasks)} services")
    
    async def restart_service(self, name: str):
        """Restart a specific service."""
        await self.stop_service(name)
        await self.start_service(name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            "running": self._running,
            "services": {
                name: {
                    "status": "running",
                    "type": service.config.get("type"),
                    "uptime": getattr(service, "uptime", None)
                }
                for name, service in self.services.items()
            }
        }
```

## Usage Examples

### Starting Services

```python
# main.py
import asyncio
from dataknobs_config import Config
from managers.service_manager import ServiceManager
import signal
import sys

async def main():
    # Load configuration
    config = Config.from_file("config/services.yaml", apply_env_overrides=True)
    
    # Create service manager
    manager = ServiceManager(config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\nShutting down services...")
        asyncio.create_task(manager.stop_all())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all services
        await manager.start_all()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
            # Print status
            status = manager.get_status()
            print(f"Services running: {len(status['services'])}")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await manager.stop_all()

if __name__ == "__main__":
    asyncio.run(main())
```

### Service Health Monitoring

```python
# monitoring/health.py
from typing import Dict, Any
import aiohttp
import asyncio

class ServiceHealthMonitor:
    """Monitor health of services."""
    
    def __init__(self, manager: ServiceManager):
        self.manager = manager
        self.health_endpoints = {}
    
    async def check_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        service = self.manager.services.get(service_name)
        if not service:
            return {"status": "not_running"}
        
        # Check based on service type
        service_type = service.config.get("type")
        
        if service_type == "fastapi":
            return await self._check_http_health(service)
        elif service_type == "celery":
            return await self._check_celery_health(service)
        else:
            return {"status": "unknown"}
    
    async def _check_http_health(self, service) -> Dict[str, Any]:
        """Check HTTP service health."""
        url = f"http://{service.config['host']}:{service.config['port']}/health"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"status": "healthy", **data}
                    else:
                        return {"status": "unhealthy", "code": response.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_celery_health(self, service) -> Dict[str, Any]:
        """Check Celery worker health."""
        if service.celery_app:
            stats = service.celery_app.control.inspect().stats()
            if stats:
                return {"status": "healthy", "workers": len(stats)}
            else:
                return {"status": "unhealthy", "workers": 0}
        return {"status": "not_initialized"}
    
    async def monitor_all(self, interval: int = 30):
        """Monitor all services continuously."""
        while True:
            results = {}
            
            for service_name in self.manager.services.keys():
                health = await self.check_health(service_name)
                results[service_name] = health
                
                if health["status"] != "healthy":
                    logger.warning(f"Service {service_name} is {health['status']}")
            
            # Log summary
            healthy = sum(1 for r in results.values() if r["status"] == "healthy")
            total = len(results)
            logger.info(f"Health check: {healthy}/{total} services healthy")
            
            await asyncio.sleep(interval)
```

## Environment-Specific Configuration

### Development Services

```yaml
# config/services.dev.yaml
services:
  - name: api
    type: fastapi
    host: localhost
    port: 8000
    workers: 1
    settings:
      debug: true
      log_level: DEBUG
      cors:
        enabled: true
        origins: ["*"]
      rate_limiting:
        enabled: false
```

### Production Services

```yaml
# config/services.prod.yaml
services:
  - name: api
    type: fastapi
    host: 0.0.0.0
    port: ${PORT:8000}
    workers: ${WEB_CONCURRENCY:4}
    settings:
      debug: false
      log_level: WARNING
      cors:
        enabled: true
        origins: ${CORS_ORIGINS}
      rate_limiting:
        enabled: true
        requests_per_minute: 100
      
  - name: worker
    type: celery
    concurrency: ${WORKER_CONCURRENCY:20}
    settings:
      task_time_limit: 7200
      result_expires: 172800
```

## Service Discovery

### Consul Integration

```python
# discovery/consul.py
import consul

class ConsulServiceDiscovery:
    """Service discovery using Consul."""
    
    def __init__(self, consul_host="localhost", consul_port=8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
    
    def register_service(self, service: ServiceBase):
        """Register service with Consul."""
        self.consul.agent.service.register(
            name=service.name,
            service_id=f"{service.name}-{id(service)}",
            address=service.config.get("host", "localhost"),
            port=service.config.get("port"),
            check=consul.Check.http(
                f"http://{service.config['host']}:{service.config['port']}/health",
                interval="30s"
            )
        )
    
    def discover_service(self, name: str):
        """Discover service by name."""
        _, services = self.consul.health.service(name, passing=True)
        return services
```

## Best Practices

1. **Use dependency injection** for database and cache connections
2. **Implement health checks** for all services
3. **Use environment variables** for sensitive configuration
4. **Set up proper logging** with appropriate levels
5. **Implement graceful shutdown** handlers
6. **Use service discovery** in distributed systems
7. **Monitor service metrics** and performance
8. **Implement circuit breakers** for resilience
9. **Use async/await** for I/O-bound services
10. **Document service dependencies** clearly
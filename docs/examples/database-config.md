# Database Configuration Example

This example demonstrates how to configure database connections using the DataKnobs Config package, including connection pooling, environment-specific settings, and factory patterns.

## Basic Database Configuration

### Simple Configuration

```yaml
# config/database.yaml
databases:
  - name: primary
    host: localhost
    port: 5432
    database: myapp
    username: postgres
    password: secret
```

```python
from dataknobs_config import Config

# Load configuration
config = Config.from_file("config/database.yaml")

# Get database configuration
db_config = config.get("databases", "primary")
print(f"Connecting to {db_config['host']}:{db_config['port']}")
```

## Advanced Database Configuration

### Multi-Database Setup

```yaml
# config/databases.yaml
databases:
  - name: primary
    host: ${DB_PRIMARY_HOST:localhost}
    port: ${DB_PRIMARY_PORT:5432}
    database: ${DB_PRIMARY_NAME:myapp}
    username: ${DB_PRIMARY_USER:postgres}
    password: ${DB_PRIMARY_PASS}
    pool:
      min_size: 5
      max_size: 20
      timeout: 30
      retry_attempts: 3
    options:
      sslmode: prefer
      connect_timeout: 10
      application_name: myapp
    
  - name: analytics
    backend: duckdb
    path: ${DUCKDB_PATH:./data/analytics.duckdb}
    table: ${DUCKDB_TABLE:records}
    timeout: ${DUCKDB_TIMEOUT:10.0}
    read_only: ${DUCKDB_READONLY:false}

  - name: warehouse
    host: ${DB_WAREHOUSE_HOST:localhost}
    port: ${DB_WAREHOUSE_PORT:5432}
    database: ${DB_WAREHOUSE_NAME:warehouse}
    username: ${DB_WAREHOUSE_USER:readonly}
    password: ${DB_WAREHOUSE_PASS}
    pool:
      min_size: 2
      max_size: 10
      timeout: 60
    options:
      sslmode: require
      statement_timeout: 300000  # 5 minutes

  - name: cache
    type: redis
    host: ${REDIS_HOST:localhost}
    port: ${REDIS_PORT:6379}
    database: ${REDIS_DB:0}
    password: ${REDIS_PASS}
    pool:
      max_connections: 50
      connection_class: redis.BlockingConnectionPool
```

## Database Factory Implementation

### PostgreSQL Factory

```python
# factories/database.py
from dataknobs_config import FactoryBase
import asyncpg
import psycopg2
from psycopg2 import pool
from typing import Union, Optional

class PostgreSQLFactory(FactoryBase):
    """Factory for PostgreSQL connections with sync/async support."""
    
    def __init__(self, async_mode: bool = False):
        self.async_mode = async_mode
        self._pools = {}
    
    def create(self, **config) -> Union[pool.ThreadedConnectionPool, asyncpg.Pool]:
        """Create PostgreSQL connection pool."""
        if self.async_mode:
            return self._create_async(**config)
        else:
            return self._create_sync(**config)
    
    def _create_sync(self, **config) -> pool.ThreadedConnectionPool:
        """Create synchronous connection pool."""
        pool_config = config.pop("pool", {})
        options = config.pop("options", {})
        
        # Build connection string
        conn_params = {
            "host": config.get("host", "localhost"),
            "port": config.get("port", 5432),
            "database": config.get("database", "postgres"),
            "user": config.get("username", "postgres"),
            "password": config.get("password"),
            **options
        }
        
        # Create connection pool
        return pool.ThreadedConnectionPool(
            minconn=pool_config.get("min_size", 5),
            maxconn=pool_config.get("max_size", 20),
            **conn_params
        )
    
    async def _create_async(self, **config) -> asyncpg.Pool:
        """Create asynchronous connection pool."""
        pool_config = config.pop("pool", {})
        options = config.pop("options", {})
        
        # Create async pool
        return await asyncpg.create_pool(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database", "postgres"),
            user=config.get("username", "postgres"),
            password=config.get("password"),
            min_size=pool_config.get("min_size", 5),
            max_size=pool_config.get("max_size", 20),
            timeout=pool_config.get("timeout", 30),
            **options
        )
```

### Redis Factory

```python
# factories/cache.py
import redis
from dataknobs_config import FactoryBase

class RedisFactory(FactoryBase):
    """Factory for Redis connections."""
    
    def create(self, **config) -> redis.Redis:
        """Create Redis connection."""
        pool_config = config.pop("pool", {})
        
        # Create connection pool
        connection_pool = redis.ConnectionPool(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("database", 0),
            password=config.get("password"),
            max_connections=pool_config.get("max_connections", 50),
            decode_responses=config.get("decode_responses", True)
        )
        
        # Return Redis client with pool
        return redis.Redis(connection_pool=connection_pool)
```

## Database Manager Pattern

### Connection Manager

```python
# managers/database.py
from dataknobs_config import Config
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and pools."""
    
    def __init__(self, config: Config):
        self.config = config
        self._connections = {}
        self._register_factories()
    
    def _register_factories(self):
        """Register database factories."""
        from factories.database import PostgreSQLFactory
        from factories.cache import RedisFactory
        
        self.config.register_factory("postgresql", PostgreSQLFactory())
        self.config.register_factory("postgresql_async", PostgreSQLFactory(async_mode=True))
        self.config.register_factory("redis", RedisFactory())
    
    def get_connection(self, name: str = "primary"):
        """Get database connection by name."""
        if name not in self._connections:
            self._connections[name] = self._create_connection(name)
        return self._connections[name]
    
    def _create_connection(self, name: str):
        """Create new database connection."""
        db_config = self.config.get("databases", name)
        
        if not db_config:
            raise ValueError(f"Database configuration not found: {name}")
        
        # Determine factory based on type
        db_type = db_config.get("type", "postgresql")
        factory_name = db_type
        
        if db_config.get("async", False):
            factory_name = f"{db_type}_async"
        
        # Set factory in config
        db_config["factory"] = factory_name
        
        # Construct and return connection
        logger.info(f"Creating connection for database: {name}")
        return self.config.construct("databases", name)
    
    def close_all(self):
        """Close all database connections."""
        for name, conn in self._connections.items():
            logger.info(f"Closing connection: {name}")
            if hasattr(conn, "close"):
                conn.close()
        self._connections.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
```

## Usage Examples

### Basic Usage

```python
# main.py
from dataknobs_config import Config
from managers.database import DatabaseManager

def main():
    # Load configuration
    config = Config.from_file("config/databases.yaml", apply_env_overrides=True)
    
    # Create database manager
    with DatabaseManager(config) as db_manager:
        # Get primary database connection
        primary_db = db_manager.get_connection("primary")
        
        # Execute query
        with primary_db.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users LIMIT 10")
                users = cursor.fetchall()
                print(f"Found {len(users)} users")
        
        # Get analytics database
        analytics_db = db_manager.get_connection("analytics")
        
        # Get cache connection
        cache = db_manager.get_connection("cache")
        cache.set("key", "value", ex=3600)

if __name__ == "__main__":
    main()
```

### Async Usage

```python
# async_main.py
import asyncio
from dataknobs_config import Config
from managers.database import DatabaseManager

async def main():
    # Load configuration
    config = Config.from_file("config/databases.yaml", apply_env_overrides=True)
    
    # Mark databases as async
    for db in config.get("databases"):
        if db["name"] in ["primary", "analytics"]:
            db["async"] = True
    
    # Create database manager
    db_manager = DatabaseManager(config)
    
    try:
        # Get async database connection
        primary_db = await db_manager.get_connection("primary")
        
        # Execute async query
        async with primary_db.acquire() as conn:
            result = await conn.fetch("SELECT * FROM users LIMIT 10")
            print(f"Found {len(result)} users")
        
        # Parallel queries
        analytics_db = await db_manager.get_connection("analytics")
        
        async with primary_db.acquire() as conn1, \
                   analytics_db.acquire() as conn2:
            
            results = await asyncio.gather(
                conn1.fetch("SELECT COUNT(*) FROM users"),
                conn2.fetch("SELECT COUNT(*) FROM events")
            )
            
            print(f"Users: {results[0][0]['count']}")
            print(f"Events: {results[1][0]['count']}")
    
    finally:
        # Close all connections
        for db in [primary_db, analytics_db]:
            if db:
                await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Environment-Specific Configuration

### Development Configuration

```yaml
# config/database.dev.yaml
databases:
  - name: primary
    host: localhost
    port: 5432
    database: myapp_dev
    username: developer
    password: devpass
    pool:
      min_size: 2
      max_size: 5
    options:
      log_statement: all
      log_duration: true
```

### Production Configuration

```yaml
# config/database.prod.yaml
databases:
  - name: primary
    host: ${DB_HOST}
    port: ${DB_PORT:5432}
    database: ${DB_NAME}
    username: ${DB_USER}
    password: ${DB_PASSWORD}
    pool:
      min_size: 20
      max_size: 100
      timeout: 30
      retry_attempts: 5
    options:
      sslmode: require
      sslcert: /etc/ssl/certs/client.crt
      sslkey: /etc/ssl/private/client.key
      connect_timeout: 10
      statement_timeout: 60000
      idle_in_transaction_session_timeout: 60000
```

### Loading Environment-Specific Config

```python
import os
from dataknobs_config import Config

def load_database_config():
    """Load database configuration based on environment."""
    env = os.getenv("APP_ENV", "development")
    
    # Load base configuration
    config = Config.from_file("config/database.yaml")
    
    # Merge environment-specific configuration
    env_config_file = f"config/database.{env}.yaml"
    if os.path.exists(env_config_file):
        config.merge_file(env_config_file)
    
    # Apply environment variable overrides
    config.apply_env_overrides()
    
    return config
```

## Connection Pooling Strategies

### Dynamic Pool Sizing

```python
class AdaptivePoolFactory(FactoryBase):
    """Factory with adaptive pool sizing based on load."""
    
    def __init__(self):
        self.metrics = {"connections": 0, "peak": 0}
    
    def create(self, **config):
        """Create pool with adaptive sizing."""
        pool_config = config.get("pool", {})
        
        # Calculate pool size based on metrics
        base_size = pool_config.get("min_size", 5)
        max_size = pool_config.get("max_size", 20)
        
        # Adaptive sizing logic
        if self.metrics["peak"] > base_size * 0.8:
            pool_config["min_size"] = min(base_size + 2, max_size)
        
        # Create pool with adjusted size
        return self._create_pool(config, pool_config)
    
    def _create_pool(self, config, pool_config):
        # Implementation specific to database type
        pass
```

### Health Check Integration

```python
class HealthCheckFactory(FactoryBase):
    """Factory with built-in health checking."""
    
    def create(self, **config):
        """Create connection with health check."""
        pool = super().create(**config)
        
        # Add health check wrapper
        return HealthCheckWrapper(pool, config)

class HealthCheckWrapper:
    """Wrapper that adds health checking to database pools."""
    
    def __init__(self, pool, config):
        self.pool = pool
        self.health_check_interval = config.get("health_check_interval", 30)
        self._start_health_check()
    
    def _start_health_check(self):
        """Start background health check."""
        import threading
        
        def check_health():
            while True:
                try:
                    # Perform health check
                    with self.pool.getconn() as conn:
                        conn.execute("SELECT 1")
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    # Trigger reconnection or alerting
                
                time.sleep(self.health_check_interval)
        
        thread = threading.Thread(target=check_health, daemon=True)
        thread.start()
```

## Best Practices

1. **Always use connection pooling** for production applications
2. **Set appropriate pool sizes** based on application load
3. **Use environment variables** for sensitive configuration
4. **Implement health checks** for critical databases
5. **Use read replicas** for analytics queries
6. **Set statement timeouts** to prevent long-running queries
7. **Enable SSL** for production databases
8. **Monitor connection pool metrics**
9. **Implement retry logic** for transient failures
10. **Use async connections** for I/O-bound applications

## Troubleshooting

### Common Issues

1. **Connection Pool Exhausted**
   ```python
   # Increase pool size
   pool:
     min_size: 10
     max_size: 50
   ```

2. **Connection Timeouts**
   ```python
   # Increase timeout values
   options:
     connect_timeout: 30
     statement_timeout: 120000
   ```

3. **SSL Certificate Issues**
   ```python
   # Verify SSL configuration
   options:
     sslmode: require
     sslrootcert: /path/to/ca.crt
   ```

### Debug Logging

```python
import logging

# Enable debug logging for database operations
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dataknobs_config")
logger.setLevel(logging.DEBUG)
```
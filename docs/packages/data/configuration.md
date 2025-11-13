# Configuration

The dataknobs-data package fully integrates with the dataknobs-config system, providing powerful configuration capabilities including environment variable substitution, factory registration, and cross-references.

## Configuration Basics

### Simple Configuration
```yaml
# config.yaml
database:
  backend: postgres
  host: localhost
  port: 5432
  database: myapp
  user: dbuser
  password: dbpass
```

```python
from dataknobs_config import Config
from dataknobs_data import DatabaseFactory

config = Config("config.yaml")
db_config = config.get("database")

factory = DatabaseFactory()
db = factory.create(**db_config)
```

### With Environment Variables
```yaml
# config.yaml with environment variable substitution
database:
  backend: ${DB_BACKEND:postgres}
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  database: ${DB_NAME:myapp}
  user: ${DB_USER:dbuser}
  password: ${DB_PASSWORD}  # Required, no default
```

Environment variables are automatically substituted:
- `${VAR}` - Use environment variable, error if not set
- `${VAR:default}` - Use environment variable with default value
- `${VAR:}` - Use environment variable, empty string if not set

## Backend-Specific Configuration

### Memory Backend
```yaml
memory_db:
  backend: memory
  # No additional configuration needed
```

### File Backend
```yaml
file_db:
  backend: file
  path: ${DATA_PATH:./data/records.json}
  format: ${FILE_FORMAT:json}  # json, csv, or parquet
```

### SQLite Backend
```yaml
sqlite_db:
  backend: sqlite
  path: ${SQLITE_PATH:./data/app.db}  # or ":memory:" for in-memory
  table: ${SQLITE_TABLE:records}  # Optional
  timeout: ${SQLITE_TIMEOUT:5.0}  # Connection timeout in seconds
  journal_mode: ${SQLITE_JOURNAL:WAL}  # WAL, DELETE, TRUNCATE, PERSIST, MEMORY, OFF
  synchronous: ${SQLITE_SYNC:NORMAL}  # FULL, NORMAL, OFF
```

### DuckDB Backend
```yaml
duckdb_db:
  backend: duckdb
  path: ${DUCKDB_PATH:./data/analytics.duckdb}  # or ":memory:" for in-memory
  table: ${DUCKDB_TABLE:records}  # Optional
  timeout: ${DUCKDB_TIMEOUT:5.0}  # Connection timeout in seconds
  read_only: ${DUCKDB_READONLY:false}  # Open in read-only mode
```

### PostgreSQL Backend
```yaml
postgres_db:
  backend: postgres
  host: ${POSTGRES_HOST:localhost}
  port: ${POSTGRES_PORT:5432}
  database: ${POSTGRES_DB:myapp}
  user: ${POSTGRES_USER:postgres}
  password: ${POSTGRES_PASSWORD}
  table: ${POSTGRES_TABLE:records}  # Optional
  pool_size: ${DB_POOL_SIZE:5}  # Connection pool size
```

### Elasticsearch Backend
```yaml
elasticsearch_db:
  backend: elasticsearch
  hosts:
    - ${ES_HOST1:localhost:9200}
    - ${ES_HOST2:}  # Optional second host
  index: ${ES_INDEX:records}
  username: ${ES_USERNAME:}  # Optional
  password: ${ES_PASSWORD:}  # Optional
  use_ssl: ${ES_USE_SSL:false}
  verify_certs: ${ES_VERIFY_CERTS:true}
```

### S3 Backend
```yaml
s3_db:
  backend: s3
  bucket: ${S3_BUCKET}
  prefix: ${S3_PREFIX:data/}
  region: ${AWS_REGION:us-east-1}
  endpoint_url: ${S3_ENDPOINT:}  # For LocalStack/MinIO
  access_key_id: ${AWS_ACCESS_KEY_ID:}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY:}
  max_workers: ${S3_MAX_WORKERS:10}
  multipart_threshold: ${S3_MULTIPART_THRESHOLD:5242880}  # 5MB
```

## Factory Registration

Register the database factory for cleaner configuration:

```python
from dataknobs_config import Config
from dataknobs_data import database_factory

# Create config and register factory
config = Config()
config.register_factory("database", database_factory)

# Now you can use factory references in config
config.load({
    "databases": [
        {
            "name": "primary",
            "factory": "database",  # Reference to registered factory
            "backend": "postgres",
            "host": "localhost"
        },
        {
            "name": "cache",
            "factory": "database",
            "backend": "memory"
        }
    ]
})

# Get instances
primary_db = config.get_instance("databases", "primary")
cache_db = config.get_instance("databases", "cache")
```

## Multiple Environments

### Environment-Specific Files
```yaml
# config.base.yaml
app:
  name: MyApp
  version: 1.0

# config.dev.yaml
database:
  backend: file
  path: ./dev_data.json

# config.prod.yaml
database:
  backend: postgres
  host: ${DB_HOST}
  database: ${DB_NAME}
  user: ${DB_USER}
  password: ${DB_PASSWORD}
```

```python
import os
from dataknobs_config import Config

# Load based on environment
env = os.environ.get("APP_ENV", "dev")
config = Config(f"config.{env}.yaml")

# Merge with base config
base_config = Config("config.base.yaml")
config.merge(base_config.data)
```

### Using Config Profiles
```yaml
# config.yaml with profiles
profiles:
  development:
    database:
      backend: memory
  
  testing:
    database:
      backend: file
      path: /tmp/test.json
  
  production:
    database:
      backend: postgres
      host: ${DB_HOST}
      database: ${DB_NAME}
      user: ${DB_USER}
      password: ${DB_PASSWORD}

# Select profile with environment variable
active_profile: ${APP_PROFILE:development}
```

## Cross-References

Reference other configuration values:

```yaml
# config.yaml
defaults:
  region: us-east-1
  bucket_prefix: myapp

databases:
  archive:
    backend: s3
    bucket: ${defaults.bucket_prefix}-archive
    region: ${defaults.region}
    prefix: ${APP_ENV:dev}/
  
  backup:
    backend: s3
    bucket: ${defaults.bucket_prefix}-backup
    region: ${defaults.region}
    prefix: backups/
```

## Configuration Validation

```python
from dataknobs_config import Config
from dataknobs_data import DatabaseFactory

def validate_database_config(config: dict) -> bool:
    """Validate database configuration."""
    factory = DatabaseFactory()
    
    # Check backend is valid
    backend = config.get("backend")
    if backend not in factory.get_available_backends():
        raise ValueError(f"Invalid backend: {backend}")
    
    # Check required parameters
    info = factory.get_backend_info(backend)
    required = info.get("required_params", [])
    
    for param in required:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    return True

# Usage
config = Config("config.yaml")
db_config = config.get("database")
validate_database_config(db_config)
```

## Dynamic Configuration

### Runtime Configuration Updates
```python
from dataknobs_config import Config
from dataknobs_data import DatabaseFactory

class DynamicDatabaseManager:
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self.factory = DatabaseFactory()
        self.databases = {}
        self._load_databases()
    
    def _load_databases(self):
        """Load all configured databases."""
        db_configs = self.config.get("databases", [])
        for db_config in db_configs:
            name = db_config.pop("name")
            self.databases[name] = self.factory.create(**db_config)
    
    def reload_config(self):
        """Reload configuration and recreate databases."""
        self.config.reload()
        self.databases.clear()
        self._load_databases()
    
    def add_database(self, name: str, config: dict):
        """Add a new database at runtime."""
        self.databases[name] = self.factory.create(**config)
        
        # Optionally persist to config
        current = self.config.get("databases", [])
        current.append({"name": name, **config})
        self.config.set("databases", current)
        self.config.save()
```

### Feature Flags
```yaml
# config.yaml
features:
  use_cache: ${USE_CACHE:true}
  archive_enabled: ${ARCHIVE_ENABLED:false}
  search_backend: ${SEARCH_BACKEND:elasticsearch}

databases:
  primary:
    backend: postgres
    host: localhost
  
  cache:
    backend: memory
    enabled: ${features.use_cache}
  
  search:
    backend: ${features.search_backend}
    hosts: ["localhost:9200"]
```

```python
config = Config("config.yaml")

# Conditionally create cache
if config.get("databases.cache.enabled", False):
    cache_db = config.get_instance("databases", "cache")
else:
    cache_db = None
```

## Testing Configuration

```python
import pytest
import tempfile
from dataknobs_config import Config
from dataknobs_data import DatabaseFactory

@pytest.fixture
def test_config():
    """Create test configuration."""
    config = Config()
    config.load({
        "databases": {
            "test": {
                "backend": "memory"
            }
        }
    })
    return config

@pytest.fixture
def test_database(test_config):
    """Create test database from config."""
    factory = DatabaseFactory()
    db_config = test_config.get("databases.test")
    return factory.create(**db_config)

def test_with_config(test_database):
    """Test using configured database."""
    record = Record({"test": "data"})
    record_id = test_database.create(record)
    assert test_database.read(record_id) is not None
```

## Configuration Best Practices

### 1. Use Environment Variables for Secrets
```yaml
# Never hardcode secrets
database:
  password: ${DB_PASSWORD}  # Good
  # password: mysecretpass  # Bad!
```

### 2. Provide Sensible Defaults
```yaml
database:
  host: ${DB_HOST:localhost}  # Defaults to localhost
  port: ${DB_PORT:5432}  # Defaults to PostgreSQL port
  pool_size: ${DB_POOL_SIZE:10}  # Default pool size
```

### 3. Group Related Configuration
```yaml
# Good: Grouped by purpose
databases:
  primary:
    backend: postgres
    # ...
  cache:
    backend: memory
    # ...

# Bad: Flat structure
primary_backend: postgres
primary_host: localhost
cache_backend: memory
```

### 4. Document Configuration
```yaml
# Database configuration
database:
  # Backend type: memory, file, postgres, elasticsearch, s3
  backend: ${DB_BACKEND:postgres}
  
  # PostgreSQL connection settings
  host: ${DB_HOST:localhost}  # Database host
  port: ${DB_PORT:5432}  # Database port
  database: ${DB_NAME:myapp}  # Database name
```

### 5. Validate Early
```python
def validate_config():
    """Validate configuration on startup."""
    config = Config("config.yaml")
    
    # Check required environment variables
    required_vars = ["DB_PASSWORD", "S3_BUCKET"]
    missing = [var for var in required_vars 
               if not os.environ.get(var)]
    
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")
    
    # Test database connections
    factory = DatabaseFactory()
    for name, db_config in config.get("databases", {}).items():
        try:
            db = factory.create(**db_config)
            db.count()  # Test connection
        except Exception as e:
            raise ValueError(f"Failed to connect to {name}: {e}")

# Call during application startup
validate_config()
```

## Example: Complete Application Configuration

```yaml
# app_config.yaml
app:
  name: DataKnobs Example
  version: ${APP_VERSION:1.0.0}
  environment: ${APP_ENV:development}

defaults:
  aws_region: ${AWS_REGION:us-east-1}
  data_prefix: ${APP_ENV}/data

databases:
  - name: primary
    factory: database
    backend: ${PRIMARY_DB:postgres}
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    database: ${DB_NAME:myapp}
    user: ${DB_USER:dbuser}
    password: ${DB_PASSWORD}
    pool_size: ${DB_POOL_SIZE:20}
  
  - name: cache
    factory: database
    backend: memory
    max_size: ${CACHE_MAX_SIZE:1000}
  
  - name: search
    factory: database
    backend: elasticsearch
    hosts:
      - ${ES_HOST:localhost:9200}
    index: ${ES_INDEX:${APP_ENV}_records}
    username: ${ES_USERNAME:}
    password: ${ES_PASSWORD:}
  
  - name: archive
    factory: database
    backend: s3
    bucket: ${S3_BUCKET}
    prefix: ${defaults.data_prefix}/archive/
    region: ${defaults.aws_region}
    max_workers: ${S3_WORKERS:10}

logging:
  level: ${LOG_LEVEL:INFO}
  file: ${LOG_FILE:./logs/app.log}

features:
  cache_enabled: ${CACHE_ENABLED:true}
  search_enabled: ${SEARCH_ENABLED:true}
  archive_old_data: ${ARCHIVE_ENABLED:false}
  archive_days: ${ARCHIVE_DAYS:365}
```
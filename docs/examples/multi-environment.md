# Multi-Environment Configuration

This example demonstrates how to manage configurations across multiple environments (development, staging, production) using the DataKnobs Config package.

## Environment Strategy

### Directory Structure

```
config/
├── base.yaml              # Base configuration (shared)
├── development.yaml       # Development overrides
├── staging.yaml          # Staging overrides
├── production.yaml       # Production overrides
├── local.yaml            # Local developer overrides (gitignored)
└── secrets/              # Sensitive configurations
    ├── development.yaml
    ├── staging.yaml
    └── production.yaml
```

## Base Configuration

### Shared Settings

```yaml
# config/base.yaml
application:
  name: MyApp
  version: ${APP_VERSION:1.0.0}
  timezone: UTC
  
databases:
  - name: primary
    type: postgresql
    database: myapp
    options:
      connect_timeout: 10
      statement_timeout: 30000
      
caches:
  - name: redis
    type: redis
    db: 0
    decode_responses: true
    
services:
  - name: api
    type: fastapi
    workers: 1
    
  - name: worker
    type: celery
    tasks:
      - tasks.email
      - tasks.reports
      
monitoring:
  metrics:
    enabled: true
    interval: 60
  logging:
    format: json
    level: INFO
    
features:
  authentication:
    enabled: true
    session_timeout: 3600
  rate_limiting:
    enabled: true
    default_limit: 100
  caching:
    enabled: true
    ttl: 300
```

## Environment-Specific Configurations

### Development Environment

```yaml
# config/development.yaml
databases:
  - name: primary
    host: localhost
    port: 5432
    username: dev_user
    password: dev_password
    pool:
      min_size: 2
      max_size: 10
    options:
      log_statement: all
      
caches:
  - name: redis
    host: localhost
    port: 6379
    
services:
  - name: api
    host: localhost
    port: 8000
    settings:
      debug: true
      reload: true
      cors:
        enabled: true
        origins: ["*"]
        
monitoring:
  logging:
    level: DEBUG
    console: true
    
features:
  rate_limiting:
    enabled: false
  caching:
    enabled: false
```

### Staging Environment

```yaml
# config/staging.yaml
databases:
  - name: primary
    host: staging-db.internal
    port: 5432
    username: ${DB_USER}
    password: ${DB_PASSWORD}
    pool:
      min_size: 5
      max_size: 20
    options:
      sslmode: require
      
caches:
  - name: redis
    host: staging-redis.internal
    port: 6379
    password: ${REDIS_PASSWORD}
    
services:
  - name: api
    host: 0.0.0.0
    port: 8000
    workers: 2
    settings:
      debug: false
      cors:
        enabled: true
        origins:
          - https://staging.example.com
          
monitoring:
  logging:
    level: INFO
    outputs:
      - console
      - file: /var/log/myapp/staging.log
      
  alerts:
    enabled: true
    webhook: ${ALERT_WEBHOOK_URL}
```

### Production Environment

```yaml
# config/production.yaml
databases:
  - name: primary
    host: ${DB_HOST}
    port: ${DB_PORT:5432}
    username: ${DB_USER}
    password: ${DB_PASSWORD}
    database: ${DB_NAME}
    pool:
      min_size: 20
      max_size: 100
      timeout: 30
    options:
      sslmode: require
      sslcert: /etc/ssl/certs/client.crt
      sslkey: /etc/ssl/private/client.key
      
  - name: replica
    host: ${DB_REPLICA_HOST}
    port: ${DB_PORT:5432}
    username: ${DB_USER}
    password: ${DB_PASSWORD}
    database: ${DB_NAME}
    readonly: true
    
caches:
  - name: redis
    host: ${REDIS_HOST}
    port: ${REDIS_PORT:6379}
    password: ${REDIS_PASSWORD}
    ssl: true
    
services:
  - name: api
    host: 0.0.0.0
    port: ${PORT:8000}
    workers: ${WEB_CONCURRENCY:4}
    settings:
      debug: false
      cors:
        enabled: true
        origins: ${CORS_ORIGINS}
        
  - name: worker
    concurrency: ${WORKER_CONCURRENCY:20}
    
monitoring:
  logging:
    level: WARNING
    outputs:
      - type: cloudwatch
        log_group: /aws/myapp/production
        
  metrics:
    backend: prometheus
    port: 9090
    
  alerts:
    enabled: true
    services:
      - pagerduty: ${PAGERDUTY_KEY}
      - slack: ${SLACK_WEBHOOK}
      
features:
  rate_limiting:
    default_limit: 1000
    redis_backend: true
  caching:
    ttl: 3600
    redis_backend: true
```

## Environment Manager Implementation

### Configuration Loader

```python
# config/loader.py
import os
from pathlib import Path
from dataknobs_config import Config, Settings
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Manages multi-environment configuration loading."""
    
    ENVIRONMENTS = ["development", "staging", "production", "test"]
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.environment = self._detect_environment()
        self.config = None
    
    def _detect_environment(self) -> str:
        """Detect current environment from various sources."""
        # Check environment variables
        env = os.getenv("APP_ENV") or \
              os.getenv("ENVIRONMENT") or \
              os.getenv("ENV")
        
        if env and env.lower() in self.ENVIRONMENTS:
            return env.lower()
        
        # Check for CI/CD indicators
        if os.getenv("CI"):
            return "test"
        
        # Default to development
        return "development"
    
    def load(self, 
             environment: Optional[str] = None,
             include_secrets: bool = True,
             include_local: bool = True) -> Config:
        """Load configuration for specified environment."""
        
        env = environment or self.environment
        logger.info(f"Loading configuration for environment: {env}")
        
        # Start with base configuration
        config = Config.from_file(self.config_dir / "base.yaml")
        
        # Merge environment-specific configuration
        env_file = self.config_dir / f"{env}.yaml"
        if env_file.exists():
            config.merge_file(str(env_file))
            logger.info(f"Merged environment config: {env_file}")
        
        # Merge secrets if they exist
        if include_secrets:
            secrets_file = self.config_dir / "secrets" / f"{env}.yaml"
            if secrets_file.exists():
                config.merge_file(str(secrets_file))
                logger.info(f"Merged secrets: {secrets_file}")
        
        # Merge local overrides (for development)
        if include_local and env == "development":
            local_file = self.config_dir / "local.yaml"
            if local_file.exists():
                config.merge_file(str(local_file))
                logger.info(f"Merged local overrides: {local_file}")
        
        # Apply environment variable overrides
        config.apply_env_overrides()
        
        # Validate configuration
        self._validate_config(config, env)
        
        self.config = config
        return config
    
    def _validate_config(self, config: Config, environment: str):
        """Validate configuration for environment."""
        # Check required fields based on environment
        if environment == "production":
            self._validate_production_config(config)
        elif environment == "staging":
            self._validate_staging_config(config)
    
    def _validate_production_config(self, config: Config):
        """Validate production configuration."""
        required = [
            ("databases", "primary", "password"),
            ("monitoring", "alerts", "enabled"),
            ("services", "api", "workers"),
        ]
        
        for path in required:
            value = config.get(*path[:-1])
            if not value or path[-1] not in value:
                raise ValueError(f"Missing required production config: {'.'.join(path)}")
    
    def _validate_staging_config(self, config: Config):
        """Validate staging configuration."""
        # Staging-specific validation
        pass
    
    def get_environment(self) -> str:
        """Get current environment name."""
        return self.environment
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
```

### Environment-Aware Application

```python
# app.py
from config.loader import EnvironmentConfig
from dataknobs_config import Config
import logging
from typing import Optional

class Application:
    """Environment-aware application."""
    
    def __init__(self, environment: Optional[str] = None):
        self.env_config = EnvironmentConfig()
        self.config = self.env_config.load(environment)
        self.environment = self.env_config.get_environment()
        self._setup_logging()
        self._setup_features()
    
    def _setup_logging(self):
        """Configure logging based on environment."""
        log_config = self.config.get("monitoring", "logging")
        
        level = getattr(logging, log_config.get("level", "INFO"))
        format_type = log_config.get("format", "simple")
        
        if format_type == "json":
            import json_logging
            json_logging.init_fastapi(enable_json=True)
        else:
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add environment-specific handlers
        if self.env_config.is_production():
            self._add_production_logging()
        elif self.env_config.is_development():
            self._add_development_logging()
    
    def _add_production_logging(self):
        """Add production logging handlers."""
        outputs = self.config.get("monitoring", "logging", "outputs", default=[])
        
        for output in outputs:
            if isinstance(output, dict) and "cloudwatch" in output:
                # Setup CloudWatch logging
                import watchtower
                handler = watchtower.CloudWatchLogHandler(
                    log_group=output["cloudwatch"]["log_group"]
                )
                logging.getLogger().addHandler(handler)
    
    def _add_development_logging(self):
        """Add development logging handlers."""
        # Development uses console logging by default
        pass
    
    def _setup_features(self):
        """Enable/disable features based on configuration."""
        features = self.config.get("features", default={})
        
        # Setup authentication
        if features.get("authentication", {}).get("enabled", False):
            self._setup_authentication()
        
        # Setup rate limiting
        if features.get("rate_limiting", {}).get("enabled", False):
            self._setup_rate_limiting()
        
        # Setup caching
        if features.get("caching", {}).get("enabled", False):
            self._setup_caching()
    
    def _setup_authentication(self):
        """Configure authentication."""
        auth_config = self.config.get("features", "authentication")
        # Authentication setup logic
        pass
    
    def _setup_rate_limiting(self):
        """Configure rate limiting."""
        rate_config = self.config.get("features", "rate_limiting")
        
        if self.env_config.is_production() and rate_config.get("redis_backend"):
            # Use Redis for distributed rate limiting
            pass
        else:
            # Use in-memory rate limiting
            pass
    
    def _setup_caching(self):
        """Configure caching."""
        cache_config = self.config.get("features", "caching")
        
        if cache_config.get("redis_backend"):
            # Use Redis cache
            cache = self.config.construct("caches", "redis")
        else:
            # Use in-memory cache
            from cachetools import TTLCache
            cache = TTLCache(maxsize=1000, ttl=cache_config.get("ttl", 300))
    
    def run(self):
        """Run the application."""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting application in {self.environment} environment")
        
        # Start services based on environment
        if self.env_config.is_production():
            self._run_production()
        elif self.env_config.is_development():
            self._run_development()
        else:
            self._run_default()
    
    def _run_production(self):
        """Run in production mode."""
        # Production-specific startup
        pass
    
    def _run_development(self):
        """Run in development mode."""
        # Development-specific startup (hot reload, debug, etc.)
        pass
    
    def _run_default(self):
        """Run in default mode."""
        pass
```

## Docker Integration

### Dockerfile with Multi-Stage Builds

```dockerfile
# Dockerfile
# Base stage
FROM python:3.11-slim as base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Development stage
FROM base as development
ENV APP_ENV=development
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["python", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]

# Production stage
FROM base as production
ENV APP_ENV=production
RUN pip install --no-cache-dir gunicorn
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker"]
```

### Docker Compose for Multiple Environments

```yaml
# docker-compose.yaml
version: '3.8'

services:
  app:
    build:
      context: .
      target: ${BUILD_TARGET:-development}
    environment:
      - APP_ENV=${APP_ENV:-development}
    env_file:
      - .env
      - .env.${APP_ENV:-development}
    volumes:
      - ./config:/app/config:ro
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: ${DB_USER:-dev_user}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-dev_password}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    command: redis-server ${REDIS_CONFIG:-}

volumes:
  postgres_data:
```

### Environment-Specific Compose Files

```yaml
# docker-compose.development.yaml
version: '3.8'

services:
  app:
    build:
      target: development
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./config:/app/config:ro
    environment:
      - DEBUG=true

  postgres:
    ports:
      - "5432:5432"

  redis:
    ports:
      - "6379:6379"
```

```yaml
# docker-compose.production.yaml
version: '3.8'

services:
  app:
    build:
      target: production
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    environment:
      - WEB_CONCURRENCY=4

  postgres:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

  redis:
    command: redis-server --requirepass ${REDIS_PASSWORD}
```

## Kubernetes Configuration

### ConfigMaps for Each Environment

```yaml
# k8s/configmap-development.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-development
data:
  APP_ENV: "development"
  LOG_LEVEL: "DEBUG"
  DATABASE_HOST: "postgres-service"
  REDIS_HOST: "redis-service"
```

```yaml
# k8s/configmap-production.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-production
data:
  APP_ENV: "production"
  LOG_LEVEL: "WARNING"
  WEB_CONCURRENCY: "4"
  WORKER_CONCURRENCY: "20"
```

### Kustomization for Environment Management

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml

configMapGenerator:
  - name: app-config
    files:
      - config/base.yaml
```

```yaml
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

patchesStrategicMerge:
  - deployment-patch.yaml

configMapGenerator:
  - name: app-config
    behavior: merge
    files:
      - config/production.yaml

secretGenerator:
  - name: app-secrets
    envs:
      - secrets.env
```

## Environment Variables Management

### .env Files

```bash
# .env.development
APP_ENV=development
DEBUG=true
DB_HOST=localhost
DB_PORT=5432
DB_USER=dev_user
DB_PASSWORD=dev_password
REDIS_HOST=localhost
REDIS_PORT=6379
```

```bash
# .env.production
APP_ENV=production
DEBUG=false
# Sensitive values should be injected by CI/CD
# DB_HOST=
# DB_PASSWORD=
# REDIS_PASSWORD=
# JWT_SECRET=
```

### Environment Variable Validation

```python
# config/validators.py
import os
from typing import List, Dict, Any

class EnvironmentValidator:
    """Validate required environment variables."""
    
    REQUIRED_VARS = {
        "development": [],
        "staging": [
            "DB_USER",
            "DB_PASSWORD",
            "REDIS_PASSWORD",
        ],
        "production": [
            "DB_HOST",
            "DB_USER",
            "DB_PASSWORD",
            "REDIS_HOST",
            "REDIS_PASSWORD",
            "JWT_SECRET",
            "CORS_ORIGINS",
        ]
    }
    
    @classmethod
    def validate(cls, environment: str):
        """Validate environment variables for given environment."""
        required = cls.REQUIRED_VARS.get(environment, [])
        missing = []
        
        for var in required:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables for {environment}: "
                f"{', '.join(missing)}"
            )
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get summary of environment variables."""
        return {
            "environment": os.getenv("APP_ENV", "unknown"),
            "configured": {
                var: "***" if "PASSWORD" in var or "SECRET" in var 
                else os.getenv(var, "not set")
                for var in cls.REQUIRED_VARS.get(
                    os.getenv("APP_ENV", "development"), []
                )
            }
        }
```

## Testing Different Environments

### Environment-Specific Tests

```python
# tests/test_environments.py
import pytest
from config.loader import EnvironmentConfig
import os

class TestEnvironmentConfig:
    """Test environment-specific configurations."""
    
    @pytest.fixture
    def reset_env(self):
        """Reset environment after test."""
        original = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original)
    
    def test_development_config(self, reset_env):
        """Test development configuration."""
        os.environ["APP_ENV"] = "development"
        
        env_config = EnvironmentConfig()
        config = env_config.load()
        
        assert env_config.is_development()
        assert config.get("services", "api", "settings", "debug") is True
        assert config.get("features", "rate_limiting", "enabled") is False
    
    def test_production_config(self, reset_env):
        """Test production configuration."""
        os.environ["APP_ENV"] = "production"
        os.environ["DB_HOST"] = "prod-db.example.com"
        os.environ["DB_PASSWORD"] = "secret"
        
        env_config = EnvironmentConfig()
        config = env_config.load()
        
        assert env_config.is_production()
        assert config.get("services", "api", "settings", "debug") is False
        assert config.get("databases", "primary", "host") == "prod-db.example.com"
    
    def test_environment_override(self, reset_env):
        """Test environment variable overrides."""
        os.environ["APP_ENV"] = "staging"
        os.environ["DATAKNOBS_DATABASE__PRIMARY__PORT"] = "5433"
        
        env_config = EnvironmentConfig()
        config = env_config.load()
        
        assert config.get("databases", "primary", "port") == 5433
```

## Best Practices

1. **Keep secrets separate** from configuration files
2. **Use base configuration** for shared settings
3. **Validate environment variables** on startup
4. **Use different database names** per environment
5. **Enable debug/development features** only in development
6. **Use environment detection** to auto-configure
7. **Implement configuration validation** for each environment
8. **Use Docker multi-stage builds** for different environments
9. **Keep local overrides** in gitignored files
10. **Document environment-specific** requirements clearly
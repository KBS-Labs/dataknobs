# Environment Variables

## Overview

This guide covers the environment variables used throughout the DataKnobs project for configuration and development.

## Core Environment Variables

### Configuration Variables

- **`DATAKNOBS_CONFIG_PATH`** - Path to configuration file (default: `config.yaml`)
- **`DATAKNOBS_ENV`** - Environment name (development, staging, production)
- **`DATAKNOBS_LOG_LEVEL`** - Logging level (DEBUG, INFO, WARNING, ERROR)
- **`DATAKNOBS_DEBUG`** - Enable debug mode (true/false)

### Database Configuration

#### PostgreSQL
- **`POSTGRES_HOST`** - PostgreSQL host (default: localhost)
- **`POSTGRES_PORT`** - PostgreSQL port (default: 5432)
- **`POSTGRES_DB`** - Database name
- **`POSTGRES_USER`** - Database username
- **`POSTGRES_PASSWORD`** - Database password
- **`POSTGRES_POOL_SIZE`** - Connection pool size (default: 10)

#### Elasticsearch
- **`ELASTICSEARCH_HOST`** - Elasticsearch host (default: localhost:9200)
- **`ELASTICSEARCH_USER`** - Elasticsearch username
- **`ELASTICSEARCH_PASSWORD`** - Elasticsearch password
- **`ELASTICSEARCH_INDEX`** - Default index name

#### Redis
- **`REDIS_HOST`** - Redis host (default: localhost)
- **`REDIS_PORT`** - Redis port (default: 6379)
- **`REDIS_DB`** - Redis database number (default: 0)
- **`REDIS_PASSWORD`** - Redis password (optional)

### AWS Configuration

- **`AWS_ACCESS_KEY_ID`** - AWS access key
- **`AWS_SECRET_ACCESS_KEY`** - AWS secret key
- **`AWS_REGION`** - AWS region (default: us-east-1)
- **`S3_BUCKET`** - Default S3 bucket name

### Development Variables

- **`UV_SYSTEM_PYTHON`** - Use system Python for UV
- **`UV_PYTHON_VERSION`** - Python version for UV
- **`PYTEST_WORKERS`** - Number of pytest workers for parallel testing
- **`MKDOCS_PORT`** - Port for MkDocs server (default: 8000)

## Usage Examples

### Setting Variables in Shell

```bash
# Export variables
export DATAKNOBS_ENV=production
export POSTGRES_HOST=db.example.com
export DATAKNOBS_LOG_LEVEL=INFO

# Run application
python app.py
```

### Using .env Files

Create a `.env` file in your project root:

```bash
# .env
DATAKNOBS_ENV=development
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=dataknobs_dev
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secret
DATAKNOBS_LOG_LEVEL=DEBUG
```

Load with python-dotenv:

```python
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
db_host = os.getenv("POSTGRES_HOST", "localhost")
log_level = os.getenv("DATAKNOBS_LOG_LEVEL", "INFO")
```

## Configuration Priority

Environment variables follow this priority order:

1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration files
4. Default values (lowest priority)

## Security Best Practices

1. **Never commit secrets** - Keep `.env` files in `.gitignore`
2. **Use secrets management** - Consider tools like AWS Secrets Manager
3. **Rotate credentials** - Regularly update passwords and keys
4. **Minimal permissions** - Use least-privilege access
5. **Encrypt sensitive data** - Use encryption for stored credentials

## Docker Configuration

When using Docker, pass environment variables via:

```yaml
# docker-compose.yml
services:
  app:
    environment:
      - DATAKNOBS_ENV=production
      - POSTGRES_HOST=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    env_file:
      - .env
```

## Testing with Environment Variables

```python
# In tests
import os
import pytest

@pytest.fixture
def test_env():
    """Set test environment variables."""
    original = os.environ.copy()
    
    os.environ["DATAKNOBS_ENV"] = "test"
    os.environ["POSTGRES_DB"] = "test_db"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original)
```

## See Also

- [Configuration System](configuration-system.md)
- [Adding Config Support](adding-config-support.md)
- [UV Environment](uv-environment.md)
- [Docker Development](ci-cd.md)
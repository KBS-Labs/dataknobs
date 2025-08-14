# Configuration Example

This example demonstrates the powerful configuration capabilities of the dataknobs-data package integrated with dataknobs-config.

## Complete Configuration Example

```python
#!/usr/bin/env python3
"""
Configuration example with DataKnobs.

This example demonstrates:
- Environment variable substitution
- Factory registration
- Cross-references in configuration
- Multi-environment configuration
- Dynamic configuration updates
- Configuration validation
"""

import os
import yaml
import json
from dataknobs_config import Config
from dataknobs_data import DatabaseFactory, Record, Query, database_factory

class ConfigurationDemo:
    """Demonstrate configuration capabilities."""
    
    def demonstrate_environment_variables(self):
        """Show environment variable substitution."""
        print("=== Environment Variable Substitution ===\n")
        
        # Set some environment variables
        os.environ["DB_HOST"] = "prod-db.example.com"
        os.environ["DB_PORT"] = "5432"
        os.environ["DB_NAME"] = "production"
        os.environ["DB_USER"] = "admin"
        os.environ["DB_PASSWORD"] = "secret123"
        os.environ["CACHE_SIZE"] = "1000"
        os.environ["ENABLE_SSL"] = "true"
        
        # Configuration with environment variables
        config_data = {
            "database": {
                "backend": "postgres",
                "host": "${DB_HOST}",  # Required env var
                "port": "${DB_PORT:5432}",  # With default
                "database": "${DB_NAME:myapp}",  # With default
                "user": "${DB_USER}",
                "password": "${DB_PASSWORD}",
                "ssl_enabled": "${ENABLE_SSL:false}",  # Boolean conversion
                "pool_size": "${DB_POOL_SIZE:10}",  # Integer conversion
                "timeout": "${DB_TIMEOUT:30.0}"  # Float conversion
            },
            "cache": {
                "backend": "memory",
                "max_size": "${CACHE_SIZE:500}",  # Integer with default
                "ttl": "${CACHE_TTL:3600}"
            }
        }
        
        config = Config()
        config.load(config_data)
        
        # Get substituted values
        db_config = config.get("database")
        print("Database Configuration:")
        print(f"  Host: {db_config['host']}")
        print(f"  Port: {db_config['port']} (type: {type(db_config['port']).__name__})")
        print(f"  Database: {db_config['database']}")
        print(f"  SSL Enabled: {db_config['ssl_enabled']} (type: {type(db_config['ssl_enabled']).__name__})")
        print(f"  Pool Size: {db_config['pool_size']} (type: {type(db_config['pool_size']).__name__})")
        
        cache_config = config.get("cache")
        print(f"\nCache Configuration:")
        print(f"  Max Size: {cache_config['max_size']} (type: {type(cache_config['max_size']).__name__})")
    
    def demonstrate_factory_registration(self):
        """Show factory registration with config."""
        print("\n=== Factory Registration ===\n")
        
        config = Config()
        
        # Register the database factory
        config.register_factory("database", database_factory)
        print(f"Registered factory: database")
        
        # Configuration using registered factory
        config.load({
            "databases": [
                {
                    "name": "primary",
                    "factory": "database",  # Reference to registered factory
                    "backend": "memory"
                },
                {
                    "name": "cache",
                    "factory": "database",
                    "backend": "memory",
                    "max_size": 100
                },
                {
                    "name": "audit",
                    "factory": "database",
                    "backend": "file",
                    "path": "./audit.json"
                }
            ]
        })
        
        # Get instances using factory
        for db_config in config.get("databases", []):
            name = db_config.get("name")
            db = config.get_instance("databases", name)
            print(f"Created {name} database using factory")
            
            # Test the database
            test_record = Record({"db": name, "test": True})
            record_id = db.create(test_record)
            assert db.read(record_id) is not None
            print(f"  ✅ {name} database working")
    
    def demonstrate_cross_references(self):
        """Show cross-references in configuration."""
        print("\n=== Cross-References ===\n")
        
        config = Config()
        config.load({
            "defaults": {
                "region": "us-east-1",
                "environment": "production",
                "app_name": "dataknobs"
            },
            "paths": {
                "data_dir": "/var/data/${defaults.app_name}",
                "log_dir": "/var/log/${defaults.app_name}",
                "backup_dir": "/backup/${defaults.environment}"
            },
            "databases": {
                "s3": {
                    "backend": "s3",
                    "bucket": "${defaults.app_name}-${defaults.environment}",
                    "region": "${defaults.region}",
                    "prefix": "data/"
                },
                "file": {
                    "backend": "file",
                    "path": "${paths.data_dir}/records.json"
                }
            }
        })
        
        # Resolve cross-references
        s3_config = config.get("databases.s3")
        print("S3 Configuration with cross-references:")
        print(f"  Bucket: {s3_config['bucket']}")
        print(f"  Region: {s3_config['region']}")
        
        file_config = config.get("databases.file")
        print(f"\nFile Configuration with cross-references:")
        print(f"  Path: {file_config['path']}")
    
    def demonstrate_multi_environment(self):
        """Show multi-environment configuration."""
        print("\n=== Multi-Environment Configuration ===\n")
        
        # Base configuration
        base_config = {
            "app": {
                "name": "DataKnobs App",
                "version": "1.0.0"
            },
            "features": {
                "cache_enabled": True,
                "max_connections": 100
            }
        }
        
        # Environment-specific overrides
        env_configs = {
            "development": {
                "database": {
                    "backend": "memory"
                },
                "features": {
                    "cache_enabled": False,
                    "debug_mode": True
                },
                "logging": {
                    "level": "DEBUG",
                    "file": "./dev.log"
                }
            },
            "testing": {
                "database": {
                    "backend": "file",
                    "path": "/tmp/test.json"
                },
                "features": {
                    "max_connections": 10
                },
                "logging": {
                    "level": "INFO",
                    "file": "/tmp/test.log"
                }
            },
            "production": {
                "database": {
                    "backend": "postgres",
                    "host": "${DB_HOST}",
                    "database": "${DB_NAME}",
                    "user": "${DB_USER}",
                    "password": "${DB_PASSWORD}",
                    "pool_size": 50
                },
                "features": {
                    "cache_enabled": True,
                    "max_connections": 500
                },
                "logging": {
                    "level": "WARNING",
                    "file": "/var/log/app.log"
                }
            }
        }
        
        # Load configuration for each environment
        for env_name, env_config in env_configs.items():
            config = Config()
            config.load(base_config)
            config.merge(env_config)
            
            print(f"{env_name.upper()} Environment:")
            print(f"  App: {config.get('app.name')} v{config.get('app.version')}")
            print(f"  Database: {config.get('database.backend')}")
            print(f"  Cache Enabled: {config.get('features.cache_enabled')}")
            print(f"  Max Connections: {config.get('features.max_connections')}")
            print(f"  Log Level: {config.get('logging.level')}")
            print()
    
    def demonstrate_yaml_configuration(self):
        """Show YAML configuration file usage."""
        print("=== YAML Configuration ===\n")
        
        # Create a YAML configuration file
        yaml_config = """
app:
  name: DataKnobs Application
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
    password: ${DB_PASSWORD:dbpass}
  
  - name: cache
    factory: database
    backend: memory
    max_size: ${CACHE_SIZE:1000}
  
  - name: search
    factory: database
    backend: elasticsearch
    hosts:
      - ${ES_HOST:localhost:9200}
    index: ${ES_INDEX:${APP_ENV}_search}
  
  - name: archive
    factory: database
    backend: s3
    bucket: ${S3_BUCKET:dataknobs-archive}
    prefix: ${defaults.data_prefix}/archive/
    region: ${defaults.aws_region}

features:
  cache:
    enabled: ${CACHE_ENABLED:true}
    ttl: ${CACHE_TTL:3600}
  
  search:
    enabled: ${SEARCH_ENABLED:true}
    min_score: ${SEARCH_MIN_SCORE:0.5}
  
  archive:
    enabled: ${ARCHIVE_ENABLED:false}
    days_to_archive: ${ARCHIVE_DAYS:365}

logging:
  level: ${LOG_LEVEL:INFO}
  format: ${LOG_FORMAT:json}
  output:
    - type: console
      enabled: true
    - type: file
      enabled: ${LOG_TO_FILE:false}
      path: ${LOG_FILE:/var/log/app.log}
"""
        
        # Save YAML file
        with open("app_config.yaml", "w") as f:
            f.write(yaml_config)
        
        # Load and use configuration
        config = Config("app_config.yaml")
        config.register_factory("database", database_factory)
        
        print("Application Configuration:")
        print(f"  Name: {config.get('app.name')}")
        print(f"  Version: {config.get('app.version')}")
        print(f"  Environment: {config.get('app.environment')}")
        
        print("\nConfigured Databases:")
        for db_config in config.get("databases", []):
            print(f"  - {db_config.get('name')}: {db_config.get('backend')}")
        
        print("\nFeatures:")
        print(f"  Cache: {'Enabled' if config.get('features.cache.enabled') else 'Disabled'}")
        print(f"  Search: {'Enabled' if config.get('features.search.enabled') else 'Disabled'}")
        print(f"  Archive: {'Enabled' if config.get('features.archive.enabled') else 'Disabled'}")
        
        # Clean up
        os.remove("app_config.yaml")
    
    def demonstrate_dynamic_configuration(self):
        """Show dynamic configuration updates."""
        print("\n=== Dynamic Configuration ===\n")
        
        config = Config()
        config.register_factory("database", database_factory)
        
        # Initial configuration
        config.load({
            "databases": [],
            "active_backends": []
        })
        
        print("Initial state: No databases configured\n")
        
        # Dynamically add databases
        database_types = [
            ("cache", {"backend": "memory", "factory": "database"}),
            ("primary", {"backend": "file", "path": "./data.json", "factory": "database"}),
            ("audit", {"backend": "file", "path": "./audit.json", "factory": "database"})
        ]
        
        for name, db_config in database_types:
            # Add to configuration
            db_config["name"] = name
            databases = config.get("databases", [])
            databases.append(db_config)
            config.set("databases", databases)
            
            # Track active backends
            active = config.get("active_backends", [])
            active.append(name)
            config.set("active_backends", active)
            
            print(f"Added {name} database")
            
            # Create instance
            db = config.get_instance("databases", name)
            
            # Test it
            test_record = Record({"dynamic": True, "name": name})
            record_id = db.create(test_record)
            print(f"  ✅ {name} database operational")
        
        print(f"\nActive backends: {', '.join(config.get('active_backends'))}")
        
        # Save configuration
        config.save("dynamic_config.json")
        print("\nConfiguration saved to dynamic_config.json")
        
        # Load and verify
        loaded_config = Config("dynamic_config.json")
        print(f"Loaded configuration has {len(loaded_config.get('databases', []))} databases")
        
        # Clean up
        os.remove("dynamic_config.json")
    
    def demonstrate_validation(self):
        """Show configuration validation."""
        print("\n=== Configuration Validation ===\n")
        
        def validate_database_config(config: dict) -> tuple[bool, str]:
            """Validate database configuration."""
            backend = config.get("backend")
            
            # Check backend is specified
            if not backend:
                return False, "Backend not specified"
            
            # Backend-specific validation
            if backend == "postgres":
                required = ["host", "database", "user", "password"]
                missing = [f for f in required if f not in config]
                if missing:
                    return False, f"Missing required fields for PostgreSQL: {', '.join(missing)}"
            
            elif backend == "elasticsearch":
                if "hosts" not in config or not config["hosts"]:
                    return False, "Elasticsearch requires at least one host"
            
            elif backend == "s3":
                if "bucket" not in config:
                    return False, "S3 requires bucket name"
            
            elif backend == "file":
                if "path" not in config:
                    return False, "File backend requires path"
            
            return True, "Valid"
        
        # Test configurations
        test_configs = [
            {
                "name": "valid_memory",
                "config": {"backend": "memory"}
            },
            {
                "name": "valid_postgres",
                "config": {
                    "backend": "postgres",
                    "host": "localhost",
                    "database": "test",
                    "user": "user",
                    "password": "pass"
                }
            },
            {
                "name": "invalid_postgres",
                "config": {
                    "backend": "postgres",
                    "host": "localhost"
                    # Missing required fields
                }
            },
            {
                "name": "invalid_s3",
                "config": {
                    "backend": "s3"
                    # Missing bucket
                }
            },
            {
                "name": "valid_file",
                "config": {
                    "backend": "file",
                    "path": "./data.json"
                }
            }
        ]
        
        for test in test_configs:
            valid, message = validate_database_config(test["config"])
            status = "✅" if valid else "❌"
            print(f"{status} {test['name']}: {message}")

def main():
    """Run configuration demonstrations."""
    print("DataKnobs Configuration Examples")
    print("=" * 50 + "\n")
    
    demo = ConfigurationDemo()
    
    # Run all demonstrations
    demo.demonstrate_environment_variables()
    demo.demonstrate_factory_registration()
    demo.demonstrate_cross_references()
    demo.demonstrate_multi_environment()
    demo.demonstrate_yaml_configuration()
    demo.demonstrate_dynamic_configuration()
    demo.demonstrate_validation()
    
    print("\n" + "=" * 50)
    print("✅ Configuration examples completed!")
    print("\nKey Takeaways:")
    print("- Environment variables with ${VAR:default} syntax")
    print("- Automatic type conversion (int, float, bool)")
    print("- Factory registration for cleaner configs")
    print("- Cross-references between config sections")
    print("- Multi-environment configuration support")
    print("- Dynamic configuration updates at runtime")
    print("- Configuration validation for reliability")

if __name__ == "__main__":
    main()
```

## Running the Example

```bash
# Install required packages
pip install dataknobs-data dataknobs-config pyyaml

# Run the example
python configuration_example.py

# With environment variables
export DB_HOST=production.db.example.com
export DB_PASSWORD=secret123
export CACHE_SIZE=5000
python configuration_example.py
```

## Configuration File Examples

### Simple JSON Configuration
```json
{
  "database": {
    "backend": "postgres",
    "host": "${DB_HOST:localhost}",
    "port": "${DB_PORT:5432}",
    "database": "${DB_NAME:myapp}",
    "user": "${DB_USER:dbuser}",
    "password": "${DB_PASSWORD}"
  }
}
```

### YAML with Multiple Backends
```yaml
databases:
  primary:
    backend: ${PRIMARY_BACKEND:postgres}
    host: ${DB_HOST:localhost}
    database: ${DB_NAME:production}
    
  cache:
    backend: memory
    max_size: ${CACHE_SIZE:1000}
    
  search:
    backend: elasticsearch
    hosts:
      - ${ES_HOST:localhost:9200}
    index: ${ES_INDEX:search}
    
  archive:
    backend: s3
    bucket: ${S3_BUCKET}
    region: ${AWS_REGION:us-east-1}
```

### Environment-Specific Configurations
```yaml
# config.base.yaml
app:
  name: MyApp
  version: 1.0.0

# config.dev.yaml
extends: config.base.yaml
database:
  backend: memory
logging:
  level: DEBUG

# config.prod.yaml
extends: config.base.yaml
database:
  backend: postgres
  host: ${DB_HOST}
  password: ${DB_PASSWORD}
logging:
  level: WARNING
```

## Key Concepts Demonstrated

1. **Environment Variables**: `${VAR}` and `${VAR:default}` substitution
2. **Type Conversion**: Automatic conversion to int, float, bool
3. **Factory Registration**: Register factories for cleaner configs
4. **Cross-References**: Reference other config values with `${section.key}`
5. **Multi-Environment**: Different configs for dev/test/prod
6. **YAML Support**: Human-readable configuration files
7. **Dynamic Updates**: Modify configuration at runtime
8. **Validation**: Ensure configuration correctness
9. **Merging**: Combine base and environment configs
10. **Persistence**: Save and load configuration state
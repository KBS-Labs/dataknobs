# Factory Pattern Example

This example demonstrates how to use the factory pattern with the dataknobs-data package for dynamic backend selection and configuration.

## Complete Factory Example

```python
#!/usr/bin/env python3
"""
Factory pattern example with DataKnobs.

This example demonstrates:
- Dynamic backend selection
- Factory registration with config
- Custom backend implementation
- Backend information querying
- Multi-environment configuration
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataknobs_data import (
    Database, 
    DatabaseFactory, 
    Record, 
    Query,
    database_factory
)
from dataknobs_config import Config, ConfigurableBase

class SmartDatabaseFactory:
    """
    Smart factory that selects backends based on requirements.
    """
    
    def __init__(self):
        self.factory = DatabaseFactory()
        self.backends_created = []
    
    def create_for_use_case(self, use_case: str, **kwargs) -> Database:
        """
        Create database based on use case requirements.
        
        Args:
            use_case: Type of use case (cache, search, archive, etc.)
            **kwargs: Additional configuration
        
        Returns:
            Appropriate database instance
        """
        backend_config = self._get_config_for_use_case(use_case, **kwargs)
        db = self.factory.create(**backend_config)
        self.backends_created.append((use_case, backend_config["backend"]))
        return db
    
    def _get_config_for_use_case(self, use_case: str, **kwargs) -> Dict[str, Any]:
        """Get configuration for specific use case."""
        configs = {
            "cache": {
                "backend": "memory",
                "description": "High-speed temporary storage"
            },
            "development": {
                "backend": "file",
                "path": kwargs.get("path", "./dev_data.json"),
                "format": "json",
                "description": "Simple file storage for development"
            },
            "testing": {
                "backend": "memory",
                "description": "Isolated in-memory storage for tests"
            },
            "production": {
                "backend": kwargs.get("backend", "postgres"),
                "host": kwargs.get("host", "localhost"),
                "database": kwargs.get("database", "production"),
                "user": kwargs.get("user", "dbuser"),
                "password": kwargs.get("password", "dbpass"),
                "description": "Production database with ACID compliance"
            },
            "search": {
                "backend": "elasticsearch",
                "hosts": kwargs.get("hosts", ["localhost:9200"]),
                "index": kwargs.get("index", "search"),
                "description": "Full-text search and analytics"
            },
            "archive": {
                "backend": "s3",
                "bucket": kwargs.get("bucket", "archive-bucket"),
                "prefix": kwargs.get("prefix", "archives/"),
                "region": kwargs.get("region", "us-east-1"),
                "description": "Long-term storage in cloud"
            },
            "analytics": {
                "backend": "elasticsearch",
                "hosts": kwargs.get("hosts", ["localhost:9200"]),
                "index": kwargs.get("index", "analytics"),
                "description": "Time-series and aggregation queries"
            },
            "audit": {
                "backend": "file",
                "path": kwargs.get("path", "./audit_log.json"),
                "format": "json",
                "description": "Immutable audit trail"
            }
        }
        
        if use_case not in configs:
            raise ValueError(f"Unknown use case: {use_case}")
        
        config = configs[use_case].copy()
        config.update(kwargs)  # Override with user-provided kwargs
        return config
    
    def get_summary(self) -> str:
        """Get summary of created backends."""
        summary = "Backends Created:\n"
        for use_case, backend in self.backends_created:
            summary += f"  - {use_case}: {backend}\n"
        return summary


class ConfigDrivenFactory:
    """
    Factory that uses configuration files for backend selection.
    """
    
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self.factory = DatabaseFactory()
        
        # Register factory with config for cleaner syntax
        self.config.register_factory("database", self.factory)
    
    def get_database(self, name: str) -> Database:
        """
        Get database instance by name from configuration.
        
        Args:
            name: Database name in configuration
        
        Returns:
            Database instance
        """
        return self.config.get_instance("databases", name)
    
    def list_databases(self) -> List[str]:
        """List all configured databases."""
        databases = self.config.get("databases", [])
        return [db.get("name") for db in databases if "name" in db]


class CustomBackendExample:
    """
    Example of registering a custom backend with the factory.
    """
    
    @staticmethod
    def create_custom_backend():
        """Create and register a custom backend."""
        
        class RedisDatabase(Database):
            """Custom Redis backend implementation."""
            
            def __init__(self, host="localhost", port=6379, db=0, **kwargs):
                self.host = host
                self.port = port
                self.db = db
                # In real implementation, would connect to Redis
                self.data = {}  # Simulated storage
            
            def create(self, record: Record) -> str:
                import uuid
                record_id = str(uuid.uuid4())
                record.metadata["id"] = record_id
                self.data[record_id] = record.to_dict()
                return record_id
            
            def read(self, record_id: str) -> Optional[Record]:
                if record_id in self.data:
                    return Record.from_dict(self.data[record_id])
                return None
            
            def update(self, record_id: str, record: Record) -> bool:
                if record_id in self.data:
                    self.data[record_id] = record.to_dict()
                    return True
                return False
            
            def delete(self, record_id: str) -> bool:
                if record_id in self.data:
                    del self.data[record_id]
                    return True
                return False
            
            def search(self, query: Query) -> List[Record]:
                # Simplified search implementation
                results = []
                for data in self.data.values():
                    record = Record.from_dict(data)
                    # Apply filters (simplified)
                    match = True
                    for filter_dict in query.filters:
                        field = filter_dict["field"]
                        op = filter_dict["operator"]
                        value = filter_dict["value"]
                        
                        if field in record.fields:
                            if op == "=" and record.fields[field] != value:
                                match = False
                                break
                    
                    if match:
                        results.append(record)
                
                return results[:query.limit] if query.limit else results
            
            def count(self, query: Optional[Query] = None) -> int:
                if query:
                    return len(self.search(query))
                return len(self.data)
            
            def clear(self) -> None:
                self.data.clear()
        
        # Register the custom backend
        factory = DatabaseFactory()
        factory.register_backend("redis", RedisDatabase)
        
        # Now it can be used like any other backend
        redis_db = factory.create(backend="redis", host="localhost", port=6379)
        return redis_db


def demonstrate_basic_factory():
    """Demonstrate basic factory usage."""
    print("=== Basic Factory Usage ===\n")
    
    factory = DatabaseFactory()
    
    # Get available backends
    backends = factory.get_available_backends()
    print(f"Available backends: {', '.join(backends)}\n")
    
    # Get backend information
    for backend in ["memory", "postgres", "s3"]:
        info = factory.get_backend_info(backend)
        print(f"{backend.upper()}:")
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Persistent: {info.get('persistent', False)}")
        if info.get('requires_install'):
            print(f"  Install: {info['requires_install']}")
        print()


def demonstrate_smart_factory():
    """Demonstrate smart factory with use case selection."""
    print("=== Smart Factory (Use Case Based) ===\n")
    
    smart_factory = SmartDatabaseFactory()
    
    # Create databases for different use cases
    use_cases = [
        ("cache", {}),
        ("development", {"path": "./dev.json"}),
        ("testing", {}),
        ("audit", {"path": "./audit.json"})
    ]
    
    databases = {}
    for use_case, config in use_cases:
        try:
            db = smart_factory.create_for_use_case(use_case, **config)
            databases[use_case] = db
            print(f"✅ Created {use_case} backend")
            
            # Test with sample data
            record = Record({"use_case": use_case, "test": True})
            record_id = db.create(record)
            assert db.read(record_id) is not None
            
        except Exception as e:
            print(f"❌ Failed to create {use_case}: {e}")
    
    print(f"\n{smart_factory.get_summary()}")


def demonstrate_config_driven_factory():
    """Demonstrate configuration-driven factory."""
    print("=== Configuration-Driven Factory ===\n")
    
    # Create configuration
    config_data = {
        "databases": [
            {
                "name": "primary",
                "factory": "database",
                "backend": os.environ.get("PRIMARY_BACKEND", "memory")
            },
            {
                "name": "cache",
                "factory": "database",
                "backend": "memory"
            },
            {
                "name": "backup",
                "factory": "database",
                "backend": "file",
                "path": "./backup.json",
                "format": "json"
            }
        ]
    }
    
    # Save config
    with open("factory_config.json", "w") as f:
        json.dump(config_data, f)
    
    # Use config-driven factory
    config_factory = ConfigDrivenFactory("factory_config.json")
    
    # List databases
    db_names = config_factory.list_databases()
    print(f"Configured databases: {', '.join(db_names)}\n")
    
    # Get and test each database
    for name in db_names:
        db = config_factory.get_database(name)
        print(f"Testing {name} database...")
        
        # Test operations
        test_record = Record({"db": name, "test": True})
        record_id = db.create(test_record)
        retrieved = db.read(record_id)
        
        if retrieved:
            print(f"  ✅ {name} working correctly")
        else:
            print(f"  ❌ {name} failed test")
    
    # Clean up
    os.remove("factory_config.json")


def demonstrate_custom_backend():
    """Demonstrate custom backend registration."""
    print("\n=== Custom Backend Registration ===\n")
    
    # Create and register custom Redis backend
    redis_db = CustomBackendExample.create_custom_backend()
    
    print("Custom Redis backend registered and created")
    
    # Test the custom backend
    records = [
        Record({"type": "user", "name": "Alice", "score": 100}),
        Record({"type": "user", "name": "Bob", "score": 85}),
        Record({"type": "admin", "name": "Charlie", "score": 95})
    ]
    
    # Create records
    record_ids = []
    for record in records:
        record_id = redis_db.create(record)
        record_ids.append(record_id)
    
    print(f"Created {len(record_ids)} records in custom backend")
    
    # Search
    users = redis_db.search(Query().filter("type", "=", "user"))
    print(f"Found {len(users)} users")
    
    # Count
    total = redis_db.count()
    print(f"Total records: {total}")
    
    # Clean up
    redis_db.clear()


def demonstrate_environment_based_selection():
    """Demonstrate environment-based backend selection."""
    print("\n=== Environment-Based Selection ===\n")
    
    class EnvironmentAwareFactory:
        """Factory that selects backend based on environment."""
        
        @staticmethod
        def create_database():
            factory = DatabaseFactory()
            env = os.environ.get("APP_ENV", "development")
            
            configs = {
                "development": {
                    "backend": "memory",
                    "description": "In-memory for fast development"
                },
                "testing": {
                    "backend": "file",
                    "path": "/tmp/test.json",
                    "format": "json",
                    "description": "File-based for test isolation"
                },
                "staging": {
                    "backend": "postgres",
                    "host": os.environ.get("DB_HOST", "staging-db"),
                    "database": "staging",
                    "description": "PostgreSQL for staging"
                },
                "production": {
                    "backend": "postgres",
                    "host": os.environ.get("DB_HOST", "prod-db"),
                    "database": "production",
                    "pool_size": 50,
                    "description": "PostgreSQL with connection pooling"
                }
            }
            
            if env not in configs:
                raise ValueError(f"Unknown environment: {env}")
            
            config = configs[env]
            print(f"Environment: {env}")
            print(f"Backend: {config['backend']}")
            print(f"Description: {config['description']}")
            
            # Remove description before creating
            desc = config.pop("description")
            
            return factory.create(**config)
    
    # Test different environments
    for env in ["development", "testing"]:
        os.environ["APP_ENV"] = env
        try:
            db = EnvironmentAwareFactory.create_database()
            print(f"✅ Created database for {env}\n")
        except Exception as e:
            print(f"❌ Failed for {env}: {e}\n")


def main():
    """Run all factory pattern demonstrations."""
    print("DataKnobs Factory Pattern Examples")
    print("=" * 50 + "\n")
    
    # Run demonstrations
    demonstrate_basic_factory()
    demonstrate_smart_factory()
    demonstrate_config_driven_factory()
    demonstrate_custom_backend()
    demonstrate_environment_based_selection()
    
    print("=" * 50)
    print("✅ Factory pattern examples completed!")
    print("\nKey Takeaways:")
    print("- Factories enable dynamic backend selection")
    print("- Configuration-driven instantiation for flexibility")
    print("- Custom backends can be registered")
    print("- Environment-aware selection for different stages")
    print("- Use case-based selection for optimal backend choice")

if __name__ == "__main__":
    main()
```

## Running the Example

```bash
# Install the package
pip install dataknobs-data dataknobs-config

# Run the example
python factory_pattern_example.py

# With different environments
APP_ENV=development python factory_pattern_example.py
APP_ENV=testing python factory_pattern_example.py
APP_ENV=production python factory_pattern_example.py
```

## Key Concepts Demonstrated

1. **Dynamic Backend Selection**: Choose backends at runtime
2. **Use Case Mapping**: Select optimal backend for each use case
3. **Configuration-Driven**: Use config files for backend setup
4. **Custom Backend Registration**: Add your own backend implementations
5. **Environment Awareness**: Different backends for dev/test/prod
6. **Backend Information API**: Query available backends and requirements
7. **Factory Registration**: Register factories with config system
8. **Error Handling**: Graceful handling of missing dependencies
9. **Testing Support**: Easy backend switching for tests
10. **Multi-Backend Applications**: Use multiple backends together
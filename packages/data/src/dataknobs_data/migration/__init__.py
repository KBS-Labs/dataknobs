"""Migration Module v2 - Clean, streaming-based data migration API.

This module provides a complete rewrite of the migration system with:
- Streaming support for memory-efficient operations
- Reversible operations for rollback support
- Fluent transformer API for data manipulation
- Clear separation between logic and progress tracking
- Parallel migration support
"""

from .factory import (
    MigrationFactory,
    MigratorFactory,
    TransformerFactory,
    migration_factory,
    migrator_factory,
    transformer_factory,
)
from .migration import Migration
from .migrator import Migrator
from .operations import (
    AddField,
    CompositeOperation,
    Operation,
    RemoveField,
    RenameField,
    TransformField,
)
from .progress import MigrationProgress
from .transformer import AddRule, ExcludeRule, MapRule, Transformer, TransformRule

__all__ = [
    # Operations
    "Operation",
    "AddField",
    "RemoveField",
    "RenameField",
    "TransformField",
    "CompositeOperation",
    # Migration
    "Migration",
    # Transformation
    "Transformer",
    "TransformRule",
    "MapRule",
    "ExcludeRule",
    "AddRule",
    # Progress
    "MigrationProgress",
    # Migrator
    "Migrator",
    # Factories
    "migration_factory",
    "transformer_factory",
    "migrator_factory",
    "MigrationFactory",
    "TransformerFactory",
    "MigratorFactory",
]

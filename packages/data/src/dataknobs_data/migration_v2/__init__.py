"""
Migration Module v2 - Clean, streaming-based data migration API.

This module provides a complete rewrite of the migration system with:
- Streaming support for memory-efficient operations
- Reversible operations for rollback support
- Fluent transformer API for data manipulation
- Clear separation between logic and progress tracking
- Parallel migration support
"""

from .operations import (
    Operation,
    AddField,
    RemoveField,
    RenameField,
    TransformField,
    CompositeOperation,
)
from .migration import Migration
from .transformer import Transformer, TransformRule, MapRule, ExcludeRule, AddRule
from .progress import MigrationProgress
from .migrator import Migrator

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
]
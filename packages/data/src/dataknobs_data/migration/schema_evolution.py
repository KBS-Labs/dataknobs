"""Schema evolution and versioning utilities."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from dataknobs_data.fields import Field, FieldType
from dataknobs_data.records import Record

logger = logging.getLogger(__name__)


class MigrationType(Enum):
    """Types of schema migrations."""
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    RENAME_FIELD = "rename_field"
    CHANGE_TYPE = "change_type"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    CUSTOM = "custom"


@dataclass
class SchemaVersion:
    """Represents a schema version."""
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    fields: Dict[str, Field] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'description': self.description,
            'fields': {
                name: {
                    'type': field.type.value if hasattr(field.type, 'value') else str(field.type),
                    'required': field.required,
                    'default': field.default,
                    'metadata': field.metadata
                }
                for name, field in self.fields.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaVersion":
        """Create from dictionary."""
        fields = {}
        for name, field_data in data.get('fields', {}).items():
            field_type = field_data.get('type', 'str')
            if hasattr(FieldType, field_type.upper()):
                field_type = FieldType[field_type.upper()]
            
            fields[name] = Field(
                type=field_type,
                required=field_data.get('required', False),
                default=field_data.get('default'),
                metadata=field_data.get('metadata', {})
            )
        
        return cls(
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            description=data.get('description', ''),
            fields=fields
        )


@dataclass
class Migration:
    """Represents a schema migration."""
    from_version: str
    to_version: str
    migration_type: MigrationType
    description: str = ""
    operations: List[Dict[str, Any]] = field(default_factory=list)
    up_function: Optional[Callable[[Record], Record]] = None
    down_function: Optional[Callable[[Record], Record]] = None
    
    def apply_forward(self, record: Record) -> Record:
        """Apply forward migration to a record."""
        if self.up_function:
            return self.up_function(record)
        
        # Apply built-in migration types
        for operation in self.operations:
            record = self._apply_operation(record, operation, forward=True)
        
        return record
    
    def apply_backward(self, record: Record) -> Record:
        """Apply backward migration to a record."""
        if self.down_function:
            return self.down_function(record)
        
        # Apply built-in migration types in reverse
        for operation in reversed(self.operations):
            record = self._apply_operation(record, operation, forward=False)
        
        return record
    
    def _apply_operation(self, record: Record, operation: Dict[str, Any], forward: bool) -> Record:
        """Apply a single migration operation."""
        op_type = MigrationType(operation['type'])
        
        if op_type == MigrationType.ADD_FIELD:
            if forward:
                field_name = operation['field_name']
                default_value = operation.get('default_value')
                if field_name not in record.fields:
                    record.fields[field_name] = Field(
                        type=operation.get('field_type', 'str'),
                        default=default_value
                    )
                    record.fields[field_name].value = default_value
            else:
                # Reverse: remove the field
                field_name = operation['field_name']
                if field_name in record.fields:
                    del record.fields[field_name]
        
        elif op_type == MigrationType.REMOVE_FIELD:
            if forward:
                field_name = operation['field_name']
                if field_name in record.fields:
                    del record.fields[field_name]
            else:
                # Reverse: add the field back with stored value
                field_name = operation['field_name']
                if field_name not in record.fields:
                    record.fields[field_name] = Field(
                        type=operation.get('field_type', 'str')
                    )
        
        elif op_type == MigrationType.RENAME_FIELD:
            old_name = operation['old_name']
            new_name = operation['new_name']
            
            if forward:
                if old_name in record.fields:
                    record.fields[new_name] = record.fields.pop(old_name)
            else:
                if new_name in record.fields:
                    record.fields[old_name] = record.fields.pop(new_name)
        
        elif op_type == MigrationType.CHANGE_TYPE:
            field_name = operation['field_name']
            
            if forward:
                new_type = operation['new_type']
                converter = operation.get('converter')
            else:
                new_type = operation['old_type']
                converter = operation.get('reverse_converter')
            
            if field_name in record.fields:
                field = record.fields[field_name]
                if converter:
                    field.value = converter(field.value)
                field.type = new_type
        
        elif op_type == MigrationType.CUSTOM:
            custom_func = operation.get('forward' if forward else 'backward')
            if custom_func:
                record = custom_func(record)
        
        return record


class SchemaEvolution:
    """Manage schema evolution and migrations."""
    
    def __init__(self):
        """Initialize schema evolution manager."""
        self.versions: Dict[str, SchemaVersion] = {}
        self.migrations: List[Migration] = []
        self.current_version: Optional[str] = None
    
    def add_version(self, version: SchemaVersion) -> None:
        """Add a schema version."""
        self.versions[version.version] = version
        if not self.current_version:
            self.current_version = version.version
        logger.info(f"Added schema version: {version.version}")
    
    def add_migration(self, migration: Migration) -> None:
        """Add a migration between versions."""
        if migration.from_version not in self.versions:
            raise ValueError(f"Unknown source version: {migration.from_version}")
        if migration.to_version not in self.versions:
            raise ValueError(f"Unknown target version: {migration.to_version}")
        
        self.migrations.append(migration)
        logger.info(f"Added migration: {migration.from_version} -> {migration.to_version}")
    
    def set_current_version(self, version: str) -> None:
        """Set the current schema version."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")
        self.current_version = version
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[Migration]:
        """Find migration path between versions."""
        if from_version == to_version:
            return []
        
        # Simple linear search for now - could be optimized with graph algorithms
        path = []
        current = from_version
        
        while current != to_version:
            found = False
            for migration in self.migrations:
                if migration.from_version == current:
                    path.append(migration)
                    current = migration.to_version
                    found = True
                    break
            
            if not found:
                # Try backward migrations
                for migration in self.migrations:
                    if migration.to_version == current:
                        path.append(migration)
                        current = migration.from_version
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"No migration path from {from_version} to {to_version}")
        
        return path
    
    def migrate_record(
        self,
        record: Record,
        from_version: str,
        to_version: str
    ) -> Record:
        """Migrate a record from one version to another."""
        migrations = self.get_migration_path(from_version, to_version)
        
        for migration in migrations:
            if migration.from_version == from_version:
                # Forward migration
                record = migration.apply_forward(record)
            else:
                # Backward migration
                record = migration.apply_backward(record)
        
        # Update record metadata
        if not record.metadata:
            record.metadata = {}
        record.metadata['schema_version'] = to_version
        
        return record
    
    def auto_detect_changes(
        self,
        old_version: SchemaVersion,
        new_version: SchemaVersion
    ) -> Migration:
        """Auto-detect changes between schema versions."""
        operations = []
        
        old_fields = set(old_version.fields.keys())
        new_fields = set(new_version.fields.keys())
        
        # Detect added fields
        for field_name in new_fields - old_fields:
            field = new_version.fields[field_name]
            operations.append({
                'type': MigrationType.ADD_FIELD.value,
                'field_name': field_name,
                'field_type': str(field.type),
                'default_value': field.default
            })
        
        # Detect removed fields
        for field_name in old_fields - new_fields:
            field = old_version.fields[field_name]
            operations.append({
                'type': MigrationType.REMOVE_FIELD.value,
                'field_name': field_name,
                'field_type': str(field.type)
            })
        
        # Detect type changes
        for field_name in old_fields & new_fields:
            old_field = old_version.fields[field_name]
            new_field = new_version.fields[field_name]
            
            if old_field.type != new_field.type:
                operations.append({
                    'type': MigrationType.CHANGE_TYPE.value,
                    'field_name': field_name,
                    'old_type': str(old_field.type),
                    'new_type': str(new_field.type)
                })
        
        return Migration(
            from_version=old_version.version,
            to_version=new_version.version,
            migration_type=MigrationType.CUSTOM,
            description=f"Auto-detected migration from {old_version.version} to {new_version.version}",
            operations=operations
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save schema evolution to JSON file."""
        data = {
            'current_version': self.current_version,
            'versions': {
                version_id: version.to_dict()
                for version_id, version in self.versions.items()
            },
            'migrations': [
                {
                    'from_version': m.from_version,
                    'to_version': m.to_version,
                    'type': m.migration_type.value,
                    'description': m.description,
                    'operations': m.operations
                }
                for m in self.migrations
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "SchemaEvolution":
        """Load schema evolution from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        evolution = cls()
        evolution.current_version = data.get('current_version')
        
        # Load versions
        for version_id, version_data in data.get('versions', {}).items():
            version = SchemaVersion.from_dict(version_data)
            evolution.versions[version_id] = version
        
        # Load migrations
        for migration_data in data.get('migrations', []):
            migration = Migration(
                from_version=migration_data['from_version'],
                to_version=migration_data['to_version'],
                migration_type=MigrationType(migration_data['type']),
                description=migration_data.get('description', ''),
                operations=migration_data.get('operations', [])
            )
            evolution.migrations.append(migration)
        
        return evolution
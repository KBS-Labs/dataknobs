"""Factory classes for migration v2 components."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from dataknobs_config import FactoryBase

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
from .transformer import Transformer

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class MigrationFactory(FactoryBase):
    """Factory for creating migrations from configuration.
    
    Configuration Options:
        from_version (str): Source version
        to_version (str): Target version
        description (str): Migration description
        operations (list): List of operation definitions
        
    Operation Types:
        - add_field: Add a new field
        - remove_field: Remove an existing field
        - rename_field: Rename a field
        - transform_field: Transform field values
        - composite: Multiple operations combined
        
    Example Configuration:
        migrations:
          - name: v1_to_v2
            factory: migration
            from_version: "1.0"
            to_version: "2.0"
            description: Add user metadata fields
            operations:
              - type: add_field
                field_name: created_at
                default_value: "2024-01-01"
              - type: rename_field
                old_name: username
                new_name: user_name
              - type: transform_field
                field_name: price
                transform: "lambda x: x * 1.1"
    """

    def create(self, **config) -> Migration:
        """Create a Migration instance from configuration.
        
        Args:
            **config: Migration configuration
            
        Returns:
            Migration instance
        """
        from_version = config.get("from_version", "0.0")
        to_version = config.get("to_version", "1.0")
        description = config.get("description")

        logger.info(f"Creating migration: {from_version} -> {to_version}")

        migration = Migration(from_version, to_version, description)

        # Add operations
        operations = config.get("operations", [])
        for op_config in operations:
            operation = self._create_operation(op_config)
            if operation:
                migration.add(operation)

        return migration

    def _create_operation(self, op_config: dict[str, Any]) -> Operation | None:
        """Create an operation from configuration.
        
        Args:
            op_config: Operation configuration
            
        Returns:
            Operation instance or None if invalid
        """
        op_type = op_config.get("type", "").lower()

        if op_type == "add_field":
            field_name = op_config.get("field_name")
            if not field_name:
                raise ValueError("add_field operation requires field_name")
            return AddField(
                field_name=field_name,
                default_value=op_config.get("default_value"),
                field_type=self._parse_field_type(op_config.get("field_type"))
            )

        elif op_type == "remove_field":
            field_name = op_config.get("field_name")
            if not field_name:
                raise ValueError("remove_field operation requires field_name")
            return RemoveField(
                field_name=field_name,
                store_removed=op_config.get("store_removed", False)
            )

        elif op_type == "rename_field":
            old_name = op_config.get("old_name")
            new_name = op_config.get("new_name")
            if not old_name or not new_name:
                raise ValueError("rename_field operation requires old_name and new_name")
            return RenameField(
                old_name=old_name,
                new_name=new_name
            )

        elif op_type == "transform_field":
            # Note: Transform functions from config strings require careful handling
            # For security, we don't eval arbitrary code. Instead, support predefined transforms.
            field_name = op_config.get("field_name")
            if not field_name:
                raise ValueError("transform_field operation requires field_name")
            transform_fn = self._get_transform_function(op_config.get("transform"))
            if not transform_fn:
                raise ValueError("transform_field operation requires transform function")
            reverse_fn = self._get_transform_function(op_config.get("reverse"))

            return TransformField(
                field_name=field_name,
                transform_fn=transform_fn,
                reverse_fn=reverse_fn
            )

        elif op_type == "composite":
            sub_operations = []
            for sub_config in op_config.get("operations", []):
                sub_op = self._create_operation(sub_config)
                if sub_op:
                    sub_operations.append(sub_op)
            return CompositeOperation(sub_operations) if sub_operations else None

        else:
            logger.warning(f"Unknown operation type: {op_type}")
            return None

    def _parse_field_type(self, type_str: str | None):
        """Parse field type from string.
        
        Args:
            type_str: Field type string
            
        Returns:
            FieldType or None
        """
        if not type_str:
            return None

        from dataknobs_data.fields import FieldType

        try:
            return FieldType[type_str.upper()]
        except KeyError:
            logger.warning(f"Unknown field type: {type_str}")
            return None

    def _get_transform_function(self, transform_spec: Any) -> Callable | None:
        """Get transform function from specification.
        
        For security, we don't eval arbitrary code. Instead, we support
        predefined transform patterns.
        
        Args:
            transform_spec: Transform specification
            
        Returns:
            Transform function or None
        """
        if not transform_spec:
            return None

        # Support predefined transforms
        if transform_spec == "uppercase":
            return lambda x: x.upper() if isinstance(x, str) else x
        elif transform_spec == "lowercase":
            return lambda x: x.lower() if isinstance(x, str) else x
        elif transform_spec == "trim":
            return lambda x: x.strip() if isinstance(x, str) else x
        elif isinstance(transform_spec, dict):
            # Support multiplication/division
            if "multiply" in transform_spec:
                factor = transform_spec["multiply"]
                return lambda x: x * factor if isinstance(x, (int, float)) else x
            elif "divide" in transform_spec:
                factor = transform_spec["divide"]
                return lambda x: x / factor if isinstance(x, (int, float)) else x

        logger.warning(f"Unsupported transform specification: {transform_spec}")
        return None


class TransformerFactory(FactoryBase):
    """Factory for creating transformers from configuration.
    
    Configuration Options:
        rules (list): List of transformation rules
        
    Rule Types:
        - map: Map field to another field with optional transformation
        - rename: Rename a field
        - exclude: Remove fields
        - add: Add new fields
        
    Example Configuration:
        transformers:
          - name: cleanup_transformer
            factory: transformer
            rules:
              - type: map
                source: old_id
                target: id
              - type: exclude
                fields: [temp_field, debug_info]
              - type: add
                field_name: processed
                value: true
    """

    def create(self, **config) -> Transformer:
        """Create a Transformer instance from configuration.
        
        Args:
            **config: Transformer configuration
            
        Returns:
            Transformer instance
        """
        logger.info("Creating transformer")

        transformer = Transformer()

        # Add rules
        rules = config.get("rules", [])
        for rule_config in rules:
            self._add_rule(transformer, rule_config)

        return transformer

    def _add_rule(self, transformer: Transformer, rule_config: dict[str, Any]) -> None:
        """Add a rule to the transformer.
        
        Args:
            transformer: Transformer to add rule to
            rule_config: Rule configuration
        """
        rule_type = rule_config.get("type", "").lower()

        if rule_type == "map":
            source = rule_config.get("source")
            target = rule_config.get("target")
            if source:
                transformer.map(source, target)

        elif rule_type == "rename":
            old_name = rule_config.get("old_name")
            new_name = rule_config.get("new_name")
            if old_name and new_name:
                transformer.rename(old_name, new_name)

        elif rule_type == "exclude":
            fields = rule_config.get("fields", [])
            if fields:
                transformer.exclude(*fields)

        elif rule_type == "add":
            field_name = rule_config.get("field_name")
            value = rule_config.get("value")
            if field_name is not None:
                transformer.add(field_name, value)

        else:
            logger.warning(f"Unknown rule type: {rule_type}")


class MigratorFactory(FactoryBase):
    """Factory for creating migrators.
    
    The Migrator doesn't require configuration, but this factory
    provides a consistent interface for the config system.
    """

    def create(self, **config) -> Migrator:
        """Create a Migrator instance.
        
        Args:
            **config: Currently unused
            
        Returns:
            Migrator instance
        """
        logger.info("Creating migrator")
        return Migrator()


# Create singleton instances for registration
migration_factory = MigrationFactory()
transformer_factory = TransformerFactory()
migrator_factory = MigratorFactory()

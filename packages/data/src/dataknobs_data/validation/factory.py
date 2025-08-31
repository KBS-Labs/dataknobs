"""Factory classes for validation v2 components."""

import logging
from typing import Any

from dataknobs_config import FactoryBase

from .coercer import Coercer
from .constraints import (
    All,
    AnyOf,
    Constraint,
    Enum,
    Length,
    Pattern,
    Range,
    Required,
    Unique,
)
from .schema import Schema

logger = logging.getLogger(__name__)


class SchemaFactory(FactoryBase):
    """Factory for creating validation schemas from configuration.
    
    Configuration Options:
        name (str): Schema name
        strict (bool): Whether to reject unknown fields (default: False)
        description (str): Optional schema description
        fields (list): List of field definitions
        
    Field Definition Options:
        name (str): Field name
        type (str): Field type (STRING, INTEGER, FLOAT, BOOLEAN, DATETIME, JSON, BINARY)
        required (bool): Whether field is required (default: False)
        default (any): Default value if field is missing
        description (str): Field description
        constraints (list): List of constraint definitions
        
    Example Configuration:
        schemas:
          - name: user_schema
            factory: schema
            strict: true
            description: User registration schema
            fields:
              - name: username
                type: STRING
                required: true
                constraints:
                  - type: length
                    min: 3
                    max: 20
                  - type: pattern
                    pattern: "^[a-zA-Z0-9_]+$"
              - name: age
                type: INTEGER
                constraints:
                  - type: range
                    min: 13
                    max: 120
    """

    def create(self, **config) -> Schema:
        """Create a Schema instance from configuration.
        
        Args:
            **config: Schema configuration
            
        Returns:
            Schema instance
        """
        name = config.get("name", "unnamed_schema")
        strict = config.get("strict", False)
        description = config.get("description")

        logger.info(f"Creating schema: {name}")

        schema = Schema(name, strict)
        if description:
            schema.with_description(description)

        # Add fields
        fields = config.get("fields", [])
        for field_config in fields:
            self._add_field_to_schema(schema, field_config)

        return schema

    def _add_field_to_schema(self, schema: Schema, field_config: dict[str, Any]) -> None:
        """Add a field to the schema based on configuration.
        
        Args:
            schema: Schema to add field to
            field_config: Field configuration
        """
        field_name = field_config.get("name")
        if not field_name:
            logger.warning("Field configuration missing 'name', skipping")
            return

        field_type = field_config.get("type", "STRING")
        required = field_config.get("required", False)
        default = field_config.get("default")
        description = field_config.get("description")

        # Build constraints
        constraints = self._build_constraints(field_config.get("constraints", []))

        schema.field(
            name=field_name,
            field_type=field_type,
            required=required,
            default=default,
            constraints=constraints,
            description=description
        )

    def _build_constraints(self, constraint_configs: list[dict[str, Any]]) -> list[Constraint]:
        """Build constraint objects from configuration.
        
        Args:
            constraint_configs: List of constraint configurations
            
        Returns:
            List of Constraint objects
        """
        constraints: list[Constraint] = []

        for config in constraint_configs:
            constraint_type = config.get("type", "").lower()

            if constraint_type == "required":
                constraints.append(Required(
                    allow_empty=config.get("allow_empty", False)
                ))

            elif constraint_type == "range":
                constraints.append(Range(
                    min=config.get("min"),
                    max=config.get("max")
                ))

            elif constraint_type == "length":
                constraints.append(Length(
                    min=config.get("min"),
                    max=config.get("max")
                ))

            elif constraint_type == "pattern":
                pattern = config.get("pattern")
                if pattern:
                    constraints.append(Pattern(pattern))

            elif constraint_type == "enum":
                values = config.get("values", [])
                if values:
                    constraints.append(Enum(values))

            elif constraint_type == "unique":
                constraints.append(Unique(
                    field_name=config.get("field_name")
                ))

            elif constraint_type == "all":
                # Recursive build for composite constraints
                sub_constraints = self._build_constraints(config.get("constraints", []))
                if sub_constraints:
                    constraints.append(All(sub_constraints))

            elif constraint_type == "any":
                # Recursive build for composite constraints
                sub_constraints = self._build_constraints(config.get("constraints", []))
                if sub_constraints:
                    constraints.append(AnyOf(sub_constraints))

            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")

        return constraints


class CoercerFactory(FactoryBase):
    """Factory for creating Coercer instances.
    
    The Coercer doesn't require configuration, but this factory
    provides a consistent interface for the config system.
    """

    def create(self, **config) -> Coercer:
        """Create a Coercer instance.
        
        Args:
            **config: Currently unused
            
        Returns:
            Coercer instance
        """
        logger.info("Creating Coercer")
        return Coercer()


# Create singleton instances for registration
schema_factory = SchemaFactory()
coercer_factory = CoercerFactory()

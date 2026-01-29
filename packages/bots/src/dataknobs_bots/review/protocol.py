"""Review protocol definitions for artifact validation.

This module provides configuration-defined review protocols that specify
how artifacts should be validated. Protocols can use:

- Persona-based reviews (LLM adopts a perspective)
- Schema validation (JSON Schema)
- Custom validation functions

Example:
    >>> protocol = ReviewProtocolDefinition.from_config(
    ...     "adversarial",
    ...     {"persona": "adversarial", "score_threshold": 0.8}
    ... )
    >>> protocol.type
    'persona'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .personas import ReviewPersona

ReviewType = Literal["persona", "schema", "custom"]


@dataclass
class ReviewProtocolDefinition:
    """Configuration-defined review protocol.

    Protocols define how artifacts should be reviewed. They can use:
    - Built-in or custom personas for LLM-based review
    - JSON Schema for structural validation
    - Custom functions for programmatic validation

    Attributes:
        id: Protocol identifier
        type: Type of review (persona, schema, custom)
        persona_id: ID of persona to use (for persona type)
        persona: Inline persona definition (for custom personas)
        schema: JSON Schema for validation (for schema type)
        schema_ref: Reference to external schema
        function_ref: Function reference for custom validation
        score_threshold: Minimum score to pass (0.0-1.0)
        required: Whether this review is required for approval
        enabled: Whether this protocol is active
        metadata: Additional protocol metadata
    """

    id: str
    type: ReviewType = "persona"
    persona_id: str | None = None
    persona: ReviewPersona | None = None
    schema: dict[str, Any] | None = None
    schema_ref: str | None = None
    function_ref: str | None = None
    score_threshold: float = 0.7
    required: bool = True
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        protocol_id: str,
        config: dict[str, Any],
    ) -> ReviewProtocolDefinition:
        """Create from configuration dict.

        The configuration can specify:
        - persona: Reference to a built-in persona ID
        - prompt_template: Custom inline persona definition
        - type: "schema" or "custom" for non-persona reviews
        - schema: JSON Schema for schema validation
        - function_ref: Function reference for custom validation

        Args:
            protocol_id: The protocol ID
            config: Configuration dictionary

        Returns:
            ReviewProtocolDefinition instance
        """
        # Handle persona reference vs inline persona
        persona_id = config.get("persona")
        persona = None

        # Check if it's a custom inline persona (has prompt_template)
        if config.get("prompt_template"):
            persona = ReviewPersona(
                id=protocol_id,
                name=config.get("name", protocol_id),
                focus=config.get("focus", ""),
                prompt_template=config["prompt_template"],
                scoring_criteria=config.get("scoring_criteria"),
                default_score_threshold=config.get("score_threshold", 0.7),
            )
            persona_id = None

        # Determine review type
        review_type: ReviewType = config.get("type", "persona")
        if config.get("schema") or config.get("schema_ref"):
            review_type = "schema"
        elif config.get("function_ref"):
            review_type = "custom"

        return cls(
            id=protocol_id,
            type=review_type,
            persona_id=persona_id,
            persona=persona,
            schema=config.get("schema"),
            schema_ref=config.get("schema_ref"),
            function_ref=config.get("function_ref"),
            score_threshold=config.get("score_threshold", 0.7),
            required=config.get("required", True),
            enabled=config.get("enabled", True),
            metadata=config.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "score_threshold": self.score_threshold,
            "required": self.required,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

        if self.persona_id:
            result["persona_id"] = self.persona_id
        if self.persona:
            result["persona"] = self.persona.to_dict()
        if self.schema:
            result["schema"] = self.schema
        if self.schema_ref:
            result["schema_ref"] = self.schema_ref
        if self.function_ref:
            result["function_ref"] = self.function_ref

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewProtocolDefinition:
        """Deserialize from dictionary."""
        persona = None
        if data.get("persona"):
            persona = ReviewPersona.from_dict(data["persona"])

        return cls(
            id=data["id"],
            type=data.get("type", "persona"),
            persona_id=data.get("persona_id"),
            persona=persona,
            schema=data.get("schema"),
            schema_ref=data.get("schema_ref"),
            function_ref=data.get("function_ref"),
            score_threshold=data.get("score_threshold", 0.7),
            required=data.get("required", True),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )

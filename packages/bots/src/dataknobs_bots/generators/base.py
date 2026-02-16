"""Generator abstraction for producing structured content from parameters.

This module provides:
- GeneratorContext: Dependencies available during generation (db, llm, etc.)
- GeneratorOutput: Result of generation with content, provenance, and validation
- Generator: Abstract base class defining the generator interface

Generators produce structured content deterministically from parameters.
Output is validated against JSON Schema and tracked via provenance.

Example:
    >>> class MyGenerator(Generator):
    ...     @property
    ...     def id(self) -> str: return "my_gen"
    ...     @property
    ...     def version(self) -> str: return "1.0.0"
    ...     ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dataknobs_bots.artifacts.provenance import ProvenanceRecord

logger = logging.getLogger(__name__)


@dataclass
class GeneratorContext:
    """Dependencies available during content generation.

    Provides optional access to data stores, LLM providers, and other
    services that generators may need for lookups or narrow NL tasks.

    Attributes:
        db: Async database for data lookups.
        llm: LLM provider for narrow NL tasks only.
        vector_store: Vector store for content retrieval.
        artifact_registry: Reference to artifact registry (avoids circular import).
        config: Additional configuration parameters.
    """

    db: Any | None = None
    llm: Any | None = None
    vector_store: Any | None = None
    artifact_registry: Any | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorOutput:
    """Result of a generation operation.

    Attributes:
        content: The generated content as a dictionary.
        provenance: How this content was generated.
        validation_errors: Any validation issues found in the output.
        metadata: Additional output metadata.
    """

    content: dict[str, Any]
    provenance: ProvenanceRecord
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "content": self.content,
            "provenance": self.provenance.to_dict(),
            "validation_errors": self.validation_errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratorOutput:
        """Deserialize from a dictionary."""
        return cls(
            content=data.get("content", {}),
            provenance=ProvenanceRecord.from_dict(data.get("provenance", {})),
            validation_errors=data.get("validation_errors", []),
            metadata=data.get("metadata", {}),
        )


class Generator(ABC):
    """Abstract base class for content generators.

    Generators produce structured content from parameters. Each generator
    defines JSON Schemas for its input parameters and output format,
    enabling validation at both ends.

    Subclasses must implement:
    - ``id``: Unique generator identifier
    - ``version``: Semantic version string
    - ``parameter_schema``: JSON Schema for input parameters
    - ``output_schema``: JSON Schema for output content
    - ``generate()``: The generation logic
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique generator identifier."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string."""

    @property
    @abstractmethod
    def parameter_schema(self) -> dict[str, Any]:
        """JSON Schema for generator input parameters."""

    @property
    @abstractmethod
    def output_schema(self) -> dict[str, Any]:
        """JSON Schema for generator output content."""

    @abstractmethod
    async def generate(
        self,
        parameters: dict[str, Any],
        context: GeneratorContext | None = None,
    ) -> GeneratorOutput:
        """Generate content from parameters.

        Args:
            parameters: Input parameters matching ``parameter_schema``.
            context: Optional dependencies for generation.

        Returns:
            GeneratorOutput with content, provenance, and validation results.
        """

    async def validate_parameters(
        self, parameters: dict[str, Any]
    ) -> list[str]:
        """Validate parameters against the parameter schema.

        Args:
            parameters: Input parameters to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        import jsonschema

        errors: list[str] = []
        try:
            jsonschema.validate(parameters, self.parameter_schema)
        except jsonschema.ValidationError as e:
            errors.append(str(e.message))
        except jsonschema.SchemaError as e:
            errors.append(f"Invalid parameter schema: {e.message}")
        return errors

    async def validate_output(
        self, output: dict[str, Any]
    ) -> list[str]:
        """Validate output content against the output schema.

        Args:
            output: Generated content to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        import jsonschema

        errors: list[str] = []
        try:
            jsonschema.validate(output, self.output_schema)
        except jsonschema.ValidationError as e:
            errors.append(str(e.message))
        except jsonschema.SchemaError as e:
            errors.append(f"Invalid output schema: {e.message}")
        return errors

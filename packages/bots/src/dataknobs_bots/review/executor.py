"""Review executor for running review protocols against artifacts.

The executor handles:
- Persona-based reviews (LLM invocation)
- Schema validation (JSON Schema)
- Custom validation (function calls)
- Runtime persona and protocol registration

Example:
    >>> executor = ReviewExecutor(
    ...     llm=llm_provider,
    ...     protocols=protocols_from_config,
    ... )
    >>>
    >>> # Run single review
    >>> review = await executor.run_review(artifact, protocol_id="adversarial")
    >>>
    >>> # Run all configured reviews for artifact type
    >>> reviews = await executor.run_artifact_reviews(artifact, definition)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from ..artifacts.models import Artifact, ArtifactReview
from .personas import BUILT_IN_PERSONAS, ReviewPersona
from .protocol import ReviewProtocolDefinition

if TYPE_CHECKING:
    from ..artifacts.models import ArtifactDefinition

logger = logging.getLogger(__name__)

# Type alias for custom validation functions
CustomValidator = Callable[[Artifact], dict[str, Any] | Any]


class ReviewExecutor:
    """Executes review protocols against artifacts.

    The executor handles:
    - Persona-based reviews (LLM invocation)
    - Schema validation (JSON Schema)
    - Custom validation (function calls)

    Attributes:
        _llm: LLM provider for persona reviews
        _protocols: Registered protocol definitions
        _personas: Available personas (built-in + custom)
        _custom_functions: Custom validation functions
    """

    def __init__(
        self,
        llm: Any | None = None,
        protocols: dict[str, ReviewProtocolDefinition] | None = None,
        custom_personas: dict[str, ReviewPersona] | None = None,
        custom_functions: dict[str, CustomValidator] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            llm: LLM provider for persona reviews
            protocols: Protocol definitions from configuration
            custom_personas: Additional personas beyond built-ins
            custom_functions: Custom validation functions
        """
        self._llm = llm
        self._protocols = protocols or {}
        self._personas: dict[str, ReviewPersona] = {**BUILT_IN_PERSONAS}
        if custom_personas:
            self._personas.update(custom_personas)
        self._custom_functions: dict[str, CustomValidator] = custom_functions or {}

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        llm: Any | None = None,
    ) -> ReviewExecutor:
        """Create executor from bot configuration.

        Args:
            config: Bot configuration with 'review_protocols' section
            llm: LLM provider for persona reviews

        Returns:
            Configured ReviewExecutor
        """
        protocols: dict[str, ReviewProtocolDefinition] = {}
        for proto_id, proto_config in config.get("review_protocols", {}).items():
            protocols[proto_id] = ReviewProtocolDefinition.from_config(
                proto_id, proto_config
            )

        # Custom functions must be registered separately via register_function
        return cls(
            llm=llm,
            protocols=protocols,
        )

    # =========================================================================
    # Runtime Registration
    # =========================================================================

    def register_persona(self, persona: ReviewPersona) -> None:
        """Register a persona at runtime.

        Allows adding personas dynamically beyond what's in configuration.
        Useful for context-specific reviewers or user-defined perspectives.

        Args:
            persona: ReviewPersona to register
        """
        self._personas[persona.id] = persona
        logger.debug("Registered persona: %s", persona.id)

    def register_protocol(self, protocol: ReviewProtocolDefinition) -> None:
        """Register a protocol at runtime.

        Allows adding protocols dynamically beyond configuration.

        Args:
            protocol: ReviewProtocolDefinition to register
        """
        self._protocols[protocol.id] = protocol
        logger.debug("Registered protocol: %s", protocol.id)

    def register_function(
        self,
        protocol_id: str,
        func: CustomValidator,
    ) -> None:
        """Register a custom validation function.

        Args:
            protocol_id: Protocol ID this function handles
            func: Validation function
        """
        self._custom_functions[protocol_id] = func
        logger.debug("Registered custom function for: %s", protocol_id)

    def get_available_personas(self) -> list[str]:
        """Get IDs of all available personas."""
        return list(self._personas.keys())

    def get_available_protocols(self) -> list[str]:
        """Get IDs of all available protocols."""
        return list(self._protocols.keys())

    def get_persona(self, persona_id: str) -> ReviewPersona | None:
        """Get a persona by ID.

        Args:
            persona_id: ID of persona to retrieve

        Returns:
            ReviewPersona if found, None otherwise
        """
        return self._personas.get(persona_id)

    def get_protocol(self, protocol_id: str) -> ReviewProtocolDefinition | None:
        """Get a protocol by ID.

        Args:
            protocol_id: ID of protocol to retrieve

        Returns:
            ReviewProtocolDefinition if found, None otherwise
        """
        return self._protocols.get(protocol_id)

    # =========================================================================
    # Review Execution
    # =========================================================================

    async def run_review(
        self,
        artifact: Artifact,
        protocol_id: str,
    ) -> ArtifactReview:
        """Run a single review protocol against an artifact.

        Args:
            artifact: Artifact to review
            protocol_id: ID of protocol to use

        Returns:
            ArtifactReview with results

        Raises:
            KeyError: If protocol not found
            ValueError: If protocol type not supported
        """
        protocol = self._protocols.get(protocol_id)
        if not protocol:
            raise KeyError(f"Protocol not found: {protocol_id}")

        if not protocol.enabled:
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol_id,
                review_type=protocol.type,
                passed=True,
                score=1.0,
                feedback=["Review skipped - protocol disabled"],
            )

        if protocol.type == "persona":
            return await self._run_persona_review(artifact, protocol)
        elif protocol.type == "schema":
            return self._run_schema_review(artifact, protocol)
        elif protocol.type == "custom":
            return await self._run_custom_review(artifact, protocol)
        else:
            raise ValueError(f"Unknown protocol type: {protocol.type}")

    async def _run_persona_review(
        self,
        artifact: Artifact,
        protocol: ReviewProtocolDefinition,
    ) -> ArtifactReview:
        """Run persona-based review using LLM.

        Args:
            artifact: Artifact to review
            protocol: Protocol definition

        Returns:
            ArtifactReview with LLM evaluation results
        """
        if not self._llm:
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="persona",
                passed=False,
                score=0.0,
                issues=["No LLM provider configured for persona reviews"],
            )

        # Get persona
        persona = protocol.persona
        if not persona and protocol.persona_id:
            persona = self._personas.get(protocol.persona_id)
        if not persona:
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="persona",
                passed=False,
                score=0.0,
                issues=[f"Persona not found for protocol: {protocol.id}"],
            )

        # Build prompt
        content_str = (
            json.dumps(artifact.content, indent=2)
            if isinstance(artifact.content, (dict, list))
            else str(artifact.content)
        )

        prompt = persona.prompt_template.format(
            artifact_type=artifact.type,
            artifact_name=artifact.name,
            artifact_purpose=artifact.metadata.purpose or "Not specified",
            artifact_content=content_str,
        )

        # Call LLM
        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # Parse response - handle different response formats
            content = response.content if hasattr(response, "content") else str(response)
            result = json.loads(content)

            score = result.get("score", 0.0)
            passed = result.get("passed", score >= protocol.score_threshold)

            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="persona",
                passed=passed,
                score=score,
                issues=result.get("issues", []),
                suggestions=result.get("suggestions", []),
                feedback=result.get("feedback", []),
                metadata={"persona": persona.id},
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="persona",
                passed=False,
                score=0.0,
                issues=[f"Invalid JSON response from LLM: {e!s}"],
            )
        except Exception as e:
            logger.error("Persona review failed: %s", e)
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="persona",
                passed=False,
                score=0.0,
                issues=[f"Review failed: {e!s}"],
            )

    def _run_schema_review(
        self,
        artifact: Artifact,
        protocol: ReviewProtocolDefinition,
    ) -> ArtifactReview:
        """Run schema validation review.

        Args:
            artifact: Artifact to validate
            protocol: Protocol definition with schema

        Returns:
            ArtifactReview with validation results
        """
        schema = protocol.schema
        if not schema and protocol.schema_ref:
            # Schema ref loading would be implemented based on schema storage
            logger.warning(
                "Schema ref loading not implemented: %s", protocol.schema_ref
            )
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="schema",
                passed=False,
                score=0.0,
                issues=[f"Schema ref loading not implemented: {protocol.schema_ref}"],
            )

        if not schema:
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="schema",
                passed=True,
                score=1.0,
                feedback=["No schema defined, validation skipped"],
            )

        try:
            import jsonschema

            jsonschema.validate(artifact.content, schema)
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="schema",
                passed=True,
                score=1.0,
                feedback=["Schema validation passed"],
            )
        except ImportError:
            logger.error("jsonschema package not installed")
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="schema",
                passed=False,
                score=0.0,
                issues=["jsonschema package not installed"],
            )
        except Exception as e:
            # Handle both ValidationError and other exceptions
            error_message = str(e)
            path_info = ""
            if hasattr(e, "message"):
                error_message = e.message  # type: ignore[attr-defined]
            if hasattr(e, "absolute_path"):
                path_info = ".".join(str(p) for p in e.absolute_path)  # type: ignore[attr-defined]

            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="schema",
                passed=False,
                score=0.0,
                issues=[f"Schema validation error: {error_message}"],
                feedback=[f"Path: {path_info}"] if path_info else [],
            )

    async def _run_custom_review(
        self,
        artifact: Artifact,
        protocol: ReviewProtocolDefinition,
    ) -> ArtifactReview:
        """Run custom validation function.

        Args:
            artifact: Artifact to validate
            protocol: Protocol definition with function reference

        Returns:
            ArtifactReview with function results
        """
        func = self._custom_functions.get(protocol.id)
        if not func:
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="custom",
                passed=False,
                score=0.0,
                issues=[f"Custom function not registered for: {protocol.id}"],
            )

        try:
            result = func(artifact)
            if asyncio.iscoroutine(result):
                result = await result

            # Result should be dict with passed, score, issues, etc.
            if isinstance(result, dict):
                return ArtifactReview(
                    artifact_id=artifact.id,
                    reviewer=protocol.id,
                    review_type="custom",
                    passed=result.get("passed", False),
                    score=result.get("score", 0.0),
                    issues=result.get("issues", []),
                    suggestions=result.get("suggestions", []),
                    feedback=result.get("feedback", []),
                )
            else:
                # Treat as boolean pass/fail
                return ArtifactReview(
                    artifact_id=artifact.id,
                    reviewer=protocol.id,
                    review_type="custom",
                    passed=bool(result),
                    score=1.0 if result else 0.0,
                )
        except Exception as e:
            logger.error("Custom review failed: %s", e)
            return ArtifactReview(
                artifact_id=artifact.id,
                reviewer=protocol.id,
                review_type="custom",
                passed=False,
                score=0.0,
                issues=[f"Custom review failed: {e!s}"],
            )

    async def run_artifact_reviews(
        self,
        artifact: Artifact,
        artifact_definition: ArtifactDefinition | None = None,
    ) -> list[ArtifactReview]:
        """Run all configured reviews for an artifact.

        Uses the artifact's definition_id to look up which reviews apply,
        or uses the provided artifact_definition directly.

        Args:
            artifact: Artifact to review
            artifact_definition: Optional definition with review list

        Returns:
            List of ArtifactReview results
        """
        # Get review protocols from artifact definition
        protocol_ids: list[str] = []
        if artifact_definition:
            protocol_ids = artifact_definition.reviews

        if not protocol_ids:
            logger.info("No reviews configured for artifact: %s", artifact.id)
            return []

        reviews: list[ArtifactReview] = []
        for protocol_id in protocol_ids:
            if protocol_id in self._protocols:
                review = await self.run_review(artifact, protocol_id)
                reviews.append(review)
            else:
                logger.warning("Review protocol not found: %s", protocol_id)

        return reviews

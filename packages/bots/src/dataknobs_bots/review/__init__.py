"""Review system for validating artifacts.

This package provides infrastructure for reviewing and validating artifacts
produced during bot workflows. It supports:

- Persona-based reviews (LLM adopts a perspective to evaluate)
- Schema validation (JSON Schema)
- Custom validation functions

Built-in personas include:
- adversarial: Edge cases, failure modes, security concerns
- skeptical: Accuracy, correctness, claim verification
- insightful: Broader context, missed opportunities
- minimalist: Simplicity, unnecessary complexity
- downstream: Usability from consumer perspective

Example:
    >>> from dataknobs_bots.review import (
    ...     ReviewExecutor,
    ...     ReviewProtocolDefinition,
    ...     BUILT_IN_PERSONAS,
    ... )
    >>>
    >>> # Create executor with a protocol
    >>> protocols = {
    ...     "adversarial": ReviewProtocolDefinition.from_config(
    ...         "adversarial",
    ...         {"persona": "adversarial", "score_threshold": 0.8}
    ...     )
    ... }
    >>> executor = ReviewExecutor(llm=llm, protocols=protocols)
    >>>
    >>> # Run a review
    >>> review = await executor.run_review(artifact, "adversarial")
"""

from .executor import ReviewExecutor
from .personas import (
    BUILT_IN_PERSONAS,
    ReviewPersona,
    get_persona,
    list_personas,
)
from .protocol import ReviewProtocolDefinition, ReviewType
from .tools import (
    GetReviewResultsTool,
    ReviewArtifactTool,
    RunAllReviewsTool,
)

__all__ = [
    # Executor
    "ReviewExecutor",
    # Personas
    "ReviewPersona",
    "BUILT_IN_PERSONAS",
    "get_persona",
    "list_personas",
    # Protocol
    "ReviewProtocolDefinition",
    "ReviewType",
    # Tools
    "GetReviewResultsTool",
    "ReviewArtifactTool",
    "RunAllReviewsTool",
]

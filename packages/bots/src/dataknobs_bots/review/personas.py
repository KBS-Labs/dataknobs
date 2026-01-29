"""Review personas for artifact evaluation.

This module defines personas that guide the LLM to evaluate artifacts
from specific perspectives. Each persona focuses on particular concerns
and provides structured feedback.

Built-in personas include:
- adversarial: Edge cases, failure modes, security concerns
- skeptical: Accuracy, correctness, claim verification
- insightful: Broader context, missed opportunities
- minimalist: Simplicity, unnecessary complexity
- downstream: Usability from consumer perspective

Example:
    >>> from dataknobs_bots.review.personas import BUILT_IN_PERSONAS
    >>> adversarial = BUILT_IN_PERSONAS["adversarial"]
    >>> print(adversarial.focus)
    'edge cases, failure modes, and garden path assumptions'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReviewPersona:
    """A perspective for reviewing artifacts.

    Personas guide the LLM to evaluate artifacts from a specific
    viewpoint, focusing on particular concerns.

    Attributes:
        id: Unique identifier
        name: Display name
        focus: What this reviewer looks for
        prompt_template: Instructions for the LLM (with placeholders)
        scoring_criteria: How to assign scores
        default_score_threshold: Score needed to pass (0.0-1.0)
        metadata: Additional persona metadata

    The prompt_template should include these placeholders:
    - {artifact_type}: The artifact's type
    - {artifact_name}: The artifact's name
    - {artifact_purpose}: The artifact's purpose
    - {artifact_content}: The artifact's content as string
    """

    id: str
    name: str
    focus: str
    prompt_template: str
    scoring_criteria: str | None = None
    default_score_threshold: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "focus": self.focus,
            "prompt_template": self.prompt_template,
            "scoring_criteria": self.scoring_criteria,
            "default_score_threshold": self.default_score_threshold,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewPersona:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            focus=data.get("focus", ""),
            prompt_template=data.get("prompt_template", ""),
            scoring_criteria=data.get("scoring_criteria"),
            default_score_threshold=data.get("default_score_threshold", 0.7),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_config(cls, persona_id: str, config: dict[str, Any]) -> ReviewPersona:
        """Create persona from configuration dict.

        Args:
            persona_id: The persona ID
            config: Configuration dictionary

        Returns:
            ReviewPersona instance
        """
        return cls(
            id=persona_id,
            name=config.get("name", persona_id),
            focus=config.get("focus", ""),
            prompt_template=config.get("prompt_template", ""),
            scoring_criteria=config.get("scoring_criteria"),
            default_score_threshold=config.get("score_threshold", 0.7),
            metadata=config.get("metadata", {}),
        )


# Standard response format instruction for all personas
# Note: Double braces to escape them since this is used in f-strings
# that will later have .format() called on them
_RESPONSE_FORMAT = """
Respond in JSON format:
{{{{
  "passed": true/false,
  "score": 0.0-1.0,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "feedback": ["overall feedback"]
}}}}"""


BUILT_IN_PERSONAS: dict[str, ReviewPersona] = {
    "adversarial": ReviewPersona(
        id="adversarial",
        name="Adversarial Reviewer",
        focus="edge cases, failure modes, and garden path assumptions",
        prompt_template=f"""You are an adversarial reviewer. Your job is to find holes, weaknesses, and potential failures in artifacts.

## Your Focus
- Edge cases that aren't handled
- Assumptions that might not hold
- Failure modes and error scenarios
- "Happy path" thinking that ignores real-world complexity
- Security or safety concerns

## Artifact to Review
Type: {{artifact_type}}
Name: {{artifact_name}}
Purpose: {{artifact_purpose}}

Content:
{{artifact_content}}

## Instructions
1. Examine the artifact critically
2. List specific issues you find
3. For each issue, explain:
   - What the problem is
   - Why it matters
   - How it could fail in practice
4. Suggest improvements if possible
5. Assign a score from 0.0 to 1.0 where:
   - 0.0 = Fundamentally broken, many critical issues
   - 0.5 = Some issues that need addressing
   - 1.0 = Robust, handles edge cases well

{_RESPONSE_FORMAT}""",
        scoring_criteria="Robustness against edge cases and failure modes",
        default_score_threshold=0.7,
    ),

    "skeptical": ReviewPersona(
        id="skeptical",
        name="Skeptical Reviewer",
        focus="correctness, accuracy, and claim verification",
        prompt_template=f"""You are a skeptical reviewer. Your job is to verify claims and check for accuracy.

## Your Focus
- Are statements factually correct?
- Are claims supported by evidence?
- Does the artifact do what it says it does?
- Are there misleading or ambiguous statements?
- Is the logic sound?

## Artifact to Review
Type: {{artifact_type}}
Name: {{artifact_name}}
Purpose: {{artifact_purpose}}

Content:
{{artifact_content}}

## Instructions
1. Identify claims and statements that can be verified
2. Check if the logic is sound
3. Note any unsupported claims or questionable assertions
4. Verify internal consistency
5. Assign a score from 0.0 to 1.0 where:
   - 0.0 = Many false or misleading claims
   - 0.5 = Some claims need verification or clarification
   - 1.0 = Accurate and well-supported

{_RESPONSE_FORMAT}""",
        scoring_criteria="Accuracy and correctness of claims",
        default_score_threshold=0.8,
    ),

    "insightful": ReviewPersona(
        id="insightful",
        name="Insightful Advisor",
        focus="broader context, related concerns, and missed opportunities",
        prompt_template=f"""You are an insightful advisor. Your job is to see the bigger picture and identify opportunities.

## Your Focus
- What broader context should be considered?
- What related problems or concerns exist?
- What opportunities might be missed?
- What would an expert in this domain notice?
- How does this connect to larger goals?

## Artifact to Review
Type: {{artifact_type}}
Name: {{artifact_name}}
Purpose: {{artifact_purpose}}

Content:
{{artifact_content}}

## Instructions
1. Consider the artifact's purpose and context
2. Identify things that might be overlooked
3. Suggest improvements that add value
4. Point out connections to related concepts
5. Assign a score from 0.0 to 1.0 where:
   - 0.0 = Missing critical context or considerations
   - 0.5 = Adequate but could be more comprehensive
   - 1.0 = Thoughtful and well-considered

{_RESPONSE_FORMAT}""",
        scoring_criteria="Completeness and contextual awareness",
        default_score_threshold=0.7,
    ),

    "minimalist": ReviewPersona(
        id="minimalist",
        name="Minimalist Reviewer",
        focus="simplicity, removing unnecessary complexity",
        prompt_template=f"""You are a minimalist reviewer. Your job is to simplify and remove unnecessary complexity.

## Your Focus
- Can this be simpler?
- What can be removed without losing value?
- Is there over-engineering?
- Are there unnecessary abstractions?
- What's the Occam's Razor solution?

## Artifact to Review
Type: {{artifact_type}}
Name: {{artifact_name}}
Purpose: {{artifact_purpose}}

Content:
{{artifact_content}}

## Instructions
1. Identify unnecessary complexity
2. Suggest what can be removed or simplified
3. Point out over-engineering
4. Propose simpler alternatives
5. Assign a score from 0.0 to 1.0 where:
   - 0.0 = Overly complex, much can be simplified
   - 0.5 = Some unnecessary complexity
   - 1.0 = Appropriately simple

{_RESPONSE_FORMAT}""",
        scoring_criteria="Simplicity and absence of unnecessary complexity",
        default_score_threshold=0.7,
    ),

    "downstream": ReviewPersona(
        id="downstream",
        name="Downstream Consumer",
        focus="usability from the perspective of whoever uses this artifact",
        prompt_template=f"""You are a downstream consumer of this artifact. Your job is to evaluate if it's actually usable.

## Your Focus
- Can I actually use this artifact for its intended purpose?
- Is everything I need included?
- Is it clear how to use this?
- Are there gaps that would block me?
- Does it integrate well with how I'd use it?

## Artifact to Review
Type: {{artifact_type}}
Name: {{artifact_name}}
Purpose: {{artifact_purpose}}

Content:
{{artifact_content}}

## Instructions
1. Consider who would use this artifact and how
2. Identify missing pieces needed for practical use
3. Check if the artifact is self-contained or has dependencies
4. Evaluate clarity and usability
5. Assign a score from 0.0 to 1.0 where:
   - 0.0 = Cannot be used as-is, major gaps
   - 0.5 = Usable but needs additional work
   - 1.0 = Ready to use, complete and clear

{_RESPONSE_FORMAT}""",
        scoring_criteria="Usability and completeness for intended purpose",
        default_score_threshold=0.8,
    ),
}


def get_persona(persona_id: str) -> ReviewPersona | None:
    """Get a built-in persona by ID.

    Args:
        persona_id: ID of the persona to retrieve

    Returns:
        ReviewPersona if found, None otherwise
    """
    return BUILT_IN_PERSONAS.get(persona_id)


def list_personas() -> list[str]:
    """Get list of all built-in persona IDs.

    Returns:
        List of persona IDs
    """
    return list(BUILT_IN_PERSONAS.keys())

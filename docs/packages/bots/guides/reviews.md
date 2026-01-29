# Review System

The review system provides infrastructure for validating artifacts through configurable review protocols. It supports persona-based LLM reviews, schema validation, and custom validation functions.

## Overview

Different artifacts need different review approaches:

| Artifact Type | Typical Reviews |
|--------------|-----------------|
| Planning docs | skeptical, minimalist |
| Content | insightful, downstream |
| Rubrics | adversarial |
| Configurations | validation (schema) |
| Test cases | adversarial, downstream |

Reviews can be:
- **LLM-based** - A persona evaluates the artifact from a specific perspective
- **Schema-based** - JSON Schema validation
- **Custom** - Your own validation functions

## Quick Start

```python
from dataknobs_bots.review import (
    ReviewExecutor,
    ReviewProtocolDefinition,
    BUILT_IN_PERSONAS,
)
from dataknobs_bots.artifacts import Artifact

# Create executor with protocols
protocols = {
    "adversarial": ReviewProtocolDefinition.from_config(
        "adversarial",
        {"persona": "adversarial", "score_threshold": 0.7}
    ),
    "validation": ReviewProtocolDefinition.from_config(
        "validation",
        {"type": "schema", "schema": my_schema}
    ),
}

executor = ReviewExecutor(llm=llm_provider, protocols=protocols)

# Run a review
review = await executor.run_review(artifact, "adversarial")
print(f"Passed: {review.passed}, Score: {review.score}")
```

## Built-in Personas

DataKnobs includes five built-in review personas:

### Adversarial

Focuses on edge cases, failure modes, and security concerns.

```python
from dataknobs_bots.review import get_persona

adversarial = get_persona("adversarial")
print(adversarial.focus)
# "edge cases, failure modes, and garden path assumptions"
```

**Good for:** Rubrics, test cases, security-sensitive configs

### Skeptical

Focuses on accuracy, correctness, and claim verification.

```python
skeptical = get_persona("skeptical")
print(skeptical.focus)
# "correctness, accuracy, and claim verification"
```

**Good for:** Documentation, factual content, technical specifications

### Insightful

Focuses on broader context and missed opportunities.

```python
insightful = get_persona("insightful")
print(insightful.focus)
# "broader context, related concerns, and missed opportunities"
```

**Good for:** Planning documents, feature designs, content

### Minimalist

Focuses on simplicity and removing unnecessary complexity.

```python
minimalist = get_persona("minimalist")
print(minimalist.focus)
# "simplicity, removing unnecessary complexity"
```

**Good for:** Code, configurations, over-engineered designs

### Downstream

Focuses on usability from the consumer perspective.

```python
downstream = get_persona("downstream")
print(downstream.focus)
# "usability from the perspective of whoever uses this artifact"
```

**Good for:** APIs, documentation, user-facing content

## Review Protocols

Review protocols define how reviews are executed:

```python
from dataknobs_bots.review import ReviewProtocolDefinition, ReviewType

# Persona-based protocol
persona_protocol = ReviewProtocolDefinition(
    id="adversarial_review",
    type="persona",
    persona_id="adversarial",
    score_threshold=0.7,
)

# Schema validation protocol
schema_protocol = ReviewProtocolDefinition(
    id="config_validation",
    type="schema",
    schema={
        "type": "object",
        "required": ["name", "llm"],
        "properties": {
            "name": {"type": "string"},
            "llm": {"type": "object"},
        }
    },
)

# Custom function protocol
custom_protocol = ReviewProtocolDefinition(
    id="custom_check",
    type="custom",
    function_ref="myapp.validators:custom_check",
)
```

### Protocol Types

| Type | Description |
|------|-------------|
| `persona` | LLM adopts a persona to evaluate |
| `schema` | JSON Schema validation |
| `custom` | Custom validation function |

## ReviewExecutor

The `ReviewExecutor` runs reviews against artifacts:

### Creating an Executor

```python
from dataknobs_bots.review import ReviewExecutor

# From configuration
executor = ReviewExecutor.from_config(
    config={
        "review_protocols": {
            "adversarial": {"persona": "adversarial", "score_threshold": 0.7},
            "validation": {"type": "schema", "schema": {...}},
        }
    },
    llm=llm_provider,
)

# Or manually
executor = ReviewExecutor(
    llm=llm_provider,
    protocols=my_protocols,
    custom_personas=my_custom_personas,
)
```

### Running Reviews

```python
# Single review
review = await executor.run_review(artifact, "adversarial")

# All configured reviews for an artifact type
reviews = await executor.run_artifact_reviews(
    artifact,
    artifact_definition,  # Has list of review protocol IDs
)
```

### Registering Custom Components

```python
# Register a custom persona at runtime
from dataknobs_bots.review import ReviewPersona

educator = ReviewPersona(
    id="educator",
    name="Educator Reviewer",
    focus="pedagogical effectiveness and learning outcomes",
    prompt_template="You are an educator reviewing...",
    default_score_threshold=0.8,
)
executor.register_persona(educator)

# Register a custom validation function
def validate_questions(artifact):
    """Custom validation for question artifacts."""
    content = artifact.content
    issues = []

    if not content.get("questions"):
        issues.append("No questions provided")

    for i, q in enumerate(content.get("questions", [])):
        if len(q) < 10:
            issues.append(f"Question {i+1} is too short")

    return {
        "passed": len(issues) == 0,
        "score": 1.0 - (len(issues) * 0.2),
        "issues": issues,
    }

executor.register_function("question_validation", validate_questions)
```

## Review Results

Reviews return an `ArtifactReview`:

```python
from dataknobs_bots.artifacts import ArtifactReview

review = await executor.run_review(artifact, "adversarial")

print(f"Passed: {review.passed}")
print(f"Score: {review.score}")  # 0.0 to 1.0
print(f"Reviewer: {review.reviewer}")
print(f"Type: {review.review_type}")  # persona, schema, custom
print(f"Issues: {review.issues}")
print(f"Suggestions: {review.suggestions}")
print(f"Feedback: {review.feedback}")
```

## Configuration

Configure reviews in your bot configuration:

```yaml
# bot_config.yaml
review_protocols:
  adversarial:
    type: persona
    persona: adversarial
    score_threshold: 0.7
    description: Check for edge cases and failure modes

  skeptical:
    type: persona
    persona: skeptical
    score_threshold: 0.8
    description: Verify accuracy and correctness

  config_validation:
    type: schema
    schema:
      type: object
      required:
        - name
        - llm
      properties:
        name:
          type: string
          minLength: 1
        llm:
          type: object
    description: Validate bot configuration structure

  custom_check:
    type: custom
    description: Custom validation logic
    enabled: true

# Link reviews to artifact definitions
artifacts:
  definitions:
    bot_config:
      type: config
      reviews:
        - config_validation
        - skeptical
      approval_threshold: 1.0
      require_all_reviews: true

    assessment_questions:
      type: content
      reviews:
        - adversarial
        - downstream
      approval_threshold: 0.8
```

## Custom Personas

Create custom personas for domain-specific reviews:

```python
from dataknobs_bots.review import ReviewPersona

educator_persona = ReviewPersona(
    id="educator",
    name="Educator Reviewer",
    focus="pedagogical effectiveness, learning outcomes, and age-appropriateness",
    prompt_template="""You are an experienced educator reviewing educational content.

## Your Focus
- Is this content pedagogically sound?
- Are learning objectives clear?
- Is it appropriate for the target audience?
- Does it follow best practices in education?

## Artifact to Review
Type: {artifact_type}
Name: {artifact_name}
Purpose: {artifact_purpose}

Content:
{artifact_content}

## Instructions
1. Evaluate the educational value
2. Check for learning objective alignment
3. Assess age-appropriateness
4. Identify areas for improvement
5. Score from 0.0 to 1.0

Respond in JSON format:
{{
  "passed": true/false,
  "score": 0.0-1.0,
  "issues": ["issue 1"],
  "suggestions": ["suggestion 1"],
  "feedback": ["overall feedback"]
}}""",
    scoring_criteria="Pedagogical effectiveness and learning outcomes",
    default_score_threshold=0.8,
)

# Register with executor
executor.register_persona(educator_persona)
```

## Integration with Artifacts

Reviews are automatically linked to artifacts:

```python
# Run review
review = await executor.run_review(artifact, "adversarial")

# Add to artifact registry
registry.add_review(artifact.id, review)

# Artifact status is updated based on definition thresholds
print(artifact.status)  # "approved" or "needs_revision"

# Access reviews from artifact
for review in artifact.reviews:
    print(f"{review.reviewer}: {review.passed}")
```

## Schema Validation

Use JSON Schema for structural validation:

```python
config_schema = {
    "type": "object",
    "required": ["name", "llm", "memory"],
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
        },
        "llm": {
            "type": "object",
            "required": ["provider", "model"],
        },
        "memory": {
            "type": "object",
            "properties": {
                "type": {"enum": ["buffer", "vector"]},
                "max_messages": {"type": "integer", "minimum": 1},
            },
        },
    },
}

protocol = ReviewProtocolDefinition(
    id="config_schema",
    type="schema",
    schema=config_schema,
)
```

## Review Tools

Built-in tools for LLM-driven review execution:

```python
from dataknobs_bots.review import (
    ReviewArtifactTool,
    RunAllReviewsTool,
    GetReviewResultsTool,
)

# Add to bot's tool registry
tools = [
    ReviewArtifactTool(),
    RunAllReviewsTool(),
    GetReviewResultsTool(),
]
```

## Best Practices

1. **Match personas to artifacts** - Use appropriate personas for each artifact type
2. **Set appropriate thresholds** - Critical artifacts need higher thresholds
3. **Combine review types** - Use both LLM and schema validation
4. **Custom personas for domains** - Create personas for your specific domain
5. **Review important artifacts** - Don't review everything, focus on critical outputs
6. **Handle failures gracefully** - Design workflows that respond to failed reviews

## Related Documentation

- [Artifact System](artifacts.md) - Managing artifacts
- [Context Accumulator](context.md) - Building context from reviews
- [Task Injection](task-injection.md) - Dynamic tasks on review events

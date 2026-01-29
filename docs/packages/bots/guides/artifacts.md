# Artifact System

The artifact system provides infrastructure for tracking work products (artifacts) produced during bot workflows. It enables bots to create, version, and manage documents, configurations, and other outputs throughout conversational interactions.

## Overview

Artifacts are first-class citizens in DataKnobs bots, representing any intermediate or final work product:

- **Planning documents** - Outlines, task breakdowns
- **Content** - Text documents, generated content
- **Data** - Structured data (JSON, YAML)
- **Configuration** - Bot configs, settings
- **Code** - Code snippets, scripts

Each artifact tracks its lifecycle status, version history, and review results.

## Quick Start

```python
from dataknobs_bots.artifacts import (
    ArtifactRegistry,
    ArtifactDefinition,
    Artifact,
)

# Create a registry
registry = ArtifactRegistry()

# Create an artifact
artifact = registry.create(
    content={"questions": ["What is 2+2?", "Explain photosynthesis"]},
    name="Assessment Questions",
    artifact_type="content",
    stage="build_questions",
    purpose="Questions for science assessment",
)

# Query artifacts
pending = registry.get_pending_review()
```

## Core Concepts

### Artifact

An `Artifact` represents a work product with:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (auto-generated) |
| `type` | Artifact type: `planning`, `content`, `data`, `config`, `code`, `composite` |
| `name` | Human-readable name |
| `content` | The actual data/content |
| `status` | Lifecycle status (see below) |
| `metadata` | Creation context (stage, task, purpose, tags) |
| `lineage` | Version tracking (parent, version number) |
| `reviews` | List of review results |

### Artifact Status

Artifacts progress through a lifecycle:

```
draft → pending_review → in_review → approved
                                   ↘ needs_revision → ...
                                   ↘ rejected
```

| Status | Description |
|--------|-------------|
| `draft` | Initial creation, work in progress |
| `pending_review` | Submitted for review |
| `in_review` | Review in progress |
| `needs_revision` | Review found issues |
| `approved` | Passed all required reviews |
| `rejected` | Failed review, won't proceed |
| `superseded` | Replaced by newer version |

### Artifact Types

```python
ArtifactType = Literal[
    "planning",    # Plans, outlines, task breakdowns
    "content",     # Text content, documents
    "data",        # Structured data (JSON, YAML)
    "config",      # Configuration files
    "code",        # Code snippets, scripts
    "composite",   # Artifact containing other artifacts
]
```

## ArtifactRegistry

The `ArtifactRegistry` manages artifacts within a conversation:

### Creating Artifacts

```python
registry = ArtifactRegistry()

# Basic creation
artifact = registry.create(
    content={"key": "value"},
    name="My Artifact",
    artifact_type="data",
)

# With full context
artifact = registry.create(
    content=document_content,
    name="User Guide",
    artifact_type="content",
    content_type="text/markdown",
    stage="documentation",
    task_id="write_guide",
    purpose="User documentation for the feature",
    tags=["documentation", "user-facing"],
)
```

### Using Definitions

Artifact definitions provide templates from configuration:

```python
# Create registry with definitions from config
registry = ArtifactRegistry.from_config({
    "artifacts": {
        "definitions": {
            "assessment_questions": {
                "type": "content",
                "name": "Assessment Questions",
                "reviews": ["adversarial", "downstream"],
                "approval_threshold": 0.8,
            }
        }
    }
})

# Create artifact using definition
artifact = registry.create(
    content=questions,
    definition_id="assessment_questions",  # Uses definition defaults
    stage="build_questions",
)
```

### Versioning

Updating an artifact creates a new version:

```python
# Original artifact
v1 = registry.create(content={"version": 1}, name="Config")

# Update creates v2
v2 = registry.update(
    artifact_id=v1.id,
    content={"version": 2},
    derived_from="Fixed validation issue",
)

assert v2.lineage.version == 2
assert v2.lineage.parent_id == v1.id
assert v1.status == "superseded"
```

### Querying

```python
# By ID
artifact = registry.get("art_abc123")

# By status
pending = registry.get_pending_review()
approved = registry.get_approved()

# By type
configs = registry.get_by_type("config")

# By definition
questions = registry.get_by_definition("assessment_questions")

# By stage
stage_artifacts = registry.get_by_stage("build_questions")

# All artifacts
all_artifacts = registry.get_all()
```

### Version Navigation

```python
# Get latest version of an artifact
latest = registry.get_latest_version(artifact_id)

# Get full version history
history = registry.get_version_history(artifact_id)
for version in history:
    print(f"v{version.lineage.version}: {version.status}")
```

### Status Transitions

```python
# Submit for review
registry.submit_for_review(artifact_id)

# Set status directly
registry.set_status(artifact_id, "approved")
```

## Artifact Metadata

Track context about how and why artifacts were created:

```python
from dataknobs_bots.artifacts import ArtifactMetadata

metadata = ArtifactMetadata(
    created_by="configbot",
    stage="configuration",
    task_id="generate_config",
    purpose="Bot configuration for math tutor",
    tags=["bot-config", "education"],
    custom={"domain": "mathematics"},
)
```

## Artifact Lineage

Track version history and relationships:

```python
from dataknobs_bots.artifacts import ArtifactLineage

lineage = ArtifactLineage(
    parent_id="art_abc123",      # Previous version
    source_ids=["art_xyz", "art_456"],  # Artifacts this was derived from
    version=2,
    derived_from="Merged from multiple sources",
)
```

## Configuration

Define artifact types in your bot configuration:

```yaml
# bot_config.yaml
artifacts:
  definitions:
    assessment_questions:
      type: content
      name: Assessment Questions
      description: Questions for assessment
      schema:
        type: object
        properties:
          questions:
            type: array
            items:
              type: string
      reviews:
        - adversarial
        - downstream
      approval_threshold: 0.8
      require_all_reviews: true
      auto_submit_for_review: true
      tags:
        - assessment
        - questions

    bot_config:
      type: config
      name: Bot Configuration
      schema_ref: schemas/bot_config.json
      reviews:
        - validation
      approval_threshold: 1.0  # Must pass
      require_all_reviews: true
```

## Lifecycle Hooks

Register callbacks for artifact events:

```python
registry = ArtifactRegistry()

# Called when artifact is created
@registry.on("on_create")
def handle_create(artifact):
    print(f"Created: {artifact.name}")

# Called when artifact is updated
@registry.on("on_update")
def handle_update(new_artifact, original):
    print(f"Updated {original.id} -> {new_artifact.id}")

# Called on status change
@registry.on("on_status_change")
def handle_status(artifact, old_status, new_status):
    print(f"{artifact.name}: {old_status} -> {new_status}")

# Called when review is added
@registry.on("on_review")
def handle_review(artifact, review):
    print(f"Review by {review.reviewer}: {'PASS' if review.passed else 'FAIL'}")
```

## Serialization

Save and restore registry state:

```python
# Save to dict
state = registry.to_dict()

# Restore from dict
restored = ArtifactRegistry.from_dict(
    state,
    definitions=my_definitions,
)
```

## Artifact Tools

Built-in tools for LLM-driven artifact management:

```python
from dataknobs_bots.artifacts import (
    CreateArtifactTool,
    UpdateArtifactTool,
    GetArtifactTool,
    QueryArtifactsTool,
    SubmitForReviewTool,
)

# Tools can be added to bot's tool registry
tools = [
    CreateArtifactTool(),
    UpdateArtifactTool(),
    GetArtifactTool(),
    QueryArtifactsTool(),
    SubmitForReviewTool(),
]
```

## Integration with Wizard Reasoning

When using `WizardReasoning`, artifacts are automatically tracked:

```python
from dataknobs_bots.reasoning import WizardReasoning

wizard = WizardReasoning.from_config({
    "wizard_config": "configs/wizards/configbot.yaml",
    "artifacts": {
        "definitions": {
            "bot_config": {
                "type": "config",
                "reviews": ["validation"],
            }
        }
    },
})

# Access the registry
registry = wizard.artifact_registry
```

## Best Practices

1. **Use definitions** - Define artifact types in configuration for consistency
2. **Track lineage** - Always provide `derived_from` when updating
3. **Meaningful names** - Use descriptive names for artifacts
4. **Tag appropriately** - Use tags for categorization and search
5. **Review important artifacts** - Configure reviews for critical outputs
6. **Handle status transitions** - Use hooks to respond to lifecycle events

## Related Documentation

- [Review System](reviews.md) - Validating artifacts with reviews
- [Context Accumulator](context.md) - Building context from artifacts
- [Task Injection](task-injection.md) - Dynamic tasks for artifact events
- [Wizard Observability](observability.md) - Tracking wizard progress

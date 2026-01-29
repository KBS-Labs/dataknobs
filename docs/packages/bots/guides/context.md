# Context Accumulator

The context accumulator provides infrastructure for building and managing conversation context, including wizard state, artifacts, assumptions, and tool history. It generates formatted context for prompt injection.

## Overview

Context accumulates throughout a conversation:

```
Turn 1: User states requirements
  → Context: { requirements: "..." }

Turn 2: Bot extracts structured data
  → Context: { requirements: "...", extracted: {...}, assumptions: [...] }

Turn 3: Bot produces artifact
  → Context: { ..., artifacts: [{ id: "q1", type: "question", ... }] }

Turn 4: Artifact is reviewed
  → Context: { ..., artifacts: [{ id: "q1", ..., reviews: [...] }] }
```

This context is:
- **Persisted** in conversation metadata
- **Injected** into prompts when needed
- **Queryable** by tools and reasoning strategies

## Quick Start

```python
from dataknobs_bots.context import (
    ConversationContext,
    ContextBuilder,
    ContextPersister,
)

# Create context
context = ConversationContext(conversation_id="conv_123")

# Add an assumption
context.add_assumption(
    content="User wants a math tutor bot",
    source="inferred",
    confidence=0.7,
)

# Generate prompt injection
prompt_context = context.to_prompt_injection(max_tokens=2000)
print(prompt_context)
```

## ConversationContext

The main data structure for accumulated context:

```python
from dataknobs_bots.context import ConversationContext

context = ConversationContext(
    conversation_id="conv_123",
    wizard_stage="collect_requirements",
    wizard_data={"domain": "education", "subject": "mathematics"},
    wizard_progress=0.25,
)
```

### Fields

| Field | Description |
|-------|-------------|
| `conversation_id` | Unique conversation identifier |
| `wizard_stage` | Current wizard stage name |
| `wizard_data` | Collected wizard data |
| `wizard_progress` | Progress (0.0 to 1.0) |
| `wizard_tasks` | Current tasks |
| `artifacts` | Artifact summaries |
| `assumptions` | Tracked assumptions |
| `tool_history` | Recent tool executions |
| `transitions` | Stage transitions |
| `sections` | Custom context sections |

## Assumption Tracking

Track what the bot has assumed about user intent:

```python
from dataknobs_bots.context import Assumption, AssumptionSource

# Add an assumption
assumption = context.add_assumption(
    content="User wants a math tutor for middle school students",
    source="inferred",  # inferred, user_stated, default, extracted
    confidence=0.7,
    related_to="target_audience",
)

# Confirm an assumption (user validated it)
context.confirm_assumption(assumption.id)

# Reject an assumption (user said it's wrong)
context.reject_assumption(assumption.id)

# Query assumptions
unconfirmed = context.get_unconfirmed_assumptions()
low_confidence = context.get_low_confidence_assumptions(threshold=0.6)
audience_assumptions = context.get_assumptions_for("target_audience")
```

### Assumption Sources

| Source | Description |
|--------|-------------|
| `inferred` | Bot reasoned this from context |
| `user_stated` | User explicitly said this |
| `default` | Applied a default value |
| `extracted` | Extracted from user input |

## Context Sections

Organize context by importance with prioritized sections:

```python
from dataknobs_bots.context import ContextSection

# Add a custom section
context.add_section(
    name="user_preferences",
    content={
        "tone": "friendly",
        "formality": "casual",
        "detail_level": "high",
    },
    priority=80,  # Higher = more important (0-100)
    formatter="summary",  # default, json, list, summary
)

# Always-include sections
context.add_section(
    name="critical_constraint",
    content="Bot must never discuss politics",
    priority=100,
    include_always=True,
)

# Query sections
section = context.get_section("user_preferences")
context.remove_section("old_section")
```

### Section Formatters

| Formatter | Output |
|-----------|--------|
| `default` | String representation |
| `json` | JSON formatted |
| `list` | Bulleted list |
| `summary` | Key-value summary |

## Prompt Injection

Generate context for prompt injection:

```python
# Basic injection
prompt_context = context.to_prompt_injection(max_tokens=2000)

# Include specific sections
prompt_context = context.to_prompt_injection(
    max_tokens=1500,
    include_sections=["wizard_progress", "unconfirmed_assumptions"],
)

# Exclude sections
prompt_context = context.to_prompt_injection(
    max_tokens=2000,
    exclude_sections=["tool_history"],
)
```

### Output Example

```markdown
## Conversation Context

### Wizard Progress
stage: collect_requirements, progress: 25%, data_collected: domain, subject

### Unconfirmed Assumptions
- assumption: User wants a math tutor, confidence: 0.7
- assumption: Target audience is middle school, confidence: 0.6

### Pending Artifacts
- name: Bot Configuration, status: draft
```

## ContextBuilder

Build context from a conversation manager:

```python
from dataknobs_bots.context import ContextBuilder

# Create builder with artifact registry
builder = ContextBuilder(artifact_registry=registry)

# Build context from conversation manager
context = builder.build(conversation_manager)

# Context now includes:
# - Wizard state and progress
# - Artifacts from registry
# - Tool execution history
# - Transitions
```

## ContextPersister

Persist context to conversation metadata:

```python
from dataknobs_bots.context import ContextPersister

persister = ContextPersister()

# Save context to conversation metadata
persister.persist(context, conversation_manager)

# Serialize context to dictionary for custom storage
state = persister.persist_to_dict(context)
# Can then be restored with ConversationContext.from_dict(state)
```

## Artifact Access

Query artifacts from context:

```python
# Get all artifacts
all_artifacts = context.artifacts

# Filter by status
approved = context.get_artifacts(status="approved")
pending = context.get_artifacts(status="pending_review")

# Filter by type
configs = context.get_artifacts(artifact_type="config")

# Filter by definition
questions = context.get_artifacts(definition_id="assessment_questions")

# Get reviews for an artifact
reviews = context.get_artifact_reviews("art_123")
```

## Serialization

Save and restore context:

```python
# To dictionary
data = context.to_dict()

# From dictionary
restored = ConversationContext.from_dict(data)
```

## Standard Sections

The context automatically builds standard sections:

### Wizard Progress (priority: 90)
```markdown
### Wizard Progress
stage: collect_requirements, progress: 25%, data_collected: domain, subject
```

### Unconfirmed Assumptions (priority: 85)
```markdown
### Unconfirmed Assumptions
- assumption: User wants a math tutor, confidence: 0.7
```

### Pending Artifacts (priority: 70)
```markdown
### Pending Artifacts
- name: Bot Config, status: draft
```

### Approved Artifacts (priority: 50)
```markdown
### Approved Artifacts
- Bot Configuration
```

### Recent Tools (priority: 40)
```markdown
### Recent Tools
- tool: extract_requirements, success: True
```

## Integration with Wizard Reasoning

Context is built using ContextBuilder with WizardReasoning:

```python
from dataknobs_bots.reasoning import WizardReasoning
from dataknobs_bots.context import ContextBuilder

wizard = WizardReasoning.from_config(config)

# Build context from conversation manager
builder = ContextBuilder(artifact_registry=wizard.artifact_registry)
context = builder.build(conversation_manager)

# Use context for prompt injection
prompt_context = context.to_prompt_injection(max_tokens=2000)
```

## Configuration

Configure context behavior in bot configuration:

```yaml
context:
  # Maximum tokens for prompt injection
  max_prompt_tokens: 2000

  # Default section priorities
  section_priorities:
    wizard_progress: 90
    unconfirmed_assumptions: 85
    pending_artifacts: 70
    approved_artifacts: 50
    recent_tools: 40

  # Sections to always include
  always_include:
    - wizard_progress
    - unconfirmed_assumptions

  # Sections to exclude
  exclude:
    - tool_history
```

## Use Cases

### Assumption Confirmation Flow

```python
# Bot makes assumptions during extraction
context.add_assumption(
    content="User wants assessment for grades 6-8",
    source="extracted",
    confidence=0.6,
    related_to="grade_level",
)

# Generate response asking for confirmation
unconfirmed = context.get_unconfirmed_assumptions()
if unconfirmed:
    low_confidence = [a for a in unconfirmed if a.confidence < 0.8]
    if low_confidence:
        # Ask user to confirm
        for a in low_confidence:
            # ... generate confirmation question
            pass

# User confirms
context.confirm_assumption(assumption.id)
```

### Context-Aware Prompting

```python
# Build context-aware system prompt
base_prompt = "You are a helpful bot configuration assistant."

context_injection = context.to_prompt_injection(max_tokens=1500)

system_prompt = f"""{base_prompt}

{context_injection}

Use the context above to provide relevant responses."""
```

## Best Practices

1. **Track assumptions explicitly** - Don't let the bot assume silently
2. **Confirm low-confidence assumptions** - Ask the user when unsure
3. **Prioritize sections** - Important context should have higher priority
4. **Manage token budget** - Use `max_tokens` to avoid prompt overflow
5. **Persist regularly** - Save context after significant changes
6. **Clean up old assumptions** - Remove confirmed/rejected assumptions

## Related Documentation

- [Artifact System](artifacts.md) - Managing artifacts
- [Review System](reviews.md) - Validating artifacts
- [Wizard Observability](observability.md) - Tracking wizard progress
- [Focus Guards](focus-guards.md) - Maintaining conversation focus

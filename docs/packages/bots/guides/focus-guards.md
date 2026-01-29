# Focus Guards

Focus guards help maintain conversation focus in ReAct reasoning by detecting and correcting conversational drift. They ensure the LLM stays on topic during multi-turn interactions.

## Overview

In multi-turn conversations, LLMs can drift off-topic, especially when:

- Users ask tangential questions
- The LLM gets "creative" with responses
- Complex topics invite exploration

Focus guards provide:

- **Detection** - Identify when responses drift off-topic
- **Context injection** - Remind the LLM of its current focus
- **Correction** - Generate prompts to redirect the conversation
- **Tangent tracking** - Allow some flexibility before correction

## Quick Start

```python
from dataknobs_bots.reasoning import FocusGuard

# Create guard
guard = FocusGuard(max_tangent_depth=2)

# Build focus context
focus_context = guard.build_context(
    primary_goal="Help user configure their bot",
    current_task="Collect the bot's name",
    collected_data={"domain": "education"},
)

# Get focus prompt to inject
focus_prompt = guard.get_focus_prompt(focus_context)
# Inject into system prompt

# After response, evaluate for drift
evaluation = guard.evaluate_response(
    response_text="Let me tell you about the history of chatbots...",
    focus_context=focus_context,
)

if evaluation.is_drifting:
    correction = guard.get_correction_prompt(evaluation)
    # Add correction to next turn
```

## FocusContext

Context defining what the conversation should focus on:

```python
from dataknobs_bots.reasoning import FocusContext

context = FocusContext(
    primary_goal="Configure a customer support bot",
    current_task="Collect the knowledge base location",
    collected_data={"name": "SupportBot", "domain": "retail"},
    required_fields=["knowledge_base_path", "response_style"],
    stage_name="configuration",
    tangent_count=0,
    max_tangent_depth=2,
    topic_keywords=["knowledge", "documents", "support", "customers"],
    off_topic_keywords=["history", "theory", "general AI"],
)

# Check if at tangent limit
if context.is_at_tangent_limit:
    # Force redirect
    pass
```

### Fields

| Field | Description |
|-------|-------------|
| `primary_goal` | Overall objective |
| `current_task` | Immediate task |
| `collected_data` | Data already gathered |
| `required_fields` | Fields still needed |
| `stage_name` | Current wizard stage |
| `tangent_count` | Consecutive off-topic turns |
| `max_tangent_depth` | Maximum allowed tangents |
| `topic_keywords` | Keywords indicating on-topic |
| `off_topic_keywords` | Keywords indicating off-topic |

## FocusEvaluation

Result of evaluating a response for drift:

```python
from dataknobs_bots.reasoning import FocusEvaluation

# After evaluation
evaluation = guard.evaluate_response(response_text, focus_context)

print(f"Drifting: {evaluation.is_drifting}")
print(f"Severity: {evaluation.drift_severity}")  # 0.0 to 1.0
print(f"Topic detected: {evaluation.detected_topic}")
print(f"Reason: {evaluation.reason}")
print(f"Suggested redirect: {evaluation.suggested_redirect}")
print(f"Tangent count: {evaluation.tangent_count}")

# Check if correction is needed
if evaluation.needs_correction:  # is_drifting and severity > 0.5
    correction = guard.get_correction_prompt(evaluation)
```

## FocusGuard

The main class for drift detection and correction:

### Creating a Guard

```python
from dataknobs_bots.reasoning import FocusGuard

# Basic guard
guard = FocusGuard()

# Customized guard
guard = FocusGuard(
    max_tangent_depth=2,      # Allow 2 tangent turns before correction
    drift_threshold=0.5,       # Severity threshold for triggering correction
    use_keyword_detection=True,  # Use keyword-based detection
    use_llm_evaluation=False,    # LLM-based evaluation (not yet implemented)
)

# From configuration
guard = FocusGuard.from_config({
    "max_tangent_depth": 2,
    "drift_threshold": 0.5,
    "use_keyword_detection": True,
})
```

### Building Focus Context

```python
# Manual context building
context = guard.build_context(
    primary_goal="Configure the bot",
    current_task="Get the bot name",
    collected_data={"domain": "education"},
    required_fields=["name", "llm_model"],
    stage_name="basic_info",
    topic_keywords=["name", "bot", "configuration"],
    off_topic_keywords=["history", "future", "theory"],
)

# From ConversationContext
from dataknobs_bots.context import ConversationContext

conv_context = ConversationContext(...)
focus_context = guard.build_context_from_conversation(
    conversation_context=conv_context,
    current_task="Collect bot name",
)
```

### Generating Focus Prompts

```python
# Generate focus prompt for injection
focus_prompt = guard.get_focus_prompt(focus_context)

# Example output:
# ## Focus Guidance
# **Primary Goal**: Configure the bot
# **Current Task**: Get the bot name
# **Still Needed**: name, llm_model
# **Already Have**: domain
#
# Stay focused on the current task. If the user asks about
# something unrelated, acknowledge briefly and redirect to
# the task at hand.
```

### Evaluating Responses

```python
# Evaluate a response for drift
evaluation = guard.evaluate_response(
    response_text=llm_response.text,
    focus_context=focus_context,
)

# Update context based on evaluation
updated_context = guard.update_context_after_evaluation(
    context=focus_context,
    evaluation=evaluation,
)
```

### Generating Corrections

```python
if evaluation.needs_correction:
    correction = guard.get_correction_prompt(evaluation)

    # Example output:
    # ## Focus Correction Needed
    # Issue: More off-topic content than on-topic
    # Redirect to: Get the bot name
    #
    # Please gently steer the conversation back to the main topic
    # while acknowledging what the user mentioned.

    # Inject into next turn
    next_system_prompt = f"{base_prompt}\n\n{correction}"
```

## Keyword Detection

The default detection uses keyword matching:

```python
# Configure topic keywords
context = guard.build_context(
    primary_goal="Build assessment questions",
    topic_keywords=[
        "assessment", "questions", "quiz", "test",
        "learning", "objective", "grade", "evaluate"
    ],
    off_topic_keywords=[
        "history", "chatbot theory", "AI research",
        "future technology", "general knowledge"
    ],
)

# Detection checks:
# 1. Off-topic keywords in response
# 2. On-topic keywords in response
# 3. Goal/task words in response
# 4. Calculates severity based on balance
```

## Tangent Tolerance

Allow some flexibility before correction:

```python
guard = FocusGuard(max_tangent_depth=2)

# Turn 1: User asks tangent question
# → evaluation.tangent_count = 1, no correction

# Turn 2: Still off-topic
# → evaluation.tangent_count = 2, correction suggested

# Turn 3: Still off-topic
# → evaluation.tangent_count >= max, FIRM correction

# If response is on-topic:
# → tangent_count resets to 0
```

## Integration with ReActReasoning

```python
from dataknobs_bots.reasoning import ReActReasoning, FocusGuard

# Create guard
guard = FocusGuard(max_tangent_depth=2)

# Build focus context for current task
focus_context = guard.build_context(
    primary_goal=wizard_goal,
    current_task=current_task_description,
)

# Add focus prompt to system message
focus_prompt = guard.get_focus_prompt(focus_context)
system_prompt = f"{base_system_prompt}\n\n{focus_prompt}"

# After each reasoning step, evaluate
evaluation = guard.evaluate_response(
    response_text=reasoning_output,
    focus_context=focus_context,
)

if evaluation.needs_correction:
    correction = guard.get_correction_prompt(evaluation)
    # Add to next iteration
```

## Configuration

Configure focus guards in bot configuration:

```yaml
# bot_config.yaml
focus_guard:
  # Maximum consecutive off-topic turns allowed
  max_tangent_depth: 2

  # Drift severity threshold (0.0-1.0)
  drift_threshold: 0.5

  # Use keyword-based detection
  use_keyword_detection: true

  # Default topic keywords (can be overridden per stage)
  default_topic_keywords:
    - configuration
    - settings
    - options

  # Default off-topic keywords
  default_off_topic_keywords:
    - history
    - theory
    - research
```

## Per-Stage Keywords

Configure keywords per wizard stage:

```yaml
# wizard_config.yaml
stages:
  collect_requirements:
    topic_keywords:
      - requirements
      - needs
      - goals
      - features
    off_topic_keywords:
      - implementation
      - code
      - technical details

  configuration:
    topic_keywords:
      - settings
      - options
      - configure
      - choose
    off_topic_keywords:
      - requirements
      - design
      - architecture
```

## Use Cases

### Guided Data Collection

```python
# User might go on tangents during data collection
focus_context = guard.build_context(
    primary_goal="Collect user information for account setup",
    current_task="Get the user's email address",
    required_fields=["email", "phone", "name"],
)

# Evaluate each response
evaluation = guard.evaluate_response(response, focus_context)

if evaluation.is_drifting:
    # Response talked about something other than email
    # Inject correction for next turn
    pass
```

### Wizard Flow Maintenance

```python
# Each wizard stage has a specific focus
stage_config = wizard.get_current_stage_config()

focus_context = guard.build_context(
    primary_goal=wizard.get_goal(),
    current_task=stage_config.get("task_description"),
    topic_keywords=stage_config.get("topic_keywords", []),
    off_topic_keywords=stage_config.get("off_topic_keywords", []),
)
```

### Tool-Using Agents

```python
# ReAct agents can get distracted during tool use
focus_context = guard.build_context(
    primary_goal="Answer the user's question about their order",
    current_task="Look up order status",
    topic_keywords=["order", "status", "shipping", "delivery"],
)

# After each ReAct step
evaluation = guard.evaluate_response(step_output, focus_context)

if evaluation.needs_correction:
    # Agent went off-topic, redirect
    pass
```

## Best Practices

1. **Set appropriate thresholds** - Too strict interrupts natural conversation
2. **Allow some tangents** - Users naturally explore; 2-3 tangents is usually fine
3. **Use relevant keywords** - Generic keywords cause false positives
4. **Update context** - Keep focus context current with wizard state
5. **Gentle corrections** - Don't be too abrupt when redirecting
6. **Test with real conversations** - Tune based on actual usage

## Limitations

- Keyword detection may miss context-dependent drift
- LLM-based evaluation (more accurate) not yet implemented
- May over-correct during legitimate exploration
- Requires manual keyword configuration per domain

## Related Documentation

- [Context Accumulator](context.md) - Building conversation context
- [Wizard Observability](observability.md) - Tracking wizard state
- [Task Injection](task-injection.md) - Dynamic task management

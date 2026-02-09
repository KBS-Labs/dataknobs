# Context-Aware Wizards

LLM-generated template variables, transition data derivation, dynamic suggestions, and extraction context for wizard stages.

## Table of Contents

- [Overview](#overview)
- [Context Generation](#context-generation)
  - [Configuration](#configuration)
  - [How It Works](#how-it-works)
  - [Prompt Rendering](#prompt-rendering)
  - [Fallback Behavior](#fallback-behavior)
- [Transition Data Derivation](#transition-data-derivation)
  - [Configuration](#derive-configuration)
  - [How It Works](#derive-how-it-works)
  - [Auto-Advance Integration](#auto-advance-integration)
  - [Override Protection](#override-protection)
- [Dynamic Suggestions](#dynamic-suggestions)
  - [Jinja2 in Suggestions](#jinja2-in-suggestions)
- [Extraction Context](#extraction-context)
  - [Bot Response in Extraction](#bot-response-in-extraction)
- [Complete Example](#complete-example)
- [Design Rationale](#design-rationale)

---

## Overview

Wizard stages use Jinja2 `response_template` fields to render deterministic prompts with collected data. This keeps stage output predictable and removes dependence on LLM compliance for structure and formatting.

However, some content benefits from dynamic generation — creative name suggestions, subject-specific examples, and contextual recommendations change based on what the user has said so far.

Context-aware wizards solve this with four complementary features:

| Feature | Purpose | Where |
|---------|---------|-------|
| **Context Generation** | Generate template variables via LLM before rendering | Stage property |
| **Transition Derivation** | Derive field values from existing data on transitions | Transition property |
| **Dynamic Suggestions** | Render suggestion buttons with Jinja2 and state data | Stage suggestions |
| **Extraction Context** | Include bot's last response in extraction input | Automatic |

The template remains the structural backbone — the LLM only fills in contextual "flavor" variables. If the LLM fails, a fallback value is used and the wizard continues without interruption.

## Context Generation

### Configuration

Add a `context_generation` block to any stage that needs LLM-generated content:

```yaml
stages:
  - name: configure_identity
    prompt: "Let's give your bot an identity."
    context_generation:
      prompt: |
        Suggest 3 creative bot names for a {{ intent|default("educational") }}
        bot about {{ subject|default("general topics") }}.
        Format each as a markdown bullet: **Name** (`slug-id`)
        Keep it to 3 short lines, no preamble.
      variable: suggested_names
      model: $resource:micro
      fallback: |
        - **Study Buddy** (`study-buddy`)
        - **Edu Coach** (`edu-coach`)
        - **Learn Lab** (`learn-lab`)
    response_template: |
      Here are some name ideas:
      {{ suggested_names }}

      Or choose your own!
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | `str` | Yes | Jinja2 template rendered with wizard state data, then sent to the LLM |
| `variable` | `str` | Yes | Name of the variable injected into the template context |
| `model` | `str` | No | LLM model or `$resource:` reference (defaults to bot's configured LLM) |
| `fallback` | `str` | No | Value used if LLM call fails, times out, or returns empty |

### How It Works

1. When a stage with `context_generation` is about to render its response, the wizard first processes the generation block.
2. The `prompt` template is rendered with current wizard state data (all collected fields).
3. The rendered prompt is sent to the configured LLM as a single user message.
4. The LLM's response is stored under the configured `variable` name.
5. The variable is passed as `extra_context` to `_render_response_template()`, making it available alongside collected data.

```
State Data: {subject: "Chemistry", intent: "tutor"}
     ↓
Render prompt: "Suggest 3 names for a tutor bot about Chemistry..."
     ↓
LLM generates: "- **Chem Coach** (`chem-coach`) ..."
     ↓
Template context: {subject: "Chemistry", intent: "tutor", suggested_names: "- **Chem Coach** ..."}
     ↓
Render response_template with full context
```

### Prompt Rendering

The `prompt` field is itself a Jinja2 template. It has access to all non-internal wizard state data (keys not starting with `_`):

```yaml
context_generation:
  prompt: |
    The user wants a {{ intent }} bot about {{ subject }}.
    Suggest names that reflect the {{ subject }} domain.
```

If a referenced variable is undefined, it renders as an empty string (using Jinja2's `Undefined` mode, not `StrictUndefined`). Use `|default()` filters for safer defaults:

```yaml
prompt: "Names for a {{ intent|default('educational') }} bot"
```

### Fallback Behavior

The `fallback` value is used when:

- The LLM call raises an exception (connection timeout, API error)
- The LLM returns an empty response
- The prompt template itself fails to render

If no fallback is provided and the LLM fails, the variable is not set (empty dict returned). The template should handle this gracefully with `{% if %}` or `|default()`.

```yaml
# In the template, handle missing variable:
response_template: |
  {% if suggested_names %}
  Here are some ideas:
  {{ suggested_names }}
  {% else %}
  Please provide a name for your bot.
  {% endif %}
```

## Transition Data Derivation

### Derive Configuration

Add a `derive` block to any transition to set field values before the transition condition is evaluated:

```yaml
transitions:
  - target: select_template
    condition: "data.get('intent') is not None"
    derive:
      template_name: "{{ intent }}"
      use_template: true
    transform: save_draft
```

#### Fields

The `derive` block is a dictionary of key-value pairs:

| Value Type | Behavior |
|------------|----------|
| String with `{{ }}` | Rendered as Jinja2 template with wizard state data |
| Literal (bool, int, etc.) | Set directly in wizard state |

### Derive How It Works

1. When the wizard evaluates transitions for the current stage, each transition's `derive` block is processed first.
2. Jinja2 string values are rendered with current wizard state data.
3. Literal values are set directly.
4. Derived values are merged into `wizard_state.data`.
5. Then the transition `condition` is evaluated as normal.

Derivations run for **all** transitions on the current stage, not just the one that ultimately fires. This ensures derived values are available for auto-advance checks on the target stage.

### Auto-Advance Integration

Derivation works with the existing `auto_advance_filled_stages` setting. When derived values satisfy a target stage's required schema fields, the wizard auto-advances past that stage:

```yaml
# Welcome stage: intent=quiz → derive template_name=quiz
# select_template stage requires template_name in enum [tutor, quiz, study_companion]
# → Schema satisfied → auto-advance fires → wizard skips select_template
```

For values that don't match the target schema (e.g., `intent: custom` derives `template_name: custom`, which is not in the enum), auto-advance correctly fails and the user sees the full stage prompt.

### Override Protection

Derivations do **not** overwrite existing values in wizard state. If a key already exists in `wizard_state.data`, the derived value is silently skipped. This protects user-provided data from being overwritten by derivation rules.

Empty Jinja2 renders (when the referenced variable is undefined) are also skipped — no empty strings are set.

## Dynamic Suggestions

### Jinja2 in Suggestions

Stage `suggestions` (shown as quick-reply buttons in WebUI) support Jinja2 rendering with wizard state data:

```yaml
suggestions:
  - "Call it '{{ subject|default('Study') }} Ace'"
  - "Name it '{{ subject|default('Edu') }} Helper'"
  - "I have my own name in mind"
```

With `subject: "Chemistry"` in wizard state, these render as:
- "Call it 'Chemistry Ace'"
- "Name it 'Chemistry Helper'"
- "I have my own name in mind"

Plain suggestions (no `{{ }}` markers) pass through unchanged — no Jinja2 processing overhead.

Internal keys (starting with `_`) are excluded from the suggestion template context, consistent with how `response_template` rendering works.

## Extraction Context

### Bot Response in Extraction

When the user responds to a stage prompt, the extraction model needs context to resolve references. For example, if the bot showed three name suggestions and the user clicks "Use the first suggestion", the extraction model must see those suggestions to extract the correct name and ID.

The wizard automatically includes the bot's most recent response in the extraction input. The extraction model receives:

```
Bot's previous message:
Here are some name ideas:
- **Grammar Guru** (`grammar-guru`)
- **Word Wizard** (`word-wizard`)
...

User's response:
Use the first suggestion
```

This allows the extraction model to resolve deictic references ("the first one", "yes to that", "use suggestion #2") against the actual content that was shown.

Bot responses longer than 1500 characters are truncated to avoid overwhelming the extraction model. When no previous bot response exists (e.g., the first stage), no prepend occurs.

## Complete Example

A wizard stage that combines all four features:

```yaml
stages:
  - name: welcome
    is_start: true
    prompt: "What kind of bot would you like to create?"
    schema:
      type: object
      properties:
        intent:
          type: string
          enum: [tutor, quiz, study_companion, custom]
        subject:
          type: string
    suggestions:
      - "I want to create a math tutor"
      - "Help me build a quiz bot for history"
    transitions:
      - target: select_template
        condition: "data.get('intent') is not None"
        derive:                              # ← Transition derivation
          template_name: "{{ intent }}"
          use_template: true

  - name: select_template
    prompt: "Which template?"
    auto_advance: true                       # ← Auto-advances when derivation fills schema
    schema:
      type: object
      properties:
        template_name:
          type: string
          enum: [tutor, quiz, study_companion]
    transitions:
      - target: configure_identity

  - name: configure_identity
    prompt: "Name your bot."
    context_generation:                      # ← LLM context generation
      prompt: |
        Suggest 3 creative bot names for a {{ intent }} bot
        about {{ subject|default("general topics") }}.
        Format: markdown bullets with **Name** (`slug-id`)
      variable: suggested_names
      fallback: |
        - **Study Buddy** (`study-buddy`)
        - **Edu Coach** (`edu-coach`)
    response_template: |
      Here are some name ideas:
      {{ suggested_names }}

      Or choose your own!
    suggestions:                             # ← Dynamic suggestions
      - "Use the first suggestion"
      - "I have my own name in mind"
    schema:
      type: object
      properties:
        domain_name:
          type: string
        domain_id:
          type: string
          pattern: "^[a-z][a-z0-9-]+$"
    transitions:
      - target: done
        condition: "data.get('domain_name')"

  - name: done
    is_end: true
    prompt: "All set!"
```

**Flow with `"I want to create a quiz bot for history"`:**

1. **welcome** — Extracts `intent=quiz`, `subject=history`
2. **welcome → select_template** transition — Derives `template_name=quiz`, `use_template=true`
3. **select_template** — Auto-advances (schema satisfied by derived `template_name`)
4. **configure_identity** — Generates context-aware name suggestions via LLM, renders template with `{{ suggested_names }}`
5. User clicks "Use the first suggestion" — Extraction model sees the bot's response containing the suggestions and extracts the correct name/ID

## Design Rationale

### Why not let the LLM generate the entire response?

Deterministic templates solve a real problem: LLM-generated responses were unreliable at following formatting instructions, providing correct field names, and maintaining consistent structure across stages. Templates guarantee structure; the LLM only fills in creative content where it adds value.

### Why a separate `context_generation` block instead of extending `llm_assist`?

`llm_assist` is triggered reactively when users ask questions. Context generation is proactive — it runs before the stage renders, producing variables the template can reference. Different trigger, different lifecycle.

### Why derive on transitions instead of using transforms?

Transforms run after a transition fires and can modify state. Derivation runs before condition evaluation, enabling the derivation to both satisfy the condition and pre-fill data for auto-advance on the target stage. Transforms can't do this because they run too late.

### Why include the bot response in extraction?

With `extraction_scope: current_message`, the extraction model only sees the user's text. When users make referential statements ("the first one", "yes", "use that name"), the model needs the bot's output for context. Including it as a prefix keeps the extraction scope setting meaningful while solving the reference resolution problem.

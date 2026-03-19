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
  - [Verbatim Capture and Deictic Resolution](#verbatim-capture-and-deictic-resolution)
  - [capture_mode Stage Field](#capture_mode-stage-field)
- [Extraction Grounding](#extraction-grounding)
  - [The Problem](#the-problem)
  - [How Grounding Works](#how-grounding-works)
  - [Configuration](#grounding-configuration)
  - [Custom Merge Filters](#custom-merge-filters)
  - [Walk-Through: Correction Scenario](#walk-through-correction-scenario)
- [Message Stages](#message-stages)
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

## Response Mode Hierarchy

Wizard stages support several response modes. Choose the **simplest mode** that meets the stage's needs:

| Priority | Mode | Config Fields | Use When |
|----------|------|---------------|----------|
| 1 (default) | **Template-only** | `response_template` + `schema` | Data collection — most stages |
| 2 | **Message stage** | `response_template` + `auto_advance: true` (no schema) | Display-only — confirmations, status updates, transitions |
| 3 | **Template + context** | `response_template` + `context_generation` | Dynamic flavor (personalized remarks, creative content) |
| 4 | **Template + LLM assist** | `response_template` + `llm_assist: true` | User may ask help questions during a stage |
| 5 | **LLM-driven** | `prompt` only (no template) | Open-ended conversation stages (`mode: conversation`) |

**Template-first is strongly recommended.** LLM-driven data-collection stages are unreliable — the LLM may ignore stage instructions, ask for different fields, or hallucinate data. The `response_template` produces consistent, deterministic output while the `schema` handles extraction.

The loader will warn if a non-end, non-conversation stage has no `schema` and no `response_template`.

---

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

### Verbatim Capture and Deictic Resolution

For trivial schemas (a single required string field with no `enum`, `pattern`, or `format` constraints), the wizard uses **verbatim capture** — the user's raw input is stored directly without calling the extraction LLM. This is a performance optimization that avoids an unnecessary LLM round-trip for simple inputs.

However, verbatim capture is automatically skipped when the bot's prior response is available in the conversation. This ensures that deictic references like "the first one", "yes to that", or "use suggestion #2" are routed through LLM extraction, where the bot's response provides the context needed to resolve them.

For example, if the bot presents:

```
I can assist with:
- **Billing** questions
- **Technical** issues
```

And the user responds "The first one", verbatim capture would store the literal string `"The first one"`. Instead, the wizard detects the bot response and routes through LLM extraction, which resolves the reference to `"billing"`.

### capture_mode Stage Field

The `capture_mode` field gives stage authors explicit control over the extraction strategy, overriding the automatic detection described above.

```yaml
stages:
  - name: welcome
    capture_mode: extract    # Force LLM extraction despite trivial schema
    schema:
      type: object
      properties:
        issue_type:
          type: string
      required:
        - issue_type
```

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Schema-based detection: trivial schemas use verbatim capture unless a bot response is available |
| `"verbatim"` | Always use verbatim capture (skip LLM extraction) |
| `"extract"` | Always use LLM extraction |

`capture_mode` can be set as a top-level stage field or nested under `collection_config`. The top-level field takes precedence when both are set.

Use `capture_mode: extract` when a stage has a trivial schema but the user's input may contain references that need resolution. Use `capture_mode: verbatim` to force the fast path even when a bot response is present (e.g., when the stage prompt never presents options).

## Extraction Grounding

### The Problem

Extraction models sometimes return values for fields the user didn't address. In multi-turn wizards, this causes good data to be overwritten with hallucinated or empty values.

For example, when a user says "make it a tutor instead, keep the same name and subject," the extraction model may return `subject: ""` and `domain_id: ""`, overwriting previously-extracted values.

### How Grounding Works

Extraction grounding verifies each extracted value against the user's actual message before allowing it to overwrite existing wizard state data. It uses type-appropriate heuristics derived from the JSON Schema definition:

| Schema Type | Grounding Strategy |
|-------------|-------------------|
| `string` | Word overlap between value and message (configurable threshold) |
| `string` + `enum` | Enum value appears as a whole word in the message |
| `boolean` | Field-related keywords found in message; by default also checks value direction --- `False` requires negation keywords, `True` requires their absence |
| `integer`/`number` | The literal number appears as a whole word in the message |
| `array` | At least one element appears as a whole word in the message; empty arrays grounded via field keyword + negation keyword |

The merge decision is conservative:

| Grounded? | Existing value? | Action |
|-----------|----------------|--------|
| Yes | Any | **Merge** (user addressed this field) |
| No | None/absent | **Merge** (first extraction, benefit of the doubt) |
| No | Has value | **Skip** (protect existing data) |

First-turn extraction is unaffected — all fields are absent, so everything merges. Grounding only gates **overwrites** of existing data.

### Configuration

Grounding is enabled by default. Control it at three levels:

#### Wizard-Level Settings

```yaml
settings:
  extraction_grounding: true          # default: true
  grounding_overlap_threshold: 0.5    # word overlap ratio for strings
  merge_filter: null                  # custom MergeFilter class (dotted path)
```

#### Per-Stage Override

```yaml
stages:
  - name: free_text_stage
    extraction_grounding: false       # disable grounding for this stage
    schema: ...
```

#### Per-Field Hints (x-extraction)

For edge cases where the inferred grounding strategy is wrong, use the `x-extraction` JSON Schema extension:

```yaml
schema:
  properties:
    tone:
      type: string
      x-extraction:
        grounding: skip               # never grounding-check this field
    domain_id:
      type: string
      x-extraction:
        grounding: exact              # require literal match in message
    description:
      type: string
      description: "Brief description of the bot"
      x-extraction:
        empty_allowed: true           # allow "" as intentional "no value"
        overlap_threshold: 0.3        # lower threshold for this field
    kb_enabled:
      type: boolean
      description: "Whether knowledge base is enabled"
      x-extraction:
        check_direction: true         # verify True/False via negation (default)
        negation_proximity: 3         # negation must be within 3 words of field keyword
```

**Supported `x-extraction` hints:**

| Key | Values | Effect |
|-----|--------|--------|
| `grounding` | `"exact"` / `"fuzzy"` / `"skip"` | Override grounding strategy |
| `empty_allowed` | `true` / `false` | Allow empty string/array to overwrite existing values |
| `overlap_threshold` | `float` | Per-field word overlap ratio override |
| `check_direction` | `true` / `false` | Boolean fields: verify value direction via negation detection (default `true`) |
| `negation_keywords` | `list[str]` | Override the default negation keyword set for this field |
| `negation_proximity` | `int` | Max word distance between negation and field keyword (`0` = anywhere in message; default `0`) |

### Custom Merge Filters

For domain-specific merge logic, provide a custom `MergeFilter` class:

```yaml
settings:
  merge_filter: mypackage.filters.ConfigBotMergeFilter
```

The class must implement the `MergeFilter` protocol
(`from dataknobs_bots.reasoning.wizard_grounding import MergeFilter`):

```python
class MergeFilter(Protocol):
    def should_merge(
        self,
        field: str,
        new_value: Any,
        existing_value: Any,
        user_message: str,
        schema_property: dict[str, Any],
    ) -> bool: ...
```

Custom filters replace the built-in grounding check entirely.

### Walk-Through: Correction Scenario

**Turn 2:** "I want a history quiz bot called History Quizzer, ID history-quizzer."

All fields extracted, no existing data → all merge normally.

**Turn 3:** "Actually, make it a tutor instead. Keep the same name and subject."

| Field | Extracted | Grounded? | Existing | Action |
|-------|-----------|-----------|----------|--------|
| intent | "tutor" | "tutor" in message | "quiz" | **Merge** (correction) |
| subject | "" | no negation keyword | "history" | **Skip** (protected) |
| domain_id | "" | no negation keyword | "history-quizzer" | **Skip** (protected) |

Result: only `intent` is updated to "tutor"; `subject` and `domain_id` are preserved.

---

## Message Stages

Message stages display informational content to the user without collecting data, then auto-advance to the next stage — all within a single user turn. They are configured using existing fields: `auto_advance: true` + `response_template`, with no `schema`.

### Configuration

```yaml
stages:
  - name: confirmation
    prompt: "Confirmation"
    auto_advance: true
    response_template: |
      Your ticket for {{ department }} has been submitted.
      Reference number: {{ ticket_id }}
    transitions:
      - target: next_stage
        condition: "true"
```

The stage needs:

| Field | Required | Purpose |
|-------|----------|---------|
| `auto_advance: true` | Yes | Tells the wizard to advance immediately |
| `response_template` | Yes | The message to display (Jinja2 with wizard state data) |
| `transitions` | Yes | Where to go next (supports conditions for routing) |
| `schema` | No | Omit — message stages don't collect data |

### How It Works

When the wizard arrives at a message stage (during `generate()` or `greet()`):

1. The `response_template` is rendered with current wizard state data
2. The rendered message is collected
3. The wizard transitions to the next stage
4. The collected message is prepended to the next stage's response

The user sees the message and the next stage's prompt combined in a single bot response.

### Per-Stage vs Global Auto-Advance

Message stages use **per-stage** `auto_advance: true`, which is distinct from the **global** `auto_advance_filled_stages` setting:

| Setting | Scope | Schema-less stages? | Use case |
|---------|-------|---------------------|----------|
| `auto_advance: true` | Single stage | Yes | Message stages, always-skip stages |
| `auto_advance_filled_stages` | All stages | No | Skip stages whose required fields are already filled |

The global setting means "skip stages whose required fields are satisfied" — it requires fields to check. Per-stage `auto_advance` means "always advance past this stage" and works with or without a schema.

### Chained Message Stages

Multiple consecutive message stages are supported. Each template is rendered and collected, then all messages are prepended to the final landing stage's response:

```yaml
stages:
  - name: step1_complete
    prompt: "Step 1"
    auto_advance: true
    response_template: "Step 1 complete: {{ item }} registered."
    transitions:
      - target: step2_info

  - name: step2_info
    prompt: "Step 2"
    auto_advance: true
    response_template: "Moving to final review..."
    transitions:
      - target: review
```

The user sees both messages followed by the review stage's prompt. The existing `max_auto_advances = 10` safety limit prevents infinite loops from misconfigured chains.

### Conditional Routing

Message stages support normal transition conditions for routing based on previously collected data:

```yaml
- name: routing_message
  prompt: "Routing"
  auto_advance: true
  response_template: "Taking you to the {{ department }} department."
  transitions:
    - target: billing_intake
      condition: "data.get('department') == 'billing'"
    - target: tech_support
      condition: "data.get('department') == 'technical'"
    - target: general_help
      condition: "true"
```

### Builder API

Use `add_structured_stage()` with `auto_advance=True` and `response_template`:

```python
builder.add_structured_stage(
    "confirmation",
    "Confirmation",
    response_template="Your ticket for {{ department }} has been submitted.",
    auto_advance=True,
)
builder.add_transition("confirmation", "next_stage", condition="true")
```

### Use Cases

- **Confirmations**: "Your order has been placed. Order #{{ order_id }}."
- **Informational transitions**: "Now let's set up your profile."
- **Conditional messages**: Different messages based on routing decisions
- **Status updates**: "Processing complete. {{ count }} items imported."
- **Greetings**: A start stage that displays a welcome message before advancing to the first data-collection stage

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

---

## Automatic Context Injection

Beyond the config-driven context features above, the wizard automatically injects
runtime context into the system prompt. These behaviors require no configuration —
they activate based on stage type and collected data.

### Collection Progress (CD-2)

During collection-mode stages (e.g. gathering ingredients), the wizard injects a
**Collection Progress** section showing what has been collected so far:

```
## Collection Progress (ingredients)
3 items collected so far:
- flour, 2 cups
- sugar, 1 cup
- chocolate chips, 1 cup
```

This compensates for conversation tree branching. Each collection iteration starts
a new sibling branch, so the LLM cannot see prior iterations in the conversation
history. The injected summary provides the full picture.

Up to 20 items are shown; beyond that, a "... and N more" summary appears.

### Collection Summary / Boundary Snapshot (CD-3)

When a stage uses ReAct reasoning (tool-driven review stages), the wizard injects a
**Collection Summary** showing all artifact fields and section records:

```
## Collection Summary
- recipe_name: Chocolate Chip Cookies

### ingredients (3 records)
- flour, 2 cups
- sugar, 1 cup
- chocolate chips, 1 cup

### instructions (3 records)
- Mix dry and wet ingredients separately
- Combine wet and dry ingredients
- Bake at 325 degrees for 12 minutes
```

This serves as a boundary snapshot at the guided-to-dynamic transition — the LLM
sees the complete artifact overview without needing tool calls to discover it.

The summary is refreshed between ReAct iterations via the `prompt_refresher`
callback, so if a tool mutates the artifact mid-loop (e.g. `load_from_catalog`),
the next iteration sees the updated data.

### Non-Happy-Path Context (CD-8)

Clarification, validation error, and restart-offer responses now receive the full
stage context — the same system prompt enhancement as normal responses. Previously,
these code paths used minimal context (~314 tokens), causing the LLM to lose track
of what was being collected.

Affected code paths:
- Clarification responses (when extraction fails or input is ambiguous)
- Validation error responses (when extracted data fails schema validation)
- Restart-offer responses (when the user's input suggests they want to start over)

### System Prompt Override Persistence (CD-10)

Every system prompt override used for an LLM call is persisted in the assistant
message's node metadata under the `system_prompt_override` key. This enables:

- **Replay**: Reconstruct the exact prompt the LLM received for any response
- **Debugging**: Compare prompts across iterations to diagnose context issues
- **Auditing**: Verify that context injection is working as expected

The override is stored by `ConversationManager._finalize_completion()` in the
`dataknobs-llm` package, following the same pattern as `config_overrides_applied`.

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
  - [Grounding Configuration](#grounding-configuration)
  - [Custom Merge Filters](#custom-merge-filters)
  - [Walk-Through: Correction Scenario](#walk-through-correction-scenario)
- [Enum Normalization](#enum-normalization)
  - [The Problem](#enum-problem)
  - [How It Works](#enum-how-it-works)
  - [Configuration](#enum-configuration)
  - [Matching Algorithm](#matching-algorithm)
- [Field Derivation Recovery](#field-derivation-recovery)
  - [The Problem](#derivation-problem)
  - [How It Works](#derivation-how-it-works)
  - [Built-In Transforms](#built-in-transforms)
  - [Template Derivation](#template-derivation)
  - [Guard Conditions](#guard-conditions)
  - [Per-Stage Override](#derivation-per-stage-override)
  - [Custom Transforms](#custom-transforms)
- [Recovery Pipeline](#recovery-pipeline)
  - [The Problem](#recovery-problem)
  - [How It Works](#recovery-how-it-works)
  - [Pipeline Configuration](#pipeline-configuration)
  - [Focused Retry Strategy](#focused-retry-strategy)
  - [Per-Stage Override](#recovery-per-stage-override)
  - [Pipeline Examples](#pipeline-examples)
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

Templates use strict undefined checking — if any referenced variable is missing, the derivation is skipped rather than producing a partial result.

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

### Grounding Configuration

Grounding is enabled by default. Control it at three levels:

#### Wizard-Level Settings

```yaml
settings:
  extraction_grounding: true          # default: true
  grounding_overlap_threshold: 0.5    # word overlap ratio for strings
  merge_filter: null                  # custom MergeFilter class (dotted path)
  extraction_hints:
    enum_normalize: true              # default: true — normalize enum values
    normalize_threshold: 0.7          # fuzzy match threshold (0.0–1.0)
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
| `normalize` | `true` / `false` | Override enum normalization for this field (see [Enum Normalization](#enum-normalization)) |
| `normalize_threshold` | `float` | Per-field token overlap threshold for fuzzy enum matching (default `0.7`) |

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

## Enum Normalization

### The Problem {#enum-problem}

Extraction models often return values that are semantically correct but syntactically wrong for an enum constraint. For example, an extraction model may return `"Tutor"`, `"TUTOR"`, or `"tutor bot"` for a field with `enum: [tutor, quiz, study_companion, custom]`. The intent is clearly `"tutor"`, but the value doesn't match the canonical entry exactly — causing downstream transition conditions like `data.get('intent') == 'tutor'` to fail.

### How It Works {#enum-how-it-works}

Enum normalization runs in `_normalize_extracted_data()` **before** the grounding check, so the grounding filter sees the canonical value:

```
1. Extract from message (SchemaExtractor)
2. Normalize extracted data (type coercion + enum normalization)  ← here
3. Merge with existing wizard_data (grounding filter)
4. Scope escalation (if enabled)
5. Apply schema defaults
6. Apply field derivations
7. Confidence gate (required fields check)
```

When enabled, each string field with an `enum` constraint is matched against the canonical entries using a tiered algorithm. Exact matches pass through untouched — normalization only acts when the extracted value doesn't already match.

### Configuration {#enum-configuration}

Enum normalization is **enabled by default**. Control it at two levels:

#### Class-Level (Wizard Settings)

Apply to all enum fields at once:

```yaml
settings:
  extraction_hints:
    enum_normalize: true          # default: true
    normalize_threshold: 0.7      # fuzzy match threshold (default: 0.7)
```

#### Per-Field Override

Override the class-level setting for individual fields via `x-extraction`:

```yaml
schema:
  properties:
    intent:
      type: string
      enum: [tutor, quiz, study_companion, custom]
      x-extraction:
        normalize: true             # redundant (default is true), shown for clarity
        normalize_threshold: 0.8    # stricter matching for this field
    provider:
      type: string
      enum: [ollama, openai, anthropic]
      x-extraction:
        normalize: false            # require exact match for this field
```

Per-field `normalize` overrides the class-level `enum_normalize` in both directions: `normalize: true` on a field enables normalization even when `enum_normalize: false`, and vice versa.

### Matching Algorithm

The normalization algorithm uses a tiered strategy. The first tier that produces a match wins:

| Tier | Strategy | Example |
|------|----------|---------|
| 1 | **Exact match** (case-sensitive) | `"tutor"` → `"tutor"` |
| 2 | **Case-insensitive** | `"Tutor"`, `"TUTOR"` → `"tutor"` |
| 3 | **Substring** (after `_`/`-` → space normalization) | `"tutor bot"` → `"tutor"`, `"study companion"` → `"study_companion"` |
| 4 | **Token overlap** ≥ threshold | `"interactive quiz"` → `"quiz"` (overlap = 0.5) |

If no tier produces a match, the original value passes through unchanged.

The `normalize_threshold` controls tier 4 sensitivity. At `0.7` (default), at least 70% of tokens must overlap. At `1.0`, tier 4 requires all tokens to match (effectively disabling fuzzy matching while still allowing tiers 1-3).

---

## Field Derivation Recovery

### The Problem {#derivation-problem}

Some fields have deterministic relationships: `domain_id` and `domain_name` are typically derivable from each other (`chess-champ` ↔ `Chess Champ`). When an extraction model captures one but misses the other, the wizard treats the missing field as unsatisfied — blocking auto-advance or forcing a clarification question for information the framework could infer.

Field derivation fills missing fields from present ones using pure functions — no LLM call, no I/O. This is the cheapest recovery strategy.

### How It Works {#derivation-how-it-works}

Derivation runs in the extraction pipeline **after** merge, scope escalation, and schema defaults, and **before** the confidence gate:

```
1. Extract from message (SchemaExtractor)
2. Normalize extracted data (type coercion + enum normalization)
3. Merge with existing wizard_data (grounding filter)
4. Scope escalation (if enabled)
5. Apply schema defaults
6. Apply field derivations  ← fills missing fields from present ones
7. Confidence gate (required fields check)
8. Transition derivations
9. FSM step
```

This ordering ensures derived values count toward the required-field check and are available for transition conditions.

### Configuration

```yaml
settings:
  derivations:
    - source: domain_id
      target: domain_name
      transform: title_case
      when: target_missing

    - source: domain_name
      target: domain_id
      transform: lower_hyphen
      when: target_missing
```

Each rule specifies:
- **source** — field to derive from (must be present)
- **target** — field to fill
- **transform** — how to transform the source value
- **when** — guard condition (default: `target_missing`)

### Built-In Transforms

| Transform | Input → Output | Use Case |
|-----------|---------------|----------|
| `title_case` | `chess-champ` → `Chess Champ` | ID → display name |
| `lower_hyphen` | `Chess Champ` → `chess-champ` | Display name → slug ID |
| `lower_underscore` | `Chess Champ` → `chess_champ` | Display name → snake_case |
| `copy` | Direct copy of source value | Aliased fields |
| `template` | Jinja2 template rendered with wizard data | Composite derivation |

### Template Derivation

For deriving a field from multiple source fields, use the `template` transform with a Jinja2 template:

```yaml
settings:
  derivations:
    - source: intent        # trigger: derive when intent is present
      target: description
      transform: template
      template: "A {{ intent }} bot for {{ subject }}"
      when: target_missing
```

The template has access to the full wizard data dict. If any referenced variable is undefined, the derivation is skipped (the template uses strict undefined checking to prevent partial renders).

### Guard Conditions

| Condition | Meaning | Default? |
|-----------|---------|----------|
| `target_missing` | Source is present, target is not | Yes |
| `target_empty` | Source is present, target is `None` or empty string | No |
| `always` | Always derive, overwriting existing values | No |

`target_missing` is the safe default — it never overwrites user-provided or extracted data. The `always` option exists for cases where derived values should take precedence (e.g., enforcing a naming convention).

### Per-Stage Override {#derivation-per-stage-override}

Disable derivation on specific stages:

```yaml
stages:
  - name: review
    derivation_enabled: false   # suppress derivation on this stage
```

### Custom Transforms

For transforms beyond the built-in set, provide a class implementing the `FieldTransform` protocol:

```yaml
settings:
  derivations:
    - source: subject
      target: domain_id
      transform: custom
      custom_class: mypackage.transforms.SubjectToId
```

The class must implement:

```python
from dataknobs_bots.reasoning.wizard_derivations import FieldTransform

class SubjectToId:
    def transform(self, value: Any, wizard_data: dict[str, Any]) -> Any:
        # Return the derived value
        return value.lower().replace(" ", "-")
```

Custom classes are loaded once at config time and cached.

### Rule Ordering

Rules are processed in order. Each rule runs at most once per turn. When two rules derive from each other (A→B and B→A), the first rule whose source is present wins. For example, with both `domain_id → domain_name` and `domain_name → domain_id` configured:

- If `domain_id` is present: first rule fires (→ `domain_name`), second rule skips (target now present)
- If `domain_name` is present: first rule skips (source missing), second rule fires (→ `domain_id`)

Derivations can also chain: rule A→B fires, then rule B→C fires in the same pass since B is now present.

---

## Recovery Pipeline

### The Problem {: #recovery-problem }

The wizard provides several extraction recovery strategies — [field derivation](#field-derivation-recovery), [scope escalation](#extraction-scope-escalation), and [enum normalization](#enum-normalization) — that each address different failure classes. However, extraction failures are often compound: a single turn might need derivation for one field AND scope escalation for another. Without a composition mechanism, strategies run in a hardcoded sequence with no awareness of each other, potentially wasting LLM calls when a cheaper strategy would have been sufficient.

### How It Works {: #recovery-how-it-works }

The recovery pipeline runs after initial extraction and merge, executing strategies in a configurable order. It **short-circuits** as soon as all required fields are satisfied — minimizing LLM calls and latency in the common case.

```
Extract → Normalize → Merge (grounded) → Schema defaults
  → Recovery pipeline (if required fields missing):
    1. derivation         [free — pure functions, no LLM call]
    2. scope_escalation   [1 LLM call — broader context]
    3. focused_retry      [1 LLM call — focused prompt]
  → Confidence gate
    → PASS: proceed to transitions
    → FAIL: clarification (ask the user)
```

After each strategy, the pipeline checks whether all required fields are now present. If they are, remaining strategies are skipped.

Key design choices:

- **Schema defaults run before the pipeline**, not as a pipeline step. Defaults fill preconfigured values that should always apply, so defaulted fields don't trigger unnecessary recovery.
- **Derivation runs first** (before scope escalation). Prior to the recovery pipeline, scope escalation ran before field derivation. The pipeline reverses this ordering because derivation is free — pure functions with no LLM call. If derivation fills the missing fields, scope escalation never fires, saving an LLM call. If you have derivation rules that depend on fields only available after escalation, list derivation twice in the pipeline: `["derivation", "scope_escalation", "derivation"]`.
- **Each LLM-backed strategy** (scope escalation, focused retry) runs normalize + merge on its results automatically, including grounding checks.

### Pipeline Configuration {: #pipeline-configuration }

Configure the pipeline under the `recovery` settings key:

```yaml
settings:
  recovery:
    pipeline:
      - derivation          # Derive missing fields (free)
      - scope_escalation    # Retry with broader scope (1 LLM call)
      - focused_retry       # Extract only missing fields (1 LLM call)
    focused_retry:
      enabled: true         # Default: false — must opt in
      max_retries: 1        # Default: 1
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `recovery.pipeline` | list of strings | `["derivation", "scope_escalation", "focused_retry"]` | Ordered list of strategies to execute |
| `recovery.focused_retry.enabled` | bool | `false` | Enable the focused retry strategy |
| `recovery.focused_retry.max_retries` | int | `1` | Maximum focused retry attempts per turn |

Valid strategy names: `derivation`, `scope_escalation`, `focused_retry`, `clarification`.

The `clarification` strategy is a no-op placeholder — clarification is handled by the confidence gate after the pipeline, regardless of whether it appears in the list. Including it documents intent but doesn't change behavior.

**Default behavior (zero-config):** When no `recovery` settings are provided, the default pipeline runs `derivation → scope_escalation → focused_retry`. However, scope escalation requires `scope_escalation.enabled: true` to fire, and focused retry requires `recovery.focused_retry.enabled: true`. So without any configuration, only derivation runs (if rules are configured).

### Focused Retry Strategy {: #focused-retry-strategy }

When scope escalation doesn't recover all fields (or is skipped), focused retry re-extracts targeting **only the missing fields**. It builds a minimal schema containing just the missing required fields, then extracts using the full wizard session context.

This works better than full re-extraction because extracting 1-2 fields from a conversation is a much simpler task than extracting 12 fields. Models that fail on the full schema often succeed on a focused subset.

```yaml
settings:
  recovery:
    pipeline:
      - derivation
      - focused_retry
    focused_retry:
      enabled: true
      max_retries: 1    # Try once with the focused schema
```

Focused retry always uses `wizard_session` scope (broadest available context) and forces LLM extraction (never verbatim capture), since the goal is to recover fields that simpler approaches missed.

### Per-Stage Override {: #recovery-per-stage-override }

Recovery can be disabled on individual stages using the `recovery_enabled` stage field:

```yaml
stages:
  - name: gather
    prompt: "Tell me about your project."
    schema: { ... }
    # Uses the global recovery pipeline (default)

  - name: confirm
    prompt: "Does this look right?"
    recovery_enabled: false   # No recovery on this stage
    schema: { ... }
```

When `recovery_enabled: false`, no recovery strategies run for that stage — the pipeline is skipped entirely. This is useful for stages where recovery would be counterproductive (e.g., confirmation stages where you want the user to explicitly provide missing information).

### Pipeline Examples {: #pipeline-examples }

**Minimal pipeline — derivation only (no LLM calls):**

```yaml
settings:
  derivations:
    - source: domain_id
      target: domain_name
      transform: title_case
  recovery:
    pipeline:
      - derivation
```

**Full pipeline — all strategies:**

```yaml
settings:
  extraction_scope: current_message
  scope_escalation:
    enabled: true
    escalation_scope: wizard_session
  derivations:
    - source: domain_id
      target: domain_name
      transform: title_case
  recovery:
    pipeline:
      - derivation
      - scope_escalation
      - focused_retry
    focused_retry:
      enabled: true
```

**Disable all recovery:**

```yaml
settings:
  recovery:
    pipeline: []   # Empty list — no strategies run
```

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

### Extraction scope escalation

When `extraction_scope: current_message` is used for speed, information from earlier turns is invisible to the extraction model. If the user spread required fields across multiple messages, or a weak extraction model missed a field, the wizard would need to ask for clarification — even though the information exists in the conversation history.

Scope escalation addresses this by automatically retrying with a broader scope when required fields are missing after the initial extraction. It only fires when (a) required fields are missing, (b) the current scope is narrower than the escalation target, and (c) there are prior user messages available. The grounding filter (if enabled) protects existing data during the escalated re-extraction.

Three extraction scopes are available, ordered from narrowest to broadest:

1. `current_message` — only the user's latest message (fast, focused)
2. `recent_messages` — the last N user messages (controlled by `scope_escalation.recent_messages_count`, default 3)
3. `wizard_session` — all user messages in the wizard session (comprehensive)

Configure escalation under the `scope_escalation` settings key:

```yaml
settings:
  extraction_scope: current_message
  scope_escalation:
    enabled: true                     # Default: false
    escalation_scope: wizard_session  # Or: recent_messages
    recent_messages_count: 3          # For recent_messages scope
```

Escalation is disabled by default for backward compatibility. The `recent_messages_count` setting applies both when `extraction_scope` is `"recent_messages"` directly and when escalation targets `"recent_messages"`.

### Why a composable recovery pipeline?

The individual recovery strategies (derivation, scope escalation, focused retry) each address different failure classes. But extraction failures are often compound — a single turn might need derivation for one field and scope escalation for another. Without composition, each strategy runs independently in a hardcoded sequence with no awareness of whether prior strategies already satisfied the requirements.

The pipeline provides three benefits: (1) **short-circuiting** — strategies stop running as soon as all required fields are present, avoiding unnecessary LLM calls; (2) **optimal ordering** — derivation (free) runs before escalation (1 LLM call) so cheap strategies get first crack; (3) **configurability** — consumers can reorder, add, or remove strategies to match their cost/latency budget.

The pipeline also introduces **focused retry** as a last-resort strategy before clarification. When all else fails, extracting 1-2 missing fields from a conversation with a minimal schema is a much simpler task than extracting 12 fields — models that fail on the full schema often succeed on a focused subset.

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

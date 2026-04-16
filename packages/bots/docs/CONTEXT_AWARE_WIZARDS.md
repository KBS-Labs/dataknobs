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
  - [Value Expansion](#value-expansion)
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
  - [Boolean Recovery Strategy](#boolean-recovery-strategy)
  - [Focused Retry Strategy](#focused-retry-strategy)
  - [Per-Stage Override](#recovery-per-stage-override)
  - [Pipeline Examples](#pipeline-examples)
  - [Clarification Grouping](#clarification-grouping)
- [Transition Re-Extraction](#transition-re-extraction)
  - [The Problem](#re-extraction-problem)
  - [Configuration](#re-extraction-configuration)
  - [How It Works](#re-extraction-how-it-works)
  - [Interaction with skip_extraction](#interaction-with-skip_extraction)
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
| 6 | **Conversation + greeting** | `mode: conversation` + `response_template` | Conversation stages with a deterministic greeting — template renders once (first turn), then LLM mode |

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

> **Security:** All template rendering uses Jinja2 `SandboxedEnvironment`. See [TEMPLATE_SECURITY.md](TEMPLATE_SECURITY.md) for the full security model and config authoring guidelines.

The `prompt` field is itself a Jinja2 template rendered with wizard state data. All state data (including `_`-prefixed transform outputs) is available as top-level template variables. Stage prompts in `WizardAdvanceResult.stage_prompt` and `metadata["stage_prompt"]` are returned fully rendered.

```yaml
stages:
  details:
    prompt: "Tell me more about {{ topic | default('your chosen topic') }}."
```

The `context_generation.prompt` field is also a Jinja2 template with access to all wizard state data:

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

All state data (including `_`-prefixed keys) is available in the suggestion template context, matching the canonical context used by all rendering sites. Suggestions in `WizardAdvanceResult.suggestions` are returned fully rendered.

See [TEMPLATE_SECURITY.md](TEMPLATE_SECURITY.md) for the full variable availability table and security model.

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

**Type-mismatch guard:** Before any type-specific check, grounding rejects values whose Python type doesn't match the declared schema type (strategy `"type_mismatch"`). This catches extraction errors like a boolean `True` for a `string` field, or a `bool` for an `integer` field (Python's `bool` is a subclass of `int`, so this requires explicit handling). The normalization layer (`_normalize_extracted_data`) performs the same check independently as defense-in-depth.

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
    reject_unmatched: true            # default: true — reject values with no enum match
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
        require_grounded: true        # reject ungrounded even on first write
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
| `require_grounded` | `true` / `false` | Reject ungrounded values even on first write (default `false` — benefit of the doubt when no existing value) |
| `empty_allowed` | `true` / `false` | Allow empty string/array to overwrite existing values |
| `overlap_threshold` | `float` | Per-field word overlap ratio override |
| `check_direction` | `true` / `false` | Boolean fields: verify value direction via negation detection (default `true`) |
| `negation_keywords` | `list[str]` | Override the default negation keyword set for this field |
| `negation_proximity` | `int` | Max word distance between negation and field keyword (`0` = anywhere in message; default `0`) |
| `normalize` | `true` / `false` | Override enum normalization for this field (see [Enum Normalization](#enum-normalization)) |
| `normalize_threshold` | `float` | Per-field token overlap threshold for fuzzy enum matching (default `0.7`) |
| `reject_unmatched` | `true` / `false` | Override enum reject behavior for this field (default `true`; see [Enum Normalization](#enum-normalization)) |
| `boolean_recovery` | `true` / `false` | Enable/disable signal-word recovery for this boolean field (default `true` when strategy is in pipeline; see [Boolean Recovery Strategy](#boolean-recovery-strategy)) |
| `affirmative_signals` | `list[str]` | Override default affirmative signal words for boolean recovery |
| `affirmative_phrases` | `list[str]` | Override default affirmative multi-word phrases for boolean recovery |
| `negative_signals` | `list[str]` | Override default negative signal words for boolean recovery |
| `negative_phrases` | `list[str]` | Override default negative multi-word phrases for boolean recovery |
| `expand_from_message` | `true` / `false` | Enable value expansion for this field (default `false`; see [Value Expansion](#value-expansion)) |

### Value Expansion

Extraction models sometimes return a partial value when the user's message contains a compound phrase. For example, a model may extract `"formal"` when the user said `"formal and academic"`. The extracted value passes grounding (it IS in the message), but it is incomplete.

Value expansion recovers the full phrase by scanning the user's message for the extracted value and expanding rightward across explicit conjunctions (`and`, `or`, `nor`). Expansion stops at natural phrase boundaries:

- Punctuation (`. , ; : ! ?`)
- Field-switching patterns (`and the`, `and set`, `and make`, `and include`, `and add`, `and use`, `but`)
- End of message

When expansion finds a longer phrase, the grounding filter returns `MergeDecision.transform(expanded_value)` instead of `MergeDecision.accept()`. The expanded value — composed entirely of the user's own words — replaces the partial extraction in wizard data.

**Expansion is opt-in** via `x-extraction.expand_from_message: true`. It is best suited for **descriptive fields** where compound phrases are common — tone, style, mood, audience. It is not appropriate for identity fields (names, IDs, topics) where the word `"and"` typically separates values for *different* fields rather than parts of a compound value.

For example, if the user says `"Alice and math"` and the LLM extracts `name: "Alice"`, default-on expansion would incorrectly expand the name to `"Alice and math"`. With opt-in, only fields explicitly marked for expansion are affected.

Expansion is also automatically skipped for:

- **Enum fields** — values are constrained to a fixed set
- **Non-string schema types** — integers, booleans, arrays, etc.

```yaml
schema:
  properties:
    tone:
      type: string
      description: "Writing tone"
      x-extraction:
        expand_from_message: true   # opt in — "formal" → "formal and academic"
    name:
      type: string
      description: "User's name"
      # no expand_from_message — default off, "Alice" stays "Alice"
```

**Example** (with `expand_from_message: true`):

| User message | Extracted | Expanded | Stored |
|---|---|---|---|
| "Set the tone to formal and academic" | `"formal"` | `"formal and academic"` | `"formal and academic"` |
| "Make it warm and inviting" | `"warm"` | `"warm and inviting"` | `"warm and inviting"` |
| "The tone should be formal." | `"formal"` | (no expansion) | `"formal"` |
| "Set the tone to formal and the style to narrative" | `"formal"` | (boundary: `and the`) | `"formal"` |

The expansion algorithm lives in `dataknobs_utils.value_expansion` as a pure text-processing utility. The `SchemaGroundingFilter` in `wizard_grounding.py` calls it for grounded string values on opted-in fields.

### Custom Merge Filters

For domain-specific merge logic, provide a custom `MergeFilter` class:

```yaml
settings:
  merge_filter: mypackage.filters.ConfigBotMergeFilter
```

The class must implement the `MergeFilter` protocol
(`from dataknobs_bots.reasoning.wizard_grounding import MergeFilter, MergeDecision`):

```python
class MergeFilter(Protocol):
    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision: ...
```

The `MergeDecision` dataclass supports three actions:
- `MergeDecision.accept()` — merge the value as-is
- `MergeDecision.reject(reason="...")` — skip the value
- `MergeDecision.transform(new_value, reason="...")` — merge a modified value

Custom filters compose with the built-in grounding check via
`CompositeMergeFilter` — grounding runs first, then the custom filter.
Set `skip_builtin_grounding: true` to bypass grounding entirely.

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
    reject_unmatched: true        # default: true — reject values with no enum match
```

When `reject_unmatched` is `true` (the default), any value that is not a valid enum entry after normalization is rejected — the field is not merged into wizard data. This prevents invalid values like `"magic"` from satisfying a required field with `enum: [ollama, openai, anthropic]`. The wizard stays at the current stage and can prompt for a valid value.

`reject_unmatched` works independently of normalization. When normalization is disabled (`enum_normalize: false`), it acts as a strict enum membership check — only exact matches are accepted.

Set `reject_unmatched: false` to restore permissive behavior where non-matching values pass through unchanged.

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
    mode:
      type: string
      enum: [interactive, batch, hybrid]
      x-extraction:
        reject_unmatched: false     # allow non-matching values for this field
```

Per-field `normalize` overrides the class-level `enum_normalize` in both directions: `normalize: true` on a field enables normalization even when `enum_normalize: false`, and vice versa. The same applies to `reject_unmatched`.

### Matching Algorithm

The normalization algorithm uses a tiered strategy. The first tier that produces a match wins:

| Tier | Strategy | Example |
|------|----------|---------|
| 1 | **Exact match** (case-sensitive) | `"tutor"` → `"tutor"` |
| 2 | **Case-insensitive** | `"Tutor"`, `"TUTOR"` → `"tutor"` |
| 3 | **Substring** (after `_`/`-` → space normalization) | `"tutor bot"` → `"tutor"`, `"study companion"` → `"study_companion"` |
| 4 | **Token overlap** ≥ threshold | `"interactive quiz"` → `"quiz"` (overlap = 0.5) |

If no tier produces a match and `reject_unmatched` is `true` (the default), the value is rejected — it is not merged into wizard data, leaving the field unset. If `reject_unmatched` is `false`, the original value passes through unchanged.

The `normalize_threshold` controls tier 4 sensitivity. At `0.7` (default), at least 70% of tokens must overlap. At `1.0`, tier 4 requires all tokens to match (effectively disabling fuzzy matching while still allowing tiers 1-3).

---

## Field Derivation Recovery

### The Problem {#derivation-problem}

Some fields have deterministic relationships: `domain_id` and `domain_name` are typically derivable from each other (`chess-champ` ↔ `Chess Champ`). When an extraction model captures one but misses the other, the wizard treats the missing field as unsatisfied — blocking auto-advance or forcing a clarification question for information the framework could infer.

Field derivation fills missing fields from present ones using pure functions — no LLM call, no I/O. This is the cheapest recovery strategy.

### How It Works {#derivation-how-it-works}

Derivation runs in the extraction pipeline **after** merge and schema defaults, and **before** the recovery pipeline and confidence gate:

```
1. Extract from message (SchemaExtractor)
2. Normalize extracted data (type coercion + enum normalization)
3. Merge with existing wizard_data (grounding filter)
4. Apply schema defaults
5. Apply field derivations  ← POST-EXTRACTION PASS (unconditional)
6. Recovery pipeline (if required fields still missing):
   a. derivation            ← no-op if post-extraction pass already filled
   b. boolean_recovery
   c. scope_escalation
   d. focused_retry
7. Confidence gate (required fields check)
8. Transition derivations
9. FSM step
```

The **post-extraction derivation pass** (step 5) runs unconditionally after every extraction, filling derivable fields regardless of whether required fields are missing. This is essential for deriving optional fields from required fields (e.g., `intent=research_assistant` → `kb_enabled=true`) — since all required fields may already be satisfied, the recovery pipeline would never run.

Guard conditions (`target_missing`, `target_empty`, `always`) ensure idempotency — values already set by extraction or defaults are not overwritten.

The recovery pipeline (step 6) may also include derivation as a strategy. If the post-extraction pass already filled a field, the recovery derivation step is a no-op for that field (guard conditions prevent double-write). Recovery derivation is still useful when earlier recovery strategies (e.g., boolean recovery or scope escalation) produce new source values that enable additional derivations.

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

The derivation system provides 22 transforms across 6 categories. See the full reference in the [MkDocs guide](../../docs/packages/bots/guides/context-aware-wizards.md).

#### String Formatting

| Transform | Input → Output | Use Case |
|-----------|---------------|----------|
| `title_case` | `chess-champ` → `Chess Champ` | ID → display name |
| `lower_hyphen` | `Chess Champ` → `chess-champ` | Display name → slug ID |
| `lower_underscore` | `Chess Champ` → `chess_champ` | Display name → snake_case |
| `copy` | Direct copy of source value | Aliased fields |
| `template` | Jinja2 template rendered with wizard data | Composite string derivation |

#### Conditional/Logical

| Transform | Config Fields | Return Type | Description |
|-----------|---------------|-------------|-------------|
| `equals` | `transform_value` | `bool` | `True` if `str(source) == str(transform_value)` |
| `not_equals` | `transform_value` | `bool` | `True` if `str(source) != str(transform_value)` |
| `constant` | `transform_value` | `Any` | Returns `transform_value` regardless of source |
| `map` | `transform_map`, `transform_default` | `Any` | Lookup `str(source)` in map; returns mapped value or default |
| `boolean` | (none) | `bool` | `True` if source is truthy |
| `one_of` | `transform_values` (list) | `bool` | `True` if source is in the values list |
| `contains` | `transform_value` | `bool` | `True` if `transform_value` is a case-insensitive substring of source |

#### Collection

| Transform | Config Fields | Return Type | Description |
|-----------|---------------|-------------|-------------|
| `first` | (none) | `Any` | First element of iterable source |
| `last` | (none) | `Any` | Last element of iterable source |
| `join` | `transform_value` (separator, default `", "`) | `str` | Join list elements into string |
| `split` | `transform_value` (separator, default `","`) | `list[str]` | Split string into list |
| `length` | (none) | `int` | Length of string/list/dict |

#### Regex

| Transform | Config Fields | Return Type | Description |
|-----------|---------------|-------------|-------------|
| `regex_match` | `transform_value` (pattern) | `bool` | `True` if source matches pattern |
| `regex_extract` | `transform_value` (pattern with group) | `str\|None` | First capture group match |
| `regex_replace` | `transform_value`, `transform_replacement` | `str` | Replace all matches |

#### Expression (General-Purpose)

| Transform | Config Fields | Return Type | Description |
|-----------|---------------|-------------|-------------|
| `expression` | `expression` (Python expression) | `Any` | Safe eval with `value`, `data`, `has()` in scope |

The `expression` transform uses the safe expression engine from `dataknobs-common` to evaluate Python expressions and return native types. Unlike `template` (string-only), `expression` returns `bool`, `int`, `list`, etc.

### Configuration Examples

```yaml
settings:
  derivations:
    # Conditional: set flag when intent matches
    - source: intent
      target: kb_enabled
      transform: equals
      transform_value: research_assistant

    # Lookup table: map intent to config
    - source: intent
      target: synthesis_style
      transform: map
      transform_map:
        research_assistant: conversational
        tutor: socratic
      transform_default: structured

    # Expression: complex logic via config
    - source: intent
      target: max_questions
      transform: expression
      expression: "10 if value == 'quiz_maker' else 5"

    # Regex: extract domain from email
    - source: email
      target: email_domain
      transform: regex_extract
      transform_value: "@([\\w.-]+)$"

    # Collection: first topic as primary
    - source: selected_topics
      target: primary_topic
      transform: first
```

### Template Derivation

For deriving a string field from multiple source fields, use the `template` transform with a Jinja2 template:

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

For typed results (boolean, integer, list, etc.), use the `expression` transform instead.

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
    derivation_enabled: false   # suppress both post-extraction and recovery derivation
```

Note: `derivation_enabled: false` suppresses derivation in **both** the post-extraction pass and the recovery pipeline. This is distinct from `recovery_enabled: false`, which only suppresses the recovery pipeline — post-extraction derivation still runs when recovery is disabled.

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

### The Problem {#recovery-problem}
The wizard provides several extraction recovery strategies — [field derivation](#field-derivation-recovery), [scope escalation](#extraction-scope-escalation), and [enum normalization](#enum-normalization) — that each address different failure classes. However, extraction failures are often compound: a single turn might need derivation for one field AND scope escalation for another. Without a composition mechanism, strategies run in a hardcoded sequence with no awareness of each other, potentially wasting LLM calls when a cheaper strategy would have been sufficient.

### How It Works {#recovery-how-it-works}
The recovery pipeline runs after initial extraction and merge, executing strategies in a configurable order. It **short-circuits** as soon as all required fields are satisfied — minimizing LLM calls and latency in the common case.

```
Extract → Normalize → Merge (grounded) → Schema defaults
  → Post-extraction derivation (unconditional — fills optional & required)
  → Recovery pipeline (if required fields still missing):
    1. derivation         [free — no-op for fields already derived above]
    2. boolean_recovery   [free — signal word matching, no LLM call]
    3. scope_escalation   [1 LLM call — broader context]
    4. focused_retry      [1 LLM call — focused prompt]
  → Confidence gate
    → PASS: proceed to transitions
    → FAIL: clarification (ask the user)
```

After each strategy, the pipeline checks whether all required fields are now present. If they are, remaining strategies are skipped.

Key design choices:

- **Post-extraction derivation runs unconditionally** — after merge and schema defaults, before the recovery pipeline check. This ensures optional fields derived from required fields are always filled, even when all required fields are satisfied and recovery is skipped.
- **Schema defaults run before derivation**, not as a pipeline step. Defaults fill preconfigured values that should always apply, so defaulted fields are available as derivation sources and don't trigger unnecessary recovery.
- **Recovery pipeline derivation runs first** (before scope escalation). Derivation is free — pure functions with no LLM call. If derivation fills the missing fields, scope escalation never fires, saving an LLM call. If you have derivation rules that depend on fields only available after escalation, list derivation twice in the pipeline: `["derivation", "scope_escalation", "derivation"]`.
- **Each LLM-backed strategy** (scope escalation, focused retry) runs normalize + merge on its results automatically, including grounding checks.

### Pipeline Configuration 
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
| `recovery.pipeline` | list of strings | `["derivation", "scope_escalation"]` | Ordered list of strategies to execute |
| `recovery.focused_retry.enabled` | bool | `false` | Enable the focused retry strategy |
| `recovery.focused_retry.max_retries` | int | `1` | Maximum focused retry attempts per turn |

Valid strategy names: `derivation`, `boolean_recovery`, `scope_escalation`, `focused_retry`, `clarification`.

The `clarification` strategy is a no-op placeholder — clarification is handled by the confidence gate after the pipeline, regardless of whether it appears in the list. Including it documents intent but doesn't change behavior.

**Default behavior (zero-config):** When no `recovery` settings are provided, the default pipeline runs `derivation → scope_escalation`. Scope escalation requires `scope_escalation.enabled: true` to fire, so without any configuration only derivation runs (if rules are configured). Add `boolean_recovery` or `focused_retry` to the pipeline explicitly to opt in.

### Boolean Recovery Strategy
When extraction fails to produce a value for a boolean field, boolean recovery scans the user's message for affirmative and negative signal words and fills the field deterministically. This is common at confirmation stages where the user says "Yes, save it!" but the extraction model fails to produce a value.

**No LLM call required** — boolean recovery uses word-boundary matching against configurable signal word lists, making it as cheap as derivation.

#### Configuration

Add `boolean_recovery` to the recovery pipeline to enable it:

```yaml
settings:
  recovery:
    pipeline:
      - derivation
      - boolean_recovery      # Must be explicitly added to pipeline
      - scope_escalation
```

Boolean recovery is enabled for all boolean fields by default when the strategy is in the pipeline. To disable it for specific fields, use per-field `x-extraction`:

```yaml
schema:
  properties:
    auto_enabled:
      type: boolean
      x-extraction:
        boolean_recovery: false   # Disable for this specific field
```

Custom signal words can be configured per-field:

```yaml
schema:
  properties:
    confirmed:
      type: boolean
      x-extraction:
        boolean_recovery: true
        affirmative_signals: ["proceed", "ship", "deploy"]
        negative_signals: ["abort", "reject", "rollback"]
```

#### How It Works

1. After initial extraction and merge, the pipeline identifies **required boolean fields** that are still missing and have `boolean_recovery` enabled.
2. For each candidate field, the user's message is scanned for signal words:
   - **Affirmative signals** (→ `True`): "yes", "confirm", "save", "approve", "correct", "sure", "ok", "yep", "yeah", and phrases like "looks good", "go ahead", "sounds good", "i confirm".
   - **Negative signals** (→ `False`): "no", "wait", "stop", "cancel", "wrong", "nope", and phrases like "not yet", "hold on", "start over", "don't save".
3. If both affirmative and negative signals are present, the result depends on signal strength: phrases beat single words. If the conflict cannot be resolved, the field is left unset (no guessing).
4. **Scope restriction**: when multiple boolean fields are missing in the same stage, recovery requires field-specific keywords (from the field name and description) to appear in the message. This prevents a generic "yes" from filling unrelated boolean fields. When only one boolean field is missing, this restriction is relaxed.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `extraction_hints.boolean_recovery` | bool | `true` | Enable boolean recovery for all boolean fields (only active when strategy is in pipeline) |
| `x-extraction.boolean_recovery` | bool | inherits class | Per-field override |
| `x-extraction.affirmative_signals` | list[str] | built-in set | Override affirmative signal words |
| `x-extraction.affirmative_phrases` | list[str] | built-in set | Override affirmative phrases |
| `x-extraction.negative_signals` | list[str] | built-in set | Override negative signal words |
| `x-extraction.negative_phrases` | list[str] | built-in set | Override negative phrases |

### Focused Retry Strategy
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

### Per-Stage Override {#recovery-per-stage-override}
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

### Pipeline Examples 
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
  extraction_hints:
    boolean_recovery: true
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
      - boolean_recovery
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

### Clarification Grouping

When the confidence gate fires and the wizard asks for missing fields, the default behavior generates a generic clarification prompt. Clarification grouping improves this by organizing related missing fields into natural questions.

#### Configuration

```yaml
settings:
  recovery:
    clarification:
      exclude_derivable: true     # Don't ask about fields that can be derived
      groups:
        - fields: [domain_id, domain_name]
          question: "What would you like to call your bot?"
        - fields: [llm_provider, llm_model]
          question: "Which LLM provider should the bot use?"
      template: |                 # Optional Jinja2 template
        I have most of your configuration. Could you also tell me:
        {% for group in field_groups %}
        - {{ group.question }}
        {% endfor %}
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `recovery.clarification.groups` | list of objects | `[]` | Field groups with `fields` (list) and `question` (string) |
| `recovery.clarification.exclude_derivable` | bool | `false` | Exclude fields that have derivation rules from clarification |
| `recovery.clarification.template` | string | `null` | Optional Jinja2 template for rendering grouped questions |

#### How It Works

When the confidence gate fires, the wizard identifies which required fields are still missing and:

1. **Excludes derivable fields** (if `exclude_derivable: true`). Fields with configured derivation rules are omitted — they'll be derived once the source field is provided, so asking the user is unnecessary. This applies even when the source field is also missing: the clarification prompt will ask for the source, and once the user provides it, derivation fills the target automatically.

2. **Matches missing fields to configured groups.** If a group contains missing fields, those fields are bundled into a single question. Only the *missing* fields from each group are included — if `domain_id` is already present but `domain_name` is missing, the group still fires but only for `domain_name`.

3. **Generates individual questions for ungrouped fields.** Missing fields that don't belong to any group get their own question, derived from their JSON Schema `description` (or their field name if no description is set).

4. **Renders the template** (if configured). The `template` receives a `field_groups` variable — a list of dicts with `fields` and `question` keys. If no template is set, the questions are rendered as a simple bullet list.

When no groups are configured, the existing behavior is preserved — the wizard generates a generic clarification prompt from the extraction issues.

---

## Transition Re-Extraction

### The Problem {#re-extraction-problem}

When a wizard transitions from stage A to stage B mid-turn, the user's message is only extracted against stage A's schema. Any data in the message relevant to stage B's schema is lost, forcing the user to repeat information in a second message.

For example, in an edit-back flow the user says "Change the tone to formal" at a finalize stage. The finalize stage extracts the routing field (`edit_section: "options"`) and transitions to the options stage — but "formal" is never extracted because it belongs to the options stage's schema, not finalize's.

### Configuration {#re-extraction-configuration}

Enable re-extraction on a stage with the `re_extract_on_entry` field:

```yaml
stages:
  - name: configure_options
    re_extract_on_entry: true   # Re-extract + relax auto-advance gates
    schema:
      properties:
        tone: { type: string, enum: [formal, casual, academic] }
      required: [tone]
    transitions:
      - target: finalize
        condition: "data.get('tone')"
```

Default: disabled (absent). Accepted values:

| Value | Extraction | Auto-advance gates |
|-------|------------|-------------------|
| `true` | Runs | Relaxed (Gate 1 + Gate 2 bypassed; only Gate 3 enforced) |
| `"capture_only"` | Runs | **Not relaxed** — all gates remain enforced |
| `false` / absent | Does not run | N/A |

When enabled (`true` or `"capture_only"`), the wizard re-runs the full extraction pipeline against this stage's schema using the same user message that triggered the transition. The extraction pipeline includes normalization, grounded merge, defaults, derivations, and recovery — the same steps as a normal extraction.

Use `"capture_only"` when the stage should absorb data from the transition message but the user must explicitly proceed — for example, confirmation or review stages where the bot should display the captured data before advancing.

### How It Works {#re-extraction-how-it-works}

The re-extraction runs after the FSM transition fires but before the post-transition lifecycle (auto-advance, hooks):

```
1. User message arrives at source stage
2. Extract from message using source stage's schema
3. Evaluate transitions → transition fires
4. Enter target stage
5. ★ Re-extract from the SAME message using target stage's schema
6. Run post-transition lifecycle (auto-advance, hooks)
7. Generate response
```

This ordering is critical: re-extracted data is available in `wizard_state.data` when auto-advance evaluates transition conditions at step 6. This enables single-turn edit-back flows:

1. **Finalize** extracts `edit_section: "options"` → transitions to **configure_options**
2. **configure_options** re-extracts `tone: "formal"` from the same message
3. Auto-advance evaluates transitions → `tone` is present → advances to **finalize**
4. Config re-saved with updated tone — single turn

Re-extraction is silently skipped when:
- No transition occurred (stayed at the same stage)
- No user message is available (e.g., greet)
- The target stage has no schema

### Interaction with auto_advance: false {#re-extraction-auto-advance}

`auto_advance: false` means "don't auto-advance during normal turn processing" — it does **not** mean "never advance under any circumstances". When `re_extract_on_entry: true` captures data at a stage, both the `auto_advance` gate and the **required-fields gate** are relaxed for that stage because the user has already expressed intent via the triggering message.

This relaxation applies only to the **immediate landing stage** (the stage where re-extraction ran). If auto-advance chains through additional stages, those stages' own `auto_advance` settings and required-fields checks apply normally.

Only the **transition condition** gate remains in effect after re-extraction — it encodes the domain logic about when advancement is appropriate. The stage must also not be an end stage. The `auto_advance` and required-fields checks are bypassed because unfilled optional fields (e.g., `llm_model = ''` in a stage with `required: []`) should not block advancement when the transition condition is satisfied. Note: stages with unconditional transitions (no `condition` on the transition) will always advance after re-extraction when using `re_extract_on_entry: true`, since Gate 3 passes unconditionally.

**`"capture_only"` mode:** If you want re-extraction to capture data but NOT relax auto-advance gates, use `re_extract_on_entry: "capture_only"`. This is useful for stages where the user should review captured data before proceeding — for example, confirmation stages or progressive-disclosure forms where you want the bot to display what was captured and let the user decide when to continue.

If re-extraction produces **no data** (empty extraction result), the `auto_advance: false` gate is **not** relaxed and the wizard stays at the landing stage (regardless of `true` vs. `"capture_only"`).

**Limitation:** Re-extraction runs only once per turn, for the stage entered by the initial transition. If re-extraction fills the target stage's required fields and auto-advance fires to a subsequent stage that also has `re_extract_on_entry`, that subsequent stage does **not** re-extract. Design edit-back flows so the re-extraction target is the final destination, not a waypoint.

The `advance()` API also supports re-extraction when called with string input (extract mode). Dict input has no raw message to re-extract from, so `re_extract_on_entry` is a no-op.

### Interaction with skip_extraction

`re_extract_on_entry` and `skip_extraction` serve opposite purposes:

| Flag | Purpose | Set by |
|------|---------|--------|
| `skip_extraction` | Prevent extraction at a landing stage after auto-advance | Auto-advance loop (one-shot) |
| `re_extract_on_entry` | Enable extraction at a landing stage from the transition message | Stage config (permanent) |

When both flags coexist on a state object, re-extraction takes priority and clears `skip_extraction`. This ensures the next turn's extraction runs normally rather than being skipped.

In the conversational path (`generate()`/`finalize_turn()`), `process_input` already clears `skip_extraction` before re-extraction runs, so this interaction is only reachable via the `advance()` API — for example, when a prior `advance()` call triggered auto-advance (setting `skip_extraction`) and the next `advance()` call lands on a stage with `re_extract_on_entry`.

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

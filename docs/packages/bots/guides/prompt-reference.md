# Prompt Key Reference

Complete catalog of all configurable prompt keys in `dataknobs-bots` and
`dataknobs-llm`. Each key can be overridden via bot configuration to
customize LLM behavior without code changes.

## Overview

Prompts are organized into **fragments** (simple text with variable
placeholders) and **meta-prompts** (Jinja2 templates that compose
fragments via `prompt_ref()`). Override a fragment to change one piece;
override a meta-prompt to restructure the composition.

- **Syntax `format`** -- uses `{var}` placeholders (Python `.format()`)
- **Syntax `jinja2`** -- uses `{{ var }}`, `{% if %}`, `prompt_ref()`, etc.

Total: **77 keys** (62 fragments + 15 meta-prompts) across 7 namespaces.

---

## Wizard Response Prompts (`wizard.*`)

Used by `WizardResponder` to generate clarification, validation,
transform error, and restart offer responses during wizard flows.

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `wizard.clarification` | meta | jinja2 | `issue_list`, `stage_prompt`, `suggestions_text` |
| `wizard.clarification.header` | fragment | format | -- |
| `wizard.clarification.preamble` | fragment | format | -- |
| `wizard.clarification.issues` | fragment | format | `issue_list` |
| `wizard.clarification.goal` | fragment | format | `stage_prompt`, `suggestions_text` |
| `wizard.clarification.instructions` | fragment | format | -- |
| `wizard.validation` | meta | jinja2 | `error_list`, `stage_prompt` |
| `wizard.validation.header` | fragment | format | -- |
| `wizard.validation.issues` | fragment | format | `error_list` |
| `wizard.validation.goal` | fragment | format | `stage_prompt` |
| `wizard.validation.instructions` | fragment | format | -- |
| `wizard.transform_error` | meta | jinja2 | `stage_name`, `error` |
| `wizard.transform_error.header` | fragment | format | -- |
| `wizard.transform_error.detail` | fragment | format | `stage_name`, `error` |
| `wizard.transform_error.instructions` | fragment | format | -- |
| `wizard.restart_offer` | meta | jinja2 | `stage_name`, `stage_prompt` |
| `wizard.restart_offer.header` | fragment | format | -- |
| `wizard.restart_offer.status` | fragment | format | `stage_name`, `stage_prompt` |
| `wizard.restart_offer.options` | fragment | format | -- |
| `wizard.restart_offer.instructions` | fragment | format | -- |

## Memory Prompts (`memory.*`)

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `memory.summary` | fragment | format | `existing_summary`, `new_messages` |

## Rubric Prompts (`rubric.*`)

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `rubric.feedback_summary.system` | fragment | format | -- |
| `rubric.feedback_summary.user` | fragment | format | `rubric_name`, `rubric_description`, `pass_fail`, `weighted_score`, `results_text` |
| `rubric.classification` | fragment | format | `criterion_name`, `criterion_description`, `level_descriptions`, `valid_ids`, `example_level_id` |

## Review Persona Prompts (`review.*`)

Shared fragments used by all personas, plus per-persona fragments and
meta-prompts. Built-in personas: adversarial, skeptical, insightful,
minimalist, downstream.

### Shared Fragments

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `review.format.response` | fragment | format | -- |
| `review.artifact_section` | fragment | format | `artifact_type`, `artifact_name`, `artifact_purpose`, `artifact_content` |

### Per-Persona Keys (pattern: `review.persona.{name}.*`)

Each persona has four keys following this pattern:

| Key Pattern | Type | Syntax | Placeholders |
|-------------|------|--------|--------------|
| `review.persona.{name}` | meta | jinja2 | `artifact_type`, `artifact_name`, `artifact_purpose`, `artifact_content` |
| `review.persona.{name}.role` | fragment | format | -- |
| `review.persona.{name}.focus` | fragment | format | -- |
| `review.persona.{name}.instructions` | fragment | format | -- |

Built-in `{name}` values: `adversarial`, `skeptical`, `insightful`,
`minimalist`, `downstream`.

## Grounded Synthesis Prompts (`grounded.*`)

Used by `GroundedReasoning` to build synthesis system prompts and
provenance templates.

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `grounded.synthesis` | meta | jinja2 | `require_citations`, `citation_format`, `allow_parametric` |
| `grounded.synthesis.base_instruction` | fragment | format | -- |
| `grounded.synthesis.citation_section` | fragment | format | -- |
| `grounded.synthesis.citation_source` | fragment | format | -- |
| `grounded.synthesis.bridge` | fragment | format | -- |
| `grounded.synthesis.strict` | fragment | format | -- |
| `grounded.synthesis.supplement` | fragment | format | -- |
| `grounded.synthesis.kb_wrapper` | fragment | format | `kb_context` |
| `grounded.provenance_template` | template | jinja2 | `results`, `results_by_source` |

## Focus Guard Prompts (`focus.*`)

Used by `FocusGuard` to generate focus guidance and drift correction
prompts.

### Guidance

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `focus.guidance` | meta | jinja2 | `primary_goal`, `current_task`*, `required_fields`*, `collected`* |
| `focus.guidance.header` | fragment | format | -- |
| `focus.guidance.goal` | fragment | format | `primary_goal` |
| `focus.guidance.task` | fragment | format | `current_task` |
| `focus.guidance.needed` | fragment | format | `required_fields` |
| `focus.guidance.collected` | fragment | format | `collected` |
| `focus.guidance.instructions` | fragment | format | -- |

*Optional -- guarded by `{% if %}` conditionals in the meta-prompt.

### Drift Correction

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `focus.drift` | meta | jinja2 | `reason`*, `suggested_redirect`*, `tangent_count`, `max_tangent_depth` |
| `focus.drift.header` | fragment | format | -- |
| `focus.drift.issue` | fragment | format | `reason` |
| `focus.drift.redirect` | fragment | format | `suggested_redirect` |
| `focus.drift.firm_message` | fragment | format | `tangent_count` |
| `focus.drift.gentle_message` | fragment | format | -- |

## Extraction Prompts (`extraction.*`)

Defined in `dataknobs-llm`. Used by `SchemaExtractor` for structured
data extraction.

| Key | Type | Syntax | Placeholders |
|-----|------|--------|--------------|
| `extraction.default` | meta | jinja2 | `schema`, `context`, `text` |
| `extraction.default.schema_section` | fragment | format | `schema` |
| `extraction.default.context_section` | fragment | format | `context` |
| `extraction.default.instructions` | fragment | format | -- |
| `extraction.default.message_section` | fragment | format | `text` |
| `extraction.with_assumptions` | meta | jinja2 | `schema`, `context`, `text` |
| `extraction.with_assumptions.instructions` | fragment | format | -- |
| `extraction.with_assumptions.example` | fragment | format | -- |
| `extraction.with_assumptions.message_section` | fragment | format | `text` |

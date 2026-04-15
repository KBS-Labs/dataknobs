"""Default extraction prompts as PromptTemplateDict constants.

This module stores the default extraction prompts that were previously
hardcoded inline in ``schema_extractor.py``. Each prompt is decomposed into
fine-grained fragments and a meta-prompt that composes them via
``prompt_ref()``. Consumers can override any fragment or the entire
meta-prompt via the prompt library configuration.

All fragment templates use ``.format()`` syntax (``{var}``) since they are
simple variable substitution. Meta-prompts use Jinja2 syntax to leverage
``prompt_ref()`` for composition.

Prompt key namespace:

- ``extraction.default.*`` — default extraction prompt fragments
- ``extraction.default`` — default extraction meta-prompt
- ``extraction.with_assumptions.*`` — assumption-tracking variant fragments
- ``extraction.with_assumptions`` — assumption-tracking meta-prompt
"""

import logging

from dataknobs_llm.prompts import ConfigPromptLibrary
from dataknobs_llm.prompts.base.types import PromptTemplateDict

logger = logging.getLogger(__name__)


# ============================================================================
# Default extraction prompt fragments
# ============================================================================

EXTRACTION_DEFAULT_SCHEMA_SECTION: PromptTemplateDict = {
    "template": (
        "## Schema\n"
        "Extract data matching this JSON Schema:\n"
        "```json\n"
        "{schema}\n"
        "```"
    ),
    "template_syntax": "format",
}

EXTRACTION_DEFAULT_CONTEXT_SECTION: PromptTemplateDict = {
    "template": "## Context\n{context}",
    "template_syntax": "format",
}

EXTRACTION_DEFAULT_INSTRUCTIONS: PromptTemplateDict = {
    "template": (
        "## Instructions\n"
        "1. Parse the user's message and extract relevant information\n"
        "2. Return ONLY a valid JSON object matching the schema\n"
        "3. If the user did NOT mention a field at all, omit it\n"
        "4. If you cannot extract the required information, return an empty object {{}}\n"
        "5. Do not include explanations - only return the JSON object\n"
        "6. For boolean fields: \"yes\"/\"enable\"/\"add\" → true; "
        "\"no\"/\"disable\"/\"skip\"/\"none\" → false\n"
        "7. Negations count as explicit values: \"no knowledge base\" → kb_enabled: false\n"
        "8. For array fields with enum constraints, \"all\" means include every enum "
        "value; \"none\" means an empty array\n"
        "9. For array fields, always return a JSON array (e.g. [\"value\"]), never a bare string"
    ),
    "template_syntax": "format",
}

EXTRACTION_DEFAULT_MESSAGE_SECTION: PromptTemplateDict = {
    "template": "## User Message\n{text}\n\n## Extracted Data (JSON only):",
    "template_syntax": "format",
}

# Meta-prompt: composes the default extraction fragments
EXTRACTION_DEFAULT_META: PromptTemplateDict = {
    "template": (
        "Extract structured data from the user's message.\n\n"
        '{{ prompt_ref("extraction.default.schema_section", schema=schema) }}\n\n'
        '{{ prompt_ref("extraction.default.context_section", context=context) }}\n\n'
        '{{ prompt_ref("extraction.default.instructions") }}\n\n'
        '{{ prompt_ref("extraction.default.message_section", text=text) }}'
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Extraction with assumptions prompt fragments
# ============================================================================

EXTRACTION_ASSUMPTIONS_INSTRUCTIONS: PromptTemplateDict = {
    "template": (
        "## Instructions\n"
        "1. Parse the user's message and extract relevant information\n"
        "2. Identify any assumptions you make about:\n"
        "   - Ambiguous terms or references\n"
        "   - Implied but not explicitly stated information\n"
        "   - Default values used when information is missing\n"
        "   - Interpretations that could have multiple meanings\n"
        "3. If the user did NOT mention a field at all, omit it from \"data\"\n"
        "4. For boolean fields: \"yes\"/\"enable\"/\"add\" → true; "
        "\"no\"/\"disable\"/\"skip\"/\"none\" → false\n"
        "5. Negations count as explicit values: \"no knowledge base\" → kb_enabled: false\n"
        "6. For array fields with enum constraints, \"all\" means include every enum "
        "value; \"none\" means an empty array\n"
        "7. For array fields, always return a JSON array (e.g. [\"value\"]), never a bare string\n"
        "8. Return a JSON object with two keys:\n"
        "   - \"data\": The extracted data matching the schema\n"
        "   - \"assumptions\": Array of assumption objects with:\n"
        "     - \"content\": Description of the assumption\n"
        "     - \"field\": Which field this relates to (or null if general)\n"
        "     - \"confidence\": How confident you are (0.0-1.0)"
    ),
    "template_syntax": "format",
}

EXTRACTION_ASSUMPTIONS_EXAMPLE: PromptTemplateDict = {
    "template": (
        "## Example Output\n"
        "```json\n"
        "{{\n"
        '  "data": {{"name": "John", "age": 30}},\n'
        '  "assumptions": [\n'
        '    {{"content": "Assumed \'John\' refers to a person\'s first name", '
        '"field": "name", "confidence": 0.9}},\n'
        '    {{"content": "Age of 30 was mentioned in a different context but '
        'likely applies here", "field": "age", "confidence": 0.7}}\n'
        "  ]\n"
        "}}\n"
        "```"
    ),
    "template_syntax": "format",
}

EXTRACTION_ASSUMPTIONS_MESSAGE_SECTION: PromptTemplateDict = {
    "template": (
        "## User Message\n{text}\n\n"
        "## Extracted Data and Assumptions (JSON only):"
    ),
    "template_syntax": "format",
}

# Meta-prompt: composes the assumptions extraction fragments
EXTRACTION_ASSUMPTIONS_META: PromptTemplateDict = {
    "template": (
        "Extract structured data from the user's message "
        "and identify any assumptions made.\n\n"
        '{{ prompt_ref("extraction.default.schema_section", schema=schema) }}\n\n'
        '{{ prompt_ref("extraction.default.context_section", context=context) }}\n\n'
        '{{ prompt_ref("extraction.with_assumptions.instructions") }}\n\n'
        '{{ prompt_ref("extraction.with_assumptions.example") }}\n\n'
        '{{ prompt_ref("extraction.with_assumptions.message_section", text=text) }}'
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Backward-compatible flat prompt constants
# ============================================================================

# These reproduce the exact text of the original hardcoded prompts for use
# in SchemaExtractor._build_extraction_prompt() which calls .format() directly.
# They are kept in sync with the fragments above — the fragments are the
# source of truth; these constants exist only for backward compatibility
# during the migration period.

DEFAULT_EXTRACTION_PROMPT = """Extract structured data from the user's message.

## Schema
Extract data matching this JSON Schema:
```json
{schema}
```

## Context
{context}

## Instructions
1. Parse the user's message and extract relevant information
2. Return ONLY a valid JSON object matching the schema
3. If the user did NOT mention a field at all, omit it
4. If you cannot extract the required information, return an empty object {{}}
5. Do not include explanations - only return the JSON object
6. For boolean fields: "yes"/"enable"/"add" → true; "no"/"disable"/"skip"/"none" → false
7. Negations count as explicit values: "no knowledge base" → kb_enabled: false
8. For array fields with enum constraints, "all" means include every enum value; "none" means an empty array
9. For array fields, always return a JSON array (e.g. ["value"]), never a bare string

## User Message
{text}

## Extracted Data (JSON only):"""

EXTRACTION_WITH_ASSUMPTIONS_PROMPT = """Extract structured data from the user's message and identify any assumptions made.

## Schema
Extract data matching this JSON Schema:
```json
{schema}
```

## Context
{context}

## Instructions
1. Parse the user's message and extract relevant information
2. Identify any assumptions you make about:
   - Ambiguous terms or references
   - Implied but not explicitly stated information
   - Default values used when information is missing
   - Interpretations that could have multiple meanings
3. If the user did NOT mention a field at all, omit it from "data"
4. For boolean fields: "yes"/"enable"/"add" → true; "no"/"disable"/"skip"/"none" → false
5. Negations count as explicit values: "no knowledge base" → kb_enabled: false
6. For array fields with enum constraints, "all" means include every enum value; "none" means an empty array
7. For array fields, always return a JSON array (e.g. ["value"]), never a bare string
8. Return a JSON object with two keys:
   - "data": The extracted data matching the schema
   - "assumptions": Array of assumption objects with:
     - "content": Description of the assumption
     - "field": Which field this relates to (or null if general)
     - "confidence": How confident you are (0.0-1.0)

## Example Output
```json
{{
  "data": {{"name": "John", "age": 30}},
  "assumptions": [
    {{"content": "Assumed 'John' refers to a person's first name", "field": "name", "confidence": 0.9}},
    {{"content": "Age of 30 was mentioned in a different context but likely applies here", "field": "age", "confidence": 0.7}}
  ]
}}
```

## User Message
{text}

## Extracted Data and Assumptions (JSON only):"""


# ============================================================================
# ExtractionPromptLibrary
# ============================================================================

def get_extraction_prompt_library() -> ConfigPromptLibrary:
    """Create a ConfigPromptLibrary with all default extraction prompts.

    Returns:
        A ``ConfigPromptLibrary`` with all extraction prompt fragments and
        meta-prompts registered as system prompts under the ``extraction.*``
        namespace.
    """
    return ConfigPromptLibrary(config={
        "system": {
            # Default extraction fragments
            "extraction.default.schema_section": EXTRACTION_DEFAULT_SCHEMA_SECTION,
            "extraction.default.context_section": EXTRACTION_DEFAULT_CONTEXT_SECTION,
            "extraction.default.instructions": EXTRACTION_DEFAULT_INSTRUCTIONS,
            "extraction.default.message_section": EXTRACTION_DEFAULT_MESSAGE_SECTION,
            "extraction.default": EXTRACTION_DEFAULT_META,
            # Assumptions extraction fragments
            "extraction.with_assumptions.instructions": EXTRACTION_ASSUMPTIONS_INSTRUCTIONS,
            "extraction.with_assumptions.example": EXTRACTION_ASSUMPTIONS_EXAMPLE,
            "extraction.with_assumptions.message_section": EXTRACTION_ASSUMPTIONS_MESSAGE_SECTION,
            "extraction.with_assumptions": EXTRACTION_ASSUMPTIONS_META,
        },
    })

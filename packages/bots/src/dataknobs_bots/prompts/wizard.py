"""Wizard response prompt fragments and meta-prompts.

Prompt keys defined here:

- ``wizard.clarification`` — Meta-prompt for clarification responses
- ``wizard.clarification.*`` — Individual fragments
- ``wizard.validation`` — Meta-prompt for validation error responses
- ``wizard.validation.*`` — Individual fragments
- ``wizard.transform_error`` — Meta-prompt for transform error responses
- ``wizard.transform_error.*`` — Individual fragments
- ``wizard.restart_offer`` — Meta-prompt for restart offer responses
- ``wizard.restart_offer.*`` — Individual fragments

All fragment templates use ``.format()`` syntax since they are simple
variable substitution. Meta-prompts use Jinja2 syntax for ``prompt_ref()``
composition.
"""

from dataknobs_llm.prompts.base.types import PromptTemplateDict


# ============================================================================
# Clarification response fragments
# ============================================================================

WIZARD_CLARIFICATION_HEADER: PromptTemplateDict = {
    "template": "## Clarification Needed",
    "template_syntax": "format",
}

WIZARD_CLARIFICATION_PREAMBLE: PromptTemplateDict = {
    "template": (
        "I wasn't able to clearly understand the user's response for this stage."
    ),
    "template_syntax": "format",
}

WIZARD_CLARIFICATION_ISSUES: PromptTemplateDict = {
    "template": "**Potential Issues**:\n{issue_list}",
    "template_syntax": "format",
}

WIZARD_CLARIFICATION_GOAL: PromptTemplateDict = {
    "template": "**What I'm Looking For**: {stage_prompt}{suggestions_text}",
    "template_syntax": "format",
}

WIZARD_CLARIFICATION_INSTRUCTIONS: PromptTemplateDict = {
    "template": (
        "Please ask a clarifying question to help gather the needed information.\n"
        "Be conversational and helpful - don't make the user feel like they did "
        "something wrong."
    ),
    "template_syntax": "format",
}

WIZARD_CLARIFICATION_META: PromptTemplateDict = {
    "template": (
        '{{ prompt_ref("wizard.clarification.header") }}\n\n'
        '{{ prompt_ref("wizard.clarification.preamble") }}\n\n'
        '{{ prompt_ref("wizard.clarification.issues", issue_list=issue_list) }}\n\n'
        '{{ prompt_ref("wizard.clarification.goal", '
        "stage_prompt=stage_prompt, suggestions_text=suggestions_text) }}\n\n"
        '{{ prompt_ref("wizard.clarification.instructions") }}'
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Validation response fragments
# ============================================================================

WIZARD_VALIDATION_HEADER: PromptTemplateDict = {
    "template": "## Validation Required",
    "template_syntax": "format",
}

WIZARD_VALIDATION_ISSUES: PromptTemplateDict = {
    "template": (
        "The user's input for this stage needs clarification:\n\n"
        "**Issues**:\n{error_list}"
    ),
    "template_syntax": "format",
}

WIZARD_VALIDATION_GOAL: PromptTemplateDict = {
    "template": "**What's Needed**: {stage_prompt}",
    "template_syntax": "format",
}

WIZARD_VALIDATION_INSTRUCTIONS: PromptTemplateDict = {
    "template": (
        "Please kindly ask the user to provide the missing or corrected "
        "information.\n"
        "Be specific about what's needed but remain friendly and helpful."
    ),
    "template_syntax": "format",
}

WIZARD_VALIDATION_META: PromptTemplateDict = {
    "template": (
        '{{ prompt_ref("wizard.validation.header") }}\n\n'
        '{{ prompt_ref("wizard.validation.issues", error_list=error_list) }}\n\n'
        '{{ prompt_ref("wizard.validation.goal", stage_prompt=stage_prompt) }}\n\n'
        '{{ prompt_ref("wizard.validation.instructions") }}'
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Transform error response fragments
# ============================================================================

WIZARD_TRANSFORM_ERROR_HEADER: PromptTemplateDict = {
    "template": "## Processing Error",
    "template_syntax": "format",
}

WIZARD_TRANSFORM_ERROR_DETAIL: PromptTemplateDict = {
    "template": (
        'An error occurred while processing the transition from the '
        '"{stage_name}" stage:\n\n'
        "**Error**: {error}"
    ),
    "template_syntax": "format",
}

WIZARD_TRANSFORM_ERROR_INSTRUCTIONS: PromptTemplateDict = {
    "template": (
        "Please apologize for the issue and let the user know they can try again.\n"
        "If the error suggests a configuration or system issue, suggest they "
        "contact support.\n"
        "Be concise and helpful."
    ),
    "template_syntax": "format",
}

WIZARD_TRANSFORM_ERROR_META: PromptTemplateDict = {
    "template": (
        '{{ prompt_ref("wizard.transform_error.header") }}\n\n'
        '{{ prompt_ref("wizard.transform_error.detail", '
        "stage_name=stage_name, error=error) }}\n\n"
        '{{ prompt_ref("wizard.transform_error.instructions") }}'
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Restart offer response fragments
# ============================================================================

WIZARD_RESTART_OFFER_HEADER: PromptTemplateDict = {
    "template": "## Multiple Clarification Attempts",
    "template_syntax": "format",
}

WIZARD_RESTART_OFFER_STATUS: PromptTemplateDict = {
    "template": (
        "We've had difficulty understanding the responses for this stage.\n\n"
        "**Current Stage**: {stage_name}\n"
        "**Goal**: {stage_prompt}"
    ),
    "template_syntax": "format",
}

WIZARD_RESTART_OFFER_OPTIONS: PromptTemplateDict = {
    "template": (
        "Please offer the user two options:\n"
        "1. Try one more time with clearer instructions\n"
        '2. Start the wizard over from the beginning (type "restart")'
    ),
    "template_syntax": "format",
}

WIZARD_RESTART_OFFER_INSTRUCTIONS: PromptTemplateDict = {
    "template": (
        "Be empathetic and helpful - acknowledge that the questions might not "
        "be clear."
    ),
    "template_syntax": "format",
}

WIZARD_RESTART_OFFER_META: PromptTemplateDict = {
    "template": (
        '{{ prompt_ref("wizard.restart_offer.header") }}\n\n'
        '{{ prompt_ref("wizard.restart_offer.status", '
        "stage_name=stage_name, stage_prompt=stage_prompt) }}\n\n"
        '{{ prompt_ref("wizard.restart_offer.options") }}\n\n'
        '{{ prompt_ref("wizard.restart_offer.instructions") }}'
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Prompt key registry
# ============================================================================

WIZARD_PROMPT_KEYS: dict[str, PromptTemplateDict] = {
    # Clarification
    "wizard.clarification.header": WIZARD_CLARIFICATION_HEADER,
    "wizard.clarification.preamble": WIZARD_CLARIFICATION_PREAMBLE,
    "wizard.clarification.issues": WIZARD_CLARIFICATION_ISSUES,
    "wizard.clarification.goal": WIZARD_CLARIFICATION_GOAL,
    "wizard.clarification.instructions": WIZARD_CLARIFICATION_INSTRUCTIONS,
    "wizard.clarification": WIZARD_CLARIFICATION_META,
    # Validation
    "wizard.validation.header": WIZARD_VALIDATION_HEADER,
    "wizard.validation.issues": WIZARD_VALIDATION_ISSUES,
    "wizard.validation.goal": WIZARD_VALIDATION_GOAL,
    "wizard.validation.instructions": WIZARD_VALIDATION_INSTRUCTIONS,
    "wizard.validation": WIZARD_VALIDATION_META,
    # Transform error
    "wizard.transform_error.header": WIZARD_TRANSFORM_ERROR_HEADER,
    "wizard.transform_error.detail": WIZARD_TRANSFORM_ERROR_DETAIL,
    "wizard.transform_error.instructions": WIZARD_TRANSFORM_ERROR_INSTRUCTIONS,
    "wizard.transform_error": WIZARD_TRANSFORM_ERROR_META,
    # Restart offer
    "wizard.restart_offer.header": WIZARD_RESTART_OFFER_HEADER,
    "wizard.restart_offer.status": WIZARD_RESTART_OFFER_STATUS,
    "wizard.restart_offer.options": WIZARD_RESTART_OFFER_OPTIONS,
    "wizard.restart_offer.instructions": WIZARD_RESTART_OFFER_INSTRUCTIONS,
    "wizard.restart_offer": WIZARD_RESTART_OFFER_META,
}

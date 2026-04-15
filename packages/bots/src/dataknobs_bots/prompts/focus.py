"""Focus guard prompt fragments and meta-prompts.

Prompt keys defined here:

- ``focus.guidance`` — Meta-prompt for focus guidance (Jinja2 conditionals)
- ``focus.guidance.*`` — Individual focus guidance fragments
- ``focus.drift`` — Meta-prompt for drift correction (Jinja2 conditionals)
- ``focus.drift.*`` — Individual drift correction fragments

The focus prompts use Jinja2 conditionals in their meta-prompts because
fragment inclusion depends on runtime context (whether current_task exists,
whether collected_data is non-empty, tangent count thresholds, etc.).
"""

from dataknobs_llm.prompts.base.types import PromptTemplateDict


# ============================================================================
# Focus guidance fragments
# ============================================================================

FOCUS_GUIDANCE_HEADER: PromptTemplateDict = {
    "template": "## Focus Guidance",
    "template_syntax": "format",
}

FOCUS_GUIDANCE_GOAL: PromptTemplateDict = {
    "template": "**Primary Goal**: {primary_goal}",
    "template_syntax": "format",
}

FOCUS_GUIDANCE_TASK: PromptTemplateDict = {
    "template": "**Current Task**: {current_task}",
    "template_syntax": "format",
}

FOCUS_GUIDANCE_NEEDED: PromptTemplateDict = {
    "template": "**Still Needed**: {required_fields}",
    "template_syntax": "format",
}

FOCUS_GUIDANCE_COLLECTED: PromptTemplateDict = {
    "template": "**Already Have**: {collected}",
    "template_syntax": "format",
}

FOCUS_GUIDANCE_INSTRUCTIONS: PromptTemplateDict = {
    "template": (
        "Stay focused on the current task. If the user asks about "
        "something unrelated, acknowledge briefly and redirect to "
        "the task at hand."
    ),
    "template_syntax": "format",
}

# Meta-prompt: composes guidance fragments with conditionals
FOCUS_GUIDANCE_META: PromptTemplateDict = {
    "template": (
        '{{ prompt_ref("focus.guidance.header") }}\n'
        '{{ prompt_ref("focus.guidance.goal", primary_goal=primary_goal) }}'
        "{% if current_task %}\n"
        '{{ prompt_ref("focus.guidance.task", current_task=current_task) }}'
        "{% endif %}"
        "{% if required_fields %}\n"
        '{{ prompt_ref("focus.guidance.needed", required_fields=required_fields) }}'
        "{% endif %}"
        "{% if collected %}\n"
        '{{ prompt_ref("focus.guidance.collected", collected=collected) }}'
        "{% endif %}\n\n"
        '{{ prompt_ref("focus.guidance.instructions") }}'
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Focus drift correction fragments
# ============================================================================

FOCUS_DRIFT_HEADER: PromptTemplateDict = {
    "template": "## Focus Correction Needed",
    "template_syntax": "format",
}

FOCUS_DRIFT_ISSUE: PromptTemplateDict = {
    "template": "Issue: {reason}",
    "template_syntax": "format",
}

FOCUS_DRIFT_REDIRECT: PromptTemplateDict = {
    "template": "Redirect to: {suggested_redirect}",
    "template_syntax": "format",
}

FOCUS_DRIFT_FIRM_MESSAGE: PromptTemplateDict = {
    "template": (
        "**IMPORTANT**: The conversation has drifted off-topic for "
        "{tangent_count} turns. Please acknowledge the "
        "tangent briefly and firmly redirect to the main task."
    ),
    "template_syntax": "format",
}

FOCUS_DRIFT_GENTLE_MESSAGE: PromptTemplateDict = {
    "template": (
        "Please gently steer the conversation back to the main topic "
        "while acknowledging what the user mentioned."
    ),
    "template_syntax": "format",
}

# Meta-prompt: composes drift correction with conditionals
FOCUS_DRIFT_META: PromptTemplateDict = {
    "template": (
        '{{ prompt_ref("focus.drift.header") }}'
        "{% if reason %}\n"
        '{{ prompt_ref("focus.drift.issue", reason=reason) }}'
        "{% endif %}"
        "{% if suggested_redirect %}\n"
        '{{ prompt_ref("focus.drift.redirect", '
        "suggested_redirect=suggested_redirect) }}"
        "{% endif %}\n\n"
        "{% if tangent_count >= max_tangent_depth %}"
        '{{ prompt_ref("focus.drift.firm_message", '
        "tangent_count=tangent_count) }}"
        "{% else %}"
        '{{ prompt_ref("focus.drift.gentle_message") }}'
        "{% endif %}"
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Prompt key registry
# ============================================================================

FOCUS_PROMPT_KEYS: dict[str, PromptTemplateDict] = {
    # Guidance
    "focus.guidance.header": FOCUS_GUIDANCE_HEADER,
    "focus.guidance.goal": FOCUS_GUIDANCE_GOAL,
    "focus.guidance.task": FOCUS_GUIDANCE_TASK,
    "focus.guidance.needed": FOCUS_GUIDANCE_NEEDED,
    "focus.guidance.collected": FOCUS_GUIDANCE_COLLECTED,
    "focus.guidance.instructions": FOCUS_GUIDANCE_INSTRUCTIONS,
    "focus.guidance": FOCUS_GUIDANCE_META,
    # Drift
    "focus.drift.header": FOCUS_DRIFT_HEADER,
    "focus.drift.issue": FOCUS_DRIFT_ISSUE,
    "focus.drift.redirect": FOCUS_DRIFT_REDIRECT,
    "focus.drift.firm_message": FOCUS_DRIFT_FIRM_MESSAGE,
    "focus.drift.gentle_message": FOCUS_DRIFT_GENTLE_MESSAGE,
    "focus.drift": FOCUS_DRIFT_META,
}

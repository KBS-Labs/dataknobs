"""Rubric evaluation prompt definitions.

Prompt keys defined here:

- ``rubric.feedback_summary.system`` — System message for feedback summarization
- ``rubric.feedback_summary.user`` — User message template for feedback summarization
- ``rubric.classification`` — System message for LLM-based level classification

The rubric prompts are more dynamic than other prompts — the user message
for feedback summary and the classification prompt are assembled from
runtime data (rubric criteria, evaluation results). The templates here
capture the static instructional portions that can be overridden by consumers.
"""

from dataknobs_llm.prompts.base.types import PromptTemplateDict


# ============================================================================
# Feedback summary prompts
# ============================================================================

RUBRIC_FEEDBACK_SUMMARY_SYSTEM: PromptTemplateDict = {
    "template": (
        "You are a feedback summarizer for content evaluation. "
        "Given structured evaluation results, produce a concise, helpful "
        "natural language summary. Focus on actionable feedback. "
        "Keep the summary to 3-5 sentences."
    ),
    "template_syntax": "format",
}

RUBRIC_FEEDBACK_SUMMARY_USER: PromptTemplateDict = {
    "template": (
        "Rubric: {rubric_name}\n"
        "Description: {rubric_description}\n"
        "Overall: {pass_fail} (score: {weighted_score})\n\n"
        "Criteria results:\n{results_text}"
    ),
    "template_syntax": "format",
}


# ============================================================================
# Classification prompt
# ============================================================================

RUBRIC_CLASSIFICATION: PromptTemplateDict = {
    "template": (
        "You are a classification evaluator for the criterion: {criterion_name}\n"
        "Description: {criterion_description}\n\n"
        "Classify the content into exactly one of these levels:\n"
        "{level_descriptions}\n\n"
        "Respond with a JSON object containing a single field \"level_id\" "
        "set to one of: {valid_ids}\n"
        "Example: {{\"level_id\": \"{example_level_id}\"}}\n"
        "Do not include any other text."
    ),
    "template_syntax": "format",
}


# ============================================================================
# Prompt key registry
# ============================================================================

RUBRIC_PROMPT_KEYS: dict[str, PromptTemplateDict] = {
    "rubric.feedback_summary.system": RUBRIC_FEEDBACK_SUMMARY_SYSTEM,
    "rubric.feedback_summary.user": RUBRIC_FEEDBACK_SUMMARY_USER,
    "rubric.classification": RUBRIC_CLASSIFICATION,
}

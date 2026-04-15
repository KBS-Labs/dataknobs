"""Grounded synthesis prompt fragments and meta-prompts.

Prompt keys defined here:

- ``grounded.synthesis`` — Meta-prompt composing synthesis fragments (Jinja2 conditionals)
- ``grounded.synthesis.*`` — Individual synthesis fragments
- ``grounded.synthesis.kb_wrapper`` — Knowledge base XML wrapper
- ``grounded.provenance_template`` — Default provenance template (Jinja2)

The grounded synthesis meta-prompt uses Jinja2 conditionals to include
fragments based on configuration (require_citations, allow_parametric, etc.).

Consumer: ``GroundedReasoning.build_synthesis_system_prompt()`` resolves
these keys via the prompt resolver with inline fallback.
"""

from dataknobs_llm.prompts.base.types import PromptTemplateDict


# ============================================================================
# Synthesis fragments
# ============================================================================

GROUNDED_SYNTHESIS_BASE_INSTRUCTION: PromptTemplateDict = {
    "template": (
        "Base your response on the knowledge base content provided above."
    ),
    "template_syntax": "format",
}

GROUNDED_SYNTHESIS_CITATION_SECTION: PromptTemplateDict = {
    "template": "Cite the relevant section heading for each claim.",
    "template_syntax": "format",
}

GROUNDED_SYNTHESIS_CITATION_SOURCE: PromptTemplateDict = {
    "template": "Cite the relevant source file for each claim.",
    "template_syntax": "format",
}

GROUNDED_SYNTHESIS_BRIDGE: PromptTemplateDict = {
    "template": (
        "You may connect and synthesize concepts across the "
        "retrieved content. Identify relationships between "
        "different sections and sources when they help answer "
        "the question. Do not introduce facts from outside "
        "the knowledge base -- only synthesize across what "
        "is provided."
    ),
    "template_syntax": "format",
}

GROUNDED_SYNTHESIS_STRICT: PromptTemplateDict = {
    "template": (
        "If the knowledge base content does not contain sufficient "
        "information to fully answer the question, explicitly state "
        "what is missing. Do not fill gaps with information from "
        "outside the knowledge base."
    ),
    "template_syntax": "format",
}

GROUNDED_SYNTHESIS_SUPPLEMENT: PromptTemplateDict = {
    "template": (
        "You may supplement with general knowledge when the "
        "knowledge base is insufficient, but clearly distinguish "
        "KB-grounded claims from general knowledge."
    ),
    "template_syntax": "format",
}

GROUNDED_SYNTHESIS_KB_WRAPPER: PromptTemplateDict = {
    "template": "\n\n<knowledge_base>\n{kb_context}\n</knowledge_base>",
    "template_syntax": "format",
}

# Meta-prompt: composes synthesis fragments based on config parameters.
# Uses Jinja2 conditionals to replicate the current Python if/elif/else logic.
GROUNDED_SYNTHESIS_META: PromptTemplateDict = {
    "template": (
        '{{ prompt_ref("grounded.synthesis.base_instruction") }}'
        "{% if require_citations %}"
        "{% if citation_format == 'section' %}"
        ' {{ prompt_ref("grounded.synthesis.citation_section") }}'
        "{% else %}"
        ' {{ prompt_ref("grounded.synthesis.citation_source") }}'
        "{% endif %}"
        "{% endif %}"
        '{% if allow_parametric == "bridge" %}'
        ' {{ prompt_ref("grounded.synthesis.bridge") }}'
        "{% elif not allow_parametric %}"
        ' {{ prompt_ref("grounded.synthesis.strict") }}'
        "{% else %}"
        ' {{ prompt_ref("grounded.synthesis.supplement") }}'
        "{% endif %}"
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Provenance template
# ============================================================================

GROUNDED_PROVENANCE_TEMPLATE: PromptTemplateDict = {
    "template": (
        "{% if results %}\n"
        "{% for source_name, source_results in results_by_source.items() %}\n"
        "### {{ source_name }}\n"
        "\n"
        "{% for r in source_results %}\n"
        "- **{{ r.metadata.get('headings', [''])|join(' > ') or r.source_id }}** "
        '(relevance: {{ "%.0f"|format(r.relevance * 100) }}%)\n'
        "  {{ r.text_preview }}\n"
        "{% endfor %}\n"
        "{% endfor %}\n"
        "---\n"
        "*{{ results|length }} result{{ 's' if results|length != 1 }} "
        "from {{ results_by_source|length }} source"
        "{{ 's' if results_by_source|length != 1 }}*\n"
        "{% else %}\n"
        "No relevant results found.\n"
        "{% endif %}"
    ),
    "template_syntax": "jinja2",
}


# ============================================================================
# Prompt key registry
# ============================================================================

GROUNDED_PROMPT_KEYS: dict[str, PromptTemplateDict] = {
    "grounded.synthesis.base_instruction": GROUNDED_SYNTHESIS_BASE_INSTRUCTION,
    "grounded.synthesis.citation_section": GROUNDED_SYNTHESIS_CITATION_SECTION,
    "grounded.synthesis.citation_source": GROUNDED_SYNTHESIS_CITATION_SOURCE,
    "grounded.synthesis.bridge": GROUNDED_SYNTHESIS_BRIDGE,
    "grounded.synthesis.strict": GROUNDED_SYNTHESIS_STRICT,
    "grounded.synthesis.supplement": GROUNDED_SYNTHESIS_SUPPLEMENT,
    "grounded.synthesis.kb_wrapper": GROUNDED_SYNTHESIS_KB_WRAPPER,
    "grounded.synthesis": GROUNDED_SYNTHESIS_META,
    "grounded.provenance_template": GROUNDED_PROVENANCE_TEMPLATE,
}

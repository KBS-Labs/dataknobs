"""Prompt resolution utility for DynaBot components.

Provides a simple interface for components to resolve and render prompts
from a prompt library. Wraps ``TemplateRenderer`` and library lookup into
a single ``resolve(key, **variables) -> str | None`` call.

Components accept an optional ``PromptResolver`` and fall back to their
inline defaults when the resolver is ``None`` or the key is not found.
"""

import logging
from typing import Any

from dataknobs_llm.prompts.base import AbstractPromptLibrary
from dataknobs_llm.prompts.base.types import TemplateMode
from dataknobs_llm.prompts.rendering.template_renderer import TemplateRenderer

logger = logging.getLogger(__name__)


class PromptResolver:
    """Resolve and render prompts from a library.

    Wraps a :class:`TemplateRenderer` (with prompt library set for
    ``prompt_ref()`` resolution) and provides a simple lookup-and-render
    API that components can use without knowing rendering internals.

    Example::

        resolver = PromptResolver(library)
        text = resolver.resolve(
            "wizard.clarification",
            issue_list="- missing name",
            stage_prompt="Enter your name",
            suggestions_text="",
        )
        if text is None:
            # Key not in library — use inline fallback
            text = build_inline_fallback(...)
    """

    def __init__(self, library: AbstractPromptLibrary) -> None:
        self._library = library
        self._renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        self._renderer.set_prompt_library(library)

    @property
    def library(self) -> AbstractPromptLibrary:
        """The underlying prompt library."""
        return self._library

    def resolve(self, key: str, **variables: Any) -> str | None:
        """Look up a prompt key and render it with the given variables.

        For meta-prompts that use ``prompt_ref()``, resolution is recursive
        — referenced prompts are resolved from the same library.

        Args:
            key: Prompt key to look up (e.g. ``"wizard.clarification"``)
            **variables: Template variables for rendering

        Returns:
            Rendered prompt string, or ``None`` if the key is not found
            in the library.
        """
        template_dict = self._library.get_system_prompt(key)
        if template_dict is None:
            template_dict = self._library.get_user_prompt(key)
        if template_dict is None:
            return None

        try:
            result = self._renderer.render_prompt_template(
                template_dict, variables,
            )
            return result.content
        except ValueError:
            logger.exception(
                "Failed to render prompt key %r — falling back to inline",
                key,
            )
            return None

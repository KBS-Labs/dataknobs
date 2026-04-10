"""Consolidated wizard template rendering.

All Jinja2 template rendering in the wizard module routes through
:class:`WizardRenderer`.  This ensures:

1. **Consistent context** — every rendering site sees the same
   canonical set of template variables.
2. **Sandboxed execution** — all templates run in a
   :class:`~jinja2.sandbox.SandboxedEnvironment`.
3. **Configurable error handling** — callers choose between
   exception propagation and fallback values.
4. **Security partitioning** — the optional ``(( ))`` conditional
   preprocessing (mixed mode) receives only author-controlled values;
   user-entered data enters only as Jinja2 context variables.
"""

from __future__ import annotations

import logging
from typing import Any

from jinja2 import TemplateError

from dataknobs_bots.utils.template_env import create_template_env

logger = logging.getLogger(__name__)

# Sentinel that distinguishes "no fallback provided" from ``None``
# (which is a valid fallback value).
_SENTINEL = object()


class WizardRenderer:
    """Single rendering layer for all wizard Jinja2 templates.

    Args:
        strict: If ``True``, missing variables raise
            :class:`~jinja2.UndefinedError` instead of rendering as
            empty strings.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self._jinja_env = create_template_env(strict=strict)

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    @staticmethod
    def get_collected_data(state: Any) -> dict[str, Any]:
        """Return non-internal keys from wizard state data.

        Filters out keys starting with ``_`` which are internal
        implementation details (e.g. ``_questions``, ``_bank_items``).

        Args:
            state: :class:`WizardState` instance.

        Returns:
            Filtered copy of ``state.data``.
        """
        return {k: v for k, v in state.data.items() if not k.startswith("_")}

    @staticmethod
    def build_template_params(
        stage: dict[str, Any],
        state: Any,
    ) -> dict[str, Any]:
        """Build author-controlled params safe for ``(( ))`` preprocessing.

        These values are substituted by
        :func:`~dataknobs_llm.template_utils.render_conditional_template`
        via regex directly into the template text.  Only author-controlled
        values (stage metadata, navigation flags, wizard progress) belong
        here — **never** user-entered wizard state data.

        Args:
            stage: Current stage metadata dict.
            state: :class:`WizardState` instance.

        Returns:
            Dict of author-controlled template parameters.
        """
        return {
            "stage_name": stage.get("name", "unknown"),
            "stage_label": stage.get("label", stage.get("name", "")),
            "stage_prompt": stage.get("prompt", ""),
            "help_text": stage.get("help_text") or "",
            "suggestions": stage.get("suggestions", []),
            "completed": state.completed,
            "history": state.history,
            "can_skip": stage.get("can_skip", False),
            "can_go_back": stage.get("can_go_back", True),
        }

    def build_context(
        self,
        stage: dict[str, Any],
        state: Any,
        *,
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build full rendering context for the Jinja2 phase.

        Includes both author-controlled params **and** user-entered state
        data.  User data is safe here because Jinja2 context variables are
        treated as data (not code) — they are never parsed as template
        syntax.

        Args:
            stage: Current stage metadata dict.
            state: :class:`WizardState` instance.
            extra_context: Additional variables to merge (e.g.
                ``bank``, ``artifact``, LLM-generated values).

        Returns:
            Full rendering context dict.
        """
        collected_data = self.get_collected_data(state)
        all_data = {**state.data, **state.transient}
        context: dict[str, Any] = {
            # Author-controlled params
            **self.build_template_params(stage, state),
            # User data as top-level variables
            **all_data,
            # Filtered and unfiltered dicts
            "collected_data": collected_data,
            "all_data": all_data,
            "raw_data": state.data,
            # MemoryBank/Artifact (None unless provided via extra_context)
            "bank": None,
            "artifact": None,
        }
        if extra_context:
            context.update(extra_context)
        return context

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        template_str: str,
        stage: dict[str, Any],
        state: Any,
        *,
        extra_context: dict[str, Any] | None = None,
        fallback: Any = _SENTINEL,
        mixed_mode: bool = False,
    ) -> str:
        """Render a Jinja2 template with wizard state context.

        Args:
            template_str: Jinja2 template string.
            stage: Current stage metadata dict.
            state: :class:`WizardState` instance.
            extra_context: Additional context variables.
            fallback: Value to return on :class:`~jinja2.TemplateError`.
                If omitted (sentinel), exceptions propagate.
            mixed_mode: If ``True``, pre-process ``(( ))`` conditional
                syntax via :func:`render_conditional_template` before
                Jinja2 rendering.  Only author-controlled
                :meth:`build_template_params` values are passed to the
                preprocessor; user data enters only as Jinja2 context
                variables.

        Returns:
            Rendered string.

        Raises:
            jinja2.TemplateError: If rendering fails and no ``fallback``
                is provided.
        """
        if not template_str:
            return template_str

        try:
            text = template_str
            logger.debug(
                "Template render: stage='%s', template_len=%d, "
                "mixed_mode=%s",
                stage.get("name", "unknown"),
                len(template_str),
                mixed_mode,
            )

            if mixed_mode:
                from dataknobs_llm.template_utils import (
                    render_conditional_template,
                )

                template_params = self.build_template_params(stage, state)
                # Enrich with extra_context keys that are author-controlled
                if extra_context:
                    template_params = {**template_params, **{
                        k: v for k, v in extra_context.items()
                        if k in template_params
                    }}
                text = render_conditional_template(text, template_params)

            full_context = self.build_context(
                stage, state, extra_context=extra_context,
            )
            template = self._jinja_env.from_string(text)
            return template.render(**full_context)

        except TemplateError as exc:
            if fallback is not _SENTINEL:
                logger.warning(
                    "Template rendering failed for stage '%s': %s — "
                    "returning fallback",
                    stage.get("name", "unknown"),
                    exc,
                )
                return fallback
            raise

    def render_list(
        self,
        items: list[str],
        stage: dict[str, Any],
        state: Any,
    ) -> list[str]:
        """Render a list of template strings with per-item fallback.

        Plain strings (no ``{{ }}`` or ``{% %}``) pass through unchanged
        for efficiency.  Items that fail to render are returned as-is.

        Args:
            items: List of template strings.
            stage: Current stage metadata dict.
            state: :class:`WizardState` instance.

        Returns:
            List of rendered strings.
        """
        if not items:
            return items

        # Quick check: if no templates, return as-is
        if not any("{%" in s or "{{" in s for s in items):
            return items

        context = self.build_context(stage, state)

        rendered = []
        for item in items:
            if "{%" not in item and "{{" not in item:
                rendered.append(item)
                continue
            try:
                rendered.append(
                    self._jinja_env.from_string(item).render(**context)
                )
            except TemplateError:
                rendered.append(item)
        return rendered

    def render_simple(
        self,
        template_str: str,
        context: dict[str, Any],
        *,
        fallback: Any = _SENTINEL,
    ) -> str:
        """Render a template with a caller-provided context dict.

        For cases where the caller has already built its own context
        (e.g. clarification templates, transition derivation values).

        Args:
            template_str: Jinja2 template string.
            context: Pre-built rendering context.
            fallback: Value to return on error.  If omitted, exceptions
                propagate.

        Returns:
            Rendered string.

        Raises:
            jinja2.TemplateError: If rendering fails and no ``fallback``
                is provided.
        """
        if not template_str:
            return template_str

        try:
            template = self._jinja_env.from_string(template_str)
            return template.render(**context)
        except TemplateError as exc:
            if fallback is not _SENTINEL:
                logger.warning(
                    "Simple template rendering failed: %s — returning fallback",
                    exc,
                )
                return fallback
            raise

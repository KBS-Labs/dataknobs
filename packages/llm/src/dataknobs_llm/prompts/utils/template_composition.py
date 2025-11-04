r"""Template composition utilities for building complex prompts from reusable parts.

This module provides the TemplateComposer class which handles:
- Section substitution in templates
- Template inheritance (extends field)
- Config merging for derived templates
- Caching for performance

Example:
    >>> from dataknobs_llm.prompts import TemplateComposer
    >>>
    >>> # Define base template with sections
    >>> base_config = {
    ...     "sections": {
    ...         "CODE_SECTION": "```{{language}}\\n{{code}}\\n```",
    ...         "INSTRUCTIONS": "Analyze for quality"
    ...     },
    ...     "user_prompts": [{
    ...         "template": "{{CODE_SECTION}}\\n\\n{{INSTRUCTIONS}}"
    ...     }]
    ... }
    >>>
    >>> # Derived template overrides one section
    >>> derived_config = {
    ...     "extends": "base_analysis",
    ...     "sections": {
    ...         "INSTRUCTIONS": "Analyze for security issues"
    ...     }
    ... }
    >>>
    >>> composer = TemplateComposer(library)
    >>> merged = composer.merge_prompt_configs(base_config, derived_config)
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TemplateComposer:
    """Handles template composition and inheritance.

    This class provides functionality for:
    1. Section substitution: Replace {{SECTION_NAME}} placeholders with section content
    2. Template inheritance: Support for 'extends' field to inherit from base templates
    3. Config merging: Merge derived template configs with base configs
    4. Caching: Cache composed templates for performance

    The composer works with prompt libraries to retrieve base templates and
    their configurations, then composes them according to inheritance rules.
    """

    def __init__(self, library: Any | None = None):
        """Initialize the template composer.

        Args:
            library: Optional prompt library for retrieving base templates.
                    If provided, enables template inheritance via 'extends' field.
        """
        self.library = library
        self._composition_cache: Dict[str, str] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}

    def compose_template(
        self,
        template: str,
        sections: Dict[str, str] | None = None,
        prompt_name: str | None = None
    ) -> str:
        r"""Compose a template by replacing section placeholders.

        Replaces all {{SECTION_NAME}} placeholders in the template with
        their corresponding section content from the sections dictionary.

        Args:
            template: Template string with section placeholders
            sections: Dictionary mapping section names to their content
            prompt_name: Optional prompt name for caching

        Returns:
            Composed template with sections expanded

        Example:
            >>> sections = {"CODE": "```python\\ncode\\n```", "NOTES": "Important!"}
            >>> template = "{{CODE}}\\n\\n{{NOTES}}"
            >>> composed = composer.compose_template(template, sections)
            >>> print(composed)
            ```python
            code
            ```

            Important!
        """
        # Check cache if prompt name provided
        if prompt_name:
            cache_key = f"{prompt_name}:{template[:50]}"
            if cache_key in self._composition_cache:
                return self._composition_cache[cache_key]

        if not sections:
            return template

        # Replace section placeholders with section content
        composed = template
        for section_name, section_content in sections.items():
            # Match both {{SECTION_NAME}} and {{ SECTION_NAME }}
            placeholder = f"{{{{{section_name}}}}}"
            placeholder_with_spaces = f"{{{{ {section_name} }}}}"

            composed = composed.replace(placeholder, section_content)
            composed = composed.replace(placeholder_with_spaces, section_content)

        # Cache if prompt name provided
        if prompt_name:
            self._composition_cache[cache_key] = composed

        return composed

    def get_sections_for_prompt(
        self,
        prompt_name: str,
        prompt_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get all sections for a prompt, including inherited sections.

        Handles template inheritance by:
        1. Getting base template sections if 'extends' field exists
        2. Merging with prompt's own sections (prompt sections override base)
        3. Recursively resolving inheritance chains

        Args:
            prompt_name: Name of the prompt
            prompt_config: Prompt configuration dictionary

        Returns:
            Dictionary of all sections (base + overrides)

        Raises:
            ValueError: If inheritance chain is circular
        """
        # Track visited prompts to detect circular inheritance
        visited = set()

        return self._get_sections_recursive(
            prompt_name,
            prompt_config,
            visited
        )

    def _get_sections_recursive(
        self,
        prompt_name: str,
        prompt_config: Dict[str, Any],
        visited: set
    ) -> Dict[str, str]:
        """Recursively resolve sections with inheritance.

        Args:
            prompt_name: Name of the current prompt
            prompt_config: Configuration for the current prompt
            visited: Set of already visited prompts (for cycle detection)

        Returns:
            Merged sections dictionary

        Raises:
            ValueError: If circular inheritance detected
        """
        # Check for circular inheritance
        if prompt_name in visited:
            raise ValueError(
                f"Circular inheritance detected: {prompt_name} already in "
                f"inheritance chain {visited}"
            )

        visited.add(prompt_name)

        # Start with empty sections
        all_sections = {}

        # If this prompt extends another, get base sections first
        extends = prompt_config.get("extends")
        if extends and self.library:
            # Try to get the base prompt configuration
            # We'll try both system and user prompts
            base_config = None

            # Try system prompts first
            try:
                base_config = self.library.get_system_prompt(extends)
            except (ValueError, KeyError):
                pass

            # Try user prompts if system didn't work
            if not base_config:
                try:
                    base_config = self.library.get_user_prompt(extends, index=0)
                except (ValueError, KeyError):
                    pass

            if base_config:
                # Recursively get base sections
                base_sections = self._get_sections_recursive(
                    extends,
                    base_config,
                    visited.copy()  # Copy to avoid affecting sibling branches
                )
                all_sections.update(base_sections)
            else:
                logger.warning(
                    f"Cannot find base template '{extends}' for '{prompt_name}'"
                )

        # Overlay this prompt's sections (overrides base)
        prompt_sections = prompt_config.get("sections", {})
        all_sections.update(prompt_sections)

        return all_sections

    def merge_prompt_configs(
        self,
        base_config: Dict[str, Any],
        derived_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge derived prompt config with base config.

        Merging rules:
        1. Sections: Child sections override parent sections
        2. Defaults: Child defaults override parent defaults
        3. Validation: Child validation overrides parent validation
        4. RAG configs: Child configs are appended to parent configs
        5. User/system prompts: Child prompts replace parent prompts
        6. Metadata: Merged with child taking priority

        Args:
            base_config: Base template configuration
            derived_config: Derived template configuration

        Returns:
            Merged configuration dictionary

        Example:
            >>> base = {
            ...     "defaults": {"lang": "python"},
            ...     "sections": {"CODE": "{{code}}"}
            ... }
            >>> derived = {
            ...     "defaults": {"style": "PEP8"},
            ...     "sections": {"NOTES": "{{notes}}"}
            ... }
            >>> merged = composer.merge_prompt_configs(base, derived)
            >>> merged["defaults"]
            {"lang": "python", "style": "PEP8"}
            >>> merged["sections"]
            {"CODE": "{{code}}", "NOTES": "{{notes}}"}
        """
        merged = {}

        # 1. Merge sections (child overrides parent)
        if "sections" in base_config or "sections" in derived_config:
            merged["sections"] = {
                **base_config.get("sections", {}),
                **derived_config.get("sections", {})
            }

        # 2. Merge defaults (child overrides parent)
        if "defaults" in base_config or "defaults" in derived_config:
            merged["defaults"] = {
                **base_config.get("defaults", {}),
                **derived_config.get("defaults", {})
            }

        # 3. Merge validation (child overrides parent completely)
        if "validation" in derived_config:
            merged["validation"] = derived_config["validation"]
        elif "validation" in base_config:
            merged["validation"] = base_config["validation"]

        # 4. Merge RAG configs (append child to parent)
        base_rag_refs = base_config.get("rag_config_refs", [])
        derived_rag_refs = derived_config.get("rag_config_refs", [])
        if base_rag_refs or derived_rag_refs:
            # Combine refs, removing duplicates while preserving order
            seen = set()
            merged["rag_config_refs"] = []
            for ref in base_rag_refs + derived_rag_refs:
                if ref not in seen:
                    seen.add(ref)
                    merged["rag_config_refs"].append(ref)

        base_rag_configs = base_config.get("rag_configs", [])
        derived_rag_configs = derived_config.get("rag_configs", [])
        if base_rag_configs or derived_rag_configs:
            merged["rag_configs"] = base_rag_configs + derived_rag_configs

        # 5. User/system prompts - child replaces parent
        # (This is for the template dict itself, not the list of prompts)
        if "template" in derived_config:
            merged["template"] = derived_config["template"]
        elif "template" in base_config:
            merged["template"] = base_config["template"]

        if "user_prompts" in derived_config:
            merged["user_prompts"] = derived_config["user_prompts"]
        elif "user_prompts" in base_config:
            merged["user_prompts"] = base_config["user_prompts"]

        if "system_prompts" in derived_config:
            merged["system_prompts"] = derived_config["system_prompts"]
        elif "system_prompts" in base_config:
            merged["system_prompts"] = base_config["system_prompts"]

        # 6. Merge metadata (child takes priority)
        if "metadata" in base_config or "metadata" in derived_config:
            merged["metadata"] = {
                **base_config.get("metadata", {}),
                **derived_config.get("metadata", {})
            }

        # 7. Copy extends field if present
        if "extends" in derived_config:
            merged["extends"] = derived_config["extends"]

        return merged

    def resolve_inheritance(
        self,
        prompt_name: str,
        prompt_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fully resolve a prompt's inheritance chain.

        This method walks up the inheritance chain (via 'extends' fields)
        and merges all configs from base to derived.

        Args:
            prompt_name: Name of the prompt to resolve
            prompt_config: Initial prompt configuration

        Returns:
            Fully resolved configuration with all inheritance applied

        Raises:
            ValueError: If circular inheritance detected

        Example:
            >>> # grandparent -> parent -> child
            >>> resolved = composer.resolve_inheritance("child", child_config)
            >>> # Returns merged config with all three levels
        """
        # Check cache
        cache_key = f"{prompt_name}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Track visited to detect cycles
        visited = []
        current_name = prompt_name
        current_config = prompt_config

        # Collect all configs in the inheritance chain
        configs_to_merge = [current_config]

        while current_config.get("extends"):
            base_name = current_config["extends"]

            # Check for circular inheritance
            if base_name in visited:
                raise ValueError(
                    f"Circular inheritance detected: {base_name} -> "
                    f"{' -> '.join(visited)} -> {base_name}"
                )

            visited.append(current_name)

            # Get base config from library
            if not self.library:
                logger.warning(
                    f"Cannot resolve inheritance for '{current_name}': "
                    f"no library provided"
                )
                break

            # Try to get base config
            base_config = None
            try:
                base_config = self.library.get_system_prompt(base_name)
            except (ValueError, KeyError):
                pass

            if not base_config:
                try:
                    base_config = self.library.get_user_prompt(base_name, index=0)
                except (ValueError, KeyError):
                    pass

            if not base_config:
                logger.warning(
                    f"Cannot find base template '{base_name}' for '{current_name}'"
                )
                break

            # Add to chain
            configs_to_merge.insert(0, base_config)  # Insert at front (base first)

            # Move up the chain
            current_name = base_name
            current_config = base_config

        # Merge all configs from base to derived
        resolved = {}
        for config in configs_to_merge:
            resolved = self.merge_prompt_configs(resolved, config)

        # Cache the result
        self._config_cache[cache_key] = resolved

        return resolved

    def clear_cache(self):
        """Clear all caches.

        Call this if templates or configs are modified after composition.
        """
        self._composition_cache.clear()
        self._config_cache.clear()

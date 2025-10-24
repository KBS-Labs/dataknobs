"""Utility functions for prompt management.

This module contains utility classes and functions:
- TemplateComposer: Support template composition and inheritance
- MessageIndexParser: Parse and validate message index definitions (future)
"""

from .template_composition import TemplateComposer

__all__ = [
    "TemplateComposer",
]

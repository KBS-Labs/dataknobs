# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for prompt management.

This module contains utility classes and functions:
- TemplateComposer: Support template composition and inheritance
- MessageIndexParser: Parse and validate message index definitions (future)
"""

from .template_composition import TemplateComposer

__all__ = [
    "TemplateComposer",
]

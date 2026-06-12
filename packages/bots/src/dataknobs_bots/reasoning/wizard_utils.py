"""Shared utility functions for wizard extraction and grounding.

The canonical home for :func:`word_in_text` is
:mod:`dataknobs_llm.intent.defaults`; it is re-exported from this
module for back-compat with extraction-layer call sites that import
``from .wizard_utils import word_in_text``.
"""

from __future__ import annotations

from dataknobs_llm.intent.defaults import word_in_text

__all__ = ["word_in_text"]

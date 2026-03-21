"""Shared utility functions for wizard extraction and grounding."""

from __future__ import annotations

import re


def word_in_text(word: str, text: str) -> bool:
    r"""Check if *word* appears as a whole word in *text*.

    Uses ``\b`` word-boundary anchors to avoid substring
    false positives (e.g. ``"base"`` matching ``"database"``).

    Works with multi-word phrases: ``word_in_text("study companion",
    "I want a study companion bot")`` returns ``True``.
    """
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))

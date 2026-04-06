"""Shared utilities for chunking operations."""

from __future__ import annotations

# Default boundary search order for split_text()
DEFAULT_BOUNDARIES: list[str] = ["\n\n", ". ", "! ", "? ", ".\n", "!\n", "?\n", " "]


def split_text(
    text: str,
    max_size: int,
    boundaries: list[str] | None = None,
) -> list[tuple[str, int, int]]:
    """Split text into chunks respecting *max_size*, returning positions.

    When *text* exceeds *max_size*, splits are placed at the best
    available boundary using a priority-based backward search:

    1. Paragraph break (double newline)
    2. Sentence-ending punctuation followed by space/newline
    3. Word break (space)
    4. Hard cut at *max_size* (only when no boundary exists)

    Each returned tuple contains ``(chunk_text, rel_start, rel_end)``
    where ``rel_start`` and ``rel_end`` are character offsets relative
    to the input *text*.  ``text[rel_start:rel_end]`` gives the
    pre-strip source span; ``chunk_text`` is the stripped version,
    so ``chunk_text == text[rel_start:rel_end].strip()``.

    Args:
        text: Text to split.
        max_size: Maximum chunk size in characters.
        boundaries: Custom boundary strings in priority order.
            Defaults to paragraph, sentence, and word boundaries.

    Returns:
        List of ``(chunk_text, rel_start, rel_end)`` tuples.
    """
    if len(text) <= max_size:
        return [(text, 0, len(text))]

    if boundaries is None:
        boundaries = DEFAULT_BOUNDARIES

    # Separate paragraph boundary (consumes 2 chars) from single-char
    # boundaries so we can correctly advance past the break.
    result: list[tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        end = start + max_size

        if end < len(text):
            # Try boundaries in priority order
            found = False
            for boundary in boundaries:
                break_pos = text.rfind(boundary, start, end)
                if break_pos > start:
                    end = break_pos + len(boundary)
                    found = True
                    break

            if not found:
                # Hard cut — end stays at start + max_size
                pass

        # Clamp to text length
        end = min(end, len(text))

        chunk_text = text[start:end].strip()
        if chunk_text:
            result.append((chunk_text, start, end))

        start = end

    return result

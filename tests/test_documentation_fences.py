"""What the shared fence reader treats as a fence, and why each rule is there.

``code_fences`` is read by two guards -- the import guard and the config-key
guard -- and each of its rules exists because a reader without it silently
read less than it reported. That is the failure both guards were written to
catch, so the reader having it too is worth pinning here rather than in
either consumer: a rule tested inside one guard is a rule the other can lose
without anything failing.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from tests._workspace import ROOT, Fence, code_fences, documentation_files, rel


@pytest.fixture
def document(tmp_path: Path) -> Callable[[str], list[Fence]]:
    """Write a markdown fixture and return the fences read out of it."""

    def write(text: str) -> list[Fence]:
        path = tmp_path / "doc.md"
        path.write_text(textwrap.dedent(text).strip("\n") + "\n", encoding="utf-8")
        return code_fences(path)

    return write


def test_a_fence_indented_under_a_list_item_is_read(document: Callable[[str], list[Fence]]) -> None:
    """A reader anchored to column zero never opens it.

    65 fences in this tree are indented under a numbered step or a list item,
    and a document whose code is all indented is indistinguishable from a
    document with no code in it.
    """
    fences = document(
        """
        1. First do this:

           ```python
           from dataknobs_data import Record
           ```
        """
    )

    assert [f.lang for f in fences] == ["python"]
    assert fences[0].lines == ["   from dataknobs_data import Record"]


def test_a_bare_backtick_run_closes_but_never_opens(document: Callable[[str], list[Fence]]) -> None:
    """The rule that keeps an unbalanced document from swallowing the next fence.

    Several documents here quote a markdown sample inside a Python string
    using the same backtick count as the fence containing it, which inverts
    every fence boundary below it. A reader that opens on a bare ``` then
    reads the real fences as the gaps between imaginary ones. Measured
    against this tree: 9 import statements in 6 documents went unread.
    """
    fences = document(
        """
        ```python
        markdown = '''
        ```python
        nested = 1
        ```
        '''
        ```

        ```python
        from dataknobs_data import Record
        ```
        """
    )

    assert [f.lang for f in fences] == ["python", "python"]
    assert fences[1].lines == ["from dataknobs_data import Record"]


def test_a_labelled_fence_inside_another_is_body_not_a_fence(
    document: Callable[[str], list[Fence]],
) -> None:
    """A ```python block quoted inside a ```markdown block is a sample of a sample.

    The document that motivated this shows a bug-report template, so the
    import inside it is not the document's own claim that anything resolves.
    """
    fences = document(
        """
        ```markdown
        ## Code Sample
        ```python
        from dataknobs_structures import Tree
        ```
        ```
        """
    )

    assert [f.lang for f in fences] == ["markdown"]
    assert "from dataknobs_structures import Tree" in fences[0].body


def test_a_marker_reaches_the_fence_below_it(document: Callable[[str], list[Fence]]) -> None:
    """An HTML comment annotates the next fence, across blank lines only."""
    fences = document(
        """
        <!-- dk-imports: illustrative -- the old spelling is the subject -->

        ```python
        from dataknobs_data import MemoryDatabase
        ```

        Ordinary prose.

        ```python
        from dataknobs_data import SyncMemoryDatabase
        ```
        """
    )

    assert fences[0].marker == "dk-imports: illustrative -- the old spelling is the subject"
    assert fences[1].marker is None


def test_an_unclosed_fence_is_still_returned(document: Callable[[str], list[Fence]]) -> None:
    """Dropping it would hide its contents behind a missing delimiter."""
    fences = document(
        """
        ```python
        from dataknobs_data import Record
        """
    )

    assert [f.lang for f in fences] == ["python"]
    assert fences[0].lines == ["from dataknobs_data import Record"]


def test_the_line_number_names_the_first_line_of_content(
    document: Callable[[str], list[Fence]],
) -> None:
    """So an offset into ``lines`` names a real line of the document."""
    fences = document(
        """
        Prose.

        ```yaml
        databases:
          - backend: memory
        ```
        """
    )

    assert fences[0].line == 4  # prose, blank, ```yaml, then content
    assert fences[0].lines[1] == "  - backend: memory"


def test_the_reader_finds_fences_across_the_documentation() -> None:
    """A reader that stops matching must fail rather than pass.

    The floor is well under what the tree holds; it exists to catch a change
    that makes the patterns stop finding anything, not to track the corpus.
    """
    counts = {rel(path): len(code_fences(path)) for path in documentation_files()}
    total = sum(counts.values())

    assert total > 2000, f"only {total} fences found across {len(counts)} documents"
    assert counts[rel(ROOT / "README.md")] > 0, "the root README reads as having no code"

"""An undeclared store width does not stop the documented fallback path.

``RAGKnowledgeBaseConfig.vector_store`` says an empty section *"does not
fail: the factory falls back to an in-process store"*, and neither that
section nor the fallback carries a ``dimensions``. So the store this
documented path produces declares ``0`` --- which is the absence of a
declaration, not a declaration of zero --- and the embedder writes 768
into it.

The width guard on ``VectorStoreBase.add_vectors`` read that sentinel as
a claim and refused the write, with ``expected 0, got 768``: a message
naming neither the config key nor the cause, on the one ingest path the
config documents as needing nothing. The guard now compares only a width
somebody stated.

This lives in ``bots`` and not beside the guard because the break is
cross-package: the refusal was added in ``dataknobs-data`` and what it
broke was a ``dataknobs-bots`` contract, which no test in ``data`` reads.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge.rag import RAGKnowledgeBase

_MARKDOWN = "# Title\n\nA paragraph with enough words in it to make one chunk.\n"


async def _kb(vector_store: dict[str, object]) -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": vector_store,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


@pytest.mark.parametrize(
    "vector_store",
    [
        pytest.param({}, id="empty-section"),
        pytest.param({"backend": "memory"}, id="backend-only"),
    ],
)
async def test_the_documented_fallback_path_can_still_ingest(
    vector_store: dict[str, object],
) -> None:
    """Both spellings the config documents, neither of which states a width."""
    kb = await _kb(vector_store)
    try:
        assert kb.vector_store.dimensions == 0, "the sentinel this test is about"
        assert await kb.load_markdown_text(_MARKDOWN, "doc1") >= 1
        assert await kb.query("paragraph", k=1)
    finally:
        await kb.close()


async def test_a_width_the_config_does_state_is_still_compared() -> None:
    """The positive control: the guard is relaxed for ``0``, not removed.

    Stating the store's width alone is *not* the way to build the
    disagreement any more, and that is this branch's other change: a width
    given only to the store now reaches the embedder through
    ``build_embedding_config``, so the two agree by construction. The
    disagreement has to be written deliberately --- a store told it is 384
    wide and an embedder told, separately, to emit 768 --- and against that
    the refusal stands.
    """
    kb = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
            "dimensions": 768,
        }
    )
    try:
        with pytest.raises(ValueError, match="384"):
            await kb.load_markdown_text(_MARKDOWN, "doc1")
    finally:
        await kb.close()

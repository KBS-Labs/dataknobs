"""One reading of "this model name denotes an embedding model", for every provider.

The Ollama and HuggingFace heuristics each carried a vocabulary of their own,
and they had drifted: HuggingFace matched ``embedding`` and so missed
``nomic-embed-text``, while Ollama matched ``bge-`` as a substring and so
granted embeddings to a ``bge`` *reranker*, a cross-encoder that embeds
nothing. Once completion models stopped being granted ``EMBEDDINGS``, the name
became the only offline evidence, so a disagreement between the two is a
capability one provider reports and the other does not.
"""

from __future__ import annotations

import pytest

from dataknobs_llm.llm.base import ModelCapability
from dataknobs_llm.llm.model_profile import is_embedding_model_name
from dataknobs_llm.llm.providers.huggingface import _hf_heuristic
from dataknobs_llm.llm.providers.ollama import _ollama_heuristic

_NAMES = {
    "nomic-embed-text": True,
    "nomic-ai/nomic-embed-text-v1": True,
    "mxbai-embed-large": True,
    "bge-m3": True,
    "BAAI/bge-large-en-v1.5": True,
    "all-minilm": True,
    "paraphrase-multilingual": True,
    "intfloat/e5-large-v2": True,
    "thenlper/gte-large": True,
    "hkunlp/instructor-large": True,
    "sentence-transformers/all-mpnet-base-v2": True,
    "bge-reranker-v2-m3": False,
    "BAAI/bge-reranker-base": False,
    "llama3.2:3b": False,
    "qwen2.5:7b": False,
    "phi3.5": False,
    "mistralai/Mistral-7B-Instruct-v0.2": False,
}


@pytest.mark.parametrize(("name", "expected"), _NAMES.items(), ids=_NAMES.keys())
def test_every_heuristic_reads_a_name_the_same_way(name: str, expected: bool) -> None:
    assert is_embedding_model_name(name) is expected

    for heuristic in (_hf_heuristic, _ollama_heuristic):
        caps = heuristic(name).capabilities or frozenset()
        assert (ModelCapability.EMBEDDINGS in caps) is expected, heuristic.__name__

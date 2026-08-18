"""Testing utilities for dataknobs-data.

Deterministic vector draws, for this package's own tests and for consumers
testing their own ``VectorStore`` implementations against the same protocol.

Every helper here builds its own ``numpy.random.Generator`` rather than reading
the process-global stream. That is what makes a draw safe to call from
anywhere: ``np.random.seed(...)`` mutates state shared by the whole process, so
one test that seeds it shifts every later unseeded draw in the session, and
outcomes start depending on which tests already ran.

Example:
    ```python
    from dataknobs_data.testing import vector, vectors

    stored = vectors(10, 128)      # ten vectors that differ, from one draw
    query = vector(128, seed=1)    # a distinct seed, so not row 0 of `stored`
    ```
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

__all__ = ["chroma_embedding_function", "text_embedding", "vector", "vectors"]


def vectors(count: int, dim: int, seed: int = 0) -> np.ndarray:
    """Draw ``count`` deterministic ``dim``-dimensional float32 vectors.

    One generator per call, seeded explicitly, so a draw here cannot shift what
    any other test draws. Pass a distinct ``seed`` where a single test needs two
    sets that differ.

    This is also the helper to reach for when a test needs N vectors that differ
    from each other. Calling :func:`vector` once per loop iteration passes the
    same default seed every time and hands back N copies of one vector — which
    keeps most assertions green while removing the thing they were checking.
    Draw the whole set here and index into it instead.
    """
    return np.random.default_rng(seed).random((count, dim), dtype=np.float32)


def vector(dim: int, seed: int = 0) -> np.ndarray:
    """Draw one deterministic ``dim``-dimensional float32 vector."""
    # Bound to a name rather than returned directly: numpy types `ndarray`
    # indexing as `Any`, so returning the subscript straight out of a function
    # declared to return `ndarray` is a `no-any-return`. The draw stays defined
    # in terms of `vectors` so there is only one of them.
    row: np.ndarray = vectors(1, dim, seed)[0]
    return row


def text_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Draw a ``dim``-dimensional vector deterministically derived from ``text``.

    The generator is built per call and seeded from the text, so
    ``same text -> same vector`` holds while nothing outside the call is
    affected.

    The seed is the sum of the first ten characters' code points, so it
    collides more readily than a hash would: two texts agreeing in their first
    ten characters share a vector, and so do two whose first ten characters are
    a permutation of each other. A test needing two embeddings that differ has
    to vary the text within that window by more than its ordering.
    """
    return np.random.default_rng(sum(ord(c) for c in text[:10])).random(dim)


@functools.cache
def _deterministic_embedding_function_class() -> Any:
    """The embedding-function class, built once and registered with chromadb.

    Defined here rather than inline in :func:`chroma_embedding_function`
    so that there is exactly one class object. A class statement inside
    that function ran on every call, so no two instances shared a type
    and chromadb saw a different embedding function each time it was
    handed one.

    The registration is what ``name`` and ``build_from_config`` below
    exist for: chromadb reconstructs a persisted collection's embedding
    function by looking its name up in ``known_embedding_functions``,
    and a name absent from that table cannot be reconstructed. It is
    applied once, here, because registering the same name twice is not
    the caching behaviour anyone wants.

    **It does not currently rescue a reopened persistent collection**,
    and the docstring says so rather than implying otherwise. Measured
    on chromadb 1.5.9: a ``PersistentClient`` collection created with
    this function and reopened through ``get_collection`` comes back
    holding ``DefaultEmbeddingFunction`` — silently, with no warning —
    whether or not the class is registered. So the reopen path falls
    back to the very model-downloading default this helper exists to
    avoid. A test that needs the guarantee must pass the embedding
    function explicitly on every open, which is what
    ``ChromaVectorStore.initialize`` does and why nothing in this repo
    is exposed. The registration stays because it is the correct
    declaration for a class carrying these methods, and because the
    fallback is chromadb's behaviour to change, not ours.

    Imported lazily, and cached, so this module stays importable without
    chromadb installed.
    """
    from chromadb.api.types import EmbeddingFunction
    from chromadb.utils.embedding_functions import register_embedding_function

    @register_embedding_function
    class _DeterministicEmbeddingFunction(EmbeddingFunction):
        """Text in, :func:`text_embedding` out."""

        def __init__(self, dim: int) -> None:
            self._dim = dim

        # ``input`` shadows the builtin because chromadb's protocol names
        # the parameter that way; the signature is theirs, not ours.
        def __call__(self, input: Any) -> Any:
            return [text_embedding(text, self._dim).tolist() for text in input]

        @staticmethod
        def name() -> str:
            return "dataknobs-deterministic"

        def get_config(self) -> dict[str, Any]:
            return {"dim": self._dim}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> Any:
            return _DeterministicEmbeddingFunction(config["dim"])

    return _DeterministicEmbeddingFunction


def chroma_embedding_function(dim: int = 8) -> Any:
    """A chromadb embedding function that embeds without downloading a model.

    A ``ChromaVectorStore`` built with no ``embedding_function`` falls
    through to chromadb's default, which fetches ~166 MB of ONNX weights
    on first use and caches them under ``~/.cache/chroma``. That makes a
    test suite pass on a developer machine whose cache is warm and
    *fail* — not skip — on a cold runner, because the download is not
    something a ``skipif`` can see coming.

    Pass this instead wherever a test exercises a document path
    (``add_documents``, ``search_documents``), which are the only paths
    that embed text rather than accepting vectors outright. Embeddings
    come from :func:`text_embedding`, so they are deterministic and
    inherit both of its properties. Texts agreeing in their first ten
    characters embed identically, so a test needing two distinct
    embeddings has to vary its texts inside that window. And every
    component is drawn from ``[0, 1)``, so any two embeddings are highly
    cosine-similar regardless of their texts — fine for asserting that a
    particular row came back, unusable for asserting a *ranking*.

    ``dim`` must match the store's configured ``dimensions``; being able
    to pick a small one is a side benefit, since nothing here needs 384.
    """
    return _deterministic_embedding_function_class()(dim)

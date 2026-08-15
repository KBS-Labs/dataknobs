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

import numpy as np

__all__ = ["text_embedding", "vector", "vectors"]


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

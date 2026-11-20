# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Provenance models for tracking artifact creation and revision history.

The records live in :mod:`dataknobs_common.provenance`, so content can carry
provenance without bots; this module re-exports the same objects, and an
artifact's provenance is a :class:`ProvenanceRecord`:

- Source references: where content came from
- Tool invocations: which tools were used during creation
- LLM invocations: when and why LLMs were consulted
- Revision records: history of changes
- Provenance records: complete creation context

The records are frozen. Fields holding several values are tuples, and a
change is a new record made with :func:`dataclasses.replace`:

Example:
    >>> import dataclasses
    >>> provenance = create_provenance(
    ...     created_by="system:generator:quiz_gen_v1",
    ...     creation_method="generator",
    ... )
    >>> provenance = dataclasses.replace(
    ...     provenance,
    ...     sources=(
    ...         *provenance.sources,
    ...         SourceReference(source_id="doc_123", source_type="document"),
    ...     ),
    ... )
"""

from __future__ import annotations

from dataknobs_common.provenance import (
    LLMInvocation,
    ProvenanceRecord,
    RevisionRecord,
    SourceReference,
    ToolInvocation,
    create_provenance,
    now_iso,
)

__all__ = [
    "LLMInvocation",
    "ProvenanceRecord",
    "RevisionRecord",
    "SourceReference",
    "ToolInvocation",
    "create_provenance",
    "now_iso",
]

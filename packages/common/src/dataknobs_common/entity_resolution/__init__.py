# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Placing a string against a vocabulary, with the reason each candidate won.

A **cascade**: rungs asked in order, stopping when ``k`` is filled. There is
no fusion and no threshold above the rungs, so a declared hit outranks a
higher-scoring vector one without a weight existing anywhere -- the exact rung
is simply asked first, and the composition is the policy::

    from dataknobs_common.entity_resolution import (
        AliasSignal, CascadingResolver, ExactNormalizedSignal,
    )

    resolver = CascadingResolver(
        [ExactNormalizedSignal(onto.entities), AliasSignal(onto.entities)],
        onto.entities,          # the authority a `within` scope is decided against
    )
    result = resolver.resolve("beagles", k=5)
    result.candidates[0].entity_id            # "beagle"
    result.explain("beagle")[0].signal        # "exact" -- which rung, and why

Every candidate carries its evidence, so *why did this win* is answerable
without re-running the query, and a stored resolution can still tell a
declared alias from a vector guess.

**This package imports nothing from** ``dataknobs_common.ontology``, which is
a property worth stating because it is easy to lose and expensive when lost.
The rungs here take an entity source as a *protocol*; the value types they
construct live in :mod:`~dataknobs_common.entity_resolution.values` rather
than in the ontology model, and the fold and the token-boundary policy they
locate a form with live in :mod:`dataknobs_common.text`. An edge back would
close a cycle through ``ontology/__init__``, which fails on import *order* --
so a suite that happens to import one side first stays green while the other
is broken.

``content_span`` and ``token_spans`` are re-exported here, beside the rungs
that need them, because writing a scanning rung is what they are for -- and
they stay in ``text`` itself, which imports nothing, so the vocabulary side
can fold with the same policy without depending on this package.

A property of the module graph, not of the process: ``dataknobs_common``
publishes the vocabulary on its own door, so importing anything from the
package imports :mod:`dataknobs_common.ontology` as well.
"""

from dataknobs_common.entity_resolution.cascade import (
    AsyncCascadingResolver,
    BridgedEntityResolver,
    CascadeState,
    CascadingResolver,
    finish,
    merge_rung,
)
from dataknobs_common.entity_resolution.protocols import (
    AliasFormSource,
    AsyncAliasFormSource,
    AsyncEntityResolver,
    AsyncMatchSignal,
    AsyncSurfaceFormCatalog,
    EntityResolver,
    MatchSignal,
    MembershipOracle,
    SurfaceFormCatalog,
)
from dataknobs_common.entity_resolution.registry import (
    async_signal_backends,
    declared_signal_metadata,
    signal_backends,
)
from dataknobs_common.entity_resolution.signals import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncDeclaredSignal,
    AsyncExactNormalizedSignal,
    AsyncLexicalSignal,
    AsyncScanningSignal,
    DeclaredSignal,
    ExactNormalizedSignal,
    LexicalSignal,
    ScanningSignal,
    declared_candidates,
)
from dataknobs_common.entity_resolution.values import (
    ENTITY_TYPE_KEY,
    CompatibilityVerdict,
    Coverage,
    EntityCandidate,
    EvidenceKind,
    FormHit,
    MatchEvidence,
    ResolutionRef,
    ResolutionResult,
    RunnerUp,
    ScopeAuthority,
    Scoring,
    Within,
    refuse_unknown_axes,
    within_admits,
    within_axes,
    within_axis_names,
    within_memberships,
)
from dataknobs_common.text import content_span, token_spans

__all__ = [
    "ENTITY_TYPE_KEY",
    "AliasFormSource",
    "AsyncAliasFormSource",
    "AliasSignal",
    "AsyncAliasSignal",
    "AsyncDeclaredSignal",
    "AsyncCascadingResolver",
    "AsyncEntityResolver",
    "AsyncExactNormalizedSignal",
    "AsyncLexicalSignal",
    "AsyncScanningSignal",
    "AsyncSurfaceFormCatalog",
    "DeclaredSignal",
    "AsyncMatchSignal",
    "BridgedEntityResolver",
    "CascadeState",
    "CascadingResolver",
    "CompatibilityVerdict",
    "Coverage",
    "EntityCandidate",
    "EntityResolver",
    "EvidenceKind",
    "ExactNormalizedSignal",
    "FormHit",
    "LexicalSignal",
    "MatchEvidence",
    "MatchSignal",
    "MembershipOracle",
    "ResolutionRef",
    "ResolutionResult",
    "RunnerUp",
    "ScanningSignal",
    "ScopeAuthority",
    "Scoring",
    "SurfaceFormCatalog",
    "Within",
    "async_signal_backends",
    "content_span",
    "declared_candidates",
    "declared_signal_metadata",
    "finish",
    "merge_rung",
    "signal_backends",
    "token_spans",
    "refuse_unknown_axes",
    "within_admits",
    "within_axes",
    "within_axis_names",
    "within_memberships",
]

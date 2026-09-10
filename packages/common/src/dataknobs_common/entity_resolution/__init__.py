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
than in the ontology model, and the normalizer they fold with lives in
:mod:`dataknobs_common.text`. An edge back would close a cycle through
``ontology/__init__``, which fails on import *order* -- so a suite that
happens to import one side first stays green while the other is broken.

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
    EntityResolver,
    MatchSignal,
    MembershipOracle,
)
from dataknobs_common.entity_resolution.registry import (
    async_signal_backends,
    signal_backends,
)
from dataknobs_common.entity_resolution.signals import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncDeclaredSignal,
    AsyncExactNormalizedSignal,
    DeclaredSignal,
    ExactNormalizedSignal,
)
from dataknobs_common.entity_resolution.values import (
    ENTITY_TYPE_KEY,
    CompatibilityVerdict,
    Coverage,
    EntityCandidate,
    EvidenceKind,
    MatchEvidence,
    ResolutionRef,
    ResolutionResult,
    ScopeAuthority,
    Scoring,
    Within,
    refuse_unknown_axes,
    within_admits,
    within_axes,
    within_axis_names,
    within_memberships,
)

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
    "MatchEvidence",
    "MatchSignal",
    "MembershipOracle",
    "ResolutionRef",
    "ResolutionResult",
    "ScopeAuthority",
    "Scoring",
    "Within",
    "async_signal_backends",
    "finish",
    "merge_rung",
    "signal_backends",
    "refuse_unknown_axes",
    "within_admits",
    "within_axes",
    "within_axis_names",
    "within_memberships",
]

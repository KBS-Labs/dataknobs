"""Configuration dataclass for the hybrid reasoning strategy."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dataknobs_common.structured_config import StructuredConfig

from dataknobs_bots.reasoning.grounded_config import GroundedReasoningConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridReasoningConfig(StructuredConfig):
    """Top-level configuration for :class:`HybridReasoning`.

    The hybrid strategy composes grounded retrieval (mandatory KB lookup)
    with a ReAct tool-use loop.  This config groups sub-configs for both
    phases plus hybrid-specific settings.

    Attributes:
        grounded: Configuration for the grounded retrieval phase
            (intent resolution, retrieval, synthesis, sources).
        react_max_iterations: Maximum ReAct tool-use iterations.
        react_verbose: Enable debug-level logging for ReAct steps.
        react_store_trace: Store ReAct reasoning trace in metadata.
        store_provenance: Record merged provenance (grounded + tool
            executions) in ``manager.metadata["retrieval_provenance"]``.
        greeting_template: Optional Jinja2 template for bot-initiated
            greetings.
    """

    grounded: GroundedReasoningConfig = field(
        default_factory=GroundedReasoningConfig,
    )
    react_max_iterations: int = 5
    react_verbose: bool = False
    react_store_trace: bool = False
    store_provenance: bool = True
    greeting_template: str | None = None

    def __post_init__(self) -> None:
        """Warn when grounded's ``store_provenance`` differs from hybrid's.

        In hybrid mode only the hybrid-level flag controls provenance
        storage — grounded's ``generate()`` is never called, only
        ``retrieve_context()`` — so a mismatched ``grounded.store_provenance``
        is silently ineffective.  Surfacing it on every construction path
        (dict-loaded *and* direct) is strictly more helpful than the
        former ``from_dict``-only warning.
        """
        if self.grounded.store_provenance != self.store_provenance:
            logger.warning(
                "Hybrid strategy: 'grounded.store_provenance' (%s) differs "
                "from hybrid-level 'store_provenance' (%s). Only the "
                "hybrid-level flag is effective.",
                self.grounded.store_provenance,
                self.store_provenance,
            )

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Flatten the nested ``react`` sub-dict onto the flat react fields.

        The ``StructuredConfig`` base rebuilds the ``grounded`` field
        recursively (it is itself a ``StructuredConfig``).  The only shape
        quirk is the ``react`` sub-dict, whose keys map to the flat
        ``react_*`` fields::

            reasoning:
              strategy: hybrid
              grounded:
                intent: {mode: extract, num_queries: 3}
                retrieval: {top_k: 5}
                synthesis: {style: conversational}
                sources:
                  - type: vector_kb
                    name: docs
              react:
                max_iterations: 5
                verbose: false
                store_trace: false
              store_provenance: true
              greeting_template: "Hello!"

        A flat key already present (the round-trip shape produced by
        ``to_dict``) takes precedence over the nested form.
        """
        react = raw.pop("react", None)
        if isinstance(react, Mapping):
            if "max_iterations" in react:
                raw.setdefault("react_max_iterations", react["max_iterations"])
            if "verbose" in react:
                raw.setdefault("react_verbose", react["verbose"])
            if "store_trace" in react:
                raw.setdefault("react_store_trace", react["store_trace"])
        return raw

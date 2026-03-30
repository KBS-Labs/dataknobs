"""Configuration dataclass for the hybrid reasoning strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from dataknobs_bots.reasoning.grounded_config import GroundedReasoningConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridReasoningConfig:
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridReasoningConfig:
        """Build config from a flat reasoning config dict.

        Expected shape::

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
        """
        grounded_data = data.get("grounded", {})
        react_data = data.get("react", {})

        hybrid_provenance = data.get("store_provenance", True)
        grounded_config = GroundedReasoningConfig.from_dict(grounded_data)

        # Warn if grounded sub-config has a different store_provenance —
        # in hybrid mode, only the hybrid-level flag controls provenance
        # storage (grounded's generate() is never called, only
        # retrieve_context()).
        if grounded_config.store_provenance != hybrid_provenance:
            logger.warning(
                "Hybrid strategy: 'grounded.store_provenance' (%s) differs "
                "from hybrid-level 'store_provenance' (%s). Only the "
                "hybrid-level flag is effective.",
                grounded_config.store_provenance,
                hybrid_provenance,
            )

        return cls(
            grounded=grounded_config,
            react_max_iterations=react_data.get("max_iterations", 5),
            react_verbose=react_data.get("verbose", False),
            react_store_trace=react_data.get("store_trace", False),
            store_provenance=hybrid_provenance,
            greeting_template=data.get("greeting_template"),
        )

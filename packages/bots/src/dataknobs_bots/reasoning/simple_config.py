"""Configuration dataclass for the simple reasoning strategy."""

from __future__ import annotations

from dataclasses import dataclass

from dataknobs_bots.reasoning.config_base import ReasoningConfig


@dataclass(frozen=True)
class SimpleReasoningConfig(ReasoningConfig):
    """Configuration for :class:`SimpleReasoning`.

    The simple strategy makes a direct LLM call with no extra reasoning
    steps, so it adds nothing to :class:`ReasoningConfig` -- its whole
    configurable surface is the universal ``greeting_template`` declared
    there.  The class still exists because ``CONFIG_CLS`` is what the
    ``StructuredConfigConsumer`` machinery dispatches on, and because a
    strategy that later gains a knob of its own has somewhere to put it.
    """
